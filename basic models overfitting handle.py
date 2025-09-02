# MNIST MLP with strong anti-overfitting + Optuna + K-Fold CV
# (Fixes: MixUp dtype; ReduceLROnPlateau vs LR schedule conflict)
# ---------------------------------------------------------------
# pip install optuna tensorflow matplotlib seaborn scikit-learn pandas

import os, time, platform, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # (optional) reproducibility

_START = time.perf_counter()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm as MaxNorm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc

import optuna


# ---------- utils ----------
def print_runtime_env():
    gpus = tf.config.list_physical_devices('GPU')
    gpu_names = []
    for g in gpus:
        try:
            details = tf.config.experimental.get_device_details(g)
            gpu_names.append(details.get('device_name', str(g)))
        except Exception:
            gpu_names.append(str(g))
    print("\n=== Runtime Environment ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python: {platform.python_version()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Accelerator: {'GPU x'+str(len(gpus))+' -> '+', '.join(gpu_names) if gpus else 'CPU (no GPU visible)'}")


def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def plot_label_distribution(y_int):
    from collections import Counter
    counts = Counter(y_int)
    df = pd.DataFrame({"digit": list(counts.keys()), "count": list(counts.values())})
    fig, ax = plt.subplots(figsize=(6,4), dpi=110)
    sns.barplot(data=df, x="digit", y="count", hue="digit", palette="cool", legend=False, ax=ax, dodge=False)
    ax.set_title("Training Set Label Distribution"); ax.set_xlabel("Digit"); ax.set_ylabel("Count")
    plt.tight_layout(); plt.show()


# --------- MixUp for tf.data (all float32) ----------
def _sample_beta(alpha, shape, dtype=tf.float32):
    alpha = tf.convert_to_tensor(alpha, dtype=dtype)
    g1 = tf.random.gamma(shape, alpha, dtype=dtype)
    g2 = tf.random.gamma(shape, alpha, dtype=dtype)
    return g1 / (g1 + g2)

def make_dataset(X, y, batch_size, training=True, mixup_alpha=0.0):
    # Ensure float32 tensors to avoid dtype mismatches
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(20000, seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    if training and mixup_alpha > 0.0:
        def mixup_map(x, y):
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.float32)
            bs = tf.shape(x)[0]
            idx = tf.random.shuffle(tf.range(bs))
            x2 = tf.gather(x, idx); y2 = tf.gather(y, idx)
            lam = _sample_beta(tf.constant(mixup_alpha, dtype=tf.float32), (bs, 1), dtype=tf.float32)
            x = lam * x + (1.0 - lam) * x2
            y = lam * y + (1.0 - lam) * y2
            return x, y
        ds = ds.map(mixup_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------- Model builder ----------
def build_mlp(trial, input_dim=28*28):
    K.clear_session()

    # Capacity & regularization (anti-overfitting knobs)
    n_hidden     = trial.suggest_int("n_hidden", 1, 4)
    units0       = trial.suggest_int("units0", 128, 1024, step=64)
    l2_coeff     = trial.suggest_float("l2_coeff", 1e-6, 1e-2, log=True)
    dropout0     = trial.suggest_float("dropout0", 0.0, 0.6, step=0.1)
    use_bn       = trial.suggest_categorical("use_batchnorm", [True, False])
    gn_std       = trial.suggest_float("gauss_noise_std", 0.0, 0.20, step=0.05)
    maxnorm_val  = trial.suggest_categorical("maxnorm", [0.0, 2.0, 3.0, 4.0])

    kernel_constraint = None if maxnorm_val == 0.0 else MaxNorm(maxnorm_val)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    if gn_std > 0: x = GaussianNoise(gn_std, name="input_noise")(x)

    x = Dense(units0, activation="relu",
              kernel_regularizer=l2(l2_coeff),
              kernel_constraint=kernel_constraint, name="Hidden-1")(x)
    if use_bn: x = BatchNormalization()(x)
    if dropout0 > 0: x = Dropout(dropout0)(x)

    prev_units = units0
    for i in range(2, n_hidden + 1):
        units_i   = trial.suggest_int(f"units{i}", 64, max(128, prev_units), step=64)
        dropout_i = trial.suggest_float(f"dropout{i}", 0.0, 0.6, step=0.1)
        x = Dense(units_i, activation="relu",
                  kernel_regularizer=l2(l2_coeff),
                  kernel_constraint=kernel_constraint, name=f"Hidden-{i}")(x)
        if use_bn: x = BatchNormalization()(x)
        if dropout_i > 0: x = Dropout(dropout_i)(x)
        prev_units = units_i

    outputs = Dense(10, activation="softmax", name="Output")(x)

    # --- Learning rate setup (choose schedule OR constant) ---
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.10, step=0.02)
    lr_mode = trial.suggest_categorical("lr_schedule", ["constant", "cosine"])
    base_lr = trial.suggest_float("adam_lr", 5e-5, 3e-3, log=True)
    clipnorm = trial.suggest_float("clipnorm", 0.5, 5.0)

    if lr_mode == "cosine":
        t0 = trial.suggest_categorical("cosine_first_decay_steps", [5, 7, 9, 11])
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_lr, first_decay_steps=t0, t_mul=1.5, m_mul=0.9
        )
    else:
        lr = base_lr  # plain float -> compatible with ReduceLROnPlateau

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

    model = tf.keras.Model(inputs, outputs, name="AntiOverfitMLP")
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
                  metrics=["accuracy"])
    return model


# ---------- Optuna objective with K-fold CV ----------
def objective_cv(trial, X, y_onehot, y_int):
    epochs     = trial.suggest_int("epochs", 10, 22)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    mixup_a    = trial.suggest_float("mixup_alpha", 0.0, 0.4, step=0.1)

    # defensive casting
    X = np.asarray(X, dtype=np.float32)
    y_onehot = np.asarray(y_onehot, dtype=np.float32)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []

    for tr_idx, val_idx in skf.split(X, y_int):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_onehot[tr_idx], y_onehot[val_idx]

        model = build_mlp(trial)

        ds_tr  = make_dataset(X_tr, y_tr, batch_size, training=True,  mixup_alpha=mixup_a)
        ds_val = make_dataset(X_val, y_val, batch_size, training=False, mixup_alpha=0.0)

        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0)
        callbacks = [es]

        # Only use ReduceLROnPlateau when LR is a float (lr_schedule == 'constant')
        lr_mode = trial.params.get("lr_schedule", "constant")
        if lr_mode == "constant":
            rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0)
            callbacks.append(rlr)

        h = model.fit(ds_tr, validation_data=ds_val, epochs=epochs, verbose=0, callbacks=callbacks)
        fold_scores.append(float(np.max(h.history["val_accuracy"])))

    return float(np.mean(fold_scores))


def main():
    set_seeds(42)
    print_runtime_env()

    # ----- data -----
    from tensorflow.keras.datasets import mnist
    (x_tr, y_tr_int), (x_test, y_test_int) = mnist.load_data()
    plot_label_distribution(y_tr_int)

    x_tr = x_tr.reshape(-1, 28*28).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28*28).astype(np.float32) / 255.0

    # one-hot as float32 (no dtype kwarg here)
    y_tr = to_categorical(y_tr_int, 10).astype(np.float32)
    y_test = to_categorical(y_test_int, 10).astype(np.float32)

    # ----- Optuna with K-fold CV -----
    study = optuna.create_study(direction="maximize", study_name="mnist_mlp_cv_anti_overfit")
    study.optimize(lambda t: objective_cv(t, x_tr, y_tr, y_tr_int), n_trials=20, show_progress_bar=False)

    print("\n[Optuna/KFold] Best mean val_acc:", f"{study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best = study.best_trial.params

    # ----- final train on full train set (with small internal val for callbacks) -----
    model = build_mlp(optuna.trial.FixedTrial(best))
    batch_size = int(best.get("batch_size", 128))
    epochs     = int(best.get("epochs", 16))
    mixup_a    = float(best.get("mixup_alpha", 0.2))
    lr_mode    = best.get("lr_schedule", "constant")

    X_train, X_val, Y_train, Y_val = train_test_split(
        x_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr_int
    )

    ds_train = make_dataset(X_train, Y_train, batch_size, training=True,  mixup_alpha=mixup_a)
    ds_val   = make_dataset(X_val,   Y_val,   batch_size, training=False, mixup_alpha=0.0)

    es  = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    ckpt = ModelCheckpoint("best_mnist_mlp.keras", monitor="val_loss", save_best_only=True, verbose=1)

    callbacks = [es, ckpt]
    if lr_mode == "constant":
        rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
        callbacks.append(rlr)

    hist = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, verbose=1)

    # ----- evaluate on test -----
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

    # ----- plots & reports -----
    # Accuracy
    plt.figure(figsize=(6,4), dpi=110)
    plt.title("Model Accuracy")
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("model_accuracy.png"); plt.show()

    # Loss
    plt.figure(figsize=(6,4), dpi=110)
    plt.title("Model Loss")
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("model_loss.png"); plt.show()

    # ROC micro
    y_score = model.predict(x_test, batch_size=batch_size, verbose=0)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    auc_micro = auc(fpr, tpr)
    plt.figure(figsize=(7,6), dpi=120)
    plt.plot(fpr, tpr, label=f"Micro-average ROC (AUC = {auc_micro:.3f})", linewidth=2)
    plt.plot([0,1],[0,1],"k--", linewidth=1)
    plt.title("ROC Curve"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.tight_layout(); plt.savefig("roc_curve.png"); plt.show()

    # Confusion matrix & F1
    y_pred_int = np.argmax(y_score, axis=1)
    cm = confusion_matrix(y_test_int, y_pred_int, labels=list(range(10)))
    plt.figure(figsize=(7,6), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.show()

    macro_f1  = f1_score(y_test_int, y_pred_int, average="macro")
    micro_f1  = f1_score(y_test_int, y_pred_int, average="micro")
    weighted_f1 = f1_score(y_test_int, y_pred_int, average="weighted")
    print(f"F1 micro={micro_f1:.4f} | macro={macro_f1:.4f} | weighted={weighted_f1:.4f}")

    rep = classification_report(y_test_int, y_pred_int, digits=4)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)
    print("\nClassification report (head):\n", "\n".join(rep.splitlines()[:8]), "\n...")

    # Save epoch table (txt)
    ep_df = pd.DataFrame({
        "epoch": np.arange(1, len(hist.history["loss"])+1),
        "loss": hist.history["loss"], "val_loss": hist.history["val_loss"],
        "accuracy": hist.history["accuracy"], "val_accuracy": hist.history["val_accuracy"]
    })
    with open("epochs_results.txt", "w", encoding="utf-8") as f:
        f.write(ep_df.to_string(index=False))

    # Example prediction
    idx = np.random.randint(0, x_test.shape[0])
    img = x_test[idx].reshape(28,28)
    p = y_score[idx]
    plt.figure(figsize=(10,6), dpi=120)
    ax1 = plt.subplot(1,2,1); ax1.imshow(img, cmap="viridis"); ax1.set_axis_off()
    ax1.set_title(f"True: {y_test_int[idx]} | Pred: {np.argmax(p)}")
    ax2 = plt.subplot(1,2,2); ax2.bar(range(10), p); ax2.set_title("Class Probability")
    ax2.set_xlabel("Digit"); ax2.set_ylabel("Probability"); ax2.set_xticks(range(10))
    plt.tight_layout(); plt.savefig("sample_prediction.png"); plt.show()

    # Save best params/epoch
    best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)
    with open("best_params.json","w") as f: json.dump(study.best_trial.params, f, indent=2)
    with open("best_epoch.txt","w") as f: f.write(str(best_epoch))

    print("\nSaved: model_* plots, roc_curve.png, confusion_matrix.png, "
          "classification_report.txt, epochs_results.txt, sample_prediction.png, "
          "best_mnist_mlp.keras, best_params.json, best_epoch.txt")

    # ---- total runtime ----
    elapsed = time.perf_counter() - _START
    print(f"\n=== Total runtime: {elapsed:.1f} s (~{elapsed/60:.2f} min) ===")


if __name__ == "__main__":
    main()
