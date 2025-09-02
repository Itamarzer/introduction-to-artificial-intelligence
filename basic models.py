# MNIST + Optuna-tuned MLP (Adam + ReLU) + Gradient Boosting baseline
# Anti-overfitting: L2, Dropout, BatchNorm, GaussianNoise, Label Smoothing, MaxNorm,
# EarlyStopping, ReduceLROnPlateau, Gradient Clipping
# -------------------------------------------------------------------
# pip install optuna tensorflow matplotlib seaborn scikit-learn pandas

import os, time, platform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TF INFO/WARN
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # (optional) disable oneDNN

# ---- start timer ----
_START = time.perf_counter()

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm as MaxNorm
from tensorflow.keras.utils import to_categorical

import optuna

from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc,
    confusion_matrix, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


# ------------- utils -------------
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
    if gpus:
        print(f"Accelerator: GPU x{len(gpus)} -> {', '.join(gpu_names)}")
    else:
        print("Accelerator: CPU (no GPU visible)")
    try:
        import psutil
        print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    except Exception:
        pass


def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_label_distribution(y_int):
    counts = Counter(y_int)
    df_counts = pd.DataFrame({"digit": list(counts.keys()), "count": list(counts.values())})
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    sns.barplot(data=df_counts, x="digit", y="count", hue="digit",
                palette="cool", ax=ax, dodge=False, legend=False)
    ax.set_xlabel("Handwritten Digits")
    ax.set_ylabel("Count")
    ax.set_title("Training Set Label Distribution")
    plt.tight_layout()
    plt.show()


def random_prediction_example(X, y, model, save_path="sample_prediction.png"):
    """Show a random test image with class-probability bar chart (like your screenshot)."""
    idx = np.random.randint(0, X.shape[0])
    img = X[idx]
    probs = model.predict(img.reshape(1, -1), verbose=0).ravel()  # shape (10,)

    print("\nSample probabilities:\n", np.array2string(probs, precision=6, suppress_small=True))
    pred = probs.argmax()
    true = (y[idx] if y.ndim == 1 else np.argmax(y[idx]))

    fig = plt.figure(figsize=(9, 6), dpi=120)
    # left: image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img.reshape(28, 28), cmap="viridis")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"True: {true} | Pred: {pred}")
    # right: probabilities
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(range(10), probs)
    ax2.set_title("Class Probability")
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_xticks(range(10))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ------------- model -------------
def build_mlp(trial: optuna.trial.Trial) -> Sequential:
    """Adam + ReLU MLP with multiple regularizers (Optuna-tuned)."""
    K.clear_session()

    # Tunables
    n_hidden      = trial.suggest_int("n_hidden", 1, 4)
    units0        = trial.suggest_int("units0", 128, 1024, step=64)
    l2_coeff      = trial.suggest_float("l2_coeff", 1e-6, 1e-2, log=True)
    dropout0      = trial.suggest_float("dropout0", 0.0, 0.6, step=0.1)
    use_bn        = trial.suggest_categorical("use_batchnorm", [True, False])
    lr            = trial.suggest_float("adam_lr", 1e-5, 5e-3, log=True)
    clipnorm      = trial.suggest_float("clipnorm", 0.5, 5.0)
    label_smooth  = trial.suggest_float("label_smoothing", 0.0, 0.10, step=0.02)
    gn_std        = trial.suggest_float("gauss_noise_std", 0.0, 0.20, step=0.05)  # input noise
    maxnorm_val   = trial.suggest_categorical("maxnorm", [0.0, 2.0, 3.0, 4.0])   # 0.0 -> off

    kernel_constraint = None if maxnorm_val == 0.0 else MaxNorm(max_value=maxnorm_val)

    model = Sequential(name="OptunaMLP")
    model.add(tf.keras.Input(shape=(28 * 28,)))
    if gn_std > 0:
        model.add(GaussianNoise(gn_std, name="input_noise"))

    # Hidden-1
    model.add(Dense(
        units0, activation="relu",
        kernel_regularizer=l2(l2_coeff),
        kernel_constraint=kernel_constraint,
        name="Hidden-1"
    ))
    if use_bn: model.add(BatchNormalization())
    if dropout0 > 0: model.add(Dropout(dropout0))

    # Additional hidden layers
    prev_units = units0
    for i in range(2, n_hidden + 1):
        units_i   = trial.suggest_int(f"units{i}", 64, max(128, prev_units), step=64)
        dropout_i = trial.suggest_float(f"dropout{i}", 0.0, 0.6, step=0.1)
        model.add(Dense(
            units_i, activation="relu",
            kernel_regularizer=l2(l2_coeff),
            kernel_constraint=kernel_constraint,
            name=f"Hidden-{i}"
        ))
        if use_bn: model.add(BatchNormalization())
        if dropout_i > 0: model.add(Dropout(dropout_i))
        prev_units = units_i

    # Output
    model.add(Dense(10, activation="softmax", name="Output"))

    # Loss with label smoothing + Adam with gradient clipping
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model


def objective_nn(trial: optuna.trial.Trial, X_tr, y_tr, X_val, y_val) -> float:
    model = build_mlp(trial)
    epochs     = trial.suggest_int("epochs", 8, 30)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    es  = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es, rlr]
    )
    return max(history.history["val_accuracy"])


def run_hgb_optuna(X_tr, y_tr_int, X_val, y_val_int, seed=42):
    def objective_hgb(trial: optuna.trial.Trial) -> float:
        n_components = trial.suggest_int("pca_components", 30, 150, step=30)
        learning_rate = trial.suggest_float("hgb_lr", 0.02, 0.3, log=True)
        max_depth = trial.suggest_int("hgb_max_depth", 3, 20)
        max_leaf_nodes = trial.suggest_int("hgb_max_leaf_nodes", 15, 63)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("hgb", HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes,
                random_state=seed
            ))
        ])
        pipe.fit(X_tr, y_tr_int)
        val_pred = pipe.predict(X_val)
        return accuracy_score(y_val_int, val_pred)

    hgb_study = optuna.create_study(direction="maximize", study_name="mnist_hgb_baseline")
    hgb_study.optimize(objective_hgb, n_trials=15, show_progress_bar=False)
    return hgb_study


# ------------- main -------------
def main():
    set_seeds(42)
    print_runtime_env()

    # Data
    from tensorflow.keras.datasets import mnist
    (x_train, y_train_int), (x_test, y_test_int) = mnist.load_data()
    plot_label_distribution(y_train_int)

    # Flatten + scale
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    y_train = to_categorical(y_train_int, num_classes=10)
    y_test  = to_categorical(y_test_int,  num_classes=10)

    # Train/Val split
    X_tr, X_val, y_tr, y_val, y_tr_int, y_val_int = train_test_split(
        x_train, y_train, y_train_int, test_size=0.2, random_state=42, stratify=y_train_int
    )

    # Optuna on NN
    study = optuna.create_study(direction="maximize", study_name="mnist_mlp_adam_relu")
    study.optimize(lambda t: objective_nn(t, X_tr, y_tr, X_val, y_val),
                   n_trials=20, show_progress_bar=False)

    print("\n[NN] Best trial:")
    print(f"  val_accuracy = {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Rebuild best model using FixedTrial
    best_params = study.best_trial.params
    best_model  = build_mlp(optuna.trial.FixedTrial(best_params))

    # Train on train+val with small internal val for callbacks
    X_full = np.concatenate([X_tr, X_val], axis=0)
    y_full = np.concatenate([y_tr, y_val], axis=0)

    final_epochs     = int(best_params.get("epochs", 15))
    final_batch_size = int(best_params.get("batch_size", 128))

    es  = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ckpt_path = "best_mnist_mlp.keras"
    mc  = ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1)

    hist = best_model.fit(
        X_full, y_full,
        validation_split=0.1,
        epochs=final_epochs,
        batch_size=final_batch_size,
        callbacks=[es, rlr, mc],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nMLP Test set -> loss={test_loss:.4f}, accuracy={test_acc:.4f}")

    # ------------------------------
    # PLOTS & REPORTS
    # ------------------------------
    # 1) Accuracy -> model_accuracy.png
    plt.figure(figsize=(6, 4), dpi=110)
    plt.title("Model Accuracy")
    plt.plot(range(1, len(hist.epoch) + 1), hist.history["accuracy"], label="train_acc")
    plt.plot(range(1, len(hist.epoch) + 1), hist.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("model_accuracy.png"); plt.show()

    # 2) Loss -> model_loss.png
    plt.figure(figsize=(6, 4), dpi=110)
    plt.title("Model Loss")
    plt.plot(range(1, len(hist.epoch) + 1), hist.history["loss"], label="train_loss")
    plt.plot(range(1, len(hist.epoch) + 1), hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("model_loss.png"); plt.show()

    # 3) ROC (micro-average) -> roc_curve.png
    y_score = best_model.predict(x_test, batch_size=final_batch_size, verbose=0)
    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.figure(figsize=(7, 6), dpi=120)
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC (AUC = {auc_micro:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC Curve"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.tight_layout(); plt.savefig("roc_curve.png"); plt.show()

    # 4) Classification report -> classification_report.txt
    y_pred_int = np.argmax(y_score, axis=1)
    report_str = classification_report(y_test_int, y_pred_int, digits=4)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)
    print("\nClassification report (head):\n", "\n".join(report_str.splitlines()[:8]), "\n...")

    # 5) Epochs results (txt table) -> epochs_results.txt
    epochs_table = pd.DataFrame({
        "epoch": np.arange(1, len(hist.epoch) + 1),
        "loss": hist.history["loss"],
        "val_loss": hist.history["val_loss"],
        "accuracy": hist.history["accuracy"],
        "val_accuracy": hist.history["val_accuracy"]
    })
    with open("epochs_results.txt", "w", encoding="utf-8") as f:
        f.write(epochs_table.to_string(index=False))

    # 6) Confusion matrix -> confusion_matrix.png
    cm = confusion_matrix(y_test_int, y_pred_int, labels=list(range(10)))
    plt.figure(figsize=(7,6), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.show()

    # 7) F1 scores (per class + macro/micro/weighted) -> f1_scores.png
    rep = classification_report(y_test_int, y_pred_int, output_dict=True, digits=4)
    per_class_f1 = [rep[str(i)]["f1-score"] for i in range(10)]
    macro_f1  = f1_score(y_test_int, y_pred_int, average="macro")
    micro_f1  = f1_score(y_test_int, y_pred_int, average="micro")
    weighted_f1 = f1_score(y_test_int, y_pred_int, average="weighted")
    print(f"F1 (micro)={micro_f1:.4f} | F1 (macro)={macro_f1:.4f} | F1 (weighted)={weighted_f1:.4f}")

    plt.figure(figsize=(8,4), dpi=120)
    plt.bar(range(10), per_class_f1)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Digit"); plt.ylabel("F1-score")
    plt.title(f"Per-class F1 (macro={macro_f1:.3f}, micro={micro_f1:.3f})")
    plt.xticks(range(10))
    plt.tight_layout(); plt.savefig("f1_scores.png"); plt.show()

    # 8) Random prediction demo (image + probs) -> sample_prediction.png
    random_prediction_example(x_test, y_test_int, best_model, save_path="sample_prediction.png")

    # ------------------------------
    # Gradient Boosting Baseline
    # ------------------------------
    hgb_study = run_hgb_optuna(X_tr, y_tr_int, X_val, y_val_int, seed=42)

    print("\n[Gradient Boosting] Best trial:")
    print(f"  val_accuracy = {hgb_study.best_value:.4f}")
    for k, v in hgb_study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_hgb_params = hgb_study.best_trial.params
    hgb_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=best_hgb_params["pca_components"], random_state=42)),
        ("hgb", HistGradientBoostingClassifier(
            learning_rate=best_hgb_params["hgb_lr"],
            max_depth=best_hgb_params["hgb_max_depth"],
            max_leaf_nodes=best_hgb_params["hgb_max_leaf_nodes"],
            random_state=42
        ))
    ])
    hgb_pipe.fit(x_train, y_train_int)
    hgb_test_pred = hgb_pipe.predict(x_test)
    hgb_test_acc = accuracy_score(y_test_int, hgb_test_pred)
    print(f"\nHGB (PCA+Gradient Boosting) Test accuracy = {hgb_test_acc:.4f}")

    # Summary
    print("\n=== Final Results ===")
    print(f"Neural Network (Adam+ReLU, Optuna-tuned) Test Accuracy: {test_acc:.4f}")
    print(f"F1 micro: {micro_f1:.4f} | macro: {macro_f1:.4f} | weighted: {weighted_f1:.4f}")
    print(f"Gradient Boosting (PCA+HGB)            Test Accuracy: {hgb_test_acc:.4f}")
    print("Saved files: model_accuracy.png, model_loss.png, roc_curve.png, "
          "confusion_matrix.png, f1_scores.png, sample_prediction.png, "
          "classification_report.txt, epochs_results.txt, best_mnist_mlp.keras")

    # ---- total runtime ----
    _elapsed = time.perf_counter() - _START
    print(f"\n=== Total runtime: {_elapsed:.1f} s (~{_elapsed/60:.2f} min) ===")


if __name__ == "__main__":
    main()
