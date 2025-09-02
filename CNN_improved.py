# mnist_cnn_fast.py
# -----------------------------------------------------------------------------
# Fast, CPU-friendly Keras CNN for MNIST with strong regularization and
# full diagnostics/artifacts (plots + txt), optimized for Windows/CPU.
#
# Requires:
#   pip install tensorflow matplotlib seaborn scikit-learn pandas psutil
#
# Saves:
#   best_mnist_cnn.keras
#   best_mnist_cnn_final.keras
#   best_epoch.txt
#   best_cnn_params.json
#   classification_report.txt
#   epochs_results.csv / epochs_results.txt
#   model_accuracy.png
#   model_loss.png
#   roc_curve.png
#   confusion_matrix.png
#   f1_scores.png
#   sample_prediction.png
#   top5_errors.png
# -----------------------------------------------------------------------------

import os, time, json, random, platform
from pathlib import Path

# Keep TF logs quiet; leave oneDNN on for CPU speed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# If you hit numerics variance and want bit-for-bit reproducibility (slower):
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import psutil
except Exception:
    psutil = None

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, f1_score
)

# -----------------------
# Fast/quality knobs
# -----------------------
SEED = 42
EPOCHS = 12            # keep low for CPU speed; 8–12 is plenty for MNIST
BATCH_SIZE = 256       # larger batch is faster on CPU if memory allows
INIT_LR = 1.5e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.04
CLIPNORM = 1.0
PATIENCE = 4           # early stopping patience
VAL_SPLIT = 0.1        # internal val split (after merging train+val)
AUG_INTENSITY = 0.08   # smaller = faster/light aug

# -----------------------
# Reproducibility
# -----------------------
def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(SEED)

# -----------------------
# Print runtime hardware
# -----------------------
def print_runtime_env():
    gpu_list = tf.config.list_physical_devices("GPU")
    accel = f"GPU: {tf.config.experimental.get_device_details(gpu_list[0]).get('device_name', 'GPU')}" \
        if gpu_list else "CPU (no GPU visible)"
    print("=== Runtime Environment ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Accelerator: {accel}")
    if psutil:
        print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")

# -----------------------
# Model (compact & fast)
# -----------------------
def build_fast_cnn(input_shape=(28,28,1),
                   wd=WEIGHT_DECAY,
                   drop_conv=0.15,
                   drop_dense=0.40):
    """
    Compact CPU-friendly CNN:
      - SeparableConv2D for fewer MACs
      - BatchNorm + ReLU
      - GlobalAveragePooling2D (no big Dense stack)
    Strong regularization: L2 + Dropout
    """
    reg = regularizers.l2(wd)

    inp = layers.Input(shape=input_shape)

    x = layers.SeparableConv2D(32, 3, padding="same", depthwise_regularizer=reg,
                               pointwise_regularizer=reg, use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.SeparableConv2D(32, 3, padding="same", depthwise_regularizer=reg,
                               pointwise_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(drop_conv)(x)

    x = layers.SeparableConv2D(64, 3, padding="same", depthwise_regularizer=reg,
                               pointwise_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.SeparableConv2D(64, 3, padding="same", depthwise_regularizer=reg,
                               pointwise_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(drop_conv)(x)

    x = layers.SeparableConv2D(128, 3, padding="same", depthwise_regularizer=reg,
                               pointwise_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(drop_conv)(x)

    x = layers.GlobalAveragePooling2D()(x)        # tiny head = speed
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(drop_dense)(x)
    out = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inp, out, name="FastMNISTCNN")
    opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, clipnorm=CLIPNORM)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model

# -----------------------
# Augmentation (light, vectorized)
# -----------------------
def make_augmenter(intensity=AUG_INTENSITY):
    return tf.keras.Sequential([
        layers.RandomTranslation(intensity, intensity, fill_mode="nearest"),
        layers.RandomRotation(intensity, fill_mode="nearest"),
        layers.RandomZoom(intensity),
        layers.RandomContrast(intensity),
    ], name="augmenter")

# -----------------------
# Datasets (cache→batch→augment→prefetch)
# -----------------------
def make_datasets(x_tr, y_tr, x_val, y_val, batch_size=BATCH_SIZE, aug=True):
    AUTOTUNE = tf.data.AUTOTUNE
    aug_layer = make_augmenter() if aug else None

    # Cache in RAM for speed
    ds_tr = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).cache().shuffle(20000, seed=SEED)
    ds_tr = ds_tr.batch(batch_size)
    if aug_layer is not None:
        ds_tr = ds_tr.map(lambda x, y: (aug_layer(x, training=True), y),
                          num_parallel_calls=AUTOTUNE)
    ds_tr = ds_tr.prefetch(AUTOTUNE)

    ds_val = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .cache()
              .batch(batch_size)
              .prefetch(AUTOTUNE))
    return ds_tr, ds_val

# -----------------------
# Plots & saves
# -----------------------
def plot_curves(hist, acc_png="model_accuracy.png", loss_png="model_loss.png"):
    e = range(1, len(hist.history["loss"])+1)
    plt.figure(figsize=(6,4), dpi=110)
    plt.plot(e, hist.history["accuracy"], label="train_acc")
    plt.plot(e, hist.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(acc_png); plt.close()

    plt.figure(figsize=(6,4), dpi=110)
    plt.plot(e, hist.history["loss"], label="train_loss")
    plt.plot(e, hist.history["val_loss"], label="val_loss")
    plt.title("Loss per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(loss_png); plt.close()

def plot_confusion(cm, out_png="confusion_matrix.png"):
    plt.figure(figsize=(7,6), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_f1_bars(y_true, y_pred, out_png="f1_scores.png"):
    f1s = [f1_score(y_true==(c), y_pred==(c)) for c in range(10)]
    plt.figure(figsize=(7,4), dpi=120)
    plt.bar(range(10), f1s)
    plt.xticks(range(10)); plt.ylim(0.9, 1.0)
    plt.title("Per-class F1"); plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_roc_curves(y_true_oh, y_prob, out_png="roc_curve.png"):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    macro_auc = float(np.mean([roc_auc[i] for i in range(10)]))

    plt.figure(figsize=(7,6), dpi=120)
    for i in range(10):
        plt.plot(fpr[i], tpr[i], alpha=0.25)
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro (AUC={roc_auc['micro']:.3f})", linewidth=2.5)
    plt.plot([0,1],[0,1],"--", linewidth=1)
    plt.title(f"MNIST ROC (macro AUC={macro_auc:.3f})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    return roc_auc["micro"], macro_auc

def plot_top5_errors(x, y_true, y_pred, y_prob, out_png="top5_errors.png"):
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return
    conf_wrong = y_prob[wrong, y_pred[wrong]]
    idx_sorted = wrong[np.argsort(-conf_wrong)]
    idx_show = idx_sorted[:min(5, len(idx_sorted))]

    plt.figure(figsize=(12,3), dpi=140)
    for i, idx in enumerate(idx_show):
        plt.subplot(1, len(idx_show), i+1)
        plt.imshow(x[idx].squeeze(-1), cmap="gray")
        plt.axis("off")
        plt.title(f"pred {y_pred[idx]} ({y_prob[idx, y_pred[idx]]:.2f})\ntrue {y_true[idx]}")
    plt.suptitle("Top-5 High-Confidence Errors")
    plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(out_png); plt.close()

def save_txt_table(df: pd.DataFrame, path_txt: str, max_rows=80):
    with open(path_txt, "w") as f:
        f.write(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            f.write(f"\n... ({len(df)-max_rows} more rows)")

# -----------------------
# Main
# -----------------------
def main():
    t0 = time.time()
    print_runtime_env()

    # Data
    from tensorflow.keras.datasets import mnist
    (x_train, y_train_int), (x_test, y_test_int) = mnist.load_data()
    # scale & add channel
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32")  / 255.0)[..., None]

    # hold out validation from train for the Optuna-like callbacks
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr_int, y_val_int = train_test_split(
        x_train, y_train_int, test_size=0.15, random_state=SEED, stratify=y_train_int
    )
    y_tr  = to_categorical(y_tr_int, num_classes=10).astype("float32")
    y_val = to_categorical(y_val_int, num_classes=10).astype("float32")
    y_test = to_categorical(y_test_int, num_classes=10).astype("float32")

    # Datasets
    ds_tr, ds_val = make_datasets(X_tr, y_tr, X_val, y_val, BATCH_SIZE, aug=True)

    # Model
    model = build_fast_cnn()

    # Callbacks
    es  = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ckpt_path = "best_mnist_cnn.keras"
    mc  = ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1)

    # Train
    hist = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[es, rlr, mc],
        verbose=1
    )

    # Merge train+val and fine-tune briefly (fast) with small val split for callbacks
    X_full = np.concatenate([X_tr, X_val], 0)
    y_full = np.concatenate([y_tr, y_val], 0)
    X_train_final, X_val_small, y_train_final, y_val_small = train_test_split(
        X_full, y_full, test_size=VAL_SPLIT, random_state=SEED, stratify=np.argmax(y_full, axis=1)
    )
    ds_final_tr, ds_final_val = make_datasets(X_train_final, y_train_final, X_val_small, y_val_small,
                                              BATCH_SIZE, aug=True)

    # Short final run (faster) using best weights already loaded
    hist2 = model.fit(
        ds_final_tr,
        validation_data=ds_final_val,
        epochs=max(2, min(5, EPOCHS // 3)),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1)],
        verbose=1
    )

    # Save final model too
    model.save("best_mnist_cnn_final.keras")

    # Evaluate on test
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print(f"\n[TEST] loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Predictions
    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred_int = np.argmax(y_prob, axis=1)

    # Reports
    report = classification_report(y_test_int, y_pred_int, digits=4)
    print("\nClassification report (head):\n", "\n".join(report.splitlines()[:12]))

    f1_micro = f1_score(y_test_int, y_pred_int, average="micro")
    f1_macro = f1_score(y_test_int, y_pred_int, average="macro")
    f1_weighted = f1_score(y_test_int, y_pred_int, average="weighted")
    print(f"F1 micro={f1_micro:.4f} | macro={f1_macro:.4f} | weighted={f1_weighted:.4f}")

    # Confusion matrix / F1 bars / ROC
    cm = confusion_matrix(y_test_int, y_pred_int)
    plot_confusion(cm)
    plot_f1_bars(y_test_int, y_pred_int)

    auc_micro, auc_macro = plot_roc_curves(y_test, y_prob)

    # Curves
    # merge two histories for plotting continuity
    h_acc = hist.history["accuracy"] + hist2.history.get("accuracy", [])
    h_val_acc = hist.history["val_accuracy"] + hist2.history.get("val_accuracy", [])
    h_loss = hist.history["loss"] + hist2.history.get("loss", [])
    h_val_loss = hist.history["val_loss"] + hist2.history.get("val_loss", [])
    class DummyHist: pass
    H = DummyHist(); H.history = {"accuracy": h_acc, "val_accuracy": h_val_acc,
                                  "loss": h_loss, "val_loss": h_val_loss}
    plot_curves(H)

    # Sample prediction
    mid = len(x_test)//2
    p_mid = y_prob[mid].max(); yhat_mid = int(np.argmax(y_prob[mid]))
    plt.figure(figsize=(3,3), dpi=160)
    plt.imshow(x_test[mid].squeeze(-1), cmap="gray"); plt.axis("off")
    plt.title(f"Sample idx={mid}\ntrue={y_test_int[mid]}, pred={yhat_mid}, p={p_mid:.3f}")
    plt.tight_layout(); plt.savefig("sample_prediction.png"); plt.close()

    # Top-5 high-confidence errors
    plot_top5_errors(x_test, y_test_int, y_pred_int, y_prob)

    # Save texts
    with open("classification_report.txt", "w") as f:
        f.write(report + "\n")
        f.write(f"F1 micro={f1_micro:.4f} | macro={f1_macro:.4f} | weighted={f1_weighted:.4f}\n")
        f.write(f"Test acc={test_acc:.4f}, loss={test_loss:.4f}\n")

    df_epochs = pd.DataFrame({
        "epoch": np.arange(1, len(h_loss)+1),
        "train_loss": h_loss,
        "val_loss": h_val_loss,
        "train_acc": h_acc,
        "val_acc": h_val_acc
    })
    df_epochs.to_csv("epochs_results.csv", index=False)
    save_txt_table(df_epochs, "epochs_results.txt", max_rows=120)

    # Best epoch (by val_acc over both phases)
    best_epoch = int(np.argmax(h_val_acc) + 1)
    with open("best_epoch.txt", "w") as f: f.write(str(best_epoch))

    artifacts = {
        "hyperparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "init_lr": INIT_LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTH,
            "clipnorm": CLIPNORM,
            "patience": PATIENCE,
            "val_split": VAL_SPLIT,
            "aug_intensity": AUG_INTENSITY
        },
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auc_micro": float(auc_micro),
        "auc_macro": float(auc_macro),
    }
    with open("best_cnn_params.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    # Done
    total = time.time() - t0
    print("\nSaved: model_* plots, roc_curve.png, confusion_matrix.png, f1_scores.png, "
          "classification_report.txt, epochs_results.csv/txt, sample_prediction.png, top5_errors.png, "
          "best_mnist_cnn.keras, best_mnist_cnn_final.keras, best_epoch.txt, best_cnn_params.json")
    print(f"=== Total runtime: {total:.1f} s ===")

if __name__ == "__main__":
    main()
