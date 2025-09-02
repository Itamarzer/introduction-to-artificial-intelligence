# mnist_mlp_regularization_sweep.py
# -----------------------------------------------------------------------------
# Eight MNIST MLP experiments to compare: baseline, only L2, only Dropout,
# only EarlyStopping, each pairwise combo, and all three together.
#
# Saves (per run under runs/<NAME>/):
#   model_accuracy.png, model_loss.png, roc_curve.png
#   confusion_matrix.png, f1_scores.png, sample_prediction.png
#   classification_report.txt, epochs_results.csv, epochs_results.txt
#   best_epoch.txt, best_params.json, best_mlp.keras
#
# Also prints a final leaderboard comparing all runs.
#
# pip install tensorflow matplotlib seaborn scikit-learn pandas psutil
# -----------------------------------------------------------------------------

import os, time, json, random, platform
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import psutil
except Exception:
    psutil = None

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, f1_score
)

# -----------------------
# Global knobs
# -----------------------
SEED = 42
EPOCHS = 12          # keep modest for CPU
BATCH_SIZE = 256
INIT_LR = 1.5e-3
L2_COEFF = 5e-4      # used when L2 enabled
DROPOUT_RATE = 0.4   # used when Dropout enabled
PATIENCE = 5         # used when EarlyStopping enabled
VAL_FRAC = 0.15      # from the 60k training split

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
# Runtime / Hardware
# -----------------------
def runtime_env_str():
    gpu_list = tf.config.list_physical_devices("GPU")
    if gpu_list:
        dev = tf.config.experimental.get_device_details(gpu_list[0]).get("device_name", "GPU")
        accel = f"GPU ({dev})"
    else:
        accel = "CPU (no GPU visible)"
    lines = [
        "=== Runtime Environment ===",
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {platform.python_version()}",
        f"TensorFlow: {tf.__version__}",
        f"Accelerator: {accel}",
    ]
    if psutil:
        lines.append(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    return "\n".join(lines)

# -----------------------
# Data
# -----------------------
def load_data():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train_int), (x_test, y_test_int) = mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]  # (N,28,28,1)
    x_test  = (x_test.astype("float32")  / 255.0)[..., None]
    # train/val split from the original train set
    X_tr, X_val, y_tr_int, y_val_int = train_test_split(
        x_train, y_train_int, test_size=VAL_FRAC, random_state=SEED, stratify=y_train_int
    )
    y_tr  = to_categorical(y_tr_int,  num_classes=10).astype("float32")
    y_val = to_categorical(y_val_int, num_classes=10).astype("float32")
    y_test = to_categorical(y_test_int, num_classes=10).astype("float32")
    return (X_tr, y_tr, y_tr_int), (X_val, y_val, y_val_int), (x_test, y_test, y_test_int)

# -----------------------
# Model builder (MLP)
# -----------------------
def build_mlp(input_shape=(28,28,1), use_l2=False, use_dropout=False):
    reg = regularizers.l2(L2_COEFF) if use_l2 else None
    inp = layers.Input(shape=input_shape)
    x = layers.Flatten()(inp)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    if use_dropout: x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    if use_dropout: x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inp, out, name="MNIST_MLP")
    opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -----------------------
# Plots
# -----------------------
def plot_curves(hist, out_dir: Path):
    e = range(1, len(hist.history["loss"])+1)
    # Accuracy
    plt.figure(figsize=(6,4), dpi=110)
    plt.plot(e, hist.history["accuracy"], label="train_acc")
    plt.plot(e, hist.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"model_accuracy.png"); plt.close()
    # Loss
    plt.figure(figsize=(6,4), dpi=110)
    plt.plot(e, hist.history["loss"], label="train_loss")
    plt.plot(e, hist.history["val_loss"], label="val_loss")
    plt.title("Loss per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"model_loss.png"); plt.close()

def plot_confusion(cm, out_dir: Path):
    plt.figure(figsize=(7,6), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out_dir/"confusion_matrix.png"); plt.close()

def plot_f1_bars(y_true, y_pred, out_dir: Path):
    f1s = [f1_score(y_true==(c), y_pred==(c)) for c in range(10)]
    plt.figure(figsize=(7,4), dpi=120)
    plt.bar(range(10), f1s)
    plt.xticks(range(10)); plt.ylim(0.9, 1.0)
    plt.title("Per-class F1"); plt.tight_layout(); plt.savefig(out_dir/"f1_scores.png"); plt.close()

def plot_roc_curves(y_true_oh, y_prob, out_dir: Path):
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
    plt.tight_layout(); plt.savefig(out_dir/"roc_curve.png"); plt.close()
    return roc_auc["micro"], macro_auc

def save_txt_table(df: pd.DataFrame, path_txt: Path, max_rows=120):
    with open(path_txt, "w") as f:
        f.write(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            f.write(f"\n... ({len(df)-max_rows} more rows)")

# -----------------------
# Single experiment
# -----------------------
def run_experiment(name, use_l2, use_dropout, use_es,
                   data, out_root: Path):
    (X_tr, y_tr, y_tr_int), (X_val, y_val, y_val_int), (x_test, y_test, y_test_int) = data
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log runtime/hardware for this model
    env_txt = out_dir / "runtime_env.txt"
    env_info = runtime_env_str()
    with open(env_txt, "w") as f: f.write(env_info + "\n")
    print(f"\n[{name}]")
    print(env_info)

    model = build_mlp(use_l2=use_l2, use_dropout=use_dropout)

    callbacks = []
    if use_es:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True, verbose=1))
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=2, min_lr=1e-5, verbose=1))
    callbacks.append(ModelCheckpoint(out_dir/"best_mlp.keras", monitor="val_accuracy",
                                     save_best_only=True, verbose=1))

    t0 = time.time()
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    runtime_s = time.time() - t0

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred_int = np.argmax(y_prob, axis=1)

    # Reports / plots
    plot_curves(hist, out_dir)
    cm = confusion_matrix(y_test_int, y_pred_int)
    plot_confusion(cm, out_dir)
    plot_f1_bars(y_test_int, y_pred_int, out_dir)
    auc_micro, auc_macro = plot_roc_curves(y_test, y_prob, out_dir)

    # sample prediction
    mid = len(x_test)//2
    p_mid = y_prob[mid].max(); yhat_mid = int(np.argmax(y_prob[mid]))
    plt.figure(figsize=(3,3), dpi=160)
    plt.imshow(x_test[mid].squeeze(-1), cmap="gray"); plt.axis("off")
    plt.title(f"Sample idx={mid}\ntrue={y_test_int[mid]}, pred={yhat_mid}, p={p_mid:.3f}")
    plt.tight_layout(); plt.savefig(out_dir/"sample_prediction.png"); plt.close()

    # Text outputs
    report = classification_report(y_test_int, y_pred_int, digits=4)
    with open(out_dir/"classification_report.txt", "w") as f:
        f.write(env_info + "\n")
        f.write(f"Split: train={1-VAL_FRAC:.2f} of 60k, val={VAL_FRAC:.2f} of 60k, test=10k original\n")
        f.write(report + "\n")
        f.write(f"F1 micro={f1_score(y_test_int, y_pred_int, average='micro'):.4f} | "
                f"macro={f1_score(y_test_int, y_pred_int, average='macro'):.4f} | "
                f"weighted={f1_score(y_test_int, y_pred_int, average='weighted'):.4f}\n")
        f.write(f"Test acc={test_acc:.4f}, loss={test_loss:.4f}\n")
        f.write(f"AUC micro={auc_micro:.4f}, macro={auc_macro:.4f}\n")
        f.write(f"Runtime (s)={runtime_s:.1f}\n")

    # History CSV/TXT
    df_epochs = pd.DataFrame({
        "epoch": np.arange(1, len(hist.history["loss"])+1),
        "train_loss": hist.history["loss"],
        "val_loss": hist.history["val_loss"],
        "train_acc": hist.history["accuracy"],
        "val_acc": hist.history["val_accuracy"]
    })
    df_epochs.to_csv(out_dir/"epochs_results.csv", index=False)
    save_txt_table(df_epochs, out_dir/"epochs_results.txt", max_rows=200)

    # Best epoch
    best_epoch = int(np.argmax(hist.history["val_accuracy"]) + 1)
    with open(out_dir/"best_epoch.txt", "w") as f: f.write(str(best_epoch))

    # Save hyperparams + metrics
    artifacts = {
        "name": name,
        "hyperparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "init_lr": INIT_LR,
            "l2_coeff": L2_COEFF if use_l2 else 0.0,
            "dropout_rate": DROPOUT_RATE if use_dropout else 0.0,
            "early_stopping": bool(use_es),
            "patience": PATIENCE if use_es else 0,
            "val_frac_of_60k": VAL_FRAC,
        },
        "metrics": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "f1_micro": float(f1_score(y_test_int, y_pred_int, average="micro")),
            "f1_macro": float(f1_score(y_test_int, y_pred_int, average="macro")),
            "f1_weighted": float(f1_score(y_test_int, y_pred_int, average="weighted")),
            "auc_micro": float(auc_micro),
            "auc_macro": float(auc_macro),
            "runtime_seconds": float(runtime_s),
        }
    }
    with open(out_dir/"best_params.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"[{name}] done. Test acc={test_acc:.4f}, F1 macro={artifacts['metrics']['f1_macro']:.4f}, "
          f"runtime={runtime_s:.1f}s")
    return artifacts

# -----------------------
# Run all experiments
# -----------------------
def main():
    print(runtime_env_str())
    data = load_data()
    out_root = Path("runs")
    out_root.mkdir(exist_ok=True)

    configs = [
        ("baseline_none",            False, False, False),
        ("only_l2",                  True,  False, False),
        ("only_earlystop",           False, False, True),
        ("only_dropout",             False, True,  False),
        ("l2_plus_earlystop",        True,  False, True),
        ("dropout_plus_earlystop",   False, True,  True),
        ("l2_plus_dropout",          True,  True,  False),
        ("l2_dropout_earlystop",     True,  True,  True),
    ]

    results = []
    for name, use_l2, use_dropout, use_es in configs:
        art = run_experiment(name, use_l2, use_dropout, use_es, data, out_root)
        art["order"] = name
        results.append(art)

    # Leaderboard
    rows = []
    for a in results:
        m = a["metrics"]
        rows.append({
            "name": a["name"],
            "test_acc": m["test_accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_micro": m["f1_micro"],
            "auc_macro": m["auc_macro"],
            "runtime_s": m["runtime_seconds"]
        })
    df = pd.DataFrame(rows).sort_values(["f1_macro", "test_acc"], ascending=False)
    df.to_csv(out_root/"leaderboard.csv", index=False)

    best = df.iloc[0]
    print("\n=== Leaderboard (top 5) by F1_macro, then accuracy ===")
    print(df.head(5).to_string(index=False))
    print(f"\nBest model: {best['name']}  "
          f"(F1_macro={best['f1_macro']:.4f}, acc={best['test_acc']:.4f}, runtime={best['runtime_s']:.1f}s)")
    with open(out_root/"leaderboard.txt", "w") as f:
        f.write("=== Leaderboard by F1_macro, then accuracy ===\n")
        f.write(df.to_string(index=False) + "\n")
        f.write(f"\nBest model: {best['name']}  "
                f"(F1_macro={best['f1_macro']:.4f}, acc={best['test_acc']:.4f}, runtime={best['runtime_s']:.1f}s)\n")

if __name__ == "__main__":
    main()
