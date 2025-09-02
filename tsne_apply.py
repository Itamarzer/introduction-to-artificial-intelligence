# tsne_oos_rf_mnist.py
# -----------------------------------------------------------------------------
# MNIST t-SNE baseline (NO PCA pre-step) + OOS mapping via KNN regressors,
# then RandomForest on the 2-D embedding (Optuna tuning).
#
# Saves under ./tsne/ :
#   runtime_env.txt, best_params.json, classification_report.txt
#   confusion_matrix.png, roc_curve.png, model_accuracy.png, model_loss.png
#   epochs_results.csv, epochs_results.txt, sample_prediction.png
#   embedding_subset_scatter.png
#
# pip install scikit-learn optuna matplotlib seaborn pandas psutil
# (TensorFlow optional; used only to fetch MNIST quickly)
# -----------------------------------------------------------------------------

import os, time, json, random, platform, warnings
from pathlib import Path

# Headless backend to avoid Tkinter crashes on Windows
import matplotlib
matplotlib.use("Agg")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, log_loss, accuracy_score
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

import optuna

try:
    import psutil
except Exception:
    psutil = None

# -----------------------
# Config
# -----------------------
SEED = 42
VAL_FRAC = 0.15

# t-SNE (no PCA)
TSNE_TRAIN_MAX = int(os.getenv("TSNE_TRAIN_MAX", "20000"))  # subset size for fitting t-SNE
TSNE_PERPLEXITY = float(os.getenv("TSNE_PERP", "60"))
TSNE_LEARNING_RATE = float(os.getenv("TSNE_LR", "200"))
TSNE_EE = float(os.getenv("TSNE_EE", "12"))
# Do NOT pass n_iter to TSNE.__init__ to avoid version incompatibilities

# Optuna trials for RF in 2D space
N_TRIALS_RF = int(os.getenv("TRIALS_RF_2D", "40"))

RNG = np.random.RandomState(SEED)


# -----------------------
# Reproducibility
# -----------------------
def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(SEED)


# -----------------------
# Runtime info
# -----------------------
def runtime_env_str():
    accel = "Unknown"
    try:
        import tensorflow as tf
        gpu_list = tf.config.list_physical_devices("GPU")
        if gpu_list:
            dev = tf.config.experimental.get_device_details(gpu_list[0]).get("device_name", "GPU")
            accel = f"GPU ({dev})"
        else:
            accel = "CPU (no GPU visible)"
        tf_ver = tf.__version__
    except Exception:
        accel = "CPU"
        tf_ver = "not available"

    lines = [
        "=== Runtime Environment ===",
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {platform.python_version()}",
        f"TensorFlow: {tf_ver}",
        f"Accelerator: {accel}",
    ]
    if psutil:
        lines.append(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    return "\n".join(lines)


# -----------------------
# Data
# -----------------------
def load_mnist():
    # Prefer TF's MNIST (fast cache)
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception:
        from sklearn.datasets import fetch_openml
        mnist_ = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist_.data.reshape(-1, 28, 28)
        y = mnist_.target.astype(int)
        x_train, x_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

    x_train = (x_train.astype("float32") / 255.0)
    x_test  = (x_test.astype("float32")  / 255.0)

    X_tr, X_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=VAL_FRAC, random_state=SEED, stratify=y_train
    )

    # Flatten -> standardize (t-SNE directly on standardized 784 dims)
    X_tr_f = X_tr.reshape(len(X_tr), -1)
    X_val_f = X_val.reshape(len(X_val), -1)
    X_te_f  = x_test.reshape(len(x_test), -1)

    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_f)
    X_val_std = scaler.transform(X_val_f)
    X_te_std  = scaler.transform(X_te_f)

    return {
        "X_tr_img": X_tr, "X_val_img": X_val, "X_te_img": x_test,
        "y_tr": y_tr, "y_val": y_val, "y_te": y_test,
        "X_tr_std": X_tr_std, "X_val_std": X_val_std, "X_te_std": X_te_std
    }


# -----------------------
# Plots
# -----------------------
def plot_confusion(cm, out_png: Path):
    plt.figure(figsize=(7,6), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_roc_multiclass(y_true, y_prob, out_png: Path, title_prefix=""):
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    macro_auc = float(np.mean([roc_auc[i] for i in range(n_classes)]))

    plt.figure(figsize=(7,6), dpi=120)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], alpha=0.22)
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro (AUC={roc_auc['micro']:.3f})", linewidth=2.3)
    plt.plot([0,1],[0,1],"--", linewidth=1)
    ttl = f"{title_prefix} ROC (macro AUC={macro_auc:.3f})".strip()
    plt.title(ttl); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()
    return roc_auc["micro"], macro_auc

def save_txt_table(df: pd.DataFrame, path_txt: Path, max_rows=200):
    with open(path_txt, "w") as f:
        f.write(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            f.write(f"\n... ({len(df)-max_rows} more rows)")

def plot_learning_curves(clf, X, y, out_dir: Path, proba_supported: bool):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    sizes, train_scores, val_scores = learning_curve(
        clf, X, y, cv=cv, train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="accuracy", n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(sizes, train_mean, marker="o", label="train_acc")
    plt.plot(sizes, val_mean, marker="o", label="cv_acc")
    plt.title("Learning Curve (Accuracy)"); plt.xlabel("Train size"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "model_accuracy.png"); plt.close()

    if proba_supported and hasattr(clf, "predict_proba"):
        sizes2, tr2, va2 = learning_curve(
            clf, X, y, cv=cv, train_sizes=np.linspace(0.2, 1.0, 5),
            scoring="neg_log_loss", n_jobs=-1
        )
        loss_train_mean = -tr2.mean(axis=1)
        loss_val_mean = -va2.mean(axis=1)
        plt.figure(figsize=(6,4), dpi=120)
        plt.plot(sizes2, loss_train_mean, marker="o", label="train_logloss")
        plt.plot(sizes2, loss_val_mean, marker="o", label="cv_logloss")
        plt.title("Learning Curve (Log Loss)"); plt.xlabel("Train size"); plt.ylabel("Log Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "model_loss.png"); plt.close()
        df = pd.DataFrame({
            "train_size": sizes2,
            "train_acc_mean": train_mean,
            "val_acc_mean": val_mean,
            "train_logloss_mean": loss_train_mean,
            "val_logloss_mean": loss_val_mean
        })
    else:
        # Proxy loss = 1 - accuracy
        plt.figure(figsize=(6,4), dpi=120)
        plt.plot(sizes, 1-train_mean, marker="o", label="1 - train_acc")
        plt.plot(sizes, 1-val_mean, marker="o", label="1 - cv_acc")
        plt.title("Learning Curve (Proxy Loss)"); plt.xlabel("Train size"); plt.ylabel("Proxy Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "model_loss.png"); plt.close()
        df = pd.DataFrame({
            "train_size": sizes,
            "train_acc_mean": train_mean,
            "val_acc_mean": val_mean,
            "train_logloss_mean": np.nan,
            "val_logloss_mean": np.nan
        })

    df.to_csv(out_dir / "epochs_results.csv", index=False)
    save_txt_table(df, out_dir / "epochs_results.txt", max_rows=200)


# -----------------------
# t-SNE + OOS mapping (KNN)
# -----------------------
def fit_tsne_subset(X_std, y, out_dir: Path):
    n = len(X_std)
    take = min(TSNE_TRAIN_MAX, n)
    idx = np.arange(n)
    RNG.shuffle(idx)
    idx_sub = idx[:take]

    print(f"[tsne] Fitting t-SNE on {take} standardized train samples (no PCA)...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LEARNING_RATE,
        early_exaggeration=TSNE_EE,
        init="random",
        random_state=SEED,
        method="barnes_hut",
        verbose=0
    )
    Z_sub = tsne.fit_transform(X_std[idx_sub])

    # Save a scatter just for quality inspection (true t-SNE points)
    show = min(5000, len(Z_sub))
    plt.figure(figsize=(6,5), dpi=130)
    plt.scatter(Z_sub[:show, 0], Z_sub[:show, 1],
                c=y[idx_sub][:show], s=6, alpha=0.6, cmap="tab10")
    plt.title("t-SNE (no PCA) on Train Subset")
    plt.tight_layout(); plt.savefig(out_dir / "embedding_subset_scatter.png"); plt.close()

    return idx_sub, Z_sub


def build_oos_knn(X_fit, Z_fit):
    # Distance-weighted KNN for nonlinear OOS mapping (reduces outliers)
    knn_x = KNeighborsRegressor(n_neighbors=30, weights="distance")
    knn_y = KNeighborsRegressor(n_neighbors=30, weights="distance")
    knn_x.fit(X_fit, Z_fit[:, 0])
    knn_y.fit(X_fit, Z_fit[:, 1])

    def transform(X):
        return np.column_stack([knn_x.predict(X), knn_y.predict(X)])
    return transform, {"oos_mapping": "KNNRegressor(n=30, weights='distance')"}


# -----------------------
# RF tuner (2-D space)
# -----------------------
def tune_rf_2d(Z_tr, y_tr, Z_val, y_val, n_trials=N_TRIALS_RF):
    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 600),
            max_depth=trial.suggest_int("max_depth", 4, 32),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
            n_jobs=-1, random_state=SEED
        )
        clf = RandomForestClassifier(**params)
        clf.fit(Z_tr, y_tr)
        return float(accuracy_score(y_val, clf.predict(Z_val)))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best.update({"n_jobs": -1, "random_state": SEED})
    return RandomForestClassifier(**best), best


# -----------------------
# Evaluate & save
# -----------------------
def evaluate_and_save(clf, Z_tr_like, Z_val_like, Z_te_like,
                      y_tr, y_val, y_te, out_dir: Path, params: dict,
                      x_te_img=None):
    t0 = time.time()
    Z_full = np.vstack([Z_tr_like, Z_val_like])
    y_full = np.concatenate([y_tr, y_val])
    clf.fit(Z_full, y_full)
    fit_time = time.time() - t0

    y_pred = clf.predict(Z_te_like)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(Z_te_like)
        test_logloss = float(log_loss(y_te, y_prob, labels=np.unique(y_te)))
    else:
        n_classes = len(np.unique(y_te))
        y_prob = np.zeros((len(y_pred), n_classes), dtype=float)
        for i, c in enumerate(y_pred):
            y_prob[i, c] = 1.0
        test_logloss = float(log_loss(y_te, y_prob, labels=np.arange(n_classes)))

    cm = confusion_matrix(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average="macro")
    f1_mic = f1_score(y_te, y_pred, average="micro")
    acc = accuracy_score(y_te, y_pred)
    auc_mic, auc_mac = plot_roc_multiclass(y_te, y_prob, out_dir / "roc_curve.png", title_prefix="t-SNE RF")

    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_learning_curves(clf, Z_full, y_full, out_dir, proba_supported=True)

    # sample
    mid = len(Z_te_like) // 2
    p_mid = y_prob[mid].max()
    yhat_mid = int(np.argmax(y_prob[mid]))
    plt.figure(figsize=(3,3), dpi=160)
    if x_te_img is not None:
        plt.imshow(x_te_img[mid], cmap="gray")
    else:
        plt.imshow(np.zeros((28,28)), cmap="gray")
    plt.axis("off")
    plt.title(f"Sample idx={mid}\ntrue={y_te[mid]}, pred={yhat_mid}, p={p_mid:.3f}")
    plt.tight_layout(); plt.savefig(out_dir / "sample_prediction.png"); plt.close()

    # report
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(runtime_env_str() + "\n")
        f.write(f"Split: train={1-VAL_FRAC:.2f} of 60k, val={VAL_FRAC:.2f} of 60k, test=10k original\n")
        f.write(json.dumps(params, indent=2) + "\n")
        f.write(classification_report(y_te, y_pred, digits=4) + "\n")
        f.write(f"F1 micro={f1_mic:.4f} | macro={f1_mac:.4f}\n")
        f.write(f"Test acc={acc:.4f}, logloss={test_logloss:.4f}\n")
        f.write(f"AUC micro={auc_mic:.4f}, macro={auc_mac:.4f}\n")
        f.write(f"Runtime fit(s)={fit_time:.2f}\n")

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"[tsne_rf] done. acc={acc:.4f}  F1_macro={f1_mac:.4f}  AUC_macro={auc_mac:.4f}  time={fit_time:.1f}s")


# -----------------------
# Main
# -----------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    out_dir = Path("learning_models/tsne")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[tsne_rf]")
    info = runtime_env_str()
    print(info)
    with open(out_dir / "runtime_env.txt", "w") as f:
        f.write(info + "\n")

    data = load_mnist()
    X_tr, X_val, X_te = data["X_tr_std"], data["X_val_std"], data["X_te_std"]
    y_tr, y_val, y_te = data["y_tr"], data["y_val"], data["y_te"]

    # 1) Fit pure t-SNE (no PCA) on a train subset
    idx_sub, Z_sub = fit_tsne_subset(X_tr, y_tr, out_dir)

    # 2) KNN OOS mapper trained on the true t-SNE subset
    oos_map, oos_meta = build_oos_knn(X_tr[idx_sub], Z_sub)

    # 3) Build embeddings for train/val/test
    Z_tr = oos_map(X_tr)
    Z_val = oos_map(X_val)
    Z_te  = oos_map(X_te)

    # 4) Tune RF on 2D embedding
    clf, best_rf = tune_rf_2d(Z_tr, y_tr, Z_val, y_val, n_trials=N_TRIALS_RF)

    params = {
        "tsne": {
            "subset_size": int(len(idx_sub)),
            "perplexity": TSNE_PERPLEXITY,
            "learning_rate": TSNE_LEARNING_RATE,
            "early_exaggeration": TSNE_EE,
            "init": "random",
            "pca_prestep": False
        },
        **oos_meta,
        "classifier": {"type": "RandomForest", **best_rf}
    }

    # 5) Evaluate & save
    evaluate_and_save(
        clf, Z_tr, Z_val, Z_te, y_tr, y_val, y_te,
        out_dir, params, x_te_img=data["X_te_img"]
    )


if __name__ == "__main__":
    main()
