# different_models.py
# -----------------------------------------------------------------------------
# MNIST classical models with Optuna (no Logistic Regression).
#
# Models:
#   - decision_tree
#   - random_forest
#   - xgboost                (skips gracefully if xgboost unavailable)
#   - lightgbm               (skips gracefully if lightgbm unavailable)
#
# Per method saves under learning_models/<method>/ :
#   runtime_env.txt, best_params.json, classification_report.txt,
#   confusion_matrix.png, roc_curve.png, model_accuracy.png, model_loss.png,
#   epochs_results.csv, epochs_results.txt, sample_prediction.png
#
# Prints a leaderboard at the end.
#
# pip install optuna scikit-learn matplotlib seaborn pandas psutil
# Optional: pip install xgboost lightgbm
# -----------------------------------------------------------------------------

import os, time, json, random, platform, warnings
from pathlib import Path

# --- avoid Tkinter GUI backend crashes on Windows ---
import matplotlib
matplotlib.use("Agg")  # headless backend
# ----------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF logs if TF exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, f1_score,
    log_loss, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge

import optuna

try:
    import psutil
except Exception:
    psutil = None

# Optional libs (skip gracefully if missing)
try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

# -----------------------
# Global knobs
# -----------------------
SEED = 42
VAL_FRAC = 0.15      # from the 60k training split
N_SPLITS_LC = 3      # CV folds for learning curves (keep small = faster)
TRAIN_SIZES = np.linspace(0.2, 1.0, 5)

# Trial counts (can override via env vars)
N_TRIALS_TREE   = int(os.getenv("TRIALS_TREE",   "40"))
N_TRIALS_RF     = int(os.getenv("TRIALS_RF",     "48"))
N_TRIALS_BOOST  = int(os.getenv("TRIALS_BOOST",  "64"))  # xgb & lgb

MAX_TRAIN_FOR_TSNE = int(os.getenv("TSNE_TRAIN_MAX", "20000"))
TSNE_ITER = int(os.getenv("TSNE_ITER", "500"))
TSNE_PERPLEXITY = int(os.getenv("TSNE_PERP", "30"))
RANDOM_STATE = np.random.RandomState(SEED)

# -----------------------
# Reproducibility
# -----------------------
def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(SEED)

# -----------------------
# Runtime / Hardware
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
    # Prefer TF's MNIST (fast & cached)
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

    # Flatten pixels
    X_tr_f = X_tr.reshape(len(X_tr), -1)
    X_val_f = X_val.reshape(len(X_val), -1)
    X_te_f  = x_test.reshape(len(x_test), -1)

    # Standardize (left here; harmless even if PCA not used)
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_f)
    X_val_std = scaler.transform(X_val_f)
    X_te_std  = scaler.transform(X_te_f)

    data = {
        "X_tr_img": X_tr, "X_val_img": X_val, "X_te_img": x_test,
        "y_tr": y_tr, "y_val": y_val, "y_te": y_test,
        "X_tr": X_tr_f, "X_val": X_val_f, "X_te": X_te_f,
        "X_tr_std": X_tr_std, "X_val_std": X_val_std, "X_te_std": X_te_std
    }
    return data

# -----------------------
# Plot helpers
# -----------------------
def save_txt_table(df: pd.DataFrame, path_txt: Path, max_rows=200):
    with open(path_txt, "w") as f:
        f.write(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            f.write(f"\n... ({len(df)-max_rows} more rows)")

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

def plot_learning_curves(clf, X, y, out_dir: Path, name: str,
                         proba_supported: bool):
    cv = StratifiedKFold(n_splits=N_SPLITS_LC, shuffle=True, random_state=SEED)
    sizes, train_scores, val_scores = learning_curve(
        clf, X, y, cv=cv, train_sizes=TRAIN_SIZES, scoring="accuracy", n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(sizes, train_mean, marker="o", label="train_acc")
    plt.plot(sizes, val_mean, marker="o", label="cv_acc")
    plt.title("Learning Curve (Accuracy)"); plt.xlabel("Train size"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "model_accuracy.png"); plt.close()

    loss_train_mean, loss_val_mean = [np.full_like(train_mean, np.nan, dtype=float) for _ in range(2)]
    if proba_supported:
        sizes2, train_scores2, val_scores2 = learning_curve(
            clf, X, y, cv=cv, train_sizes=TRAIN_SIZES, scoring="neg_log_loss", n_jobs=-1
        )
        if not np.all(sizes2 == sizes):
            sizes = sizes2
        loss_train_mean = -train_scores2.mean(axis=1)
        loss_val_mean = -val_scores2.mean(axis=1)

        plt.figure(figsize=(6,4), dpi=120)
        plt.plot(sizes, loss_train_mean, marker="o", label="train_logloss")
        plt.plot(sizes, loss_val_mean, marker="o", label="cv_logloss")
        plt.title("Learning Curve (Log Loss)"); plt.xlabel("Train size"); plt.ylabel("Log Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "model_loss.png"); plt.close()
    else:
        plt.figure(figsize=(6,4), dpi=120)
        plt.plot(sizes, 1-train_mean, marker="o", label="1 - train_acc")
        plt.plot(sizes, 1-val_mean, marker="o", label="1 - cv_acc")
        plt.title("Learning Curve (Proxy Loss)"); plt.xlabel("Train size"); plt.ylabel("Proxy Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "model_loss.png"); plt.close()

    df_lc = pd.DataFrame({
        "train_size": sizes,
        "train_acc_mean": train_mean,
        "val_acc_mean": val_mean,
        "train_logloss_mean": loss_train_mean,
        "val_logloss_mean": loss_val_mean
    })
    df_lc.to_csv(out_dir / "epochs_results.csv", index=False)
    save_txt_table(df_lc, out_dir / "epochs_results.txt", max_rows=200)

# -----------------------
# Booster fit helpers (version-robust early stopping)
# -----------------------
def xgb_fit_with_es(clf, X_tr, y_tr, X_val, y_val, rounds=30, verbose=False):
    """
    Works across XGBoost versions:
      - Prefer setting eval_metric via set_params (not fit kwarg) to avoid older
        wrappers raising TypeError('eval_metric' unexpected).
      - Try early_stopping_rounds; if not supported, fall back to callbacks; if
        still not available, fit without early stopping.
    """
    # Prefer metric as an estimator param
    try:
        clf.set_params(eval_metric="mlogloss")
    except Exception:
        pass

    try:
        return clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=rounds,
            verbose=verbose
        )
    except TypeError:
        try:
            from xgboost.callback import EarlyStopping as XgbEarlyStopping
            return clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[XgbEarlyStopping(rounds=rounds, save_best=True)],
                verbose=verbose
            )
        except Exception:
            return clf.fit(X_tr, y_tr)

def lgb_fit_with_es(clf, X_tr, y_tr, X_val, y_val, rounds=30, verbose=False):
    """LightGBM ES across wrapper variants (callbacks or kwarg)."""
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=rounds, verbose=verbose)]
        return clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=callbacks
        )
    except Exception:
        try:
            return clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                early_stopping_rounds=rounds
            )
        except Exception:
            return clf.fit(X_tr, y_tr)

# -----------------------
# Generic evaluation/saving
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_runtime(out_dir: Path):
    info = runtime_env_str()
    with open(out_dir / "runtime_env.txt", "w") as f:
        f.write(info + "\n")
    print(info)

def evaluate_and_save(name, clf, X_tr_like, X_val_like, X_te_like, y_tr, y_val, y_te,
                      out_dir: Path, params_dict, proba_supported=True,
                      x_te_img=None, extra_plots_fn=None):
    t_start = time.time()

    # Fit on train+val (keep test clean)
    X_full = np.vstack([X_tr_like, X_val_like])
    y_full = np.concatenate([y_tr, y_val])

    clf.fit(X_full, y_full)
    fit_time = time.time() - t_start

    # Predict
    y_pred = clf.predict(X_te_like)
    if proba_supported and hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te_like)
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
    auc_mic, auc_mac = plot_roc_multiclass(y_te, y_prob, out_dir / "roc_curve.png", title_prefix=name)

    # Plots & learning curves
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_learning_curves(clf, X_full, y_full, out_dir, name,
        proba_supported=proba_supported and hasattr(clf, "predict_proba"))

    # Sample image
    mid = len(X_te_like) // 2
    p_mid = y_prob[mid, :].max()
    yhat_mid = int(np.argmax(y_prob[mid, :]))
    plt.figure(figsize=(3,3), dpi=160)
    if x_te_img is not None:
        plt.imshow(x_te_img[mid], cmap="gray")
    else:
        try:
            plt.imshow(X_te_like[mid].reshape(28,28), cmap="gray")
        except Exception:
            plt.imshow(np.zeros((28,28)), cmap="gray")
    plt.axis("off")
    plt.title(f"Sample idx={mid}\ntrue={y_te[mid]}, pred={yhat_mid}, p={p_mid:.3f}")
    plt.tight_layout(); plt.savefig(out_dir / "sample_prediction.png"); plt.close()

    if extra_plots_fn is not None:
        extra_plots_fn(out_dir)

    # Reports
    report = classification_report(y_te, y_pred, digits=4)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(runtime_env_str() + "\n")
        f.write(f"Split: train={1-VAL_FRAC:.2f} of 60k, val={VAL_FRAC:.2f} of 60k, test=10k original\n")
        f.write(json.dumps(params_dict, indent=2) + "\n")
        f.write(report + "\n")
        f.write(f"F1 micro={f1_mic:.4f} | macro={f1_mac:.4f}\n")
        f.write(f"Test acc={acc:.4f}, logloss={test_logloss:.4f}\n")
        f.write(f"AUC micro={auc_mic:.4f}, macro={auc_mac:.4f}\n")
        f.write(f"Runtime fit(s)={fit_time:.2f}\n")

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(params_dict, f, indent=2)

    metrics = {
        "name": name,
        "test_accuracy": acc,
        "f1_macro": f1_mac,
        "f1_micro": f1_mic,
        "auc_macro": auc_mac,
        "logloss": test_logloss,
        "runtime_seconds": fit_time
    }
    print(f"[{name}] done. acc={acc:.4f}  F1_macro={f1_mac:.4f}  AUC_macro={auc_mac:.4f}  time={fit_time:.1f}s")
    return metrics

# -----------------------
# Tuners (expanded search spaces)
# -----------------------
def tune_decision_tree(Xt, yt, Xv, yv, n_trials=N_TRIALS_TREE):
    def obj(trial):
        params = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 8, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            ccp_alpha=trial.suggest_float("ccp_alpha", 1e-6, 1e-3, log=True),
            random_state=SEED
        )
        clf = DecisionTreeClassifier(**params)
        clf.fit(Xt, yt)
        return float(accuracy_score(yv, clf.predict(Xv)))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best["random_state"] = SEED
    final_clf = DecisionTreeClassifier(**best)
    return final_clf, best

def tune_random_forest(Xt, yt, Xv, yv, n_trials=N_TRIALS_RF):
    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 700),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            max_depth=trial.suggest_int("max_depth", 12, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
            n_jobs=-1, random_state=SEED
        )
        clf = RandomForestClassifier(**params)
        clf.fit(Xt, yt)
        return float(accuracy_score(yv, clf.predict(Xv)))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best.update({"n_jobs": -1, "random_state": SEED})
    final_clf = RandomForestClassifier(**best)
    return final_clf, best

def tune_xgb(Xt, yt, Xv, yv, n_trials=N_TRIALS_BOOST):
    if not XGB_OK:
        return None, {"note": "xgboost unavailable"}

    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 900),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            colsample_bynode=trial.suggest_float("colsample_bynode", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 8.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            max_bin=trial.suggest_int("max_bin", 128, 512),
            grow_policy=trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            tree_method="hist",
            objective="multi:softprob",
            num_class=10,
            n_jobs=-1,
            random_state=SEED
        )
        clf = xgb.XGBClassifier(**params)
        xgb_fit_with_es(clf, Xt, yt, Xv, yv, rounds=40, verbose=False)
        return float(accuracy_score(yv, clf.predict(Xv)))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best.update({"tree_method": "hist", "objective": "multi:softprob",
                 "num_class": 10, "n_jobs": -1, "random_state": SEED})
    final_clf = xgb.XGBClassifier(**best)
    return final_clf, best

def tune_lgbm(Xt, yt, Xv, yv, n_trials=N_TRIALS_BOOST):
    if not LGB_OK:
        return None, {"note": "lightgbm unavailable"}

    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 900),
            max_depth=trial.suggest_int("max_depth", -1, 24),  # -1 = no limit
            num_leaves=trial.suggest_int("num_leaves", 31, 512),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),               # bagging_fraction
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0), # feature_fraction
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 8.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-3, 1.0, log=True),
            objective="multiclass",
            n_jobs=-1,
            random_state=SEED
        )
        clf = lgb.LGBMClassifier(**params)
        lgb_fit_with_es(clf, Xt, yt, Xv, yv, rounds=40, verbose=False)
        return float(accuracy_score(yv, clf.predict(Xv)))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    best = study.best_params
    best.update({"objective": "multiclass", "n_jobs": -1, "random_state": SEED})
    final_clf = lgb.LGBMClassifier(**best)
    return final_clf, best

# -----------------------
# Generic method runner
# -----------------------
def run_method(name, tuner_fn, X_for_fit, X_for_val, X_for_test, y_tr, y_val, y_te,
               root: Path, proba_supported=True, x_te_img=None):
    out_dir = root / name
    ensure_dir(out_dir)
    print(f"\n[{name}]")
    write_runtime(out_dir)

    clf, best_params = tuner_fn(X_for_fit, y_tr, X_for_val, y_val)

    return evaluate_and_save(
        name, clf, X_for_fit, X_for_val, X_for_test,
        y_tr, y_val, y_te, out_dir, best_params,
        proba_supported=proba_supported, x_te_img=x_te_img
    )

# -----------------------
# Main
# -----------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    print(runtime_env_str())

    data = load_mnist()

    root = Path("learning_models")
    ensure_dir(root)

    Xtr_raw, Xv_raw, Xte_raw = data["X_tr"], data["X_val"], data["X_te"]
    Xtr_std, Xv_std, Xte_std = data["X_tr_std"], data["X_val_std"], data["X_te_std"]
    ytr, yv, yte = data["y_tr"], data["y_val"], data["y_te"]

    results = []

    # Decision Tree (raw pixels)
    results.append(run_method(
        "decision_tree", tune_decision_tree, Xtr_raw, Xv_raw, Xte_raw, ytr, yv, yte, root,
        proba_supported=True, x_te_img=data["X_te_img"]
    ))

    # Random Forest (raw pixels)
    results.append(run_method(
        "random_forest", tune_random_forest, Xtr_raw, Xv_raw, Xte_raw, ytr, yv, yte, root,
        proba_supported=True, x_te_img=data["X_te_img"]
    ))

    # XGBoost (raw pixels) - skip if unavailable
    if XGB_OK:
        results.append(run_method(
            "xgboost", tune_xgb, Xtr_raw, Xv_raw, Xte_raw, ytr, yv, yte, root,
            proba_supported=True, x_te_img=data["X_te_img"]
        ))
    else:
        print("[xgboost] Skipped (package not available).")

    # LightGBM (raw pixels) - skip if unavailable
    if LGB_OK:
        results.append(run_method(
            "lightgbm", tune_lgbm, Xtr_raw, Xv_raw, Xte_raw, ytr, yv, yte, root,
            proba_supported=True, x_te_img=data["X_te_img"]
        ))
    else:
        print("[lightgbm] Skipped (package not available).")

    # Leaderboard
    df = pd.DataFrame(results).sort_values(
        ["f1_macro", "test_accuracy", "runtime_seconds"], ascending=[False, False, True]
    )
    print("\n=== Leaderboard (by F1_macro → Acc → Runtime) ===")
    print(df.to_string(index=False))

    df.to_csv(root / "leaderboard.csv", index=False)
    with open(root / "leaderboard.txt", "w") as f:
        f.write("=== Leaderboard (by F1_macro → Acc → Runtime) ===\n")
        f.write(df.to_string(index=False) + "\n")

if __name__ == "__main__":
    main()
