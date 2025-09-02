# mnist_tsne_unsupervised.py
# -----------------------------------------------------------------------------
# Unsupervised t-SNE embedding on MNIST + clustering in the t-SNE space.
# No labels are used for embedding or clustering; labels are only used later
# for evaluation (ARI/NMI/etc.) and a mapped confusion matrix.
#
# Artifacts saved to ./tsne_unsupervised :
#   runtime_env.txt, run_params.json, summary.txt
#   clusters_scatter_pred.png      (colored by predicted clusters)
#   clusters_scatter_true.png      (colored by true digits - evaluation only)
#   cluster_counts.png
#   confusion_matrix.png           (after cluster->digit mapping)
#
# Speed knobs via env vars:
#   TSNE_SUBSET=20000          # how many samples to embed/cluster
#   TSNE_PCA_COMP=50           # PCA dim before t-SNE (0 disables PCA)
#   TSNE_PERPLEXITIES=30,50    # grid to try; comma-separated
#   TSNE_K_RANGE=8,9,10,11,12  # KMeans k values to try; comma-separated
#   TSNE_ITER=1000             # (if your sklearn supports n_iter; otherwise ignored)
#
# pip install scikit-learn matplotlib seaborn pandas psutil scipy
# -----------------------------------------------------------------------------

import os, json, time, random, platform, warnings
from pathlib import Path

# Avoid Tk GUI backend issues on Windows
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, confusion_matrix,
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, classification_report,
    accuracy_score
)

# Hungarian assignment to map clusters -> digits for interpretability
from scipy.optimize import linear_sum_assignment as hungarian

try:
    import psutil
except Exception:
    psutil = None

# -----------------------
# Config (env overridable)
# -----------------------
SEED = 42
RNG = np.random.RandomState(SEED)

SUBSET = int(os.getenv("TSNE_SUBSET", "20000"))
N_PCA  = int(os.getenv("TSNE_PCA_COMP", "50"))  # set 0 to disable PCA pre-reduction

def _parse_int_list(envname, default_csv):
    s = os.getenv(envname, default_csv)
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return [int(x) for x in default_csv.split(",")]

def _parse_float_list(envname, default_csv):
    s = os.getenv(envname, default_csv)
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return [float(x) for x in default_csv.split(",")]

PERPLEXITIES = _parse_float_list("TSNE_PERPLEXITIES", "30,50")
K_RANGE      = _parse_int_list("TSNE_K_RANGE", "8,9,10,11,12")
TSNE_ITER    = int(os.getenv("TSNE_ITER", "1000"))  # used only if supported

OUT_DIR = Path("learning_models/tsne_unsupervised")

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
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            dev = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
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

def write_runtime(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    info = runtime_env_str()
    with open(out_dir / "runtime_env.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(info + "\n")
    print(info)

# -----------------------
# Data
# -----------------------
def load_mnist_flat():
    # Prefer TF's MNIST (fast/cached)
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

    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 784)
    x_test  = (x_test.astype("float32")  / 255.0).reshape(-1, 784)
    X = np.vstack([x_train, x_test])
    y = np.hstack([y_train, y_test])
    return X, y

# -----------------------
# t-SNE helper (robust to sklearn versions)
# -----------------------
def fit_tsne(X, perplexity=30.0, random_state=42, n_iter=1000):
    """
    Version-robust t-SNE:
    - Only passes kwargs that your sklearn.manifold.TSNE actually supports.
    - Uses numeric learning_rate (200.0) to avoid 'auto' incompatibilities.
    - Returns (Z, used_iter) where used_iter is None if your TSNE doesn't support it.
    """
    from sklearn.manifold import TSNE
    import inspect

    # What parameters does this sklearn build support?
    sig_params = set(inspect.signature(TSNE.__init__).parameters.keys())

    params = {
        "n_components": 2,
        "perplexity": float(perplexity),
        "random_state": random_state,
    }
    if "init" in sig_params:
        params["init"] = "pca"
    if "learning_rate" in sig_params:
        # 'auto' may not be supported on older sklearn; numeric is always safe
        params["learning_rate"] = 200.0
    if "early_exaggeration" in sig_params:
        params["early_exaggeration"] = 12.0
    if "n_iter" in sig_params:
        params["n_iter"] = int(n_iter)
    if "metric" in sig_params:
        params["metric"] = "euclidean"
    if "square_distances" in sig_params:
        params["square_distances"] = True
    if "verbose" in sig_params:
        params["verbose"] = 0
    if "method" in sig_params:
        # Let sklearn choose default; if you want to force, uncomment:
        # params["method"] = "barnes_hut"
        pass

    tsne = TSNE(**params)
    Z = tsne.fit_transform(X)
    used_iter = params.get("n_iter", None)
    return Z, used_iter


# -----------------------
# Cluster -> label mapping (Hungarian)
# -----------------------
def map_clusters_to_digits(y_true, clusters, n_classes=10):
    labels_unique = np.unique(clusters)
    K = len(labels_unique)
    # Build cost matrix (rows=clusters, cols=digits), cost = -count
    cost = np.zeros((K, n_classes), dtype=int)
    idx_of = {c: i for i, c in enumerate(labels_unique)}
    for c in labels_unique:
        i = idx_of[c]
        mask = clusters == c
        lab, cnt = np.unique(y_true[mask], return_counts=True)
        for l, ct in zip(lab, cnt):
            cost[i, int(l)] = -int(ct)
    rows, cols = hungarian(cost)
    mapping = { int(labels_unique[i]): int(cols[j]) for j, i in enumerate(rows) }
    return mapping

# -----------------------
# Plots
# -----------------------
def scatter_tsne(Z, colors, out_png: Path, title="t-SNE (2D)"):
    plt.figure(figsize=(7,6), dpi=130)
    # If too many points, sample for the figure for speed/size
    n = len(Z)
    max_plot = 12000
    if n > max_plot:
        idx = np.random.RandomState(123).choice(n, size=max_plot, replace=False)
    else:
        idx = np.arange(n)
    sc = plt.scatter(Z[idx,0], Z[idx,1], c=colors[idx], s=6, alpha=0.7, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_confusion(cm, out_png: Path):
    plt.figure(figsize=(7,6), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix (cluster->digit mapped)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_cluster_counts(labels, out_png: Path):
    vc = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(7,4), dpi=120)
    vc.plot(kind="bar")
    plt.title("Cluster sizes (KMeans in t-SNE space)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# -----------------------
# Main
# -----------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(runtime_env_str())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_runtime(OUT_DIR)

    # 1) Load & standardize
    X, y = load_mnist_flat()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 2) Optional PCA pre-reduction (speed + denoise)
    if N_PCA and N_PCA > 0:
        pca = PCA(n_components=N_PCA, random_state=SEED)
        Xf = pca.fit_transform(Xs)
        reducer_note = {"use_pca": True, "n_components": int(N_PCA)}
    else:
        Xf = Xs
        pca = None
        reducer_note = {"use_pca": False, "n_components": None}

    # 3) Subsample for embedding (speed)
    n = len(Xf)
    take = min(SUBSET, n)
    idx = RNG.choice(n, size=take, replace=False)
    Xc, yc = Xf[idx], y[idx]

    # 4) Grid over perplexities and KMeans k to pick the best (unsupervised)
    best = None
    t0 = time.time()
    for perp in PERPLEXITIES:
        Z, used_iter = fit_tsne(Xc, perplexity=perp, random_state=SEED, n_iter=TSNE_ITER)
        # Try a few k and score by silhouette; tie-break by Davies-Bouldin (lower better)
        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            clusters = km.fit_predict(Z)
            # Silhouette requires at least 2 clusters and more samples than clusters
            try:
                sil = silhouette_score(Z, clusters, metric="euclidean")
            except Exception:
                sil = -1.0
            try:
                dbi = davies_bouldin_score(Z, clusters)
            except Exception:
                dbi = np.inf

            record = {
                "perplexity": float(perp),
                "k": int(k),
                "silhouette": float(sil),
                "db_index": float(dbi),
                "clusters": clusters,
                "Z": Z,
                "used_iter": used_iter
            }
            if (best is None or
                (record["silhouette"] > best["silhouette"]) or
                (np.isclose(record["silhouette"], best["silhouette"]) and record["db_index"] < best["db_index"])):
                best = record

    fit_time = time.time() - t0

    # Best embedding & clustering
    Z = best["Z"]
    clusters = best["clusters"]
    perp = best["perplexity"]
    k = best["k"]
    sil = best["silhouette"]
    dbi = best["db_index"]

    # 5) External evaluation vs. ground truth (post-hoc, for reporting only)
    ari  = adjusted_rand_score(yc, clusters)
    ami  = adjusted_mutual_info_score(yc, clusters, average_method="arithmetic")
    nmi  = normalized_mutual_info_score(yc, clusters, average_method="arithmetic")
    homo = homogeneity_score(yc, clusters)
    comp = completeness_score(yc, clusters)
    vms  = v_measure_score(yc, clusters)

    # Map clusters -> digits for human-readable confusion matrix & "mapped accuracy"
    mapping = map_clusters_to_digits(yc, clusters, n_classes=10)
    y_pred_digits = np.array([mapping.get(int(c), 0) for c in clusters])
    cm = confusion_matrix(yc, y_pred_digits)
    acc = accuracy_score(yc, y_pred_digits)

    # 6) Plots
    plot_cluster_counts(clusters, OUT_DIR / "cluster_counts.png")
    scatter_tsne(Z, clusters, OUT_DIR / "clusters_scatter_pred.png",
                 title=f"t-SNE (perp={perp}, k={k}) — colored by clusters")
    scatter_tsne(Z, yc, OUT_DIR / "clusters_scatter_true.png",
                 title=f"t-SNE (perp={perp}, k={k}) — colored by true digits")
    plot_confusion(cm, OUT_DIR / "confusion_matrix.png")

    # 7) Save params + summary
    params_payload = {
        "subset_used": int(take),
        "reducer": reducer_note,
        "tsne": {
            "chosen_perplexity": float(perp),
            "used_iter": (int(best["used_iter"]) if best["used_iter"] is not None else None),
            "perplexity_grid": PERPLEXITIES
        },
        "kmeans": {
            "chosen_k": int(k),
            "k_grid": K_RANGE
        },
        "scores": {
            "silhouette": float(sil),
            "davies_bouldin": float(dbi)
        },
        "external_eval": {
            "ARI": float(ari),
            "AMI": float(ami),
            "NMI": float(nmi),
            "homogeneity": float(homo),
            "completeness": float(comp),
            "v_measure": float(vms),
            "mapped_accuracy": float(acc)
        },
        "cluster_to_digit_map": {int(a): int(b) for a, b in mapping.items()},
        "runtime_seconds": float(fit_time),
        "note": "All text saved with UTF-8; figures use headless backend."
    }
    with open(OUT_DIR / "run_params.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(params_payload, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(runtime_env_str() + "\n\n")
        f.write(f"Subset embedded: {take} samples (out of {len(X)} total)\n")
        f.write(f"PCA: {reducer_note}\n")
        f.write(f"Chosen t-SNE perplexity: {perp}\n")
        f.write(f"Chosen KMeans k: {k}\n")
        f.write(f"Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}\n")
        f.write(f"ARI={ari:.4f}, AMI={ami:.4f}, NMI={nmi:.4f}\n")
        f.write(f"Homogeneity={homo:.4f}, Completeness={comp:.4f}, V-measure={vms:.4f}\n")
        f.write(f"Mapped accuracy (cluster->digit) = {acc:.4f}\n\n")
        f.write("Classification report after cluster->digit mapping (interpretability):\n")
        f.write(classification_report(yc, y_pred_digits, digits=4))

    print("\n[t-SNE unsupervised] results")
    print(f"  subset={take}, perp={perp}, k={k}")
    print(f"  silhouette={sil:.4f}, DB-index={dbi:.4f}")
    print(f"  ARI={ari:.4f}, AMI={ami:.4f}, NMI={nmi:.4f}, "
          f"Homog={homo:.4f}, Compl={comp:.4f}, V={vms:.4f}")
    print(f"  Mapped accuracy={acc:.4f}   (interpretability only)")
    print(f"  Artifacts saved in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
