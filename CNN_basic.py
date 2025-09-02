# fast_mnist_cnn.py
# TensorFlow/Keras CNN for MNIST – fast, with diagnostics & overfitting guards

import os, time, json, random, platform, psutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # hide TF INFO/WARN
# If you want fully deterministic CPU runs (slower): os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks as cb
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -----------------------
# Reproducibility & threads
# -----------------------
SEED = 42
def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
set_seeds()

# Make CPU training snappy (tune if you like)
try:
    ncpu = psutil.cpu_count(logical=True) or 4
    tf.config.threading.set_intra_op_parallelism_threads(ncpu)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, ncpu // 2))
except Exception:
    pass

# Mixed precision if GPU is available
try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# -----------------------
# Runtime / hardware print
# -----------------------
def print_env():
    accel = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print("=== Runtime Environment ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Accelerator: {accel}")
    print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
print_env()

t0 = time.time()

# -----------------------
# Data loading & prep
# -----------------------
from tensorflow.keras.datasets import mnist
(x_train, y_train_int), (x_test, y_test_int) = mnist.load_data()

x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0
x_train = np.expand_dims(x_train, -1)   # (N,28,28,1)
x_test  = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train_int, 10)
y_test  = to_categorical(y_test_int, 10)

# Simple train/val split
val_fraction = 0.1
n_val = int(len(x_train)*val_fraction)
X_val, y_val = x_train[:n_val], y_train[:n_val]
X_tr,  y_tr  = x_train[n_val:], y_train[n_val:]
y_tr_int = np.argmax(y_tr, axis=1)
y_val_int = np.argmax(y_val, axis=1)

# -----------------------
# Data pipeline (fast)
# -----------------------
BATCH = 256         # good on CPU; try 512 on GPU
USE_AUG = False     # set True if you want small augment (CPU cost)

augment = tf.keras.Sequential([
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.06),
], name="augment")

AUTOTUNE = tf.data.AUTOTUNE
def ds_train(X, Y):
    ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(10000, seed=SEED).batch(BATCH)
    if USE_AUG:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

def ds_eval(X, Y):  # val/test
    return tf.data.Dataset.from_tensor_slices((X, Y)).batch(BATCH).cache().prefetch(AUTOTUNE)

train_ds = ds_train(X_tr, y_tr)
val_ds   = ds_eval(X_val, y_val)
test_ds  = ds_eval(x_test, y_test)

# -----------------------
# Model (compact & accurate)
# -----------------------
L2 = 5e-4
DropC = 0.15
DropD = 0.2

inputs = layers.Input((28,28,1))
x = inputs
# Block 1
x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.MaxPooling2D(2)(x); x = layers.Dropout(DropC)(x)
# Block 2
x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.MaxPooling2D(2)(x); x = layers.Dropout(DropC)(x)
# Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.Dropout(DropD)(x)
outputs = layers.Dense(10, activation="softmax", dtype="float32")(x)  # float32 head for metrics with mixed precision
model = tf.keras.Model(inputs, outputs, name="FastCNN")
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------
# Callbacks
# -----------------------
ckpt_path = "best_mnist_cnn.keras"
cbs = [
    cb.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=3e-5, verbose=1),
    cb.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
]

# -----------------------
# Train (short & sweet)
# -----------------------
EPOCHS = 12
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=1)

# -----------------------
# Evaluate + predictions
# -----------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n[TEST] loss={test_loss:.4f}  acc={test_acc:.4f}")

y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
y_true = y_test_int

# Classification report & F1s
report = classification_report(y_true, y_pred, digits=4)
f1_micro = f1_score(y_true, y_pred, average="micro")
f1_macro = f1_score(y_true, y_pred, average="macro")
print("\nClassification report (head):\n", "\n".join(report.splitlines()[:12]))
print(f"F1 micro={f1_micro:.4f} | macro={f1_macro:.4f}")

# Save TXT artifacts
with open("classification_report.txt", "w") as f:
    f.write(report)
with open("epochs_results.txt", "w") as f:
    f.write(json.dumps({
        "best_val_acc": float(np.max(hist.history["val_accuracy"])),
        "test_acc": float(test_acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "epochs": len(hist.history["loss"])
    }, indent=2))

# -----------------------
# Plots
# -----------------------
# Accuracy / Loss
plt.figure(figsize=(6,4), dpi=110)
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy per Epoch"); plt.legend(); plt.tight_layout()
plt.savefig("model_accuracy.png"); plt.close()

plt.figure(figsize=(6,4), dpi=110)
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch"); plt.legend(); plt.tight_layout()
plt.savefig("model_loss.png"); plt.close()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,7), dpi=120)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.close()

# Per-class F1
per_class_f1 = [f1_score(y_true, y_pred, average=None)[i] for i in range(10)]
plt.figure(figsize=(8,4), dpi=120)
plt.bar(range(10), per_class_f1)
plt.ylim(0.9,1.0); plt.title("Per-class F1"); plt.tight_layout()
plt.savefig("f1_scores.png"); plt.close()

# Example prediction (one sample)
idx = np.random.randint(len(x_test))
img = x_test[idx]
pred_label = int(np.argmax(model.predict(img[None,...], verbose=0)))
true_label = int(y_test_int[idx])
plt.figure(figsize=(3,3), dpi=140); plt.imshow(img.squeeze(), cmap="gray"); plt.axis("off")
plt.title(f"Pred: {pred_label} | True: {true_label}")
plt.tight_layout(); plt.savefig("sample_prediction.png"); plt.close()

# Top-5 errors (highest wrong-confidence)
wrong = np.where(y_pred != y_true)[0]
if len(wrong) >= 5:
    conf = y_prob[wrong, :]
    pred_conf = conf.max(axis=1)
    top5_idx = wrong[np.argsort(-pred_conf)[:5]]
    fig, axes = plt.subplots(1, 5, figsize=(12,3), dpi=120)
    for ax, i in zip(axes, top5_idx):
        ax.imshow(x_test[i].squeeze(), cmap="gray"); ax.axis("off")
        ax.set_title(f"p={y_pred[i]} (conf {y_prob[i].max():.2f})\nt={y_true[i]}")
    fig.suptitle("Top-5 Errors (most confident wrong preds)")
    plt.tight_layout(); plt.savefig("top5_errors.png"); plt.close()

# Save model & params
model.save("best_mnist_cnn_final.keras")
with open("best_cnn_params.json", "w") as f:
    json.dump({
        "batch_size": BATCH,
        "use_augmentation": USE_AUG,
        "l2": L2, "drop_conv": DropC, "drop_dense": DropD,
        "test_accuracy": float(test_acc),
        "f1_micro": float(f1_micro), "f1_macro": float(f1_macro),
    }, f, indent=2)

# -----------------------
# Final runtime
# -----------------------
t1 = time.time()
print(f"=== Total runtime: {t1 - t0:.1f} s ===")
print("Saved: model_* plots, confusion_matrix.png, f1_scores.png, top5_errors.png, "
      "classification_report.txt, epochs_results.txt, best_cnn_params.json, best_mnist_cnn_final.keras, best_mnist_cnn.keras")
