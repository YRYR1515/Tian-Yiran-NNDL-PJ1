"""
Direction 5: Error Analysis & Visualization
  - Confusion matrix (MLP & CNN)
  - Misclassified samples (MLP & CNN)
  - MLP Layer-1 weight visualization (neurons as 28x28 images)
  - MLP Layer-2 weight matrix heatmap
  - CNN Conv1 & Conv2 kernel visualization

Run from project root:  python codes/weight_visualization.py
Switch ANALYZE_MLP / ANALYZE_CNN to control which sections are shown.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────────────
ANALYZE_MLP = True
ANALYZE_CNN = True

MLP_MODEL_PATH = r'./codes/best_models/mlp/best_model.pickle'
CNN_MODEL_PATH = r'./codes/best_models/cnn/best_model.pickle'
TEST_IMAGES_PATH = r'./codes/dataset/MNIST/t10k-images-idx3-ubyte.gz'
TEST_LABELS_PATH = r'./codes/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

N_MISCLASSIFIED = 20   # how many misclassified samples to display
N_MLP_NEURONS   = 20   # how many Layer-1 neurons to visualize

# ── load test data ───────────────────────────────────────────────────────────
with gzip.open(TEST_IMAGES_PATH, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(TEST_LABELS_PATH, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs_flat = test_imgs / test_imgs.max()   # [10000, 784], float

# ── helpers ──────────────────────────────────────────────────────────────────
def build_confusion_matrix(logits, labels, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    preds = np.argmax(logits, axis=1)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, title, ax):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=6,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)


def plot_misclassified(imgs_flat, true_labs, pred_labs, n, title):
    wrong = np.where(true_labs != pred_labs)[0]
    n = min(n, len(wrong))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=12)
    axes = axes.reshape(-1)
    for i in range(n):
        idx = wrong[i]
        axes[i].imshow(imgs_flat[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'T:{true_labs[idx]} P:{pred_labs[idx]}', fontsize=7)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()


# ════════════════════════════════════════════════════════════════════════════
# MLP ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if ANALYZE_MLP:
    print("=" * 50)
    print("Loading MLP model ...")
    mlp = nn.models.Model_MLP()
    mlp.load_model(MLP_MODEL_PATH)

    mlp_logits = mlp(test_imgs_flat)
    mlp_preds  = np.argmax(mlp_logits, axis=1)
    mlp_acc    = (mlp_preds == test_labs).mean()
    print(f"MLP Test Accuracy: {mlp_acc:.4f}")

    # -- per-class accuracy
    print("MLP Per-class Accuracy:")
    for c in range(10):
        mask = test_labs == c
        acc_c = (mlp_preds[mask] == test_labs[mask]).mean()
        print(f"  Class {c}: {acc_c:.4f}  (n={mask.sum()})")

    # -- confusion matrix
    cm_mlp = build_confusion_matrix(mlp_logits, test_labs)

    # -- Layer-1 weight visualization (each column = one neuron, reshape to 28x28)
    W1 = mlp.layers[0].params['W']   # [784, 600]
    cols_w = 5
    rows_w = (N_MLP_NEURONS + cols_w - 1) // cols_w
    fig_w1, axes_w1 = plt.subplots(rows_w, cols_w, figsize=(cols_w * 2, rows_w * 2))
    fig_w1.suptitle(f'MLP Layer-1 Weights: first {N_MLP_NEURONS} neurons (28×28)', fontsize=11)
    axes_w1 = axes_w1.reshape(-1)
    for i in range(N_MLP_NEURONS):
        neuron = W1[:, i].reshape(28, 28)
        vmax = np.abs(neuron).max()
        axes_w1[i].imshow(neuron, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes_w1[i].set_title(f'N{i}', fontsize=7)
        axes_w1[i].axis('off')
    for i in range(N_MLP_NEURONS, len(axes_w1)):
        axes_w1[i].axis('off')
    plt.tight_layout()

    # -- Layer-2 weight heatmap  [600, 10]
    W2 = mlp.layers[2].params['W']   # [600, 10]
    fig_w2, ax_w2 = plt.subplots(figsize=(7, 4))
    im2 = ax_w2.imshow(W2.T, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im2, ax=ax_w2, fraction=0.03)
    ax_w2.set_title('MLP Layer-2 Weight Matrix  (10 classes × 600 hidden units)', fontsize=10)
    ax_w2.set_xlabel('Hidden Unit Index')
    ax_w2.set_ylabel('Output Class')
    ax_w2.set_yticks(range(10))
    plt.tight_layout()

    # -- misclassified samples
    plot_misclassified(test_imgs_flat, test_labs, mlp_preds,
                       N_MISCLASSIFIED, f'MLP Misclassified Samples  (test acc={mlp_acc:.4f})')


# ════════════════════════════════════════════════════════════════════════════
# CNN ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if ANALYZE_CNN:
    print("=" * 50)
    print("Loading CNN model ...")
    cnn = nn.models.Model_CNN()
    cnn.load_model(CNN_MODEL_PATH)

    test_imgs_cnn = test_imgs_flat.reshape(-1, 1, 28, 28)
    cnn_logits = cnn(test_imgs_cnn)
    cnn_preds  = np.argmax(cnn_logits, axis=1)
    cnn_acc    = (cnn_preds == test_labs).mean()
    print(f"CNN Test Accuracy: {cnn_acc:.4f}")

    # -- per-class accuracy
    print("CNN Per-class Accuracy:")
    for c in range(10):
        mask = test_labs == c
        acc_c = (cnn_preds[mask] == test_labs[mask]).mean()
        print(f"  Class {c}: {acc_c:.4f}  (n={mask.sum()})")

    # -- confusion matrix (side-by-side with MLP if both enabled)
    cm_cnn = build_confusion_matrix(cnn_logits, test_labs)

    # -- Conv1 kernels  [4, 1, 3, 3]
    W_c1 = cnn.layers[0].params['W']
    n_c1 = W_c1.shape[0]
    fig_c1, axes_c1 = plt.subplots(1, n_c1, figsize=(n_c1 * 2, 2))
    fig_c1.suptitle('CNN Conv1 Kernels  (4 filters × 1 channel × 3×3)', fontsize=11)
    for i in range(n_c1):
        k = W_c1[i, 0]
        vmax = max(np.abs(k).max(), 1e-8)
        axes_c1[i].imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes_c1[i].set_title(f'F{i}', fontsize=9)
        axes_c1[i].axis('off')
    plt.tight_layout()

    # -- Conv2 kernels  [8, 4, 3, 3]
    W_c2 = cnn.layers[2].params['W']
    n_out, n_in = W_c2.shape[:2]
    fig_c2, axes_c2 = plt.subplots(n_out, n_in, figsize=(n_in * 1.8, n_out * 1.8))
    fig_c2.suptitle('CNN Conv2 Kernels  (8 out-filters × 4 in-channels × 3×3)', fontsize=11)
    for oc in range(n_out):
        for ic in range(n_in):
            k = W_c2[oc, ic]
            vmax = max(np.abs(k).max(), 1e-8)
            axes_c2[oc, ic].imshow(k, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes_c2[oc, ic].axis('off')
    for ic in range(n_in):
        axes_c2[0, ic].set_title(f'In{ic}', fontsize=8)
    for oc in range(n_out):
        axes_c2[oc, 0].set_ylabel(f'Out{oc}', fontsize=8, rotation=0, labelpad=28)
    plt.tight_layout()

    # -- misclassified samples
    plot_misclassified(test_imgs_flat, test_labs, cnn_preds,
                       N_MISCLASSIFIED, f'CNN Misclassified Samples  (test acc={cnn_acc:.4f})')


# ── confusion matrix plot (combined) ────────────────────────────────────────
if ANALYZE_MLP and ANALYZE_CNN:
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 5))
    plot_confusion_matrix(cm_mlp, f'MLP Confusion Matrix  (acc={mlp_acc:.4f})', axes_cm[0])
    plot_confusion_matrix(cm_cnn, f'CNN Confusion Matrix  (acc={cnn_acc:.4f})', axes_cm[1])
    plt.tight_layout()
elif ANALYZE_MLP:
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    plot_confusion_matrix(cm_mlp, f'MLP Confusion Matrix  (acc={mlp_acc:.4f})', ax_cm)
    plt.tight_layout()
elif ANALYZE_CNN:
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    plot_confusion_matrix(cm_cnn, f'CNN Confusion Matrix  (acc={cnn_acc:.4f})', ax_cm)
    plt.tight_layout()

plt.show()
