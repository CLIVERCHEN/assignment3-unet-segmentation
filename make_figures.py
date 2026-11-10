"""Build paper figures directly from the saved experiment artifacts."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from src.data import SHAPES
from src.metrics import binary_metrics

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT/f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(6.8, 3.35))
    ax.set(xlim=(-0.15, 7.55), ylim=(-0.3, 3.8))
    ax.axis("off")
    def box(x, y, text, color, width=1.9):
        ax.add_patch(FancyBboxPatch((x, y), width, 0.64,
                     boxstyle="round,pad=0.04", facecolor=color,
                     edgecolor="#536477", linewidth=0.8))
        ax.text(x+width/2, y+0.32, text, ha="center", va="center", fontsize=12)
    def arrow(a, b, label=None, dx=0, dy=0):
        ax.annotate("", b, a, arrowprops={"arrowstyle": "->", "color": "#435466", "lw": 1})
        if label:
            ax.text((a[0]+b[0])/2+dx, (a[1]+b[1])/2+dy,
                    label, ha="center", va="center", fontsize=11,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    box(0.15, 2.35, "Encoder 1\n16 x 128 x 128", "#e0edf7", 2.45)
    box(0.15, 1.1, "Encoder 2\n32 x 64 x 64", "#e0edf7", 2.45)
    box(2.55, -0.15, "Bottleneck\n64 x 32 x 32", "#e7e1f3", 2.45)
    box(4.9, 1.1, "Decoder 2\n32 x 64 x 64", "#e0f0eb", 2.45)
    box(4.9, 2.35, "Decoder 1\n16 x 128 x 128", "#e0f0eb", 2.45)
    arrow((1.375, 3.5), (1.375, 3.02))
    ax.text(1.375, 3.62, "Input: 1 x 128 x 128", ha="center", fontsize=12)
    arrow((1.375, 2.31), (1.375, 1.78), "2 x 2 max pool")
    arrow((1.375, 1.06), (2.55, 0.49), "pool", dx=-0.25)
    arrow((5.0, 0.49), (6.125, 1.06), "up", dx=0.16)
    arrow((6.125, 1.78), (6.125, 2.31), "up")
    arrow((2.64, 2.67), (4.86, 2.67), "skip / concat")
    arrow((2.64, 1.42), (4.86, 1.42), "skip / concat")
    arrow((6.125, 3.02), (6.125, 3.5))
    ax.text(6.125, 3.64, "1 x 1 Conv + sigmoid", ha="center", fontsize=12)
    ax.text(3.85, -0.42, "up = 2 x 2 transposed convolution, stride 2; dimensions = C x H x W", ha="center", fontsize=10)
    save(fig, "architecture")


def learning_curve():
    with open("results/history.csv") as f: rows = list(csv.DictReader(f))
    with open("results/epochs.csv") as f: epochs = list(csv.DictReader(f))
    x = np.array([int(r["update"]) for r in rows])
    y = np.array([float(r["bce"]) for r in rows])
    fig, ax = plt.subplots(figsize=(3.48, 1.85), layout="constrained")
    ax.plot(x, y, color="#95b9d4", linewidth=0.65, label="Mini-batch BCE")
    ax.plot([int(r["update"]) for r in epochs], [float(r["train_bce"]) for r in epochs],
            "o-", color="#173d63", markersize=2.5, linewidth=1, label="Epoch mean")
    ax.set(xlabel="Training update", ylabel="Binary cross-entropy", yscale="log", xlim=(0, x[-1]))
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    save(fig, "learning_curve")


def examples():
    d = np.load("results/selected_examples.npz")
    fig, axes = plt.subplots(2, 9, figsize=(7.15, 2.25))
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.78, hspace=0.65, wspace=0.06)
    for group, shape in enumerate(SHAPES):
        for example in range(2):
            i = group*2+example
            for column, key in enumerate(["images", "masks", "predictions"]):
                ax = axes[example, group*3+column]
                ax.imshow(d[key][i], cmap="gray", vmin=0, vmax=255 if key == "images" else 1,
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)
                ax.set_title(["Input", "Ground truth", "Prediction"][column], fontsize=6.2, pad=3)
                if column == 1:
                    iou = binary_metrics(d["predictions"][i], d["masks"][i])["iou"]
                    ax.text(0.5, 1.35, f'{shape.capitalize()} #{d["indices"][i]} | IoU {iou:.4f}',
                            transform=ax.transAxes, ha="center", fontsize=7)
    save(fig, "test_examples")


if __name__ == "__main__":
    architecture()
    learning_curve()
    examples()
    metrics = json.loads(Path("results/metrics.json").read_text())
    config = json.loads(Path("results/config.json").read_text())
    macros = {"TestAcc": f'{100*metrics["test"]["accuracy"]:.4f}',
              "TestIoU": f'{metrics["test"]["iou"]:.6f}',
              "TrainAcc": f'{100*metrics["train"]["accuracy"]:.4f}',
              "TrainIoU": f'{metrics["train"]["iou"]:.6f}',
              "TrainBCE": f'{metrics["train"]["bce"]:.6f}',
              "TestBCE": f'{metrics["test"]["bce"]:.6f}',
              "MeanIoU": f'{metrics["mean_image_iou"]:.6f}',
              "TrainSeconds": f'{metrics["training_seconds"]:.1f}',
              "Parameters": f'{config["parameters"]:,}',
              "BackgroundAcc": f'{100*metrics["background_baseline"]["accuracy"]:.4f}'}
    for key in ["tp", "tn", "fp", "fn"]:
        macros[key.upper()] = f'{metrics["test"][key]:,}'
    for shape in SHAPES:
        macros[shape.capitalize()+"Acc"] = f'{100*metrics["by_shape"][shape]["accuracy"]:.4f}'
        macros[shape.capitalize()+"IoU"] = f'{metrics["by_shape"][shape]["iou"]:.6f}'
    Path("report/numbers.tex").write_text("% Generated from results/metrics.json; do not edit.\n"+
        "\n".join(f'\\newcommand{{\\{k}}}{{{v}}}' for k,v in macros.items())+"\n")
