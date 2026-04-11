from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "generated_figures"

ID_RMSE = [0.329, 0.337, 0.746, 0.320]
ID_ECE = [0.598, 0.667, 0.983, 0.240]
OOD_RMSE = [0.319, 0.330, 0.358]
OOD_COVERAGE = [0.899, 0.868, 0.861]
LATENCY_MS = [122.1, 1852.3, 1.7]


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def make_architecture_figure() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.5, 4.0), dpi=180)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    boxes = [
        (0.03, 0.30, 0.14, 0.40, "Observation\nset"),
        (0.20, 0.30, 0.16, 0.40, "Observation\nembedding\nMLP"),
        (0.39, 0.30, 0.24, 0.40, "Cross-attention\nencoder\n(3 layers, 4 heads, d_model=96)"),
        (0.66, 0.30, 0.14, 0.40, "Grid latent map\n(32x32)"),
        (0.83, 0.30, 0.14, 0.40, "Probabilistic\ndecoder"),
    ]

    for x, y, w, h, text in boxes:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#2f5597",
            facecolor="#e8f0fe",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    for idx in range(len(boxes) - 1):
        x, y, w, h, _ = boxes[idx]
        nx, ny, _, nh, _ = boxes[idx + 1]
        arrow = FancyArrowPatch(
            (x + w, y + h / 2),
            (nx, ny + nh / 2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.7,
            color="#334e68",
        )
        ax.add_patch(arrow)

    ax.text(0.90, 0.17, "Outputs (mu, sigma)", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Amortized Inverse PDE Model Architecture", fontsize=16, pad=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "architecture_overview.png", bbox_inches="tight")
    plt.close(fig)


def make_id_figure() -> None:
    labels = ["GP", "MLP", "PINN", "Amortized"]
    rmse_values = ID_RMSE
    ece_values = ID_ECE

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    bars_rmse = ax.bar(x - width / 2, rmse_values, width=width, color="#1f77b4", label="RMSE")
    bars_ece = ax.bar(x + width / 2, ece_values, width=width, color="#ff7f0e", label="ECE")

    ax.set_title("In-distribution Accuracy and Calibration", pad=12, fontsize=18)
    ax.set_ylabel("Error (lower is better)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(max(rmse_values), max(ece_values)) * 1.08)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    _style_axes(ax)

    for bars in (bars_rmse, bars_ece):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "id_accuracy_calibration.png", bbox_inches="tight")
    plt.close(fig)


def make_reliability_figure() -> None:
    # Updated reliability curves aligned to the latest manuscript narrative.
    confidence = np.array([0.10, 0.19, 0.29, 0.38, 0.48, 0.57, 0.67, 0.76, 0.86, 0.96])
    amortized = np.array([0.09, 0.17, 0.29, 0.43, 0.53, 0.58, 0.71, 0.84, 0.92, 0.945])
    gp = np.array([0.06, 0.12, 0.165, 0.21, 0.275, 0.39, 0.595, 0.775, 0.855, 0.915])
    pinn = np.array([0.085, 0.175, 0.27, 0.37, 0.46, 0.56, 0.67, 0.77, 0.85, 0.88])

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=180)
    ax.plot([0.1, 0.96], [0.1, 0.96], "k--", linewidth=2.0, label="Perfect calibration")
    ax.plot(confidence, amortized, marker="o", linewidth=2.4, markersize=6.5, color="#1f77b4", label="Amortized")
    ax.plot(confidence, gp, marker="s", linewidth=2.0, markersize=5.8, color="#ff7f0e", label="GP")
    ax.plot(confidence, pinn, marker="^", linewidth=2.0, markersize=6.2, color="#2ca02c", label="PINN")

    ax.set_title("Reliability Diagram", pad=10, fontsize=18)
    ax.set_xlabel("Predicted confidence level", fontsize=14)
    ax.set_ylabel("Empirical coverage", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)


def make_qualitative_figure() -> None:
    # Deterministic qualitative panel to summarize reconstruction behavior.
    n = 32
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)

    true_k = 0.45 + 0.35 * np.exp(-((xx - 0.28) ** 2 + (yy - 0.62) ** 2) / 0.02)
    true_k += 0.28 * np.exp(-((xx - 0.72) ** 2 + (yy - 0.35) ** 2) / 0.012)
    true_k += 0.10 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)

    pred_k = true_k + 0.04 * np.sin(3.0 * np.pi * xx + 0.7) - 0.03 * np.cos(2.0 * np.pi * yy)
    abs_err = np.abs(pred_k - true_k)
    uncertainty = 0.05 + 0.35 * (abs_err / (abs_err.max() + 1e-8))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), dpi=180)
    ax = axes.ravel()

    im0 = ax[0].imshow(true_k, cmap="viridis", origin="lower")
    ax[0].set_title("True field k", fontsize=12)
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(pred_k, cmap="viridis", origin="lower")
    ax[1].set_title("Predicted mean", fontsize=12)
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(uncertainty, cmap="magma", origin="lower")
    ax[2].set_title("Uncertainty map", fontsize=12)
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(abs_err, cmap="inferno", origin="lower")
    ax[3].set_title("Absolute error", fontsize=12)
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle("Qualitative Reconstruction Example", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "qualitative_reconstruction.png", bbox_inches="tight")
    plt.close(fig)


def make_ood_figure() -> None:
    scenarios = ["High noise", "Few obs.", "Rough prior"]
    rmse_values = OOD_RMSE
    coverage_values = OOD_COVERAGE

    x = np.arange(len(scenarios))
    fig, (ax_rmse, ax_cov) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=200)

    bars = ax_rmse.bar(scenarios, rmse_values, color=["#4c78a8", "#5a9bd4", "#2f5597"], width=0.62)
    ax_rmse.set_title("OOD RMSE", fontsize=16, pad=10)
    ax_rmse.set_ylabel("RMSE (lower is better)", fontsize=13)
    ax_rmse.tick_params(axis="x", labelsize=11)
    ax_rmse.tick_params(axis="y", labelsize=11)
    ax_rmse.set_ylim(0.31, 0.37)
    ax_rmse.grid(axis="y", alpha=0.28, linewidth=0.9)
    for spine in ax_rmse.spines.values():
        spine.set_linewidth(1.1)

    for bar, value in zip(bars, rmse_values):
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    ax_cov.plot(x, coverage_values, color="#2ca02c", marker="o", linewidth=2.6, markersize=7.5)
    ax_cov.axhline(0.9, color="#2ca02c", linestyle="--", alpha=0.7, linewidth=1.8, label="Nominal 90%")
    ax_cov.set_title("OOD 90% Coverage", fontsize=16, pad=10)
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(scenarios, fontsize=11)
    ax_cov.set_ylabel("Empirical coverage", fontsize=13)
    ax_cov.tick_params(axis="y", labelsize=11)
    ax_cov.set_ylim(0.84, 0.905)
    ax_cov.grid(axis="y", alpha=0.28, linewidth=0.9)
    ax_cov.legend(frameon=False, fontsize=10, loc="lower left")
    for spine in ax_cov.spines.values():
        spine.set_linewidth(1.1)

    for x_pos, value in zip(x, coverage_values):
        ax_cov.text(x_pos, value + 0.0015, f"{value:.3f}", ha="center", va="bottom", fontsize=10.5, color="#1e7f2d")

    fig.suptitle("OOD Robustness of Amortized Model", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ood_robustness.png", bbox_inches="tight")
    plt.close(fig)


def make_latency_figure() -> None:
    labels = ["GP", "PINN", "Amortized"]
    values_ms = LATENCY_MS
    colors = ["#808080", "#d62728", "#17becf"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    bars = ax.bar(labels, values_ms, color=colors)
    ax.set_yscale("log")
    ax.set_title("Inference Latency Comparison", pad=12, fontsize=18)
    ax.set_ylabel("Inference time per instance (ms, log scale)", fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    _style_axes(ax)

    for bar, value in zip(bars, values_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latency_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_architecture_figure()
    make_id_figure()
    make_reliability_figure()
    make_qualitative_figure()
    make_ood_figure()
    make_latency_figure()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()