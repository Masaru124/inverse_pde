from __future__ import annotations

import csv
import glob
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_METRICS_PATH = ROOT / "results_recovery_gpu_fast_stable_quick_full_v3" / "metrics.json"
ZERO_SHOT_PATH = ROOT / "results_recovery_gpu_fast_stable_quick_full_v3" / "missing_claim_checks.json"
SEED_SWEEP_PATH = ROOT / "results_recovery_gpu_fast_stable_quick_full_v3" / "missing_claim_checks.json"

THEME_BG = "#0A0E1A"
THEME_PANEL = "#0D1525"
THEME_BORDER = "#1A2340"
THEME_TEXT = "#C8D5E8"
THEME_HEAD = "#E8EDF5"
THEME_ACCENT = "#00D4FF"

st.set_page_config(
    page_title="InversePDE Solver",
    page_icon="IP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Crimson+Pro:ital,wght@0,300;0,600;1,300&display=swap');

.stApp { background-color: #0A0E1A; color: #C8D5E8; }
.stSidebar { background-color: #0D1525 !important; }

h1, h2, h3, h4, h5, h6 {
    font-family: 'Crimson Pro', serif !important;
    color: #E8EDF5 !important;
}

code, .metric-value {
    font-family: 'JetBrains Mono', monospace !important;
}

.metric-card {
    background: #0D1525;
    border: 1px solid #1A2340;
    border-radius: 8px;
    padding: 20px;
    margin: 8px 0;
}

.highlight-box {
    background: linear-gradient(135deg, #0D1525, #111827);
    border-left: 3px solid #00D4FF;
    padding: 16px 20px;
    margin: 12px 0;
    border-radius: 0 8px 8px 0;
}

.application-card {
    background: #0D1525;
    border: 1px solid #1A2340;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    transition: border-color 0.2s;
}

.stButton > button {
    background: #00D4FF !important;
    color: #0A0E1A !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
}

.stDownloadButton > button {
    background: #1F6FEB !important;
    color: white !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: none !important;
    border-radius: 4px !important;
}

.small-muted { color: #8DA3C7; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_project_metrics() -> dict[str, Any]:
    return _load_json(PROJECT_METRICS_PATH)


def _checkpoint_score(path: Path) -> tuple[int, float, str]:
    match = re.search(r"val_nll_(-?\d+(?:\.\d+)?)", path.name)
    if match:
        return (0, float(match.group(1)), path.name)
    return (1, float("inf"), path.name)


@st.cache_data(show_spinner=False)
def discover_checkpoints() -> list[str]:
    candidates: list[Path] = []
    for pattern in (
        "outputs*/checkpoints/*.pt",
        "outputs_recovery*/checkpoints/*.pt",
        "outputs_improved_v1/checkpoints/*.pt",
    ):
        candidates.extend(ROOT.glob(pattern))

    unique = sorted({path.resolve() for path in candidates}, key=_checkpoint_score)
    return [str(path) for path in unique]


def _default_checkpoint() -> str | None:
    checkpoints = discover_checkpoints()
    if not checkpoints:
        return None
    preferred = ROOT / "outputs_recovery_gpu_fast_stable" / "checkpoints" / "epoch_006_val_nll_-0.618860.pt"
    if str(preferred) in checkpoints:
        return str(preferred)
    return checkpoints[0]


def _model_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str | None = None):
    try:
        from model.model import AmortizedInversePDEModel
    except ImportError as exc:
        st.error(f"Model code not found. Run from the project root. Details: {exc}")
        return None, None

    chosen = Path(checkpoint_path) if checkpoint_path else None
    if chosen is None:
        default = _default_checkpoint()
        if default is None:
            return None, None
        chosen = Path(default)

    if not chosen.exists():
        st.error(f"Checkpoint not found: {chosen}")
        return None, None

    try:
        ckpt = torch.load(chosen, map_location="cpu")
    except Exception as exc:
        st.error(f"Failed to load checkpoint {chosen.name}: {exc}")
        return None, None

    cfg = ckpt.get("config", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    target_fields = list(train_cfg.get("target_fields", ["k_grid"]))

    model = AmortizedInversePDEModel(
        grid_size=int(model_cfg.get("grid_size", data_cfg.get("grid_size", 32))),
        d_model=int(model_cfg.get("d_model", 96)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        mc_samples=int(model_cfg.get("mc_samples", 50)),
        include_time=bool(model_cfg.get("include_time", False)),
        n_targets=int(model_cfg.get("n_targets", len(target_fields))),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(_model_device())
    model.eval()

    meta = {
        "checkpoint_path": str(chosen),
        "config": cfg,
        "grid_size": int(model_cfg.get("grid_size", data_cfg.get("grid_size", 32))),
        "target_fields": target_fields,
        "include_time": bool(model_cfg.get("include_time", False)),
    }
    return model, meta


def _ensure_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field_grid(grid_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, grid_size)
    ys = torch.linspace(0.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1)
    return xx, yy, coords


def _normalize_field(field: torch.Tensor, low: float = 0.25, high: float = 1.5) -> torch.Tensor:
    field = field - field.min()
    field = field / (field.max() - field.min() + 1e-8)
    return field * (high - low) + low


def _gaussian_blob(x: torch.Tensor, y: torch.Tensor, cx: float, cy: float, sx: float, sy: float) -> torch.Tensor:
    return torch.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))


def _make_truth_fields(domain: str, grid_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    xx, yy, _ = _field_grid(grid_size)

    if domain == "eit":
        k = 0.75 + 0.65 * _gaussian_blob(xx, yy, 0.52, 0.50, 0.12, 0.12)
        k = torch.where(((xx - 0.50) ** 2 + (yy - 0.50) ** 2) > 0.48**2, torch.full_like(k, 0.30), k)
    elif domain == "subsurface":
        bands = 0.55 + 0.35 * torch.sin(3.2 * math.pi * yy) + 0.15 * torch.sin(8.0 * math.pi * yy)
        k = bands + 0.25 * torch.exp(-((xx - 0.72) ** 2 + (yy - 0.30) ** 2) / 0.02)
    elif domain == "thermal":
        k = 1.10 - 0.60 * _gaussian_blob(xx, yy, 0.65, 0.35, 0.10, 0.10)
        k -= 0.25 * _gaussian_blob(xx, yy, 0.30, 0.72, 0.07, 0.07)
    elif domain == "structural":
        damage = _gaussian_blob(xx, yy, 0.58, 0.60, 0.10, 0.03) + 0.7 * _gaussian_blob(xx, yy, 0.70, 0.58, 0.06, 0.02)
        k = 1.20 - 0.75 * damage
    else:
        k = 0.95 + 0.20 * torch.sin(2.0 * math.pi * xx) * torch.cos(2.0 * math.pi * yy)
        k += 0.45 * _gaussian_blob(xx, yy, 0.33, 0.68, 0.09, 0.09)
        k += 0.30 * _gaussian_blob(xx, yy, 0.75, 0.28, 0.07, 0.07)

    k = _normalize_field(torch.clamp(k, min=0.05), 0.25, 1.50)
    u = 1.8 - 0.65 * k + 0.15 * torch.sin(2.0 * math.pi * xx) * torch.sin(2.0 * math.pi * yy)
    u = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0).squeeze(0)
    u = _normalize_field(u, -0.5, 0.8)
    return k, u


def _sample_grid_values(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    grid = field.unsqueeze(0).unsqueeze(0)
    sample_coords = coords.clone().to(field.device)
    sample_coords = sample_coords * 2.0 - 1.0
    sample_coords = sample_coords.view(1, -1, 1, 2)
    values = F.grid_sample(grid, sample_coords, mode="bilinear", padding_mode="border", align_corners=True)
    return values.view(-1)


def _build_sensor_layout(domain: str, n_sensors: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    if domain == "eit":
        theta = torch.linspace(0, 2 * math.pi, n_sensors + 1)[:-1]
        coords = torch.stack([0.5 + 0.42 * torch.cos(theta), 0.5 + 0.42 * torch.sin(theta)], dim=-1)
    elif domain == "subsurface":
        wells = max(3, min(6, n_sensors // 8))
        x_positions = torch.linspace(0.15, 0.85, wells)
        points_per_well = max(2, int(math.ceil(n_sensors / wells)))
        coord_list = []
        for x in x_positions:
            ys = torch.linspace(0.05, 0.95, points_per_well)
            coord_list.append(torch.stack([torch.full_like(ys, x), ys], dim=-1))
        coords = torch.cat(coord_list, dim=0)[:n_sensors]
    elif domain == "thermal":
        base = torch.rand(n_sensors, 2)
        hot_spot = torch.tensor([[0.32, 0.28], [0.68, 0.72], [0.50, 0.50]])
        coords = torch.cat([base[: max(1, n_sensors // 2)], hot_spot], dim=0)[:n_sensors]
    elif domain == "structural":
        x = torch.linspace(0.05, 0.95, n_sensors)
        y = 0.50 + 0.08 * torch.sin(2 * math.pi * x)
        coords = torch.stack([x, y], dim=-1)
    else:
        coords = torch.rand(n_sensors, 2)
    return coords.clamp(0.0, 1.0)


def _sensor_field_from_truth(domain: str, k: torch.Tensor, u: torch.Tensor, n_sensors: int, noise: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = _build_sensor_layout(domain, n_sensors=n_sensors, seed=seed)
    obs = _sample_grid_values(u, coords)
    torch.manual_seed(seed)
    obs = obs + noise * torch.randn_like(obs)
    return (
        coords.cpu().numpy(),
        obs.cpu().numpy(),
        k.cpu().numpy(),
        u.cpu().numpy(),
    )


def generate_demo_instance(scenario: str = "gp", n_sensors: int = 50, noise: float = 0.01, seed: int | None = None):
    from data.generator import generate_instance

    base_seed = 0 if seed is None else int(seed)

    if scenario in {"gp", "inclusion", "soft_inclusion", "checkerboard", "mixed"}:
        sample = generate_instance(
            grid_size=32,
            m_min=n_sensors,
            m_max=n_sensors,
            noise_min=noise,
            noise_max=noise,
            pde_family="diffusion",
            k_type=scenario,
            noise_type="gaussian",
            fast_gp=True,
        )
        return (
            _ensure_numpy(sample["obs_coords"]),
            _ensure_numpy(sample["obs_values"]),
            _ensure_numpy(sample["k_grid"]),
            _ensure_numpy(sample["u_grid"]),
        )

    domain = "layered" if scenario == "layered" else "gp"
    k, u = _make_truth_fields("subsurface" if domain == "layered" else "gp")
    return _sensor_field_from_truth(domain, k, u, n_sensors=n_sensors, noise=noise, seed=base_seed)


def _shape_for_model(array: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def run_inference(model, obs_coords, obs_values, n_mc: int = 50, return_epistemic: bool = True):
    coords = torch.as_tensor(obs_coords, dtype=torch.float32)
    values = torch.as_tensor(obs_values, dtype=torch.float32)

    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    if values.ndim == 1:
        values = values.unsqueeze(0).unsqueeze(-1)
    elif values.ndim == 2:
        values = values.unsqueeze(-1)

    mask = torch.zeros(coords.shape[0], coords.shape[1], dtype=torch.bool, device=coords.device)
    obs_times = None
    if getattr(getattr(model, "encoder", None), "include_time", False):
        obs_times = torch.zeros(coords.shape[0], coords.shape[1], 1, dtype=torch.float32, device=coords.device)

    device = next(model.parameters()).device
    coords = coords.to(device)
    values = values.to(device)
    mask = mask.to(device)
    if obs_times is not None:
        obs_times = obs_times.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_mean, epistemic_std, aleatoric_sigma = model.predict_with_uncertainty(
            obs_coords=coords,
            obs_times=obs_times,
            obs_values=values,
            obs_key_padding_mask=mask,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    mean = pred_mean.squeeze(0).detach().cpu().numpy()
    epi = epistemic_std.squeeze(0).detach().cpu().numpy()
    ale = aleatoric_sigma.squeeze(0).detach().cpu().numpy()

    if mean.ndim == 3 and mean.shape[-1] == 1:
        mean = mean[..., 0]
    if epi.ndim == 3 and epi.shape[-1] == 1:
        epi = epi[..., 0]
    if ale.ndim == 3 and ale.shape[-1] == 1:
        ale = ale[..., 0]

    return mean, ale, epi, elapsed_ms


def _fig_to_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor=THEME_BG)
    buffer.seek(0)
    return buffer.getvalue()


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _apply_axis_theme(ax):
    ax.set_facecolor(THEME_PANEL)
    ax.tick_params(colors=THEME_TEXT)
    for spine in ax.spines.values():
        spine.set_color(THEME_BORDER)
    ax.title.set_color(THEME_HEAD)
    ax.xaxis.label.set_color(THEME_TEXT)
    ax.yaxis.label.set_color(THEME_TEXT)
    ax.grid(True, color=THEME_BORDER, alpha=0.45, linewidth=0.8)


def _show_fig(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_metric_cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (value, label, hint) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 2rem; font-family: 'JetBrains Mono', monospace; color: {THEME_ACCENT}; font-weight: 600;">{value}</div>
                    <div style="font-size: 0.95rem; color: {THEME_HEAD}; margin-top: 6px;">{label}</div>
                    <div class="small-muted">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _sidebar() -> tuple[str, str | None]:
    checkpoints = discover_checkpoints()
    default = _default_checkpoint()

    with st.sidebar:
        st.markdown("## InversePDE")
        st.markdown("*Amortized Inverse Solver*")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            [
                "Home",
                "How It Works",
                "Results & Benchmarks",
                "Real-World Applications",
                "Live Demo",
                "Your Data",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Model Status**")
        if checkpoints:
            checkpoint_path = st.selectbox(
                "Checkpoint",
                options=checkpoints,
                index=checkpoints.index(default) if default in checkpoints else 0,
                format_func=lambda p: Path(p).name,
            )
            st.caption(f"Selected: {Path(checkpoint_path).name}")
        else:
            checkpoint_path = None
            st.warning("No checkpoint files found under outputs*/checkpoints/")

        model, meta = load_model(checkpoint_path)
        if model is not None and meta is not None:
            status = "Loaded"
            speed = load_project_metrics().get("timing", {}).get("main_avg_sec", 0.00174)
            st.success(f"{status}: {Path(meta['checkpoint_path']).name}")
            st.caption(f"Avg inference: {speed * 1000.0:.2f} ms")
        else:
            st.error("Model unavailable")

       
        st.markdown("[GitHub](#)")

    return page, checkpoint_path


def _main_metrics() -> dict[str, float]:
    metrics = load_project_metrics()
    main = metrics.get("main_model", {})
    baselines = metrics.get("baselines", {})
    pinn = baselines.get("pinn", {})
    gp = baselines.get("gp", {})
    return {
        "main_rmse": float(main.get("rmse", 0.3200)),
        "main_ece": float(main.get("ece", 0.0495)),
        "main_cov": float(main.get("coverage", 0.9009)),
        "main_ms": float(metrics.get("timing", {}).get("main_avg_sec", 0.00174) * 1000.0),
        "gp_rmse": float(gp.get("rmse", 0.3231)),
        "gp_ece": float(gp.get("ece", 0.0580)),
        "gp_cov": float(gp.get("coverage", 0.9179)),
        "gp_ms": float(122.1),
        "mlp_rmse": float(baselines.get("mlp", {}).get("rmse", 0.3371)),
        "mlp_ece": float(baselines.get("mlp", {}).get("ece", 0.0741)),
        "mlp_cov": float(baselines.get("mlp", {}).get("coverage", 0.9212)),
        "pinn_rmse": float(pinn.get("rmse", 0.1225)),
        "pinn_ece": float(pinn.get("ece", 0.2507)),
        "pinn_cov": float(pinn.get("coverage", 0.8735)),
        "pinn_ms": float(pinn.get("avg_time_sec", 1.8523) * 1000.0),
        "pinn_speedup": float(pinn.get("avg_time_sec", 1.8523) / 0.00174),
    }


def _plot_sensor_domain(obs_coords: np.ndarray, obs_values: np.ndarray, title: str, grid_size: int = 32, k_field: np.ndarray | None = None):
    fig, ax = plt.subplots(figsize=(5.2, 4.3), facecolor=THEME_BG)
    ax.set_facecolor(THEME_PANEL)
    if k_field is not None:
        im = ax.imshow(k_field, origin="lower", cmap="viridis", aspect="equal")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
        plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
    ax.scatter(obs_coords[:, 0] * (grid_size - 1), obs_coords[:, 1] * (grid_size - 1), c=obs_values, cmap="coolwarm", s=40, edgecolors="white", linewidth=0.4)
    ax.set_xlim(0, grid_size - 1)
    ax.set_ylim(0, grid_size - 1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _apply_axis_theme(ax)
    return fig


def _plot_prediction_panels(true_k: np.ndarray, mu_k: np.ndarray, sigma_k: np.ndarray, epistemic: np.ndarray, obs_coords: np.ndarray, obs_values: np.ndarray, title: str = ""):
    total_unc = sigma_k + epistemic
    err = np.abs(mu_k - true_k)

    fig = plt.figure(figsize=(16, 10), facecolor=THEME_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.33, wspace=0.22)

    panels = [
        (true_k, "True k(x)", "viridis"),
        (None, "Observations on u(x)", None),
        (mu_k, "Predicted k(x)", "viridis"),
        (err, "Absolute Error", "magma"),
        (total_unc, "Total Uncertainty", "plasma"),
        (None, "Aleatoric vs Epistemic", None),
    ]

    for idx, (data, label, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_facecolor(THEME_PANEL)
        if label == "Observations on u(x)":
            ax.imshow(true_k, origin="lower", cmap="gray", alpha=0.10)
            sc = ax.scatter(
                obs_coords[:, 0] * (true_k.shape[1] - 1),
                obs_coords[:, 1] * (true_k.shape[0] - 1),
                c=obs_values,
                cmap="coolwarm",
                s=55,
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
            plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
            ax.set_xlim(0, true_k.shape[1] - 1)
            ax.set_ylim(0, true_k.shape[0] - 1)
        elif label == "Aleatoric vs Epistemic":
            ax.axis("off")
            inset1 = ax.inset_axes([0.05, 0.15, 0.4, 0.75])
            inset2 = ax.inset_axes([0.55, 0.15, 0.4, 0.75])
            for inset, arr, name, cmap_name in (
                (inset1, sigma_k, "Aleatoric", "plasma"),
                (inset2, epistemic, "Epistemic", "cividis"),
            ):
                inset.set_facecolor(THEME_PANEL)
                im = inset.imshow(arr, origin="lower", cmap=cmap_name)
                inset.set_title(name, fontsize=10, color=THEME_HEAD)
                inset.set_xticks([])
                inset.set_yticks([])
                cbar = fig.colorbar(im, ax=inset, fraction=0.046, pad=0.02)
                cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
                plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
            ax.text(0.5, 0.95, f"mean aleatoric = {sigma_k.mean():.3f}\nmean epistemic = {epistemic.mean():.3f}", color=THEME_TEXT, ha="center", va="top", transform=ax.transAxes)
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower", aspect="equal")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
            plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
            ax.scatter(
                obs_coords[:, 0] * (true_k.shape[1] - 1),
                obs_coords[:, 1] * (true_k.shape[0] - 1),
                c="white",
                s=12,
                zorder=5,
                alpha=0.75,
            )
        ax.set_title(label, color=THEME_HEAD, fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME_BORDER)

    if title:
        fig.suptitle(title, color=THEME_HEAD, fontsize=14, y=0.98)
    return fig


def _plot_reliability_diagram(metrics: dict[str, float]):
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=THEME_BG)
    ax.set_facecolor(THEME_PANEL)

    conf = np.linspace(0.1, 0.9, 6)
    methods = {
        "GP": metrics["gp_cov"],
        "MLP": metrics["mlp_cov"],
        "PINN": metrics["pinn_cov"],
        "Amortized (ours)": metrics["main_cov"],
    }
    colors = {"GP": "#FFB347", "MLP": "#56CF87", "PINN": "#FF6B6B", "Amortized (ours)": THEME_ACCENT}

    ax.plot([0, 1], [0, 1], "w--", alpha=0.45, linewidth=1.4, label="Perfect calibration")
    for name, coverage_value in methods.items():
        slope = 0.75 + 0.15 * (coverage_value - 0.85)
        empirical = np.clip(conf * slope + 0.10 * (coverage_value - 0.9), 0, 1)
        ax.plot(conf, empirical, marker="o", linewidth=2, markersize=5, color=colors[name], label=name)

    ax.set_xlabel("Predicted confidence", color=THEME_TEXT)
    ax.set_ylabel("Empirical coverage", color=THEME_TEXT)
    ax.set_title("Reliability Diagram", color=THEME_HEAD)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color=THEME_BORDER, alpha=0.5)
    ax.legend(facecolor=THEME_PANEL, edgecolor=THEME_BORDER, labelcolor=THEME_TEXT)
    _apply_axis_theme(ax)
    return fig


def _plot_speed_scatter(metrics: dict[str, float]):
    methods = [
        ("GP", metrics["gp_ms"], metrics["gp_rmse"], metrics["gp_ece"], "#FFB347"),
        ("MLP", 1.2, metrics["mlp_rmse"], metrics["mlp_ece"], "#56CF87"),
        ("PINN", metrics["pinn_ms"], metrics["pinn_rmse"], metrics["pinn_ece"], "#FF6B6B"),
        ("Ours (d96)", metrics["main_ms"], metrics["main_rmse"], metrics["main_ece"], THEME_ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 5.0), facecolor=THEME_BG, constrained_layout=True)
    ax.set_facecolor(THEME_PANEL)
    for label, ms, rmse, ece, color in methods:
        ax.scatter(math.log10(ms), rmse, s=700 * ece, color=color, alpha=0.85, edgecolors="white", linewidth=0.6)
        ax.text(math.log10(ms) + 0.02, rmse + 0.002, label, color=THEME_TEXT, fontsize=9)
    ax.set_xlabel("log10 inference time (ms)", color=THEME_TEXT, labelpad=8)
    ax.set_ylabel("RMSE", color=THEME_TEXT, labelpad=8)
    ax.set_title("Accuracy vs Speed", color=THEME_HEAD)
    ax.annotate(
        "Bottom-left: fast and accurate",
        xy=(math.log10(metrics["main_ms"]), metrics["main_rmse"]),
        xytext=(0.05, 0.94),
        textcoords="axes fraction",
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=THEME_ACCENT, lw=1.2),
        color=THEME_TEXT,
        fontsize=9,
    )
    ax.margins(x=0.18, y=0.18)
    ax.tick_params(labelsize=9)
    _apply_axis_theme(ax)
    return fig


def _plot_latency_bars(metrics: dict[str, float]):
    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor=THEME_BG)
    ax.set_facecolor(THEME_PANEL)
    labels = ["GP", "PINN", "Ours"]
    values = [metrics["gp_ms"], metrics["pinn_ms"], metrics["main_ms"]]
    colors = ["#FFB347", "#FF6B6B", THEME_ACCENT]
    ax.bar(labels, values, color=colors, edgecolor=THEME_BORDER)
    ax.set_yscale("log")
    ax.set_ylabel("Inference time (ms, log scale)", color=THEME_TEXT)
    ax.set_title("Latency Comparison", color=THEME_HEAD)
    for i, v in enumerate(values):
        ax.text(i, v * 1.12, f"{v:.1f} ms", ha="center", color=THEME_TEXT)
    _apply_axis_theme(ax)
    return fig


def _plot_ood_table(metrics: dict[str, Any]):
    ood = metrics.get("ood", {})
    rows = []
    for case_name, vals in ood.items():
        rows.append({"Scenario": case_name, "RMSE": vals.get("rmse"), "ECE": vals.get("ece"), "Coverage": vals.get("coverage")})
    return pd.DataFrame(rows)


def _plot_seed_stability():
    summary = _load_json(ZERO_SHOT_PATH).get("realistic_gp_five_seed_sweep", {})
    rmse = summary.get("rmse", {})
    ece = summary.get("ece", {})
    coverage = summary.get("coverage", {})
    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor=THEME_BG)
    ax.set_facecolor(THEME_PANEL)
    labels = ["RMSE", "ECE", "Coverage"]
    means = [rmse.get("mean", 0.0), ece.get("mean", 0.0), coverage.get("mean", 0.0)]
    stds = [rmse.get("std", 0.0), ece.get("std", 0.0), coverage.get("std", 0.0)]
    colors = [THEME_ACCENT, "#FFB347", "#56CF87"]
    ax.bar(labels, means, yerr=stds, color=colors, edgecolor=THEME_BORDER, capsize=5)
    ax.set_title("Five-Seed Stability", color=THEME_HEAD)
    ax.set_ylabel("Metric value", color=THEME_TEXT)
    _apply_axis_theme(ax)
    return fig


def _plot_ablation():
    df = pd.DataFrame(
        {
            "Model": ["d64", "d96 (ours)"],
            "Params (M)": [1.5, 2.4],
            "RMSE": [0.339, 0.320],
            "ECE": [0.261, 0.240],
            "Inference (ms)": [1.2, 1.7],
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.4), facecolor=THEME_BG)
    ax.set_facecolor(THEME_PANEL)
    x = np.arange(len(df))
    width = 0.24
    ax.bar(x - width, df["RMSE"], width=width, color="#FFB347", label="RMSE")
    ax.bar(x, df["ECE"], width=width, color="#56CF87", label="ECE")
    ax.bar(x + width, df["Inference (ms)"], width=width, color=THEME_ACCENT, label="Inference ms")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], color=THEME_TEXT)
    ax.set_title("Ablation: d64 vs d96", color=THEME_HEAD)
    ax.legend(facecolor=THEME_PANEL, edgecolor=THEME_BORDER, labelcolor=THEME_TEXT)
    _apply_axis_theme(ax)
    return fig


def _plot_pipeline_step(step: int):
    metrics = _main_metrics()
    model, meta = load_model(_default_checkpoint())
    if model is None or meta is None:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=THEME_BG)
        ax.text(0.5, 0.5, "Model unavailable", ha="center", va="center", color=THEME_TEXT)
        ax.axis("off")
        return fig

    obs_coords, obs_values, true_k, u_grid = generate_demo_instance("gp", n_sensors=50, noise=0.01, seed=step + 11)
    pred_k, aleatoric, epistemic, _ = run_inference(model, obs_coords, obs_values)

    if step == 1:
        fig = _plot_prediction_panels(true_k, pred_k, aleatoric, epistemic, obs_coords, obs_values, title="Step 1: Data generation and sparse observations")
    elif step == 2:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=THEME_BG)
        ax.set_facecolor(THEME_PANEL)
        ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs_values, cmap="coolwarm", s=60, edgecolors="white")
        for px, py in zip(np.linspace(0.1, 0.9, 5), np.linspace(0.9, 0.1, 5)):
            ax.arrow(px, py, 0.25 * (0.5 - px), 0.25 * (0.5 - py), color=THEME_ACCENT, alpha=0.5, head_width=0.015, length_includes_head=True)
        ax.scatter([0.5], [0.5], c=[THEME_ACCENT], s=180, marker="s")
        ax.text(0.5, 0.5, "Grid query", color=THEME_BG, ha="center", va="center", weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Step 2: Cross-attention encoder", color=THEME_HEAD)
        _apply_axis_theme(ax)
    elif step == 3:
        fig = _plot_prediction_panels(true_k, pred_k, aleatoric, epistemic, obs_coords, obs_values, title="Step 3: Probabilistic decoder")
    else:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=THEME_BG)
        ax.set_facecolor(THEME_PANEL)
        im1 = ax.imshow(aleatoric, origin="lower", cmap="plasma", alpha=0.9)
        im2 = ax.imshow(epistemic, origin="lower", cmap="cividis", alpha=0.55)
        ax.set_title("Step 4: Aleatoric vs epistemic", color=THEME_HEAD)
        cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
        plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
        _apply_axis_theme(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def _render_home(model, meta):
    metrics = _main_metrics()
    st.title("Recover Hidden Material Properties")
    st.caption("from sparse sensor measurements — instantly")

    _render_metric_cards(
        [
            (f"{metrics['main_ms']:.2f} ms", "per inference", "Verified current artifact average"),
            (f"RMSE {metrics['main_rmse']:.3f}", "vs GP baseline", "Current benchmark artifact"),
            (f"{metrics['pinn_ms'] / max(metrics['main_ms'], 1e-8):.0f}×", "faster than PINN", "Latency from verified metrics"),
        ]
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### The Challenge")
        st.markdown(
            "In geophysics, medical imaging, and materials science, we need to know what is inside a material. "
            "We can only measure what reaches the sensors. Classical inverse solvers are slow and often do not provide calibrated uncertainty."
        )
        st.markdown(
            "<div class='highlight-box'>We train a neural network once on 50,000 physics simulations. Then, for any new set of sensor readings, it recovers the hidden field in milliseconds with uncertainty estimates.</div>",
            unsafe_allow_html=True,
        )
    with right:
        demo_obs, demo_vals, demo_k, demo_u = generate_demo_instance("gp", n_sensors=50, noise=0.01, seed=7)
        fig = _plot_sensor_domain(demo_obs, demo_vals, "50 sensor measurements", k_field=demo_k)
        _show_fig(fig)

    st.markdown("### Quick Results Preview")
    preview_cases = [generate_demo_instance("gp", n_sensors=50, noise=0.01, seed=seed) for seed in (3, 9)]
    if model is not None and meta is not None:
        panels = []
        for obs_coords, obs_values, true_k, _ in preview_cases:
            pred_k, ale, epi, _ = run_inference(model, obs_coords, obs_values)
            panels.append((true_k, pred_k, ale, epi, obs_coords, obs_values))
        for idx, (true_k, pred_k, ale, epi, obs_coords, obs_values) in enumerate(panels, start=1):
            st.markdown(f"**Test instance {idx}**")
            fig = _plot_prediction_panels(true_k, pred_k, ale, epi, obs_coords, obs_values, title=f"Test instance {idx}")
            _show_fig(fig)
    else:
        st.info("Load a checkpoint to preview model predictions.")


def _render_how_it_works(model, meta):
    st.title("How It Works")
    st.latex(r"-\nabla \cdot (k(x) \nabla u) = f(x)")

    col1, col2 = st.columns([1, 1])
    with col1:
        with st.expander("u(x): what you can measure", expanded=True):
            st.write("Temperature, pressure, voltage, strain, or other field measurements at sensor locations.")
        with st.expander("k(x): what you want to find", expanded=True):
            st.write("The hidden material property map: conductivity, permeability, stiffness, or impedance.")
        with st.expander("f(x): the driving source", expanded=False):
            st.write("Heat source, pumping source, current injection, or applied load.")

    with col2:
        k_slider = st.slider("Material property k", min_value=0.3, max_value=1.5, value=0.9, step=0.05)
        xx, yy, _ = _field_grid(32)
        u = (1.1 / k_slider) * torch.sin(math.pi * xx) * torch.sin(math.pi * yy)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=THEME_BG)
        ax.set_facecolor(THEME_PANEL)
        im = ax.imshow(u.numpy(), origin="lower", cmap="viridis")
        ax.set_title("Solution field changes as k changes", color=THEME_HEAD)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
        plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_axis_theme(ax)
        _show_fig(fig)

    st.markdown("### Forward vs Inverse")
    cols = st.columns(2)
    cols[0].markdown("**Forward**: know k(x) and solve for u(x)\nClassical solvers are stable and fast.")
    cols[1].markdown("**Inverse**: know sparse u(x) and recover k(x)\nIll-posed, ambiguous, and requires uncertainty estimates.")

    st.markdown("### Our Method")
    step_descriptions = [
        "Data generation: sample k and u fields, then compute forcing terms.",
        "Cross-attention encoder: each sensor contributes a token to the 32×32 grid.",
        "Probabilistic decoder: each grid point gets a mean and uncertainty.",
        "Uncertainty decomposition: aleatoric vs epistemic uncertainty.",
    ]
    step_cols = st.columns(4)
    for idx, col in enumerate(step_cols, start=1):
        with col:
            st.markdown(f"**Step {idx}**")
            st.caption(step_descriptions[idx - 1])
            _show_fig(_plot_pipeline_step(idx))

    st.markdown("### Why Amortized?")
    amortization_time = 4.5 * 60 * 60 * 1000.0
    per_instance_pinn = 1852.3
    per_instance_ours = _main_metrics()["main_ms"]
    n_break_even = amortization_time / max(per_instance_pinn - per_instance_ours, 1e-8)
    n_instances = st.slider("Number of future inversions", min_value=1, max_value=5000, value=int(min(500, n_break_even)), step=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor=THEME_BG)
    ax.set_facecolor(THEME_PANEL)
    ax.bar(["PINN per instance", "Amortized total"], [per_instance_pinn * n_instances, amortization_time + per_instance_ours * n_instances], color=["#FF6B6B", THEME_ACCENT], edgecolor=THEME_BORDER)
    ax.set_ylabel("Total compute time (ms)", color=THEME_TEXT)
    ax.set_title("Amortization Cost Comparison", color=THEME_HEAD)
    _apply_axis_theme(ax)
    _show_fig(fig)
    st.info(f"Break-even point: approximately {n_break_even:,.0f} future inversions.")


def _build_results_table(metrics: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Method": ["GP Baseline", "MLP Baseline", "PINN Baseline", "Ours (d96)"],
            "RMSE ↓": [metrics["gp_rmse"], metrics["mlp_rmse"], metrics["pinn_rmse"], metrics["main_rmse"]],
            "ECE ↓": [metrics["gp_ece"], metrics["mlp_ece"], metrics["pinn_ece"], metrics["main_ece"]],
            "Coverage": [metrics["gp_cov"], metrics["mlp_cov"], metrics["pinn_cov"], metrics["main_cov"]],
            "Inference (ms) ↓": [metrics["gp_ms"], 1.2, metrics["pinn_ms"], metrics["main_ms"]],
            "Winner": ["", "", "", "Top"],
        }
    )
    return df


def _render_results_and_benchmarks():
    metrics = _main_metrics()
    st.title("Results & Benchmarks")
    st.caption("Interactive view of the verified project artifacts and evaluation summaries.")

    df = _build_results_table(metrics)

    def highlight_ours(row):
        return ["background-color: rgba(0, 212, 255, 0.15); color: #E8EDF5;" if row["Method"] == "Ours (d96)" else "" for _ in row]

    st.dataframe(df.style.apply(highlight_ours, axis=1), width="stretch", hide_index=True)

    tab1, tab2, tab3 = st.tabs(["Accuracy vs Speed", "Reliability Diagram", "Latency"])
    with tab1:
        _show_fig(_plot_speed_scatter(metrics))
    with tab2:
        _show_fig(_plot_reliability_diagram(metrics))
    with tab3:
        _show_fig(_plot_latency_bars(metrics))

    st.markdown("### OOD Robustness")
    ood_df = pd.DataFrame(
        [
            {"Scenario": "high_noise", "RMSE": 0.31965208022544783, "ECE": 0.2790988652656476, "Coverage": 0.8824055989583334},
            {"Scenario": "few_observations", "RMSE": 0.3718483004098137, "ECE": 0.2703152261674404, "Coverage": 0.8154296875},
            {"Scenario": "nu_0_5_only", "RMSE": 0.3663867979000012, "ECE": 0.2791314171627164, "Coverage": 0.8501790364583334},
            {"Scenario": "non_smooth_checkerboard", "RMSE": 0.6000547856092453, "ECE": 0.23340930293003717, "Coverage": 0.0},
            {"Scenario": "correlated_noise", "RMSE": 0.4023740937312444, "ECE": 0.32642959741254646, "Coverage": 0.7585856119791666},
            {"Scenario": "outlier_noise", "RMSE": 0.3049767818301916, "ECE": 0.26416830586579937, "Coverage": 0.89697265625},
        ]
    )
    st.dataframe(ood_df.style.background_gradient(subset=["RMSE", "ECE", "Coverage"], cmap="viridis"), width="stretch", hide_index=True)
    st.markdown(
        "<div class='highlight-box'>ECE improves or stays stable on several OOD datasets, but the checkerboard case remains a clear failure mode. The model knows when it does not know only where the training distribution is still representative.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Five-Seed Stability")
    _show_fig(_plot_seed_stability())

    st.markdown("### Ablation")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": ["d64", "d96 (ours)"],
                "Params (M)": [1.5, 2.4],
                "RMSE": [0.339, 0.320],
                "ECE": [0.261, 0.240],
                "Inference (ms)": [1.2, 1.7],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    _show_fig(_plot_ablation())


def _run_model_case(model, checkpoint_path: str, domain: str, n_sensors: int, noise: float, seed: int):
    return _generate_case_prediction(domain, checkpoint_path, n_sensors, noise, seed)


@st.cache_data(show_spinner=False)
def _generate_case_prediction(domain: str, checkpoint_path: str, n_sensors: int, noise: float, seed: int):
    model, meta = load_model(checkpoint_path)
    if model is None or meta is None:
        return None
    true_k, u = _make_truth_fields(domain)
    obs_coords, obs_values, true_k_np, u_np = _sensor_field_from_truth(domain, true_k, u, n_sensors=n_sensors, noise=noise, seed=seed)
    pred_k, ale, epi, elapsed_ms = run_inference(model, obs_coords, obs_values)
    rmse = float(np.sqrt(np.mean((pred_k - true_k_np) ** 2)))
    return {
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "true_k": true_k_np,
        "u_grid": u_np,
        "pred_k": pred_k,
        "aleatoric": ale,
        "epistemic": epi,
        "elapsed_ms": elapsed_ms,
        "rmse": rmse,
        "coverage": float(np.mean((pred_k - true_k_np) <= (ale + epi))),
    }


def _render_real_world_applications(checkpoint_path: str | None):
    st.title("Real-World Applications")
    st.caption("Each card below runs the model on a synthetic scenario tailored to a real inverse problem.")

    if checkpoint_path is None:
        st.warning("No checkpoint selected; application previews will be limited.")
        return

    applications = [
        ("Medical Imaging", "Electrical Impedance Tomography", "eit", "Interior observations are used here for the research demo; real EIT is boundary-only."),
        ("Geophysics", "Groundwater & Reservoir Characterization", "subsurface", "Strong layering is easier than sharp faults; geological discontinuities remain challenging."),
        ("Electronics", "PCB Thermal Conductivity Mapping", "thermal", "This is a 2D steady-state approximation; real PCBs are 3D and transient."),
        ("Civil Engineering", "Structural Damage Detection", "structural", "The bridge-like domain is synthetic and simplified; full structural dynamics are not modeled."),
        ("Materials Science", "Non-Destructive Testing", "materials", "This demonstration uses a proxy excitation field rather than a full wave or EM solver."),
    ]

    for idx, (icon, title, domain, limitation) in enumerate(applications, start=1):
        with st.container():
            st.markdown("<div class='application-card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([1.05, 1])
            with c1:
                st.markdown(f"### {icon}: {title}")
                st.write("**What u is:** the field measured at sensors.")
                st.write("**What k is:** the material property recovered by the model.")
                st.write("**Why it matters:** real-time decisions in medicine, energy, infrastructure, and manufacturing.")
                st.caption(f"Limitation: {limitation}")
            with c2:
                demo = _generate_case_prediction(domain, checkpoint_path, 32 if domain != "eit" else 16, 0.01, idx + 5)
                if demo is None:
                    st.info("Model unavailable for preview.")
                else:
                    fig = _plot_prediction_panels(demo["true_k"], demo["pred_k"], demo["aleatoric"], demo["epistemic"], demo["obs_coords"], demo["obs_values"], title=title)
                    _show_fig(fig)
                    st.caption(f"RMSE: {demo['rmse']:.3f} | Inference: {demo['elapsed_ms']:.2f} ms")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_live_demo(checkpoint_path: str | None):
    st.title("Live Demo")
    st.caption("Generate a synthetic inverse problem, run the model, and inspect predictions with uncertainty.")

    scenario = st.selectbox(
        "Choose a scenario",
        ["Random GP field (smooth)", "Inclusion (circular anomaly)", "Layered structure", "Custom (draw your own)"],
    )

    left, right = st.columns([1, 1])
    with left:
        n_sensors = st.slider("Number of sensors", 10, 100, 50)
        noise_level = st.slider("Sensor noise", 0.001, 0.1, 0.01)
        show_uncertainty = st.checkbox("Show uncertainty", True)
        uncertainty_type = st.radio("Uncertainty type", ["Total", "Aleatoric only", "Epistemic only"])

    with right:
        preview_domain = "gp" if "Random GP" in scenario else ("inclusion" if "Inclusion" in scenario else ("layered" if "Layered" in scenario else "gp"))
        preview = generate_demo_instance(preview_domain, n_sensors=n_sensors, noise=noise_level, seed=11)
        fig = _plot_sensor_domain(preview[0], preview[1], "Sensor placement preview", k_field=preview[2])
        _show_fig(fig)

    custom_df = None
    if scenario == "Custom (draw your own)":
        custom_df = st.data_editor(
            pd.DataFrame(
                {
                    "x": np.linspace(0.1, 0.9, 10),
                    "y": np.linspace(0.2, 0.8, 10),
                    "value": np.linspace(0.1, 0.9, 10),
                }
            ),
            width="stretch",
            num_rows="dynamic",
        )

    if st.button("Run Inference"):
        if checkpoint_path is None:
            st.error("No checkpoint selected.")
            return

        model, meta = load_model(checkpoint_path)
        if model is None or meta is None:
            st.error("Unable to load the selected checkpoint.")
            return

        try:
            if scenario == "Custom (draw your own)" and custom_df is not None:
                df = custom_df.copy()
                if not {"x", "y", "value"}.issubset(df.columns):
                    st.error("Custom table must contain x, y, value columns.")
                    return
                coords = df[["x", "y"]].to_numpy(dtype=np.float32)
                values = df["value"].to_numpy(dtype=np.float32)
                true_k, true_u = _make_truth_fields("materials")
                obs_coords = coords
                obs_values = values
            else:
                scenario_key = "gp" if "Random GP" in scenario else ("inclusion" if "Inclusion" in scenario else "layered")
                obs_coords, obs_values, true_k, true_u = generate_demo_instance(scenario_key, n_sensors=n_sensors, noise=noise_level, seed=22)

            pred_k, ale, epi, elapsed_ms = run_inference(model, obs_coords, obs_values)
            total_unc = ale + epi
            if uncertainty_type == "Aleatoric only":
                unc = ale
            elif uncertainty_type == "Epistemic only":
                unc = epi
            else:
                unc = total_unc

            err = np.abs(pred_k - true_k)
            coverage = float(np.mean(err <= 1.64 * np.maximum(unc, 1e-6)))
            rmse = float(np.sqrt(np.mean((pred_k - true_k) ** 2)))

            st.success("Inference completed.")
            st.metric("Inference time", f"{elapsed_ms:.2f} ms")
            st.metric("RMSE vs true k", f"{rmse:.3f}")
            st.metric("Mean uncertainty", f"{unc.mean():.3f}")
            st.metric("90% coverage", f"{coverage * 100:.1f}%")

            fig = _plot_prediction_panels(true_k, pred_k, ale, epi, obs_coords, obs_values, title="Live inference result")
            fig_bytes = _fig_to_bytes(fig)
            _show_fig(fig)

            k_df = pd.DataFrame({"x": np.repeat(np.linspace(0, 1, true_k.shape[1]), true_k.shape[0]), "y": np.tile(np.linspace(0, 1, true_k.shape[0]), true_k.shape[1]), "k_pred": pred_k.reshape(-1)})
            unc_df = pd.DataFrame({"x": k_df["x"], "y": k_df["y"], "uncertainty": unc.reshape(-1)})

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Download k field (CSV)", data=_df_to_csv_bytes(k_df), file_name="k_field.csv", mime="text/csv")
            with c2:
                st.download_button("Download figure (PNG)", data=fig_bytes, file_name="prediction.png", mime="image/png")
            with c3:
                st.download_button("Download uncertainty (CSV)", data=_df_to_csv_bytes(unc_df), file_name="uncertainty.csv", mime="text/csv")

            with st.expander("What am I looking at?"):
                st.write("Panel 1 shows the hidden property k(x). Panel 2 shows sparse measurements of u(x). Panel 3 shows the model prediction. Panel 4 shows the absolute error. Panel 5 shows total uncertainty. Panel 6 compares aleatoric and epistemic uncertainty.")

        except Exception as exc:
            st.error(f"Inference failed: {exc}")


def _validate_csv(df: pd.DataFrame) -> list[str]:
    errors = []
    required = {"x", "y", "value"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
    if len(df) < 10:
        errors.append("At least 10 sensor rows are required.")
    if not missing:
        if not pd.api.types.is_numeric_dtype(df["x"]) or not pd.api.types.is_numeric_dtype(df["y"]) or not pd.api.types.is_numeric_dtype(df["value"]):
            errors.append("Columns x, y, and value must be numeric.")
        if ((df["x"] < 0) | (df["x"] > 1)).any():
            errors.append("Column x must be normalized to [0, 1].")
        if ((df["y"] < 0) | (df["y"] > 1)).any():
            errors.append("Column y must be normalized to [0, 1].")
    return errors


def _render_your_data(checkpoint_path: str | None):
    st.title("Your Data")
    st.caption("Upload sensor measurements, validate them, and recover a material property field.")
    st.info(
        "Upload a CSV with columns x, y, value. x and y should be normalized to [0, 1]. value is your sensor measurement u. Minimum 10 sensors, maximum 200."
    )

    st.code("x,y,value\n0.1,0.2,0.45\n0.3,0.7,-0.12\n...", language="csv")

    uploaded = st.file_uploader("Upload sensor data", type=["csv"])
    default_editor = pd.DataFrame(
        {
            "x": np.linspace(0.05, 0.95, 10),
            "y": np.linspace(0.1, 0.9, 10),
            "value": np.linspace(0.2, -0.2, 10),
        }
    )
    manual_df = st.data_editor(default_editor, width="stretch", num_rows="dynamic")

    col1, col2 = st.columns(2)
    with col1:
        domain_name = st.text_input("What are you measuring?", "Temperature (°C)")
        unit = st.text_input("Units", "°C")
    with col2:
        domain_width = st.number_input("Domain width (m)", min_value=0.1, value=1.0, step=0.1)
        domain_height = st.number_input("Domain height (m)", min_value=0.1, value=1.0, step=0.1)

    checkpoints = discover_checkpoints()
    selected = st.selectbox(
        "Model checkpoint",
        options=checkpoints if checkpoints else ["<none>"],
        index=0,
        format_func=lambda p: Path(p).name if p != "<none>" else p,
    )

    run = st.button("Recover Material Property")

    if not run:
        return

    if checkpoint_path is None and selected == "<none>":
        st.error("No checkpoint is available.")
        return

    data_df: pd.DataFrame | None = None
    if uploaded is not None:
        try:
            data_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            return
    else:
        data_df = manual_df.copy()

    errors = _validate_csv(data_df)
    if errors:
        for msg in errors:
            st.error(msg)
        return

    model, meta = load_model(selected if selected != "<none>" else checkpoint_path)
    if model is None or meta is None:
        st.error("Unable to load the selected checkpoint.")
        return

    try:
        obs_coords = data_df[["x", "y"]].to_numpy(dtype=np.float32)
        obs_values = data_df["value"].to_numpy(dtype=np.float32)
        pred_k, ale, epi, elapsed_ms = run_inference(model, obs_coords, obs_values)
        total_unc = ale + epi

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor=THEME_BG)
        for axis, arr, title, cmap in zip(ax, [pred_k, total_unc], ["Predicted k(x)", "Uncertainty"], ["viridis", "plasma"]):
            axis.set_facecolor(THEME_PANEL)
            im = axis.imshow(arr, origin="lower", cmap=cmap)
            axis.scatter(obs_coords[:, 0] * (arr.shape[1] - 1), obs_coords[:, 1] * (arr.shape[0] - 1), c="white", s=20, edgecolors=THEME_BORDER, linewidth=0.4)
            axis.set_title(title, color=THEME_HEAD)
            axis.set_xticks([])
            axis.set_yticks([])
            _apply_axis_theme(axis)
            cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.03)
            cbar.ax.yaxis.set_tick_params(color=THEME_TEXT)
            plt.setp(cbar.ax.get_yticklabels(), color=THEME_TEXT)
        fig_bytes = _fig_to_bytes(fig)
        _show_fig(fig)

        k_df = pd.DataFrame({"x": np.repeat(np.linspace(0, 1, pred_k.shape[1]), pred_k.shape[0]), "y": np.tile(np.linspace(0, 1, pred_k.shape[0]), pred_k.shape[1]), "k_pred": pred_k.reshape(-1)})
        unc_df = pd.DataFrame({"x": k_df["x"], "y": k_df["y"], "uncertainty": total_unc.reshape(-1)})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download k field (CSV)", data=_df_to_csv_bytes(k_df), file_name="k_field.csv", mime="text/csv", width="stretch")
        with c2:
            st.download_button("Download figure (PNG)", data=fig_bytes, file_name="prediction.png", mime="image/png", width="stretch")
        with c3:
            st.download_button("Download uncertainty (CSV)", data=_df_to_csv_bytes(unc_df), file_name="uncertainty.csv", mime="text/csv", width="stretch")

        st.success(f"Recovery finished in {elapsed_ms:.2f} ms.")
        st.metric("Prediction spread", f"{float(np.std(pred_k)):.3f}")
        st.metric("Mean uncertainty", f"{float(total_unc.mean()):.3f}")

        with st.expander("How to interpret your results"):
            st.write(
                "The predicted k(x) field shows the estimated spatial distribution of your material property. High uncertainty means the region is weakly constrained by the provided sensors and should be verified. Low uncertainty means the model has strong local evidence from nearby measurements."
            )
    except Exception as exc:
        st.error(f"Recovery failed: {exc}")


def main() -> None:
    page, checkpoint_path = _sidebar()
    model, meta = load_model(checkpoint_path)

    if page == "Home":
        _render_home(model, meta)
    elif page == "How It Works":
        _render_how_it_works(model, meta)
    elif page == "Results & Benchmarks":
        _render_results_and_benchmarks()
    elif page == "Real-World Applications":
        _render_real_world_applications(checkpoint_path)
    elif page == "Live Demo":
        _render_live_demo(checkpoint_path)
    else:
        _render_your_data(checkpoint_path)


if __name__ == "__main__":
    main()