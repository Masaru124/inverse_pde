from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from model.model import AmortizedInversePDEModel

ROOT = Path(__file__).resolve().parent


def _load_comparison_csv() -> pd.DataFrame | None:
    csv_path = ROOT / "results_final_summary" / "comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _discover_metrics() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for metrics_path in ROOT.glob("results*/metrics.json"):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            runs[metrics_path.parent.name] = data
        except Exception:
            continue
    return dict(sorted(runs.items()))


def _discover_checkpoints() -> list[str]:
    ckpts = []
    for ckpt in ROOT.glob("outputs*/checkpoints/*.pt"):
        ckpts.append(str(ckpt.relative_to(ROOT)).replace("\\", "/"))
    return sorted(ckpts)


@st.cache_resource(show_spinner=False)
def _load_model_from_checkpoint(ckpt_rel_path: str) -> tuple[AmortizedInversePDEModel, dict]:
    ckpt_path = ROOT / ckpt_rel_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    target_fields = list(train_cfg.get("target_fields", ["k_grid"]))

    model = AmortizedInversePDEModel(
        grid_size=int(data_cfg.get("grid_size", 32)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mc_samples=int(model_cfg.get("mc_samples", 8)),
        include_time=bool(model_cfg.get("include_time", False)),
        n_targets=len(target_fields),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _build_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, grid_size, device=device)
    y = torch.linspace(0.0, 1.0, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _div_k_grad_u(coords: torch.Tensor, u_flat: torch.Tensor, k_flat: torch.Tensor) -> torch.Tensor:
    grad_u = torch.autograd.grad(u_flat.sum(), coords, create_graph=True, retain_graph=True)[0]
    flux = k_flat.unsqueeze(-1) * grad_u
    div_x = torch.autograd.grad(flux[:, 0].sum(), coords, create_graph=True, retain_graph=True)[0][:, 0]
    div_y = torch.autograd.grad(flux[:, 1].sum(), coords, create_graph=True, retain_graph=True)[0][:, 1]
    return div_x + div_y


def _laplacian(coords: torch.Tensor, u_flat: torch.Tensor) -> torch.Tensor:
    grad_u = torch.autograd.grad(u_flat.sum(), coords, create_graph=True, retain_graph=True)[0]
    u_x = grad_u[:, 0]
    u_y = grad_u[:, 1]
    u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True, retain_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True, retain_graph=True)[0][:, 1]
    return u_xx + u_yy


def _generate_equation_case(
    equation_name: str,
    grid_size: int,
    n_obs: int,
    noise_sigma: float,
    seed: int,
    alpha: float,
    beta_x: float,
    beta_y: float,
) -> dict:
    device = torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    coords = _build_grid(grid_size, device=device).to(torch.float64).requires_grad_(True)
    x = coords[:, 0]
    y = coords[:, 1]
    pi = math.pi

    # Shared smooth test fields.
    u_flat = (
        torch.sin(pi * x) * torch.sin(pi * y)
        + 0.15 * torch.sin(2.0 * pi * x) * torch.sin(2.0 * pi * y)
    )
    k_flat = 1.0 + 0.25 * torch.sin(2.0 * pi * x) * torch.cos(2.0 * pi * y)
    k_flat = torch.clamp(k_flat, min=1e-3)

    if equation_name == "Diffusion ( -div(k grad u) = f )":
        f_flat = -_div_k_grad_u(coords=coords, u_flat=u_flat, k_flat=k_flat)
    elif equation_name == "Poisson ( -Delta u = f )":
        k_flat = torch.ones_like(k_flat)
        f_flat = -_laplacian(coords=coords, u_flat=u_flat)
    elif equation_name == "Reaction-Diffusion ( -div(k grad u) + a u = f )":
        f_flat = -_div_k_grad_u(coords=coords, u_flat=u_flat, k_flat=k_flat) + alpha * u_flat
    elif equation_name == "Advection-Diffusion ( -div(k grad u) + b.grad(u) = f )":
        grad_u = torch.autograd.grad(u_flat.sum(), coords, create_graph=True, retain_graph=True)[0]
        advection = beta_x * grad_u[:, 0] + beta_y * grad_u[:, 1]
        f_flat = -_div_k_grad_u(coords=coords, u_flat=u_flat, k_flat=k_flat) + advection
    else:
        raise ValueError(f"Unknown equation: {equation_name}")

    total_pts = grid_size * grid_size
    n_obs = max(5, min(n_obs, total_pts))
    obs_idx = torch.randperm(total_pts, generator=g, device=device)[:n_obs]
    obs_coords = coords.detach()[obs_idx].to(torch.float32)
    clean_obs = u_flat.detach()[obs_idx].to(torch.float32)
    noise = noise_sigma * torch.randn(n_obs, generator=g, device=device, dtype=torch.float32)
    obs_values = clean_obs + noise

    return {
        "equation": equation_name,
        "coords": coords.detach().to(torch.float32).cpu(),
        "obs_coords": obs_coords.cpu(),
        "obs_values": obs_values.cpu(),
        "u_grid": u_flat.detach().reshape(grid_size, grid_size).to(torch.float32).cpu(),
        "k_grid": k_flat.detach().reshape(grid_size, grid_size).to(torch.float32).cpu(),
        "f_grid": f_flat.detach().reshape(grid_size, grid_size).to(torch.float32).cpu(),
    }


def _run_checkpoint_inference(case: dict, ckpt_rel_path: str) -> dict:
    model, _ = _load_model_from_checkpoint(ckpt_rel_path)

    obs_coords = case["obs_coords"].unsqueeze(0)
    obs_values = case["obs_values"].unsqueeze(0).unsqueeze(-1)
    mask = torch.zeros(1, obs_coords.shape[1], dtype=torch.bool)

    with torch.no_grad():
        obs_times = torch.zeros(obs_coords.shape[0], obs_coords.shape[1], 1, device=obs_coords.device)
        mu, sigma = model(
            obs_coords=obs_coords,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_key_padding_mask=mask,
            mc_dropout=False,
        )

    mu = mu[0].cpu()
    sigma = sigma[0].cpu()
    k_true = case["k_grid"]

    rmse = torch.sqrt(torch.mean((mu - k_true) ** 2)).item()
    mae = torch.mean(torch.abs(mu - k_true)).item()
    return {
        "k_pred": mu,
        "sigma_pred": sigma,
        "rmse": rmse,
        "mae": mae,
    }


def _plot_case(case: dict, pred: dict | None = None) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    u = case["u_grid"].numpy()
    k = case["k_grid"].numpy()
    f = case["f_grid"].numpy()
    obs = case["obs_coords"].numpy()

    im0 = axes[0, 0].imshow(u, origin="lower", cmap="viridis")
    axes[0, 0].set_title("u(x,y)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(k, origin="lower", cmap="viridis")
    axes[0, 1].set_title("k_true(x,y)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(f, origin="lower", cmap="coolwarm")
    axes[0, 2].set_title("f(x,y)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(u, origin="lower", cmap="gray")
    axes[1, 0].scatter(
        obs[:, 0] * (u.shape[1] - 1),
        obs[:, 1] * (u.shape[0] - 1),
        s=12,
        c="red",
        alpha=0.8,
    )
    axes[1, 0].set_title("Observation Locations")

    if pred is not None:
        k_pred = pred["k_pred"].numpy()
        err = abs(k_pred - k)
        im4 = axes[1, 1].imshow(k_pred, origin="lower", cmap="viridis")
        axes[1, 1].set_title("k_pred")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.046)

        im5 = axes[1, 2].imshow(err, origin="lower", cmap="magma")
        axes[1, 2].set_title("|k_pred - k_true|")
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _main_baseline_table(run_data: dict) -> pd.DataFrame:
    main = run_data.get("main_model", {})
    baselines = run_data.get("baselines", {})
    rows = [
        {
            "model": "main",
            "rmse": main.get("rmse"),
            "ece": main.get("ece"),
            "coverage": main.get("coverage"),
        }
    ]

    for name, vals in baselines.items():
        rows.append(
            {
                "model": name,
                "rmse": vals.get("rmse"),
                "ece": vals.get("ece"),
                "coverage": vals.get("coverage"),
            }
        )

    return pd.DataFrame(rows)


def _ood_table(run_data: dict) -> pd.DataFrame:
    ood = run_data.get("ood", {})
    rows = []
    for case, vals in ood.items():
        rows.append(
            {
                "case": case,
                "rmse": vals.get("rmse"),
                "ece": vals.get("ece"),
                "coverage": vals.get("coverage"),
            }
        )
    return pd.DataFrame(rows)


def _render_results_explorer() -> None:
    st.subheader("Results Explorer")

    st.set_page_config(page_title="Inverse PDE Results Viewer", layout="wide")
    summary_df = _load_comparison_csv()
    runs = _discover_metrics()

    top_left, top_right = st.columns([1, 1])
    with top_left:
        show_summary = st.checkbox("Show final summary table", value=True)
    with top_right:
        run_names = list(runs.keys())
        selected_run = st.selectbox("Select run", options=run_names if run_names else ["<none>"])

    if show_summary:
        st.subheader("Final Comparison")
        if summary_df is None:
            st.info("No summary CSV found at results_final_summary/comparison.csv")
        else:
            st.dataframe(summary_df, use_container_width=True)

    if not runs:
        st.warning("No metrics.json files found under results*/")
        return

    if selected_run == "<none>":
        st.warning("Select a run to continue.")
        return

    run_data = runs[selected_run]

    st.subheader(f"Run: {selected_run}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Main RMSE", f"{run_data.get('main_model', {}).get('rmse', float('nan')):.4f}")
    col2.metric("Main ECE", f"{run_data.get('main_model', {}).get('ece', float('nan')):.3f}")
    col3.metric("Main Coverage", f"{run_data.get('main_model', {}).get('coverage', float('nan')):.3f}")
    col4.metric("Main Time (s)", f"{run_data.get('timing', {}).get('main_avg_sec', float('nan')):.4f}")

    st.markdown("### Main vs Baselines")
    mb_df = _main_baseline_table(run_data)
    st.dataframe(mb_df, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**RMSE by Model**")
        st.bar_chart(mb_df.set_index("model")[["rmse"]])
    with right:
        st.markdown("**ECE and Coverage by Model**")
        st.line_chart(mb_df.set_index("model")[["ece", "coverage"]])

    st.markdown("### OOD Metrics")
    ood_df = _ood_table(run_data)
    if ood_df.empty:
        st.info("No OOD metrics available in this run.")
    else:
        st.dataframe(ood_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**OOD RMSE**")
            st.bar_chart(ood_df.set_index("case")[["rmse"]])
        with c2:
            st.markdown("**OOD Coverage**")
            st.bar_chart(ood_df.set_index("case")[["coverage"]])

    st.markdown("### Calibration and Timing Diagnostics")
    cal = run_data.get("calibration", {})
    pinn_timing = run_data.get("pinn_timing", {})

    diag_rows = [
        ["temperature_scaling_enabled", cal.get("temperature_scaling_enabled")],
        ["temperature", cal.get("temperature")],
        ["objective", cal.get("objective")],
        ["val_objective_at_temperature", cal.get("val_objective_at_temperature")],
        ["val_coverage_at_temperature", cal.get("val_coverage_at_temperature")],
        ["feasible_candidate_count", cal.get("feasible_candidate_count")],
        ["fallback_reason", cal.get("fallback_reason")],
        ["main_avg_sec", run_data.get("timing", {}).get("main_avg_sec")],
        ["pinn_avg_sec", pinn_timing.get("pinn_avg_sec")],
        ["pinn_avg_steps", pinn_timing.get("pinn_avg_steps")],
        ["pinn_median_steps", pinn_timing.get("pinn_median_steps")],
        ["pinn_converged_fraction", pinn_timing.get("pinn_converged_fraction")],
    ]

    st.table(pd.DataFrame(diag_rows, columns=["metric", "value"]))

    st.markdown("### Raw JSON")
    st.json(run_data)


def _render_equation_playground() -> None:
    st.subheader("Equation Playground")
    st.caption("Generate a synthetic PDE case, choose an equation type, and test a trained checkpoint interactively.")

    ckpt_options = _discover_checkpoints()
    eq_options = [
        "Diffusion ( -div(k grad u) = f )",
        "Poisson ( -Delta u = f )",
        "Reaction-Diffusion ( -div(k grad u) + a u = f )",
        "Advection-Diffusion ( -div(k grad u) + b.grad(u) = f )",
    ]

    c1, c2, c3, c4 = st.columns(4)
    equation = c1.selectbox("Equation", eq_options)
    grid_size = c2.selectbox("Grid size", [16, 24, 32, 40], index=2)
    n_obs = c3.slider("Observations", min_value=10, max_value=300, value=80, step=5)
    noise_sigma = c4.slider("Noise sigma", min_value=0.0, max_value=0.1, value=0.02, step=0.001)

    seed_col, ckpt_col = st.columns([1, 2])
    seed = seed_col.number_input("Seed", min_value=0, max_value=100000, value=42, step=1)
    ckpt_choice = ckpt_col.selectbox(
        "Checkpoint for inference",
        options=["<no checkpoint>"] + ckpt_options,
        index=0,
    )

    alpha = 1.0
    beta_x = 0.5
    beta_y = 0.25
    if "Reaction-Diffusion" in equation:
        alpha = st.slider("Reaction coefficient a", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    if "Advection-Diffusion" in equation:
        b1, b2 = st.columns(2)
        beta_x = b1.slider("Advection beta_x", min_value=-2.0, max_value=2.0, value=0.5, step=0.1)
        beta_y = b2.slider("Advection beta_y", min_value=-2.0, max_value=2.0, value=0.25, step=0.1)

    run_btn = st.button("Generate Case And Run")

    if run_btn:
        case = _generate_equation_case(
            equation_name=equation,
            grid_size=grid_size,
            n_obs=n_obs,
            noise_sigma=noise_sigma,
            seed=int(seed),
            alpha=float(alpha),
            beta_x=float(beta_x),
            beta_y=float(beta_y),
        )

        pred = None
        if ckpt_choice != "<no checkpoint>":
            try:
                pred = _run_checkpoint_inference(case=case, ckpt_rel_path=ckpt_choice)
            except Exception as exc:
                st.error(f"Checkpoint inference failed: {exc}")

        st.session_state["play_case"] = case
        st.session_state["play_pred"] = pred
        st.session_state["play_ckpt"] = ckpt_choice

    case = st.session_state.get("play_case")
    pred = st.session_state.get("play_pred")
    used_ckpt = st.session_state.get("play_ckpt", "<no checkpoint>")

    if case is None:
        st.info("Configure parameters and click 'Generate Case And Run'.")
        return

    st.markdown(f"**Selected equation:** {case['equation']}")
    st.markdown(f"**Checkpoint:** {used_ckpt}")
    if pred is not None:
        m1, m2 = st.columns(2)
        m1.metric("RMSE vs k_true", f"{pred['rmse']:.4f}")
        m2.metric("MAE vs k_true", f"{pred['mae']:.4f}")
    else:
        st.caption("No checkpoint selected. Showing only generated fields and observations.")

    _plot_case(case=case, pred=pred)


def main() -> None:
    st.set_page_config(page_title="Inverse PDE Results Viewer", layout="wide")
    st.title("Inverse PDE Experiment Viewer")
    st.caption("Explore saved experiments and interactively test checkpoint behavior on selectable equation families.")

    tab1, tab2 = st.tabs(["Results", "Equation Playground"])
    with tab1:
        _render_results_explorer()
    with tab2:
        _render_equation_playground()


if __name__ == "__main__":
    main()
