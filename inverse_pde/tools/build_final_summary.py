import csv
import json
from pathlib import Path

FILES = [
    ("d64_no_temp", "results_full_run_final_eval_notemp_v2/metrics.json"),
    ("d64_temp_safe", "results_full_run_final_eval_safecal_v5/metrics.json"),
    ("d96_no_temp", "results_full_run_d96_eval_notemp_v2/metrics.json"),
    ("d96_temp_safe", "results_full_run_d96_eval_safecal_v2/metrics.json"),
    ("d96_temp_1_2", "results_full_run_d96_eval_temp_1_2/metrics.json"),
]

out = Path("results_final_summary")
out.mkdir(exist_ok=True)

rows = []
for name, path in FILES:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    m = d["main_model"]
    gp = d["baselines"]["gp"]
    cal = d.get("calibration", {})
    pt = d.get("pinn_timing", {})
    rows.append(
        {
            "config": name,
            "rmse": m["rmse"],
            "ece": m["ece"],
            "coverage": m["coverage"],
            "gp_rmse": gp["rmse"],
            "rmse_gap_vs_gp": m["rmse"] - gp["rmse"],
            "temperature": cal.get("temperature"),
            "calibration_fallback": cal.get("fallback_reason"),
            "main_avg_sec": d["timing"]["main_avg_sec"],
            "pinn_avg_sec": pt.get("pinn_avg_sec"),
            "pinn_avg_steps": pt.get("pinn_avg_steps"),
            "pinn_converged_fraction": pt.get("pinn_converged_fraction"),
        }
    )

with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Final Comparison (Corrected Baselines)",
    "",
    "| Config | RMSE | ECE | Coverage | GP RMSE | Gap vs GP | Temp | Main sec | PINN sec | PINN steps | PINN conv. |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {config} | {rmse:.4f} | {ece:.3f} | {coverage:.3f} | {gp_rmse:.4f} | {rmse_gap_vs_gp:+.4f} | {temperature} | {main_avg_sec:.4f} | {pinn_avg_sec:.4f} | {pinn_avg_steps:.1f} | {pinn_converged_fraction:.2f} |".format(
            **r
        )
    )

(out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote", out / "comparison.csv")
print("Wrote", out / "summary.md")
