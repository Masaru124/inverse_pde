import json

# Load both metrics files
with open("results_recovery_gpu_fast_stable_quick_full_v3/metrics.json") as f:
    quick_v3 = json.load(f)

with open("results_recovery_gpu_fast_stable_quick_full_v3/controlled_comparison_results.json") as f:
    controlled = json.load(f)

print("=" * 70)
print("CRITICAL ANALYSIS: MODEL vs BASELINE COMPARISONS")
print("=" * 70)

print("\n1. PINN BASELINE - CONVERGENCE ISSUE:")
print("-" * 70)
quick_pinn = quick_v3['baselines']['pinn']
print(f"Quick evaluation (metrics.json):")
print(f"  Converged: {quick_pinn['converged_fraction']:.1%} of instances")
print(f"  RMSE: {quick_pinn['rmse']:.4f}")
print(f"  Max steps: 1000, Avg steps taken: {quick_pinn['avg_steps']:.0f}")

controlled_pinn = controlled['pinn_controlled']
print(f"\nControlled evaluation (full convergence):")
print(f"  Converged: {controlled_pinn['converged_fraction']:.1%} of instances")
print(f"  RMSE: {controlled_pinn['rmse_mean']:.4f} ± {controlled_pinn['rmse_std']:.4f}")
print(f"  Max steps: {controlled_pinn['max_steps_allowed']}, Avg steps: {controlled_pinn['avg_steps']:.0f}")

print(f"\n⚠️  PINN RMSE DISCREPANCY: {quick_pinn['rmse']:.4f} vs {controlled_pinn['rmse_mean']:.4f}")
print(f"   With full convergence, PINN is {controlled_pinn['rmse_mean']/quick_pinn['rmse']:.1f}× WORSE!")

print("\n2. MODEL vs BASELINES - FAIR COMPARISON:")
print("-" * 70)
model_rmse = quick_v3['main_model']['rmse']
gp_rmse = quick_v3['baselines']['gp']['rmse']
pinn_rmse_fair = controlled_pinn['rmse_mean']  # Use fully converged PINN

print(f"Model RMSE:     {model_rmse:.4f}")
print(f"GP RMSE:        {gp_rmse:.4f}  (Model is {(gp_rmse/model_rmse - 1)*100:.1f}% better)")
print(f"PINN RMSE:      {pinn_rmse_fair:.4f}  (Model is {(pinn_rmse_fair/model_rmse - 1)*100:.1f}% better)")

print("\n3. MODEL CALIBRATION/RELIABILITY:")
print("-" * 70)
model_ece = quick_v3['main_model']['ece']
gp_ece = quick_v3['baselines']['gp']['ece']
pinn_ece = controlled_pinn['ece_mean']

print(f"Model ECE:      {model_ece:.4f}")
print(f"GP ECE:         {gp_ece:.4f}    (Model is {(gp_ece/model_ece):.1f}× better calibrated)")
print(f"PINN ECE:       {pinn_ece:.4f}    (Model is {(pinn_ece/model_ece):.1f}× better calibrated)")

print("\n4. POTENTIAL IMPROVEMENT ANGLES:")
print("-" * 70)

# Check if model performs worse on specific k_types
by_k = quick_v3['main_model']['by_k_type']
print("\nPerformance by K Type (model's breakdown):")
for k_type, metrics in by_k.items():
    print(f"  {k_type:20s}: RMSE {metrics['rmse']:.4f}, ECE {metrics['ece']:.4f}")

# Check OOD performance
print("\nOOD Robustness (model's weak points):")
for ood_type, metrics in quick_v3['ood'].items():
    print(f"  {ood_type:25s}: RMSE {metrics['rmse']:.4f}, ECE {metrics['ece']:.4f}, Coverage {metrics['coverage']:.1%}")

print("\n5. STATISTICAL SIGNIFICANCE:")
print("-" * 70)
print(f"\nModel vs GP: Δ RMSE = {abs(model_rmse - gp_rmse):.4f}")
print(f"  - Difference: {abs(model_rmse - gp_rmse):.2%} of model error")
print(f"  - Is this within noise? Need std dev to assess.")

print(f"\nModel vs PINN: Δ RMSE = {pinn_rmse_fair - model_rmse:.4f}")
print(f"  - Model is {(pinn_rmse_fair/model_rmse):.2f}× better (substantial)")
print(f"  - PINN std dev: {controlled_pinn['rmse_std']:.4f}")
