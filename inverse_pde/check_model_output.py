import torch
from model.model import AmortizedInversePDEModel
from data.dataset import build_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')

train_loader, _, _ = build_dataloaders(
    'data/generated', 40000, 5000, 5000, 64, seed=42,
    num_workers=0, pin_memory=True, normalize_k_per_instance=True
)

model = AmortizedInversePDEModel(
    grid_size=32, d_model=96, n_heads=4, n_layers=3,
    dropout=0.15, include_time=False, n_targets=1
).to(device)
model.eval()

batch = next(iter(train_loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

with torch.no_grad():
    mu, sigma = model(
        batch['obs_coords'],
        batch.get('obs_times'),
        batch['obs_values'],
        batch['obs_key_padding_mask']
    )

print(f'mu mean={mu.mean().item():.6f} mu std={mu.std().item():.6f}')
print(f'mu min={mu.min().item():.6f} mu max={mu.max().item():.6f}')
print(f'mu sample (first 5)={mu[0].flatten()[:5]}')
print(f'target (k_grid) mean={batch["k_grid"].mean().item():.6f} target std={batch["k_grid"].std().item():.6f}')
print(f'target sample (first 5)={batch["k_grid"][0].flatten()[:5]}')
