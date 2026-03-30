import torch
from model.model import AmortizedInversePDEModel
from model.encoder import CrossAttentionEncoder
from model.decoder import ProbabilisticDecoder
from data.dataset import build_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')

train_loader, _, _ = build_dataloaders(
    'data/generated', 40000, 5000, 5000, 64, seed=42,
    num_workers=0, pin_memory=True, normalize_k_per_instance=True
)

encoder = CrossAttentionEncoder(
    grid_size=32, d_model=96, n_heads=4, n_layers=3, dropout=0.15, include_time=False
).to(device)

decoder = ProbabilisticDecoder(d_model=96, dropout=0.15, n_targets=1).to(device)

encoder.eval()
decoder.eval()

batch = next(iter(train_loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

with torch.no_grad():
    latent = encoder(
        batch['obs_coords'],
        batch.get('obs_times'),
        batch['obs_values'],
        batch['obs_key_padding_mask']
    )
    print(f'latent shape={latent.shape}')
    print(f'latent mean={latent.mean().item():.6f} latent std={latent.std().item():.6f}')
    print(f'latent min={latent.min().item():.6f} latent max={latent.max().item():.6f}')
    
    mu, sigma, log_sigma = decoder(latent)
    print(f'\nmu (decoder output) mean={mu.mean().item():.6f} mu std={mu.std().item():.6f}')
    print(f'mu min={mu.min().item():.6f} mu max={mu.max().item():.6f}')
    print(f'log_sigma mean={log_sigma.mean().item():.6f}')
    print(f'\ntarget (k_grid) mean={batch["k_grid"].mean().item():.6f} target std={batch["k_grid"].std().item():.6f}')
