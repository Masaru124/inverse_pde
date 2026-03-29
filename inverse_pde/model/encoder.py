from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, force_mc: bool = False) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=self.training or force_mc)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = MCDropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.store_weights = False
        self._last_attn_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * head_dim)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        mc_dropout: bool = False,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,M)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self.store_weights:
            self._last_attn_weights = attn_weights.detach().clone()
        attn_weights = self.dropout(attn_weights, force_mc=mc_dropout)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = self._merge_heads(attn_out)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out, force_mc=mc_dropout))
        return x


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        grid_size: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        include_time: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        self.include_time = include_time

        obs_input_dim = 4 if include_time else 3
        self.obs_proj = nn.Linear(obs_input_dim, d_model)

        self.grid_embed = nn.Linear(2, d_model)
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        coords = self._build_grid(grid_size)
        self.register_buffer("grid_coords", coords, persistent=False)

    @staticmethod
    def _build_grid(grid_size: int) -> torch.Tensor:
        x = torch.linspace(0.0, 1.0, grid_size)
        y = torch.linspace(0.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    def forward(
        self,
        obs_coords: torch.Tensor,
        obs_times: torch.Tensor | None,
        obs_values: torch.Tensor,
        obs_key_padding_mask: torch.Tensor | None = None,
        mc_dropout: bool = False,
    ) -> torch.Tensor:
        bsz = obs_coords.shape[0]

        if self.include_time:
            if obs_times is None:
                obs_times = torch.zeros(obs_coords.shape[0], obs_coords.shape[1], 1, device=obs_coords.device)
            obs_raw = torch.cat([obs_coords, obs_times, obs_values], dim=-1)
        else:
            obs_raw = torch.cat([obs_coords, obs_values], dim=-1)
        obs_tokens = self.obs_proj(obs_raw)

        query_coords = self.grid_coords.unsqueeze(0).expand(bsz, -1, -1).to(obs_coords.device)
        query_tokens = self.grid_embed(query_coords)

        x = query_tokens
        for layer in self.layers:
            x = layer(
                query=x,
                key_value=obs_tokens,
                key_padding_mask=obs_key_padding_mask,
                mc_dropout=mc_dropout,
            )

        return x.view(bsz, self.grid_size, self.grid_size, self.d_model)

    def enable_attention_capture(self) -> None:
        for layer in self.layers:
            layer.store_weights = True

    def disable_attention_capture(self) -> None:
        for layer in self.layers:
            layer.store_weights = False

    def get_last_attention_weights(self) -> dict[int, torch.Tensor | None]:
        return {idx: layer._last_attn_weights for idx, layer in enumerate(self.layers)}
