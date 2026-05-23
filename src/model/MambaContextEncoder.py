import torch
import torch.nn as nn
from mamba_ssm import Mamba2



class MambaContextEncoder(nn.Module):
    def __init__(self, in_channels=4, mamba_dim=256, num_mamba_layers=4, dropout=0.1):
        super().__init__()

        # GroupNorm remplace BatchNorm pour la stabilité en microbatching
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, mamba_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, mamba_dim // 2), 
            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, mamba_dim)
        )

        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba2(
                    d_model=mamba_dim,
                    d_state=64,
                    headdim=32,
                ),
                'norm': nn.LayerNorm(mamba_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_mamba_layers)
        ])

        self.context_head = nn.Sequential(
            nn.Linear(mamba_dim, mamba_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mamba_dim, mamba_dim)
        )

    def forward(self, x_past):
        h = self.input_proj(x_past)
        h = h.transpose(1, 2).contiguous() # [B, L, D]

        for layer in self.mamba_layers:
            residual = h
            h = layer['norm'](h)
            h = layer['mixer'](h)
            h = layer['dropout'](h)
            h = h + residual

        final_rep = h[:, -1, :] 
        return self.context_head(torch.nn.functional.layer_norm(final_rep, final_rep.shape[1:]))
