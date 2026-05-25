import math
import torch
import torch.nn as nn



def get_1d_sincos_pos_embed(embed_dim, length):
    """Génère un encodage positionnel sinusoïdal de type Transformer."""
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1) # [length, embed_dim]

def timestep_embedding(timesteps, dim, max_period=10000):
    """Crée des embeddings sinusoïdaux pour le temps d'intégration (t)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiTBlock1D(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        # Self-Attention avec dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)

        # Cross-Attention (Mamba context) avec dropout
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)

        # MLP avec dropout
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # AdaLN Modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        # Initialisation neutre pour la stabilité
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb, mamba_context):
        shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        # 1. Self-Attention
        x_mod = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.self_attn(x_mod, x_mod, x_mod)
        x = x + attn_out

        # 2. Cross-Attention
        x_cross_mod = self.norm2(x) * (1 + scale_cross.unsqueeze(1)) + shift_cross.unsqueeze(1)
        cross_out, _ = self.cross_attn(query=x_cross_mod, key=mamba_context, value=mamba_context)
        x = x + cross_out

        # 3. MLP
        x_mlp_mod = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(x_mlp_mod)

        return x


class DiT1D(nn.Module):
    """
    Le Réseau Principal de Diffusion (Flow Matching) utilisant l'architecture DiT.
    """
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=128, num_blocks=4, num_heads=4, max_len=128, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Réseau d'embedding du Temps
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Projection d'entrée
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Encodage Positionnel
        pos_emb = get_1d_sincos_pos_embed(hidden_dim, max_len).unsqueeze(0)
        self.register_buffer("pos_embedding", pos_emb)
        
        # Blocs Transformer avec dropout propagé
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_dim, num_heads, dropout=dropout) for _ in range(num_blocks)
        ])

        # Tête de sortie
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

        # Initialisation à zéro de la dernière couche
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, t, mamba_context):
        """
        x : [Batch, 4, Length] (Trajectoire de Flow)
        t : [Batch] (Temps entre 0 et 1)
        mamba_context : [Batch, Length, Hidden_Dim] (Sortie de Mamba)
        """
        # 1. Préparation de x
        x = x.transpose(1, 2).contiguous()
        seq_len = x.shape[1]

        # Projection et ajout du Positional Encoding
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :seq_len, :]

        # 2. Préparation du temps
        t_freq = timestep_embedding(t * 1000.0, self.hidden_dim)
        t_emb = self.time_embed(t_freq)

        # 3. Passage dans les blocs DiT
        for block in self.blocks:
            h = block(h, t_emb, mamba_context)

        # 4. Projection finale vers les 2 canaux de vélocité
        h = self.out_norm(h)
        v_pred = self.out_proj(h)

        # 5. Formatage final
        v_pred = v_pred.transpose(1, 2).contiguous()
        return v_pred
