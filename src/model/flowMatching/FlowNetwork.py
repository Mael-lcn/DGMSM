from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .nn import (
    SiLU,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock1D(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.use_checkpoint = use_checkpoint

        # Couches de traitement de la trajectoire (1D)
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(1, channels, channels, 3, padding=1),
        )

        # Projection du "Super Contexte" (Temps + Mamba) pour FiLM
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(emb_channels, 2 * channels), # Scale & Shift
        )

        self.out_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(1, channels, channels, 3, padding=1)),
        )

    def forward(self, x, emb):
        """
        x: [Batch, Channels, 16]
        emb: [Batch, emb_channels] (Temps + Mamba)
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)

        # Projection du conditionnement et alignement des dimensions
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out.unsqueeze(-1) # [Batch, 2*Channels, 1]

        # Application du Scale and Shift (FiLM)
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = h * (1 + scale) + shift

        h = self.out_layers(h)
        return x + h


class AttentionBlock1D(nn.Module):
    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention() # On réutilise la classe QKVAttention de unet.py
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, length = x.shape
        # On traite la dimension temporelle comme la dimension spatiale
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, length)
        h = self.attention(qkv)
        h = h.reshape(b, -1, length)
        h = self.proj_out(h)
        return x + h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])



class FlowNetwork1D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=2,
        model_channels=128,
        num_blocks=4,
        mamba_dim=256,
        dropout=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.model_channels = model_channels

        # Embedding du Temps d'intégration (tau)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Dimension totale du conditionnement (Temps + Mamba)
        self.cond_dim = time_embed_dim + mamba_dim

        # Projection initiale (Accepte les 4 canaux)
        self.input_proj = conv_nd(1, in_channels, model_channels, 3, padding=1)

        # Empilement de blocs plats (sans changement de résolution)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                ResBlock1D(model_channels, self.cond_dim, dropout, use_checkpoint)
            )
            self.blocks.append(
                AttentionBlock1D(model_channels, num_heads=4, use_checkpoint=use_checkpoint)
            )

        # Tête de sortie (Prédiction de la vélocité v_t sur 2 canaux)
        self.out = nn.Sequential(
            normalization(model_channels),
            SiLU(),
            zero_module(conv_nd(1, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, t, mamba_context):
        """
        x: [Batch, 4, 16] - Trajectoire projetée en Sin/Cos
        t: [Batch] - Temps d'intégration flow matching
        mamba_context: [Batch, 256] - Résumé de l'historique
        """
        # Embedding du temps 
        t_emb = self.time_embed(timestep_embedding(t, self.model_channels))

        # Fusion du conditionnement (Super Contexte)
        super_cond = th.cat([t_emb, mamba_context], dim=-1)

        # Passage dans le réseau
        h = self.input_proj(x)
        for block in self.blocks:
            if isinstance(block, ResBlock1D):
                h = block(h, super_cond)
            else:
                h = block(h)
        
        return self.out(h)
