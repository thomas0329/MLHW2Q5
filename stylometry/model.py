# -*- coding: utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class TokenTransformerEncoder(nn.Module):
    def __init__(
            self,
            num_tokens: int = 362,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 3,
            ff_dim: int = 256,
            dropout: float = 0.1,
            emb_dim: int = 128,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L] int64, mask: [B, L] bool. Returns L2-normalized embeddings [B, D]."""
        x = self.token_emb(tokens)  # [B, L, D]
        x = self.pos_enc(x)
        # Transformer key padding mask expects True where positions should be masked
        key_pad = ~mask  # invert: True = pad
        x = self.encoder(x, src_key_padding_mask=key_pad)
        # masked mean over time
        m = mask.float().unsqueeze(-1)  # [B, L, 1]
        sum_x = (x * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp_min(1.0)
        pooled = sum_x / cnt
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class GameEncoderTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 3, ff_dim: int = 512, dropout: float = 0.1, emb_dim: int = 256):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, move_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # move_feats: [B, T, D], mask: [B, T] True for valid
        x = self.pos_enc(move_feats)
        key_pad = ~mask
        x = self.encoder(x, src_key_padding_mask=key_pad)
        m = mask.float().unsqueeze(-1)
        sum_x = (x * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp_min(1.0)
        pooled = sum_x / cnt
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


class BoardStylometryModel(nn.Module):
    def __init__(self, in_channels: int = 4, move_dim: int = 256, d_model: int = 256, n_layers: int = 3, n_heads: int = 4, ff_dim: int = 512, emb_dim: int = 256):
        super().__init__()
        from .move_encoder import MoveEncoderCNN
        self.move_enc = MoveEncoderCNN(in_channels=in_channels, out_dim=move_dim)
        self.proj = nn.Linear(move_dim, d_model)
        self.game_enc = GameEncoderTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim, emb_dim=emb_dim)

    def forward(self, boards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # boards: [B, T, C, H, W], mask: [B, T]
        B, T, C, H, W = boards.shape
        x = boards.view(B * T, C, H, W)
        mf = self.move_enc(x)  # [B*T, move_dim]
        mf = self.proj(mf).view(B, T, -1)  # [B, T, d_model]
        emb = self.game_enc(mf, mask)
        return emb
