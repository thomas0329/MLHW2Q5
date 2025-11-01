# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    """Generalized End-to-End loss for speaker-style verification.

    Inputs:
    - embeddings: [N*M, D]
    - players_per_batch N, games_per_player M
    Returns:
    - scalar loss
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    @staticmethod
    def _centroids(emb: torch.Tensor, N: int, M: int) -> torch.Tensor:
        # emb: [N*M, D] → [N, M, D]
        D = emb.size(-1)
        e = emb.view(N, M, D)
        c = e.mean(dim=1)  # [N, D]
        return c

    @staticmethod
    def _leave_one_out_centroid(e: torch.Tensor, j: int, i: int) -> torch.Tensor:
        # e: [N, M, D]
        Ej = e[j]  # [M, D]
        M = Ej.size(0)
        if M <= 1:
            return Ej[i]
        return (Ej.sum(dim=0) - Ej[i]) / float(M - 1)

    def forward(self, embeddings: torch.Tensor, N: int, M: int) -> torch.Tensor:
        # Normalize embeddings
        e = F.normalize(embeddings, p=2, dim=-1).view(N, M, -1)  # [N, M, D]
        D = e.size(-1)
        C = e.mean(dim=1)  # [N, D]
        # Similarity matrix S_{ji,k}
        S = []
        for j in range(N):
            row = []
            for i in range(M):
                # leave-one-out centroid for true class j
                c_j_ex = (e[j].sum(dim=0) - e[j][i]) / max(1.0, float(M - 1))
                # cosine vs all centroids
                cos_pos = F.cosine_similarity(e[j][i], c_j_ex, dim=-1)
                cos_negs = F.cosine_similarity(e[j][i].unsqueeze(0), C, dim=-1)  # [N]
                # replace the j-th with leave-one-out value
                cos_all = cos_negs.clone()
                cos_all[j] = cos_pos
                row.append(cos_all)
            row_t = torch.stack(row, dim=0)  # [M, N]
            S.append(row_t)
        S = torch.stack(S, dim=0)  # [N, M, N]
        S = self.w * S + self.b
        # GE2E loss: -S_{ji,j} + logsumexp_k S_{ji,k}
        pos = torch.diagonal(S, dim1=0, dim2=2)  # [M, N] but diag picks [M] per class, we want gather differently
        # Build per (j,i) indices
        loss = 0.0
        for j in range(N):
            for i in range(M):
                logsum = torch.logsumexp(S[j, i, :], dim=-1)
                loss = loss + (-S[j, i, j] + logsum)
        loss = loss / (N * M)
        return loss
