"""Attention-based multiple-instance learning head."""
import torch
import torch.nn as nn


class AttnMILHead(nn.Module):
    def __init__(self, d_in, d_hid=64, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(d_hid, d_hid)    # gated attention (Ilse et al., 2018)
        self.att_U = nn.Linear(d_hid, d_hid)
        self.att_w = nn.Linear(d_hid, 1)
        self.cls = nn.Linear(d_hid, 1)

    def forward(self, x):                       # x: (B, W, D) per-window latents
        h = self.proj(x)                        # (B, W, d_hid)
        s = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).squeeze(-1)  # (B, W) scores
        a = torch.softmax(s, dim=1)             # attention weights, sum to 1 per exam
        z = (a.unsqueeze(-1) * h).sum(1)        # (B, d_hid) exam embedding
        return self.cls(z).squeeze(-1), a       # exam logit, per-window attention
