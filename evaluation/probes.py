"""The two probe architectures."""
import torch
import torch.nn as nn


class SoftmaxProbe(nn.Module):
    """Linear layer + softmax (used for the mean-pooled bag probe and the instance probe)."""
    def __init__(self, d, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.fc(x)


class ABMIL(nn.Module):
    """Gated-attention MIL (Ilse et al. 2018). Bag (B, N, d_in) + mask (B, N) -> (logits (B, n_classes), attention (B, N)).
    hidden=None: attention directly over the input features (minimal head);
    hidden=int : a Linear+ReLU+Dropout projection to `hidden` dims first."""

    def __init__(self, d_in, hidden=64, attn_dim=32, dropout=0.25, n_classes=2):
        super().__init__()
        if hidden:
            self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        else:
            self.proj = nn.Identity()
        d = hidden or d_in
        self.V = nn.Linear(d, attn_dim)
        self.U = nn.Linear(d, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, mask=None):
        h = self.proj(x)                                                    # (B, N, d)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)   # (B, N)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))                         # padded windows get zero weight
        a = torch.softmax(a, dim=1)
        z = self.drop((a.unsqueeze(-1) * h).sum(1))                         # (B, d)
        return self.head(z), a
