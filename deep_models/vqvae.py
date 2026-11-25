"""ConvVQVAE for (bands, C, C) connectivity matrices, plain PyTorch.

ResNet/GroupNorm architecture (17 -> 9 -> 5 latent grid), VectorQuantizer learned by gradient (no EMA),
configurable output activation (default sigmoid) and off-diagonal mask.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from . import config as c

OUT_ACTS = ("sigmoid", "tanh", "none")


class VectorQuantizer(nn.Module):
    """EMA VectorQuantizer with input LayerNorm and dead-code revival."""
    def __init__(self, code_book_size=64, embedding_dim=10, commitment_cost=0.25,
                 decay=0.99, eps=1e-5):
        super().__init__()
        self.code_book_size = code_book_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.norm = nn.GroupNorm(1, embedding_dim)  # keeps z_e scale bounded
        self.embedding = nn.Embedding(code_book_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)

        self.register_buffer("ema_cluster_size", torch.zeros(code_book_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, inputs):
        inputs = self.norm(inputs)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()          # (B, H, W, C)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)          # (N, C)

        with torch.no_grad():
            distances = (
                flat_input.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2.0 * flat_input @ self.embedding.weight.t()
            )
            encoding_indices = torch.argmin(distances, dim=1)     # (N,)

        quantized = self.embedding(encoding_indices).view(input_shape)

        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices, self.code_book_size).float()
                self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
                dw = encodings.t() @ flat_input
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing + update codebook
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.code_book_size * self.eps) * n
                )
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

                # Revive dead codes (usage < 1.0) using random encoder vectors from current batch
                dead = self.ema_cluster_size < 1.0
                if dead.any():
                    rand_idx = torch.randint(0, flat_input.size(0), (int(dead.sum().item()),), device=flat_input.device)
                    replacements = flat_input[rand_idx].detach()
                    self.embedding.weight.data[dead] = replacements
                    self.ema_w[dead] = replacements
                    self.ema_cluster_size[dead] = 1.0

        loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())

        if self.training:
            quantized = inputs + (quantized - inputs).detach()

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.view(input_shape[0], -1)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        skip = x
        x = F.elu(self.norm1(x))
        x = F.elu(self.norm2(self.conv1(x)))
        return self.conv2(x) + skip


class DownBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.bn1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)
        self.bn2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)

    def forward(self, x):
        x = F.elu(self.bn1(x))
        x_skip = self.conv3(x)
        x = F.elu(self.bn2(self.conv1(x)))
        return self.conv2(x) + x_skip


class UpBlock(nn.Module):
    """Upsamples to an explicit `size` (17 is odd, so scale_factor=2 would give 18/20)."""
    def __init__(self, channels_in, channels_out, size):
        super().__init__()
        self.bn1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_in, 3, 1, 1)
        self.bn2 = nn.GroupNorm(8, channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        self.up_nn = nn.Upsample(size=size, mode="nearest")

    def forward(self, x_in):
        x = self.up_nn(F.elu(self.bn1(x_in)))
        x_skip = self.conv3(x)
        x = F.elu(self.bn2(self.conv1(x)))
        return self.conv2(x) + x_skip


class VQEncoder(nn.Module):
    def __init__(self, in_ch=5, init_channels=16, latent_channels=10):
        super().__init__()
        # size path (17x17 input): 17 -> 9 -> 5
        self.conv_1 = nn.Conv2d(in_ch, init_channels, 3, 1, 1)
        self.conv_block1 = DownBlock(init_channels, init_channels * 2)      # 17 -> 9
        self.conv_block2 = DownBlock(init_channels * 2, init_channels * 4)  # 9 -> 5
        self.res_block_1 = ResBlock(init_channels * 4)
        self.res_block_2 = ResBlock(init_channels * 4)
        self.res_block_3 = ResBlock(init_channels * 4)
        self.conv_out = nn.Conv2d(4 * init_channels, latent_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = F.elu(self.res_block_3(x))
        return self.conv_out(x)


class VQDecoder(nn.Module):
    def __init__(self, out_ch=5, sizes=(9, 17), init_channels=16, latent_channels=10,
                 out_act="sigmoid", mask_diag=True):
        super().__init__()
        if out_act not in OUT_ACTS:
            raise ValueError(f"out_act must be one of {OUT_ACTS} (got {out_act!r}).")
        self.out_act, self.mask_diag = out_act, mask_diag
        self.conv1 = nn.Conv2d(latent_channels, 4 * init_channels, 3, 1, 1)
        self.res_block_1 = ResBlock(init_channels * 4)
        self.res_block_2 = ResBlock(init_channels * 4)
        self.res_block_3 = ResBlock(init_channels * 4)
        # size path: 5 -> 9 -> 17
        self.conv_block1 = UpBlock(4 * init_channels, 2 * init_channels, sizes[0])  # 5 -> 9
        self.conv_block2 = UpBlock(2 * init_channels, init_channels, sizes[1])      # 9 -> 17
        self.conv_out = nn.Conv2d(init_channels, out_ch, 3, 1, 1)
        if mask_diag:
            self.register_buffer("offdiag_mask", (1.0 - torch.eye(sizes[1])))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_out(x)
        if self.out_act == "sigmoid":
            x = torch.sigmoid(x)
        elif self.out_act == "tanh":
            x = torch.tanh(x)
        if self.mask_diag:
            x = x * self.get_buffer("offdiag_mask")
        return x


class ConvVQVAE(nn.Module):
    """Tutorial VQVAE on a (bands, C, C) input. Latent grid: 5x5 for 17x17 inputs."""
    def __init__(self, latent_channels=10, code_book_size=64, commitment_cost=0.25,
                 init_channels=16, in_ch=None, eeg_channels=None, out_act="sigmoid", mask_diag=True):
        super().__init__()
        in_ch = len(c.freq_bands) if in_ch is None else in_ch
        eeg_channels = c.N_channels if eeg_channels is None else eeg_channels
        down = lambda n: (n + 2 - 3) // 2 + 1                      # k=3, s=2, p=1
        s1 = down(eeg_channels)                                    # 9
        self.grid = down(s1)                                       # 5
        self.out_act, self.mask_diag = out_act, mask_diag
        self.encoder = VQEncoder(in_ch, init_channels, latent_channels)
        self.vq = VectorQuantizer(code_book_size, latent_channels, commitment_cost)
        self.decoder = VQDecoder(in_ch, (s1, eeg_channels), init_channels, latent_channels,
                                 out_act, mask_diag)
        # constructor arguments (resolved), e.g. to store in summary.json next to a checkpoint
        self.config = dict(latent_channels=latent_channels, code_book_size=code_book_size,
                           commitment_cost=commitment_cost, init_channels=init_channels, in_ch=in_ch,
                           eeg_channels=eeg_channels, out_act=out_act, mask_diag=mask_diag)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """He init for conv layers, Xavier for the two output convs; the VQ codebook keeps uniform(-1/K, 1/K)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for layer in [self.encoder.conv_out, self.decoder.conv_out]:
            nn.init.xavier_normal_(layer.weight)

    @torch.no_grad()
    def set_output_bias(self, band_means):
        """Start the output near each band's typical value (inverse of the output activation)."""
        bias = self.decoder.conv_out.bias
        p = band_means.to(device=bias.device, dtype=bias.dtype).clamp(1e-6, 1 - 1e-6) #type:ignore
        if self.out_act == "sigmoid":
            p = torch.logit(p)
        elif self.out_act == "tanh":
            p = torch.atanh(p)
        bias.copy_(p)  #type:ignore

    def forward(self, x):
        z_e = self.encoder(x)
        vq_loss, quantized, idx = self.vq(z_e)
        return self.decoder(quantized), vq_loss, idx


# ======================================================================================
# Codebook diagnostics (indices: (exams, windows, H*W) integer array from vqvae_train.encode_bags)
# ======================================================================================
def active_codes(indices, code_book_size, thr=1e-4):
    """indices: (exams, windows, H*W) integer numpy array from encode_bags.
    Returns (n_active, probs, perplexity)."""
    counts = np.bincount(indices.ravel(), minlength=code_book_size).astype(np.float64)
    probs = counts / counts.sum()
    perplexity = float(np.exp(-np.sum(probs * np.log(probs + 1e-10))))
    n_active = int((probs > thr).sum())
    return n_active, probs, perplexity


def plot_codebook_usage(model, indices, thr=1e-4):
    """Per-code usage frequency and codebook L2 norms, sorted by usage (most frequent first).
    indices: (exams, windows, H*W) integer numpy array from encode_bags on the validation set.
    Returns (fig, order, n_active); order[:n] are the code indices, most frequent first."""
    K = model.vq.code_book_size
    n_active, probs, perplexity = active_codes(indices, K, thr)
    with torch.no_grad():
        code_norms = model.vq.embedding.weight.detach().cpu().norm(dim=1).numpy()

    order = np.argsort(-probs)
    x = np.arange(K)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x, probs[order])
    ax[0].axhline(1.0 / K, ls=":", c="k", label=f"uniform (1/{K})")
    ax[0].axhline(thr, ls="--", c="r", label=f"threshold {thr}")
    ax[0].set_title(f"code usage ({n_active}/{K} active, perplexity = {perplexity:.1f})")
    ax[0].set_ylabel("fraction of spatial positions")
    ax[0].legend()

    ax[1].bar(x, code_norms[order])
    ax[1].set_title("codebook vector L2 norm (sorted by usage)")
    ax[1].set_ylabel("||e_k||_2")

    for a in ax:
        a.set_xlabel("code index (sorted by usage)")
    fig.tight_layout()
    print(f"active codes: {n_active}/{K}   perplexity: {perplexity:.1f}/{K}")
    return fig, order, n_active
