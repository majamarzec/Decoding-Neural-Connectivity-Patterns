"""BrainNetCNN-style VAE / CVAE for directed (bands, C, C) connectivity matrices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as c


class E2EConv(nn.Module):
    """Edge-to-Edge: separate 1xC (row) and Cx1 (col) filters summed via broadcasting."""
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.row_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, size), bias=True)
        self.col_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(size, 1), bias=False)

    def forward(self, x):
        return self.row_conv(x) + self.col_conv(x)


class DirectedE2N(nn.Module):
    """Edge-to-Node: aggregates both outgoing (row) and incoming (col) edges per channel."""
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.row_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, size), bias=True)
        self.col_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(size, 1), bias=False)

    def forward(self, x):
        # (N, out_ch, C, 1) + (N, out_ch, 1, C)^T -> (N, out_ch, C, 1)
        return self.row_conv(x) + self.col_conv(x).transpose(-1, -2)


class DirectedN2E(nn.Module):
    """Node-to-Edge: expands node embeddings (N, in_ch, C, 1) into directed (N, out_ch, C, C)."""
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.row_deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, size), bias=True)
        self.col_deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(size, 1), bias=False)

    def forward(self, x):
        return self.row_deconv(x) + self.col_deconv(x.transpose(-1, -2))


class BrainNetCNNVAE(nn.Module):
    def __init__(self, latent=64, e2e_ch=(32, 32), e2n_ch=64, n2g_ch=256,
                 batchnorm=False, in_ch=None, size=None, out_act="sigmoid",
                 mask_diag=True, fc_act=True, logvar_clamp=(-10.0, 2.0),
                 cond_dim=0, cond_in_encoder=True):
        super().__init__()
        in_ch = len(c.freq_bands) if in_ch is None else in_ch
        size = c.N_channels if size is None else size

        if not e2e_ch:
            raise ValueError("e2e_ch must be non-empty.")
        if out_act not in ("sigmoid", "none"):
            raise ValueError(f"out_act must be 'sigmoid' or 'none' (got {out_act!r}).")
        if cond_dim < 0:
            raise ValueError("cond_dim must be >= 0.")

        def norm_act(ch):
            return ([nn.BatchNorm2d(ch)] if batchnorm else []) + [nn.ReLU()]

        # --- Encoder: E2E -> ... -> E2N -> N2G ---
        enc = []
        ch_prev = in_ch
        for ch in e2e_ch:
            enc += [E2EConv(ch_prev, ch, size), *norm_act(ch)]
            ch_prev = ch
        enc += [DirectedE2N(ch_prev, e2n_ch, size), *norm_act(e2n_ch)]
        enc += [nn.Conv2d(e2n_ch, n2g_ch, kernel_size=(size, 1)), *norm_act(n2g_ch)]

        self.latent, self.cond_dim = latent, cond_dim
        self.out_act, self.mask_diag, self.fc_act = out_act, mask_diag, fc_act
        self.logvar_clamp = tuple(logvar_clamp) if logvar_clamp is not None else None
        self.cond_in_encoder = bool(cond_in_encoder) and cond_dim > 0
        self.n2g_ch = n2g_ch

        enc_dim = n2g_ch + (cond_dim if self.cond_in_encoder else 0)
        self.encoder = nn.Sequential(*enc, nn.Flatten())
        self.fc_mu = nn.Linear(enc_dim, latent)
        self.fc_logvar = nn.Linear(enc_dim, latent)

        # --- Decoder: FC -> G2N -> N2E -> E2E -> ... -> out_e2e ---
        self.fc = nn.Linear(latent + cond_dim, n2g_ch)
        dec = [
            nn.ConvTranspose2d(n2g_ch, e2n_ch, kernel_size=(size, 1)), *norm_act(e2n_ch),
            DirectedN2E(e2n_ch, e2e_ch[-1], size), *norm_act(e2e_ch[-1]),
        ]
        for ch_in, ch_out in zip(reversed(e2e_ch[1:]), reversed(e2e_ch[:-1])):
            dec += [E2EConv(ch_in, ch_out, size), *norm_act(ch_out)]
        self.decoder = nn.Sequential(*dec)
        self.out_e2e = E2EConv(e2e_ch[0], in_ch, size)  # raw output layer

        if mask_diag:
            self.register_buffer("offdiag_mask", 1.0 - torch.eye(size))

        self.config = dict(
            arch="BrainNetCNNVAE", latent=latent, e2e_ch=tuple(e2e_ch),
            e2n_ch=e2n_ch, n2g_ch=n2g_ch, batchnorm=batchnorm,
            in_ch=in_ch, size=size, out_act=out_act, mask_diag=mask_diag,
            fc_act=fc_act, logvar_clamp=self.logvar_clamp,
            cond_dim=cond_dim, cond_in_encoder=cond_in_encoder,
        )
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if self.fc_act:
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @torch.no_grad()
    def set_output_bias(self, band_means):
        p = band_means.to(device=self.out_e2e.row_conv.bias.device, #type:ignore
                          dtype=self.out_e2e.row_conv.bias.dtype) #type:ignore
        bias = torch.logit(p.clamp(1e-6, 1 - 1e-6)) if self.out_act == "sigmoid" else p
        self.out_e2e.row_conv.bias.copy_(bias) #type:ignore

    # ---------------------------------------------------------------- forward
    def _cond(self, cond, n):
        if cond is None:
            raise ValueError(f"model was built with cond_dim={self.cond_dim}: "
                             f"pass cond of shape (N, {self.cond_dim}).")
        if cond.shape != (n, self.cond_dim):
            raise ValueError(f"cond must have shape ({n}, {self.cond_dim}), got {tuple(cond.shape)}.")
        return cond

    def encode(self, x, cond=None):
        if self.mask_diag:
            x = x * self.get_buffer("offdiag_mask")
        h = self.encoder(x)
        if self.cond_in_encoder:
            h = torch.cat([h, self._cond(cond, x.shape[0])], dim=1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        if self.logvar_clamp is not None:
            logvar = torch.clamp(logvar, *self.logvar_clamp)
        return mu, logvar

    def decode(self, z, cond=None):
        if self.cond_dim:
            z = torch.cat([z, self._cond(cond, z.shape[0])], dim=1)
        h = self.fc(z)
        if self.fc_act:
            h = F.relu(h)
        x = self.out_e2e(self.decoder(h.view(z.shape[0], self.n2g_ch, 1, 1)))
        if self.out_act == "sigmoid":
            x = torch.sigmoid(x)
        if self.mask_diag:
            x = x * self.get_buffer("offdiag_mask")
        return x

    def forward(self, x, cond=None):
        mu, logvar = self.encode(x, cond)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decode(z, cond), mu, logvar