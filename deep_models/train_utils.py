"""Helpers shared by vae_train.py and vqvae_train.py."""
import torch


@torch.no_grad()
def mean_offdiag_per_band(loader, n_batches=None, offdiag=True):
    """Mean value per frequency band over all windows (diagonal excluded by default).
    Shapes are inferred from the data: batches are (x, ...) with x = (exams, windows, bands, C, C)."""
    total, count = None, 0
    for i, batch in enumerate(loader):
        x = batch[0].flatten(0, 1).double()                          # (N, B, C, C)
        if offdiag:
            x = x[..., ~torch.eye(x.shape[-1], dtype=torch.bool)]    # (N, B, C*(C-1)), diagonal removed
        else:
            x = x.flatten(2)                                         # (N, B, C*C)
        s = x.sum(dim=(0, 2))
        total = s if total is None else total + s
        count += x.shape[0] * x.shape[2]
        if n_batches is not None and i + 1 >= n_batches:
            break
    if total is None:
        raise ValueError("empty loader")
    return (total / count).float()                                   # (B,)


def unpack_batch(batch, device, cond_dim=0):
    """(bag, label[, cond]) -> (x, cond) with windows flattened into the batch dimension.

    x:    (exams * windows, bands, C, C)
    cond: (exams * windows, cond_dim) -- each exam's condition repeated for all its windows -- or None.
    """
    x = batch[0]
    n_windows = x.shape[1]
    x = x.flatten(0, 1).to(device)
    cond = None
    if cond_dim:
        if len(batch) < 3:
            raise ValueError(f"model expects a condition (cond_dim={cond_dim}): "
                             f"build the dataset with return_cond=True")
        cond = batch[2].to(device).repeat_interleave(n_windows, dim=0)
    return x, cond


class FoldR2:
    """Fold-wide R^2 (variance_weighted) against the 'always predict the mean matrix' baseline.

    Accumulates batch by batch, so the result is the R^2 of the WHOLE loader, not a mean of
    per-batch values. offdiag=True ignores the diagonal (it is masked to 0 in the models)."""

    def __init__(self, offdiag=True):
        self.offdiag = offdiag
        self.n, self.sse = 0, 0.0
        self.sum_x = self.sum_x2 = self.mask = None

    def update(self, x, x_rec):
        xd, rd = x.double(), x_rec.double()
        if self.mask is None:
            mask = torch.ones(xd.shape[1:], dtype=torch.float64, device=xd.device)
            if self.offdiag:
                mask = mask * (1.0 - torch.eye(xd.shape[-1], dtype=torch.float64, device=xd.device))
            self.mask = mask
            self.sum_x, self.sum_x2 = torch.zeros_like(mask), torch.zeros_like(mask)
        self.n += xd.shape[0]
        self.sum_x += xd.sum(0)                      # type: ignore
        self.sum_x2 += (xd ** 2).sum(0)              # type: ignore
        self.sse += (((rd - xd) ** 2) * self.mask).sum().item()

    def compute(self):
        sst = ((self.sum_x2 - self.sum_x ** 2 / self.n) * self.mask).sum().item()   # type: ignore
        return 1.0 - self.sse / sst
