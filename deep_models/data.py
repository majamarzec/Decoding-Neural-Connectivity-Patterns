"""Metadata access, memmap loaders and MIL datasets (exam = bag, window = instance)."""
import numpy as np
import torch
from pandas import DataFrame, read_csv
from torch.utils.data import Dataset

from . import config as c

def load_folds(folds_path):
    with np.load(folds_path) as s:
        folds = [s[f"fold_{k}"] for k in range(len(s.files))]
    return folds

def make_splits(dev_folds):
    n = len(dev_folds)
    for k in range(n):
        test_k, val_k = k, (k + 1) % n
        train = np.concatenate([f for j, f in enumerate(dev_folds) if j not in (test_k, val_k)])
        yield k, train, dev_folds[val_k], dev_folds[test_k]

# ---------------------------------------------------------------- loaders
def read_metadata(metadata_path: str = c.metadata_path) -> DataFrame:
    return read_csv(metadata_path)


def load_s00_conn_memmap():
    meta = read_metadata()
    i_exams = np.where(np.asarray(meta['site']) == 'S00')[0]
    n_exams = len(i_exams)
    mm = np.memmap(c.s00_conn_path,
                   dtype=c.dtype,
                   mode='r',  # only reading
                   shape=(n_exams, c.N_windows, c.N_channels, c.N_channels, len(c.freq_bands)))
    return mm, i_exams, n_exams


def load_conn_memmap():
    i_exams = np.arange(0, c.N_exams, 1)
    mm = np.memmap(c.all_sites_conn_path,
                   dtype=c.dtype,
                   mode='r',  # only reading
                   shape=(c.N_exams, c.N_windows, c.N_channels, c.N_channels, len(c.freq_bands)))
    return mm, i_exams


# --------------------------------------------------------------- datasets
class _BaseMILDataset(Dataset):
    """Shared logic: window subsampling, permute, labels, condition.
    Subclasses define the data source (_open_memmap), optionally filter the
    metadata rows (_select_meta) and optionally change the condition (_build_cond)."""

    def __init__(self, exam_idx, max_instances=None, train=False, return_cond=False,
                 age_center=40.0, age_scale=25.0):
        self.exam_idx = np.asarray(exam_idx)
        self.max_instances = max_instances   # max windows taken from one exam (bag)
        self.train = train                   # True: random window subsampling; False: deterministic
        self.return_cond = return_cond       # True -> also return the condition vector
        self._mm = None                      # lazy (big file)

        meta = self._select_meta(read_metadata())   # rows must line up with the memmap
        self.labels = np.asarray(meta['pathology']).astype(int)
        self.cond = self._build_cond(meta, age_center, age_scale)   # (n_exams, cond_dim)
        self.cond_dim = self.cond.shape[1]

    # ---- hooks -----------------------------------------------------------
    def _select_meta(self, meta):
        return meta

    def _open_memmap(self):
        raise NotImplementedError

    def _build_cond(self, meta, age_center, age_scale):
        """Default condition: [age scaled to about [-1, 1], female 0/1]."""
        age = np.asarray(meta['age'], dtype=np.float32)
        female = np.asarray(meta['female'], dtype=np.float32)
        return np.stack([(age - age_center) / age_scale, female], axis=1)

    # ---- Dataset API -----------------------------------------------------
    def __len__(self):
        return len(self.exam_idx)  # exams (bags), not individual instances

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = self._open_memmap()   # (exams, windows, channels, channels, freq_bands)
        exam_i = self.exam_idx[i]
        bag = self._mm[exam_i]               # (windows, channels, channels, freq_bands)

        n = bag.shape[0]
        if self.max_instances and n > self.max_instances:
            if self.train:
                keep = np.sort(np.random.choice(n, self.max_instances, replace=False))
            else:
                keep = np.linspace(0, n - 1, self.max_instances).astype(int)
            bag = bag[keep]

        bag = torch.as_tensor(np.array(bag, dtype=c.dtype))
        bag = bag.permute(0, 3, 1, 2)        # (windows, freq_bands, channels, channels)

        label = torch.tensor(self.labels[exam_i], dtype=torch.float32)
        if self.return_cond:
            return bag, label, torch.as_tensor(self.cond[exam_i])   # cond: (cond_dim,) float32
        return bag, label


class MILDataset(_BaseMILDataset):
    """
    All sites. Condition = [age, female, site one-hot...].
    """

    def __init__(self, *args, site_categories=None, **kw):
        self.site_categories = site_categories   # must be set before super().__init__ uses it
        super().__init__(*args, **kw)

    def _open_memmap(self):
        return load_conn_memmap()[0]

    def _build_cond(self, meta, age_center, age_scale):
        base = super()._build_cond(meta, age_center, age_scale)       # [age, female]
        sites = np.asarray(meta['site'])
        if self.site_categories is None:
            self.site_categories = sorted(np.unique(sites).tolist())
        cats = np.asarray(self.site_categories)
        if not np.isin(sites, cats).all():
            raise ValueError("Metadata contains sites not in site_categories.")
        onehot = (sites[:, None] == cats[None, :]).astype(np.float32)  # (n_exams, n_sites)
        return np.concatenate([base, onehot], axis=1)


class S00MILDataset(_BaseMILDataset):
    """
    S00 only. Condition = [age, female]. 
    exam_idx indexes the S00 subset (0 .. n_S00 - 1).
    """

    def _select_meta(self, meta):
        return meta[meta['site'] == 'S00'].reset_index(drop=True)

    def _open_memmap(self):
        return load_s00_conn_memmap()[0]
