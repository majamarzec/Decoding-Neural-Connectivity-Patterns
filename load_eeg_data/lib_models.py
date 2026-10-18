import config as c
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import DataFrame, read_csv

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as L 
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class S00MILDataset(Dataset):
    def __init__(self, exam_idx, max_instances=None, train=False):

        self.exam_idx = np.asarray(exam_idx)              
        self.max_instances = max_instances #that will be taken from one exam (bag)
        self.train = train #(bool), to define train vs val/ hold-out
        self._mm = None #lazy (big file)

        #from metadata - exams_processed.csv
        meta = read_metadata() #open eagerly (small file)
        self.labels = np.asarray(meta['pathology']).astype(int)


    def __len__(self):
        return len(self.exam_idx) #the length corresponds to exams (bags) not individual instances

    def __getitem__(self, i): 
        if self._mm is None:
            self._mm, _,  _ = load_s00_conn_memmap() # shape = (exams, windows, channels, channels, freq_bands)
        exam_i = self.exam_idx[i]
        bag = self._mm[exam_i] # shape = (windows, channels, channels, freq_bands)

        N_instances = bag.shape[0]
        if self.max_instances and N_instances > self.max_instances:
            if self.train:
                keep = np.sort(np.random.choice(N_instances, self.max_instances, replace=False))
            else:
                keep = np.linspace(0, N_instances - 1, self.max_instances).astype(int)
            bag = bag[keep]

        bag = torch.as_tensor(np.array(bag, dtype=c.dtype))
        bag = bag.permute(0, 3, 1, 2) # shape = (windows, freq_bands, channels, channels)
        return bag, torch.tensor(self.labels[exam_i], dtype=torch.float32)

# 3x3 kernels
class ConvAE(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(5, 16, 3, stride=2, padding=1), 
                                     nn.BatchNorm2d(16), 
                                     nn.ReLU(), # 17 -> 9

                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), 
                                     nn.BatchNorm2d(32), 
                                     nn.ReLU(), # 9 -> 5

                                    nn.Flatten(), 
                                    nn.Linear(32 * 5 * 5, latent))
        
        self.fc = nn.Linear(latent, 32 * 5 * 5)

        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1), 
                                       nn.BatchNorm2d(16), 
                                       nn.ReLU(),# 5 -> 9

                                       nn.ConvTranspose2d(16, 5, 3, stride=2, padding=1))# 9 -> 17

    def forward(self, x):                             
        z = self.encoder(x)
        return self.decoder(self.fc(z).view(-1, 32, 5, 5)), z

# 5x5 kernels
class ConvAE2(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(5, 16, 5, stride=2, padding=2), 
                                     nn.BatchNorm2d(16), 
                                     nn.ReLU(), # 17 -> 9

                                     nn.Conv2d(16, 32, 5, stride=2, padding=2), 
                                     nn.BatchNorm2d(32), 
                                     nn.ReLU(), # 9 -> 5

                                    nn.Flatten(), 
                                    nn.Linear(32 * 5 * 5, latent))
        
        self.fc = nn.Linear(latent, 32 * 5 * 5)

        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2), 
                                       nn.BatchNorm2d(16), 
                                       nn.ReLU(),# 5 -> 9

                                       nn.ConvTranspose2d(16, 5, 5, stride=2, padding=2))# 9 -> 17

    def forward(self, x):                             
        z = self.encoder(x)
        return self.decoder(self.fc(z).view(-1, 32, 5, 5)), z

#not reducing spacial resolution (inspired by http://dx.doi.org/10.1109/JBHI.2019.2941222)
class ConvAE3(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(5, 16, 5, stride=1, padding=2), 
                                     nn.BatchNorm2d(16), 
                                     nn.ReLU(), # 17 -> 17

                                     nn.Conv2d(16, 32, 5, stride=1, padding=2), 
                                     nn.BatchNorm2d(32), 
                                     nn.ReLU(), # 17 -> 17

                                    nn.Flatten())
        
        self.fc_encode = nn.Linear(32 * 17 * 17, latent)
        self.fc_decode = nn.Linear(latent, 32 * 17 * 17)

        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 5, stride=1, padding=2), 
                                       nn.BatchNorm2d(16), 
                                       nn.ReLU(),# 17 -> 17

                                       nn.ConvTranspose2d(16, 5, 5, stride=1, padding=2))# 17 -> 17

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        h = self.fc_encode(z)
        h_reshaped = self.fc_decode(h).view(batch_size, 32, 17, 17)
        reconstructed = self.decoder(h_reshaped)
        return reconstructed, z

class BagEncoder(L.LightningModule):
    def __init__(self, ae, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.ae = ae
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['ae'])

    def forward(self, x): 
        # x shape = (exams (selected bags), windows (moderated by max_instances), freq_bands, channels, channels)
        n_exams, n_windows = x.shape[:2]
        recon, z = self.ae(x.flatten(0, 1))
        return recon.view(n_exams, n_windows, *recon.shape[1:]), z.view(n_exams, n_windows, -1)

    def _step(self, batch, stage):
        x, _ = batch #not using labels in unsupervised AE
        recon, _ = self(x) # self == model
        loss = F.mse_loss(recon, x)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True,
                 batch_size=x.shape[0])
        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        self._step(batch, "val")

    def predict_step(self, batch):
        x, _ = batch
        recon, z = self(x)
        return recon, z

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.Adam(self.parameters(), 
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, 
                             "monitor": "val_loss"}}

##########
def read_metadata(metadata_path: str = c.metadata_path) -> DataFrame:
    return read_csv(metadata_path)

def load_s00_conn_memmap():
    meta = read_metadata()
    i_exams = np.where(np.asarray(meta['site']) == 'S00')[0]
    n_exams = len(i_exams)
    mm = np.memmap(c.s00_conn_path, 
                dtype = c.dtype, 
                mode = 'r', #only reading
                shape = (n_exams, c.N_windows, c.N_channels, c.N_channels, len(c.freq_bands)))
    return mm, i_exams, n_exams

def heatmap_bands(mats, cmap='viridis', input_type='original'):
    """
    input_type='original'      : mats is (C, C, B)
    input_type='reconstructed' : mats is (B, C, C), torch tensor or numpy array
    """
    if isinstance(mats, torch.Tensor):
        mats = mats.detach().cpu().numpy()
    mats = np.asarray(mats)

    if input_type == 'reconstructed':
        mats = np.moveaxis(mats, 0, -1)#(B,C,C) -> (C,C,B)
    elif input_type != 'original':
        raise ValueError(f"unknown input_type: {input_type!r}")

    band_names = list(c.freq_bands.keys())
    vmin, vmax = mats.min(), mats.max()

    fig, axes = plt.subplots(1, mats.shape[-1], figsize=(15, 5), squeeze=False)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        sns.heatmap(mats[:, :, i].T, ax=ax, xticklabels=c.channels, yticklabels=c.channels,
                    cmap=cmap, vmin=vmin, vmax=vmax, square=True, cbar=False)
        ax.set_title(band_names[i])
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap)  # type: ignore
    fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.02, label='Connectivity strength')
    fig.suptitle(f"{input_type} band-aggregated ffDTF", fontweight='bold', fontsize=15, y=0.8)
    return fig

def compare_heatmaps(val_ds, connectivity_mm, reconstructed, recon_i, window_i):
    val_exam_ids = val_ds.exam_idx 
    heatmap_bands(connectivity_mm[val_exam_ids[recon_i]][window_i], input_type='original')
    heatmap_bands(reconstructed[recon_i][window_i], input_type='reconstructed')
    plt.show()
###########