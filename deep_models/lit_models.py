"""
Lightning module for the attention-MIL bag classifier (stage 2).
The stage-1 encoders (ConvVAE / ConvVQVAE) are trained in vae_train.py / vqvae_train.py.
"""
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import BinaryAUROC

from .mil import AttnMILHead


class BagClassifier(L.LightningModule):
    def __init__(self, d_in, d_hid=64, dropout=0.25, lr=1e-3, weight_decay=1e-2, pos_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.head = AttnMILHead(d_in, d_hid, dropout)
        self.pos_weight: torch.Tensor
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.val_auc, self.test_auc = BinaryAUROC(), BinaryAUROC()

    def forward(self, x):
        return self.head(x)

    def _step(self, batch, stage, metric=None):
        x, y = batch
        logit, _ = self(x)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=self.pos_weight)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(y))
        if metric is not None:
            metric.update(torch.sigmoid(logit), y.int())
            self.log(f"{stage}_auc", metric, on_epoch=True, prog_bar=True, batch_size=len(y))
        return loss

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch, i):
        self._step(batch, "val", self.val_auc)

    def test_step(self, batch, i):
        self._step(batch, "test", self.test_auc)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(self.parameters(), 
                                lr=self.hparams["lr"], 
                                weight_decay=self.hparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               factor=0.5, 
                                                               patience=5)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
