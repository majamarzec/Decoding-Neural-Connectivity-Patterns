"""
k-fold CV for the BagEncoder+ convolutional autoencoder: 
   train (train & val) -> 
   test (holdout) -> 
   original vs reconstructed band-agreggated ffDTF heatmaps
"""

import numpy as np
import json
import os

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as L 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lib_models import MILDataset, ConvAE, ConvAE2, ConvAE3, BagEncoder


folds = [0,1,2,3,4,5,6,7,8,9]
max_epochs = 100
patience = 5
batch_size = 10
num_workers = 6
initial_lr = 1e-3
weight_decay = 1e-5
seed = 0 #for reproducibility


###
chosen_conv_ae = ConvAE
latent = 256
max_instances = 150 #for now let's train/val/test on ALL windows
save_preds = True #save test reconstructions + latents per fold

####
run_name = "model_conv_ae1_l256_mi150"
splits_dir = "splits/all_sites_10_folds.npz"
log_dir = "logs_all"
out_dir = "result_all/" + run_name
os.makedirs(out_dir, exist_ok=True)

results = []

for k in folds:
    with np.load(splits_dir) as s: #type:ignore
        train_idx, val_idx, test_idx = s[f"fold{k}_train"], s[f"fold{k}_val"], s[f"fold{k}_test"]
    print(f"\n=== Fold {k}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ===")

    L.seed_everything(seed + k, workers=True)

    #create Datasets)
    train_dataset = MILDataset(train_idx, max_instances=max_instances,train=True) 
    val_dataset = MILDataset(val_idx, max_instances=max_instances,train=False) 
    test_dataset = MILDataset(test_idx, max_instances=max_instances, train=False) 

    #create DataLoaders
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True,  
                             num_workers=num_workers, 
                             pin_memory=True)

    val_loader = DataLoader(val_dataset,  
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True)

    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers, 
                             pin_memory=True)



    #initialize TensorBoard logger (visualizations in Dashboard)
    #access by $tensorboard --logdir logs  
    logger = TensorBoardLogger(save_dir=log_dir, name=run_name, version=f"fold{k}")

    #initialize model checkpoint; update each time when the val_loss is best (the lowest)
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", filename="best-{epoch}-{val_loss:.4f}") 

    #initialize trainer (wrapper on pure torch), use defined logger and model_checkpoint
    trainer = L.Trainer(max_epochs=max_epochs,
                        accelerator="auto",
                        logger=logger,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=patience), 
                                   model_checkpoint,
                                   LearningRateMonitor(logging_interval='epoch')])


    #initialize the model (Adam optimizer, lr scheduler)
    model = BagEncoder(chosen_conv_ae(latent = latent), lr = initial_lr, weight_decay=weight_decay)

    #TRAINING
    trainer.fit(model, train_loader, val_loader)

    #testing
    #mean squared error per matrix element, averaged over all test exams (each exam weighted equally, up to 150 windows each)
    test_averaged_mse = trainer.test(model, test_loader, ckpt_path="best")[0] #load the best epoch's weights

    if save_preds:
        predictions = trainer.predict(model, test_loader, ckpt_path="best") # (recon, z) of all exams in test sample
        recons  = torch.cat([r for r, _ in predictions]).numpy() #type:ignore
        latents = torch.cat([z for _, z in predictions]).numpy() #type:ignore
        np.savez_compressed(out_dir + f"/fold{k}_test_pred.npz", recon=recons, z=latents, exam_idx=test_idx)

    result = {"fold" : k,
              "best_model_checkpoint": model_checkpoint.best_model_path,
              "best_val_loss": float(model_checkpoint.best_model_score), # type: ignore
              "test_loss": float(test_averaged_mse['test_loss'])
              }
    print(result)
    results.append(result)

#most basic test log (which fold performed the best?)
test = np.array([r["test_loss"] for r in results])
print(f"\nTest loss over {len(results)} folds: {test.mean():.5f} +/- {test.std():.5f}")


with open(out_dir + "/summary.json", "w") as file:
    json.dump({"folds": results, "test_loss_mean": float(test.mean()), "test_loss_std": float(test.std())}, file, indent=2)



