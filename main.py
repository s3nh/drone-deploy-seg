import os 
import re  
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from models.enet import LightningEnet
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

def main():
    model = LightningEnet(num_classes = 6)
    os.makedirs('enet_results', exist_ok = True)
    checkpoint_callback = ModelCheckpoint(
            filepath = "enet_results", 
            verbose = True, 
            monitor = "val_loss", 
            mode = "min", 
            prefix = "", 
            )

    trainer = pl.Trainer(gpus = 1, 
            default_save_path = 'enet_results', 
            max_epochs = 100, 
            checkpoint_callback = checkpoint_callback, 
            show_progres_bar = True, 
            )

    trainer.fit(model)

if __name__ == "__main__":
    main()
