import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.models 
import pytorch_lightning as pl 


def swish(input):
    return input * torch.sigmoid(input)

class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)

def dconv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), 
        #nn.ReLU(inplace =True),
        Swish(), 
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1), 
        #nn.ReLU(inplace = True)
        Swish(), 
    )

class LightningUNet(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()

        self.n_class = n_class
        self.dconv1 = dconv(3, 64)
        self.dconv2 = dconv(64, 128)
        self.dconv3 = dconv(128, 256)
        self.dconv4 = dconv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        self.uconv3 = dconv(256 + 512, 256)
        self.uconv2 = dconv(128 + 256, 128)
        self.uconv1 = dconv(64 + 128, 64)
        self.lconv  = nn.Conv2d(64, self.n_class, 1)

    def forward(self, x):

        conv1 = self.dconv1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv2(x)

        x = self.maxpool(conv2)
        conv3 = self.dconv3(x)

        x = self.maxpool(conv3)
        x = self.dconv4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim = 1)

        x = self.uconv3(x)
        x = self.upsample(x)

        x = torch.cat([x, conv2], dim =1)
        x = self.uconv2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim =1)

        x = self.uconv1(x)
        out = self.lconv(x)
        return out

    def loss(self, logits, labels):
        return nn.BCEWithLogitsLoss(logits, labels)

    def validation_step(self, val_batch, val_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {'val_loss' : loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack[x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'vall_loss' : avg_loss}
        return {'avg_val_loss'  avg_loss, 'log' : tensorboard_logs}
    
    def prepare_data(self):
        # Add after data loader preparation
        raise NotImplementedError


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = 32)


    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = 32)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = 32)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
        return optimizer 

def main():

    model = LightningUNet()
    trainer = pl.Trainer()

    trainer.fit(model)
