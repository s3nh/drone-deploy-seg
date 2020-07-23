import albumentations
import os 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.models 
import pytorch_lightning as pl 
from src.dataset import DroneDeployDataset

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
        nn.ReLU(inplace =True),
        #Swish(), 
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1), 
        nn.ReLU(inplace = True)
        #Swish(), 
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
        self.lconv  = nn.Conv2d(64, self.n_class, kernel_size = 1)

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

    #def loss(self):
    #    return nn.NLLLoss()

    def training_step(self, train_batch, batch_idx):
        x = train_batch['features'].to(device='cuda', dtype = torch.float32)
        y = train_batch['masks'].to(device='cuda', dtype = torch.long)
        logits = self.forward(x)
        tr_loss = nn.CrossEntropyLoss()(logits, y[:, :, :, 0])
        return {'loss' : tr_loss}

    def validation_step(self, val_batch, val_idx):
        x_val = val_batch['features'].to(device='cuda', dtype = torch.float32)
        y_val = val_batch['masks'].to(device='cuda', dtype = torch.long)
        val_logits = self.forward(x_val)
        val_loss = nn.CrossEntropyLoss()(val_logits, y_val[:, :, :, 0])
        return {'val_loss' : val_loss}


    def get_samples(self):
        label_chips = os.listdir('dataset-medium/label-chips/')
        image_chips = os.listdir('dataset-medium/image-chips/')
        samples = [(os.path.join('dataset-medium/image-chips',x), os.path.join('dataset-medium/label-chips/', y) ) for (x, y) in zip(label_chips, image_chips)]
        return  samples
    
    def prepare_data(self):
        samples = self.get_samples()
        dataset = DroneDeployDataset(samples = samples, transform = albumentations.Compose([ albumentations.LongestMaxSize(max_size = 256, p=1)], p=1))
        
        self.train_data, self.test_data = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = 8)


    def val_dataloader(self):
        return DataLoader(self.test_data, shuffle = False,   batch_size = 8)

    #def test_dataloader(self):
    #    return DataLoader(self.test_data, batch_size = 32)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
        return optimizer 

def main():

    model = LightningUNet(n_class = 6)
    os.makedirs('results', exist_ok=True)
    trainer = pl.Trainer(gpus=1, default_save_path='results')

    trainer.fit(model)


if __name__ == "__main__":
    main()
