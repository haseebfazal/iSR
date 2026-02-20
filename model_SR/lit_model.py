import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary

# import PSNR and SSIM metrics from torchmetrics
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .arch import rt4ksr_rep


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (pseudo-L1)"""
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon ** 2))
        return loss

class LitRT4KSR_Rep(pl.LightningModule):
    def __init__(
        self, 
        config
    ):
        super().__init__()
        self.config = config
        self.lr = config.learning_rate
        self.model = rt4ksr_rep(config)
        # summary(self.model, (3, 128, 128), device='cpu')

        # self.l1_loss_fn = nn.L1Loss()
        self.l1_loss_fn = CharbonnierLoss(epsilon=1e-6)
        self.hidden_state = None  # Initialize the hidden state
        self.total_batch_counter = 0  # Initialize batch counter
        # add metrics to monitor during training
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.val_psnr_best = MaxMetric()
        
    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        # Reset hidden state at the beginning of each epoch
        self.hidden_state = None

    def on_validation_epoch_start(self):
        # Reset hidden state at the beginning of each epoch
        self.hidden_state = None
    
    def training_step(self, batch, batch_idx):
        self.total_batch_counter += 1

        # Reset the hidden state at the start of every 30th batch
        if self.total_batch_counter % 75 == 0:
            self.hidden_state = None
        # self.hidden_state = None  # Reset hidden state for each batch
        image_lr, image_hr = batch['lr'], batch['hr']
        image_lr = image_lr.to(self.device)
        image_sr, new_hidden_state = self.model(image_lr, self.hidden_state)
        # self.hidden_state = new_hidden_state.detach()
        h_next, c_next = new_hidden_state
        self.hidden_state = (h_next.detach(), c_next.detach())


        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.train_psnr(image_sr, image_hr)
        ssim = self.train_ssim(image_sr, image_hr)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        # log images to wandb log 4 times per epoch
        if batch_idx % 500 == 0:
            grid = torchvision.utils.make_grid(torch.cat((image_sr[:1], image_hr[:1]), dim=0))
            self.logger.experiment.add_image('train_images', grid, self.global_step)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def validation_step(self, batch, batch_idx):
        # self.hidden_state = None  # Reset hidden state for each batch
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr, new_hidden_state = self.model(image_lr, self.hidden_state)
        # self.hidden_state = new_hidden_state.detach()
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.val_psnr(image_sr, image_hr)
        ssim = self.val_ssim(image_sr, image_hr)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx % 500 == 0:
            # show random single images lr, sr, hr in tensorboard
            index = torch.randint(0, image_sr.shape[0], (1,))
            grid = torchvision.utils.make_grid(torch.cat((image_sr[index:index+1], image_hr[index:index+1]), dim=0))
            self.logger.experiment.add_image('val_images', grid, self.global_step)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def test_step(self, batch, batch_idx):
        image_lr, image_hr = batch['lr'], batch['hr']
        image_sr = self.forward(image_lr)
        loss = self.l1_loss_fn(image_sr, image_hr)
        psnr = self.val_psnr(image_sr, image_hr)
        ssim = self.val_ssim(image_sr, image_hr)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "psnr": psnr, "ssim": ssim}
    
    def predict_step(self, image_lr, external_hidden_state=None):
        image_sr, new_hidden_state= self.model(image_lr, external_hidden_state)
        return image_sr, new_hidden_state
    
    def configure_optimizers(self):
        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5, 
                                          betas=(0.9, 0.9999), amsgrad=False)
        elif self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5,
                                         betas=(0.9, 0.9999), amsgrad=False)
        elif self.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        else:
            raise NotImplementedError("Optimizer not implemented")
        
        # create learning rate scheduler with halved for every 200000 steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.multistepLR_milestones, 
            gamma=self.config.multistepLR_gamma, 
            verbose=False,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": self.config.lr_monitor_logging_interval,
                "frequency": 1,
            },
        }
    
