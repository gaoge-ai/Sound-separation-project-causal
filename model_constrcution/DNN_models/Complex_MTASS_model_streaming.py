

import torch
import pytorch_lightning as pl
from DNN_models.Complex_MTASS_streaming import *
from DNN_models.Complex_MTASS_Solver import *

class ComplexMTASSLightningStreaming(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        model_class,
        loss_class,
        is_causal=True,
        mse_loss_weight=1.0,
        sisdr_loss_weight=1.0,
        l1_loss_weight=0.0,
        magnitude_l1_loss_weight=0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class', 'loss_class'])
        self.model = model_class(is_causal=is_causal)
        self.loss_wrapper = loss_class
        self.learning_rate = learning_rate
        self.mse_loss_weight = mse_loss_weight
        self.sisdr_loss_weight = sisdr_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.magnitude_l1_loss_weight = magnitude_l1_loss_weight

    def forward(self, x):
        return self.model(x)

    def reset_streaming_state(self):
        reset_fn = getattr(self.model, 'reset_streaming_state', None)
        if callable(reset_fn):
            reset_fn()

    def forward_streaming(self, x):
        stream_fn = getattr(self.model, 'forward_streaming', None)
        if not callable(stream_fn):
            raise AttributeError('Underlying model does not implement forward_streaming().')
        return stream_fn(x)

    def training_step(self, batch, batch_idx):
        X1 = batch[0]
        Y_targets = batch[1:4]
        R_targets = batch[4:7]

        Z1, Z2, Z3 = self(X1)

        loss, mse_loss, sisdr_loss, l1_loss, magnitude_l1_loss = self.loss_wrapper.compute_out_cost(
            Z1,
            Z2,
            Z3,
            Y_targets,
            R_targets,
            mse_loss_weight=self.mse_loss_weight,
            sisdr_loss_weight=self.sisdr_loss_weight,
            l1_loss_weight=self.l1_loss_weight,
            magnitude_l1_loss_weight=self.magnitude_l1_loss_weight,
        )

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('sisdr_loss', -sisdr_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('l1_loss', l1_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('magnitude_l1_loss', magnitude_l1_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        X1 = batch[0]
        Y_targets = batch[1:4]
        R_targets = batch[4:7]
        
        with torch.no_grad():
            Z1, Z2, Z3 = self(X1)
            loss, mse_loss, sisdr_loss, l1_loss, magnitude_l1_loss = self.loss_wrapper.compute_out_cost(
                Z1,
                Z2,
                Z3,
                Y_targets,
                R_targets,
                mse_loss_weight=self.mse_loss_weight,
                sisdr_loss_weight=self.sisdr_loss_weight,
                l1_loss_weight=self.l1_loss_weight,
                magnitude_l1_loss_weight=self.magnitude_l1_loss_weight,
            )

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_sisdr_loss', -sisdr_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_l1_loss', l1_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_magnitude_l1_loss', magnitude_l1_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        grad_norm = self._compute_grad_norm()
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                                        min_lr=5e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": schedular,
                "interval": "epoch",
                "monitor": "val_loss"
            },
        }

    def _compute_grad_norm(self):
        grad_norm_sq = torch.zeros((), device=self.device)
        for param in self.parameters():
            if param.grad is None:
                continue
            param_grad_norm = torch.norm(param.grad.detach(), p=2)
            grad_norm_sq = grad_norm_sq + param_grad_norm.pow(2)
        return torch.sqrt(grad_norm_sq)
