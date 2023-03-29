#!/usr/bin/env python3
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra


class LitModule(pl.LightningModule):
    """A generic boilerplate PyTorch-Lightning module
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = hydra.utils.instantiate(cfg.model)
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        train_loss, pred = self.model.compute_loss(batch, batch_idx)
        step_output = self.model.log_step(
            train_loss, pred, batch, batch_idx, prefix="train/")

        # log step output
        for key, value in step_output.items():
            self.log(key, value, on_step=True, on_epoch=False,
                     prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, pred = self.model.compute_loss(batch, batch_idx)
        step_output = self.model.log_step(
            val_loss, pred, batch, batch_idx, prefix="val/")

        # log step output
        for key, value in step_output.items():
            self.log(key, value, on_step=True, on_epoch=False,
                     prog_bar=False, logger=True)
        self.validation_step_outputs.append(step_output)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def val_test_epoch_end(self):
        # accumulate the validation step outputs for each key
        for key in self.validation_step_outputs[0].keys():
            values = torch.stack([x[key]
                                 for x in self.validation_step_outputs])
            mean_value = values.mean()
            self.log("epoch_"+key, mean_value, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.val_test_epoch_end()

    def on_test_epoch_end(self):
        self.val_test_epoch_end()

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer.name
        optimizer_lr = self.hparams.optimizer.lr
        optimizer_weight_decay = self.hparams.optimizer.weight_decay

        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(
            self.parameters(),
            lr=optimizer_lr, weight_decay=optimizer_weight_decay)

        return optimizer


@hydra.main(version_base="1.3", config_path="config", config_name="train")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(42)

    datamodule = hydra.utils.instantiate(cfg.data)

    model = LitModule(cfg)
    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val)

    # Train the model and compute the validation metrics
    trainer.fit(model, datamodule=datamodule)
    results = trainer.test(model, datamodule=datamodule)
    print('Validation accuracy:', results)


if __name__ == "__main__":
    run()
