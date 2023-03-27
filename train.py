#!/usr/bin/env python3
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from datamodule.mnist import MNISTDataModule
import hydra


class MNISTModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.validation_step_outputs = []

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.model.hidden_dim)
        self.dropout = torch.nn.Dropout(self.hparams.model.dropout)
        self.l2 = torch.nn.Linear(self.hparams.model.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = torch.nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).float() / y.size(0)

        self.log('val_loss', val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        step_output = {"val_loss": val_loss, "val_acc": acc}
        self.validation_step_outputs.append(step_output)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def val_test_epoch_end(self):
        avg_loss = torch.stack([x['val_loss']
                               for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc']
                              for x in self.validation_step_outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc', avg_acc, on_epoch=True, prog_bar=True)
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

    datamodule = MNISTDataModule(cfg.data)

    model = MNISTModel(cfg)
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
