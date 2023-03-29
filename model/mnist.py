import torch
import torch.nn as nn
from typing import Dict


class MNISTModel(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()

        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

    def compute_loss(self, batch, batch_idx):
        """Return forward results and loss
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss, y_hat

    def log_step(self, loss, pred, batch, batch_idx, prefix) -> Dict:
        _, y = batch
        preds = torch.argmax(pred, dim=1)
        acc = torch.sum(preds == y).float() / y.size(0)
        step_output = {prefix+"loss": loss, prefix+"acc": acc}
        return step_output
