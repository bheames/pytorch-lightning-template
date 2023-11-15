from typing import cast

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LitModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(x))

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(
        self, batch: list[torch.Tensor], batch_idx: int, prefix: str
    ) -> None:
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(f"{prefix}_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
