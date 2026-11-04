# Определяем LightningModule, в котором описаны прямой проход, метрики train/validation/test и оптимизатор AdamW.

import logging
import os
import warnings
from pathlib import Path
from typing import TypeAlias, Union

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

TensorPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
DeviceLikeType: TypeAlias = Union[str, torch.device, int]

# Отключаем ненужное логирование.
os.environ["LT_CLOUD"] = "0"

warnings.filterwarnings("ignore")

logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, custom_params: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Сохраняем словарь с дополнительными параметрами
        self.custom_params = custom_params or {}

    def format_checkpoint_name(self, metrics, filename=None, ver=None):
        # Объединяем дефолтные метрики логирования и наши кастомные параметры
        combined_metrics = {**metrics, **self.custom_params}

        # Передаем объединенный словарь в родительский метод форматирования
        return super().format_checkpoint_name(combined_metrics, filename, ver)


class LitModel(L.LightningModule):
    """
    Полносвязный классификатор изображений MNIST.

    Args:
        model: архитектура сети.
        lr: learning rate оптимизатора AdamW.
        weight_decay: коэффициент затухания весов в AdamW.
        **kwargs: параметры сети.
    """

    def __init__(
        self,
        model: type[nn.Module],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__()

        # Сохраняем гиперпараметры, чтобы архитектуру можно было восстановить из чекпойнта.
        self.save_hyperparameters(ignore="model")

        # Создаём сеть
        self.network = model(**kwargs)
        self.name = model.__name__.lower()

        # CrossEntropyLoss принимает логиты, поэтому Softmax в network не добавляем.
        self.criterion = nn.CrossEntropyLoss()

    def fit(
        self,
        train_loader: DataLoader[TensorPair],
        validation_loader: DataLoader[TensorPair],
        epochs: int = 20,
        patience: int = 3,
        seed: int = 42,
    ):
        """Автоматический процесс обучения. Возвращает датафрейм с метриками."""

        L.seed_everything(seed, workers=True, verbose=False)

        checkpoint = CustomModelCheckpoint(
            dirpath="checkpoints",
            filename="{name}-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            custom_params={"name": self.name},
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
        )

        csv_logger = CSVLogger(save_dir="logs", name=self.name)

        self.trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            logger=csv_logger,
            callbacks=[checkpoint, early_stopping],
            enable_progress_bar=True,
        )

        self.trainer.fit(
            self, train_dataloaders=train_loader, val_dataloaders=validation_loader
        )

        best_path = checkpoint.best_model_path

        if best_path:
            best_state = torch.load(
                best_path, map_location=self.device, weights_only=True
            )

            self.load_state_dict(best_state["state_dict"])

        metrics_path = Path(csv_logger.log_dir) / "metrics.csv"

        metrics_df = pd.read_csv(metrics_path)

        return metrics_df

    def test(self, test_loader: DataLoader[TensorPair]):
        """Тестирование модели на тестовом датасете."""
        return self.trainer.test(self, dataloaders=test_loader, verbose=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты модели для входного батча."""
        return self.network(x)

    def _shared_step(self, batch: TensorPair) -> TensorPair:
        """Выполняет общий прямой проход и вычисляет loss и accuracy."""

        features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()

        return loss, accuracy

    def training_step(self, batch: TensorPair, batch_idx: int) -> torch.Tensor:
        """Вычисляет loss и логирует метрики на обучающем батче."""
        loss, accuracy = self._shared_step(batch)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[1].size(0),
        )

        self.log(
            "train_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch[1].size(0),
        )

        return loss

    def validation_step(self, batch: TensorPair, batch_idx: int) -> None:
        """Вычисляет и логирует метрики на validation-батче."""
        loss, accuracy = self._shared_step(batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[1].size(0),
        )

        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch[1].size(0),
        )

    def test_step(self, batch: TensorPair, batch_idx: int) -> None:
        """Вычисляет и логирует метрики на тестовом батче."""
        loss, accuracy = self._shared_step(batch)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch[1].size(0),
        )

        self.log(
            "test_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch[1].size(0),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
