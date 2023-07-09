import logging
from typing import Any, Dict, List

import numpy as np
import torch

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    """
    A class representing a Trainer, which is responsible for training a model.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training data.
    test_loader : torch.utils.data.DataLoader
        The DataLoader for the testing data.
    loss_function : torch.nn.Module
        The loss function to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    learning_rate : float
        The learning rate to use for training.
    device : torch.device
        The device (e.g., CPU or GPU) to use for training.
    """
    model: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    loss_function: nn.Module
    optimizer: Optimizer
    learning_rate: float
    device: torch.device
    metrics: Dict[str, Any]


    def __post_init__(self):
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")


    def train(self, epochs: int):
        """Trains the model for a number of epochs.

        Parameters
        ----------
        epochs : int
            Iterations over the dataset
        """
        total_loss = 0.0
        for epoch in tqdm(range(epochs)):
            logger.info(f"Training: Epoch {epoch + 1}...")

            self.model.train()

            for _, data in tqdm(enumerate(self.train_loader)):

                batch, labels = data[0].float(), data[1]

                batch = batch.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(batch)

                loss = self.loss_function(logits, labels)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            self.metrics["loss"].append(total_loss)

            logger.info(f"Training Loss: {total_loss/len(self.train_loader):.2f}")

            # Evaluation
            logger.info(f"Evaluating: Epoch {epoch + 1}...")

            metrics = self.evaluate(epoch=epoch)

            logger.info(f"Accuracy: {metrics['accuracy']}")
            

    def evaluate(self, epoch: int):
        self.model.eval()
    
        metrics = {}
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].float(), data[1]

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)

                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu())
                all_targets.extend(labels.cpu())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        self.metrics["epoch"].append(epoch)

        self.metrics["accuracy"].append(
            accuracy_score(all_targets, all_preds)
        )

        metrics["conf_matrix"] = confusion_matrix(all_targets, all_preds)

        return metrics
