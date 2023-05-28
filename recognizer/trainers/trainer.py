import logging
import time
import time

import torch

from dataclasses import dataclass

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    loss_function: nn.Module
    optimizer: Optimizer
    learning_rate: float
    device: torch.device


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
            logger.info("Beginning training...")

            self.model.train()

            for i, data in tqdm(enumerate(self.train_loader)):

                batch, labels = data[0].float(), data[1]

                batch = batch.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                logger.info(f"Epoch {epoch+1} | Batch {i+1}...")

                logits = self.model(batch)

                loss = self.loss_function(logits, labels)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1}, Training Loss: {total_loss/len(self.train_loader):.2f}")

            # Evaluation
            logger.info("Beginning of testing...")
            self.model.eval()

            accuracy = 0.0
            count = 0

            with torch.no_grad():
                for data in tqdm(self.test_loader):
                    batch, labels = data[0].float(), data[1]

                    batch = batch.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(batch)

                    accuracy += (torch.argmax(logits, 1) == labels).float().sum()
                    count += len(labels)

                    accuracy /= count

            logger.info(f"Epoch {epoch+1}: Test Accuracy: {(100 * accuracy):.2f}")
