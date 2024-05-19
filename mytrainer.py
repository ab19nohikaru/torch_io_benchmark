import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import logging, time
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        dataloader_type: str, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.dataloader_type = dataloader_type

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        t0 = time.time()
        for batch in self.train_data:
            if self.dataloader_type == "tensorclass":
                source = batch.images.contiguous().to(self.gpu_id)
                targets = batch.targets.contiguous().to(self.gpu_id)
            else:
                source, targets = batch
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        logger.info(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} |"
                    f"Time: {time.time() - t0: 4.4f} s")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
