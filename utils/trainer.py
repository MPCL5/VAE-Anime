import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    # TODO: Too many dependencies. should be refactored!
    def __init__(self, name: str, max_patience: int, num_epochs: int,
                 model: nn.Module, optimizer, training_loader: DataLoader, val_loader: DataLoader, data_transfer=None) -> None:
        self.name = name
        self.max_patience = max_patience
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.val_loader = val_loader
        self.data_transfer = data_transfer

    def train(self):
        self.model.train() # put our model in train mode
        for _, batch in enumerate(self.training_loader):
            # batch = batch.float()
            # batch = batch.to(DEVICE)
            batch = self.data_transfer(batch)

            if hasattr(self.model, 'dequantization'):
                if self.model.dequantization:
                    batch = batch + torch.rand(batch.shape)

            loss = self.model.forward(batch)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def evaluate(self, epoch):
        pass

    def start_engine(self):
        nll_val = []
        best_nll = 1000.
        patience = 0

        for e in range(self.num_epochs):
            self.train()
            loss_val = self.evaluate(e)
            
            nll_val.append(loss_val)  # save for plotting
