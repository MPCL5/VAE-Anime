import torch
import numpy as np


class BestModelFinder:
    def __init__(self, model_name: str, max_patience: int, result_dir='./results') -> None:
        self.model_name = model_name
        self.result_dir = result_dir
        self.best_nll = np.inf
        self.max_patience = max_patience
        self.patience = 0
        self.model = None

    def __save(self, model, loss_val):
        print(f'Model {self.model_name} saved!')
        torch.save(model, self.model_name + '.model')
        self.best_nll = loss_val

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def proceed(self, epoch, model):
        # TODO: Check if this way is right
        if model is None:
            raise Exception("Model would be given befor proceed.")

        pass

    def isBreakable(self):
        return self.max_patience < self.patience
