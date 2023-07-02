import torch
import numpy as np


class TrainSupervisor:
    def __init__(self, model_name: str, max_patience: int, result_dir='./results') -> None:
        self.model_name = model_name
        self.result_dir = result_dir
        self.best_nll = np.inf
        self.max_patience = max_patience
        self.patience = 0
        self.model = None
        self.on_save = None

    def __save(self):
        print(f'Model {self.model_name} saved!')
        torch.save(self.model, self.result_dir + self.model_name + '.model')

        if self.on_save:
            self.on_save()

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def proceed(self, loss_val):
        # TODO: Check if this way is right
        if self.model is None:
            raise Exception("Model would be given befor proceeding.")

        if loss_val < self.best_nll:
            self.__save()
            self.best_nll = loss_val
            return

        self.patience += 1

    def is_breakable(self):
        return self.max_patience < self.patience
