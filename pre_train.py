import numpy as np
import torch
import torch.nn as nn
from pytorch_model_summary import summary

from train import LATENT_SIZE, NUM_CHANNELS, NUM_VALS, SIZE_OF_FEATURE_MAP, ensure_structure, get_data_loaders, get_optimizer, plot_curve, weights_init
from utils.train_supervisor import TrainSupervisor

MODEL_NAME = 'StackedEncoder'
MAX_PATIENCE = 20  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
RESULT_DIR = './pretrained/'
NUM_EPOCHS = 1000  # max. number of epochs

DEVICE = torch.device('cuda')


class StackedVAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(StackedVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(NUM_CHANNELS, SIZE_OF_FEATURE_MAP, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(SIZE_OF_FEATURE_MAP),
                                     nn.Conv2d(SIZE_OF_FEATURE_MAP, SIZE_OF_FEATURE_MAP *
                                               2, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(2 * SIZE_OF_FEATURE_MAP),
                                     nn.Conv2d(SIZE_OF_FEATURE_MAP * 2, SIZE_OF_FEATURE_MAP *
                                               4, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(SIZE_OF_FEATURE_MAP * 4),
                                     nn.Flatten(),
                                     nn.Linear(256 * 8 * 8, 256 * 2),
                                     nn.Linear(256 * 2, LATENT_SIZE),
                                     )

        self.decoder = nn.Sequential(nn.Linear(LATENT_SIZE, 256 * 2),
                                     nn.Linear(256 * 2, 256 * 8 * 8),
                                     nn.Unflatten(1, (256, 8, 8)),
                                     nn.ConvTranspose2d(SIZE_OF_FEATURE_MAP * 4, SIZE_OF_FEATURE_MAP * 2, kernel_size=4, stride=2,
                                                        padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(SIZE_OF_FEATURE_MAP * 2),
                                     nn.ConvTranspose2d(SIZE_OF_FEATURE_MAP * 2, SIZE_OF_FEATURE_MAP, kernel_size=4, stride=2,
                                                        padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(SIZE_OF_FEATURE_MAP),
                                     nn.ConvTranspose2d(
                                         SIZE_OF_FEATURE_MAP, NUM_CHANNELS, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     #  nn.Flatten(),
                                     #  nn.Unflatten(
                                     #      1, (NUM_CHANNELS, 64, 64, NUM_VALS)),
                                     # nn.Softmax(dim=4)
                                     )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat


def on_save(name, num):
    pass


mse_loss = nn.MSELoss().cuda()


class Trainer:
    # TODO: Too many dependencies. should be refactored!
    def __init__(self, supervisor: TrainSupervisor, num_epochs: int,
                 model: nn.Module, optimizer, training_loader, val_loader, data_transfer=None) -> None:
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.val_loader = val_loader
        self.data_transfer = data_transfer
        self.supervisor = supervisor

    def train(self):
        self.model.train()  # put our model in train mode
        for _, batch in enumerate(self.training_loader):
            if self.data_transfer:
                batch = self.data_transfer(batch)

            x_hat = self.model.forward(batch)
            loss = mse_loss(batch, x_hat)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def evaluate(self, epoch):
        self.model.eval()
        loss = 0.
        N = 0.

        for _, test_batch in enumerate(self.val_loader):
            if self.data_transfer:
                test_batch = self.data_transfer(test_batch)

            x_hat = self.model.forward(test_batch)
            loss_t = mse_loss(test_batch, x_hat)
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]

        loss = loss / N

        print(
            f'Epoch: {epoch if epoch is not None else "Final"}, val nll={loss}'
        )

        return loss

    def start_training(self):
        nll_val = []
        self.supervisor.set_model(self.model)

        for e in range(self.num_epochs):
            self.train()
            loss_val = self.evaluate(e)

            nll_val.append(loss_val)  # save for plotting
            self.supervisor.proceed(loss_val)

            if self.supervisor.is_breakable():
                break

        nll_val = np.asarray(nll_val)

        return nll_val


def data_transformer(data):
    data = data.float().to(DEVICE)

    return data


if __name__ == '__main__':
    ensure_structure()
    training_loader, val_loader, test_loader = get_data_loaders()

    model = StackedVAE().to(DEVICE)
    model.apply(weights_init)
    print("ENCODER:\n", summary(model, torch.zeros(1, 3, 64, 64,
                                                   device=DEVICE), show_input=False, show_hierarchical=True))

    optimizer = get_optimizer(model)

    supervisor = TrainSupervisor(
        MODEL_NAME, MAX_PATIENCE, RESULT_DIR, on_save=on_save)
    trainer = Trainer(supervisor, NUM_EPOCHS, model,
                      optimizer, training_loader, val_loader, data_transformer)

    nll_val = trainer.start_training()
    print(nll_val)
    plot_curve(RESULT_DIR + MODEL_NAME, nll_val)
