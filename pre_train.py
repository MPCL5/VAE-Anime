import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from pytorch_model_summary import summary

from train import LATENT_SIZE, NUM_CHANNELS, SIZE_OF_FEATURE_MAP, ensure_structure, get_data_loaders, get_img_transform, get_optimizer, plot_curve, weights_init
from utils.train_supervisor import TrainSupervisor

BATCH_SIZE = 20
LR = 5e-4  # learning rate
MODEL_NAME = 'test'
MAX_PATIENCE = 20  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
RESULT_DIR = './pretrained/'
NUM_EPOCHS = 1000  # max. number of epochs
DATA_DIR = './data'

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
                 model: nn.Module, optimizer, training_loader) -> None:
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.training_loader = training_loader
        # self.val_loader = val_loader
        self.supervisor = supervisor

    def train(self):
        self.model.train()  # put our model in train mode

        overall_loss = 0.
        N = 0
        for _, batch in enumerate(self.training_loader):
            batch = batch[0].float().to(DEVICE)

            x_hat = self.model.forward(batch)
            loss = mse_loss(batch, x_hat)
            overall_loss += loss.item()
            N += batch.shape[0]

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        return overall_loss / N

    # @torch.no_grad()
    # def evaluate(self, epoch):
    #     self.model.eval()
    #     loss = 0.
    #     N = 0.

    #     for _, test_batch in enumerate(self.val_loader):

    #         x_hat = self.model.forward(test_batch)
    #         loss_t = mse_loss(test_batch, x_hat)
    #         loss = loss + loss_t.item()
    #         N = N + test_batch.shape[0]

    #     loss = loss / N

    #     print(
    #         f'Epoch: {epoch if epoch is not None else "Final"}, val nll={loss}'
    #     )

    #     return loss

    def start_training(self):
        nll_val = []
        self.supervisor.set_model(self.model)

        for e in range(self.num_epochs):
            loss_val = self.train()
            # loss_val = self.evaluate(e

            nll_val.append(loss_val)  # save for plotting
            self.supervisor.proceed(loss_val)

            if self.supervisor.is_breakable():
                break

        nll_val = np.asarray(nll_val)

        return nll_val


if __name__ == '__main__':
    ensure_structure()
    dataset = ImageFolder(root=DATA_DIR, transform=get_img_transform())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=2)

    model = StackedVAE().to(DEVICE)
    model.apply(weights_init)
    print(
        "Model:\n",
        summary(model, torch.zeros(1, 3, 64, 64, device=DEVICE),
                show_input=True, show_hierarchical=True)
    )

    optimizer = torch.optim.Adamax(
        [p for p in model.parameters() if p.requires_grad == True], lr=LR)

    supervisor = TrainSupervisor(
        MODEL_NAME, MAX_PATIENCE, RESULT_DIR, on_save=on_save)
    trainer = Trainer(supervisor, NUM_EPOCHS, model,
                      optimizer, dataloader)

    nll_val = trainer.start_training()
    print(nll_val)
    plot_curve(RESULT_DIR + MODEL_NAME, nll_val)
