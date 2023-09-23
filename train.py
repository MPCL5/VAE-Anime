import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pytorch_model_summary import summary

from data.anime_dataset import AnimeDataset
from model.VAE import VAE
from utils.train_supervisor import TrainSupervisor
from utils.trainer import Trainer

MODEL_NAME = 'VAE'
IMG_SIZE = 64  # input dimension
BATCH_SIZE = 20
L = 16  # number of latents
M = 256  # the number of neurons in scale (s) and translation (t) nets

LR = 1e-3  # learning rate
NUM_EPOCHS = 1000  # max. number of epochs
MAX_PATIENCE = 20  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
SIZE_OF_FEATURE_MAP = 64
NUM_CHANNELS = 3
LIKELIHOOD_TYPE = 'categorical'
NUM_VALS = 256

RESULT_DIR = './results/'
DATA_DIR = './data/images/'

GENERATED_NUM_X = 4 # number of images in the X axis of generated result
GENERATED_NUM_Y = 4 # number of images in the Y axis of generated result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')

def ensure_structure():
    if (os.path.exists(RESULT_DIR)):
        return

    os.mkdir(RESULT_DIR)


def get_data_loaders():
    img_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip()
    ])

    train_data = AnimeDataset(mode='train', data_dir=[
                              DATA_DIR], transformer=img_transform)
    val_data = AnimeDataset(mode='val', data_dir=[
                            DATA_DIR], transformer=img_transform)
    test_data = AnimeDataset(mode='test', data_dir=[
                             DATA_DIR], transformer=img_transform)

    training_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return (training_loader, val_loader, test_loader)


def data_transformer(data):
    data = data.float()
    data = data.to(DEVICE)

    return data


# TODO: extract the codes related to model itself.
def get_model():
    encoder = nn.Sequential(nn.Conv2d(NUM_CHANNELS, SIZE_OF_FEATURE_MAP, kernel_size=4, stride=2, padding=1),
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
                            nn.Linear(256 * 2, L * 2),
                            ).to(DEVICE)

    decoder = nn.Sequential(nn.Linear(L, 256 * 2),
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
                                SIZE_OF_FEATURE_MAP, NUM_CHANNELS * NUM_VALS, kernel_size=4, stride=2, padding=1),
                            nn.ReLU(),
                            nn.Flatten(),
                            # nn.Unflatten(1, (3, 64, 64, num_vals)),
                            # nn.Softmax(dim=4)
                            ).to(DEVICE)

    model = VAE(encoder_net=encoder, decoder_net=decoder,
                num_vals=NUM_VALS, L=L, likelihood_type=LIKELIHOOD_TYPE).to(DEVICE)

    print("ENCODER:\n", summary(encoder, torch.zeros(1, 3, 64, 64,
                                                     device=DEVICE), show_input=False, show_hierarchical=False))

    print("\nDECODER:\n", summary(decoder, torch.zeros(
        1, L, device=DEVICE), show_input=False, show_hierarchical=False))

    return model


def get_optimizer():
    return torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=LR)


def samples_generated(model: nn.Module, extra_name=''):
    model.eval()

    x = model.sample(GENERATED_NUM_X * GENERATED_NUM_Y)

    _, ax = plt.subplots(GENERATED_NUM_X, GENERATED_NUM_Y)
    for i, ax in enumerate(ax.flatten()):
        ax.imshow(x[i].permute(1, 2, 0).cpu().detach().numpy())
        ax.axis('off')

    plt.savefig(RESULT_DIR + MODEL_NAME + '_generated_images' +
                extra_name + '.pdf', bbox_inches='tight')
    plt.close()


# TODO: need to be refactored
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# TODO: need some revisions.
def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    ensure_structure()
    training_loader, val_loader, test_loader = get_data_loaders()

    model = get_model()
    model.apply(weights_init)
    optimizer = get_optimizer()

    def on_save(name, num):
        return samples_generated(name, f'best_{num}')

    supervisor = TrainSupervisor(
        MODEL_NAME, MAX_PATIENCE, RESULT_DIR, on_save=on_save)
    trainer = Trainer(supervisor, NUM_EPOCHS, model,
                      optimizer, training_loader, val_loader, data_transformer)

    nll_val = trainer.start_training()
    print(nll_val)
    plot_curve(RESULT_DIR + MODEL_NAME, nll_val)
