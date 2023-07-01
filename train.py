import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

from data.anime_dataset import AnimeDataset


IMG_SIZE = 64  # input dimension
BATCH_SIZE = 32
L = 16  # number of latents
M = 256  # the number of neurons in scale (s) and translation (t) nets

LR = 1e-3  # learning rate
NUM_EPOCHS = 1000  # max. number of epochs
MAX_PATIENCE = 20  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
SIZE_OF_FEATURE_MAP = 64
NUM_CHANNELS = 3

RESULT_DIR = './results/'
DATA_DIR = './data/images/'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return (training_loader, val_loader, test_loader)

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for _, batch in enumerate(training_loader):
            batch = batch.float()
            batch = batch.to(DEVICE)
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + torch.rand(batch.shape)
                    
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val

if __name__ == '__main__':
    ensure_structure()
    training_loader, val_loader, test_loader = get_data_loaders()
