import os
import torch
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset

from model.vanilla_vae import VanillaVAE
from utils.experiment import VAEXperiment


LOGING_PARAM = {
    'save_dir': './results'
}

MODEL_PARAM = {
    'name': 'Vanilla VAE',
    'in_channels': 3,
    'latent_dim': 128
}

EXP_PARAM = {
    'LR': 0.005,
    'weight_decay': 0.0,
    'scheduler_gamma': 0.95,
    'kld_weight': 0.00025,
    'manual_seed': 1265,
}

TRAINER_PARAM = {
    'accelerator': 'gpu',
    'max_epochs': 100,
}

DATA_PARAM = {
    'data_path': './data',
    'train_batch_size': 64,
    'val_batch_size':  64,
    'patch_size': 64,
    'num_workers': 4,
    'pin_memory': False,
    'fast_dev_run': True,
}

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    tb_logger = TensorBoardLogger(save_dir=LOGING_PARAM['save_dir'],
                                  name=MODEL_PARAM['name'],)

    # For reproducibility
    # seed_everything(config['exp_params']['manual_seed'], True)

    model = VanillaVAE(
        in_channels=MODEL_PARAM['in_channels'],
        latent_dim=MODEL_PARAM['latent_dim']
    )
    experiment = VAEXperiment(model, EXP_PARAM)

    data = VAEDataset(**DATA_PARAM)

    # data.setup()

    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(
                                             tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     **TRAINER_PARAM)

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {MODEL_PARAM} =======")
    runner.fit(experiment, datamodule=data)
