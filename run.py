import os
import torch
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset

from model.vanilla_vae import VanillaVAE
from utils.experiment import VAEXperiment


LOGING_PARAM = {
    'log_dir': './logs',
    'save_dir': './results',
    'log_model': 'all',
    'project': 'VAE',
    'offline': True
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
    'pretrained_path': './results/checkpoints/last-v1.ckpt',
    'use_pretrained': True,
}

TRAINER_PARAM = {
    'accelerator': 'gpu',
    'max_epochs': 100,
}

DATA_PARAM = {
    'data_path': './data',
    'train_batch_size': 64,
    'val_batch_size':  64,
    'patch_size': (64, 64),
    'num_workers': 4,
    'pin_memory': False,
    'fast_dev_run': True,
}


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # tb_logger = TensorBoardLogger(save_dir=LOGING_PARAM['save_dir'],
    #                               name=MODEL_PARAM['name'],)
    wandb_logger = WandbLogger(
        name=MODEL_PARAM['name'],
        project=LOGING_PARAM['project'],
        save_dir=LOGING_PARAM['log_dir'],
        offline=LOGING_PARAM['offline']
    )

    # For reproducibility
    # seed_everything(config['exp_params']['manual_seed'], True)

    model = VanillaVAE(
        in_channels=MODEL_PARAM['in_channels'],
        latent_dim=MODEL_PARAM['latent_dim']
    )

    if EXP_PARAM['use_pretrained']:
        loaded = torch.load(EXP_PARAM['pretrained_path'])['state_dict']
        pretrained_state = model.state_dict()

        for key, value in loaded.items():
            if ('model.encoder' in key or 'model.decoder' in key) and 'encoder_output' not in key:
                pretrained_state[key.replace('model.', '')] = value

        model.load_state_dict(pretrained_state)

    # print("ENCODER:\n", summary(model, torch.zeros(1, 3, 64, 64),
    #   show_input=True, show_hierarchical=False))
    experiment = VAEXperiment(model, EXP_PARAM)

    data = VAEDataset(**DATA_PARAM)

    # data.setup()

    runner = Trainer(logger=wandb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(
                                             LOGING_PARAM['save_dir'], "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True,
                                         filename='{epoch}-{step}-' + MODEL_PARAM['name']),
                     ],
                     **TRAINER_PARAM)

    Path(
        f"{LOGING_PARAM['log_dir']}/Samples").mkdir(exist_ok=True, parents=True)
    Path(
        f"{LOGING_PARAM['log_dir']}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {MODEL_PARAM} =======")
    runner.fit(experiment, datamodule=data)
