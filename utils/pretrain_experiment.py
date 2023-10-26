from typing import Any, Dict
from torch import optim
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.utils as vutils

from model.types_ import *
from utils.gmm_experiment import GMMExperiment


class PreTrainExperiment(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 params: dict) -> None:
        super(PreTrainExperiment, self).__init__()
        self.model = model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              # al_img.shape[0]/ self.num_train_imgs,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item()
                      for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            # real_img.shape[0]/ self.num_val_imgs,
                                            M_N=1.0,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item()
                      for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images() 

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model.eval()

        gmm_experiment = GMMExperiment(
            num_centorids=self.model.num_centroids,
            model=self.model,
            data_loader=self.trainer.datamodule.train_dataloader(),
            device=self.curr_device
        )

        params = gmm_experiment.fit()

        checkpoint['u_p'] = torch.tensor(params[1], dtype=torch.float32)
        checkpoint['lambda_p'] = torch.tensor(params[2], dtype=torch.float32)
        
        self.model.train()
        
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model.u_p = checkpoint['u_p']
        self.model.lambda_p = checkpoint['u_p']

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(
            iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels=test_label)
        self.logger.log_image(
            key="Reconstructions",
            images=[vutils.make_grid(recons, nrow=12, normalize=True)],
        )

        try:
            if hasattr(self.model, 'sample'):
                samples = self.model.sample(144,
                                            self.curr_device,
                                            labels=test_label).cpu().data

                self.logger.log_image(
                    key="Samples",
                    images=[vutils.make_grid(samples, nrow=12, normalize=True)]
                )

        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
