from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.types_ import *


class GMMExperiment():
    def __init__(self, num_centorids: int, model: nn.Module, data_loader, device) -> None:
        super(GMMExperiment, self).__init__()
        
        self.model = model
        self.data_loader = data_loader
        self.gmm = GaussianMixture(
            n_components=num_centorids,
            covariance_type='diag'
        )
        self.device = device
    
    @torch.no_grad()
    def __init_data(self):
        z = torch.tensor([], device=self.device)
        for (data, _) in self.data_loader:
            batch_z = self.model.encode(data.to(self.device))
            z = torch.cat((z, batch_z), 0)
            
        self.latent_data = z.detach().cpu().numpy()
            

    def fit(self):
        print('GMM fit is called')
        self.__init_data()
        self.gmm.fit(self.latent_data)
        
        return (self.gmm.weights_, self.gmm.means_.T, self.gmm.covariances_.T)
