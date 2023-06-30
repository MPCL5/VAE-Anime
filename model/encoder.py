import torch
import torch.nn as nn

from utils.probability_distributions import log_normal_diag


class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()

        self.encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

    def encode(self, x):
        h_e = self.encoder(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

    # TODO: decrease dependency injection between this methods and its relatives
    def sample(self, mu_e=None, log_var_e=None):
        if (mu_e is None) or (log_var_e is None):
            raise ValueError('mu and log-var can`t be None!')

        z = self.reparameterization(mu_e, log_var_e)
        return z

    def sample_with_x(self, x):
        mu_e, log_var_e = self.encode(x)

        return self.sample(mu_e, log_var_e)

    # TODO: decrease dependency injection between this methods and its relatives
    def log_prob(self, mu_e=None, log_var_e=None, z=None):
        if (mu_e is None) or (log_var_e is None) or (z is None):
            raise ValueError('mu, log-var and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

    def log_prob_with_x(self, x=None):
        mu_e, log_var_e = self.encode(x)
        z = self.sample(mu_e=mu_e, log_var_e=log_var_e)

        return self.log_prob(mu_e, log_var_e, z)

    def forward(self, x, type='log_prob'):
        assert type in [
            'encode', 'log_prob'], 'Type could be either encode or log_prob'

        if type == 'log_prob':
            return self.log_prob(x)

        return self.sample(x)
