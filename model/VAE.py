import torch
import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder
from model.prior import Prior


class VAE(nn.Module):
    def __init__(self, encoder_net, decoder_net, num_vals=256, L=16, likelihood_type='categorical'):
        super(VAE, self).__init__()

        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(distribution=likelihood_type,
                               decoder_net=decoder_net, num_vals=num_vals)
        self.prior = Prior(L=L)

        self.num_vals = num_vals

        self.likelihood_type = likelihood_type

    def forward(self, x, reduction='avg'):
        x = x.float()
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        outs = self.decoder.decoder(z)

        # ELBO
        # RE = self.decoder.log_prob(x, z)  # reconstruction
        RE = nn.functional.cross_entropy(
            x.view((x.shape[0], 3*64*64)), outs.view((outs.shape[0], 3*64*64)), reduction='sum')
        # RE = nn.functional.binary_cross_entropy(x, outs, reduction='sum')
        # RE = get_re(x, z)  # reconstruction\
        # RE = 0
        KL = -0.5 * torch.sum(1 + log_var_e - mu_e.pow(2) - log_var_e.exp())
        # KL = 0

        return RE + KL

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)

        return self.decoder.decoder(z)
