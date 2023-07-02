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

        # ELBO
        RE = self.decoder.log_prob(x, z)  # reconstruction
        # RE = get_re(x, z)  # reconstruction
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e,
              log_var_e=log_var_e, z=z)).sum(-1)  # generalization

        if reduction == 'sum':
            return -(RE + KL).sum()

        # otherwise it's average.
        return -(RE + KL).mean()

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)

        return self.decoder.sample(z)
