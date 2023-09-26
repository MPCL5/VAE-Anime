import torch
import torch.nn as nn

from utils.probability_distributions import log_bernoulli, log_categorical

SUPPORTED_DISTROS = ['categorical', 'bernoulli']


# def get_re(x, p):
#     p = torch.argmax(p, dim=-1)
#     x = x.view(x.size(0), -1)
#     p = p.view(p.size(0), -1)
#     reconstruction_loss = nn.functional.mse_loss(p, x, reduction='none')
#     reconstruction_loss = torch.mean(reconstruction_loss, dim=-1)

#     return reconstruction_loss


class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_vals=None):
        super(Decoder, self).__init__()

        if distribution not in SUPPORTED_DISTROS:
            raise ValueError('Either `categorical` or `bernoulli`')

        self.decoder = decoder_net
        self.num_vals = num_vals
        self.distribution = distribution

    def decode(self, z):
        h_d = self.decoder(z)
        mu_d = None

        match self.distribution:
            case 'categorical':
                # TODO: should moved to another method
                b = h_d.shape[0]
                h_d = h_d.view(b, 3, 64, 64, self.num_vals)
                mu_d = torch.softmax(h_d, 3)
                # mu_d = torch.softmax(h_d, dim=4)
                # return [h_d]

            case 'bernoulli':
                # TODO: should moved to another method
                mu_d = torch.sigmoid(h_d)

        return [mu_d]

    def sample(self, z):
        outs = self.decode(z)
        x_new = None

        match self.distribution:
            case 'categorical':
                # TODO: should moved to another method
                mu_d = outs[0]
                x_new = torch.argmax(outs[0], dim=-1)

            case 'bernoulli':
                # TODO: should moved to another method
                mu_d = outs[0]
                x_new = torch.bernoulli(mu_d)

        return x_new

    def log_prob(self, x, z):
        outs = self.decode(z)

        match self.distribution:
            case 'categorical':
                # # TODO: should moved to another method
                # mu_d = outs[0]
                # log_p = get_re(x, mu_d)
                mu_d = outs[0]
                log_p = log_categorical(
                    x, mu_d, reduction='sum', dim=-1).sum(-1).sum(-1).sum(-1)

            case 'bernoulli':
                # TODO: should moved to another method
                mu_d = outs[0]
                log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder',
                        'log_prob'], 'Type could be either decode or log_prob'

        if type == 'log_prob':
            return self.log_prob(x, z)

        return self.sample(z)
