import torch
from torch import nn
from torch.nn import functional as F

from model.base import BaseVAE
from .types_ import *


class VaDE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 cluster_count: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VaDE, self).__init__()

        self.latent_dim = latent_dim
        self.num_centroids = cluster_count

        self.__init_gmm_params()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def __init_gmm_params(self):
        """
        Initiate GMM Params
        """
        self.theta_p = nn.Parameter(
            torch.ones(self.num_centroids, dtype='float32') /
            self.num_centroids
        )
        self.u_p = nn.Parameter(
            torch.rand(self.latent_dim, self.num_centroids)
        )
        self.lambda_p = nn.Parameter(
            # I should think on this 10.
            torch.rand(self.latent_dim, self.num_centroids) * 10
        )

    def __get_gamma(self, z):
        temp_Z = z.unsqueeze(2).repeat(1, 1, self.n_centroid)
        temp_u_tensor3 = self.u_p.unsqueeze(0).repeat(temp_Z.size(0), 1, 1)
        temp_lambda_tensor3 = self.lambda_p.unsqueeze(
            0).repeat(temp_Z.size(0), 1, 1)
        temp_theta_tensor3 = self.theta_p.unsqueeze(0).unsqueeze(
            0) * torch.ones((temp_Z.size(0), temp_Z.size(1), self.n_centroid), device=next(self.parameters()).device)

        p_c_z = torch.exp(torch.sum((torch.log(temp_theta_tensor3) - 0.5 * torch.log(2 * torch.pi * temp_lambda_tensor3) -
                                     torch.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)), dim=1)) + 1e-10

        gamma = p_c_z / torch.sum(p_c_z, dim=-1, keepdim=True)
        return gamma

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        gamma = self.__get_gamma(z)
        return [self.decode(z), input, mu, log_var, gamma]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        gamma = args[4]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 +
        #                       log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        kld_loss = torch.sum(0.5*gamma*(self.latent_dim*torch.log(torch.pi*2)+torch.log(
            self.lambda_p)+torch.exp(log_var)/self.lambda_p+torch.square(mu-self.u_p)/self.lambda_p), dim=(1, 2))\
            - 0.5*torch.sum(log_var + 1, dim=-1)\
            - torch.sum(torch.log(self.theta_p.unsqueeze(0).repeat(input.shape[0], 1)) * gamma, dim=-1)\
            + torch.sum(torch.log(gamma)*gamma, dim=-1)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.zeros((num_samples * self.num_centroids, self.latent_dim))

        for i in range(len(self.u_p)):
            z[i * self.num_centroids: (i+1) * self.num_centroids] = torch.normal(
                mean=self.u_p[i],
                std=self.theta_p[i],
                size=(num_samples, self.latent_dim)
            )

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
