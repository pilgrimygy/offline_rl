from typing import Union, Sequence
import gym
import torch
import torch.nn as nn
import numpy as np

from unstable_baselines.common.networks import BasePolicyNetwork, get_network, get_act_cls, MLPNetwork
from unstable_baselines.common import util


class VAE(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 max_action: float,
                 hidden_dims: Union[int, list],
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 log_std_min: int = -4, 
                 log_std_max: int = 15, 
                 **kwargs
    ) -> None:
        super(VAE, self).__init__()

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]

        self.encoder = MLPNetwork(obs_dim + action_dim, 2*latent_dim, hidden_dims, act_fn, out_act_fn)

        self.decoder = MLPNetwork(obs_dim + latent_dim, action_dim, hidden_dims, **kwargs)

        self.max_action = max_action
        self.latent_dim = latent_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        # encode
        z = self.encoder(torch.cat([obs, action], 1))
        mean = z[:, :self.latent_dim]
        log_std = z[:, self.latent_dim:].clamp(self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        # reparameterize
        z = mean + std * torch.randn_like(std)

        # decode
        u = self.decode(obs, z)

        return u, mean, std
    
    def decode(self, obs: torch.Tensor, z: torch.Tensor = None):
        if z is None:
            # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
            z = torch.randn((obs.shape[0], self.latent_dim)).to(util.device).clamp(-0.5, 0.5)
        
        action_pre_tanh = self.decoder(torch.cat([obs, z], 1))

        return self.max_action * torch.tanh(action_pre_tanh)
    
    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return super(VAE, self).to(device)


class PerturbationPolicyNetwork(BasePolicyNetwork):
    def __init__(self, 
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int],
                 phi: float = 0.05,
                 act_fn: str = "relu",
                 out_act_fn: str = "identity",
                 *args, **kwargs
        ):
        super(PerturbationPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn, *args, **kwargs)

        final_network = get_network([hidden_dims[-1], self.action_dim])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        self.phi = phi
        self.max_action = np.max(action_space.high)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        action_preturb = self.phi * self.max_action * self.networks(torch.cat([obs, action], 1))

        return (action + action_preturb).clamp(-self.max_action, self.max_action)
    
    def sample(self, obs: torch.Tensor):
        return self.forward(obs)

    def evaluate_actions(self, obs: torch.Tensor):
        return self.forward(obs)