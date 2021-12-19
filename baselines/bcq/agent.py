import torch
import torch.nn.functional as F
import gym 
import os
from operator import itemgetter
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import MLPNetwork, get_optimizer
from unstable_baselines.offline_rl.common.networks import VAE, PerturbationPolicyNetwork
from unstable_baselines.common.buffer import ReplayBuffer
import numpy as np
from unstable_baselines.common import util


class BCQAgent(torch.nn.Module, BaseAgent):
    def __init__(self, observation_space, action_space,
                 update_target_network_interval=50, 
                 target_smoothing_tau=0.1,
                 **kwargs
        ):
        super(BCQAgent, self).__init__()

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        latent_dim = action_dim * 2
        max_action = np.max(action_space.high)

        #save parameters
        self.args = kwargs

        #initilze networks
        self.q1_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q1_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q2_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.policy_network = PerturbationPolicyNetwork(obs_dim + action_dim, action_space, **kwargs["policy_network"])
        self.target_policy_network = PerturbationPolicyNetwork(obs_dim + action_dim, action_space, **kwargs["policy_network"])
        self.vae = VAE(obs_dim, action_dim, latent_dim, max_action, **kwargs['vae'])

        #sync network parameters
        util.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        util.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.target_policy_network = self.target_policy_network.to(util.device)
        self.vae = self.vae.to(util.device)

        #register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            'target_q1_network': self.target_q1_network,
            'target_q2_network': self.target_q2_network,
            'policy_network': self.policy_network,
            'vae': self.vae
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        self.vae_optimizer = get_optimizer(kwargs['vae']['optimizer_class'], self.vae, kwargs['vae']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.lamda = kwargs['lambda']
        self.repeat_num = kwargs['repeat_num']

        self.tot_update_count = 0 
        self.update_target_network_interval = update_target_network_interval
        self.target_smoothing_tau = target_smoothing_tau
    
    def update(self, data_batch):
        obs_batch = data_batch['obs']
        action_batch = data_batch['action'] 
        next_obs_batch = data_batch['next_obs'] 
        reward_batch = data_batch['reward']
        done_batch = data_batch['done']

        # VAE training
        recon, mean, std = self.vae(obs_batch, action_batch)
        recon_loss = F.mse_loss(recon, action_batch)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Critic training
        next_obs = torch.repeat_interleave(next_obs_batch, self.repeat_num, 0)

        target_next_action = self.target_policy_network(next_obs, self.vae.decode(next_obs))

        target_q1 = self.target_q1_network(torch.cat([next_obs, target_next_action], dim=1))
        target_q2 = self.target_q2_network(torch.cat([next_obs, target_next_action], dim=1))

        target_q = self.lamda * torch.min(target_q1, target_q2) + (1 - self.lamda) * torch.max(target_q1, target_q2)

        target_q = target_q.reshape(-1, self.repeat_num).max(1)[0].reshape(-1, 1)

        target_q = reward_batch + self.gamma * (1. - done_batch) * target_q
        
        curr_q1 = self.q1_network(torch.cat([obs_batch, action_batch], dim=1))
        curr_q2 = self.q2_network(torch.cat([obs_batch, action_batch], dim=1))

        q1_loss = F.mse_loss(curr_q1, target_q.detach())
        q2_loss = F.mse_loss(curr_q2, target_q.detach())

        q1_loss_value = q1_loss.detach().cpu().numpy()
        q2_loss_value = q2_loss.detach().cpu().numpy()

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Actor Training
        sampled_action = self.vae.decode(obs_batch)
        perturbed_action = self.policy_network(obs_batch, sampled_action)

        policy_loss = -self.q1_network(torch.cat([obs_batch, perturbed_action], dim=1)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        policy_loss_value = policy_loss.detach().cpu().numpy()

        self.tot_update_count += 1

        self.try_update_target_network()

        return {
            "loss/q1": q1_loss_value, 
            "loss/q2": q2_loss_value, 
            "loss/policy": policy_loss_value, 
            "misc/current_state_q1_value": torch.norm(curr_q1.squeeze().detach().clone().cpu(), p=1) / len(curr_q1.squeeze()), 
            "misc/current_state_q2_value": torch.norm(curr_q2.squeeze().detach().clone().cpu(), p=1) / len(curr_q2.squeeze()),
            "misc/q_diff": torch.norm((curr_q2-curr_q1).squeeze().detach().clone().cpu(), p=1) / len(curr_q2.squeeze())
        }

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            util.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
            util.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)
            util.soft_update_network(self.policy_network, self.target_policy_network, self.target_smoothing_tau)
    
    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).repeat(self.repeat_num, 1).to(util.device)
            action = self.policy_network(obs, self.vae.decode(obs))
            q1 = self.q1_network(torch.cat([obs, action], dim=1))
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()
    
    def get_gradient(self):
        ret = {}
        for name in ["q1_network", "q2_network", "policy_network"]:
            network = self.networks[name]
            grads = []
            for param in network.parameters():
                if param.requires_grad:
                    grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            ret[name] = torch.norm(grads.detach().clone().cpu(), p=1)
        return ret