"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    1. get_action (line 111)
    2. forward (line 126)
    3. update (line 141)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from aai4160.infrastructure import pytorch_util as ptu
from aai4160.policies.base_policy import BasePolicy
from aai4160.config import parse_args


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        
        args = parse_args()
        
        if args.criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.criterion == "L1":
            self.criterion = nn.L1Loss()
        elif args.criterion == "SmoothL1":
            self.criterion = nn.SmoothL1Loss()

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(

                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # HINT 1: DO NOT forget to change the type of observation.
        # Take a close look at `infrastructure/pytorch_util.py`.
        # HINT 2: We would use self.forward function to get the distribution,
        # And we will sample actions from the distribution.
        # HINT 3: Return a numpy action, not torch tensor
        dist = self.forward(ptu.from_numpy(observation))
        action = ptu.to_numpy(dist.sample())
        return action


    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action_dist: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. 
        # We are only considering continuous action cases.
        # So, we would like to return a normal distirbution from which we can sample actions.
        # HINT 1: Search up documentation `torch.distributions.Distribution` object
        # And design the function to return such a distribution object.
        # HINT 2: In self.get_action and self.update, we will sample from this distribution.
        # HINT 3: Think about how to convert logstd to regular std.
        if self.discrete:
            logits = self.logits_na(observation)
            action_dist = distributions.Categorical(logits)
        else:
            # 배치를 고려해야하나..?
            # torhc.diag(torch.exp(self.logstd))
            # batch_dim = batch_mean.shape[0]
            # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            # using scale_tril is more efficient, to represent the covariance
            # you can take std as a bias > 0 in network ,so we need to take exp(), you could choose not to, results are the same
            mean = self.mean_net(observation)
            std = torch.diag(torch.exp(self.logstd))
            action_dist = distributions.MultivariateNormal(mean, std)
        return action_dist
            

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        # HINT 1: DO NOT forget to call zero_grad to clear gradients from the previous update.
        # HINT 2: DO NOT forget to change the type of observations and actions, just like get_action.
        # HINT 3: Think about what function to call to sample an action
        # with which we will compute gradient to optimize.
        # https://stackoverflow.com/questions/60533150/what-is-the-difference-between-sample-and-rsample
        self.optimizer.zero_grad()
        pred_dist = self.forward(ptu.from_numpy(observations))
        pred_x = pred_dist.rsample()
        x = ptu.from_numpy(actions)

        loss = self.criterion(pred_x, x)
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

