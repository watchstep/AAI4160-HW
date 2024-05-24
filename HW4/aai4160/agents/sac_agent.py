from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
import numpy as np

import aai4160.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
            "none",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(student): Implement the different backup strategies.
        if self.target_critic_backup_type == "doubleq":
            # swap next_qs
            next_qs = next_qs[[1, 0]]
        elif self.target_critic_backup_type == "min":
            next_qs = next_qs.min(dim=0).values()
        else:
            # Default, we don't need to do anything.
            pass

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # NOTE: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student): Sample from the actor
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample()

            # TODO(student)
            # Compute the next Q-values using `self.critic` for the sampled actions
            next_qs = self.critic(obs=next_obs, action=next_action)

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (for double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                # NOTE: use `self.entropy()`
                print('호', next_action)
                next_action_entropy = self.entropy(next_action)
                next_qs += next_action_entropy

            # TODO(student): Compute the target Q-value
            # HINT: implement Equation (1) in Homework 4
            target_values: torch.Tensor = reward + self.discount * (1 - done.float()) * next_qs
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # TODO(student): Predict Q-values using `self.critic`
        q_values = self.critic(obs)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # TODO(student): Compute loss using `self.critic_loss`
        loss: torch.Tensor =  self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(student): Compute the entropy of the action distribution
        # HINT: use one action sample for estimating the entropy
        # HINT: different to Homework 2, we use tanh_transformed distribution
        # in SAC, so log_prob does not need sum over the last dimension
        # NOTE: think about whether to use .rsample() or .sample() here
        print("확인")
        action = action_distribution.sample()
        entropy = -action_distribution.log_prob(action)

        assert entropy.shape == action.shape[:-1]
        return entropy

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # TODO(student): Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            # TODO(student): Draw num_actor_samples samples from the action distribution for each batch element
            action = [action_distribution.sample() for _ in range(self.num_actor_samples)]
            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), action.shape

            # TODO(student): Compute Q-values for the current state-action pair
            # HINT: need to add one dimension with `self.num_actor_samples` at the beginning of `obs`
            # HINT: for this, you can use either `repeat` or `expand`
            q_values = self.critic(obs.unsequeeze().repeat(self.num_actor_samples, 1, 1), action)
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)

        # Do REINFORCE (without baseline)
        # TODO(student): Calculate log-probs
        log_probs = action_distribution.log_prob(action)
        assert log_probs.shape == q_values.shape

        # TODO(student): Compute policy gradient using log-probs and Q-values
        loss = self.critic_loss(log_probs, q_values)

        return loss, torch.mean(self.entropy(action_distribution))
    
    # 헷갈
    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(student): Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        action = action_distribution.rsample()

        # TODO(student): Compute Q-values for the sampled state-action pair
        q_values = self.critic(obs, action)

        # TODO(student): Compute the actor loss using Q-values
        loss = -torch.mean(action_distribution.log_prob(action)*q_values)

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)
        else:
            loss, entropy = 0., self.entropy(self.actor(obs)).mean()

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        critic_infos = []
        for _ in range(self.num_critic_updates):
            critic_infos.append(self.update_critic(observations, actions, rewards, next_observations, dones))

        # TODO(student): Update the actor
        actor_info = self.update_actor(observations)
        print(actor_info)

        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        # For hard target updates, you should do it every self.target_update_period step
        # For soft target updates, you should do it every step
        # HINT: use `self.update_target_critic` or `self.soft_update_target_critic`
        if self.target_update_period is None:
            if not step % self.target_update_period:
                self.update_target_critic()
        else:
            self.soft_update_target_critic()


        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
