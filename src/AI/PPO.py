import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from typing import Dict, Any, Optional
from torch.distributions import Categorical


class Memory:
    """
    Per-agent memory identical in structure to TAAC.Memory for compatibility.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.action_state_values = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.action_state_values[:]


class MLPActorCriticNetwork(nn.Module):
    """
    PPO-style actor-critic with simple MLPs, but exposing the same forward signatures as TAAC.
    - actor_forward: accepts [B, N, state_dim] and returns [B, N, action_size]
    - critic_forward: accepts [B, N, state_dim] and action indices [B, N], returns [B, N]
    - actor_forward_update: returns (action_probs, similarity_loss=0.0) to match TAAC
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.temperature = 1.0
        self.similarity_loss_cap = 0.0  # kept for interface compatibility
    
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.action_size),
        )

        # Critic takes state and chosen action one-hot (like TAAC critic)
        critic_input_size = self.state_size + self.action_size
        self.critic_mlp = nn.Sequential(
            nn.Linear(critic_input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
        )
       
        print(f"PPO Network created with {sum(p.numel() for p in self.parameters())} parameters")
        print(f"State size: {state_size}, Action size: {action_size}, Action type: discrete")

    def actor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, state_dim]
        returns: [B, N, action_size] (action probabilities)
        """
        B, N, D = x.shape
        flat = x.reshape(B * N, D)
        logits = self.actor_mlp(flat)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs.view(B, N, self.action_size)

    def actor_forward_update(self, x: torch.Tensor):
        """
        Same as actor_forward, but returns a tuple (probs, similarity_loss) to match TAAC.
        For PPO (no attention), similarity_loss = 0.0.
        """
        probs = self.actor_forward(x)
        similarity_loss = torch.tensor(0.0, device=x.device)
        return probs, similarity_loss

    def critic_forward(self, x: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, state_dim]
        action_idx: [B, N]
        returns: [B, N] state-action values
        """
        B, N, D = x.shape
        action_one_hot = torch.zeros(B, N, self.action_size, device=x.device)
        action_one_hot.scatter_(-1, action_idx.unsqueeze(-1), 1)
        critic_input = torch.cat([x, action_one_hot], dim=-1)  # [B, N, D + A]
        critic_input = critic_input.reshape(B * N, -1)
        values = self.critic_mlp(critic_input).view(B, N)
        return values

    @torch.no_grad()
    def multi_agent_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a per-agent baseline V(s) as expectation over actions of Q(s,a)
        using the actor's current policy. Returns [B, N].
        """
        B, N, D = x.shape
        # Actor probabilities for all agents
        probs = self.actor_forward(x)  # [B, N, A]

        # Build inputs for all actions per agent
        x_expanded = x.unsqueeze(2).repeat(1, 1, self.action_size, 1)  # [B, N, A, D]
        all_action_one_hot = torch.eye(self.action_size, device=x.device).view(1, 1, self.action_size, self.action_size)
        all_action_one_hot = all_action_one_hot.repeat(B, N, 1, 1)  # [B, N, A, A]
        critic_input_all = torch.cat([x_expanded, all_action_one_hot], dim=-1)  # [B, N, A, D+A]
        critic_input_all = critic_input_all.view(B * N * self.action_size, -1)
        q_values_all = self.critic_mlp(critic_input_all).view(B, N, self.action_size)  # [B, N, A]

        baseline = (q_values_all * probs).sum(dim=-1)  # [B, N]
        return baseline


class PPO:
    """
    PPO agent matching TAAC’s public API so it can be swapped via config.
    """
    def __init__(self, env_config: Dict[str, Any], training_config: Optional[Dict[str, Any]] = None, mode: str = "train"):
        self.mode = mode
        self.memories = []

        # Env specifications
        self.state_size = int(env_config['state_size'])
        self.action_size = int(env_config['action_size'])
        self.number_of_agents = int(env_config['num_agents'])

        # Hyperparameters
        training_config = training_config or {}
        self.gamma = float(training_config.get('gamma', 0.99))
        self.epsilon_clip = float(training_config.get('epsilon_clip', 0.2))
        self.K_epochs = int(training_config.get('K_epochs', 10))
        self.learning_rate = float(training_config.get('learning_rate', 3e-4))
        self.c_entropy = float(training_config.get('c_entropy', 0.01))
        self.max_grad_norm = float(training_config.get('max_grad_norm', 0.5))
        self.c_value = float(training_config.get('c_value', 0.5))
        self.lam = float(training_config.get('lam', 0.95))
        self.base_batch_size = int(training_config.get('batch_size', 64))
        self.min_learning_rate = float(training_config.get('min_learning_rate', 1e-6))
        self.episodes = int(training_config.get('episodes', 1000))
        self.num_heads = int(training_config.get('num_heads', 1))  # unused but kept for parity
        self.embedding_dim = int(training_config.get('embedding_dim', 256))  # unused but parity
        self.hidden_size = int(training_config.get('hidden_size', 256))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.policy = self._build_model().to(self.device)
        self.policy_old = self._build_model().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer & LR schedule (match TAAC behavior)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Loss
        self.MseLoss = nn.MSELoss()

    def lr_lambda(self, epoch):
        initial_lr = float(self.learning_rate)
        final_lr = float(self.min_learning_rate)
        total_epochs = self.episodes
        lr = final_lr + (initial_lr - final_lr) * (1 - epoch / total_epochs)
        return max(lr / initial_lr, final_lr / initial_lr)

    def _build_model(self) -> nn.Module:
        network = MLPActorCriticNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
        )
        return network

    def select_action(self, state):
        """
        state: list of per-agent observations -> returns dicts matching TAAC.
        """
        try:
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_array).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # [1, N, D]
            action_probs = self.policy.actor_forward(state_tensor).squeeze(0)  # [N, A]
            dist = Categorical(action_probs)
            actions = dist.sample()  # [N]
            log_probs = dist.log_prob(actions)  # [N]
            entropies = dist.entropy()  # [N]

            return (
                {f"agent_{i}": actions[i].item() for i in range(actions.size(0))},
                {f"agent_{i}": log_probs[i].item() for i in range(log_probs.size(0))},
                {f"agent_{i}": entropies[i].item() for i in range(entropies.size(0))}
            )
        except Exception as e:
            print(f"An unexpected error occurred in select_action: {e}")
            raise e

    def get_actions(self, states):
        """
        Get actions for all agents and store experience (TAAC-compatible).
        """
        with torch.no_grad():
            actions, log_probs, entropies = self.select_action(states)
        self.store_experience(states, actions, log_probs)
        return actions, log_probs, entropies

    def memory_prep(self, number_of_agents):
        if self.mode != "train":
            return
        for memory in self.memories:
            memory.clear()
        self.memories = []
        for _ in range(number_of_agents):
            self.memories.append(Memory())

    def store_experience(self, states, actions, log_probs):
        if self.mode != "train":
            return
        current_num_agents = len(states)
        # Ensure memories are initialized
        if len(self.memories) != current_num_agents:
            self.memory_prep(current_num_agents)
        for i in range(current_num_agents):
            self.memories[i].states.append(states[i])
            self.memories[i].actions.append(actions[f"agent_{i}"])
            self.memories[i].log_probs.append(log_probs[f"agent_{i}"])

    def store_rewards(self, rewards, done):
        if self.mode != "train":
            return
        # Ensure memories are initialized
        if len(self.memories) < len(rewards):
            self.memory_prep(len(rewards))
        for i in range(len(rewards)):
            self.memories[i].rewards.append(rewards[i])
            self.memories[i].dones.append(done)

    def update(self):
        if self.mode != "train":
            return 0.0
        if len(self.memories) == 0:
            return 0.0

        final_similarity_loss = torch.tensor(0.0, device=self.device)

        actual_num_agents = len(self.memories)
        old_states = []
        old_actions = []
        old_log_probs = []
        rewards_list = []
        dones_list = []

        for j in range(actual_num_agents):
            if len(self.memories[j].states) > 0:
                old_states.append(self.memories[j].states)
                old_actions.append(self.memories[j].actions)
                old_log_probs.append(self.memories[j].log_probs)
                rewards_list.append(self.memories[j].rewards)
                dones_list.append(self.memories[j].dones)

        if len(old_states) == 0:
            return 0.0

        sequence_lengths = [len(s) for s in old_states]
        if len(set(sequence_lengths)) > 1:
            min_length = min(sequence_lengths)
            old_states = [s[:min_length] for s in old_states]
            old_actions = [a[:min_length] for a in old_actions]
            old_log_probs = [lp[:min_length] for lp in old_log_probs]
            rewards_list = [r[:min_length] for r in rewards_list]
            dones_list = [d[:min_length] for d in dones_list]

        if len(old_states[0]) == 0:
            return 0.0

        # [B, N, state_size]
        old_states = torch.from_numpy(np.array(old_states, dtype=np.float32)).permute(1, 0, 2).to(self.device)
        old_actions = torch.LongTensor(old_actions).permute(1, 0).to(self.device)  # [B, N]
        old_log_probs = torch.FloatTensor(old_log_probs).permute(1, 0).to(self.device)  # [B, N]
        rewards_tensor = torch.FloatTensor(rewards_list).permute(1, 0).to(self.device)  # [B, N]
        dones_tensor = torch.FloatTensor(dones_list).permute(1, 0).to(self.device)  # [B, N]

        baseline_values = self.policy_old.multi_agent_baseline(old_states)  # [B, N]
        gae_returns_tensor, advantages_tensor = self.compute_gae(rewards_tensor, dones_tensor, baseline_values, actual_num_agents)

        # Shuffle
        states = old_states
        actions = old_actions
        log_probs = old_log_probs
        advantages = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)
        gae_returns = gae_returns_tensor

        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size)
        states = states[indices]
        actions = actions[indices]
        log_probs = log_probs[indices]
        advantages = advantages[indices]
        gae_returns = gae_returns[indices]

        mini_batch_size = max(1, int(self.base_batch_size // actual_num_agents))
        num_mini_batches = max(1, dataset_size // mini_batch_size)

        for _ in range(self.K_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                mini_states = states[start:end]
                mini_actions = actions[start:end]
                mini_log_probs = log_probs[start:end]
                mini_advantages = advantages[start:end]
                mini_gae_returns = gae_returns[start:end]

                action_probs, similarity_loss = self.policy.actor_forward_update(mini_states)
                final_similarity_loss = similarity_loss  # zero for PPO
                state_values_new = self.policy.critic_forward(mini_states, mini_actions)  # [B, N]
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(mini_actions)
                dist_entropy = dist.entropy()

                ratios = torch.exp(action_log_probs - mini_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages

                # Calculate loss
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_new.squeeze(), mini_gae_returns) 
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * dist_entropy.mean()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()
        return float(final_similarity_loss.item())

    def compute_gae(self, rewards, dones, baseline_values, num_agents=None):
        if self.mode != "train":
            return
        if num_agents is None:
            num_agents = self.number_of_agents

        baseline_values = baseline_values.reshape(-1)
        dones = dones.reshape(-1)
        rewards = rewards.reshape(-1)

        baseline_values = baseline_values.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        rewards = rewards.detach().cpu().numpy()
        
        gamma = self.gamma
        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            next_value = 0 if i == len(rewards) - 1 else baseline_values[i + 1]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - baseline_values[i]
            gae = delta + gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + baseline_values

        returns = torch.FloatTensor(returns).reshape(-1, num_agents).to(self.device)
        advantages = torch.FloatTensor(advantages).reshape(-1, num_agents).to(self.device)
        return returns, advantages

    def save_model(self, model_path: str):
        print(f"--> Saving model to {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.policy_old.state_dict(), model_path)

    def load_model(self, model_path: str, test: bool = False) -> bool:
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            if test:
                self.policy.load_state_dict(state)
            else:
                self.policy.load_state_dict(state)
                self.policy_old.load_state_dict(state)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} does not exist.")
            return False
        
    def clone(self):
        return self.__class__(
            env_config={
                'state_size': self.state_size,
                'action_size': self.action_size,
                'num_agents': self.number_of_agents,
            },
            training_config={
                'gamma': self.gamma,
                'epsilon_clip': self.epsilon_clip,
                'K_epochs': self.K_epochs,
                'learning_rate': self.learning_rate,
                'c_entropy': self.c_entropy,
                'max_grad_norm': self.max_grad_norm,
                'c_value': self.c_value,
                'lam': self.lam,
                'batch_size': self.base_batch_size,
                'min_learning_rate': self.min_learning_rate,
                'episodes': self.episodes,
                'hidden_size': self.hidden_size,
            },
            mode=self.mode,
        )
    
    def assign_device(self, device):
        self.device = device
        self.policy.to(device)
        self.policy_old.to(device)
    
    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)
    
    def state_dict(self):
        return self.policy.state_dict()