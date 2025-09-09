import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from params import AIHyperparameters
import copy
import os

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 526),
            nn.LeakyReLU(),
            nn.Linear(526, 526),
            nn.LeakyReLU(),
            nn.Linear(526, 526),
            nn.LeakyReLU(),
            nn.Linear(526, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, 526),
            nn.LeakyReLU(),
            nn.Linear(526, 526),
            nn.LeakyReLU(),
            nn.Linear(526, 526),
            nn.LeakyReLU(),
            nn.Linear(526, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
       
        AI_PARAMS = AIHyperparameters()
        self.temperature = AI_PARAMS.temperature

        print(f"params: {sum(p.numel() for p in self.parameters())}")

    def actor_forward(self, x):
        action_probs = torch.softmax(self.actor(x)/ self.temperature, dim=-1)
        return action_probs

    def critic_forward(self, x, action_idx):
        action_one_hot = torch.zeros(x.shape[0], 18).to(x.device)
        action_one_hot.scatter_(-1, action_idx.unsqueeze(-1), 1)
        critic_input = torch.cat([x, action_one_hot], dim=-1)
        state_value = self.critic(critic_input)
        return state_value
    
    def multi_agent_baseline(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            possible_actions = torch.eye(18).to(x.device)
            possible_actions = possible_actions.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 18, 18]
            x_expanded = x.unsqueeze(1).expand(-1, 18, -1)  # [batch_size, 18, state_size]
            critic_input = torch.cat([x_expanded, possible_actions], dim=-1) # [batch_size, 18, state_size + action_size]
            state_action_value = self.critic(critic_input) # [batch_size, 18, 1]
            state_action_value = state_action_value.squeeze(-1) # [batch_size, 18]
            action_probs = torch.softmax(self.actor(x)/ self.temperature, dim=-1) # [batch_size, 18]
            state_value = (state_action_value * action_probs).sum(dim=-1) # [batch_size]
        return state_value


class PPOAgent:
    def __init__(self, mode="train"):
        AI_PARAMS = AIHyperparameters()

        self.mode = "train"
        self.memories = []

        self.state_size = AI_PARAMS.STATE_SIZE
        self.action_size = AI_PARAMS.ACTION_SIZE
        self.gamma = AI_PARAMS.gamma
        self.epsilon_clip = AI_PARAMS.epsilon_clip
        self.K_epochs = AI_PARAMS.K_epochs
        self.learning_rate = AI_PARAMS.learning_rate
        self.c_entropy = AI_PARAMS.c_entropy
        self.max_grad_norm = AI_PARAMS.max_grad_norm
        self.c_value = AI_PARAMS.c_value
        self.lam = AI_PARAMS.lam
        self.mini_batch_size = AI_PARAMS.batch_size
        self.min_learning_rate = AI_PARAMS.min_learning_rate
        self.episodes = AI_PARAMS.episodes
        self.td_dif_N = AI_PARAMS.TD_difference_N

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy network and old policy network
        self.policy = self._build_model().to(self.device)
        self.policy_old = self._build_model().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Loss function
        self.MseLoss = nn.MSELoss()

    def lr_lambda(self, epoch):
        initial_lr = self.learning_rate
        final_lr = self.min_learning_rate
        total_epochs = self.episodes
        lr = final_lr + (initial_lr - final_lr) * (1 - epoch / total_epochs)
        return max(lr / initial_lr, final_lr / initial_lr)


    def _build_model(self):
        return ActorCriticNetwork(self.state_size, self.action_size)

    def select_action(self, state):
        try :
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs = self.policy_old.actor_forward(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        except ValueError as e:
            print(e)
            print(f"state tensor: {state_tensor}")
            print(f"action_probs: {action_probs}")
            raise ValueError("Break")

        return action.item(), action_log_prob.item(), entropy.item()
    
    def _action_to_input(self, action):
        action_mapping = {
            0: [1, 0, 0, 0, 0],  # Up - drible
            1: [0, 1, 0, 0, 0],  # Down - drible
            2: [0, 0, 1, 0, 0],  # Left - drible
            3: [0, 0, 0, 1, 0],  # Right - drible
            4: [1, 0, 1, 0, 0],  # Up-Left - drible
            5: [1, 0, 0, 1, 0],  # Up-Right - drible
            6: [0, 1, 1, 0, 0],  # Down-Left - drible
            7: [0, 1, 0, 1, 0],  # Down-Right - drible
            8: [0, 0, 0, 0, 0],  # No movement - drible
            9: [1, 0, 0, 0, 1],  # Up - shoot
            10: [0, 1, 0, 0, 1],  # Down - shoot
            11: [0, 0, 1, 0, 1],  # Left - shoot
            12: [0, 0, 0, 1, 1],  # Right - shoot
            13: [1, 0, 1, 0, 1],  # Up-Left - shoot
            14: [1, 0, 0, 1, 1],  # Up-Right - shoot
            15: [0, 1, 1, 0, 1],  # Down-Left - shoot
            16: [0, 1, 0, 1, 1],  # Down-Right - shoot
            17: [0, 0, 0, 0, 1]  # No movement - shoot
        }
        return action_mapping.get(action, [0, 0, 0, 0])

    def update(self):
        if self.mode != "train":
            return

        # Combine experiences from all memories
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        gae_returns = []
        advantages = []

        for memory in self.memories:
            # Convert lists to tensors
            old_states = torch.FloatTensor(memory.states).to(self.device)
            old_actions = torch.LongTensor(memory.actions).to(self.device)
            old_log_probs = torch.FloatTensor(memory.log_probs).to(self.device)

            # Compute returns and advantages for this memory
            rewards = memory.rewards
            dones = memory.dones
            multi_agent_baseline = self.policy_old.multi_agent_baseline(old_states) 
            gae_returns_tensor, advantages_tensor = self.compute_gae(rewards, dones, multi_agent_baseline)

            # Append to the combined lists
            states.append(old_states)
            actions.append(old_actions)
            log_probs.append(old_log_probs)
            advantages.append(advantages_tensor)
            gae_returns.append(gae_returns_tensor)

            # Clear memory after processing
            memory.clear()

        # Concatenate experiences from all agents
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        advantages = torch.cat(advantages, dim=0)
        gae_returns = torch.cat(gae_returns, dim=0)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Shuffle the data
        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size)
        states = states[indices]
        actions = actions[indices]
        log_probs = log_probs[indices]
        advantages = advantages[indices]
        gae_returns = gae_returns[indices]

        # Define mini-batch size
        mini_batch_size = self.mini_batch_size  # e.g., 64
        num_mini_batches = dataset_size // mini_batch_size

        # PPO policy update with mini-batching
        for _ in range(self.K_epochs):
            for i in range(num_mini_batches):
                # Define the start and end of the mini-batch
                start = i * mini_batch_size
                end = start + mini_batch_size

                # Slice the mini-batch
                mini_states = states[start:end]
                mini_actions = actions[start:end]
                mini_log_probs = log_probs[start:end]
                mini_advantages = advantages[start:end]
                mini_gae_returns = gae_returns[start:end]

                # Forward pass
                action_probs = self.policy.actor_forward(mini_states)
                state_values_new = self.policy.critic_forward(mini_states, mini_actions)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(mini_actions)
                dist_entropy = dist.entropy()

                # Calculate the ratios
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

            # Handle any remaining data not fitting into mini-batches
            remainder = dataset_size % mini_batch_size
            if remainder != 0:
                start = num_mini_batches * mini_batch_size
                mini_states = states[start:]
                mini_actions = actions[start:]
                mini_log_probs = log_probs[start:]
                mini_advantages = advantages[start:]
                mini_gae_returns = gae_returns[start:]

                # Forward pass
                action_probs = self.policy.actor_forward(mini_states)
                state_values_new = self.policy.critic_forward(mini_states, mini_actions)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(mini_actions)
                dist_entropy = dist.entropy()

                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_new.squeeze(), mini_gae_returns)
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * dist_entropy.mean()
                # Calculate loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
        
        # Update old policy parameters with new policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()
        return 0


    
    def temporal_difference(self, rewards, dones, action_state_values):
        """
        Computes N-step returns (bootstrapped) for each time step, 
        then returns a list/array that you can compare with predicted Q-values.
        """
        with torch.no_grad():
            N = self.td_dif_N
            gamma = self.gamma
            T = len(rewards)
    
            # Convert predicted Q-values to CPU if needed, for easy manipulation
            # We'll convert back to CUDA after computation if necessary
            action_state_values = action_state_values.detach().cpu().numpy()
    
            # Prepare an array for the N-step targets
            bootstrapped_return = np.zeros_like(action_state_values, dtype=np.float32)
    
            for t in range(T):
                G = 0.0
                discount = 1.0
    
                # Accumulate discounted rewards for up to N steps or until 'done'
                episode_ended = False
                for i in range(N):
                    idx = t + i
                    if idx < T:
                        G += discount * rewards[idx]
                        discount *= gamma
                        if dones[idx]:
                            episode_ended = True
                            break
                    else:
                        break
    
                # add the estimated state value at time step t+N if it's not the end of the episode
                final_idx = t + N
                if not episode_ended and final_idx < T and not dones[final_idx - 1]:
                    G += discount * action_state_values[final_idx]
    
                bootstrapped_return[t] = G
    
            # Convert the bootstrapped_return back to a torch.Tensor on the same device
            bootstrapped_return = torch.tensor(bootstrapped_return, dtype=torch.float32, device=self.device)
    
        return bootstrapped_return
    
    def compute_gae(self, rewards, dones, baseline_values):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        :param rewards: List of rewards for an episode.
        :param dones: List of done flags for an episode.
        :param baseline_values: Tensor of state baseline_values.
        :return: Tensors of returns.
        """
        if self.mode != "train":
            return
        
        gamma = self.gamma
        advantages = []
        gae = 0
        baseline_values = baseline_values.detach().cpu().numpy()
        #state_action_values = state_action_values.detach().cpu().numpy()
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else baseline_values[i + 1]
            else:
                next_value = baseline_values[i + 1]
            #delta = rewards[i] + gamma * next_value * (1 - dones[i]) - state_action_values[i]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - baseline_values[i]
            gae = delta + gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        #returns = advantages + state_action_values
        returns = advantages + baseline_values
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)

    
    def save_model(self, model_name="PPO_model_giant"):
        path = f"files/Models/{model_name}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model_name="PPO_model_big", test=False):
        path = f"files/Models/{model_name}.pth"
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            if test:
                self.policy.eval()
                self.policy_old.eval()
            else:
                self.policy.train()
                self.policy_old.train()
            print(f"Model loaded from {path}")
        else:
            print(f"Model file {path} does not exist.")
    
    def memory_prep(self, number_of_agents):
        if self.mode != "train":
            return
        
        for memory in self.memories:
            memory.clear()

        self.memories = []
        for _ in range(number_of_agents):
            self.memories.append(Memory())

    
    def get_actions(self, states):
        actions = []
        entropies = []
        for i, state in enumerate(states):
            action, log_prob, entropy = self.select_action(state)
            if self.mode == "train":
                self.memories[i].states.append(state)
                self.memories[i].actions.append(action)
                self.memories[i].log_probs.append(log_prob)

            actions.append(self._action_to_input(action))
            entropies.append(entropy)
        
        return actions, entropies
    
    def store_rewards(self, rewards, done):
        if self.mode != "train":
            return

        for i in range(len(rewards)):
            self.memories[i].rewards.append(rewards[i])
            self.memories[i].dones.append(done)
        
    def clone(self):
        return copy.deepcopy(self)
    
    def assign_device(self, device):
        self.device = device
        self.policy.to(device)
        self.policy_old.to(device)
    
    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)
    
    def state_dict(self):
        return self.policy.state_dict()