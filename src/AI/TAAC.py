import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import math
import copy
import os
from typing import Dict, Any, Optional, Union
from torch.distributions import Categorical
from tqdm import tqdm


class Memory:
    """
    This is a per agent class that stores the experience of the agent.
    It can be conbined with other episodes agents to combine the experience of all agents for parallel training.
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


class AttentionActorCriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, num_heads=4, embedding_dim=256, hidden_size=526):
        super(AttentionActorCriticNetwork, self).__init__()

        # Make hyperparameters configurable instead of importing from external file
        self.temperature = 1.0  # Default value, can be overridden
        self.similarity_loss_cap = -0.5  # Default value, can be overridden
        self.action_size = action_size
        self.action_space_type = "discrete"  # Only supporting discrete actions
        self.emb_dim = embedding_dim

        self.hidden_size = hidden_size

        self.actor_embedding = nn.Sequential(
            nn.Linear(state_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, embedding_dim),
        )

        # Multi-head self-attention block
        # We treat each agent as a 'token' in the sequence dimension
        self.actor_attention_block = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True  # so input can be [batch_size, seq_len, embed_dim]
        )

        # Actor output for discrete action space
        self.actor_out = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, action_size)
        )

        # Critic network
        critic_input_size = state_size + action_size  # For discrete actions

        self.critic_embedding = nn.Sequential(
            nn.Linear(critic_input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, embedding_dim),
        )

        # Multi-head self-attention block
        # We treat each agent as a 'token' in the sequence dimension
        self.critic_attention_block = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True  # so input can be [batch_size, seq_len, embed_dim]
        )

        # critic head: transforms each post-attention embedding -> value
        self.critic_out = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        print(f"Network created with {sum(p.numel() for p in self.parameters())} parameters")
        print(f"State size: {state_size}, Action size: {action_size}, Action type: discrete")

    def actor_forward(self, x):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          action_probs [batch_size, num_agents, action_size]
        """
        B, N, D = x.shape
        actor_input = x.reshape(B*N, -1) # [B*N, state_size]
        actor_input = self.actor_embedding(actor_input) # [B*N, embedding_dim]
        actor_input = actor_input.reshape(B, N, -1) # [B, N, embedding_dim]
        attn_output, _ = self.actor_attention_block(actor_input, actor_input, actor_input) # [B, N, embedding_dim]

        #concatenate the attention output with the actor input
        attn_output = torch.cat([attn_output, actor_input], dim=-1) # [B, N, 2*embedding_dim]

        action_logits = self.actor_out(attn_output) # [B, N, action_size]
        action_probs = torch.softmax(action_logits / self.temperature, dim=-1)
        return action_probs

    def actor_forward_update(self, x):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        
        Additionally calculate similarity loss for training
        Returns:
          action_probs [batch_size, num_agents, action_size]
        """
        B, N, D = x.shape
        actor_input = x.reshape(B*N, -1) # [B*N, state_size]
        actor_input = self.actor_embedding(actor_input) # [B*N, embedding_dim]
        actor_input = actor_input.reshape(B, N, -1) # [B, N, embedding_dim]
        attn_output, _ = self.actor_attention_block(actor_input, actor_input, actor_input) # [B, N, embedding_dim]

        #concatenate the attention output with the actor input
        attn_output = torch.cat([attn_output, actor_input], dim=-1) # [B, N, 2*embedding_dim]

        # similarity loss 
        normalized_attn_output = attn_output / attn_output.norm(dim=-1, keepdim=True) # [B, N, embedding]
        similarity_matrix = torch.matmul(normalized_attn_output, normalized_attn_output.transpose(-2, -1)) # [B, N, N]
        mask = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1).bool() # [B, N, N]
        similarity_matrix = similarity_matrix.masked_fill(mask, 0) # [B, N, N]
        similarity_loss = torch.sum(similarity_matrix, dim=(1, 2)) / (N * (N - 1))  # [1]

        # remove negative values
        similarity_loss = torch.clamp(similarity_loss, min=self.similarity_loss_cap)
        similarity_loss = similarity_loss.mean()

        action_logits = self.actor_out(attn_output)
        action_probs = torch.softmax(action_logits / self.temperature, dim=-1)
        return action_probs, similarity_loss
    
    def critic_forward(self, x, action_idx):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          action_probs: [batch_size, num_agents, action_size]
        """
        B, N, D = x.shape
        action_one_hot = torch.zeros(B, N, self.action_size).to(x.device) # [B, N, action_size]
        action_one_hot.scatter_(-1, action_idx.unsqueeze(-1), 1) # [B, N, action_size]
        critic_input = torch.cat([x, action_one_hot], dim=-1) # [B, N, state_size + action_size]
        critic_input = critic_input.reshape(B*N, -1) # [B*N, state_size + action_size]
        critic_input = self.critic_embedding(critic_input) # [B*N, embedding_dim]
        critic_input = critic_input.reshape(B, N, -1) # [B, N, embedding_dim]
        attn_output, _ = self.critic_attention_block(critic_input, critic_input, critic_input) # [B, N, embedding_dim]
        attn_output = torch.cat([attn_output, critic_input], dim=-1) # [B, N, 2*embedding_dim]
        values = self.critic_out(attn_output).squeeze(-1) # [B, N]
        values = values.reshape(B, N) # [B, N]
        return values


    def multi_agent_baseline(self, x, action_idx): # this was hard
        """
        x.shape = [B, N, state_dim]
        action_idx.shape = [B, N]
        Returns:
        baseline_values: [B, N]
        """
        B, N, state_dim = x.shape

        with torch.no_grad():
            #  Get actor probabilities (for weighting)
            full_action_probs = self.actor_forward(x)  # [B, N, A]

            #  Embedding for chosen action (one per agent/batch):
            chosen_action_one_hot = torch.zeros(B, N, self.action_size, device=x.device)
            chosen_action_one_hot.scatter_(-1, action_idx.unsqueeze(-1), 1)
            chosen_critic_input = torch.cat([x, chosen_action_one_hot], dim=-1)  # [B, N, state_dim + A]
            chosen_critic_input = chosen_critic_input.view(B*N, -1)              # [B*N, state_dim + A]
            chosen_emb = self.critic_embedding(chosen_critic_input)              # [B*N, emb_dim]
            chosen_emb = chosen_emb.view(B, N, -1)                               # [B, N, emb_dim]

            # Embedding for every possible action for every agent:
            x_expanded = x.unsqueeze(2).repeat(1, 1, self.action_size, 1)
            range_actions = torch.arange(self.action_size, device=x.device).view(1, 1, -1)
            action_range = range_actions.repeat(B, N, 1)  # [B, N, A]
            all_action_one_hot = torch.zeros(B, N, self.action_size, self.action_size, device=x.device)
            all_action_one_hot.scatter_( -1, action_range.unsqueeze(-1), 1)
            critic_input_all = torch.cat([x_expanded, all_action_one_hot], dim=-1)

            # Flatten to feed into critic_embedding
            critic_input_all = critic_input_all.view(B*N*self.action_size, -1)  # [B*N*A, state_dim + A]
            all_actions_emb = self.critic_embedding(critic_input_all)           # [B*N*A, emb_dim]
            all_actions_emb = all_actions_emb.view(B, N, self.action_size, -1)  # [B, N, A, emb_dim]

            # For each agent i, build the final “attention input”
            baseline_values = torch.zeros(B, N, device=x.device)

            for i in range(N):
                # Build a list of embeddings for all agents:
                agent_emb_list = chosen_emb.clone()  # shape => [B, N, emb_dim]

                # Replace i-th agent’s embedding with the “all possible actions” embeddings
                agent_emb_list = agent_emb_list.unsqueeze(2).repeat(1, 1, self.action_size, 1)
                agent_emb_list[:, i, :, :] = all_actions_emb[:, i, :, :]

                # Flatten to pass through attention -> [B*A, N, emb_dim]
                agent_emb_list = agent_emb_list.permute(0, 2, 1, 3)  # => [B, A, N, emb_dim]
                agent_emb_list = agent_emb_list.reshape(B*self.action_size, N, -1)  # => [B*A, N, emb_dim]

                # Pass through attention
                attn_output, _ = self.critic_attention_block(agent_emb_list, agent_emb_list, agent_emb_list) # [B*A, N, emb_dim]
                attn_output = torch.cat([attn_output, agent_emb_list], dim=-1)  # [B*A, N, 2*emb_dim]

                # Extract the i-th agent’s embedding 
                agent_output = attn_output[:, i, :]  # [B*A, emb_dim]

                agent_values = self.critic_out(agent_output).squeeze(-1)  # [B*A]
                agent_values = agent_values.view(B, self.action_size) # [B, A]
                agent_probs = full_action_probs[:, i, :]  # [B, A]
                agent_baseline = (agent_values * agent_probs).sum(dim=1)  # [B]
                baseline_values[:, i] = agent_baseline

            return baseline_values

class TAAC:
    def __init__(self, env_config: Dict[str, Any], training_config: Optional[Dict[str, Any]] = None, mode="train"):
        """
        Initialize TAAC with environment configuration
        
        Args:
            env_config: Dictionary containing environment configuration
            training_config: Dictionary containing training hyperparameters
            mode: "train" or "test"
        """
        self.mode = mode
        self.memories = []

        # Extract environment info
        self.state_size = env_config['state_size']
        self.action_size = env_config['action_size']
        self.action_space_type = "discrete"  
        self.number_of_agents = env_config['num_agents']
        
        # Training hyperparameters with defaults
        if training_config is None:
            training_config = {}
            
        # Convert values to correct types (handles YAML string conversion issues)
        self.gamma = float(training_config.get('gamma', 0.99))
        self.epsilon_clip = float(training_config.get('epsilon_clip', 0.2))
        self.K_epochs = int(training_config.get('K_epochs', 10))
        self.learning_rate = float(training_config.get('learning_rate', 3e-4))
        self.c_entropy = float(training_config.get('c_entropy', 0.01))
        self.max_grad_norm = float(training_config.get('max_grad_norm', 0.5))
        self.c_value = float(training_config.get('c_value', 0.5))
        self.lam = float(training_config.get('lam', 0.95))
        self.base_batch_size = int(training_config.get('batch_size', 64))  # Store original batch size
        self.min_learning_rate = float(training_config.get('min_learning_rate', 1e-6))
        self.episodes = int(training_config.get('episodes', 1000))
        self.similarity_loss_coef = float(training_config.get('similarity_loss_coef', 0.01))
        self.similarity_loss_cap = float(training_config.get('similarity_loss_cap', 0))
        print(f"Similarity loss cap set to: {self.similarity_loss_cap}")
        self.num_heads = int(training_config.get('num_heads', 4))
        self.embedding_dim = int(training_config.get('embedding_dim', 256))
        self.hidden_size = int(training_config.get('hidden_size', 526))
        
        # Calculate mini_batch_size dynamically based on actual agent count during update
        self.mini_batch_size = self.base_batch_size
        #self.mini_batch_size = min(self.base_batch_size * int(math.sqrt(self.number_of_agents)), 4098)  # Keep it manageable
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
        initial_lr = float(self.learning_rate)
        final_lr = float(self.min_learning_rate)
        total_epochs = self.episodes
        lr = final_lr + (initial_lr - final_lr) * (1 - epoch / total_epochs)
        return max(lr / initial_lr, final_lr / initial_lr)

    def _build_model(self):
        network = AttentionActorCriticNetwork(
            state_size=self.state_size, 
            action_size=self.action_size, 
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size
        )
        # Set hyperparameters that need to be configured
        network.similarity_loss_cap = self.similarity_loss_cap
        return network

    def select_action(self, state):
        # 'state' is a list of observations for each agent
        try:
            state_array = np.array(state, dtype=np.float32)

            state_tensor = torch.from_numpy(state_array).to(self.device)
            state_tensor = state_tensor.unsqueeze(0) # add batch dimension
            action_probs = self.policy.actor_forward(state_tensor)
            action_probs = action_probs.squeeze(0) # remove batch dimension
            dist = Categorical(action_probs)
            actions = dist.sample()
            
            # Get log probabilities and entropy
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            # Return actions and log probs as dictionaries
            return {f"agent_{i}": act.item() for i, act in enumerate(actions)}, \
                   {f"agent_{i}": lp.item() for i, lp in enumerate(log_probs)}, \
                   {f"agent_{i}": ent.item() for i, ent in enumerate(entropies)}

        except ValueError as e:
            print(f"Error creating state tensor: {e}")
            print("This usually means the observation format from the environment is incorrect.")
            print(f"Received state shape: {len(state)} agents, with individual shapes: {[s.shape for s in state if hasattr(s, 'shape')]}")
            # Re-raise the error to stop execution, as this is a critical issue
            raise e
        except Exception as e:
            # Catch other potential errors
            print(f"An unexpected error occurred in select_action: {e}")
            raise e


    def update(self):
        """
        B = batch size or number of timesteps
        N = number of agents (variable)
        G = number of games
        """
        if self.mode != "train":
            return
        
        final_similarity_loss = torch.tensor(0.0, device=self.device)

        # For dynamic agent training, we process all memories as one episode
        # instead of chunking by fixed number of agents
        if len(self.memories) == 0:
            return final_similarity_loss.item()

        # Determine actual number of agents from memories
        actual_num_agents = len(self.memories)
        
        # Combine experiences from all memories
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        gae_returns = []
        advantages = []

        # Process as one episode with variable agents
        old_states = [] # [N, B, state_size]
        old_actions = [] # [N, B]
        old_log_probs = [] # [N, B]
        rewards_list = [] # [N, B]
        dones_list = [] # [N, B]

        for j in range(actual_num_agents): # for each agent
            if len(self.memories[j].states) > 0:  # Only add if agent has experiences
                old_states.append(self.memories[j].states)
                old_actions.append(self.memories[j].actions)
                old_log_probs.append(self.memories[j].log_probs)
                rewards_list.append(self.memories[j].rewards)
                dones_list.append(self.memories[j].dones)
        
        # Skip if no agents have experiences
        if len(old_states) == 0:
            return final_similarity_loss.item()
            
        # Check if all sequences have the same length
        sequence_lengths = [len(states) for states in old_states]
        if len(set(sequence_lengths)) > 1:
            # If sequences have different lengths, find the minimum length and truncate all to that
            min_length = min(sequence_lengths)
            old_states = [states[:min_length] for states in old_states]
            old_actions = [actions[:min_length] for actions in old_actions]
            old_log_probs = [log_probs[:min_length] for log_probs in old_log_probs]
            rewards_list = [rewards[:min_length] for rewards in rewards_list]
            dones_list = [dones[:min_length] for dones in dones_list]
            
        # Skip if sequences are empty after truncation
        if len(old_states[0]) == 0:
            return final_similarity_loss.item()
            
        # Convert lists to tensors and reshape (optimize for performance)
        old_states = torch.from_numpy(np.array(old_states, dtype=np.float32)).permute(1, 0, 2).to(self.device) # [B, N, state_size]
        old_actions = torch.LongTensor(old_actions).permute(1, 0).to(self.device) # [B, N]
        old_log_probs = torch.FloatTensor(old_log_probs).permute(1, 0).to(self.device) # [B, N]
        rewards_tensor = torch.FloatTensor(rewards_list).permute(1, 0).to(self.device) # [B, N]
        dones_tensor = torch.FloatTensor(dones_list).permute(1, 0).to(self.device) # [B, N]

        multi_agent_baseline = self.policy_old.multi_agent_baseline(old_states, old_actions) # [B, N]
        gae_returns_tensor, advantages_tensor = self.compute_gae(rewards_tensor, dones_tensor, multi_agent_baseline, actual_num_agents) # [B, N], [B, N]

        # Append to the combined lists
        states.append(old_states) 
        actions.append(old_actions)
        log_probs.append(old_log_probs)
        advantages.append(advantages_tensor)
        gae_returns.append(gae_returns_tensor)

        # Clear memory after processing
        for j in range(actual_num_agents):
            self.memories[j].clear()

        # Concatenate experiences from all agents
        states = torch.cat(states, dim=0) # [B*G , N, state_size]
        actions = torch.cat(actions, dim=0) # [B*G , N]
        log_probs = torch.cat(log_probs, dim=0) # [B*G , N]
        advantages = torch.cat(advantages, dim=0) # [B*G , N] 
        gae_returns = torch.cat(gae_returns, dim=0) # [B*G , N]

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

        # Define mini-batch size dynamically based on actual agent count
        mini_batch_size = max(1, int(self.base_batch_size // actual_num_agents))
        num_mini_batches = dataset_size // mini_batch_size

        # PPO policy update with mini-batching
        for _ in range(self.K_epochs):
            for i in range(num_mini_batches):
                # Define the start and end of the mini-batch
                start = i * mini_batch_size
                end = start + mini_batch_size

                # Slice the mini-batch
                mini_states = states[start:end] # [mini_batch_size, N, state_size]
                mini_actions = actions[start:end] # [mini_batch_size, N]
                mini_log_probs = log_probs[start:end] # [mini_batch_size, N]
                mini_advantages = advantages[start:end] # [mini_batch_size, N]
                mini_gae_returns = gae_returns[start:end] # [mini_batch_size, N]
                
                # Forward pass
                action_probs, similarity_loss = self.policy.actor_forward_update(mini_states) # [mini_batch_size, N, action_size]
                final_similarity_loss = similarity_loss  # Update the final similarity loss
                state_values_new = self.policy.critic_forward(mini_states, mini_actions) # [mini_batch_size, N]
                dist = torch.distributions.Categorical(action_probs) # [mini_batch_size, N]
                action_log_probs = dist.log_prob(mini_actions) # [mini_batch_size, N]
                dist_entropy = dist.entropy() # [mini_batch_size, N]
                
                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs) # [mini_batch_size, N]

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages

                # Calculate loss
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_new.squeeze(), mini_gae_returns) 

                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * dist_entropy.mean() + self.similarity_loss_coef * similarity_loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
        
        # Update old policy parameters with new policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()

        return final_similarity_loss.item()

    
    def compute_gae(self, rewards, dones, baseline_values, num_agents=None):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
    
        :param rewards: List of rewards for an episode. # [B, N]
        :param dones: List of done flags for an episode. # [B, N]
        :param baseline_values: Tensor of state baseline_values. # [B, N]
        :param num_agents: Number of agents (for dynamic agent support)
        :return: Tensors of returns.
        """
        if self.mode != "train":
            return

        # Use provided num_agents or fall back to default
        if num_agents is None:
            num_agents = self.number_of_agents

        # reshaping
        baseline_values = baseline_values.reshape(-1) # [B*N]
        dones = dones.reshape(-1) # [B*N]
        rewards = rewards.reshape(-1) # [B*N]

        baseline_values = baseline_values.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        rewards = rewards.detach().cpu().numpy()

        gamma = self.gamma
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                # For the last timestep, next_value is always 0 (terminal state)
                next_value = 0
            else:
                next_value = baseline_values[i + 1]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - baseline_values[i]
            gae = delta + gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + baseline_values

        #reshaping and converting to tensor [B, N]
        returns = torch.FloatTensor(returns).reshape(-1, num_agents).to(self.device)
        advantages = torch.FloatTensor(advantages).reshape(-1, num_agents).to(self.device)
        return returns, advantages

    
    def save_model(self, model_path: str):
        """Save the model to the specified path."""
        print(f"--> Saving model to {model_path}")
        torch.save(self.policy_old.state_dict(), model_path)

    def load_model(self, model_path: str, test=False) -> bool:
        """Load the model from the specified path."""
        if os.path.exists(model_path):
            if test:
                    # For evaluation, only load the main policy
                    self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Model loaded from {model_path}")
            else:
                    # For training, load both old and new policies
                    self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.policy_old.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} does not exist.")
            return False

    
    def memory_prep(self, number_of_agents):
        if self.mode != "train":
            return
        
        for memory in self.memories:
            memory.clear()

        self.memories = []
        for _ in range(number_of_agents):
            self.memories.append(Memory())
        
    
    def store_experience(self, states, actions, log_probs):
        """
        Store experience in memory for training
        
        Args:
            states: List of states for each agent
            actions: Actions tensor from select_action
            log_probs: Log probabilities tensor from select_action
        """
        if self.mode != "train":
            return

        # Use the actual number of states/agents in this episode instead of fixed number_of_agents
        current_num_agents = len(states)
        
        # Store in memory for each agent
        for i in range(current_num_agents):
            self.memories[i].states.append(states[i])
            self.memories[i].actions.append(actions[f"agent_{i}"])
            self.memories[i].log_probs.append(log_probs[f"agent_{i}"])
    
    def get_actions(self, states):
        """
        Get actions for all agents from the current policy
        """
        with torch.no_grad():
            actions, log_probs, entropies = self.select_action(states)
        
        self.store_experience(states, actions, log_probs)
        return actions, log_probs, entropies
    

    def store_rewards(self, rewards, done):
        if self.mode != "train":
            return

        # Use actual number of rewards instead of fixed number of agents
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