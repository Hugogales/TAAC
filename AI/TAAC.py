import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
import os
from typing import Dict, Any, Optional, Union


class Memory:
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
    
    def __init__(self, state_size, action_size, action_space_type="discrete", num_heads=4, embedding_dim=256, hidden_size=526):
        super(AttentionActorCriticNetwork, self).__init__()

        # Make hyperparameters configurable instead of importing from external file
        self.temperature = 1.0  # Default value, can be overridden
        self.similarity_loss_cap = -0.5  # Default value, can be overridden
        self.action_size = action_size
        self.action_space_type = action_space_type  # "discrete" or "continuous"
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

        # Actor output depends on action space type
        if action_space_type == "discrete":
            self.actor_out = nn.Sequential(
                nn.Linear(embedding_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, action_size)
            )
        else:  # continuous
            # For multi-dimensional continuous actions, we need separate mean and std networks
            self.actor_mean = nn.Sequential(
                nn.Linear(embedding_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, action_size)
            )
            # Learnable log standard deviation for each action dimension
            self.actor_logstd = nn.Parameter(torch.zeros(action_size))
            
            # Optional: Action bounds for clamping (can be set externally)
            self.action_low = -1.0  # Default bounds
            self.action_high = 1.0

        # Critic network - input size depends on action space type
        critic_input_size = state_size + (action_size if action_space_type == "discrete" else action_size)
        
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
        print(f"State size: {state_size}, Action size: {action_size}, Action type: {action_space_type}")
        if action_space_type == "continuous":
            print(f"Continuous action dimensions: {action_size}")
            print(f"Actor mean output shape: [batch_size, num_agents, {action_size}]")
            print(f"Actor std parameters: {action_size} learnable log-std values")

    def actor_forward(self, x):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          For discrete: action_probs [batch_size, num_agents, action_size]
          For continuous: (mean, std) tuple
        """
        B, N, D = x.shape
        actor_input = x.reshape(B*N, -1) # [B*N, state_size]
        actor_input = self.actor_embedding(actor_input) # [B*N, embedding_dim]
        actor_input = actor_input.reshape(B, N, -1) # [B, N, embedding_dim]
        attn_output, _ = self.actor_attention_block(actor_input, actor_input, actor_input) # [B, N, embedding_dim]
        
        if self.action_space_type == "discrete":
            action_logits = self.actor_out(attn_output) # [B, N, action_size]
            action_probs = torch.softmax(action_logits / self.temperature, dim=-1)
            return action_probs
        else:  # continuous
            action_mean = self.actor_mean(attn_output)
            action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
            return action_mean, action_std


    def actor_forward_update(self, x):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        
        Additionally calculate similarity loss for training
        Returns:
          For discrete: action_probs [batch_size, num_agents, action_size]
          For continuous: (mean, std) tuple
        """
        B, N, D = x.shape
        actor_input = x.reshape(B*N, -1) # [B*N, state_size]
        actor_input = self.actor_embedding(actor_input) # [B*N, embedding_dim]
        actor_input = actor_input.reshape(B, N, -1) # [B, N, embedding_dim]
        attn_output, _ = self.actor_attention_block(actor_input, actor_input, actor_input) # [B, N, embedding_dim]

        # similarity loss 
        normalized_attn_output = attn_output / attn_output.norm(dim=-1, keepdim=True) # [B, N, embedding]
        similarity_matrix = torch.matmul(normalized_attn_output, normalized_attn_output.transpose(-2, -1)) # [B, N, N]
        mask = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1).bool() # [B, N, N]
        similarity_matrix = similarity_matrix.masked_fill(mask, 0) # [B, N, N]
        similarity_loss = torch.sum(similarity_matrix, dim=(1, 2)) / (N * (N - 1))  # [1]

        # remove negative values
        similarity_loss = torch.clamp(similarity_loss, min=self.similarity_loss_cap)
        similarity_loss = similarity_loss.mean()
        
        if self.action_space_type == "discrete":
            action_logits = self.actor_out(attn_output)
            action_probs = torch.softmax(action_logits / self.temperature, dim=-1)
            return action_probs, similarity_loss
        else:  # continuous
            action_mean = self.actor_mean(attn_output)
            action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
            return (action_mean, action_std), similarity_loss
    
    def critic_forward(self, x, actions):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          action_probs: [batch_size, num_agents, action_size]
        """
        B, N, D = x.shape
        
        if self.action_space_type == "discrete":
            # Convert discrete actions to one-hot
            action_one_hot = torch.zeros(B, N, self.action_size).to(x.device)
            action_one_hot.scatter_(-1, actions.unsqueeze(-1), 1)
            critic_input = torch.cat([x, action_one_hot], dim=-1)
        else:  # continuous
            # Use action values directly
            critic_input = torch.cat([x, actions], dim=-1)
            
        critic_input = critic_input.reshape(B*N, -1)
        critic_input = self.critic_embedding(critic_input)
        critic_input = critic_input.reshape(B, N, -1)
        attn_output, _ = self.critic_attention_block(critic_input, critic_input, critic_input)
        attn_output = torch.cat([attn_output, critic_input], dim=-1)
        values = self.critic_out(attn_output).squeeze(-1)
        return values.reshape(B, N)

    def multi_agent_baseline(self, x, actions):
        """
        Compute baseline values for multi-agent setting
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        actions.shape = [batch_size, num_agents]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - value is the id of the action
        Returns:
          baseline_values: [batch_size, num_agents]
        """
        if self.action_space_type == "continuous":
            # For continuous actions, we cannot use the multi-agent baseline
            return self.critic_forward(x, actions)
            
        # Original implementation for discrete actions
        B, N, state_dim = x.shape

        with torch.no_grad():
            #  Get actor probabilities (for weighting)
            full_action_probs = self.actor_forward(x)  # [B, N, A]

            #  Embedding for chosen action (one per agent/batch):
            chosen_action_one_hot = torch.zeros(B, N, self.action_size, device=x.device)
            chosen_action_one_hot.scatter_(-1, actions.unsqueeze(-1), 1)
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
        self.action_space_type = env_config.get('action_space_type', 'discrete')
        self.number_of_agents = env_config['num_agents']
        
        # Training hyperparameters with defaults
        if training_config is None:
            training_config = {}
            
        self.gamma = training_config.get('gamma', 0.99)
        self.epsilon_clip = training_config.get('epsilon_clip', 0.2)
        self.K_epochs = training_config.get('K_epochs', 10)
        self.learning_rate = training_config.get('learning_rate', 3e-4)
        self.c_entropy = training_config.get('c_entropy', 0.01)
        self.max_grad_norm = training_config.get('max_grad_norm', 0.5)
        self.c_value = training_config.get('c_value', 0.5)
        self.lam = training_config.get('lam', 0.95)
        self.mini_batch_size = training_config.get('batch_size', 64)
        self.min_learning_rate = training_config.get('min_learning_rate', 1e-6)
        self.episodes = training_config.get('episodes', 1000)
        self.similarity_loss_coef = training_config.get('similarity_loss_coef', 0.01)
        self.similarity_loss_cap = training_config.get('similarity_loss_cap', 0)
        self.num_heads = training_config.get('num_heads', 4)
        self.embedding_dim = training_config.get('embedding_dim', 256)
        self.hidden_size = training_config.get('hidden_size', 526)
        
        self.mini_batch_size = int(self.mini_batch_size // self.number_of_agents)
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
        network = AttentionActorCriticNetwork(
            state_size=self.state_size, 
            action_size=self.action_size, 
            action_space_type=self.action_space_type,
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size
        )
        # Set hyperparameters that need to be configured
        network.similarity_loss_cap = self.similarity_loss_cap
        return network

    def select_action(self, state):
        """
        Select actions for all agents given their states
        
        Args:
            state: List of states for each agent
            
        Returns:
            actions, log_probs, entropy
        """
        try:
            # Convert to numpy array first for efficiency, then to tensor
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.action_space_type == "discrete":
                    action_probs = self.policy_old.actor_forward(state_tensor)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    action_log_prob = dist.log_prob(action)
                    entropy = dist.entropy()
                else:  # continuous - handles multi-dimensional actions
                    action_mean, action_std = self.policy_old.actor_forward(state_tensor)
                    
                    # Create distribution for each action dimension
                    dist = torch.distributions.Normal(action_mean, action_std)
                    raw_action = dist.sample()
                    
                    # Apply tanh transformation for bounded actions [-1, 1]
                    action = torch.tanh(raw_action)
                    
                    # Scale to actual action bounds if needed
                    if hasattr(self.policy_old, 'action_low') and hasattr(self.policy_old, 'action_high'):
                        action_low = self.policy_old.action_low
                        action_high = self.policy_old.action_high
                        # Scale from [-1, 1] to [action_low, action_high]
                        action = action_low + (action_high - action_low) * (action + 1) / 2
                    
                    # Adjust log probabilities for tanh transformation
                    # For multi-dimensional actions, sum over action dimensions
                    action_log_prob = dist.log_prob(raw_action).sum(dim=-1)
                    action_log_prob -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1)
                    
                    # Entropy sum over action dimensions for multi-dimensional actions
                    entropy = dist.entropy().sum(dim=-1)
                    
        except ValueError as e:
            print(e)
            print(f"state tensor: {state_tensor}")
            print(f"state tensor shape: {state_tensor.shape}")
            if self.action_space_type == "continuous":
                print(f"action_size: {self.action_size}")
            raise ValueError("Error in action selection")

        return action, action_log_prob, entropy

    

    def update(self):
        """
        B = batch size or number of timesteps
        N = number of agents
        G = number of games
        """

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

        for i in range(0, len(self.memories), self.number_of_agents): # for each game
            old_states = [] # [N, B, state_size]
            old_actions = [] # [N, B]
            old_log_probs = [] # [N, B]
            rewards = [] # [N, B]
            dones = [] # [N, B]

            for j in range(self.number_of_agents): # for each agent
                old_states.append(self.memories[i+j].states)
                old_actions.append(self.memories[i+j].actions)
                old_log_probs.append(self.memories[i+j].log_probs)
                rewards.append(self.memories[i+j].rewards)
                dones.append(self.memories[i+j].dones)
            
            # Convert lists to tensors and reshape (optimize for performance)
            old_states = torch.from_numpy(np.array(old_states, dtype=np.float32)).permute(1, 0, 2).to(self.device) # [B, N, state_size]
            
            # Handle different action types
            if self.action_space_type == "discrete":
                old_actions = torch.LongTensor(old_actions).permute(1, 0).to(self.device) # [B, N]
            else:  # continuous
                old_actions = torch.FloatTensor(old_actions).permute(1, 0, 2).to(self.device) # [B, N, action_size]
                
            old_log_probs = torch.FloatTensor(old_log_probs).permute(1, 0).to(self.device) # [B, N]
            rewards = torch.FloatTensor(rewards).permute(1, 0).to(self.device) # [B, N]
            dones = torch.FloatTensor(dones).permute(1, 0).to(self.device) # [B, N]

            multi_agent_baseline = self.policy_old.multi_agent_baseline(old_states, old_actions) # [B, N]
            gae_returns_tensor, advantages_tensor = self.compute_gae(rewards, dones, multi_agent_baseline) # [B, N], [B, N]

            # Append to the combined lists
            states.append(old_states) 
            actions.append(old_actions)
            log_probs.append(old_log_probs)
            advantages.append(advantages_tensor)
            gae_returns.append(gae_returns_tensor)

            # Clear memory after processing
            for j in range(self.number_of_agents):
                self.memories[i+j].clear()

        # Concatenate experiences from all agents
        states = torch.cat(states, dim=0) # [B*G , N, state_size]
        actions = torch.cat(actions, dim=0) # [B*G , N] for discrete, [B*G, N, action_size] for continuous
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
                mini_states = states[start:end] # [mini_batch_size, N, state_size]
                mini_actions = actions[start:end] # [mini_batch_size, N] for discrete, [mini_batch_size, N, action_size] for continuous
                mini_log_probs = log_probs[start:end] # [mini_batch_size, N]
                mini_advantages = advantages[start:end] # [mini_batch_size, N]
                mini_gae_returns = gae_returns[start:end] # [mini_batch_size, N]
                
                # Forward pass
                if self.action_space_type == "discrete":
                    action_probs, similarity_loss= self.policy.actor_forward_update(mini_states) # [mini_batch_size, N, action_size]
                    state_values_new = self.policy.critic_forward(mini_states, mini_actions) # [mini_batch_size, N]
                    dist = torch.distributions.Categorical(action_probs) # [mini_batch_size, N]
                    action_log_probs = dist.log_prob(mini_actions) # [mini_batch_size, N]
                    dist_entropy = dist.entropy() # [mini_batch_size, N]
                else: # continuous - multi-dimensional actions
                    (action_mean, action_std), similarity_loss = self.policy.actor_forward_update(mini_states) # [mini_batch_size, N, action_size]
                    state_values_new = self.policy.critic_forward(mini_states, mini_actions) # [mini_batch_size, N]
                    dist = torch.distributions.Normal(action_mean, action_std) # Multi-dimensional normal distribution
                    # For multi-dimensional actions, sum log probabilities across action dimensions
                    action_log_probs = dist.log_prob(mini_actions).sum(dim=-1) # [mini_batch_size, N]
                    # For multi-dimensional actions, sum entropy across action dimensions  
                    dist_entropy = dist.entropy().sum(dim=-1) # [mini_batch_size, N]
                
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

            # Handle any remaining data not fitting into mini-batches
            remainder = dataset_size % mini_batch_size
            if remainder != 0:
                start = num_mini_batches * mini_batch_size
                mini_states = states[start:]
                mini_actions = actions[start:]
                mini_log_probs = log_probs[start:]
                mini_advantages = advantages[start:]
                mini_gae_returns = gae_returns[start:]

                if self.action_space_type == "discrete":
                    action_probs, similarity_loss = self.policy.actor_forward_update(mini_states)
                    state_values_new = self.policy.critic_forward(mini_states, mini_actions)
                    dist = torch.distributions.Categorical(action_probs)
                    action_log_probs = dist.log_prob(mini_actions)
                    dist_entropy = dist.entropy()
                else: # continuous - multi-dimensional actions
                    (action_mean, action_std), similarity_loss = self.policy.actor_forward_update(mini_states)
                    state_values_new = self.policy.critic_forward(mini_states, mini_actions)
                    dist = torch.distributions.Normal(action_mean, action_std) # Multi-dimensional normal distribution
                    # For multi-dimensional actions, sum log probabilities across action dimensions
                    action_log_probs = dist.log_prob(mini_actions).sum(dim=-1)
                    # For multi-dimensional actions, sum entropy across action dimensions
                    dist_entropy = dist.entropy().sum(dim=-1)
                
                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_new.squeeze(), mini_gae_returns)
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * dist_entropy.mean() + self.similarity_loss_coef * similarity_loss
                # Calculate loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
        
        # Update old policy parameters with new policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()

        return similarity_loss.item()

    
    def compute_gae(self, rewards, dones, baseline_values):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
    
        :param rewards: List of rewards for an episode. # [B, N]
        :param dones: List of done flags for an episode. # [B, N]
        :param baseline_values: Tensor of state baseline_values. # [B, N]
        :return: Tensors of returns.
        """
        if self.mode != "train":
            return

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
        returns = torch.FloatTensor(returns).reshape(-1, self.number_of_agents).to(self.device)
        advantages = torch.FloatTensor(advantages).reshape(-1, self.number_of_agents).to(self.device)
        return returns, advantages

    
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

    
    def set_action_bounds(self, action_low, action_high):
        """
        Set action bounds for continuous action spaces
        
        Args:
            action_low: Lower bound for actions (scalar or array)
            action_high: Upper bound for actions (scalar or array)
        """
        if self.action_space_type == "continuous":
            self.policy.action_low = action_low
            self.policy.action_high = action_high
            self.policy_old.action_low = action_low
            self.policy_old.action_high = action_high
            print(f"Action bounds set: [{action_low}, {action_high}]")
        else:
            print("Warning: Action bounds only applicable for continuous action spaces")

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
            
        # Convert tensors to appropriate format for storage
        if self.action_space_type == "discrete":
            # actions shape: [1, num_agents] -> convert to numpy
            actions_np = actions.squeeze(0).cpu().numpy()
        else:  # continuous - multi-dimensional actions
            # actions shape: [1, num_agents, action_size] -> convert to numpy
            actions_np = actions.squeeze(0).cpu().numpy()
            
        log_probs_np = log_probs.squeeze(0).cpu().numpy()
        
        # Store in memory for each agent
        for i in range(self.number_of_agents):
            self.memories[i].states.append(states[i])
            
            if self.action_space_type == "discrete":
                self.memories[i].actions.append(actions_np[i])
            else:  # continuous - store full action vector
                self.memories[i].actions.append(actions_np[i])
                
            self.memories[i].log_probs.append(log_probs_np[i])

    def get_actions(self, states):
        """
        Get actions for all agents given their states (used during training/testing)
        
        Args:
            states: List of states for each agent
            
        Returns:
            Dictionary of actions for each agent, entropy dictionary
        """
        actions, log_probs, entropies = self.select_action(states)
        
        # Store experience for training
        if self.mode == "train":
            self.store_experience(states, actions, log_probs)
        
        # Convert to dictionary format expected by environment
        action_dict = {}
        entropy_dict = {}
        
        # Handle different tensor shapes for discrete vs continuous actions
        if self.action_space_type == "discrete":
            # actions shape: [1, num_agents]
            for i in range(self.number_of_agents):
                action_dict[f"agent_{i}"] = actions[0, i].item()
                entropy_dict[f"agent_{i}"] = entropies[0, i].item()
        else:  # continuous - multi-dimensional actions
            # actions shape: [1, num_agents, action_size]
            for i in range(self.number_of_agents):
                if self.action_size == 1:
                    # Single dimensional action
                    action_dict[f"agent_{i}"] = actions[0, i, 0].item()
                else:
                    # Multi-dimensional action - return as numpy array
                    action_dict[f"agent_{i}"] = actions[0, i].detach().cpu().numpy()
                entropy_dict[f"agent_{i}"] = entropies[0, i].item()
        
        return action_dict, entropy_dict
    

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