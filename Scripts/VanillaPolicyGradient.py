import torch
from torch import nn
import tensorflow as tf
import torch.optim as optim
import torch.nn.functional as F
from tensorflow.keras import layers, optimizers, losses
import os
import sys
import random
import numpy as np
import pandas as pd

"""Observations are returned by the env as dictionary of values that need to be preprocessed"""
def process_observation(obs, size, max_steps, lives):
    # Flatten grid
    grid_flat = obs['grid'].flatten()

    # Normalize agent position
    agent_x, agent_y = obs['agent_pos']
    normalized_x = agent_x / size
    normalized_y = agent_y / size

    # Normalize steps left and lives
    steps_left = np.array([obs['steps_left'] / max_steps])
    lives = np.array([obs['lives'] / lives])

    # Flatten special treasures as its 81*2
    special_treasures_flat = obs['special_treasures'].flatten()

    # Merge other features
    other_features = np.concatenate([
        np.array([normalized_x, normalized_y]),
        steps_left,
        lives,
        special_treasures_flat
    ])
    return grid_flat, other_features



"""Process the observation to give probabilities for the actions"""
class PolicyNetwork(nn.Module):
    def __init__(self, grid_size, feature_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        
        # features =  81 (grid) + 81 (special treasures) + 2 (agent pos) + 1 (steps left) + 1 (lives) = 166
        actual_feature_size = 166
        
        # Define grid input processing
        self.grid_flatten = nn.Flatten()
        self.grid_dense = nn.Sequential(
            nn.Linear(grid_size * grid_size, 64),
            nn.ReLU()
        )
        
        # Use the actual feature size from your data
        self.feature_dense = nn.Sequential(
            nn.Linear(actual_feature_size, 64),
            nn.ReLU()
        )
        
        # Rest of the network remains the same
        self.combined_dense = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU()
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.output = nn.Linear(32, action_size)
        
    def forward(self, grid, features):
        # Process grid through grid network
        grid_out = self.grid_dense(self.grid_flatten(grid))
        
        # Process features through feature network
        feature_out = self.feature_dense(features)
        
        # Combine both grid and feature outputs
        combined_out = torch.cat((grid_out, feature_out), dim=-1)
        combined_out = self.combined_dense(combined_out)
        
        # Pass through hidden layers
        hidden1_out = self.hidden1(combined_out)
        hidden2_out = self.hidden2(hidden1_out)
        
        # Output layer
        action_values = self.output(hidden2_out)
        
        return F.softmax(action_values, dim=-1)  # Apply softmax to get proper probabilities
    
  
"""Based on the policy network processing, the agent network selects action and updates the policy"""    
class PolicyGradientAgent:
    def __init__(self, grid_dim, feature_dim, action_dim, learning_rate=0.001, gamma=0.90, hidden_size=128, batch_size=32,
                 grad_clip=1.0, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.policy = PolicyNetwork(grid_dim, feature_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.log_probs = []
        self.rewards = []
        self.losses = []
        
    def select_action(self, state):
        grid, features = state
        grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)      
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0) 
    
        action_probs = self.policy(grid, features).squeeze(0) 
        dist = torch.distributions.Categorical(action_probs)
        
        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(action_probs))
        else:
            action = dist.sample().item()
    
        self.log_probs.append(dist.log_prob(torch.tensor(action)))
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        discounted_rewards = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)

        # Calculate the discounted rewards and normalize using std and mean to prevent large variations in the rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # Consatant is added to avoid dividing by 0 
        
        # Calculate entropy using the discounted rewards
        loss = 0
        for log_prob, G in zip(self.log_probs, discounted_rewards):
            loss -= log_prob * G

        self.optimizer.zero_grad() # Clear the optimizer from last pass
        loss.backward() # Perform backprop
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip) # Apply gradient clipping
        self.optimizer.step() # Update parameters

        self.losses.append(loss.item())
        self.log_probs = []
        self.rewards = []

    def learn(self, env, num_episodes=100000,log_every=5,size=9,max_steps=1000,lives=3):
        logs = {
        'episode': [],
        'total_reward': [],
        'epsilon': [],
        'loss': [],
        'episode_length': [],
        } 
        reward_per_step_logs = {
        'episode': [],
        'reward_per_step': [],
        }

        for episode in range(num_episodes):
            obs = env.reset()[0]
            grid, features = process_observation(obs, size, max_steps, lives)

            total_reward = 0
            steps = 0
            
            while True:
                action = self.select_action((grid,features))
                next_obs, reward, done, _, _ = env.step(action)
                    
                self.store_reward(reward)
                grid, features = process_observation(next_obs, size, max_steps, lives)
                total_reward += reward
                steps+=1
                reward_per_step_logs['episode'].append(episode)
                reward_per_step_logs['reward_per_step'].append(reward)
                if done:
                    break

            # After the defined batch_size number of episodes, update policy
            if (episode + 1) % self.batch_size == 0 or episode == num_episodes - 1:
                self.update_policy()


            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            # Log data
            logs['episode'].append(episode)
            logs['total_reward'].append(total_reward)
            logs['epsilon'].append(self.epsilon)
            logs['loss'].append(self.losses[-1] if self.losses else 0)
            logs['episode_length'].append(steps)
            
            if episode % log_every == 0 or episode==num_episodes-1:
              last_loss = self.losses[-1] if self.losses else 0
              print(f"Episode {episode}, Reward: {total_reward}, Loss: {last_loss}, Epsilon: {self.epsilon:.3f}")

        return logs,reward_per_step_logs    