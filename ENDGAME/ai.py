# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from collections import deque


class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 3, 1)
    self.conv2 = nn.Conv2d(8, 16, 3, 1)
    self.conv3 = nn.Conv2d(16, 16, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.avgpooling = nn.AvgPool2d(16)
    self.fc1 = nn.Linear(16, 32)
    self.fc2 = nn.Linear(32, action_dim)
    self.max_action = max_action

  def forward(self, state):
    x = self.conv1(state) # 38x38x8
    x = F.relu(x)
    x = self.conv2(x) # 36x36x16
    x = F.relu(x)
    x = F.max_pool2d(x, 2) # 18x18x16
    x = self.dropout1(x) 
    x = self.conv3(x) # 16x16x16
    x = self.avgpooling(x) #1x1x16
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    actions = self.fc2(x)
    return self.max_action * torch.tanh(actions)



class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.conv1 = nn.Conv2d(1, 8, 3, 1)
    self.conv2 = nn.Conv2d(8, 16, 3, 1)
    self.conv3 = nn.Conv2d(16, 16, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.avgpooling1 = nn.AvgPool2d(16)
    self.fc1 = nn.Linear(17, 32)
    self.fc2 = nn.Linear(32, 1)

    # Defining the second Critic neural network
    self.conv4 = nn.Conv2d(1, 8, 3, 1)
    self.conv5 = nn.Conv2d(8, 16, 3, 1)
    self.conv6 = nn.Conv2d(16, 16, 3, 1)
    self.dropout3 = nn.Dropout2d(0.25)
    self.dropout4 = nn.Dropout2d(0.5)
    self.avgpooling2 = nn.AvgPool2d(16)
    self.fc3 = nn.Linear(17, 32)
    self.fc4 = nn.Linear(32, 1)

  def forward(self, x, u):
    
    # Forward-Propagation on the first Critic Neural Network
    x1 = self.conv1(x) # 38x38x8
    x1 = F.relu(x1)
    x1 = self.conv2(x1) # 36x36x16
    x1 = F.relu(x1)
    x1 = F.max_pool2d(x1, 2) # 18x18x16
    x1 = self.dropout1(x1) 
    x1 = self.conv3(x1) # 16x16x16
    x1 = self.avgpooling1(x1) #1x1x16
    x1 = torch.flatten(x1, 1)
    x1 = torch.cat([x1, u], 1)
    x1 = self.fc1(x1)
    x1 = F.relu(x1)
    x1 = self.dropout2(x1)
    x1 = self.fc2(x1)
    
    # Forward-Propagation on the second Critic Neural Network
    x2 = self.conv4(x) # 38x38x8
    x2 = F.relu(x2)
    x2 = self.conv5(x2) # 36x36x16
    x2 = F.relu(x2)
    x2 = F.max_pool2d(x2, 2) # 18x18x16
    x2 = self.dropout3(x2) 
    x2 = self.conv6(x2) # 16x16x16
    x2 = self.avgpooling2(x2) #1x1x16
    x2 = torch.flatten(x2, 1)
    x2 = torch.cat([x2, u], 1)
    x2 = self.fc3(x2)
    x2 = F.relu(x2)
    x2 = self.dropout4(x2)
    x2 = self.fc4(x2)
    return x1, x2

  def Q1(self, x, u):
    x1 = self.conv1(x) # 38x38x8
    x1 = F.relu(x1)
    x1 = self.conv2(x1) # 36x36x16
    x1 = F.relu(x1)
    x1 = F.max_pool2d(x1, 2) # 18x18x16
    x1 = self.dropout1(x1) 
    x1 = self.conv3(x1) # 16x16x16
    x1 = self.avgpooling1(x1) #1x1x16
    x1 = torch.flatten(x1, 1)
# concatenting the image after flattening and action to critic target 
    x1 = torch.cat([x1, u], 1)
    x1 = self.fc1(x1)
    x1 = F.relu(x1)
    x1 = self.dropout2(x1)
    x1 = self.fc2(x1)
    return x1

class ReplayBuffer(object):

  def __init__(self, max_size=1e5):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# Building the whole Training Process into a class
class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.001)
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action
    self.reward_window = []
    self.memory = ReplayBuffer()
    self.last_state = torch.zeros(state_dim)
    self.last_action = 0
    self.last_reward = 0
    self.action_dim = action_dim
    self.state_dim = state_dim
    # self.episode_timesteps = 0
    self.episode_reward = 0
    self.episode_num = 0
    

  def select_action(self, state):
    state = torch.Tensor(state.unsqueeze(0)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations ,batch_size=100, discount=0.99, tau=0.005, policy_noise=0.9, noise_clip=0.5,policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)

      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device).unsqueeze(1)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
     
      #print("TIMESTAMP",it)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      # print("test------test",next_action.shape)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip).unsqueeze(1)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      # print("test -----test",next_state.shape, action.shape, next_action.shape,state.shape,reward.shape,dones.shape)
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
      # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
      # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  def update(self, reward, new_signal,episode_timesteps,image_step,Done,):
        
        
        batch_size = 50 # Size of the batch
        discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
        tau = 0.005 # Target network update rate
        policy_noise = 1 # STD of Gaussian noise added to the actions for the exploration purposes
        noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
        policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
      
      # print("new state",new_signal.shape)
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
      # print("new state",new_state.shape)       
        self.memory.add((self.last_state, new_state, self.last_action, self.last_reward, Done))
        #print("memory size -----------outside done",len(self.memory.storage))
       
      # train once episode i done for all te time stamps it took to complete the episode
        if Done:
          print("memory size -------------inside DONE ",len(self.memory.storage))
          print("......Total Episode reward",self.episode_reward)
          print("Total time steps for episode....",episode_timesteps)
         
          self.train(self.memory, episode_timesteps,batch_size, discount, tau, policy_noise, noise_clip,policy_freq)

          self.episode_reward = 0
          episode_timesteps = 0
          self.episode_num += 1
        
      # After  1000 random steps get the next actions from actor based on the new state of the car
        if image_step > 1000:
            action = self.select_action(new_state)
            action = action[0]
        else:  # Run intially 1000 random steps without policy gradient 
            action = random.uniform(-5,5)
        self.last_action = action
       
       
        self.last_state = new_state
        self.last_reward = reward

        self.episode_reward += reward
        episode_timesteps += 1

        return action, episode_timesteps


  def save(self):
    torch.save(self.actor.state_dict(), 'actor.pth')
    torch.save(self.critic.state_dict(), 'critic.pth')
    
  def load(self):
    self.actor.load_state_dict(torch.load('actor.pth'))
    self.critic.load_state_dict(torch.load('critic.pth'))
