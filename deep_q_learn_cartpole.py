from __future__ import print_function
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter
import matplotlib.pyplot as plt
import gym
import math
from PIL import Image

# Neural Network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		final_pixel_val = num_pixels
		self.conv1 = nn.Conv2d(3, 6, 8)
		final_pixel_val -= 7
		self.conv2 = nn.Conv2d(6, 16, 4)
		final_pixel_val -= 3
		self.fc1 = nn.Linear(16 * final_pixel_val * final_pixel_val, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 2)

	def forward(self, x):
		final_pixel_val = num_pixels
		x = F.relu(self.conv1(x))
		final_pixel_val -= 7
		x = F.relu(self.conv2(x))
		final_pixel_val -= 3
		x = x.view(-1, 16 * final_pixel_val * final_pixel_val)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Memory():
	def __init__(self, capacity=10000):
		self.capacity = capacity
		self.memory = []
	def add_elem(self, elem):
		if(len(self.memory) > self.capacity):
			del(self.memory[0])
		self.memory.append(elem)
	def get_memory(self, max_to_include=500):
		num_to_include = max_to_include
		if(len(self.memory) < num_to_include):
			num_to_include = len(self.memory)
		random.shuffle(self.memory)
		return self.memory[:num_to_include-1]

# Train the network
mem = Memory()
num_pixels = 32
net = Net()
lr = .5
dv = .9
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001)
env = gym.make("CartPole-v1")
num_episodes = 1000
num_episodes += 1
for episode in range(0, num_episodes):
	print("Episode: " + str(episode))
	current_state = env.reset()
	done = False
	while(not done):
		# Exploit or Explore
		epsilon = math.exp(-4.6*episode/num_episodes)
		rand_num = random.random()
		if(rand_num > epsilon):
			#Exploit
			current_state = env.render(mode='rgb_array')
			current_state = current_state[160:320, :, :]
			current_image = Image.fromarray(current_state)
			current_image = current_image.resize((num_pixels, num_pixels), Image.ANTIALIAS)
			X = np.zeros((1, num_pixels, num_pixels, 3))
			X[0] = np.array(current_image)
			X = np.swapaxes(X, 1, 3)
			X = X.astype(np.float32)
			X = torch.from_numpy(X)
			q_guess = net(X)
			action = np.argmax(q_guess.detach().numpy())
		else:
			#Explore
			action = env.action_space.sample()
		# Make Move
		current_state = env.render(mode='rgb_array')
		new_state, reward, done, info = env.step(action)
		new_state = env.render(mode='rgb_array')
		crop_current_state = current_state[160:320, :, :]
		current_image = Image.fromarray(crop_current_state)
		current_image = current_image.resize((num_pixels, num_pixels), Image.ANTIALIAS)
		crop_new_state = new_state[160:320, :, :]
		new_image = Image.fromarray(crop_new_state)
		new_image = new_image.resize((num_pixels, num_pixels), Image.ANTIALIAS)
		current_state = new_state

		# Add to memory
		mem.add_elem((current_image, action, reward, new_image, done))

	# Train
	mem_list = mem.get_memory(max_to_include=100)
	for i in range(len(mem_list)):
		optimizer.zero_grad()
		X = np.zeros((1, num_pixels, num_pixels, 3))
		new_X = np.zeros((1, num_pixels, num_pixels, 3))
		current_image, action, reward, new_image, done = mem_list[i]
		X[0] = np.array(current_image)
		new_X[0] = np.array(new_image)
		if(done):
			X = np.swapaxes(X, 1, 3)
			X = X.astype(np.float32)
			X = torch.from_numpy(X)
			q_guess = net(X)
			q_target = q_guess.clone()
			q_target[0][action] = (1 - lr) * q_guess[0][action] + lr * reward
			loss = criterion(q_guess, q_target.detach())
			loss.backward()
			optimizer.step()
		else:
			X = np.swapaxes(X, 1, 3)
			X = X.astype(np.float32)
			X = torch.from_numpy(X)
			q_guess = net(X)
			new_X = np.swapaxes(new_X, 1, 3)
			new_X = new_X.astype(np.float32)
			new_X = torch.from_numpy(new_X)
			new_q_vals = net(new_X)
			max_new_state_q_val = np.amax(new_q_vals[0].detach().numpy())
			q_target = q_guess.clone()
			q_target[0][action] = (1 - lr) * q_guess[0][action] + lr * (reward + dv * max_new_state_q_val)
			loss = criterion(q_guess, q_target.detach())
			loss.backward()
			optimizer.step()

	# Save the Neural Net
	if(episode % 20 == 0):
		torch.save(net, "deep_q_net_" + str(episode) + ".pt")
