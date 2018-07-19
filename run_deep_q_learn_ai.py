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
from PIL import Image
import sys

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

# Load the network
net = torch.load(sys.argv[1])

# Run the game
done = False
num_pixels = 32 
env = gym.make("CartPole-v1")
current_state = env.reset()
while(not done):
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
	print(action, [float(q_guess[0, 0]), float(q_guess[0, 1])])
	new_state, reward, done, info = env.step(action)
	time.sleep(.1)
