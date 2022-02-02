# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 02:16:58 2021

@author: amtul, ryan
"""

import random as r

import sys
import os

import nrrd

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# fixes some potential compatibility problems between libraries
os.environ['KMP_DUPLICATE_LIB_OK']='True'
OUT_OF_BOUNDS_REWARD = -2
WINDOW_SIZE = 50
SEPARATE_CRITIC = False


def euclidean_distance(pos1, pos2):
    """
    Returns Euclidean distance between points.
    """
    distance_squared = np.square(pos1[0] - pos2[0]) + np.square(pos1[1] - pos2[1]) + np.square(pos1[2] - pos2[2])
    distance = np.sqrt(distance_squared)
    return distance


def reward(current_pos, new_pos, target_pos):
    """
    Returns reward given two states and the target.
    """
    reward = 0

    current_distance = euclidean_distance(target_pos, current_pos)
    new_distance = euclidean_distance(target_pos, new_pos)

    # If the distance of the target from the new position is less than the distance of the target from the current position, a positive reward will be given
    reward = np.sign(current_distance - new_distance)

    return reward


def step_function(a):
    """
    Given a direction, returns a step of that size.
    """
    eta = 2  # length of a unit step
    if torch.sign(a) == 1:
        U = eta
    elif torch.sign(a) == -1:
        U = -eta
    else:
        U = 0

    return U


def transition_function(q, a):
    """
    Returns a new state given an action and state.
    """
    q1 = np.array((q[0] + step_function(a[0]), q[1] + step_function(a[1]), q[2] + step_function(a[2])))

    return q1


def env(q, a, target_pos, bounds):
    """
    Handles transition to next state and reward
    """
    #Next position q1
    q1 = transition_function(q, a)

    # check if we went out of bounds of the picture
    if any(q1 >= bounds) or any(q1 < (0, 0, 0)):
        q1=q
        # negative reward for moving out
        r = OUT_OF_BOUNDS_REWARD
        
    else:    
        #Reward
        r = reward(q, q1, target_pos) #target_pos is from the measurement.txt file. But is it L1, or L2, or L3...
    
    return q1, r
    

def select_action(probs):
    """
    Picks an action based on the probabilities given
    """
    actions = torch.tensor([-1, 1])
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)
    index = m.sample()
    return actions[index], m.log_prob(index)


def read_image(filename):
    """
    Reads a 2D image.
    """
    image_data = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(image_data)
    return torch.Tensor(nda).unsqueeze(0)


def plot_lines(ax, history, data, target_pos, c1, c2):
    """
    Plots a single trajectory.
    """
    # for some reason we need to transpose the coordinates to get them to match image
    ax.scatter(target_pos[c1], target_pos[c2], color="green")
    ax.scatter(history[0, c1], history[0, c2], color="red")
    ax.plot(history[:, c1],history[:, c2], "-", color="red")
    ax.set_aspect(data.shape[c1]/data.shape[c2])

def plot_history(history, data, target_pos, filename = None):
    """
    Plots history of our positions
    """
    plt.figure(figsize = (30, 10))
    ax = plt.subplot(1, 3, 1)
    plt.imshow(data[target_pos[0],:,:])
    plot_lines(ax, history, data, target_pos, 2, 1)

    ax = plt.subplot(1, 3, 2)
    plt.imshow(data[:, target_pos[1], :])
    plot_lines(ax, history, data, target_pos, 2, 0)

    ax = plt.subplot(1, 3, 3)
    plt.imshow(data[:, :, target_pos[2]])
    plot_lines(ax, history, data, target_pos, 1, 0)

    # save figure if needed
    if filename is not None:
        plt.savefig(filename)
        plt.close()


def compute_target(reward, discount_factor, value_net_new_state, value_net_old_state):
    """
    Computes the TD target and error.
    """
    TD_target = reward + (discount_factor * value_net_new_state)
    TD_error = TD_target - value_net_old_state

    return TD_target, TD_error


def state(pos, data):
    """
    Takes 3 m*m windows of the image centered at the position,
    along each plane (sagittal axial coronal) and returns them stacked.
    This will be the state that we feed to the policy net and value net.
    """
    window_size = WINDOW_SIZE
    data_padded = F.pad(data.unsqueeze(0).unsqueeze(0),
                        (window_size, window_size, window_size, window_size, window_size, window_size),
                        "replicate")
    xy = data_padded[:,:,pos[0] : pos[0]+2*window_size+1,
                     pos[1] : pos[1]+2*window_size+1,
                     pos[2]]
    yz = data_padded[:,:,pos[0],
                     pos[1] : pos[1]+2*window_size+1,
                     pos[2] : pos[2]+2*window_size+1]
    xz = data_padded[:,:,pos[0] : pos[0]+2*window_size+1,
                     pos[1],
                     pos[2] : pos[2]+2*window_size+1]
    return torch.stack((xy.squeeze(), yz.squeeze(), xz.squeeze())).unsqueeze(0) # add a batch dimension at the start

class CNN(nn.Module):
    """
    Common CNN module for all value and policy estimators.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5), padding=0, padding_mode="replicate")
        self.conv2 = nn.Conv2d(32, 32, (5, 5), padding=0, padding_mode="replicate")
        self.conv3 = nn.Conv2d(32, 64, (5, 5), padding=0, padding_mode="replicate")
        self.conv4 = nn.Conv2d(64, 128, (5, 5), padding=0, padding_mode="replicate")

    def forward(self, x):
        # common convolutional layers
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=(2, 2))

        return x

class Partial_Policy(nn.Module):
    """
    Contains both policy and value estimator.
    """
    def __init__(self):
        super(Partial_Policy, self).__init__()

        # actor layer
        self.policy_lin1 = nn.Linear(128*2*2, 20)
        self.policy_lin2 = nn.Linear(20, 2)

        # critic layer
        self.value_lin1 = nn.Linear(128*2*2, 20)
        self.value_lin2 = nn.Linear(20, 1)

    def forward(self, x, critic = False):
        '''
        Critic = True will use the critic attached to this policy.
        '''
        # flatten out x
        x = x.view(*x.shape[:-3], -1)

        # actor: returns probability of each action
        pol1 = F.relu(self.policy_lin1(x))
        action_prob = F.softmax(self.policy_lin2(pol1), dim=-1)

        # critic: estimates value of being in current state
        val1 = F.relu(self.value_lin1(x))
        state_value = self.value_lin2(val1)

        # returns a tuple of (action probabilities (tensor), estimated value)
        if critic:
            return action_prob, state_value
        return action_prob

class Critic(nn.Module):
    '''
    Standalone critic to be used when testing a universal critic.
    '''
    def __init__(self):
        super(Critic, self).__init__()

        # critic layer
        self.value_lin1 = nn.Linear(128*2*2, 20)
        self.value_lin2 = nn.Linear(20, 1)

    def forward(self, x):
        # flatten out x
        x = x.view(*x.shape[:-3], -1)

        # critic: estimates value of being in current state
        val1 = F.relu(self.value_lin1(x))
        state_value = self.value_lin2(val1)

        # returns a tuple of (action probabilities (tensor), estimated value)
        return state_value

if __name__ == "__main__":
    filename = ".\\sample_data.nrrd"
    readdata, header = nrrd.read(filename)
        
    data = torch.Tensor(readdata)

    policies = [Partial_Policy(), Partial_Policy(), Partial_Policy()]
    cnn = CNN()
    critic = Critic()

    # each policy has an Adam optimizer
    optimizers = [optim.RMSprop(p.parameters(), lr=1e-4) for p in policies]
    cnn_optimizer = optim.RMSprop(cnn.parameters(), lr=1e-4)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=1e-4)

    discount_factor = 0.9

    target_pos = np.array((256, 256, 60))
    steps = 100

    # run multiple episodes, starting from random location
    for e in range(1000):

        bounds = data.shape # outer bounds for this image
        q = np.random.randint(bounds) # random location inside image

        history = np.zeros([steps, 3])

        #for each step sequence do
        for step in range(steps):
            if step%20 == 0: print(f'Episode {e}, step {step}')

            # loop through coordinates
            for coord in range(3):

                # apply policy and estimate value of current state
                prob, val0 = policies[coord](cnn(state(q, data)), critic = True)
                if SEPARATE_CRITIC:
                    val0 = critic(cnn(state(q, data)))

                # sample an action from policy distribution
                partial_a, log_prob = select_action(prob)
                a = torch.zeros(3)
                a[coord] = partial_a

                # move through the environment and receive reward
                q1, r = env(q, a, target_pos, bounds)

                # estimate value of new state
                if SEPARATE_CRITIC:
                    val1 = critic(cnn(state(q1, data)))
                else:
                    _, val1 = policies[coord](cnn(state(q1, data)), critic = True)

                # TD-target
                TD_target, TD_error = compute_target(r, discount_factor, val1, val0)

                # calculate loss
                policy_loss = -TD_error * log_prob
                value_loss = F.mse_loss(TD_target, val0)
                loss = policy_loss + value_loss

                # calculate the gradients and modify weights
                optimizers[coord].zero_grad()
                cnn_optimizer.zero_grad()
                if SEPARATE_CRITIC:
                    critic_optimizer.zero_grad()

                loss.backward()

                optimizers[coord].step()
                cnn_optimizer.step()
                if SEPARATE_CRITIC:
                    critic_optimizer.step()

                # record position in history
                history[step] = q

                q = q1

                # terminate epoch if we leave
                # if (r == OUT_OF_BOUNDS_REWARD): break
            # if (r == OUT_OF_BOUNDS_REWARD): break


        plot_history(history, data, target_pos, f'output\\episode{e}.png')