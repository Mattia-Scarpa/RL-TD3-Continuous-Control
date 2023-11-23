# RL Continuous Control
The main goal of this project is to train an agent for the continuous control of a robotic arm in a simulated environment.


# Project 2: Continuous Control

### Introduction

For this project it has been worked with the environment [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project it has been provided 2 version of the environment. For the next steps, and our algorithm design we will refer only to the single agent version.

### Instruction

The project has been developed exploiting a dedicated conda environment. In order to run it correctly it is strongly suggested to launch the command 

'./install.sh'

to install the correct environment with all the required package.

The project is built on:

* test_continuous_control.py -> The main file, the executable to run to see the project
* agent.py -> Agent Class
* model.py -> the NN model for the TD3 algorithm
* checkpoint_actor.pth -> the weights of the best policy model found during training
* checkpoint_critic.pth -> the weights of the best action-value function found during training
* report.md -> Project report of the overall implementation
* arm_control.mp4 -> The video of the final trained model for some episode