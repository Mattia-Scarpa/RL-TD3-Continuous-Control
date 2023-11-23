## Environment and Agent

The first step is to load the environment built on unity. Then we extract 'brain' information, which is the responsible of taking action in the enviroment.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The first step brefore to train our agent is to reset the environment, namely placing it in a random point in its available square space. The final goal for the agent will be to track the reference sphere for as much as possibile (which will provide a reward of +0.1), while avoiding other random movement (which provide no reward).

The environment will be considered solved if the agent will collect at least an average overall reaward &ge;30 along 100 episode.

## Agent training
In our analysis is mandatory to recall that the choice of our algorithm, namely Twin Delayed DDPG, is, as the name suggest an evolution of the DDPG algorithm. The latter suffers of overestimation issues for the **Q**-value function, and consequently our algorithm will follow the same fate.

To overcome this issues some trick has been adopted.
While explaining them however it is interesting also to explain a bit more in details the learning algorithm.

### Learning Algorithm
Twin Delayed Deep Deterministic Policy Gradient (TD3) is a RL algorithm for continuous control algorithm. It aims to improve the classical DDPG algorithm with 3 main variations:

1. Double **Q**-network: To improve action value overestimation the critic model exploit 2 different network and use the minimun value to compute the Q value target: $Q_{target} = min_{i=1,2}Q_{\theta_i'}(s', clip(\pi_{\pi'}(s')+\epsilon), a_{low}, a_{high})$ where $\epsilon$ is an additive noise used to induce exploration of the action.

2. In order to avoid the policy variance the actor training is delayed with respect the critic. This allow to collect a bit more experience under the same policy helping to reduce the variance.

3. Exploration Noise: As said in point $1.$ at the policy target an additive noise is added. This noise, clipped according to action range allowed, help the network to not get stucked in earlier suboptimal policy.

The weigths update is based on the loss computation among the current Q estimated and the Q target, updating consequently the network according to the gradient descent:

$\theta_i \leftarrow$