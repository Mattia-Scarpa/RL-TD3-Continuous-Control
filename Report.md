## Environment and Agent

The first step is to load the environment built on unity. Then we extract 'brain' information, which is the responsible of taking action in the enviroment.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The first step brefore to train our agent is to reset the environment, namely placing it in a random point in its available square space. The final goal for the agent will be to track the reference sphere for as much as possibile (which will provide a reward of +0.1), while avoiding other random movement (which provide no reward).

The environment will be considered solved if the agent will collect at least an average overall reaward &ge;30 along 100 episode.