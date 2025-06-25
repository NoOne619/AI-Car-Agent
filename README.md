AI Car TORCS Agent Using Q-Learning
Overview
This project implements an AI agent for lane-keeping in The Open Racing Car Simulator (TORCS) using Q-learning, a model-free reinforcement learning algorithm. The agent learns to control a car to stay on the track by optimizing actions (steering, acceleration, and braking) based on sensor inputs and a reward function. The implementation is designed for research and experimentation in autonomous driving within the TORCS environment.
The agent uses a Q-table to store state-action values, discretizing the state space to manage the high-dimensional sensory input from TORCS. The goal is to keep the car on the track, avoid getting stuck, and maximize the distance traveled.

Features

Q-Learning Algorithm: The agent learns an optimal policy by updating Q-values based on rewards and state transitions.
TORCS Integration: Communicates with the TORCS simulator via a client-server model using UDP connections.
State Representation: Utilizes key sensor inputs such as track position, car angle, speed, and opponent sensors (optional).
Action Space: Controls steering, acceleration, and braking (braking treated as negative acceleration).
Reward Function: Encourages staying on the track and penalizes going off-track or getting stuck.
Stuck Detection: Terminates episodes if the car is stuck (angle > 45° for > 25 ticks or minimal distance traveled).


Prerequisites
To run this project, ensure you have the following installed:

TORCS: The Open Racing Car Simulator (tested with version 1.3.6 or later).
Python: Version 3.6+.
Dependencies:
numpy for numerical computations.
gym_torcs (optional, for a Gym-like interface, if used).
Other dependencies listed in requirements.txt.



Install dependencies using:
pip install -r requirements.txt


Installation

Install TORCS:

Download and install TORCS from http://torcs.sourceforge.net/.
Ensure the scr_server is configured for client-server communication (refer to TORCS documentation).


Clone the Repository:
git clone <your-repository-url>
cd <repository-directory>


Set Up the Environment:

If using gym_torcs, follow the setup instructions from https://github.com/ugo-nama-kun/gym_torcs.
Alternatively, configure the TORCS server manually:
Run wtorcs.exe (Windows) or torcs (Linux) from the torcs_server directory.
Navigate to Race > Practice > Configure Race, select a track, and ensure scr_server 1 is selected as the client.
Start the race by selecting New Race.




Install Python Dependencies:
pip install -r requirements.txt




Usage

Start the TORCS Server:

Launch TORCS and configure the race as described above.
The server will wait for a client connection.


Run the Agent:

Execute the main script to start the Q-learning agent:python main.py


The agent will connect to the TORCS server, receive sensory inputs, and send control actions (steering, acceleration/braking).


Training:

The agent trains by interacting with the environment, updating the Q-table based on rewards.
Training parameters (e.g., learning rate, discount factor, epsilon for exploration) can be adjusted in config.py or the main script.


Testing:

After training, the agent can be tested by setting epsilon=0 to disable exploration and use the learned policy.
Run:python main.py --test






Implementation Details

State Space:

Key features: track position (distance from track axis), car angle, speed, and optionally opponent proximity (from 36 sensors, reduced to a single max value for efficiency).
States are discretized (e.g., 16 discrete values per feature, represented in 4 bits) to manage the Q-table size.


Action Space:

Actions include:
Steering: Continuous or discretized values between [-1, 1].
Acceleration/Braking: Treated as a single dimension (positive for acceleration, negative for braking).
Gear: Optional, typically fixed for simplicity.


Example action selection uses an epsilon-greedy policy.


Reward Function:

Positive reward for traveling a long distance on the track (range [-1, 1]).
Negative reward (-1) if the car goes off-track (abs(track position) > 1).
Episode terminates if the car is stuck (angle > 45° for > 25 ticks or distance < 0.01m).


Q-Learning Parameters:

Learning Rate (α): Controls the update rate of Q-values.
Discount Factor (γ): Balances immediate vs. future rewards.
Exploration (ε): Epsilon-greedy exploration, decaying over time.
Example values: α=0.1, γ=0.9, ε starts at 1.0 and decays to 0.01.


Game Loop:

Each tick, the server requests an action via the drive function.
The agent checks for stuck conditions and selects an action using the Q-table.
The Q-table is updated using the reward and next state.


Q-Table Management:

New states are initialized with zero Q-values for all actions.
If multiple actions have the same Q-value, a heuristic or random selection is used.




File Structure
├── main.py                 # Main script to run the agent
├── agent.py               # Q-learning agent implementation
├── config.py              # Configuration for hyperparameters
├── requirements.txt       # Python dependencies
├── torcs_server/          # TORCS server files (e.g., wtorcs.exe)
├── data/                  # Optional: Saved Q-table or training logs
└── README.md              # This file


Training Tips

Hyperparameter Tuning: Adjust α, γ, and ε decay in config.py to balance exploration and exploitation.
State Discretization: Reduce the number of discrete states to avoid an infeasibly large Q-table (e.g., use a single sensor instead of all 36 opponent sensors).
Track Selection: Start with simpler tracks (e.g., oval tracks) for faster learning.
Exploration Strategy: Use epsilon-greedy with decay or consider heuristic exploration for new states.


Limitations

Scalability: Q-learning with a Q-table struggles with high-dimensional state spaces. For complex tracks or visual inputs, consider Deep Q-Networks (DQN).
Training Time: Requires significant episodes for convergence, especially with large state spaces.
Generalization: The agent may overfit to specific tracks; transfer learning or multi-track training can help.


Future Improvements

Integrate Deep Q-Networks (DQN) for handling continuous or high-dimensional inputs like visual data.
Experiment with Double DQN or Dueling DQN to improve stability and performance.
Incorporate experience replay to enhance sample efficiency.
Add support for multi-track training or racing against opponents.
Explore advanced exploration strategies (e.g., Boltzmann exploration or Bootstrapped DQN).


References

TORCS: http://torcs.sourceforge.net/
Q-Learning: https://en.wikipedia.org/wiki/Q-learning
Gym-TORCS: https://github.com/ugo-nama-kun/gym_torcs
Inspired by: https://github.com/A-Raafat/Torcs---Reinforcement-Learning-using-Q-Learning


Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug fixes, improvements, or new features.

License
This project is licensed under the MIT License. See the LICENSE file for details.
