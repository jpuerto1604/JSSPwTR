import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =====================================================
# 1. Data Loading and Transport/Processing Times
# =====================================================

# Transport layout times (for different layouts)
df1 = pd.DataFrame(np.array([[0,6,8,10,12],[12,0,6,8,10],[10,6,0,6,8],[8,8,6,0,6],[6,10,8,6,0]]), columns=["LU","M1","M2",'M3',"M4"], index=["LU","M1","M2",'M3',"M4"])
df2 = pd.DataFrame(np.array([[0,4,6,8,6],[6,0,2,4,2],[8,12,0,2,4],[6,10,12,0,2],[4,8,10,12,0]]), columns=["LU","M1","M2",'M3',"M4"], index=["LU","M1","M2",'M3',"M4"])
df3 = pd.DataFrame(np.array([[0,2,4,10,12],[12,0,2,8,10],[10,12,0,6,8],[4,6,8,0,2],[2,4,6,12,0]]), columns=["LU","M1","M2",'M3',"M4"], index=["LU","M1","M2",'M3',"M4"])
df4 = pd.DataFrame(np.array([[0,4,8,10,14],[18,0,4,6,10],[20,14,0,8,6],[12,8,6,0,6],[14,14,12,6,0]]), columns=["LU","M1","M2",'M3',"M4"], index=["LU","M1","M2",'M3',"M4"])

# Load processing times and machine data from Excel
xls = pd.read_excel('/Users/julian/Documentos/Thesis/Data.xlsx', sheet_name='Macrodata', usecols='F:H, J:P, R:X')
data = pd.DataFrame(xls)
data.loc[:, 'nj'] = data.loc[:, 'nj'] + 1  # Adjust job numbers (increment by 1)
data = data.fillna('')

# Processing times and machine assignments
p_times = pd.DataFrame(data.iloc[:, :10].to_numpy(), columns=["Set", "Job", "nj", "P1", "P2", "P3", "P4", "P5", "P6", "P7"])
m_data = pd.DataFrame(data.iloc[:, [0, 1, 2] + list(range(10, data.shape[1]))].to_numpy(), 
                      columns=["Set", "Job", "nj"] + [f"M{i}" for i in range(1, data.shape[1] - 9)])

# Function to retrieve transport times between locations
def t_times(layout, start, end):
    match layout:
        case 1:
            return df1.loc[start, end]
        case 2:
            return df2.loc[start, end]
        case 3:
            return df3.loc[start, end]
        case 4:
            return df4.loc[start, end]

# Retrieve job data based on a specific set
def jobs(nset):
    return m_data[m_data['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

# Retrieve processing times based on a specific set
def processing(nset):
    return p_times[p_times['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

# =====================================================
# 2. Optimized Job Shop Scheduling Environment
# =====================================================

class JobShopEnv:
    def __init__(self, layout, nset, jobs_data, processing_data):
        """
        Optimized environment class.
        - jobs_data: Preloaded job data to avoid redundant loading during training.
        - processing_data: Preloaded processing times.
        """
        self.layout = layout
        self.nset = nset
        self.jobs_data = jobs_data
        self.processing_data = processing_data
        self.current_time = 0
        self.done = False
        self.machine_status = np.zeros(5)  # Assuming 5 machines
        self.transport_status = np.zeros(5)  # Assuming 5 transports

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.current_time = 0
        self.done = False
        self.machine_status.fill(0)
        self.transport_status.fill(0)
        return self._get_state()

    def step(self, action):
        """
        Execute the given action and update the environment state.
        """
        job, machine, transport = action
        process_time = self.processing_data.iloc[job - 1, machine + 2]
        transport_time = t_times(self.layout, "M1", f"M{machine}")
        self.current_time += process_time + transport_time
        reward = -self.current_time  # Reward is negative to minimize makespan
        self.done = self._check_done()
        next_state = self._get_state()
        return next_state, reward, self.done

    def _get_state(self):
        """
        Return the current state (machine and transport status).
        """
        state = np.concatenate([self.machine_status, self.transport_status])
        return state

    def _check_done(self):
        """
        Check if all jobs are completed.
        """
        return np.all(self.machine_status == 0)

# =====================================================
# 3. Optimized Deep Q-Learning (DQN) Model
# =====================================================

class DQNScheduler(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Optimized neural network model for resource efficiency.
        - Reduced the number of neurons in the hidden layers to minimize memory usage.
        """
        super(DQNScheduler, self).__init__()

        # Reduced number of neurons in the hidden layers
        self.fc1 = nn.Linear(input_dim, 64)  # Reduced from 128 to 64 neurons
        self.fc2 = nn.Linear(64, 32)  # Reduced from 64 to 32 neurons
        self.fc3 = nn.Linear(32, output_dim)  # Output layer (unchanged)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the optimized network.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Output layer (Q-values for each action)

# =====================================================
# 4. Optimized Replay Buffer
# =====================================================

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Replay buffer to store past experiences.
        - Reduced capacity to optimize memory usage.
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        """
        Store a new experience in the buffer.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

# =====================================================
# 5. Optimized Training Loop and Batch Updates
# =====================================================

# Check if MPS is available for the M1 GPU, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the dimensions of the input and output for the DQN model
state_dim = 10  # Adjust according to the environment state size
action_dim = 5  # Output dimension (number of possible actions)

# Initialize the DQN model and optimizer
dqn = DQNScheduler(state_dim, action_dim).to(device)

# Optimizer using Adam (Learning rate can be adjusted if needed)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)

# Initialize replay buffer with a reduced size to optimize memory
replay_buffer = ReplayBuffer(5000)  # Reduced from 10000

# Hyperparameters
batch_size = 32  # Reduced from 64
gamma = 0.99  # Discount factor for future rewards
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Rate at which exploration decreases

def train_dqn_batch(dqn, replay_buffer, batch_size, gamma):
    """
    Train the DQN model using batch updates from the replay buffer.
    - dqn: The Deep Q-Network model
    - replay_buffer: The buffer storing past experiences
    - batch_size: Number of experiences to sample from the buffer
    - gamma: The discount factor for future rewards
    """
    if replay_buffer.size() < batch_size:
        return  # Don't train until we have enough experiences in the buffer

    # Sample a batch of experiences from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert the experiences to PyTorch tensors and move to the device
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Compute Q-values for the current states
    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute the target Q-values for the next states
    with torch.no_grad():
        next_q_values = dqn(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute the loss (Mean Squared Error) between current Q-values and target Q-values
    loss = nn.MSELoss()(q_values, target_q_values)

    # Perform the backward pass to compute gradients and update the model's weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =====================================================
# 6. Optimized Simulation and Training with Early Stopping
# =====================================================

# Preload job and processing data for all sets
all_jobs_data = {nset: jobs(nset) for nset in range(1, 11)}  # 10 sets
all_processing_data = {nset: processing(nset) for nset in range(1, 11)}  # 10 sets

num_episodes = 200  # Reduced from 500 to save time
early_stopping_threshold = -100  # Threshold for early stopping

# Loop over layouts and sets
for layout in range(1, 5):  # 4 layouts
    for nset in range(1, 11):  # 10 sets
        print(f"Training for Layout {layout} and Set {nset}")
        
        # Initialize environment for the current layout and set
        jobs_data = all_jobs_data[nset]
        processing_data = all_processing_data[nset]
        env = JobShopEnv(layout, nset, jobs_data, processing_data)

        for episode in range(num_episodes):
            # Reset environment for new episode
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Epsilon-greedy action selection (exploration vs exploitation)
                if np.random.rand() <= epsilon:
                    action = random.choice([0, 1, 2, 3, 4])  # Random action (exploration)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    q_values = dqn(state_tensor)
                    action = torch.argmax(q_values).item()  # Best action (exploitation)

                next_state, reward, done = env.step((1, action, action))

                # Store the experience in the replay buffer
                replay_buffer.store((state, action, reward, next_state, done))

                # Train the DQN model with batch updates
                train_dqn_batch(dqn, replay_buffer, batch_size, gamma)

                # Update the current state
                state = next_state
                episode_reward += reward

            # Epsilon decay after each episode
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Early stopping if reward threshold is met
            if episode_reward > early_stopping_threshold:
                print(f"Early stopping at episode {episode} for Layout {layout} and Set {nset}")
                break

        # Optional: Save the model after training each set-layout combination
        torch.save(dqn.state_dict(), f'model_layout{layout}_set{nset}.pth')
