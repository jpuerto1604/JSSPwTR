import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Data Loading and Transport/Processing Times

# Define transport times between locations for different layouts
df1 = pd.DataFrame(np.array([
    [0, 6, 8, 10, 12],
    [12, 0, 6, 8, 10],
    [10, 6, 0, 6, 8],
    [8, 8, 6, 0, 6],
    [6, 10, 8, 6, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

df2 = pd.DataFrame(np.array([
    [0, 4, 6, 8, 6],
    [6, 0, 2, 4, 2],
    [8, 12, 0, 2, 4],
    [6, 10, 12, 0, 2],
    [4, 8, 10, 12, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

df3 = pd.DataFrame(np.array([
    [0, 2, 4, 10, 12],
    [12, 0, 2, 8, 10],
    [10, 12, 0, 6, 8],
    [4, 6, 8, 0, 2],
    [2, 4, 6, 12, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

df4 = pd.DataFrame(np.array([
    [0, 4, 8, 10, 14],
    [18, 0, 4, 6, 10],
    [20, 14, 0, 8, 6],
    [12, 8, 6, 0, 6],
    [14, 14, 12, 6, 0]]),
    columns=["LU", "M1", "M2", "M3", "M4"], index=["LU", "M1", "M2", "M3", "M4"])

# Load processing times and machine assignments from Excel
# Adjust the file path as needed
xls = pd.read_excel('/Users/julian/Documentos/Thesis/Data.xlsx', sheet_name='Macrodata', usecols='F:H, J:P, R:X')
data = pd.DataFrame(xls)
data.loc[:, 'nj'] = data.loc[:, 'nj'] + 1  # Adjust job numbers
data = data.fillna('')

# Extract processing times and machine assignments
p_times = pd.DataFrame(data.iloc[:, :10].to_numpy(), columns=["Set", "Job", "nj", "P1", "P2", "P3", "P4", "P5", "P6", "P7"])
m_data = pd.DataFrame(data.iloc[:, [0, 1, 2] + list(range(10, data.shape[1]))].to_numpy(),columns=["Set", "Job", "nj"] + [f"M{i}" for i in range(1, data.shape[1] - 9)])


def t_times(layout, start, end):
    '''Parameters:
    - layout: The layout number for transport times (1 to 4)
    - start: Starting location
    - end: Destination

    Returns:
    -The transport time between the start and end locations for the given layout.
    '''
    if layout == 1:
        return df1.loc[start, end]
    elif layout == 2:
        return df2.loc[start, end]
    elif layout == 3:
        return df3.loc[start, end]
    elif layout == 4:
        return df4.loc[start, end]
    else:
        raise ValueError("Invalid layout number.")
def jobs(nset):
    '''
    Parameters:
    -nset: the specific job set being used

    Returns:
    -The job data for the given set, including the job number and machine assignments.
    '''
    return m_data[m_data['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

def processing(nset):
    '''
    Parameters:
    -nset: the specific job set being used

    Returns:
    -The processing times for the given set, including the job number and processing times for each machine.
    '''
    return p_times[p_times['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

# Job Shop Scheduling Environment with Time Tracking

class JobShopEnv:
    def __init__(self, layout, nset, jobs_data, processing_data, num_agvs):
        """
        Job Shop Scheduling environment with 4 machines (M1 to M4) and LU (Loading-Unloading area).
        Supports multiple AGVs (from 1 to 5) for transport.
        Tracks start and end times for each job on each machine dynamically.

        Parameters:
        - layout: The layout number for transport times (1 to 4).
        - nset: The specific job set being used.
        - jobs_data: DataFrame containing job sequences (including LU).
        - processing_data: DataFrame containing processing times for each job and machine.
        - num_agvs: Number of AGVs available for transport (from 1 to 5).
        """
        self.layout = layout
        self.nset = nset
        self.jobs_data = jobs_data
        self.processing_data = processing_data
        self.num_agvs = num_agvs  # Number of AGVs available
        self.current_time = 0
        self.done = False
        self.machine_status = np.zeros(4)  # Status of machines M1 to M4; LU is always available
        self.agv_status = np.zeros(num_agvs)  # Status of each AGV (0 for free, 1 for busy)
        self.agv_locations = ["LU"] * num_agvs  # Current location of each AGV

        # Track remaining times for each job at each machine in its sequence
        self.job_times = {job_id: np.zeros(len(self.jobs_data.iloc[job_id, 2:].dropna())) for job_id in range(len(self.jobs_data))}

        # Dictionary to track which machine each job needs to go to next in its sequence
        self.job_next_machine = {job_id: 0 for job_id in range(len(self.jobs_data))}

        # Data Structure for Time Tracking
        # Dictionary to store start and end times for each job on each machine
        # Structure: {job_id: {machine: {'start': start_time, 'end': end_time}}}
        self.job_machine_times = {job_id: {} for job_id in range(len(self.jobs_data))}

        # Current location of each job (starting at LU)
        self.job_locations = {job_id: "LU" for job_id in range(len(self.jobs_data))}

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - state: The initial state of the environment.
        """
        self.current_time = 0
        self.done = False
        self.machine_status.fill(0)  # All machines (M1 to M4) start free
        self.agv_status.fill(0)      # All AGVs start free
        self.agv_locations = ["LU"] * self.num_agvs  # All AGVs at LU

        # Reset job-machine assignment tracking and job times
        self.job_next_machine = {job_id: 0 for job_id in range(len(self.jobs_data))}
        self.job_times = {job_id: np.zeros(len(self.jobs_data.iloc[job_id, 2:].dropna())) for job_id in range(len(self.jobs_data))}
        self.job_locations = {job_id: "LU" for job_id in range(len(self.jobs_data))}

        # Reset the job-machine times tracking
        self.job_machine_times = {job_id: {} for job_id in range(len(self.jobs_data))}

        return self._get_state()

    def step(self, action):
        """
        Execute the given action in the environment.

        Parameters:
        - action: A tuple (job, machine), assigning a job to a machine.

        Returns:
        - next_state: The updated state after the action.
        - reward: The reward obtained from taking the action.
        - done: A boolean indicating if all jobs are completed.
        """
        job, machine = action  # The action specifies the job and the machine (or LU)

        # Get the job's machine sequence and the correct machine it needs to go to
        job_sequence = self.jobs_data.iloc[job, 2:].dropna().tolist()
        next_machine_index = self.job_next_machine[job]
        if next_machine_index >= len(job_sequence):
            # Job has already completed its sequence
            return self._get_state(), 0, self._check_done()
        correct_machine = job_sequence[next_machine_index]

        # Ensure the job is assigned to the correct machine (or LU)
        if correct_machine != machine:
            raise ValueError(f"Job {job} cannot be assigned to machine {machine}. It should go to machine {correct_machine}.")

        if machine != "LU":
            # Check for an available AGV
            available_agv = self._find_available_agv()
            if available_agv is None:
                raise ValueError("No AGVs are currently available.")

            # Check if the machine is available
            machine_index = int(machine[1]) - 1  # Convert 'M1' to index 0
            if self.machine_status[machine_index] == 1:  # If the machine is busy
                raise ValueError(f"Machine {machine} is currently busy.")

            # Get the current location of the AGV
            agv_location = self.agv_locations[available_agv]

            # Get the current location of the job
            job_location = self.job_locations[job]

            # Calculate transport time from AGV's location to the job's location (if different)
            if agv_location != job_location:
                empty_move_time = t_times(self.layout, agv_location, job_location)
            else:
                empty_move_time = 0

            # Calculate transport time from the job's location to the destination machine
            loaded_move_time = t_times(self.layout, job_location, machine)

            transport_time = empty_move_time + loaded_move_time

            # Mark the machine and AGV as busy
            self.machine_status[machine_index] = 1
            self.agv_status[available_agv] = 1  # Mark the AGV as busy

            # Update the AGV's location
            self.agv_locations[available_agv] = machine

            # Update the job's location
            self.job_locations[job] = machine

            # Record the start time for the job on the machine
            start_time = self.current_time + transport_time

            # Get the processing time for this job-machine combination
            process_time = self.processing_data.iloc[job, 3 + next_machine_index]  # Adjust index as needed

            # Update the job's remaining time on this machine
            self.job_times[job][next_machine_index] = process_time

            # Update the current time
            total_time = transport_time + process_time
            self.current_time += total_time

            # Record the end time for the job on the machine
            end_time = start_time + process_time

            # Store the start and end times
            self.job_machine_times[job][machine] = {'start': start_time, 'end': end_time}

            # Negative reward to minimize makespan
            reward = -total_time

            # Mark this job's machine assignment as completed
            self.job_next_machine[job] += 1

            # Update availability after processing
            self.machine_status[machine_index] = 0  # Free the machine
            self.agv_status[available_agv] = 0      # Free the AGV

        else:
            # If the machine is LU
            # Check for an available AGV
            available_agv = self._find_available_agv()
            if available_agv is None:
                raise ValueError("No AGVs are currently available.")

            # Get the current location of the AGV
            agv_location = self.agv_locations[available_agv]

            # Get the current location of the job
            job_location = self.job_locations[job]

            # Calculate transport time from AGV's location to the job's location (if different)
            if agv_location != job_location:
                empty_move_time = t_times(self.layout, agv_location, job_location)
            else:
                empty_move_time = 0

            # Calculate transport time from the job's location to LU
            loaded_move_time = t_times(self.layout, job_location, "LU")

            transport_time = empty_move_time + loaded_move_time

            # Update the AGV's location
            self.agv_locations[available_agv] = "LU"

            # Update the job's location
            self.job_locations[job] = "LU"

            # Record the start time for the job at LU
            start_time = self.current_time + transport_time

            # No processing time at LU
            process_time = 0

            # Update the current time
            total_time = transport_time + process_time
            self.current_time += total_time

            # Record the end time for the job at LU
            end_time = start_time + process_time

            # Store the start and end times
            self.job_machine_times[job][machine] = {'start': start_time, 'end': end_time}

            # Negative reward to minimize makespan
            reward = -total_time

            # Mark this job's machine assignment as completed
            self.job_next_machine[job] += 1

            # Free the AGV
            self.agv_status[available_agv] = 0

        # Check if all jobs are finished
        self.done = self._check_done()

        # Get the next state of the system
        next_state = self._get_state()
        return next_state, reward, self.done

    def _find_available_agv(self):
        """
        Find an available AGV.

        Returns:
        - Index of an available AGV, or None if all are busy.
        """
        for idx, status in enumerate(self.agv_status):
            if status == 0:  # AGV is free
                return idx
        return None  # No AGV is available

    def _get_state(self):
        """
        Return the current state of the environment.

        State includes:
        - Machine statuses (M1 to M4).
        - AGV statuses.
        - Job-machine times for each job.

        Returns:
        - state: A numpy array representing the current state.
        """
        # Combine machine statuses and AGV statuses
        state = np.concatenate([self.machine_status, self.agv_status])

        # Add the job-machine times for all jobs
        for job_id in range(len(self.jobs_data)):
            state = np.concatenate([state, self.job_times[job_id]])

        # Add AGV locations
        agv_location_indices = [machine_to_index(loc) for loc in self.agv_locations]
        state = np.concatenate([state, agv_location_indices])

        # Add job locations
        job_location_indices = [machine_to_index(self.job_locations[job_id]) for job_id in range(len(self.jobs_data))]
        state = np.concatenate([state, job_location_indices])

        return state

    def _check_done(self):
        """
        Check if all jobs have completed their sequences.
        
        Returns:
        - done: True if all jobs are completed, False otherwise.
        """
        return all(self.job_next_machine[job] == len(self.jobs_data.iloc[job, 2:].dropna()) for job in range(len(self.jobs_data)))

# Deep Q-Network (DQN) Model

class DQNScheduler(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Deep Q-Network model to estimate Q-values for each action.

        Parameters:
        - input_dim: The number of features in the state representation.
        - output_dim: The number of possible actions (job-machine pairings).
        """
        super(DQNScheduler, self).__init__()

        # Neural network layers
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer with 64 neurons
        self.fc3 = nn.Linear(64, output_dim)  # Output layer to predict Q-values for actions

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the DQN.

        Parameters:
        - x: Input state tensor.

        Returns:
        - Output tensor with Q-values for each action.
        """
        x = self.relu(self.fc1(x))  # First hidden layer
        x = self.relu(self.fc2(x))  # Second hidden layer
        return self.fc3(x)          # Output layer

# Replay Buffer for Experience Storage

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Replay buffer to store experiences during training.

        Parameters:
        - capacity: Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        """
        Store a new experience in the buffer.

        Parameters:
        - experience: A tuple (state, action, reward, next_state, done).
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Parameters:
        - batch_size: Number of experiences to sample.

        Returns:
        - Tuple of arrays: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        """
        Return the current size of the buffer.

        Returns:
        - Integer representing the number of experiences stored.
        """
        return len(self.buffer)

# Training the DQN with Batch Updates

def train_dqn_batch(dqn, replay_buffer, batch_size, gamma, optimizer):
    """
    Train the DQN model using batch updates from the replay buffer.

    Parameters:
    - dqn: The Deep Q-Network model.
    - replay_buffer: The buffer storing past experiences.
    - batch_size: Number of experiences to sample from the buffer.
    - gamma: Discount factor for future rewards.
    - optimizer: Optimizer for updating the DQN's weights.
    """
    if replay_buffer.size() < batch_size:
        return  # Skip training if not enough experiences in the buffer

    # Sample a batch of experiences from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert the experiences to PyTorch tensors and move them to the device (MPS or CPU)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    # Convert complex actions (job, machine) to indices if necessary
    # For simplicity, we can encode actions as integers
    action_indices = torch.tensor([action[0] * 5 + machine_to_index(action[1]) for action in actions], dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Compute Q-values for the current states
    q_values = dqn(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)

    # Compute the target Q-values for the next states
    with torch.no_grad():
        next_q_values = dqn(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute the loss (MSE between current Q-values and target Q-values)
    loss = nn.MSELoss()(q_values, target_q_values)

    # Backpropagation to update the model's weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Helper function to convert machine names to indices
def machine_to_index(machine):
    machine_list = ['LU', 'M1', 'M2', 'M3', 'M4']
    return machine_list.index(machine)

# Modify the action selection to consider the availability of AGVs and machines
def select_action(state, env, dqn, epsilon):
    if np.random.rand() <= epsilon:
        # Exploration: randomly choose a valid action
        valid_actions = []
        for job in range(len(env.jobs_data)):
            next_machine_index = env.job_next_machine[job]
            job_sequence = env.jobs_data.iloc[job, 2:].dropna().tolist()
            if next_machine_index >= len(job_sequence):
                continue  # Skip if job is already completed
            machine = job_sequence[next_machine_index]
            # Optionally check AGV and machine availability here
            valid_actions.append((job, machine))
        if valid_actions:
            return random.choice(valid_actions)
        else:
            return None
    else:
        # Exploitation: choose the best action based on the current policy
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = dqn(state_tensor)
        sorted_indices = torch.argsort(q_values, descending=True)
        for action_idx in sorted_indices.cpu().numpy():
            job, machine = decode_action(action_idx)
            # Check if the action is valid
            next_machine_index = env.job_next_machine.get(job, None)
            if next_machine_index is None:
                continue
            job_sequence = env.jobs_data.iloc[job, 2:].dropna().tolist()
            if next_machine_index >= len(job_sequence):
                continue
            correct_machine = job_sequence[next_machine_index]
            if machine != correct_machine:
                continue
            # Check availability of AGV and machine
            if machine != "LU":
                if env._find_available_agv() is None:
                    continue
                machine_index = int(machine[1]) - 1
                if env.machine_status[machine_index] == 1:
                    continue
            return (job, machine)
        return None  # No valid actions

def encode_action(job, machine):
    machine_idx = machine_to_index(machine)
    return job * 5 + machine_idx

def decode_action(action_idx):
    job = action_idx // 5
    machine_idx = action_idx % 5
    machine_list = ['LU', 'M1', 'M2', 'M3', 'M4']
    machine = machine_list[machine_idx]
    return job, machine


# Simulation and Training Loop with Epsilon-Greedy Policy

# Device configuration (use MPS if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Number of AGVs to test (from 1 to 5)
for num_agvs in range(1, 6):
    print(f"Training with {num_agvs} AGVs")
    # Preload job and processing data for all sets (avoiding repeated loading)
    all_jobs_data = {nset: jobs(nset) for nset in range(1, 11)}        # 10 sets
    all_processing_data = {nset: processing(nset) for nset in range(1, 11)}  # 10 sets

    num_episodes = 500  # Number of episodes per layout/set combination
    early_stopping_threshold = -100  # Threshold for early stopping based on reward

    # Initialize performance metrics storage
    performance_metrics = {}

    # Loop through all layouts and sets
    for layout in range(1, 5):  # 4 layouts
        for nset in range(1, 11):  # 10 sets
            print(f"Training for Layout {layout}, Set {nset}, with {num_agvs} AGVs")

            # Retrieve jobs and processing data
            jobs_data = all_jobs_data[nset]
            processing_data = all_processing_data[nset]
            num_jobs = len(jobs_data)
            steps_per_job = len(jobs_data.iloc[0, 2:].dropna())

            # Calculate state and action dimensions
            state_dim = (
                4 +  # Machine statuses
                num_agvs +  # AGV statuses
                num_agvs +  # AGV locations (indices)
                num_jobs * steps_per_job +  # Job-machine times
                num_jobs  # Job locations (indices)
            )
            action_dim = num_jobs * 5  # Each job can go to 5 locations (LU, M1 to M4)

            # Initialize DQN model and optimizer
            dqn = DQNScheduler(state_dim, action_dim).to(device)
            optimizer = optim.Adam(dqn.parameters(), lr=0.001)  # Adjust learning rate if needed

            # Initialize replay buffer
            replay_buffer = ReplayBuffer(5000)  # Adjust capacity if needed

            # Initialize environment
            env = JobShopEnv(layout, nset, jobs_data, processing_data, num_agvs)

            # Hyperparameters
            batch_size = 32
            gamma = 0.99
            epsilon = 1.0
            epsilon_min = 0.1
            epsilon_decay = 0.995

            total_rewards = []

            # Training loop
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action = select_action(state, env, dqn, epsilon)
                    if action is None:
                        # No valid actions available, proceed accordingly
                        continue

                    try:
                        next_state, reward, done = env.step(action)
                    except ValueError as e:
                        # Invalid action, assign a large negative reward
                        reward = -1000
                        next_state = state
                        done = False

                    # Store experience and train
                    replay_buffer.store((state, action, reward, next_state, done))
                    train_dqn_batch(dqn, replay_buffer, batch_size, gamma, optimizer)

                    state = next_state
                    episode_reward += reward

                # Epsilon decay after each episode
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                total_rewards.append(episode_reward)

                # Early stopping if reward threshold is met
                if episode_reward > early_stopping_threshold:
                    print(f"Early stopping at episode {episode} for Layout {layout}, Set {nset}, AGVs {num_agvs}")
                    break

            # Store performance metrics
            performance_metrics[(layout, nset, num_agvs)] = {
                "average_reward": np.mean(total_rewards),
                "total_time": env.current_time,
                "job_machine_times": env.job_machine_times  # Include the start and end times
            }

            # Save the model after training each set-layout combination
            torch.save(dqn.state_dict(), f'model_layout{layout}_set{nset}_agvs{num_agvs}.pth')

    # Analyze the performance metrics
    for key, metrics in performance_metrics.items():
        layout, nset, agvs = key
        print(f"Layout: {layout}, Set: {nset}, AGVs: {agvs}, "
              f"Average Reward: {metrics['average_reward']}, Total Time: {metrics['total_time']}")

        # Access the start and end times
        job_machine_times = metrics['job_machine_times']
        for job_id, machine_times in job_machine_times.items():
            print(f"Job {job_id}:")
            for machine, times in machine_times.items():
                print(f"  Machine {machine}: Start at {times['start']}, End at {times['end']}")