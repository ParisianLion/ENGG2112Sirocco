import numpy as np
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from SimConnect import *
import time


# Define constants
STATE_SIZE = 9  # position_x, position_y, velocity, angle
ACTION_SIZE = 22  # move left, move right, go forward, go backward
EPISODES = 1000
REPLAY_MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.95  # Discount factor

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model = self._build_model()
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=STATE_SIZE, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)  # Explore
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create a connection to the simulator
simconnect = SimConnect()

# Define the aircraft variables
class Aircraft:
    def __init__(self, simconnect):
        self.simconnect = simconnect
        self.data = AircraftData()
    
    def get_position_x(self):
        return self.data.Position.X
    
    def get_position_y(self):
        return self.data.Position.Y
    
    def get_position_z(self):
        return self.data.Position.Z
    
    def get_velocity(self):
        return self.data.Velocity.X, self.data.Velocity.Y, self.data.Velocity.Z
    
    def get_pitch(self):
        return self.data.Pitch
    
    def get_roll(self):
        return self.data.Roll
    
    def get_yaw(self):
        return self.data.Yaw
    
    def get_altitude(self):
        return self.data.Altitude
    
    def get_heading(self):
        return self.data.Heading
    
    def update(self):
        self.simconnect.call(REQUESTS.AIRCRAFT_DATA, self.data)

class AircraftData:
    # Define the structure for the aircraft data
    def __init__(self):
        self.Position = PositionData()
        self.Velocity = VelocityData()
        self.Pitch = 0.0
        self.Roll = 0.0
        self.Yaw = 0.0
        self.Altitude = 0.0
        self.Heading = 0.0

class PositionData:
    def __init__(self):
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0

class VelocityData:
    def __init__(self):
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0

# Initialize the aircraft object
aircraft = Aircraft(simconnect)

def get_current_state():
    # Update the aircraft data from the simulator
    aircraft.update()
    
    # Retrieve relevant state variables from the simulator
    position_x = aircraft.get_position_x()
    position_y = aircraft.get_position_y()
    position_z = aircraft.get_position_z()
    velocity = aircraft.get_velocity()
    pitch = aircraft.get_pitch()
    roll = aircraft.get_roll()
    yaw = aircraft.get_yaw()
    altitude = aircraft.get_altitude()
    heading = aircraft.get_heading()

def send_action(action_id):
    # Define your actions here based on the action_id
    if action_id == 0:  # Example: Move Left
        aircraft.set_control('AILERON_TRIM_LEFT', 1)
    elif action_id == 1:  # Move Right
        aircraft.set_control('AILERON_TRIM_RIGHT', 1)
    elif action_id == 2:  # Go Forward
        aircraft.set_control('THROTTLE', 1)  # Full throttle
    elif action_id == 3:  # Go Backward
        aircraft.set_control('THROTTLE', -1)  # Reverse throttle
    # Add more actions based on the action ID
    # ...

def send_velocities(xvel, yvel, zvel):
    # Implement the function to send the velocities to MSFS
    pass

def simulate_environment(action_id, velocity_index):
    # Get the velocities from the pre-made matrix
    xvel, yvel, zvel = velocity_matrix[velocity_index]

    # Send velocities to MSFS
    send_velocities(xvel, yvel, zvel)

    # Send action to MSFS
    send_action(action_id)

    # Wait briefly to allow the command to take effect
    time.sleep(0.1)

    # Get the new state from the simulator
    current_state = get_current_state()

    #get new xvel, yvel, zvel

    # Define a reward structure (this will need to be more complex based on your application)
    reward = 0  # Compute your reward based on the new state or goals
    done = False  # Define your termination condition based on the state

    return current_state, reward, done

def main():
    agent = DQNAgent()
    for episode in range(EPISODES):
        state = get_current_state()  # Initial state
        state = np.reshape(state, [1, STATE_SIZE])
        
        for time_step in range(200):  # Max time steps per episode
            action_id = agent.act(state)
            next_state, reward, done = simulate_environment(action_id)
            next_state = np.reshape(next_state, [1, STATE_SIZE])
            
            # Store experience in memory
            agent.remember(state, action_id, reward, next_state, done)
            
            # Update the current state
            state = next_state
            
            # Replay and learn from past experiences
            agent.replay()
            
            if done:
                print(f"Episode: {episode}/{EPISODES}, score: {time_step}, e: {agent.epsilon:.2}")
                break

if __name__ == "__main__":
    main()
