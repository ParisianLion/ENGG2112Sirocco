import tensorflow as tf
from tensorflow.keras import layers, models



def create_dqn_model(state_space, action_space):
    model = models.Sequential()
    model.add(layers.Input(shape=(state_space,)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))  # Output Q-values for each action
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

action_mapping = {
    0: "Increase_Altitude",
    1: "Decrease_Altitude",
    2: "Turn_Left",
    3: "Turn_Right",
    4: "Throttle_Up",
    5: "Throttle_Down",
    6: "Land",
    7: "Take_Off"
}

class DQNAgent:
    def __init__(self, state_space, action_space):
        self.model = create_dqn_model(state_space, action_space)
        self.memory = []
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(action_space)  # Explore
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size)
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    simconnect = SimConnect()
    env = FlightSimEnv(simconnect)
    agent = DQNAgent(state_space=NUM_STATES, action_space=NUM_ACTIONS)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, NUM_STATES])  # Reshape for the model input
        total_reward = 0
        done = False

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward = env.step(action)
            next_state = np.reshape(next_state, [1, NUM_STATES])  # Reshape for the model input
            
            # Store the experience in memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()  # Learn from the experiences
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    main()
