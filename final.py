import gym 								 # Used to import the enviroment

import keras
from keras.models import Sequential 	 # Used for the CNN
from keras.models import load_model      # Used to load and save the model
from keras.layers import Dense			 # For storing moves 
from keras.optimizers import Adam		 # Optimization algorithm used to produce/update network weights 

import random 							 # Maths imports used for calculations
import math
import numpy as np 
from collections import deque 			 # Deque allow for faster append and pop operation / O(1) time complexity

# These values will be used in training the DQN
gamma = 1.0 					# Discount Factor
epsilon = 1.0					# Exploration (Randomly chooses action for the best long term affect)
epsilon_min = 0.01
epislon_decay = 0.995 			# Used to lower the epsilon values
learning_rate = 0.01 			# Learning rate
learning_rate_decay = 0.01 		# Used to lower the learning rate value value
batch_size = 64 				# Used as the size limit of the previous experiances
memory = deque(maxlen = 2000)	# Stores all experiances


def load_game(): 
	""" Step 1: Import an OpenAI universe/gym game 20% """
	# Load enviroment / Loading CartPole
	env = gym.make("CartPole-v1")
	return env

def load_CNN(): 
	""" Step 2: Creating a network 20% """
	# Creating the neural network
	model = Sequential()	# Creates the foundation of the layers
	state_size = 4 # Input layer
	# "Dense" is the basic form of the Neural network layer
	model.add(Dense(24, input_dim = state_size, activation = "relu")) # Creating the CNN Layers, first with 24 nodes, second with 48 nodes
	model.add(Dense(48, activation = "relu"))
	model.add(Dense(2, activation = "relu")) # 2 is used as this is the amount of actions we can take (move left or right)
	model.compile(loss = "mse", optimizer = Adam(lr = learning_rate, decay = learning_rate_decay)) # Create model based on the information above

	# Loads the trained model I have created
	model = load_model("trained_DQN.p5") 
	print(model.summary()) # Displays the params at each layer
	print("Loaded trained DQN Model")
	return model


""" List of functions required for DQN: """

# Stores a list of all previous experiances and observation
# Used to re-train the model again with previous experiances.
def remember(state, action, reward, next_state, done): 
	memory.append((state, action, reward, next_state, done))

def choose_action(state, epsilon, model, env):
	if (np.random.random() <= epsilon): # If a random value is below epsilon, then return a random action
		return env.action_space.sample()
	else:
		action_value = model.predict(state) # Predit the reward value based on the given state
		return np.argmax(action_value[0])	# Return the best action based on the predicted reward (either action 0/1, left/right)

def get_epsilon(episode): # Lowers the epsilon based on amount of episodes
	new_ep = min(epsilon, 1.0 - math.log10((episode + 1) * epislon_decay))
	if epsilon_min > new_ep:
		return epsilon_min
	else:
		return new_ep

def replay(batch_size, epsilon, model):
	
	states_batch, q_values = [], []
	mini_batch = random.sample(memory, min(len(memory), batch_size)) # Sample some experiances from memory (previous experiances)
	for state, action, reward, next_state, done in mini_batch: # Extracting information from each memory
		
		# Make agent try map the current state to the future discounted reward.
		q_update = model.predict(state)

		if done: # If done, make our target reward
			q_update[0][action] = reward 
		else: # Predict the future discounted reward
		# Predict the future discounted reward
		# Calculating new Q value by taking the Max Q for a given action (predicted valaue of the next best state)
		# & multiplying it by the gamma value, and then lastly storing it to the current state reward.
			q_update[0][action] = reward + gamma * np.max(model.predict(next_state)[0])

		states_batch.append(state[0]) 	# Adding to the states list
		q_values.append(q_update[0])	# Adding to the Q values list

	# Train the neural network using the states list and Q values list 
	model.fit(np.array(states_batch), np.array(q_values), verbose = 0)

	# Want to update epsilon / will decrease epsilon depending if result is not good
	if epsilon > epsilon_min:
		epsilon = epsilon * epislon_decay


# Training the network
""" Step 4: Deep reinforcement learning model 30% """
def run():

	env = load_game() # Stores the enviorment
	model = load_CNN() # Stores the trained model

	number_episodes = 2000			# Total episodes to play
	win_goal = 195					# Average score of past 100 episodes must be higher than 195
	highest_score = 0 				# Stores the highest score reached by the DQN

	# Stores the scores achieved by the agent
	scores = deque(maxlen = number_episodes)	 # Stores every score of every episode played
	average_scores_list = deque(maxlen = 100)	 # Stores the last 100 episodes played
	observation_space = env.observation_space.shape[0] # Values describing the what the env observes, such as cart velocity etc

	for eps in range(number_episodes):

		state = env.reset() # Resets the state at the start of every game.
		state = np.reshape(state, [1, observation_space])
		done = False # Reset after every episode
		i = 0		 # Reset i after every episode, stores how long the agent survived

		while not done:
			action = choose_action(state, get_epsilon(eps), model, env) # Choose an action / Either 1/0, left/right
			
			# Advance the game to the next frame based on the action choosen. 
			# The reward is +1 point for every every the pole is balanced
			next_state, reward, done, info = env.step(action)
			
			# env.render() # Uncomment to show gameplay
			next_state = np.reshape(next_state, [1, observation_space])
			
			# Remember the previous experiance, such as: (state, action, reward, next_state, done)
			remember(state, action, reward, next_state, done) 
			state = next_state # Make the next_state the new current state for the next frame.
			i = i + 1 # Increment based on every frame the agent has survived

		if done: # Episode is over
			scores.append(i) # Game is over, stores amount of wins ticks
			#print("Eps: {} Score: {} ".format(eps, scores[-1]))

			average_scores_list.append(i)
			average_score = np.mean(average_scores_list)
			if eps % 100 is 0 and eps is not 0:
				print("Episode {} - average score over last 100 episodes was: {}".format(eps, average_score))

			if (average_score > win_goal):
				print("Reached the target of: {}. In {} episodes. Average score was: {}. ".format(win_goal, eps, average_score))
				model.save("trained_DQN.p5")
				print("Saved trained DQN model!")
				break

			if max(scores) > highest_score: # Displays the highest score ever achived 
				highest_score = max(scores)
				print("Highest score so far is: {} ".format(highest_score))
			
			replay(batch_size, get_epsilon(eps), model) # Can train the network based of results we have gotten

	if eps is number_episodes and highest_score < win_goal: # If we have reached the desired score after all games
		print("did not solve after: {} episodes".format(eps))
		print("highest score was: {} ".format(highest_score))


if __name__ == "__main__":
	run()





