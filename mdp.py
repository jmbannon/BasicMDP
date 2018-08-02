import numpy as np
import sys
import time
import os

MAP = np.matrix([
  [' ', ' ', ' ', ' ', ' ', '░', '$', 'X', ' ', 'X', 'O', ' ', '$'],
  [' ', ' ', 'X', 'X', 'X', 'X', '░', 'X', ' ', 'X', 'X', 'X', 'X'],
  ['X', ' ', ' ', ' ', ' ', ' ', '░', 'X', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', 'X', 'X', ' ', ' ', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', 'X', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', '░', ' ', '░', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', 'X', 'X', 'X', 'X', '░', ' ', ' ', ' ', ' ', 'X', ' '],
  ['X', ' ', ' ', ' ', '░', '$', '░', ' ', ' ', ' ', ' ', 'X', ' '],
  [' ', ' ', ' ', ' ', ' ', '░', ' ', ' ', ' ', ' ', ' ', 'X', ' '],
  [' ', 'X', 'X', 'O', ' ', ' ', ' ', ' ', ' ', 'X', 'X', ' ', ' '],
  [' ', 'X', 'X', ' ', ' ', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', 'X', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', '░', ' ', '░', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', 'X', 'X', 'X', 'X', '░', ' ', ' ', ' ', ' ', ' ', ' '],
  ['X', ' ', ' ', ' ', ' ', ' ', '░', ' ', ' ', ' ', 'X', 'X', '░'],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X', 'X', 'X', '$'],
  [' ', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
  [' ', 'X', 'X', ' ', ' ', 'X', 'X', ' ', ' ', 'X', 'X', 'X', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', 'X', 'X', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', '░', '$', '░', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
])

class World:
	def __init__(self, world, complete_reward, progress_reward, lava_reward, teleport_reward):
		self.world = world
		self.actions = {}
		self.rewards = {}
		self.complete_reward = complete_reward
		self.progress_reward = progress_reward
		self.lava_reward = lava_reward
		self.teleport_reward = teleport_reward

		self.teleporters = []
		rows, cols = world.shape
		for i in range(rows):
			for j in range(cols):
				if world[i, j] == 'O':
					self.teleporters.append((i, j))

	def in_bounds(self, i, j):
		rows, cols = self.world.shape
		return i >= 0 and i < rows and j >= 0 and j < cols and self.world[i, j] != 'X'

	def get_tile_actions(self, i, j):
		if not self.in_bounds(i, j):
			return {}
		if self.world[i, j] == '$':
			return {'E': {(i, j): 1.0}}

		actions = {}
		if self.world[i, j] == 'O':
			tele_prob = 1.0 / len(self.teleporters)
			actions['T'] = {tele: tele_prob for tele in self.teleporters}
		if self.in_bounds(i + 1, j):
			# actions.append((i + 1, j))
			actions['D'] = {(i + 1, j): 1.0}
		if self.in_bounds(i - 1, j):
			# actions.append((i - 1, j))
			actions['U'] = {(i - 1, j): 1.0}
		if self.in_bounds(i, j + 1):
			# actions.append((i, j + 1))
			actions['R'] = {(i, j + 1): 1.0}
		if self.in_bounds(i, j - 1):
			# actions.append((i, j - 1))
			actions['L'] = {(i, j - 1): 1.0}

		# print(f'({i}, {j}), {actions}')
		# equal_proba = 1.0 / len(actions)
		# actions_proba = {}
		# for action in actions:
		# 	actions_proba[action] = equal_proba

		return actions


	def generate_actions_rewards(self):
		rows, cols = self.world.shape
		for i in range(rows):
			for j in range(cols):
				if self.in_bounds(i, j):
					self.actions[(i, j)] = self.get_tile_actions(i, j)

					if self.world[i, j] == '$':
						self.rewards[(i, j)] = self.complete_reward
					elif self.world[i, j] == '░':
						self.rewards[(i, j)] = self.lava_reward
					elif self.world[i, j] == 'O':
						self.rewards[(i, j)] = self.teleport_reward
					else:
						self.rewards[(i, j)] = self.progress_reward

	def move(self, position, action):
		state = list(self.actions[position][action].keys())[0]
		return state

	def print_world(self, world):
		print(f'Completion Reward ($): {self.complete_reward}')
		print(f'Progress Reward ( ): {self.progress_reward}')
		print(f'Lava Reward (░): {self.lava_reward}')
		print(f'Teleport Reward (O): {self.teleport_reward}')

		rows, cols = world.shape
		def in_world(i, j):
			return i > 0 and i < rows + 1 and j > 0 and j < cols + 1

		for i in range(rows + 2):
			for j in range(cols + 2):
				if i == 0 and j == 0:
					print('┌', end='')
				elif i == 0 and j == cols + 1:
					print('┐', end='')
				elif i == rows + 1 and j == 0:
					print('└', end='')
				elif i == rows + 1 and j == cols + 1:
					print('┘', end='')
				elif i == 0 or i == rows + 1:
					print('─', end='')
				elif j == 0 or j == cols + 1:
					print('│', end='')
				else:
					print(world[i - 1, j - 1], end='')
			print('')

	def play(self, position, policy):
		i, j = position
		assert(self.in_bounds(i, j))
		worldmap = self.world.copy()
		worldmap[i, j] = '•'

		while self.world[i, j] != '$':
			# print(position)
			time.sleep(1)
			os.system('clear')
			self.print_world(worldmap)
			action = policy[(i, j)]
			if action == 'T':
				worldmap[i, j] = 'Φ'
			elif worldmap[i, j] != 'Φ':
				worldmap[i, j] = '*'
			position = self.move(position, action)
			i, j = position
			if action == 'T':
				worldmap[i, j] = 'Φ'
			else:
				worldmap[i, j] = '•'

		worldmap[i, j] = '•'
		time.sleep(1)
		os.system('clear')
		self.print_world(worldmap)
				

class MDP:
	def __init__(self, transition, reward, gamma=0.9):
		self.states = transition.keys()
		self.transition = transition
		self.reward = reward
		self.gamma = gamma

	# Return reward
	def R(self, state):
		return self.reward[state]

	# Returns set of actions
	def actions(self, state):
		return self.transition[state].keys()

	# state
	#   action
	#     (state', prob)
	#     (state', prob)
	#   action
	#     (state', prob)
	# state
	#   ...

	# For a given state and action, return pairs of (result-state, probability)
	# This will always be 1
	def T(self, state, action):
		return self.transition[state][action].items()

	def utility(self, epsilon=0.001):
		R = self.R
		T = self.T
		V = {s: 0 for s in self.states}

		while True:
			V1 = V.copy()
			delta = 0
			for s in self.states:
				V[s] = R(s) + self.gamma * max( [sum( [p * V[s_] for (s_, p) in T(s, a)] ) for a in self.actions(s)] )
				delta = max(delta, abs(V1[s] - V[s]))

			# print(delta)
			if delta < epsilon * (1 - self.gamma) / self.gamma:
				return V1

	def policy(self):
		V = self.utility()
		P = {}

		def expected_utility(s, a):
			return sum([ p * V[s_] for (s_, p) in self.T(s, a) ])

		for s in self.states:
			P[s] = max(self.actions(s), key=lambda a: expected_utility(s, a))

		return P




world = World(MAP, 1, -.003, -100, -1)
world.generate_actions_rewards()
mdp = MDP(world.actions, world.rewards)
world.play((0, 0), mdp.policy())


