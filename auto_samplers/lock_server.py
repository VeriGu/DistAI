import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def connect_prec(x, y):
	if not ((semaphore[y])):
		return False
	return True

def connect(x, y):
	link[x,y] = True
	semaphore[y] = False

def disconnect_prec(x, y):
	if not ((link[x,y])):
		return False
	return True

def disconnect(x, y):
	link[x,y] = False
	semaphore[y] = True

func_from_name = {'connect': connect, 'connect_prec': connect_prec, 'disconnect': disconnect, 'disconnect_prec': disconnect_prec}

def instance_generator():
	client_num = rng.integers(1, 5)
	server_num = rng.integers(1, 5)
	return client_num, server_num

def sample(max_iter=50):
	global client_num, server_num, link, semaphore
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		client_num, server_num = instance_generator()
		link = rng.integers(0, 2, size=(client_num, server_num), dtype=bool)
		semaphore = rng.integers(0, 2, size=(server_num), dtype=bool)
		
		for W in range(server_num):
			semaphore[W] = True
		for X in range(client_num):
			for Y in range(server_num):
				link[X,Y] = False

		action_pool = ['connect', 'disconnect']
		argument_pool = dict()
		argument_pool['connect'] = []
		for x in range(client_num):
			for y in range(server_num):
				argument_pool['connect'].append((x, y))
		argument_pool['disconnect'] = []
		for x in range(client_num):
			for y in range(server_num):
				argument_pool['disconnect'].append((x, y))

		for curr_iter in range(max_iter):
			rng.shuffle(action_pool)
			action_selected, args_selected = None, None
			for action in action_pool:
				rng.shuffle(argument_pool[action])
				argument_candidates = argument_pool[action]
				for args_candidate in argument_candidates:
					if func_from_name[action + '_prec'](*args_candidate):
						action_selected, args_selected = action, args_candidate
						break
				if action_selected is not None:
					break
			if action_selected is None:
				# action pool exhausted, start a new simulation
				break
			func_from_name[action_selected](*args_selected)

			# generate subsamples from the current state (sample)
			for k in range(3):
				client_indices = rng.choice(list(range(client_num)), 1, replace=False)
				client_indices = sorted(client_indices)
				server_indices = rng.choice(list(range(server_num)), 1, replace=False)
				server_indices = sorted(server_indices)
				for C1, in permutations(client_indices):
					for S1, in permutations(server_indices):
						df_data.add((link[C1,S1], semaphore[S1]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['link(C1,S1)', 'semaphore(S1)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/lock_server.csv', index=False)
	print('Simulation finished. Trace written to traces/lock_server.csv')
