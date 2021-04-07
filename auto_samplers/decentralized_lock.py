import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def send_prec(src, dst):
	if not ((has_lock[src])):
		return False
	return True

def send(src, dst):
	message[src, dst] = True
	has_lock[src] = False

def recv_prec(src, dst):
	if not ((message[src, dst])):
		return False
	return True

def recv(src, dst):
	message[src, dst] = False
	has_lock[dst] = True

func_from_name = {'send': send, 'send_prec': send_prec, 'recv': recv, 'recv_prec': recv_prec}

def instance_generator():
	node_num = rng.integers(4, 8)
	return node_num

def sample(max_iter=50):
	global node_num, message, has_lock, start_node
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num = instance_generator()
		message = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		has_lock = rng.integers(0, 2, size=(node_num), dtype=bool)
		start_node = rng.integers(0, node_num)
		
		for Src in range(node_num):
			for Dst in range(node_num):
				message[Src, Dst] = False
		for X in range(node_num):
			has_lock[X] = X == start_node

		action_pool = ['send', 'recv']
		argument_pool = dict()
		argument_pool['send'] = []
		for src in range(node_num):
			for dst in range(node_num):
				argument_pool['send'].append((src, dst))
		argument_pool['recv'] = []
		for src in range(node_num):
			for dst in range(node_num):
				argument_pool['recv'].append((src, dst))

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
				node_indices = rng.choice(list(range(node_num)), 4, replace=False)
				node_indices = sorted(node_indices)
				for N1, N2, N3, N4, in permutations(node_indices):
					df_data.add((message[N1,N1], message[N1,N2], message[N1,N3], message[N1,N4], message[N2,N1], message[N2,N2], message[N2,N3], message[N2,N4], message[N3,N1], message[N3,N2], message[N3,N3], message[N3,N4], message[N4,N1], message[N4,N2], message[N4,N3], message[N4,N4], has_lock[N1], has_lock[N2], has_lock[N3], has_lock[N4], start_node==N1, start_node==N2, start_node==N3, start_node==N4))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['message(N1,N1)', 'message(N1,N2)', 'message(N1,N3)', 'message(N1,N4)', 'message(N2,N1)', 'message(N2,N2)', 'message(N2,N3)', 'message(N2,N4)', 'message(N3,N1)', 'message(N3,N2)', 'message(N3,N3)', 'message(N3,N4)', 'message(N4,N1)', 'message(N4,N2)', 'message(N4,N3)', 'message(N4,N4)', 'has_lock(N1)', 'has_lock(N2)', 'has_lock(N3)', 'has_lock(N4)', 'start_node=N1', 'start_node=N2', 'start_node=N3', 'start_node=N4'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/decentralized_lock.csv', index=False)
	print('Simulation finished. Trace written to traces/decentralized_lock.csv')
