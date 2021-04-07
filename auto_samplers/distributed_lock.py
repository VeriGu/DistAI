import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def grant_prec(n1, n2, e):
	if not ((held[n1])):
		return False
	if not ((not (e <=  ep[n1]))):
		return False
	return True

def grant(n1, n2, e):
	transfer[e, n2] = True
	held[n1] = False

def accept_prec(n, e):
	if not ((transfer[e,n])):
		return False
	return True

def accept(n, e):
	if (not (e <=  ep[n])):
		held[n] = True
		ep[n] = e
		locked[e, n] = True

func_from_name = {'grant': grant, 'grant_prec': grant_prec, 'accept': accept, 'accept_prec': accept_prec}

def instance_generator():
	node_num = rng.integers(2, 6)
	epoch_num = rng.integers(2, 6)
	return node_num, epoch_num

def sample(max_iter=50):
	global node_num, epoch_num, held, transfer, locked, zero, first, e, ep
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num, epoch_num = instance_generator()
		held = rng.integers(0, 2, size=(node_num), dtype=bool)
		transfer = rng.integers(0, 2, size=(epoch_num, node_num), dtype=bool)
		locked = rng.integers(0, 2, size=(epoch_num, node_num), dtype=bool)
		zero = rng.integers(0, epoch_num)
		first = rng.integers(0, node_num)
		e = rng.integers(0, epoch_num)
		ep = rng.integers(0, epoch_num, size=(node_num))
		zero = 0
		e = rng.integers(1, epoch_num)
		
		for X in range(node_num):
			held[X] = X==first
		for N in range(node_num):
			ep[N] = zero
		ep[first] = e
		for E in range(epoch_num):
			for N in range(node_num):
				transfer[E, N] = False
		for E in range(epoch_num):
			for N in range(node_num):
				locked[E, N] = False

		action_pool = ['grant', 'accept']
		argument_pool = dict()
		argument_pool['grant'] = []
		for n1 in range(node_num):
			for n2 in range(node_num):
				for e in range(epoch_num):
					argument_pool['grant'].append((n1, n2, e))
		argument_pool['accept'] = []
		for n in range(node_num):
			for e in range(epoch_num):
				argument_pool['accept'].append((n, e))

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
				node_indices = rng.choice(list(range(node_num)), 2, replace=False)
				node_indices = sorted(node_indices)
				epoch_indices = rng.choice(list(range(epoch_num)), 2, replace=False)
				epoch_indices = sorted(epoch_indices)
				for N1, N2, in permutations(node_indices):
					E1, E2, = epoch_indices
					df_data.add((held[N1], held[N2], transfer[E1,N1], transfer[E1,N2], transfer[E2,N1], transfer[E2,N2], locked[E1,N1], locked[E1,N2], locked[E2,N1], locked[E2,N2], (E1 <= ep[N1]), (E2 <= ep[N1]), (ep[N1] <= ep[N2]), (ep[N2] <= ep[N1]), (E1 <= ep[N2]), (E2 <= ep[N2]), first==N1, first==N2))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['held(N1)', 'held(N2)', 'transfer(E1,N1)', 'transfer(E1,N2)', 'transfer(E2,N1)', 'transfer(E2,N2)', 'locked(E1,N1)', 'locked(E1,N2)', 'locked(E2,N1)', 'locked(E2,N2)', 'le(E1,ep(N1))', 'le(E2,ep(N1))', 'le(ep(N1),ep(N2))', 'le(ep(N2),ep(N1))', 'le(E1,ep(N2))', 'le(E2,ep(N2))', 'first=N1', 'first=N2'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/distributed_lock.csv', index=False)
	print('Simulation finished. Trace written to traces/distributed_lock.csv')
