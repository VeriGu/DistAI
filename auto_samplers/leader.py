import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def send_prec(n, n1):
	tmp_var_1 = True
	for Z in range(node_num):
		if not (not ((Z != n and Z != n1)) or (btw[n, n1, Z])):
			tmp_var_1 = False
			break
	if not (tmp_var_1):
		return False
	return True

def send(n, n1):
	pending[idn[n], n1] = True

def become_leader_prec(n):
	if not ((pending[idn[n], n])):
		return False
	return True

def become_leader(n):
	leader[n] = True

def receive_prec(n, m, n1):
	if not ((pending[m, n])):
		return False
	tmp_var_2 = True
	for Z in range(node_num):
		if not (not ((Z != n and Z != n1)) or (btw[n, n1, Z])):
			tmp_var_2 = False
			break
	if not (tmp_var_2):
		return False
	return True

def receive(n, m, n1):
	if (idn[n] <=  m):
		pending[m, n1] = True

func_from_name = {'send': send, 'send_prec': send_prec, 'become_leader': become_leader, 'become_leader_prec': become_leader_prec, 'receive': receive, 'receive_prec': receive_prec}

def instance_generator():
	node_num = rng.integers(3, 7)
	id_num = node_num
	return node_num, id_num

def sample(max_iter=50):
	global node_num, id_num, leader, pending, idn, btw
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num, id_num = instance_generator()
		leader = rng.integers(0, 2, size=(node_num), dtype=bool)
		pending = rng.integers(0, 2, size=(id_num, node_num), dtype=bool)
		idn = rng.permutation(node_num)
		# build ring topology
		btw = np.zeros((node_num, node_num, node_num), dtype=bool)
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					if x != y and x != z and y != z:
						btw[x, y, z] = (x < y < z) | (z < x < y) | (y < z < x)
		
		for N in range(node_num):
			leader[N] = False
		for V in range(id_num):
			for N in range(node_num):
				pending[V,N] = False

		action_pool = ['send', 'become_leader', 'receive']
		argument_pool = dict()
		argument_pool['send'] = []
		for n in range(node_num):
			for n1 in range(node_num):
				argument_pool['send'].append((n, n1))
		argument_pool['become_leader'] = []
		for n in range(node_num):
			argument_pool['become_leader'].append((n,))
		argument_pool['receive'] = []
		for n in range(node_num):
			for m in range(id_num):
				for n1 in range(node_num):
					argument_pool['receive'].append((n, m, n1))

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
				node_indices = rng.choice(list(range(node_num)), 3, replace=False)
				node_indices = sorted(node_indices)
				N1, N2, N3, = node_indices
				I1, I2, I3, = idn[N1], idn[N2], idn[N3],
				tmp_list = [(N1, I1), (N2, I2), (N3, I3)]
				tmp_list.sort(key=lambda x: x[1])
				(N1, I1), (N2, I2), (N3, I3) = tmp_list
				df_data.add((leader[N1], leader[N2], leader[N3], pending[I1,N1], pending[I1,N2], pending[I1,N3], pending[I2,N1], pending[I2,N2], pending[I2,N3], pending[I3,N1], pending[I3,N2], pending[I3,N3], btw[N1,N2,N3]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['leader(N1)', 'leader(N2)', 'leader(N3)', 'pending(I1,N1)', 'pending(I1,N2)', 'pending(I1,N3)', 'pending(I2,N1)', 'pending(I2,N2)', 'pending(I2,N3)', 'pending(I3,N1)', 'pending(I3,N2)', 'pending(I3,N3)', 'ring.btw(N1,N2,N3)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/leader.csv', index=False)
	print('Simulation finished. Trace written to traces/leader.csv')
