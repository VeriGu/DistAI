import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def find_block_prec(n, b):
	t = leader_f[n]
	return True

def find_block(n, b):
	t = leader_f[n]
	block_found[n,b,t] = True

def add_transaction_prec(tr, b):
	return True

def add_transaction(tr, b):
	transaction_in_block[tr,b] = True

def begin_broadcast_prec(n, b, t):
	if not ((leader[n,t]) and (block_found[n,b,t]) and (not broadcasted[n])):
		return False
	return True

def begin_broadcast(n, b, t):
	broadcastable[n,b,t] = True

def begin_broadcast_adversary_prec(n, b, t):
	if not ((not honest[n])):
		return False
	return True

def begin_broadcast_adversary(n, b, t):
	broadcastable[n,b,t] = True

def byzantine_broadcast_prec(n, b, t):
	if not ((broadcastable[n,b,t])):
		return False
	tmp_var_1 = True
	for TR in range(transaction_num):
		for T in range(time_num):
			if not (not (honest[n] and transaction_time[TR,T] and (T <= t) and not transaction_confirmed[TR,n]) or (transaction_in_block[TR,b])):
				tmp_var_1 = False
				break
	if not (tmp_var_1):
		return False
	tmp_var_2 = True
	for TR in range(transaction_num):
		for T in range(time_num):
			if not (not (honest[n] and transaction_in_block[TR,b]) or (transaction_time[TR,T] and (T <= t) and not transaction_confirmed[TR,n])):
				tmp_var_2 = False
				break
	if not (tmp_var_2):
		return False
	return True

def byzantine_broadcast(n, b, t):
	for N in range(node_num):
		for B in range(block_num):
			block_confirmed[N,B,t] = rng.integers(0, 2, dtype=bool)
	broadcasted[n] = True
	broadcastable[n,b,t] = False
	for TR in range(transaction_num):
		for N in range(node_num):
			transaction_confirmed[TR,N] = transaction_confirmed[TR,N] or (transaction_in_block[TR,b]) if honest[n] else transaction_confirmed[TR,N]
	if honest[n] or rng.random() > 0.5:
		for tmp_loop_var in range(node_num):
			block_confirmed[tmp_loop_var, b, t] = True

def sabotage_prec(n):
	if not ((not honest[n])):
		return False
	return True

def sabotage(n):
	for B in range(block_num):
		for T in range(time_num):
			block_confirmed[n,B,T] = rng.integers(0, 2, dtype=bool)
	for TR in range(transaction_num):
		transaction_confirmed[TR,n] = rng.integers(0, 2, dtype=bool)

func_from_name = {'find_block': find_block, 'find_block_prec': find_block_prec, 'add_transaction': add_transaction, 'add_transaction_prec': add_transaction_prec, 'begin_broadcast': begin_broadcast, 'begin_broadcast_prec': begin_broadcast_prec, 'begin_broadcast_adversary': begin_broadcast_adversary, 'begin_broadcast_adversary_prec': begin_broadcast_adversary_prec, 'byzantine_broadcast': byzantine_broadcast, 'byzantine_broadcast_prec': byzantine_broadcast_prec, 'sabotage': sabotage, 'sabotage_prec': sabotage_prec}

def instance_generator():
	node_num = rng.integers(2, 6)
	block_num = rng.integers(1, 5)
	transaction_num = rng.integers(1, 5)
	time_num = rng.integers(2, 6)
	return node_num, block_num, transaction_num, time_num

def sample(max_iter=25):
	global node_num, block_num, transaction_num, time_num, leader, honest, broadcastable, broadcasted, block_found, block_confirmed, transaction_time, transaction_in_block, transaction_confirmed, leader_f, transaction_time_f
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num, block_num, transaction_num, time_num = instance_generator()
		honest = rng.integers(0, 2, size=(node_num), dtype=bool)
		broadcastable = rng.integers(0, 2, size=(node_num, block_num, time_num), dtype=bool)
		broadcasted = rng.integers(0, 2, size=(node_num), dtype=bool)
		block_found = rng.integers(0, 2, size=(node_num, block_num, time_num), dtype=bool)
		block_confirmed = rng.integers(0, 2, size=(node_num, block_num, time_num), dtype=bool)
		transaction_in_block = rng.integers(0, 2, size=(transaction_num, block_num), dtype=bool)
		transaction_confirmed = rng.integers(0, 2, size=(transaction_num, node_num), dtype=bool)
		leader = np.zeros((node_num, time_num), dtype=bool)
		leader_f = rng.integers(0, time_num, size=(node_num))
		for i in range(node_num):
			leader[i, leader_f[i]] = True
		transaction_time = np.zeros((transaction_num, time_num), dtype=bool)
		transaction_time_f = rng.integers(0, time_num, size=(transaction_num))
		for i in range(transaction_num):
			transaction_time[i, transaction_time_f[i]] = True
		
		for N in range(node_num):
			for B in range(block_num):
				for T in range(time_num):
					block_found[N,B,T] = False
		for N in range(node_num):
			for B in range(block_num):
				for T in range(time_num):
					block_confirmed[N,B,T] = False
		for TR in range(transaction_num):
			for B in range(block_num):
				transaction_in_block[TR,B] = False
		for TR in range(transaction_num):
			for N in range(node_num):
				transaction_confirmed[TR,N] = False
		for N in range(node_num):
			broadcasted[N] = False
		for N in range(node_num):
			for B in range(block_num):
				for T in range(time_num):
					broadcastable[N,B,T] = False

		action_pool = ['find_block', 'add_transaction', 'begin_broadcast', 'begin_broadcast_adversary', 'byzantine_broadcast', 'sabotage']
		argument_pool = dict()
		argument_pool['find_block'] = []
		for n in range(node_num):
			for b in range(block_num):
				argument_pool['find_block'].append((n, b))
		argument_pool['add_transaction'] = []
		for tr in range(transaction_num):
			for b in range(block_num):
				argument_pool['add_transaction'].append((tr, b))
		argument_pool['begin_broadcast'] = []
		for n in range(node_num):
			for b in range(block_num):
				for t in range(time_num):
					argument_pool['begin_broadcast'].append((n, b, t))
		argument_pool['begin_broadcast_adversary'] = []
		for n in range(node_num):
			for b in range(block_num):
				for t in range(time_num):
					argument_pool['begin_broadcast_adversary'].append((n, b, t))
		argument_pool['byzantine_broadcast'] = []
		for n in range(node_num):
			for b in range(block_num):
				for t in range(time_num):
					argument_pool['byzantine_broadcast'].append((n, b, t))
		argument_pool['sabotage'] = []
		for n in range(node_num):
			argument_pool['sabotage'].append((n,))

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
				block_indices = rng.choice(list(range(block_num)), 1, replace=False)
				block_indices = sorted(block_indices)
				transaction_indices = rng.choice(list(range(transaction_num)), 1, replace=False)
				transaction_indices = sorted(transaction_indices)
				time_indices = rng.choice(list(range(time_num)), 2, replace=False)
				time_indices = sorted(time_indices)
				for N1, N2, in permutations(node_indices):
					for B1, in permutations(block_indices):
						for TR1, in permutations(transaction_indices):
							TI1, TI2, = time_indices
							df_data.add((leader[N1,TI1], leader[N1,TI2], leader[N2,TI1], leader[N2,TI2], honest[N1], honest[N2], broadcastable[N1,B1,TI1], broadcastable[N1,B1,TI2], broadcastable[N2,B1,TI1], broadcastable[N2,B1,TI2], broadcasted[N1], broadcasted[N2], block_found[N1,B1,TI1], block_found[N1,B1,TI2], block_found[N2,B1,TI1], block_found[N2,B1,TI2], block_confirmed[N1,B1,TI1], block_confirmed[N1,B1,TI2], block_confirmed[N2,B1,TI1], block_confirmed[N2,B1,TI2], transaction_time[TR1,TI1], transaction_time[TR1,TI2], transaction_in_block[TR1,B1], transaction_confirmed[TR1,N1], transaction_confirmed[TR1,N2]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 500 or (simulation_round > 10 and df_size_history[-1] == df_size_history[-11])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['leader(N1,TI1)', 'leader(N1,TI2)', 'leader(N2,TI1)', 'leader(N2,TI2)', 'honest(N1)', 'honest(N2)', 'broadcastable(N1,B1,TI1)', 'broadcastable(N1,B1,TI2)', 'broadcastable(N2,B1,TI1)', 'broadcastable(N2,B1,TI2)', 'broadcasted(N1)', 'broadcasted(N2)', 'block_found(N1,B1,TI1)', 'block_found(N1,B1,TI2)', 'block_found(N2,B1,TI1)', 'block_found(N2,B1,TI2)', 'block_confirmed(N1,B1,TI1)', 'block_confirmed(N1,B1,TI2)', 'block_confirmed(N2,B1,TI1)', 'block_confirmed(N2,B1,TI2)', 'transaction_time(TR1,TI1)', 'transaction_time(TR1,TI2)', 'transaction_in_block(TR1,B1)', 'transaction_confirmed(TR1,N1)', 'transaction_confirmed(TR1,N2)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/blockchain.csv', index=False)
	print('Simulation finished. Trace written to traces/blockchain.csv')
