import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def do_abort_prec(op, kw, kr, luwkw, lurkw, luwkr, lcwkr):
	tx = op_in_tx_f[op]
	n = op_node_f[op]
	if not ((not abort_tx[tx]) and (not commit_tx[tx])):
		return False
	tmp_var_3 = True
	for X in range(operation_num):
		for N in range(node_num):
			if not (not ((oporder[X, op] and X != op and op_node[X, N])) or (precommit_tx[tx, N])):
				tmp_var_3 = False
				break
	if not (tmp_var_3):
		return False
	if not ((not precommit_tx[tx, n])):
		return False
	tmp_var_4 = True
	for K in range(key_num):
		if not (not op_writes_key[op, K] or op_writes_key[op, kw]):
			tmp_var_4 = False
			break
	if not (tmp_var_4):
		return False
	if not ((not (op_writes_key[op, kw]) or (node_for_key[kw, n]))):
		return False
	tmp_var_5 = True
	for K in range(key_num):
		if not (not op_reads_key[op, K] or op_reads_key[op, kr]):
			tmp_var_5 = False
			break
	if not (tmp_var_5):
		return False
	if not ((not (op_reads_key[op, kr]) or (node_for_key[kr, n]))):
		return False
	tmp_var_6 = True
	for T in range(transaction_num):
		if not (not (write_tx[T, kw]) or (((T <=  luwkw) or abort_tx[T]))):
			tmp_var_6 = False
			break
	if not (tmp_var_6):
		return False
	tmp_var_7 = True
	for T in range(transaction_num):
		if not (not (read_tx[T, kw]) or ((T <=  lurkw) or abort_tx[T])):
			tmp_var_7 = False
			break
	if not (tmp_var_7):
		return False
	tmp_var_8 = True
	for T in range(transaction_num):
		if not (not (write_tx[T, kr]) or (((T <=  luwkr) or (tx <=  T) or abort_tx[T]))):
			tmp_var_8 = False
			break
	if not (tmp_var_8):
		return False
	tmp_var_9 = True
	for T in range(transaction_num):
		if not (not ((commit_tx[T] and write_tx[T, kr])) or (((T <=  lcwkr) or (tx <=  T)))):
			tmp_var_9 = False
			break
	if not (tmp_var_9):
		return False
	if not (((op_writes_key[op, kw] and ((tx <=  luwkw) or (tx <=  lurkw))) or (op_reads_key[op, kr] and luwkr != lcwkr and (luwkr <=  tx)))):
		return False
	return True

def do_abort(op, kw, kr, luwkw, lurkw, luwkr, lcwkr):
	tx = op_in_tx_f[op]
	n = op_node_f[op]
	abort_tx[tx] = True

def do_progress_prec(op, kw, kr, luwkw, lurkw, luwkr, lcwkr):
	tx = op_in_tx_f[op]
	n = op_node_f[op]
	if not ((not abort_tx[tx]) and (not commit_tx[tx])):
		return False
	tmp_var_10 = True
	for X in range(operation_num):
		for N in range(node_num):
			if not (not ((oporder[X, op] and X != op and op_node[X, N])) or (precommit_tx[tx, N])):
				tmp_var_10 = False
				break
	if not (tmp_var_10):
		return False
	if not ((not precommit_tx[tx, n])):
		return False
	tmp_var_11 = True
	for K in range(key_num):
		if not (not op_writes_key[op, K] or op_writes_key[op, kw]):
			tmp_var_11 = False
			break
	if not (tmp_var_11):
		return False
	if not ((not (op_writes_key[op, kw]) or (node_for_key[kw, n]))):
		return False
	tmp_var_12 = True
	for K in range(key_num):
		if not (not op_reads_key[op, K] or op_reads_key[op, kr]):
			tmp_var_12 = False
			break
	if not (tmp_var_12):
		return False
	if not ((not (op_reads_key[op, kr]) or (node_for_key[kr, n]))):
		return False
	tmp_var_13 = True
	for T in range(transaction_num):
		if not (not (write_tx[T, kw]) or (((T <=  luwkw) or abort_tx[T]))):
			tmp_var_13 = False
			break
	if not (tmp_var_13):
		return False
	tmp_var_14 = True
	for T in range(transaction_num):
		if not (not (read_tx[T, kw]) or ((T <=  lurkw) or abort_tx[T])):
			tmp_var_14 = False
			break
	if not (tmp_var_14):
		return False
	tmp_var_15 = True
	for T in range(transaction_num):
		if not (not (write_tx[T, kr]) or (((T <=  luwkr) or (tx <=  T) or abort_tx[T]))):
			tmp_var_15 = False
			break
	if not (tmp_var_15):
		return False
	tmp_var_16 = True
	for T in range(transaction_num):
		if not (not ((commit_tx[T] and write_tx[T, kr])) or (((T <=  lcwkr) or (tx <=  T)))):
			tmp_var_16 = False
			break
	if not (tmp_var_16):
		return False
	if not ((not ((op_writes_key[op, kw] and ((tx <=  luwkw) or (tx <=  lurkw))) or (op_reads_key[op, kr] and luwkr != lcwkr and (luwkr <=  tx))))):
		return False
	return True

def do_progress(op, kw, kr, luwkw, lurkw, luwkr, lcwkr):
	tx = op_in_tx_f[op]
	n = op_node_f[op]
	if (op_writes_key[op, kw]):
		write_tx[tx, kw] = True
	if (op_reads_key[op, kr]):
		write_tx[tx, kw] = True
		depends_tx[tx, kr, lcwkr] = True
		read_tx[tx, kr] = True
	precommit_tx[tx, n] = True
	tmp_var_17 = True
	for O in range(operation_num):
		if not (not (oporder[op, O]) or (O == op)):
			tmp_var_17 = False
			break
	if tmp_var_17:
		write_tx[tx, kw] = True
		depends_tx[tx, kr, lcwkr] = True
		read_tx[tx, kr] = True
		commit_tx[tx] = True

func_from_name = {'do_abort': do_abort, 'do_abort_prec': do_abort_prec, 'do_progress': do_progress, 'do_progress_prec': do_progress_prec}

def instance_generator():
	transaction_num = rng.integers(2, 6)
	node_num = rng.integers(2, 6)
	key_num = rng.integers(1, 5)
	operation_num = rng.integers(2, 6)
	return transaction_num, node_num, key_num, operation_num

def sample(max_iter=25):
	global transaction_num, node_num, key_num, operation_num, op_reads_key, op_writes_key, op_node, node_for_key, op_in_tx, oporder, precommit_tx, abort_tx, commit_tx, depends_tx, read_tx, write_tx, zero, op_reads_key_f, op_writes_key_f, op_node_f, node_for_key_f, op_in_tx_f
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		transaction_num, node_num, key_num, operation_num = instance_generator()
		oporder = rng.integers(0, 2, size=(operation_num, operation_num), dtype=bool)
		precommit_tx = rng.integers(0, 2, size=(transaction_num, node_num), dtype=bool)
		abort_tx = rng.integers(0, 2, size=(transaction_num), dtype=bool)
		commit_tx = rng.integers(0, 2, size=(transaction_num), dtype=bool)
		depends_tx = rng.integers(0, 2, size=(transaction_num, key_num, transaction_num), dtype=bool)
		read_tx = rng.integers(0, 2, size=(transaction_num, key_num), dtype=bool)
		write_tx = rng.integers(0, 2, size=(transaction_num, key_num), dtype=bool)
		zero = rng.integers(0, transaction_num)
		zero = 0
		# the following code block applies rejection sampling to generate predicates that satisfy axiom:
		# op_reads_key(Op, K1) & op_writes_key(Op, K2) -> K1 ~= K2
		# you may consider manually improving its efficiency
		predicates_valid = False
		for retry in range(10):
			op_reads_key = np.zeros((operation_num, key_num), dtype=bool)
			op_reads_key_f = rng.integers(0, key_num, size=(operation_num))
			for i in range(operation_num):
				op_reads_key[i, op_reads_key_f[i]] = True
			op_writes_key = np.zeros((operation_num, key_num), dtype=bool)
			op_writes_key_f = rng.integers(0, key_num, size=(operation_num))
			for i in range(operation_num):
				op_writes_key[i, op_writes_key_f[i]] = True
			tmp_var_1 = True
			for Op in range(operation_num):
				for K1 in range(key_num):
					for K2 in range(key_num):
						if not (not (op_reads_key[Op, K1] and op_writes_key[Op, K2]) or (K1 != K2)):
							tmp_var_1 = False
							break
			if (tmp_var_1):
				predicates_valid = True
				break
		if not predicates_valid:
			continue
		
		# the following code block applies rejection sampling to generate predicates that satisfy axiom:
		# op_in_tx(T, O1) & op_in_tx(T, O2) & O1 ~= O2 & op_node(O1, N1) & op_node(O2, N2) -> N1 ~= N2
		# you may consider manually improving its efficiency
		predicates_valid = False
		for retry in range(10):
			op_in_tx = np.zeros((transaction_num, operation_num), dtype=bool)
			op_in_tx_f = rng.integers(0, transaction_num, size=(operation_num))
			for i in range(operation_num):
				op_in_tx[op_in_tx_f[i], i] = True
			op_node = np.zeros((operation_num, node_num), dtype=bool)
			op_node_f = rng.integers(0, node_num, size=(operation_num))
			for i in range(operation_num):
				op_node[i, op_node_f[i]] = True
			tmp_var_2 = True
			for O1 in range(operation_num):
				for N1 in range(node_num):
					for O2 in range(operation_num):
						for N2 in range(node_num):
							for T in range(transaction_num):
								if not (not (op_in_tx[T, O1] and op_in_tx[T, O2] and O1 != O2 and op_node[O1, N1] and op_node[O2, N2]) or (N1 != N2)):
									tmp_var_2 = False
									break
			if (tmp_var_2):
				predicates_valid = True
				break
		if not predicates_valid:
			continue
		
		node_for_key = np.zeros((key_num, node_num), dtype=bool)
		node_for_key_f = rng.integers(0, node_num, size=(key_num))
		for i in range(key_num):
			node_for_key[i, node_for_key_f[i]] = True
		
		# conditional total order
		oporder = np.zeros((operation_num, operation_num), dtype=bool)
		for X in range(operation_num):
			for Y in range(operation_num):
				oporder[X, Y] = (op_in_tx_f[X] == op_in_tx_f[Y] and X <= Y)
		
		for T in range(transaction_num):
			for N in range(node_num):
				precommit_tx[T, N] = T == zero
		for T in range(transaction_num):
			abort_tx[T] = False
		for T in range(transaction_num):
			commit_tx[T] = T == zero
		for T1 in range(transaction_num):
			for K in range(key_num):
				for T2 in range(transaction_num):
					depends_tx[T1, K, T2] = (T1 == zero and T2 == zero)
		for Tx in range(transaction_num):
			for K in range(key_num):
				read_tx[Tx, K] = Tx == zero
		for Tx in range(transaction_num):
			for K in range(key_num):
				write_tx[Tx, K] = Tx == zero

		action_pool = ['do_abort', 'do_progress']
		argument_pool = dict()
		argument_pool['do_abort'] = []
		for op in range(operation_num):
			for kw in range(key_num):
				for kr in range(key_num):
					for luwkw in range(transaction_num):
						for lurkw in range(transaction_num):
							for luwkr in range(transaction_num):
								for lcwkr in range(transaction_num):
									argument_pool['do_abort'].append((op, kw, kr, luwkw, lurkw, luwkr, lcwkr))
		argument_pool['do_progress'] = []
		for op in range(operation_num):
			for kw in range(key_num):
				for kr in range(key_num):
					for luwkw in range(transaction_num):
						for lurkw in range(transaction_num):
							for luwkr in range(transaction_num):
								for lcwkr in range(transaction_num):
									argument_pool['do_progress'].append((op, kw, kr, luwkw, lurkw, luwkr, lcwkr))

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
				transaction_indices = rng.choice(list(range(transaction_num)), 2, replace=False)
				transaction_indices = sorted(transaction_indices)
				node_indices = rng.choice(list(range(node_num)), 2, replace=False)
				node_indices = sorted(node_indices)
				key_indices = rng.choice(list(range(key_num)), 1, replace=False)
				key_indices = sorted(key_indices)
				operation_indices = rng.choice(list(range(operation_num)), 2, replace=False)
				operation_indices = sorted(operation_indices)
				T1, T2, = transaction_indices
				for N1, N2, in permutations(node_indices):
					for K1, in permutations(key_indices):
						for O1, O2, in permutations(operation_indices):
							df_data.add((op_reads_key[O1,K1], op_reads_key[O2,K1], op_writes_key[O1,K1], op_writes_key[O2,K1], op_node[O1,N1], op_node[O1,N2], op_node[O2,N1], op_node[O2,N2], node_for_key[K1,N1], node_for_key[K1,N2], precommit_tx[T1,N1], precommit_tx[T1,N2], precommit_tx[T2,N1], precommit_tx[T2,N2], abort_tx[T1], abort_tx[T2], commit_tx[T1], commit_tx[T2], depends_tx[T1,K1,T1], depends_tx[T1,K1,T2], depends_tx[T2,K1,T1], depends_tx[T2,K1,T2], read_tx[T1,K1], read_tx[T2,K1], write_tx[T1,K1], write_tx[T2,K1]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 500 or (simulation_round > 10 and df_size_history[-1] == df_size_history[-11])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['op_reads_key(O1,K1)', 'op_reads_key(O2,K1)', 'op_writes_key(O1,K1)', 'op_writes_key(O2,K1)', 'op_node(O1,N1)', 'op_node(O1,N2)', 'op_node(O2,N1)', 'op_node(O2,N2)', 'node_for_key(K1,N1)', 'node_for_key(K1,N2)', 'precommit_tx(T1,N1)', 'precommit_tx(T1,N2)', 'precommit_tx(T2,N1)', 'precommit_tx(T2,N2)', 'abort_tx(T1)', 'abort_tx(T2)', 'commit_tx(T1)', 'commit_tx(T2)', 'depends_tx(T1,K1,T1)', 'depends_tx(T1,K1,T2)', 'depends_tx(T2,K1,T1)', 'depends_tx(T2,K1,T2)', 'read_tx(T1,K1)', 'read_tx(T2,K1)', 'write_tx(T1,K1)', 'write_tx(T2,K1)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/database_chain_replication.csv', index=False)
	print('Simulation finished. Trace written to traces/database_chain_replication.csv')
