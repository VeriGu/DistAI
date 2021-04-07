import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def vote1_prec(n):
	if not ((alive[n])):
		return False
	if not ((not vote_no[n])):
		return False
	if not ((not decide_commit[n])):
		return False
	if not ((not decide_abort[n])):
		return False
	return True

def vote1(n):
	vote_yes[n] = True

def vote2_prec(n):
	if not ((alive[n])):
		return False
	if not ((not vote_yes[n])):
		return False
	if not ((not decide_commit[n])):
		return False
	if not ((not decide_abort[n])):
		return False
	return True

def vote2(n):
	vote_no[n] = True
	abort_flag[0] = True
	decide_abort[n] = True

def fail_prec(n):
	if not ((alive[n])):
		return False
	return True

def fail(n):
	alive[n] = False
	abort_flag[0] = True

def go1_prec():
	tmp_var_1 = True
	for N in range(node_num):
		if not (not go_commit[N]):
			tmp_var_1 = False
			break
	if not (tmp_var_1):
		return False
	tmp_var_2 = True
	for N in range(node_num):
		if not (not go_abort[N]):
			tmp_var_2 = False
			break
	if not (tmp_var_2):
		return False
	tmp_var_3 = True
	for N in range(node_num):
		if not (vote_yes[N]):
			tmp_var_3 = False
			break
	if not (tmp_var_3):
		return False
	return True

def go1():
	for N in range(node_num):
		go_commit[N] = True

def go2_prec():
	tmp_var_4 = True
	for N in range(node_num):
		if not (not go_commit[N]):
			tmp_var_4 = False
			break
	if not (tmp_var_4):
		return False
	tmp_var_5 = True
	for N in range(node_num):
		if not (not go_abort[N]):
			tmp_var_5 = False
			break
	if not (tmp_var_5):
		return False
	tmp_var_6 = False
	for N in range(node_num):
		if (vote_no[N] or not alive[N]):
			tmp_var_6 = True
			break
	if not (tmp_var_6):
		return False
	return True

def go2():
	for N in range(node_num):
		go_abort[N] = True

def commit_prec(n):
	if not ((alive[n])):
		return False
	if not ((go_commit[n])):
		return False
	return True

def commit(n):
	decide_commit[n] = True

def abort_prec(n):
	if not ((alive[n])):
		return False
	if not ((go_abort[n])):
		return False
	return True

def abort(n):
	decide_abort[n] = True

func_from_name = {'vote1': vote1, 'vote1_prec': vote1_prec, 'vote2': vote2, 'vote2_prec': vote2_prec, 'fail': fail, 'fail_prec': fail_prec, 'go1': go1, 'go1_prec': go1_prec, 'go2': go2, 'go2_prec': go2_prec, 'commit': commit, 'commit_prec': commit_prec, 'abort': abort, 'abort_prec': abort_prec}

def instance_generator():
	node_num = rng.integers(2, 6)
	return node_num

def sample(max_iter=50):
	global node_num, vote_yes, vote_no, alive, go_commit, go_abort, decide_commit, decide_abort, abort_flag
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num = instance_generator()
		vote_yes = rng.integers(0, 2, size=(node_num), dtype=bool)
		vote_no = rng.integers(0, 2, size=(node_num), dtype=bool)
		alive = rng.integers(0, 2, size=(node_num), dtype=bool)
		go_commit = rng.integers(0, 2, size=(node_num), dtype=bool)
		go_abort = rng.integers(0, 2, size=(node_num), dtype=bool)
		decide_commit = rng.integers(0, 2, size=(node_num), dtype=bool)
		decide_abort = rng.integers(0, 2, size=(node_num), dtype=bool)
		abort_flag = rng.integers(0, 2, size=(1), dtype=bool)
		
		for N in range(node_num):
			vote_yes[N] = False
		for N in range(node_num):
			vote_no[N] = False
		for N in range(node_num):
			alive[N] = True
		for N in range(node_num):
			go_commit[N] = False
		for N in range(node_num):
			go_abort[N] = False
		for N in range(node_num):
			decide_commit[N] = False
		for N in range(node_num):
			decide_abort[N] = False
		abort_flag[0] = False

		action_pool = ['vote1', 'vote2', 'fail', 'go1', 'go2', 'commit', 'abort']
		argument_pool = dict()
		argument_pool['vote1'] = []
		for n in range(node_num):
			argument_pool['vote1'].append((n,))
		argument_pool['vote2'] = []
		for n in range(node_num):
			argument_pool['vote2'].append((n,))
		argument_pool['fail'] = []
		for n in range(node_num):
			argument_pool['fail'].append((n,))
		argument_pool['go1'] = []
		argument_pool['go1'].append(())
		argument_pool['go2'] = []
		argument_pool['go2'].append(())
		argument_pool['commit'] = []
		for n in range(node_num):
			argument_pool['commit'].append((n,))
		argument_pool['abort'] = []
		for n in range(node_num):
			argument_pool['abort'].append((n,))

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
				for N1, N2, in permutations(node_indices):
					df_data.add((vote_yes[N1], vote_yes[N2], vote_no[N1], vote_no[N2], alive[N1], alive[N2], go_commit[N1], go_commit[N2], go_abort[N1], go_abort[N2], decide_commit[N1], decide_commit[N2], decide_abort[N1], decide_abort[N2], abort_flag[0]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['vote_yes(N1)', 'vote_yes(N2)', 'vote_no(N1)', 'vote_no(N2)', 'alive(N1)', 'alive(N2)', 'go_commit(N1)', 'go_commit(N2)', 'go_abort(N1)', 'go_abort(N2)', 'decide_commit(N1)', 'decide_commit(N2)', 'decide_abort(N1)', 'decide_abort(N2)', 'abort_flag'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/two_phase_commit.csv', index=False)
	print('Simulation finished. Trace written to traces/two_phase_commit.csv')
