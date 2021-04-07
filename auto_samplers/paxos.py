import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def send_a_prec(r):
	if not ((r != none)):
		return False
	return True

def send_a(r):
	one_a[r] = True

def join_round_case1_prec(n, r, maxr, v):
	if not ((r != none)):
		return False
	if not ((one_a[r])):
		return False
	if not ((not left_rnd[n,r])):
		return False
	if not ((maxr == none)):
		return False
	tmp_var_1 = True
	for MAXR in range(round_num):
		for V in range(value_num):
			if not (not (not (r <= MAXR) and vote[n,MAXR,V])):
				tmp_var_1 = False
				break
	if not (tmp_var_1):
		return False
	return True

def join_round_case1(n, r, maxr, v):
	one_b_max_vote[n,r,maxr,v] = True
	one_b[n,r] = True
	for R in range(round_num):
		left_rnd[n,R] = left_rnd[n,R] or not (r <= R)

def join_round_case2_prec(n, r, maxr, v):
	if not ((r != none)):
		return False
	if not ((one_a[r])):
		return False
	if not ((not left_rnd[n,r])):
		return False
	if not ((maxr != none)):
		return False
	if not ((not (r <= maxr))):
		return False
	if not ((vote[n,maxr,v])):
		return False
	tmp_var_2 = True
	for MAXR in range(round_num):
		for V in range(value_num):
			if not (not ((not (r <= MAXR) and vote[n,MAXR,V])) or ((MAXR <= maxr))):
				tmp_var_2 = False
				break
	if not (tmp_var_2):
		return False
	return True

def join_round_case2(n, r, maxr, v):
	one_b_max_vote[n,r,maxr,v] = True
	one_b[n,r] = True
	for R in range(round_num):
		left_rnd[n,R] = left_rnd[n,R] or not (r <= R)

def propose_case1_prec(r, q, maxr, v):
	if not ((r != none)):
		return False
	tmp_var_3 = True
	for V in range(value_num):
		if not (not proposal[r,V]):
			tmp_var_3 = False
			break
	if not (tmp_var_3):
		return False
	tmp_var_4 = True
	for N in range(node_num):
		if not (not (member[N, q]) or (one_b[N,r])):
			tmp_var_4 = False
			break
	if not (tmp_var_4):
		return False
	if not ((maxr == none)):
		return False
	tmp_var_5 = True
	for N in range(node_num):
		for MAXR in range(round_num):
			for V in range(value_num):
				if not (not (member[N, q] and not (r <= MAXR) and vote[N,MAXR,V])):
					tmp_var_5 = False
					break
	if not (tmp_var_5):
		return False
	return True

def propose_case1(r, q, maxr, v):
	proposal[r, v] = True

def propose_case2_prec(r, q, maxr, v):
	if not ((r != none)):
		return False
	tmp_var_6 = True
	for V in range(value_num):
		if not (not proposal[r,V]):
			tmp_var_6 = False
			break
	if not (tmp_var_6):
		return False
	tmp_var_7 = True
	for N in range(node_num):
		if not (not (member[N, q]) or (one_b[N,r])):
			tmp_var_7 = False
			break
	if not (tmp_var_7):
		return False
	if not ((maxr != none)):
		return False
	tmp_var_8 = tmp_var_9 = False
	if (not (r <= maxr)):
		tmp_var_8 = False
		for N in range(node_num):
			if (member[N, q]):
				tmp_var_8 = True
				break
		if tmp_var_8:
			tmp_var_9 = True
			for N in range(node_num):
				if not (vote[N,maxr,v]):
					tmp_var_9 = False
					break
	if not (tmp_var_8 and tmp_var_9):
		return False
	tmp_var_10 = True
	for N in range(node_num):
		for MAXR in range(round_num):
			for V in range(value_num):
				if not (not ((member[N, q] and not (r <= MAXR) and vote[N,MAXR,V])) or ((MAXR <= maxr))):
					tmp_var_10 = False
					break
	if not (tmp_var_10):
		return False
	return True

def propose_case2(r, q, maxr, v):
	proposal[r, v] = True

def cast_vote_prec(n, v, r):
	if not ((r != none)):
		return False
	if not ((not left_rnd[n,r])):
		return False
	if not ((proposal[r, v])):
		return False
	return True

def cast_vote(n, v, r):
	vote[n, r, v] = True

def decide_prec(n, r, v, q):
	if not ((r != none)):
		return False
	tmp_var_11 = True
	for N in range(node_num):
		if not (not (member[N, q]) or (vote[N, r, v])):
			tmp_var_11 = False
			break
	if not (tmp_var_11):
		return False
	return True

def decide(n, r, v, q):
	decision[n, r, v] = True

func_from_name = {'send_a': send_a, 'send_a_prec': send_a_prec, 'join_round_case1': join_round_case1, 'join_round_case1_prec': join_round_case1_prec, 'join_round_case2': join_round_case2, 'join_round_case2_prec': join_round_case2_prec, 'propose_case1': propose_case1, 'propose_case1_prec': propose_case1_prec, 'propose_case2': propose_case2, 'propose_case2_prec': propose_case2_prec, 'cast_vote': cast_vote, 'cast_vote_prec': cast_vote_prec, 'decide': decide, 'decide_prec': decide_prec}

def instance_generator():
	node_num = rng.integers(4, 8)
	value_num = rng.integers(1, 5)
	quorum_num = rng.integers(2, 6)
	round_num = rng.integers(2, 6)
	return node_num, value_num, quorum_num, round_num

def sample(max_iter=37):
	global node_num, value_num, quorum_num, round_num, member, one_a, one_b_max_vote, one_b, left_rnd, proposal, vote, decision, none
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num, value_num, quorum_num, round_num = instance_generator()
		member = rng.integers(0, 2, size=(node_num, quorum_num), dtype=bool)
		one_a = rng.integers(0, 2, size=(round_num), dtype=bool)
		one_b_max_vote = rng.integers(0, 2, size=(node_num, round_num, round_num, value_num), dtype=bool)
		one_b = rng.integers(0, 2, size=(node_num, round_num), dtype=bool)
		left_rnd = rng.integers(0, 2, size=(node_num, round_num), dtype=bool)
		proposal = rng.integers(0, 2, size=(round_num, value_num), dtype=bool)
		vote = rng.integers(0, 2, size=(node_num, round_num, value_num), dtype=bool)
		decision = rng.integers(0, 2, size=(node_num, round_num, value_num), dtype=bool)
		none = rng.integers(0, round_num)
		member = np.zeros((node_num, quorum_num), dtype=bool)
		for q in range(quorum_num):
			qsize = rng.integers(np.ceil(node_num/2), node_num + 1)
			tmp_members = rng.choice(list(range(node_num)), qsize, replace=False)
			for m in tmp_members:
				member[m,q] = True
		
		for R in range(round_num):
			one_a[R] = False
		for N in range(node_num):
			for R1 in range(round_num):
				for R2 in range(round_num):
					for V in range(value_num):
						one_b_max_vote[N,R1,R2,V] = False
		for N in range(node_num):
			for R in range(round_num):
				one_b[N,R] = False
		for N in range(node_num):
			for R in range(round_num):
				left_rnd[N,R] = False
		for R in range(round_num):
			for V in range(value_num):
				proposal[R,V] = False
		for N in range(node_num):
			for R in range(round_num):
				for V in range(value_num):
					vote[N,R,V] = False
		for N in range(node_num):
			for R in range(round_num):
				for V in range(value_num):
					decision[N,R,V] = False

		action_pool = ['send_a', 'join_round_case1', 'join_round_case2', 'propose_case1', 'propose_case2', 'cast_vote', 'decide']
		argument_pool = dict()
		argument_pool['send_a'] = []
		for r in range(round_num):
			argument_pool['send_a'].append((r,))
		argument_pool['join_round_case1'] = []
		for n in range(node_num):
			for r in range(round_num):
				for maxr in range(round_num):
					for v in range(value_num):
						argument_pool['join_round_case1'].append((n, r, maxr, v))
		argument_pool['join_round_case2'] = []
		for n in range(node_num):
			for r in range(round_num):
				for maxr in range(round_num):
					for v in range(value_num):
						argument_pool['join_round_case2'].append((n, r, maxr, v))
		argument_pool['propose_case1'] = []
		for r in range(round_num):
			for q in range(quorum_num):
				for maxr in range(round_num):
					for v in range(value_num):
						argument_pool['propose_case1'].append((r, q, maxr, v))
		argument_pool['propose_case2'] = []
		for r in range(round_num):
			for q in range(quorum_num):
				for maxr in range(round_num):
					for v in range(value_num):
						argument_pool['propose_case2'].append((r, q, maxr, v))
		argument_pool['cast_vote'] = []
		for n in range(node_num):
			for v in range(value_num):
				for r in range(round_num):
					argument_pool['cast_vote'].append((n, v, r))
		argument_pool['decide'] = []
		for n in range(node_num):
			for r in range(round_num):
				for v in range(value_num):
					for q in range(quorum_num):
						argument_pool['decide'].append((n, r, v, q))

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
				value_indices = rng.choice(list(range(value_num)), 1, replace=False)
				value_indices = sorted(value_indices)
				quorum_indices = rng.choice(list(range(quorum_num)), 2, replace=False)
				quorum_indices = sorted(quorum_indices)
				round_indices = rng.choice(list(range(round_num)), 2, replace=False)
				round_indices = sorted(round_indices)
				for N1, N2, N3, N4, in permutations(node_indices):
					for V1, in permutations(value_indices):
						for Q1, Q2, in permutations(quorum_indices):
							R1, R2, = round_indices
							df_data.add((member[N1,Q1], member[N1,Q2], member[N2,Q1], member[N2,Q2], member[N3,Q1], member[N3,Q2], member[N4,Q1], member[N4,Q2], one_a[R1], one_a[R2], one_b_max_vote[N1,R1,R1,V1], one_b_max_vote[N1,R1,R2,V1], one_b_max_vote[N1,R2,R1,V1], one_b_max_vote[N1,R2,R2,V1], one_b_max_vote[N2,R1,R1,V1], one_b_max_vote[N2,R1,R2,V1], one_b_max_vote[N2,R2,R1,V1], one_b_max_vote[N2,R2,R2,V1], one_b_max_vote[N3,R1,R1,V1], one_b_max_vote[N3,R1,R2,V1], one_b_max_vote[N3,R2,R1,V1], one_b_max_vote[N3,R2,R2,V1], one_b_max_vote[N4,R1,R1,V1], one_b_max_vote[N4,R1,R2,V1], one_b_max_vote[N4,R2,R1,V1], one_b_max_vote[N4,R2,R2,V1], one_b[N1,R1], one_b[N1,R2], one_b[N2,R1], one_b[N2,R2], one_b[N3,R1], one_b[N3,R2], one_b[N4,R1], one_b[N4,R2], left_rnd[N1,R1], left_rnd[N1,R2], left_rnd[N2,R1], left_rnd[N2,R2], left_rnd[N3,R1], left_rnd[N3,R2], left_rnd[N4,R1], left_rnd[N4,R2], proposal[R1,V1], proposal[R2,V1], vote[N1,R1,V1], vote[N1,R2,V1], vote[N2,R1,V1], vote[N2,R2,V1], vote[N3,R1,V1], vote[N3,R2,V1], vote[N4,R1,V1], vote[N4,R2,V1], decision[N1,R1,V1], decision[N1,R2,V1], decision[N2,R1,V1], decision[N2,R2,V1], decision[N3,R1,V1], decision[N3,R2,V1], decision[N4,R1,V1], decision[N4,R2,V1]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 750 or (simulation_round > 15 and df_size_history[-1] == df_size_history[-16])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['member(N1,Q1)', 'member(N1,Q2)', 'member(N2,Q1)', 'member(N2,Q2)', 'member(N3,Q1)', 'member(N3,Q2)', 'member(N4,Q1)', 'member(N4,Q2)', 'one_a(R1)', 'one_a(R2)', 'one_b_max_vote(N1,R1,R1,V1)', 'one_b_max_vote(N1,R1,R2,V1)', 'one_b_max_vote(N1,R2,R1,V1)', 'one_b_max_vote(N1,R2,R2,V1)', 'one_b_max_vote(N2,R1,R1,V1)', 'one_b_max_vote(N2,R1,R2,V1)', 'one_b_max_vote(N2,R2,R1,V1)', 'one_b_max_vote(N2,R2,R2,V1)', 'one_b_max_vote(N3,R1,R1,V1)', 'one_b_max_vote(N3,R1,R2,V1)', 'one_b_max_vote(N3,R2,R1,V1)', 'one_b_max_vote(N3,R2,R2,V1)', 'one_b_max_vote(N4,R1,R1,V1)', 'one_b_max_vote(N4,R1,R2,V1)', 'one_b_max_vote(N4,R2,R1,V1)', 'one_b_max_vote(N4,R2,R2,V1)', 'one_b(N1,R1)', 'one_b(N1,R2)', 'one_b(N2,R1)', 'one_b(N2,R2)', 'one_b(N3,R1)', 'one_b(N3,R2)', 'one_b(N4,R1)', 'one_b(N4,R2)', 'left_rnd(N1,R1)', 'left_rnd(N1,R2)', 'left_rnd(N2,R1)', 'left_rnd(N2,R2)', 'left_rnd(N3,R1)', 'left_rnd(N3,R2)', 'left_rnd(N4,R1)', 'left_rnd(N4,R2)', 'proposal(R1,V1)', 'proposal(R2,V1)', 'vote(N1,R1,V1)', 'vote(N1,R2,V1)', 'vote(N2,R1,V1)', 'vote(N2,R2,V1)', 'vote(N3,R1,V1)', 'vote(N3,R2,V1)', 'vote(N4,R1,V1)', 'vote(N4,R2,V1)', 'decision(N1,R1,V1)', 'decision(N1,R2,V1)', 'decision(N2,R1,V1)', 'decision(N2,R2,V1)', 'decision(N3,R1,V1)', 'decision(N3,R2,V1)', 'decision(N4,R1,V1)', 'decision(N4,R2,V1)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/paxos.csv', index=False)
	print('Simulation finished. Trace written to traces/paxos.csv')
