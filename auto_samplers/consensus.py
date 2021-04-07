import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def send_request_vote_prec(src, dst):
	return True

def send_request_vote(src, dst):
	for N1 in range(node_num):
		for N2 in range(node_num):
			vote_request_msg[N1, N2] = vote_request_msg[N1, N2] or (N1==src and N2==dst)

def send_vote_prec(src, dst):
	if not ((not voted[src]) and (vote_request_msg[dst, src])):
		return False
	return True

def send_vote(src, dst):
	for N1 in range(node_num):
		for N2 in range(node_num):
			vote_msg[N1, N2] = vote_msg[N1, N2] or (N1 == src and N2 == dst)
	for N in range(node_num):
		voted[N] = voted[N] or N==src

def recv_vote_prec(n, sender):
	if not ((vote_msg[sender, n])):
		return False
	return True

def recv_vote(n, sender):
	for N1 in range(node_num):
		for N2 in range(node_num):
			votes[N1, N2] = votes[N1, N2] or (N1 == n and N2 == sender)

def choose_voting_quorum_prec(q, sn):
	tmp_var_1 = True
	for N in range(node_num):
		if not (not (member[N, q]) or (votes[sn, N])):
			tmp_var_1 = False
			break
	if not (tmp_var_1):
		return False
	return True

def choose_voting_quorum(q, sn):
	for Q in range(quorum_num):
		voting_quorum[Q] = Q == q

def become_leader_prec(n, q):
	tmp_var_2 = True
	for N in range(node_num):
		if not (not (member[N, q]) or (votes[n, N])):
			tmp_var_2 = False
			break
	if not (tmp_var_2):
		return False
	return True

def become_leader(n, q):
	for N in range(node_num):
		for Q in range(quorum_num):
			leader[N, Q] = leader[N, Q] or (N == n and Q == q)

def decide_prec(n, q, v):
	tmp_var_3 = False
	if (leader[n, q]):
		tmp_var_3 = True
		for Q in range(quorum_num):
			for V in range(value_num):
				if not (not decided[n, Q, V]):
					tmp_var_3 = False
					break
	if not (tmp_var_3):
		return False
	return True

def decide(n, q, v):
	for N in range(node_num):
		for Q in range(quorum_num):
			for V in range(value_num):
				decided[N, Q, V] = decided[N, Q, V] or (N==n and Q==q and V==v)

func_from_name = {'send_request_vote': send_request_vote, 'send_request_vote_prec': send_request_vote_prec, 'send_vote': send_vote, 'send_vote_prec': send_vote_prec, 'recv_vote': recv_vote, 'recv_vote_prec': recv_vote_prec, 'choose_voting_quorum': choose_voting_quorum, 'choose_voting_quorum_prec': choose_voting_quorum_prec, 'become_leader': become_leader, 'become_leader_prec': become_leader_prec, 'decide': decide, 'decide_prec': decide_prec}

def instance_generator():
	quorum_num = rng.integers(1, 5)
	node_num = rng.integers(3, 7)
	value_num = rng.integers(1, 5)
	return quorum_num, node_num, value_num

def sample(max_iter=37):
	global quorum_num, node_num, value_num, member, vote_request_msg, voted, vote_msg, votes, leader, voting_quorum, decided
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		quorum_num, node_num, value_num = instance_generator()
		member = rng.integers(0, 2, size=(node_num, quorum_num), dtype=bool)
		vote_request_msg = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		voted = rng.integers(0, 2, size=(node_num), dtype=bool)
		vote_msg = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		votes = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		leader = rng.integers(0, 2, size=(node_num, quorum_num), dtype=bool)
		voting_quorum = rng.integers(0, 2, size=(quorum_num), dtype=bool)
		decided = rng.integers(0, 2, size=(node_num, quorum_num, value_num), dtype=bool)
		member = np.zeros((node_num, quorum_num), dtype=bool)
		for q in range(quorum_num):
			qsize = rng.integers(np.ceil(node_num/2), node_num + 1)
			tmp_members = rng.choice(list(range(node_num)), qsize, replace=False)
			for m in tmp_members:
				member[m,q] = True
		
		for N1 in range(node_num):
			for N2 in range(node_num):
				vote_request_msg[N1, N2] = False
		for N in range(node_num):
			voted[N] = False
		for N1 in range(node_num):
			for N2 in range(node_num):
				vote_msg[N1, N2] = False
		for N1 in range(node_num):
			for N2 in range(node_num):
				votes[N1, N2] = False
		for N in range(node_num):
			for Q in range(quorum_num):
				leader[N, Q] = False
		for N in range(node_num):
			for Q in range(quorum_num):
				for V in range(value_num):
					decided[N, Q, V] = False
		for Q in range(quorum_num):
			voting_quorum[Q] = False

		action_pool = ['send_request_vote', 'send_vote', 'recv_vote', 'choose_voting_quorum', 'become_leader', 'decide']
		argument_pool = dict()
		argument_pool['send_request_vote'] = []
		for src in range(node_num):
			for dst in range(node_num):
				argument_pool['send_request_vote'].append((src, dst))
		argument_pool['send_vote'] = []
		for src in range(node_num):
			for dst in range(node_num):
				argument_pool['send_vote'].append((src, dst))
		argument_pool['recv_vote'] = []
		for n in range(node_num):
			for sender in range(node_num):
				argument_pool['recv_vote'].append((n, sender))
		argument_pool['choose_voting_quorum'] = []
		for q in range(quorum_num):
			for sn in range(node_num):
				argument_pool['choose_voting_quorum'].append((q, sn))
		argument_pool['become_leader'] = []
		for n in range(node_num):
			for q in range(quorum_num):
				argument_pool['become_leader'].append((n, q))
		argument_pool['decide'] = []
		for n in range(node_num):
			for q in range(quorum_num):
				for v in range(value_num):
					argument_pool['decide'].append((n, q, v))

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
				quorum_indices = rng.choice(list(range(quorum_num)), 1, replace=False)
				quorum_indices = sorted(quorum_indices)
				node_indices = rng.choice(list(range(node_num)), 3, replace=False)
				node_indices = sorted(node_indices)
				value_indices = rng.choice(list(range(value_num)), 1, replace=False)
				value_indices = sorted(value_indices)
				for Q1, in permutations(quorum_indices):
					for N1, N2, N3, in permutations(node_indices):
						for V1, in permutations(value_indices):
							df_data.add((member[N1,Q1], member[N2,Q1], member[N3,Q1], vote_request_msg[N1,N1], vote_request_msg[N1,N2], vote_request_msg[N1,N3], vote_request_msg[N2,N1], vote_request_msg[N2,N2], vote_request_msg[N2,N3], vote_request_msg[N3,N1], vote_request_msg[N3,N2], vote_request_msg[N3,N3], voted[N1], voted[N2], voted[N3], vote_msg[N1,N1], vote_msg[N1,N2], vote_msg[N1,N3], vote_msg[N2,N1], vote_msg[N2,N2], vote_msg[N2,N3], vote_msg[N3,N1], vote_msg[N3,N2], vote_msg[N3,N3], votes[N1,N1], votes[N1,N2], votes[N1,N3], votes[N2,N1], votes[N2,N2], votes[N2,N3], votes[N3,N1], votes[N3,N2], votes[N3,N3], leader[N1,Q1], leader[N2,Q1], leader[N3,Q1], voting_quorum[Q1], decided[N1,Q1,V1], decided[N2,Q1,V1], decided[N3,Q1,V1]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 750 or (simulation_round > 15 and df_size_history[-1] == df_size_history[-16])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['member(N1,Q1)', 'member(N2,Q1)', 'member(N3,Q1)', 'vote_request_msg(N1,N1)', 'vote_request_msg(N1,N2)', 'vote_request_msg(N1,N3)', 'vote_request_msg(N2,N1)', 'vote_request_msg(N2,N2)', 'vote_request_msg(N2,N3)', 'vote_request_msg(N3,N1)', 'vote_request_msg(N3,N2)', 'vote_request_msg(N3,N3)', 'voted(N1)', 'voted(N2)', 'voted(N3)', 'vote_msg(N1,N1)', 'vote_msg(N1,N2)', 'vote_msg(N1,N3)', 'vote_msg(N2,N1)', 'vote_msg(N2,N2)', 'vote_msg(N2,N3)', 'vote_msg(N3,N1)', 'vote_msg(N3,N2)', 'vote_msg(N3,N3)', 'votes(N1,N1)', 'votes(N1,N2)', 'votes(N1,N3)', 'votes(N2,N1)', 'votes(N2,N2)', 'votes(N2,N3)', 'votes(N3,N1)', 'votes(N3,N2)', 'votes(N3,N3)', 'leader(N1,Q1)', 'leader(N2,Q1)', 'leader(N3,Q1)', 'voting_quorum(Q1)', 'decided(N1,Q1,V1)', 'decided(N2,Q1,V1)', 'decided(N3,Q1,V1)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/consensus.csv', index=False)
	print('Simulation finished. Trace written to traces/consensus.csv')
