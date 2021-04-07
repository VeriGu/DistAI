import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def request_prec(requester, responder):
	if not ((not requested[requester, responder])):
		return False
	if not ((requester != responder)):
		return False
	return True

def request(requester, responder):
	requested[requester, responder] = True

def reply_prec(requester, responder):
	if not ((not replied[requester, responder])):
		return False
	if not ((not holds[responder])):
		return False
	if not ((not replied[responder, requester])):
		return False
	if not ((requested[requester, responder])):
		return False
	if not ((requester != responder)):
		return False
	return True

def reply(requester, responder):
	requested[requester, responder] = False
	replied[requester, responder] = True

def enter_prec(requester):
	tmp_var_1 = True
	for N in range(node_num):
		if not (not (N != requester) or (replied[requester, N])):
			tmp_var_1 = False
			break
	if not (tmp_var_1):
		return False
	return True

def enter(requester):
	holds[requester] = True

def leave_prec(requester):
	if not ((holds[requester])):
		return False
	return True

def leave(requester):
	holds[requester] = False
	for N in range(node_num):
		replied[requester, N] = False

func_from_name = {'request': request, 'request_prec': request_prec, 'reply': reply, 'reply_prec': reply_prec, 'enter': enter, 'enter_prec': enter_prec, 'leave': leave, 'leave_prec': leave_prec}

def instance_generator():
	node_num = rng.integers(2, 6)
	return node_num

def sample(max_iter=50):
	global node_num, requested, replied, holds
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num = instance_generator()
		requested = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		replied = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		holds = rng.integers(0, 2, size=(node_num), dtype=bool)
		
		for N1 in range(node_num):
			for N2 in range(node_num):
				requested[N1, N2] = False
		for N1 in range(node_num):
			for N2 in range(node_num):
				replied[N1, N2] = False
		for N in range(node_num):
			holds[N] = False

		action_pool = ['request', 'reply', 'enter', 'leave']
		argument_pool = dict()
		argument_pool['request'] = []
		for requester in range(node_num):
			for responder in range(node_num):
				argument_pool['request'].append((requester, responder))
		argument_pool['reply'] = []
		for requester in range(node_num):
			for responder in range(node_num):
				argument_pool['reply'].append((requester, responder))
		argument_pool['enter'] = []
		for requester in range(node_num):
			argument_pool['enter'].append((requester,))
		argument_pool['leave'] = []
		for requester in range(node_num):
			argument_pool['leave'].append((requester,))

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
					df_data.add((requested[N1,N1], requested[N1,N2], requested[N2,N1], requested[N2,N2], replied[N1,N1], replied[N1,N2], replied[N2,N1], replied[N2,N2], holds[N1], holds[N2]))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['requested(N1,N1)', 'requested(N1,N2)', 'requested(N2,N1)', 'requested(N2,N2)', 'replied(N1,N1)', 'replied(N1,N2)', 'replied(N2,N1)', 'replied(N2,N2)', 'holds(N1)', 'holds(N2)'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/Ricart-Agrawala.csv', index=False)
	print('Simulation finished. Trace written to traces/Ricart-Agrawala.csv')
