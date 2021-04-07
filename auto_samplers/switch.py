import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def new_packet_prec(p):
	return True

def new_packet(p):
	pending[p, src[p], src[p]] = True

def flood_prec(p, sw0, sw1, sw2):
	if not ((pending[p, sw0, sw1])):
		return False
	if not ((not route_dom[dst[p], sw1])):
		return False
	return True

def flood(p, sw0, sw1, sw2):
	if (not route_dom[src[p], sw1]) and (src[p] != sw1):
		route_dom[src[p], sw1] = True
		for X in range(node_num):
			for Y in range(node_num):
				route_tc[src[p], X, Y] = route_tc[src[p], X, Y] or (route_tc[src[p], X, sw1] and route_tc[src[p], sw0, Y])
	if (dst[p] != sw1):
		route_dom[src[p], sw1] = True
		for X in range(node_num):
			for Y in range(node_num):
				route_tc[src[p], X, Y] = route_tc[src[p], X, Y] or (route_tc[src[p], X, sw1] and route_tc[src[p], sw0, Y])
		for Y in range(node_num):
			pending[p, sw1, Y] = link[sw1, Y] and Y != sw0

def route_prec(p, sw0, sw1, sw2):
	if not ((pending[p, sw0, sw1])):
		return False
	if not ((route_dom[dst[p], sw1])):
		return False
	tmp_var_1 = True
	for Z in range(node_num):
		if not (not ((route_tc[dst[p], sw1, Z] and sw1 != Z)) or (route_tc[dst[p], sw2, Z])):
			tmp_var_1 = False
			break
	if not (tmp_var_1):
		return False
	return True

def route(p, sw0, sw1, sw2):
	if (not route_dom[src[p], sw1]) and (src[p] != sw1):
		route_dom[src[p], sw1] = True
		for X in range(node_num):
			for Y in range(node_num):
				route_tc[src[p], X, Y] = route_tc[src[p], X, Y] or (route_tc[src[p], X, sw1] and route_tc[src[p], sw0, Y])
	if (dst[p] != sw1):
		route_dom[src[p], sw1] = True
		for X in range(node_num):
			for Y in range(node_num):
				route_tc[src[p], X, Y] = route_tc[src[p], X, Y] or (route_tc[src[p], X, sw1] and route_tc[src[p], sw0, Y])
		pending[p, sw1, sw2] = True

func_from_name = {'new_packet': new_packet, 'new_packet_prec': new_packet_prec, 'flood': flood, 'flood_prec': flood_prec, 'route': route, 'route_prec': route_prec}

def instance_generator():
	packet_num = rng.integers(1, 5)
	node_num = rng.integers(3, 7)
	return packet_num, node_num

def sample(max_iter=50):
	global packet_num, node_num, pending, link, route_dom, route_tc, src, dst
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		packet_num, node_num = instance_generator()
		pending = rng.integers(0, 2, size=(packet_num, node_num, node_num), dtype=bool)
		link = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		route_dom = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		route_tc = rng.integers(0, 2, size=(node_num, node_num, node_num), dtype=bool)
		src = rng.integers(0, node_num, size=(packet_num))
		dst = rng.integers(0, node_num, size=(packet_num))
		for i in range(node_num):
			link[i,i] = False
		for i in range(1, node_num):
			for j in range(0, i):
				link[i,j] = link[j,i]
		
		for N in range(node_num):
			for X in range(node_num):
				route_dom[N, X] = False
		for N in range(node_num):
			for X in range(node_num):
				for Y in range(node_num):
					route_tc[N, X, Y] = X == Y
		for P in range(packet_num):
			for S in range(node_num):
				for T in range(node_num):
					pending[P, S, T] = False

		action_pool = ['new_packet', 'flood', 'route']
		argument_pool = dict()
		argument_pool['new_packet'] = []
		for p in range(packet_num):
			argument_pool['new_packet'].append((p,))
		argument_pool['flood'] = []
		for p in range(packet_num):
			for sw0 in range(node_num):
				for sw1 in range(node_num):
					for sw2 in range(node_num):
						argument_pool['flood'].append((p, sw0, sw1, sw2))
		argument_pool['route'] = []
		for p in range(packet_num):
			for sw0 in range(node_num):
				for sw1 in range(node_num):
					for sw2 in range(node_num):
						argument_pool['route'].append((p, sw0, sw1, sw2))

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
				packet_indices = rng.choice(list(range(packet_num)), 1, replace=False)
				packet_indices = sorted(packet_indices)
				node_indices = rng.choice(list(range(node_num)), 3, replace=False)
				node_indices = sorted(node_indices)
				for P1, in permutations(packet_indices):
					for N1, N2, N3, in permutations(node_indices):
						df_data.add((pending[P1,N1,N1], pending[P1,N1,N2], pending[P1,N1,N3], pending[P1,N2,N1], pending[P1,N2,N2], pending[P1,N2,N3], pending[P1,N3,N1], pending[P1,N3,N2], pending[P1,N3,N3], link[N1,N1], link[N1,N2], link[N1,N3], link[N2,N1], link[N2,N2], link[N2,N3], link[N3,N1], link[N3,N2], link[N3,N3], route_dom[N1,N1], route_dom[N1,N2], route_dom[N1,N3], route_dom[N2,N1], route_dom[N2,N2], route_dom[N2,N3], route_dom[N3,N1], route_dom[N3,N2], route_dom[N3,N3], route_tc[N1,N1,N1], route_tc[N1,N1,N2], route_tc[N1,N1,N3], route_tc[N1,N2,N1], route_tc[N1,N2,N2], route_tc[N1,N2,N3], route_tc[N1,N3,N1], route_tc[N1,N3,N2], route_tc[N1,N3,N3], route_tc[N2,N1,N1], route_tc[N2,N1,N2], route_tc[N2,N1,N3], route_tc[N2,N2,N1], route_tc[N2,N2,N2], route_tc[N2,N2,N3], route_tc[N2,N3,N1], route_tc[N2,N3,N2], route_tc[N2,N3,N3], route_tc[N3,N1,N1], route_tc[N3,N1,N2], route_tc[N3,N1,N3], route_tc[N3,N2,N1], route_tc[N3,N2,N2], route_tc[N3,N2,N3], route_tc[N3,N3,N1], route_tc[N3,N3,N2], route_tc[N3,N3,N3], src[P1]==N1, src[P1]==N2, src[P1]==N3, dst[P1]==N1, dst[P1]==N2, dst[P1]==N3))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['pending(P1,N1,N1)', 'pending(P1,N1,N2)', 'pending(P1,N1,N3)', 'pending(P1,N2,N1)', 'pending(P1,N2,N2)', 'pending(P1,N2,N3)', 'pending(P1,N3,N1)', 'pending(P1,N3,N2)', 'pending(P1,N3,N3)', 'link(N1,N1)', 'link(N1,N2)', 'link(N1,N3)', 'link(N2,N1)', 'link(N2,N2)', 'link(N2,N3)', 'link(N3,N1)', 'link(N3,N2)', 'link(N3,N3)', 'route_dom(N1,N1)', 'route_dom(N1,N2)', 'route_dom(N1,N3)', 'route_dom(N2,N1)', 'route_dom(N2,N2)', 'route_dom(N2,N3)', 'route_dom(N3,N1)', 'route_dom(N3,N2)', 'route_dom(N3,N3)', 'route_tc(N1,N1,N1)', 'route_tc(N1,N1,N2)', 'route_tc(N1,N1,N3)', 'route_tc(N1,N2,N1)', 'route_tc(N1,N2,N2)', 'route_tc(N1,N2,N3)', 'route_tc(N1,N3,N1)', 'route_tc(N1,N3,N2)', 'route_tc(N1,N3,N3)', 'route_tc(N2,N1,N1)', 'route_tc(N2,N1,N2)', 'route_tc(N2,N1,N3)', 'route_tc(N2,N2,N1)', 'route_tc(N2,N2,N2)', 'route_tc(N2,N2,N3)', 'route_tc(N2,N3,N1)', 'route_tc(N2,N3,N2)', 'route_tc(N2,N3,N3)', 'route_tc(N3,N1,N1)', 'route_tc(N3,N1,N2)', 'route_tc(N3,N1,N3)', 'route_tc(N3,N2,N1)', 'route_tc(N3,N2,N2)', 'route_tc(N3,N2,N3)', 'route_tc(N3,N3,N1)', 'route_tc(N3,N3,N2)', 'route_tc(N3,N3,N3)', 'src(P1)=N1', 'src(P1)=N2', 'src(P1)=N3', 'dst(P1)=N1', 'dst(P1)=N2', 'dst(P1)=N3'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/switch.csv', index=False)
	print('Simulation finished. Trace written to traces/switch.csv')
