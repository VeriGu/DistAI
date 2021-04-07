import numpy as np
from collections import defaultdict
from scipy.special import comb
import time
import pandas as pd
from itertools import product, permutations

rng = np.random.default_rng(0)

def join_prec(x, y):
	if not ((not a[x])):
		return False
	if not ((a[y])):
		return False
	if not ((not btw[x, org, y])):
		return False
	return True

def join(x, y):
	a[x] = True
	for Y in range(node_num):
		s1[x, Y] = y == Y
	in_s1[x] = True
	for Y in range(node_num):
		s2[x, Y] = False
	in_s2[x] = False
	for Y in range(node_num):
		p[x, Y] = False

def stabilize_prec(x, y, z):
	if not ((a[x])):
		return False
	if not ((s1[x, y])):
		return False
	if not ((a[y])):
		return False
	if not ((p[y, z])):
		return False
	if not ((btw[x, z, y])):
		return False
	return True

def stabilize(x, y, z):
	for Z in range(node_num):
		s1[x, Z] = Z == z
	in_s1[x] = True
	for Y in range(node_num):
		s2[x, Y] = Y == y
	in_s2[x] = True

def notify_prec(x, y, z):
	if not ((a[x])):
		return False
	if not ((s1[x, y])):
		return False
	if not ((a[y])):
		return False
	tmp_var_1 = True
	for X in range(node_num):
		if not (p[y, z] or not p[y, X]):
			tmp_var_1 = False
			break
	if not (tmp_var_1):
		return False
	if not ((btw[z, x, y])):
		return False
	return True

def notify(x, y, z):
	for X in range(node_num):
		p[y, X] = X == x

def inherit_prec(x, y, z):
	if not ((a[x])):
		return False
	if not ((s1[x, y])):
		return False
	if not ((a[y])):
		return False
	if not ((s1[y, z])):
		return False
	return True

def inherit(x, y, z):
	for Z in range(node_num):
		s2[x, Z] = Z == z
	in_s2[x] = True

def remove_prec(x, y, z):
	if not ((a[x])):
		return False
	if not ((s1[x, y])):
		return False
	if not ((not a[y])):
		return False
	if not ((s2[x, z])):
		return False
	return True

def remove(x, y, z):
	for Z in range(node_num):
		s1[x, Z] = Z == z
	in_s1[x] = True
	for Y in range(node_num):
		s2[x, Y] = False
	in_s2[x] = False

def fail_prec(x):
	if not ((a[x])):
		return False
	if not ((x != org)):
		return False
	tmp_var_2 = True
	for Y in range(node_num):
		if not (not ((s1[Y, x])) or (in_s2[Y])):
			tmp_var_2 = False
			break
	if not (tmp_var_2):
		return False
	tmp_var_3 = True
	for Z in range(node_num):
		for Y in range(node_num):
			if not (not ((s1[Y, x] and s2[Y, Z])) or (a[Z])):
				tmp_var_3 = False
				break
	if not (tmp_var_3):
		return False
	tmp_var_4 = True
	for Y in range(node_num):
		for X in range(node_num):
			if not (not ((s1[X, Y] and s2[X, x])) or ((Y != x and a[Y]))):
				tmp_var_4 = False
				break
	if not (tmp_var_4):
		return False
	return True

def fail(x):
	a[x] = False
	for Y in range(node_num):
		p[x, Y] = False
	for Y in range(node_num):
		s1[x, Y] = False
	in_s1[x] = False
	for Y in range(node_num):
		s2[x, Y] = False
	in_s2[x] = False

def reach_org_prec(x, y, z):
	if not (((s1[x, y] and a[y] and reach[y]) or (s1[x, y] and not a[y] and s2[x, z] and a[z] and reach[z]))):
		return False
	return True

def reach_org(x, y, z):
	reach[x] = True

def remove_org_prec(x, y, z):
	if not ((x != org)):
		return False
	if not ((s1[x, y])):
		return False
	if not ((not a[y] or not reach[y])):
		return False
	tmp_var_5 = True
	for Z in range(node_num):
		if not (not (not a[y]) or ((not s2[x, Z] or s2[x, z]))):
			tmp_var_5 = False
			break
	if not (tmp_var_5):
		return False
	if not ((not ((not a[y] and s2[x, z])) or ((not a[z] or not reach[z])))):
		return False
	return True

def remove_org(x, y, z):
	reach[x] = False

func_from_name = {'join': join, 'join_prec': join_prec, 'stabilize': stabilize, 'stabilize_prec': stabilize_prec, 'notify': notify, 'notify_prec': notify_prec, 'inherit': inherit, 'inherit_prec': inherit_prec, 'remove': remove, 'remove_prec': remove_prec, 'fail': fail, 'fail_prec': fail_prec, 'reach_org': reach_org, 'reach_org_prec': reach_org_prec, 'remove_org': remove_org, 'remove_org_prec': remove_org_prec}

def instance_generator():
	node_num = rng.integers(3, 7)
	return node_num

def sample(max_iter=50):
	global node_num, a, s1, in_s1, s2, in_s2, p, reach, error, org, other, btw
	df_data = set()
	stopping_criteria = False
	simulation_round = 0
	df_size_history = [0]
	while stopping_criteria is False:
		# protocol initialization
		node_num = instance_generator()
		a = rng.integers(0, 2, size=(node_num), dtype=bool)
		s1 = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		in_s1 = rng.integers(0, 2, size=(node_num), dtype=bool)
		s2 = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		in_s2 = rng.integers(0, 2, size=(node_num), dtype=bool)
		p = rng.integers(0, 2, size=(node_num, node_num), dtype=bool)
		reach = rng.integers(0, 2, size=(node_num), dtype=bool)
		error = rng.integers(0, 2, size=(node_num), dtype=bool)
		org = rng.integers(0, node_num)
		other = rng.integers(0, node_num)
		# build ring topology
		btw = np.zeros((node_num, node_num, node_num), dtype=bool)
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					if x != y and x != z and y != z:
						btw[x, y, z] = (x < y < z) | (z < x < y) | (y < z < x)
		other, org = rng.choice(node_num, 2, replace=False)
		
		for X in range(node_num):
			a[X] = X == org or X == other
		for X in range(node_num):
			for Y in range(node_num):
				s1[X, Y] = (X == org and Y == other) or (X == other and Y == org)
		for X in range(node_num):
			in_s1[X] = X == org or X == other
		for X in range(node_num):
			for Y in range(node_num):
				s2[X, Y] = False
		for X in range(node_num):
			in_s2[X] = False
		for X in range(node_num):
			for Y in range(node_num):
				p[X, Y] = (X == org and Y == other) or (X == other and Y == org)
		for X in range(node_num):
			reach[X] = X == org
		for X in range(node_num):
			error[X] = False

		action_pool = ['join', 'stabilize', 'notify', 'inherit', 'remove', 'fail', 'reach_org', 'remove_org']
		argument_pool = dict()
		argument_pool['join'] = []
		for x in range(node_num):
			for y in range(node_num):
				argument_pool['join'].append((x, y))
		argument_pool['stabilize'] = []
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					argument_pool['stabilize'].append((x, y, z))
		argument_pool['notify'] = []
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					argument_pool['notify'].append((x, y, z))
		argument_pool['inherit'] = []
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					argument_pool['inherit'].append((x, y, z))
		argument_pool['remove'] = []
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					argument_pool['remove'].append((x, y, z))
		argument_pool['fail'] = []
		for x in range(node_num):
			argument_pool['fail'].append((x,))
		argument_pool['reach_org'] = []
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					argument_pool['reach_org'].append((x, y, z))
		argument_pool['remove_org'] = []
		for x in range(node_num):
			for y in range(node_num):
				for z in range(node_num):
					argument_pool['remove_org'].append((x, y, z))

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
				for N1, N2, N3, in permutations(node_indices):
					df_data.add((a[N1], a[N2], a[N3], s1[N1,N1], s1[N1,N2], s1[N1,N3], s1[N2,N1], s1[N2,N2], s1[N2,N3], s1[N3,N1], s1[N3,N2], s1[N3,N3], in_s1[N1], in_s1[N2], in_s1[N3], s2[N1,N1], s2[N1,N2], s2[N1,N3], s2[N2,N1], s2[N2,N2], s2[N2,N3], s2[N3,N1], s2[N3,N2], s2[N3,N3], in_s2[N1], in_s2[N2], in_s2[N3], reach[N1], reach[N2], reach[N3], btw[N1,N2,N3], org==N1, org==N2, org==N3, other==N1, other==N2, other==N3))
		simulation_round += 1
		df_size_history.append(len(df_data))
		stopping_criteria = simulation_round > 1000 or (simulation_round > 20 and df_size_history[-1] == df_size_history[-21])
	return list(df_data)

if __name__ == '__main__':
	start_time = time.time()
	df_data = sample()
	df = pd.DataFrame(df_data, columns=['a(N1)', 'a(N2)', 'a(N3)', 's1(N1,N1)', 's1(N1,N2)', 's1(N1,N3)', 's1(N2,N1)', 's1(N2,N2)', 's1(N2,N3)', 's1(N3,N1)', 's1(N3,N2)', 's1(N3,N3)', 'in_s1(N1)', 'in_s1(N2)', 'in_s1(N3)', 's2(N1,N1)', 's2(N1,N2)', 's2(N1,N3)', 's2(N2,N1)', 's2(N2,N2)', 's2(N2,N3)', 's2(N3,N1)', 's2(N3,N2)', 's2(N3,N3)', 'in_s2(N1)', 'in_s2(N2)', 'in_s2(N3)', 'reach(N1)', 'reach(N2)', 'reach(N3)', 'ring.btw(N1,N2,N3)', 'org=N1', 'org=N2', 'org=N3', 'other=N1', 'other=N2', 'other=N3'])
	df = df.drop_duplicates().astype(int)
	end_time = time.time()
	df.to_csv('../traces/chord.csv', index=False)
	print('Simulation finished. Trace written to traces/chord.csv')
