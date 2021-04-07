#include "Solver.h"

Solver::Solver(string problem, int num_attempt)
{
	problem_name = problem;
	string csv_file = "../traces/" + problem + ".csv";
	string config_file = "../configs/" + problem + ".txt";
	init_data_mat.data_ptr = NULL;
	config.max_literal = 0;
	config.hard = false;
	read_config(config_file, &config);
	config.max_literal = config.max_literal + num_attempt;
	helper.config = config;
	encoder.config = config;
	read_trace(csv_file, full_predicates, init_data_mat);
	add_negation(init_data_mat);
	extended_same_type_groups = config.same_type;
	if (config.total_order_exists)
	{
		extended_same_type_groups.insert(extended_same_type_groups.begin(), config.total_order);
	}
}

void Solver::get_subtables_by_varnum(DataMatrix& init_data_mat)
{
	vector<string> all_vars;
	for (vector<string>& group : extended_same_type_groups)
	{
		all_vars.insert(all_vars.end(), group.begin(), group.end());
	}
	predicates_dict[all_vars] = full_predicates;
	data_mat_dict[all_vars] = init_data_mat;
	for (vector<string>& curr_group : extended_same_type_groups)
	{
		map<vars_t, vector<string>> curr_predicates_dict = predicates_dict; // cannot reassign reference in C++
		map<vars_t, DataMatrix> curr_data_mat_dict = data_mat_dict;
		vector<string> curr_group_remaining = curr_group;
		for (reverse_iterator<vector<string>::iterator> it = curr_group.rbegin(); std::next(it) != curr_group.rend(); it++)
		{
			string var_to_remove = *it;
			map<vars_t, vector<string>> new_predicates_dict;
			map<vars_t, DataMatrix> new_data_mat_dict;
			for (map<vars_t, vector<string>>::iterator it = curr_predicates_dict.begin(); it != curr_predicates_dict.end(); it++)
			{
				const vars_t& vars = it->first;
				vector<string>& predicates = it->second;
				DataMatrix& data_mat = curr_data_mat_dict[vars];
				vector<string> new_predicates;
				DataMatrix new_data_mat = {NULL, 0, 0};
				reduce_table(predicates, data_mat, new_predicates, new_data_mat, curr_group_remaining, var_to_remove);
				vars_t new_vars = vars;
				vars_t::iterator it_remove = std::find(new_vars.begin(), new_vars.end(), var_to_remove);
				assert(it_remove != new_vars.end());
				new_vars.erase(it_remove);
				new_predicates_dict[new_vars] = new_predicates;
				new_data_mat_dict[new_vars] = new_data_mat;
			}
			predicates_dict.insert(new_predicates_dict.begin(), new_predicates_dict.end());
			data_mat_dict.insert(new_data_mat_dict.begin(), new_data_mat_dict.end());
			curr_predicates_dict = new_predicates_dict;
			curr_data_mat_dict = new_data_mat_dict;
			curr_group_remaining.pop_back();
		}
		assert(curr_group_remaining.size() == 1);
	}
}

void Solver::get_column_indices_dict()
{
	for (map<vars_t, vector<string>>::iterator it = predicates_dict.begin(); it != predicates_dict.end(); it++)
	{
		const vars_t& vars = it->first;
		const vector<string>& predicates = it->second;
		map<string, vector<int>> var_in_p;
		map<string, int> p_to_idx;
		helper.parse_predicates(predicates, var_in_p, p_to_idx);
		var_in_p_dict[vars] = var_in_p;
		p_to_idx_dict[vars] = p_to_idx;
	}

	for (map<vars_t, vector<string>>::iterator it = predicates_dict.begin(); it != predicates_dict.end(); it++)
	{
		const vars_t& vars = it->first;
		const vector<string>& predicates = it->second;
		bool ever_extensibe = false;   // only full variables is never extensible
		for (const vector<string>& group : extended_same_type_groups)
		{
			int group_size = group.size();
			bool extensible = false;
			int idx = -1;
			for (int i = 0; i < group_size; i++)
			{
				if (std::find(vars.begin(), vars.end(), group[i]) == vars.end())
				{
					assert(i > 0);              // first variable in each group is always reserved during table reduction
					if (!extensible)
					{
						extensible = true;
						ever_extensibe = true;
						idx = i;
					}
				}
				else assert(!extensible);
			}
			if (extensible)
			{
				string var_to_add = group[idx];
				vars_t super_vars = vars;
				vars_t::const_iterator it = std::find(super_vars.begin(), super_vars.end(), group[idx - 1]);  // predecessor of var_to_add in group
				super_vars.insert(std::next(it), var_to_add);                                                 // insert after the predecessor
				vector<map<string, string>> vars_mappings;
				vector<string> subgroup(group.begin(), group.begin() + idx + 1);
				helper.calc_vars_mapping(subgroup, var_to_add, vars_mappings);

				int num_super_predicates = predicates_dict[super_vars].size();
				for (const map<string, string>& vars_map : vars_mappings)
				{
					vector<string> remapped_predicates;
					helper.remap_predicates(predicates, vars_map, remapped_predicates);
					vector<int> column_indices;
					for (const string& p : remapped_predicates)
					{
						column_indices.push_back(p_to_idx_dict[super_vars][p]);
					}
					int column_indices_half_count = column_indices.size();
					for (int i = 0; i < column_indices_half_count; i++)
					{
						column_indices.push_back(column_indices[i] + num_super_predicates);
					}
					column_indices_dict[vars][super_vars].push_back(column_indices);
				}
			}
		}
		if (!ever_extensibe)
		{
			column_indices_dict[vars];  // just for completeness
		}
	}
}

void Solver::get_vars_traversal_order()
{
	vars_t genesis_vars;
	for (const vector<string>& group : extended_same_type_groups) genesis_vars.push_back(group[0]);
	vars_traversal_order.push_back(genesis_vars);
	for (const vector<string>& group : extended_same_type_groups)
	{
		vector<vars_t> curr_traversal_order = vars_traversal_order;
		int curr_group_size = group.size();
		for (int j = 1; j < curr_group_size; j++)
		{
			vector<vars_t> new_traversal_order;
			for (const vars_t& vars : curr_traversal_order)
			{
				vars_t new_vars = vars;
				vars_t::iterator it = std::find(new_vars.begin(), new_vars.end(), group[j-1]);  // iterator to the (j-1)-th element of the current group in new_vars
				assert(it != new_vars.end());
				new_vars.insert(std::next(it), group[j]);
				new_traversal_order.push_back(new_vars);
			}
			vars_traversal_order.insert(vars_traversal_order.end(), new_traversal_order.begin(), new_traversal_order.end());
			curr_traversal_order = new_traversal_order;
		}
	}
}

void Solver::reduce_table(vector<string>& old_predicates, DataMatrix& old_data_mat, vector<string>& new_predicates, DataMatrix& new_data_mat, const vector<string>& curr_group_remaining, const string var_to_remove)
{
	assert(int(2 * old_predicates.size()) == old_data_mat.ncol);
	assert(std::find(curr_group_remaining.begin(), curr_group_remaining.end(), var_to_remove) != curr_group_remaining.end());
	assert(new_predicates.size() == 0);
	assert(new_data_mat.data_ptr == NULL);
	map<string, vector<int>> var_in_p;
	map<string, int> p_to_idx;
	helper.parse_predicates(old_predicates, var_in_p, p_to_idx);
	int num_predicates = old_predicates.size();
	vector<int> p_to_remove = var_in_p[var_to_remove];
	if (config.one_to_one_exists)
	{
		map<string, string>::iterator it = config.one_to_one_bidir.find(var_to_remove);
		if (it != config.one_to_one_bidir.end())
		{
			vector<int>& additional_p_to_remove = var_in_p[it->second];
			p_to_remove.insert(p_to_remove.end(), additional_p_to_remove.begin(), additional_p_to_remove.end());
		}
	}
	for (int i = 0; i < num_predicates; i++)
	{
		if (std::find(p_to_remove.begin(), p_to_remove.end(), i) == p_to_remove.end())
		{
			new_predicates.push_back(old_predicates[i]);
		}
	}

	vector<map<string, string>> vars_mappings;
	helper.calc_vars_mapping(curr_group_remaining, var_to_remove, vars_mappings);
	vector<vector<int>> new_data_mat_with_duplicates;
	unordered_set<vector<int>, VectorHash> deduplicated_data_mat;
	for (const map<string, string>& vars_map : vars_mappings)
	{
		vector<string> remapped_predicates;
		helper.remap_predicates(new_predicates, vars_map, remapped_predicates);
		vector<int> column_indices;
		for (const string& p : remapped_predicates)
		{
			column_indices.push_back(p_to_idx[p]);
		}
		int column_indices_half_count = column_indices.size();
		for (int i=0; i<column_indices_half_count; i++)
		{
			column_indices.push_back(column_indices[i] + num_predicates);
		}
		for (int i = 0; i < old_data_mat.nrow; i++)
		{
			int* row = old_data_mat.data_ptr[i];
			vector<int> reduced_row(2*column_indices_half_count);
			int k = 0;
			for (int idx : column_indices)
			{
				reduced_row[k++] = row[idx];
			}
			deduplicated_data_mat.insert(reduced_row);
		}
	}
	int nrow = deduplicated_data_mat.size();
	assert(nrow > 0);
	int ncol = (*deduplicated_data_mat.begin()).size();
	assert(ncol > 0);
	new_data_mat.data_ptr = contiguous_2d_array(nrow, ncol);
	new_data_mat.nrow = nrow;
	new_data_mat.ncol = ncol;
	int row_count = 0;
	for (const vector<int>& row : deduplicated_data_mat)
	{
		std::copy(row.begin(), row.end(), new_data_mat.data_ptr[row_count]);
		row_count++;
	}
}

void Solver::enumerate_disj(const DataMatrix& data_mat, const vars_t& vars, int max_literal, unordered_set<inv_t, VectorHash>& inv_results, unordered_set<inv_t, VectorHash>& extended_invs)
{
	/* We assume that the original data matrix is symmetric. All possible self-mutations are included.
	   The current Python implementation does not exploit the fact that when a candidate is proved wrong, so will its equivalent forms be. */
	// int num_sample = data_mat.nrow;
	int num_literal_total = data_mat.ncol;
	assert((num_literal_total > 0) && (num_literal_total % 2 == 0));
	int num_atomic = num_literal_total / 2;
	precompute_vars_self_mapping(vars);
	for (int num_literal_curr = 1; num_literal_curr <= max_literal; num_literal_curr++)
	{
		vector<inv_t> combs;
		vector<int> base_seq(num_literal_total);
		for (int i = 0; i < num_literal_total; i++) base_seq[i] = i;
		calc_combinations(base_seq, num_literal_curr, combs);
		
		vector<int> shifted_comb(num_literal_curr);
		for (const inv_t& comb : combs)
		{
			for (int i = 0; i < num_literal_curr; i++) shifted_comb[i] = comb[i] - num_atomic;
			bool p_or_not_p = false;
			for (int e : shifted_comb)
			{
				if (e >= 0)
				{
					bool found = false;
					for (int ee : comb) if (ee == e) found = true;
					if (found)
					{
						p_or_not_p = true;
						break;
					}
				}
			}
			if (p_or_not_p) continue;

			bool inv_already_in_extended_invs = (extended_invs.find(comb) != extended_invs.end());
			if (inv_already_in_extended_invs) continue;

			bool subcomb_is_inv = false;
			vector<inv_t> subcombs;
			calc_combinations(comb, num_literal_curr - 1, subcombs);

			for (const inv_t& subcomb : subcombs)
			{
				if (extended_invs.find(subcomb) != extended_invs.end())
				{
					extended_invs.insert(comb);
					subcomb_is_inv = true;
					break;
				}
			}
			if (subcomb_is_inv) continue;

			bool comb_is_inv = check_if_inv(data_mat, comb);
			if (comb_is_inv)
			{
				inv_results.insert(comb);
				permute_inv(comb, vars, extended_invs);
				extended_invs.insert(comb);
			}
		}
	}
}

bool Solver::check_if_inv(const DataMatrix& data_mat, const inv_t& comb)
{
	int nrow = data_mat.nrow;
	int** data_ptr = data_mat.data_ptr;
	for (int row = 0; row < nrow; row++)
	{
		bool this_row_is_satisfied = false;
		for (int col : comb)
		{
			if (data_ptr[row][col] != 0)
			{
				this_row_is_satisfied = true;
				break;
			}
		}
		if (!this_row_is_satisfied) return false;
	}
	return true;
}

void Solver::precompute_vars_self_mapping(const vars_t& vars)
{
	const vector<string>& predicates = predicates_dict[vars];
	int num_predicates = predicates.size();
	vector<map<string, string>> vars_mappings;
	helper.calc_vars_self_mapping(vars, extended_same_type_groups, vars_mappings);
	for (const map<string, string>& vars_map : vars_mappings)
	{
		vector<string> new_predicates;
		helper.remap_predicates(predicates, vars_map, new_predicates);
		self_mapped_new_predicates_dict[vars].push_back(new_predicates);
	}
    /* Predicate ring.btw(X,Y,Z) means Y is on the clockwise route from X to Z in a ring topology
	   From the truth value of ring.btw(X,Y,Z), we immediately know the truth value of ring.btw(X,Z,Y), ring.btw(Z,X,Y), etc.
	   You can extend the following code block if you introduce your own module with similar properties */
	if ((p_to_idx_dict[vars].find("ring.btw(X,Y,Z)") != p_to_idx_dict[vars].end()) && (p_to_idx_dict[vars].find("ring.btw(X,Z,Y)") == p_to_idx_dict[vars].end()))
	{
		p_to_idx_dict[vars]["ring.btw(X,Z,Y)"] = p_to_idx_dict[vars]["ring.btw(X,Y,Z)"] + num_predicates;
		p_to_idx_dict[vars]["ring.btw(Y,X,Z)"] = p_to_idx_dict[vars]["ring.btw(X,Y,Z)"] + num_predicates;
		p_to_idx_dict[vars]["ring.btw(Y,Z,X)"] = p_to_idx_dict[vars]["ring.btw(X,Y,Z)"];
		p_to_idx_dict[vars]["ring.btw(Z,X,Y)"] = p_to_idx_dict[vars]["ring.btw(X,Y,Z)"];
		p_to_idx_dict[vars]["ring.btw(Z,Y,X)"] = p_to_idx_dict[vars]["ring.btw(X,Y,Z)"] + num_predicates;
	}
}

void Solver::permute_inv(const inv_t& inv, const vars_t& vars, unordered_set<inv_t, VectorHash>& extended_invs)
{
	int inv_size = inv.size();
	for (const vector<string>& new_predicates : self_mapped_new_predicates_dict[vars])
	{
		inv_t new_inv(inv_size);
		int num_predicates = new_predicates.size();
		for (int i = 0; i < inv_size; i++)
		{
			int idx = inv[i];
			if (idx < num_predicates)
			{
				new_inv[i] = p_to_idx_dict[vars][new_predicates[idx]];
			}
			else
			{
				new_inv[i] = (p_to_idx_dict[vars][new_predicates[idx - num_predicates]] + num_predicates) % (2 * num_predicates);
			}
		}
		std::sort(new_inv.begin(), new_inv.end());
		extended_invs.insert(new_inv);
	}
}

void Solver::auto_solve()
{
	get_subtables_by_varnum(init_data_mat);
	get_column_indices_dict();
	get_vars_traversal_order();

	for (const vars_t& vars : vars_traversal_order)
	{
		const DataMatrix& data_mat = data_mat_dict[vars];
		unordered_set<inv_t, VectorHash> invs;
		enumerate_disj(data_mat, vars, config.max_literal, invs, extended_invs_dict[vars]);
		invs_dict[vars] = invs;
		for (map<vars_t, vector<vector<int>>>::iterator it = column_indices_dict[vars].begin(); it != column_indices_dict[vars].end(); it++)
		{
			const vars_t& successor = it->first;
			const vector<vector<int>>& column_indices_list = it->second;
			unordered_set<inv_t, VectorHash> new_extended_invs;
			helper.generalize_invs(extended_invs_dict[vars], column_indices_list, new_extended_invs);
			extended_invs_dict[successor].insert(new_extended_invs.begin(), new_extended_invs.end());
		}
	}
	cout << "Invariant enumeration finished" << endl;
}

void Solver::encode_and_output(const string& infile, const string& outfile, map<int, pair<vars_t, inv_t>>& id_to_inv, const vector<string>& more_invs)
{
	vector<string> str_invs;
	encoder.encode_invs_dict(invs_dict, predicates_dict, extended_same_type_groups, str_invs, id_to_inv, more_invs);
	encoder.append_invs_ivy(infile, outfile, str_invs);
}
