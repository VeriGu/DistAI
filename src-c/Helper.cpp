#include "Helper.h"

void Helper::parse_predicates(const vector<string>& predicates, map<string, vector<int>>& var_in_p, map<string, int>& p_to_idx)
{
	assert(var_in_p.size() == 0);
	assert(p_to_idx.size() == 0);
	int num_predicates = predicates.size();
	regex pattern("[A-Z]+[0-9]*");
	for (int idx = 0; idx < num_predicates; idx++)
	{
		string p = predicates[idx];
		p_to_idx[p] = idx;
		smatch match;
		string remaining_str = p;
		set<string> variables_in_this_p;
		while (regex_search(remaining_str, match, pattern))
		{
			string variable = match.str(0);
			if (variables_in_this_p.find(variable) != variables_in_this_p.end()) {}
			else
			{
				variables_in_this_p.insert(variable);
				var_in_p[variable].push_back(idx);
				var_in_p[variable].push_back(idx + num_predicates);  // ~p
			}
			remaining_str = match.suffix().str();
		}
	}
}

void Helper::calc_vars_mapping(const vector<string>& old_group, const string& var_to_remove, vector<map<string, string>>& vars_mappings)
{
	bool one_to_one_exists_for_var_to_remove = false;
	string var_to_remove_counterpart;
	if (config.one_to_one_exists)
	{
		map<string, string>::iterator it = config.one_to_one_bidir.find(var_to_remove);
		if (it != config.one_to_one_bidir.end())
		{
			one_to_one_exists_for_var_to_remove = true;
			var_to_remove_counterpart = it->second;
		}
	}

	bool total_order_exists_for_var_to_remove_or_its_counterpart = false;
	if (config.total_order_exists)
	{
		if (std::find(config.total_order.begin(), config.total_order.end(), var_to_remove) != config.total_order.end())
		{
			total_order_exists_for_var_to_remove_or_its_counterpart = true;
		}
		else if (one_to_one_exists_for_var_to_remove && (std::find(config.total_order.begin(), config.total_order.end(), var_to_remove_counterpart) != config.total_order.end()))
		{
			total_order_exists_for_var_to_remove_or_its_counterpart = true;
		}
	}

	vector<vector<string>> vars_combs;
	if (total_order_exists_for_var_to_remove_or_its_counterpart)
	{
		calc_combinations(old_group, old_group.size() - 1, vars_combs);
	}
	else
	{
		calc_permutations(old_group, old_group.size() - 1, vars_combs);
	}

	vector<string> new_group;
	for (const string& var : old_group)
	{
		if (var != var_to_remove) new_group.push_back(var);
	}
	auto new_group_size = new_group.size();
	assert(new_group_size == old_group.size() - 1);
	for (const vector<string>& vars_comb : vars_combs)
	{
		map<string, string> vars_map;
		for (vars_t::size_type i = 0; i < new_group_size; i++)
		{
			vars_map[new_group[i]] = vars_comb[i];
			if (one_to_one_exists_for_var_to_remove)
			{
				vars_map[config.one_to_one_bidir[new_group[i]]] = config.one_to_one_bidir[vars_comb[i]];
			}
		}
		vars_mappings.push_back(vars_map);
	}
}

void Helper::calc_vars_self_mapping(const vars_t& vars, const vector<vector<string>>& extended_same_type_groups, vector<map<string, string>>& vars_mappings)
{
	assert(vars_mappings.size() == 0);
	vector<vector<map<string, string>>> mappings_for_each_group;
	for (const vector<string>& group : extended_same_type_groups)
	{
		vector<string> curr_group;
		for (const string& variable : group)
		{
			if (std::find(vars.begin(), vars.end(), variable) != vars.end())
			{
				curr_group.push_back(variable);
			}
		}
		auto curr_group_size = curr_group.size();
		if (config.total_order_exists && (std::find(config.total_order.begin(), config.total_order.end(), group[0]) != config.total_order.end())) continue;
		if (curr_group_size == 1) continue;
		assert(curr_group_size >= 2);
		vector<map<string, string>> mappings_for_curr_group;
		vector<vector<string>> perms;
		calc_permutations(curr_group, curr_group_size, perms);
		for (const vector<string>& perm : perms)
		{
			map<string, string> curr_mapping;
			assert(curr_group_size == perm.size());
			for (vars_t::size_type i = 0; i < curr_group_size; i++) curr_mapping[curr_group[i]] = perm[i];
			mappings_for_curr_group.push_back(curr_mapping);
		}
		mappings_for_each_group.push_back(mappings_for_curr_group);
	}

	if (mappings_for_each_group.size() == 0) return;

	vector<vector<map<string, string>>> cart_product_result = cart_product(mappings_for_each_group);  // can be optimized
	for (const vector<map<string, string>>& element : cart_product_result)
	{
		map<string, string> tmp_dict;
		for (const map<string, string>& group_dict : element)
		{
			tmp_dict.insert(group_dict.begin(), group_dict.end());
		}
		assert(tmp_dict.size() > 0);
		vars_mappings.push_back(tmp_dict);
	}
}

void Helper::remap_predicates(const vector<string>& old_predicates, const map<string, string>& vars_map, vector<string>& new_predicates)
{
	assert(new_predicates.size() == 0);
	vector<string> old_vars;
	for (auto const& imap : vars_map)
	{
		old_vars.push_back(imap.first);
	}
	string pattern_str;
	join_string(old_vars, '|', pattern_str);
	regex pattern(pattern_str);
	smatch match;
	for (const string& p : old_predicates)
	{
		string remaining_str = p;
		string mapped_str;
		while (regex_search(remaining_str, match, pattern))
		{
			// cout << "prefix: " << match.prefix().str() << endl;
			// cout << "matched: " << match.str(0) << endl;
			// cout << "suffix: " << match.suffix().str() << endl;
			string new_var = vars_map.at(match.str(0));
			mapped_str += match.prefix().str() + new_var;
			remaining_str = match.suffix().str();
		}
		mapped_str += remaining_str;
		new_predicates.push_back(mapped_str);
	}
}

void Helper::generalize_invs(const unordered_set<inv_t, VectorHash>& pred_extended_invs, const vector<vector<int>>& column_indices_list, unordered_set<inv_t, VectorHash>& succ_extended_invs)
{
	assert(succ_extended_invs.size() == 0);
	for (const inv_t& inv : pred_extended_invs)
	{
		int inv_size = inv.size();
		for (const vector<int>& column_indices : column_indices_list)
		{
			inv_t mapped_inv(inv_size);
			for (int i = 0; i < inv_size; i++)
			{
				mapped_inv[i] = column_indices[inv[i]];
			}
			succ_extended_invs.insert(mapped_inv);
		}
	}
}


void Helper::reconstruct_var_group(const vars_t& vars, const vector<vector<string>>& extended_same_type_groups, vector<vector<string>>& vars_grouped)
{
	vars_grouped.clear();
	if (extended_same_type_groups.size() == 0) return;
	vector<vars_t>::const_iterator cur_group = extended_same_type_groups.begin();
	vars_t::const_iterator cur_var = cur_group -> begin();
	bool first = true;
	for (const string& var : vars)
	{
		bool new_group = false;
		for (bool find = false; !find; )
		{
			for (; cur_var != cur_group -> end() && *cur_var != var; cur_var++);
			if (cur_var == cur_group -> end())
			{
				cur_group++;
				cur_var = cur_group -> begin();
				new_group = true;
			}
			else
			{
				find = true;
			}
		}
		if (first || new_group)
		{
			vars_grouped.push_back(vars_t());
		}
		vars_grouped.back().push_back(var);
		first = false;
	}
}