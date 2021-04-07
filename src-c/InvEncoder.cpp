#include "InvEncoder.h"

void InvEncoder::encode_invs_dict(const map<vars_t, unordered_set<inv_t, VectorHash>>& invs_dict, const map<vars_t, vector<string>>& predicates_dict, const vector<vector<string>>& extended_same_type_groups, vector<string>& str_invs, map<int, pair<vars_t, inv_t>>& id_to_inv, const vector<string>& more_invs)
{
	// encode an entire inv_dict into a vector of strings, each represents an invariant
	// The API is different from the Python version, "use_refined_invs" can be ignored
	// invs_dict maps vars (e.g., ['N1', 'N2', 'K0']) to invariants, each invariant is a vector of indices (e.g., [3,4,7])
	// predicates_dict maps vars to predicates, each predicate is a string (e.g., 'replied(N1,N2)'). The indices [3,4,7] above mean the 3rd,4th,7th predicates here
	assert(str_invs.size() == 0);  // the results will be written into str_invs
	int start_idx = 1000;
	for (map<vars_t, unordered_set<inv_t, VectorHash>>::const_iterator it = invs_dict.begin(); it != invs_dict.end(); it++)
	{
		const vars_t& vars = it->first;
		const unordered_set<inv_t, VectorHash>& invs = it->second;
		vector<string> str_invs_for_this_vars;
		encode_invs(vars, predicates_dict.at(vars), invs, extended_same_type_groups, str_invs_for_this_vars, id_to_inv, start_idx);
		str_invs.insert(str_invs.end(), str_invs_for_this_vars.begin(), str_invs_for_this_vars.end());
		start_idx += 1000;
	}
	for (const string& more_inv : more_invs) 
	{
		str_invs.push_back("invariant [" + std::to_string(start_idx) + "] " + more_inv);
		start_idx++;
	}
}

void InvEncoder::encode_invs(const vars_t& vars, const vector<string>& predicates, const unordered_set<inv_t, VectorHash>& invs, const vector<vector<string>>& extended_same_type_groups, vector<string>& str_invs, map<int, pair<vars_t, inv_t>>& id_to_inv, int start_idx)
{
	// called by encode_invs_dict, "prefix" in Python can be ignored
	assert(str_invs.size() == 0);  // the results will be written into str_invs
	vector<string> all_predicates(predicates);
	for (const string& p : predicates)
	{
		all_predicates.push_back("~" + p);
	}
	map<string, vector<int>> var_in_p;
	map<string, int> p_to_idx;
	helper.parse_predicates(predicates, var_in_p, p_to_idx);
	for (const inv_t& inv : invs)
	{
		assert(id_to_inv.find(start_idx) == id_to_inv.end());
		id_to_inv[start_idx] = pair<vars_t,inv_t>(vars, inv);
		string line = "invariant [" + std::to_string(start_idx++) + "] ";
		vector<string> curr_predicates;
		for (const int& e : inv)
		{
			curr_predicates.push_back(all_predicates[e]);
		}
		vector<string> prefix_segments;
		if (config.one_to_one_exists && config.one_to_one.size() > 0)
		{
			vector<string> pairs;
			for (const auto& x : var_in_p)
			{
				string var = x.first;
				if (config.one_to_one.count(var) > 0)
				{
					pairs.push_back(config.one_to_one_f + "(" + var + ")=" + config.one_to_one[var]);
				}
			}
			string s_join;
			join_string(pairs, " & ", s_join);
			prefix_segments.push_back(s_join);
		}
		vector<vector<string>> vars_grouped;

		helper.reconstruct_var_group(vars, extended_same_type_groups, vars_grouped);

		for (const vector<string>& group : vars_grouped)
		{
			assert(group.size() > 0);
			vector<string> pairs;
			if(config.total_order_exists && std::count(config.total_order.begin(), config.total_order.end(), group[0]) > 0)
			{
				for (vars_t::size_type j = 0; j < group.size() - 1; j++)
				{
					pairs.push_back("le(" + group[j] + ", " + group[j + 1] + ") & " + group[j] + " ~= " + group[j + 1]);
				}
			}
			else
			{
				for (vars_t::size_type j = 0; j < group.size() - 1; j++)
				{
					for (vars_t::size_type k = j + 1; k < group.size(); k++)
					{
						pairs.push_back(group[j] + " ~= " + group[k]);
					}
				}		
			}
			if (pairs.size() > 0)
			{
				string s_join;
				join_string(pairs, " & ", s_join);
				prefix_segments.push_back(s_join);
			}
		}
		if (prefix_segments.size() > 0)
		{
			string prefix_join;
			join_string(prefix_segments, " & ", prefix_join);
			line += prefix_join + " -> ";
		}
		string curr_join;
		join_string(curr_predicates, " | ", curr_join);
		line += curr_join;
		str_invs.push_back(line);
	}
}

void InvEncoder::append_invs_ivy(const string& infile, const string& outfile, const vector<string>& str_invs)
{
	// read an Ivy file, and append the invariants to the end
	// learned from https://stackoverflow.com/questions/10195343/copy-a-file-in-a-sane-safe-and-efficient-way
	ifstream source(infile, std::ios::binary);
	ofstream dest(outfile, std::ios::binary);
	dest << source.rdbuf();
	for (const string& str_inv : str_invs)
	{
		dest << str_inv << endl;
	}
}
