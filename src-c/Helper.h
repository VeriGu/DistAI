#ifndef HELPER_H
#define HELPER_H

#include "basics.h"
#include <regex>

using std::regex; using std::regex_search; using std::smatch;

class Helper
{
public:
	Config config;
	void parse_predicates(const vector<string>& predicates, map<string, vector<int>>& var_in_p, map<string, int>& p_to_idx);
	void calc_vars_mapping(const vector<string>& old_group, const string& var_to_remove, vector<map<string, string>>& vars_mappings);
	void calc_vars_self_mapping(const vars_t& vars, const vector<vector<string>>& extended_same_type_groups, vector<map<string, string>>& vars_mappings);
	void remap_predicates(const vector<string>& old_predicates, const map<string, string>& vars_map, vector<string>& new_predicates);
	void generalize_invs(const unordered_set<inv_t, VectorHash>& pred_extended_invs, const vector<vector<int>>& column_indices_list, unordered_set<inv_t, VectorHash>& succ_extended_invs);
	void reconstruct_var_group(const vars_t& vars, const vector<vector<string>>& extended_same_type_groups, vector<vector<string>>& vars_grouped);
};

#endif
