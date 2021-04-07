#ifndef INVENCODER_H
#define INVENCODER_H

#include "basics.h"
#include "Helper.h"

class InvEncoder
{
	Helper helper;
	void encode_invs(const vars_t& vars, const vector<string>& predicates, const unordered_set<inv_t, VectorHash>& invs, const vector<vector<string>>& extended_same_type_groups, vector<string>& str_invs, map<int, pair<vars_t, inv_t>>& id_to_inv, int start_idx = 100);
public:
	Config config;
	void encode_invs_dict(const map<vars_t, unordered_set<inv_t, VectorHash>>& invs_dict, const map<vars_t, vector<string>>& predicates_dict, const vector<vector<string>>& extended_same_type_groups, vector<string>& str_invs, map<int, pair<vars_t, inv_t>>& id_to_inv, const vector<string>& more_invs);
	void append_invs_ivy(const string& infile, const string& outfile, const vector<string>& str_invs);
};
#endif // INVENCODER_H
