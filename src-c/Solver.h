#ifndef SOLVER_H
#define SOLVER_H

#include "basics.h"
#include "preprocessing.h"
#include "Helper.h"
#include "InvEncoder.h"

#define MAX_CSV_COLUMN 60

class Solver
{
private:
	vector<string> full_predicates;
	DataMatrix init_data_mat;
	map<vars_t, DataMatrix> data_mat_dict;
	map<vars_t, vector<vector<string>>> self_mapped_new_predicates_dict;
	void get_subtables_by_varnum(DataMatrix& init_data_mat);
	void get_column_indices_dict();
	void get_vars_traversal_order();
	void reduce_table(vector<string>& old_predicates, DataMatrix& old_data_mat, vector<string>& new_predicates, DataMatrix& new_data_mat,
		const vector<string>& curr_group_remaining, const string var_to_remove);
	void enumerate_disj(const DataMatrix& data_mat, const vars_t& vars, int max_literal, unordered_set<inv_t, VectorHash>& inv_results, unordered_set<inv_t, VectorHash>& extended_invs);
	bool check_if_inv(const DataMatrix& data_mat, const inv_t& comb);
	void precompute_vars_self_mapping(const vars_t& vars);
	void permute_inv(const inv_t& comb, const vars_t& vars, unordered_set<inv_t, VectorHash>& extended_invs);

protected:  // visible to derived class InvRefiner
	string problem_name;
	Helper helper;
	InvEncoder encoder;
	Config config;
	vector<vector<string>> vars_traversal_order;
	// key of both predicates_dict and data_mat_dict are set of variables, e.g., ['X', 'Y', 'A']
	// suppose there are three same-type groups with size n1 n2 n3, then the number of keys is n1*n2*n3
	// value of predicates_dict is the reduced predicates, predicates that are well-defined given current variables
	// value of data_mat_dict is the reduced data matrix, with ncol == |reduced_predicates|
	map<vars_t, vector<string>> predicates_dict;
	map<vars_t, map<vars_t, vector<vector<int>>>> column_indices_dict;
	map<vars_t, map<string, vector<int>>> var_in_p_dict;
	map<vars_t, map<string, int>> p_to_idx_dict;
	// invs_dict: key: subtemplate; value: checked invariants
	map<vars_t, unordered_set<inv_t, VectorHash>> invs_dict;
	// extended_invs_dict: key: subtemplate; value: checked invariants and invariants projected from lower subtemplates
	map<vars_t, unordered_set<inv_t, VectorHash>> extended_invs_dict;
	void encode_and_output(const string& infile, const string& outfile, map<int, pair<vars_t, inv_t>>& id_to_inv, const vector<string>& more_invs);
	
public:
	vector<vector<string>> extended_same_type_groups;  // groups of variables with the same type, either same-type or total-order in config
	Solver(string problem, int num_attempt);
	void auto_solve();
};

#endif
