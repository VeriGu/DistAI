#ifndef INVREFINER_H
#define INVREFINER_H

#define REFINE_EXTEND_SECCESSOR false
#define REFINE_EXTEND_DISJUNCTION false

#include "Solver.h"

#define SAFETY_PROPERTY_ID 1000000
#define IVY_CHECK_PATH "/home/me/anaconda3/envs/py2/bin/ivy_check"  // change this to absolute path of ivy_check on your machine

class InvRefiner:
	public Solver
{
private:
	int ivy_call_count;
	int countereg_count;
	map<int, pair<vars_t, inv_t>> id_to_inv;
	void ivy_check_curr_invs();
	bool parse_log(set<int>& failed_inv_ids);
	void refine_one_countereg(const vars_t& vars, const inv_t& inv);
	void extend_disjunctions(const vars_t& vars, const inv_t& inv);
	void extend_successors(const vars_t& vars, const inv_t& inv);
	void infer_more_invs(vector<string>& more_invs);
public:
	bool is_last_attempt;
	InvRefiner(string problem, int num_attempt) : Solver(problem, num_attempt), ivy_call_count(0), countereg_count(0), is_last_attempt(false) {}
	bool auto_refine();
	int get_countereg_count();
	int get_invariant_count();
};

void write_log(bool success, int countereg_count, int invariant_num, int enumeration_time_total, int refinement_time_total);
#endif
