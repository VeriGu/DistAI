#include "Solver.h"
#include "InvRefiner.h"


int main(int argc, char* argv[])
{
	if (!(argc >= 2)) {
		cout << "Please specify a protocol" << endl;
		exit(-1);
	}

	auto start_time = std::chrono::steady_clock::now();
	string problem = argv[1];

	int max_retry = 1;
	bool add_disj = true, add_proj = true;
	for (int i = 2; i < argc; i++) {
		string arg_str = argv[i];
		if (arg_str.rfind("--max_retry=", 0) == 0) {
			max_retry = atoi(arg_str.substr(12).c_str());
		}
		else if (arg_str.rfind("--add_disj=", 0) == 0) {
			string add_disj_str = arg_str.substr(11);
			assert((add_disj_str == "true") || (add_disj_str == "false"));
			add_disj = add_disj_str == "true";
		}
		else if (arg_str.rfind("--add_proj=", 0) == 0) {
			string add_proj_str = arg_str.substr(11);
			assert((add_proj_str == "true") || (add_proj_str == "false"));
			add_proj = add_proj_str == "true";
		}
	}
	assert(!(!add_disj) && (add_proj));

	bool success = false;
	int num_attempt = 0;
	int counterexample_count = 0;
	auto enumeration_time = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
	auto refinement_time = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
	int invariant_num = 0;
	bool lower_literal_inv_discarded = false, lower_subtemplate_inv_discarded = false;

	// increase the max-literal after each failed refinement
	while(!success && num_attempt <= max_retry)
	{
		if (num_attempt > 0) 
		{
			cout << "Retry with new template/max-literal" << endl;
		}
		InvRefiner refiner(problem, num_attempt++);
		Solver& solver = refiner;
		solver.auto_solve();
		auto refine_start_time = std::chrono::steady_clock::now();
		success = refiner.auto_refine(add_disj, add_proj);
		auto refine_end_time = std::chrono::steady_clock::now();
		refinement_time += refine_end_time - refine_start_time;
		counterexample_count += refiner.get_countereg_count();
		invariant_num = refiner.get_invariant_count();
		lower_literal_inv_discarded = refiner.get_lower_literal_inv_discarded();
		lower_subtemplate_inv_discarded = refiner.get_lower_subtemplate_inv_discarded();
	}

	auto end_time = std::chrono::steady_clock::now();
	int enumeration_time_total = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time - refinement_time).count();
	int refinement_time_total = (int)std::chrono::duration_cast<std::chrono::milliseconds>(refinement_time).count();

	write_log(problem, success, counterexample_count, invariant_num, enumeration_time_total, refinement_time_total, lower_literal_inv_discarded, lower_subtemplate_inv_discarded);
	return 0;
}
