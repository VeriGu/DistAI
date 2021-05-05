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

	int max_retry;
	if (argc >= 3) {
		max_retry = atoi(argv[2]);
	}
	else max_retry = 1;

	bool success = false;
	int num_attempt = 0;
	int counterexample_count = 0;
	auto enumeration_time = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
	auto refinement_time = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
	int invariant_num = 0;
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
		success = refiner.auto_refine();
		auto refine_end_time = std::chrono::steady_clock::now();
		refinement_time += refine_end_time - refine_start_time;
		counterexample_count += refiner.get_countereg_count();
		invariant_num = refiner.get_invariant_count();
	}

	auto end_time = std::chrono::steady_clock::now();
	int enumeration_time_total = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time - refinement_time).count();
	int refinement_time_total = (int)std::chrono::duration_cast<std::chrono::milliseconds>(refinement_time).count();

	write_log(success, counterexample_count, invariant_num, enumeration_time_total, refinement_time_total);
	return 0;
}