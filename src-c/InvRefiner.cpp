#include "InvRefiner.h"

void InvRefiner::ivy_check_curr_invs()
{
	string original_ivy_file = "../protocols/" + problem_name + "/" + problem_name + ".ivy";
	string target_output_file = "../outputs/" + problem_name + "/" + problem_name + "_inv" + ".ivy";
	id_to_inv.clear();
	vector<string> more_invs;
	if (config.hard) infer_more_invs(more_invs);
	add_checked_invs(more_invs);
	encode_and_output(original_ivy_file, target_output_file, id_to_inv, more_invs);
	int retval = system((string(IVY_CHECK_PATH) + " " + target_output_file + " > runtime/" + problem_name + "/ivy_check_log.txt").c_str());
	(void)retval;
	ivy_call_count++;
}

bool InvRefiner::parse_log(set<int>& failed_inv_ids)
{
	ifstream in("runtime/" + problem_name + "/ivy_check_log.txt");
	if (!in) {
		cout << "Cannot open Ivy log file" << endl;
		exit(-1);
	}
	string line;
	bool succeed = false;
	while (getline(in, line))
	{
		if (line.find("FAIL") != string::npos)
		{
			vector<string> substrings;
			split_string(line, ' ', substrings);
			if (substrings.size() != 6) cout << line << endl;
			assert(substrings.size() == 6);
			int failed_inv = stoi(substrings[3]);
			failed_inv_ids.insert(failed_inv);
		}
		else if (line.substr(0,2) == "OK")
		{
			succeed = true;
		}
	}
	in.close();
	assert(!(succeed && failed_inv_ids.size() > 0));
	if ((!succeed) && (failed_inv_ids.size() == 0))
	{
		cout << "Ivy check failed. Check ivy_check_log.txt for details" << endl;
		exit(-1);
	}
	return(succeed);
}

void InvRefiner::refine_one_countereg(const vars_t& vars, const inv_t& inv)
{
	// remove the broken invariant
	invs_dict[vars].erase(inv);
	extended_invs_dict[vars].erase(inv);
	// optionally add new candidate invariants 
	// if REFINE_EXTEND_DISJUNCTION == REFINE_EXTEND_SECCESSOR == true, the system is theoretically guaranteed to find exist-free invariants
	// but the runtime can be prohibitively slow
	if (refine_extend_disjunction)
	{
		// if invariant p \/ !q is invalidated by SMT solver, try p \/ !q \/ r for all r
		extend_disjunctions(vars, inv);
	}
	else
	{
		// mark if we possibly overshoot
		if (!lower_literal_inv_discarded)
		{
			if (int(inv.size()) < config.max_literal) lower_literal_inv_discarded = true;
		}
	}
	if (refine_extend_successor)
	{
		// if forall X. p(X) is invalidated by SMT solver, try forall X < Y. p(X)
		extend_successors(vars, inv);
	}
	else
	{
		// mark if we possibly overshoot
		if (!lower_subtemplate_inv_discarded)
		{
			if (column_indices_dict[vars].size() > 0) lower_subtemplate_inv_discarded = true;
		}
	}
}

void InvRefiner::extend_disjunctions(const vars_t& vars, const inv_t& inv)
{
	// 1) If we can parse the counterexample, either r or !r can be discarded.
	int inv_literal = int(inv.size());
	if (inv_literal == config.max_literal) return;
	const unordered_set<inv_t, VectorHash>& extended_invs = extended_invs_dict[vars];
	int num_predicates = predicates_dict[vars].size();
	int num_predicates_2 = 2 * num_predicates;
	for (int i = 0; i < num_predicates_2; i++)
	{
		int not_i = (i + num_predicates) % num_predicates_2;
		if ((std::find(inv.begin(), inv.end(), i) == inv.end()) && (std::find(inv.begin(), inv.end(), not_i) == inv.end()))
		{
			inv_t new_inv = inv;
			new_inv.push_back(i);
			std::sort(new_inv.begin(), new_inv.end());

			// if p \/ r is currently an invariant, we do not need to consider p \/ !q \/ r
			bool subcomb_is_inv = false;
			vector<inv_t> subcombs;
			calc_combinations(new_inv, inv_literal - 1, subcombs);

			for (const inv_t& subcomb : subcombs)
			{
				if (extended_invs.find(subcomb) != extended_invs.end())
				{
					subcomb_is_inv = true;
					break;
				}
			}

			if (!subcomb_is_inv)
			{
				invs_dict[vars].insert(new_inv);
			}
		}
	}
}

void InvRefiner::extend_successors(const vars_t& vars, const inv_t& inv)
{
	for (map<vars_t, vector<vector<int>>>::iterator it = column_indices_dict[vars].begin(); it != column_indices_dict[vars].end(); it++)
	{
		// cout << "Extending successor" << endl;
		const vars_t& successor = it->first;
		const vector<vector<int>>& column_indices_list = it->second;
		unordered_set<inv_t, VectorHash> new_extended_invs;
		unordered_set<inv_t, VectorHash> inv_as_set;
		inv_as_set.insert(inv);
		helper.generalize_invs(inv_as_set, column_indices_list, new_extended_invs);
		invs_dict[successor].insert(new_extended_invs.begin(), new_extended_invs.end());
	}
}

void InvRefiner::add_checked_invs(vector<string>& more_invs)
{
	for (const vector<string> checked_inv_tuple : config.checked_inv_tuples)
	{
		const string& relation_name = checked_inv_tuple[0];
		int arity = std::stoi(checked_inv_tuple[1]);
		int index = std::stoi(checked_inv_tuple[2]) + 1;
		const string& type_name = checked_inv_tuple[3];
		const string& type_abbr = checked_inv_tuple[4];
		string line;
		vector<string> segments;
		for (int i = 1; i <= arity + 1; i++) segments.push_back(type_abbr + std::to_string(i) + ":" + type_name);
		join_string(segments, ", ", line);
		line = "forall " + line + ". ";
		string inequality = type_abbr + std::to_string(index) + " ~= " + type_abbr + std::to_string(arity + 1);
		vector<string> param_range;
		for (int i = 1; i <= arity; i++) param_range.push_back(type_abbr + std::to_string(i));
		string predicate1;
		join_string(param_range, ',', predicate1);
		predicate1 = "~" + relation_name + "(" + predicate1 + ")";
		param_range[index - 1] = type_abbr + std::to_string(arity + 1);
		string predicate2;
		join_string(param_range, ',', predicate2);
		predicate2 = "~" + relation_name + "(" + predicate2 + ")";
		line += inequality + " -> " + predicate1 + " | " + predicate2;
		more_invs.push_back(line);
	}
}


void replaceAll(std::string& str, const std::string& from, const std::string& to) {
	if (from.empty())
		return;
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}


void InvRefiner::infer_more_invs(vector<string>& more_invs)
{
	vars_t& base_vars = vars_traversal_order[0];
	unordered_set<inv_t, VectorHash>& base_invs = invs_dict[base_vars];
	map<pair<string, string>, bool > candidates;
	for (map<string, vector<int>>::iterator it = var_in_p_dict[base_vars].begin(); it != var_in_p_dict[base_vars].end(); it++)
	{
		string var_name = it->first;
		const vector<int>& predicate_indices = it->second;
		const vector<string>& base_predicates = predicates_dict[base_vars];
		int num_predicates = int(base_predicates.size());
		for (int left : predicate_indices)
		{
			int adjusted_left = left < num_predicates ? left : left - num_predicates;
			for (int right : predicate_indices)
			{
				vector<int> candidate = { left, right };
				if (base_invs.find(candidate) != base_invs.end())
				{
					int adjusted_right = right < num_predicates ? right : right - num_predicates;
					if (left < right)
					{
						string p1 = base_predicates[adjusted_left];
						string p2 = base_predicates[adjusted_right];
						if ((p1.find(',') == string::npos) && (p2.find(',') == string::npos))
						{
							string relation1 = p1.substr(0, p1.find('('));
							string relation2 = p2.substr(0, p2.find('('));
							if (p1.find('(') == string::npos || p2.find('(') == string::npos) continue;
							if ((left < num_predicates && right < num_predicates) || (left >= num_predicates && right >= num_predicates))
							{
								candidates[pair<string, string>(relation1, relation2)] = true;
							}
							else
							{
								candidates[pair<string, string>(relation1, relation2)] = false;
							}
						}
					}
				}
			}
		}
	}
	for (map<pair<string, string>, bool >::iterator it = candidates.begin(); it != candidates.end(); it++)
	{
		const pair<string, string>& str_pair = it->first;
		const string& relation1 = str_pair.first;
		const string& relation2 = str_pair.second;
		bool need_negation = it->second;
		for (const string& safety : config.safety_properties)
		{
			string new_safety = safety;
			if (need_negation)
			{
				replaceAll(new_safety, " " + relation1, " ~" + relation2);
				if (new_safety != safety) more_invs.push_back(new_safety);
				new_safety = safety;
				replaceAll(new_safety, " " + relation2, " ~" + relation1);
				if (new_safety != safety) more_invs.push_back(new_safety);
			}
			else
			{
				replaceAll(new_safety, " " + relation1, " " + relation2);
				if (new_safety != safety) more_invs.push_back(new_safety);
				new_safety = safety;
				replaceAll(new_safety, " " + relation2, " " + relation1);
				if (new_safety != safety) more_invs.push_back(new_safety);
			}
		}
	}
}

bool InvRefiner::auto_refine(bool add_disj, bool add_proj)
{
	bool success = false;
	map<vars_t, unordered_set<inv_t, VectorHash>> invs_dict_before_refine = invs_dict;
	map<vars_t, unordered_set<inv_t, VectorHash>> extended_invs_dict_before_refine = extended_invs_dict;
	for (int mode = 0; mode < 3; mode++)
	{
		if (mode == 0)           // only remove broken invariants
		{
			refine_extend_disjunction = false;
			refine_extend_successor = false;
		}
		else 
		{
			if (mode == 1)       // enable second step of minimum weakening --- disjunct literals
			{
				if (!add_disj) break;
				if (!lower_literal_inv_discarded) break;
				refine_extend_disjunction = true;
				refine_extend_successor = false;
				cout << "Enable second step of minimum weakening" << endl;
			}
			else if (mode == 2)  // enable third step of minimum weakening --- project to higher subtemplates
			{
				if (!add_proj) break;
				if (!lower_subtemplate_inv_discarded) break;
				refine_extend_disjunction = true;
				refine_extend_successor = true;
				cout << "Enable third step of minimum weakening" << endl;
			}
			invs_dict = invs_dict_before_refine;
			extended_invs_dict = extended_invs_dict_before_refine;
		}
		
		success = false;
		bool safety_property_failed = false;
		int refinement_round = 0;
		while ((!success) && (!safety_property_failed))
		{
			refinement_round++;
			ivy_check_curr_invs();
			set<int> failed_inv_ids;
			bool ivy_check_passed = parse_log(failed_inv_ids);
			cout << "Refinement round " << refinement_round << ": ";
			if (ivy_check_passed)
			{
				cout << "Invariants validated, protocol proved." << endl;
				success = true;
			}
			else
			{
				if (failed_inv_ids.find(SAFETY_PROPERTY_ID) != failed_inv_ids.end())
				{
					cout << "Safety property failed." << endl;
					safety_property_failed = true;
				}
				else
				{
					for (int id : failed_inv_ids)
					{
						// cout << id << endl;
						assert(id_to_inv.find(id) != id_to_inv.end());
						vars_t& vars = id_to_inv[id].first;
						inv_t& inv = id_to_inv[id].second;
						refine_one_countereg(vars, inv);
						countereg_count++;
					}
					cout << failed_inv_ids.size() << " invariants not inductive" << endl;
				}
			}
		}
		if (success) break;
	}
	return success;
}

int InvRefiner::get_countereg_count()
{
	return countereg_count;
}

int InvRefiner::get_invariant_count()
{
	return id_to_inv.size();
}

bool InvRefiner::get_lower_literal_inv_discarded()
{
	return lower_literal_inv_discarded;
}

bool InvRefiner::get_lower_subtemplate_inv_discarded()
{
	return lower_subtemplate_inv_discarded;
}

void write_log(string problem_name, bool success, int countereg_count, int invariant_num, int enumeration_time_total, int refinement_time_total, bool lower_literal_inv_discarded, bool lower_subtemplate_inv_discarded)
{
	ofstream out("runtime/" + problem_name + "/refiner_log.txt");
	if (!out) {
		cout << "Cannot create refiner log file" << endl;
		exit(-1);
	}

	out << "Success? " << BOOL_TO_STR(success) << endl;
	out << "Invariants: " << invariant_num << endl;
	out << "Counterexamples: " << countereg_count << endl;
	out << "Enumeration time: " << enumeration_time_total << endl;
	out << "Refinement time: " << refinement_time_total << endl;
	out.flush();
	out.close();
}
