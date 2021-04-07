#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "basics.h"

#define FILE_BUFFER_SIZE (1 << 24)

/*
#define MAX_ONE_TO_ONE 5
#define MAX_TOTAL_ORDER_VARS 5
#define MAX_SAME_TYPE_GROUP 5
#define MAX_SAME_TYPE_VARS_PER_GROUP 5
struct Config
{
	int max_literal;
	char* one_to_one[MAX_ONE_TO_ONE][2];
	char* one_to_one_bidir[2*MAX_ONE_TO_ONE][2];
	char* one_to_one_f;
	char* total_order[MAX_TOTAL_ORDER_VARS];
	char* same_type[MAX_SAME_TYPE_GROUP][MAX_SAME_TYPE_VARS_PER_GROUP];
};
*/

void read_config(const string& config_file, Config* config);
void read_trace(const string& csv_file, vector<string>& predicates, DataMatrix& init_data_mat);
void add_negation(DataMatrix& data_mat);
void semicolon_comma_parse_line(const string& line, vector<vector<string>>& parsed_results);
void merge_quoted_comma(const vector<string>& before_merging, vector<string>& after_merging);

#endif
