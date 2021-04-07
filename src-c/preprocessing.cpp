#include "preprocessing.h"

void read_config(const string& config_file, Config* config)
{
	ifstream in(config_file.c_str());
	if (!in)
	{
		cout << "Can't open config file " << config_file << endl;
		exit(-1);
	}
	config->max_literal = 0;
	config->one_to_one_exists = false;
	config->total_order_exists = false;
	string line;
	while (getline(in, line)) 
	{	
		if (line[0] == '#') continue;  // comment line
		else if (line.rfind("one-to-one:", 0) == 0)  // equivalent as line.startswith("one-to-one") in Python
		{
			assert(config->one_to_one.size() == 0);
			assert(config->one_to_one_bidir.size() == 0);
			vector<vector<string>> pairs;
			semicolon_comma_parse_line(line.substr(11), pairs);
			for (vector<string>& curr_pair : pairs)
			{
				assert(curr_pair.size() == 2);
				config->one_to_one[curr_pair[0]] = curr_pair[1];
				config->one_to_one_bidir[curr_pair[0]] = curr_pair[1];
				config->one_to_one_bidir[curr_pair[1]] = curr_pair[0];
				config->one_to_one_exists = true;
			}
		}
		else if (line.rfind("one-to-one-f:", 0) == 0)
		{
			assert(config->one_to_one_f.size() == 0);
			config->one_to_one_f = trim_string(line.substr(13));
		}
		else if (line.rfind("total-order:", 0) == 0)
		{
			assert(config->total_order.size() == 0);
			vector<string> literals;
			split_string(line.substr(12), ',', literals);
			if (literals.size() > 0)
			{
				config->total_order = literals;
				config->total_order_exists = true;
			}
		}
		else if (line.rfind("same-type:", 0) == 0)
		{
			assert(config->same_type.size() == 0);
			vector<vector<string>> groups;
			semicolon_comma_parse_line(line.substr(10), groups);
			config->same_type = groups;
		}
		else if (line.rfind("max-literal:", 0) == 0)
		{
			assert(config->max_literal == 0);
			int max_literal = std::stoi(line.substr(12));
			config->max_literal = max_literal;
		}
		else if (line.rfind("hard: true", 0) == 0)
		{
			config->hard = true;
		}
		else if (line.rfind("safety-property:", 0) == 0)
		{
			if (config->hard)
			{
				config->safety_properties.push_back(line.substr(16));
			}
		}
		else
		{
			cout << "Unparsable line in config" << line << endl;
		}
	}
}

void read_trace(const string& csv_file, vector<string>& full_predicates, DataMatrix& init_data_mat)
{
	// read the sample file, outputs the predicates (i.e., first line of csv file) and the 0/1 sample matrix
	assert(full_predicates.size() == 0);
	assert(init_data_mat.data_ptr == NULL);
	char* file_read_buffer = new char[FILE_BUFFER_SIZE];
	FILE* fp = fopen(csv_file.c_str(), "r");
	size_t newLen = 0;
	if (fp != NULL) {
		newLen = fread(file_read_buffer, sizeof(char), FILE_BUFFER_SIZE, fp);
		if (newLen >= FILE_BUFFER_SIZE - 1) {
			delete[] file_read_buffer;
			file_read_buffer = new char[16 * FILE_BUFFER_SIZE];
			fclose(fp);
			fp = fopen(csv_file.c_str(), "r");
			newLen = fread(file_read_buffer, sizeof(char), 16 * FILE_BUFFER_SIZE, fp);
		}
		if (newLen >= 16 * FILE_BUFFER_SIZE - 1) {   // if still fails, return
			cout << "Trace file exceeds input buffer size. You can increase FILE_BUFFER_SIZE at the beginning of preprocessig.h" << endl;
			cout << "But in fact, our tool has effectively failed. The huge trace file will stall the invariant learner or exhausts memory during enumeration" << endl;
			exit(-1);
		}
		if (ferror(fp) != 0) {
			fputs("Error reading file", stderr);
		}
		fclose(fp);
	}
	else {
		cout << "Cannot open trace file " << csv_file << endl;
		exit(-1);
	}

	int landmark;
	for (int i = 0;; i++)
	{
		if (file_read_buffer[i] == '\n')
		{
			landmark = i;
			break;
		}
	}
	int newline_c = 1;
	if (file_read_buffer[landmark - 1] == '\r') newline_c = 2;
	string line(file_read_buffer, landmark);
	vector<string> before_merging;
	split_string(line, ',', before_merging); 
	merge_quoted_comma(before_merging, full_predicates);
	int ncol = full_predicates.size();
	vector<int> temp_data;
	int nchar_per_row = 2 * ncol + newline_c - 1;
	landmark++;
	assert((newLen - landmark) % nchar_per_row == 0);
	int nrow = (newLen - landmark) / nchar_per_row;
	assert((nrow > 0) && (ncol > 0));
	for (int row = 0; row < nrow; row++)
	{
		int end = landmark + 2 * ncol;
		for (int i = landmark; i < end; i += 2)
		{
			if (file_read_buffer[i] == '1') temp_data.push_back(1);
			else
			{
				assert(file_read_buffer[i] == '0');
				temp_data.push_back(0);
			}
		}
		landmark += nchar_per_row;
	}
	assert(temp_data.size() % ncol == 0);
	assert(nrow == int(temp_data.size() / ncol));
	init_data_mat.data_ptr = contiguous_2d_array(nrow, ncol);
	std::copy(temp_data.begin(), temp_data.end(), init_data_mat.data_ptr[0]);
	init_data_mat.nrow = nrow;
	init_data_mat.ncol = ncol;
	delete[] file_read_buffer;
}

void add_negation(DataMatrix& data_mat)
{
	int nrow = data_mat.nrow, ncol = data_mat.ncol;
	int** old_data_ptr = data_mat.data_ptr;
	int** new_data_ptr = contiguous_2d_array(nrow, 2 * ncol);
	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
		{
			new_data_ptr[i][j] = old_data_ptr[i][j];
			new_data_ptr[i][j + ncol] = 1 - old_data_ptr[i][j];
		}
	}
	data_mat.ncol = 2 * ncol;
	data_mat.data_ptr = new_data_ptr;
	// warning: currently no memory free (delete []) for old_data_ptr
}

void semicolon_comma_parse_line(const string& line, vector<vector<string>>& parsed_results)
{
	vector<string> segments;
	split_string(line, ';', segments);
	for (string segment : segments)
	{
		vector<string> segment_parsed_result;
		split_string(segment, ',', segment_parsed_result);
		if (segment_parsed_result.size() > 0)
		{
			parsed_results.push_back(segment_parsed_result);
		}
	}
}

void merge_quoted_comma(const vector<string>& before_merging, vector<string>& after_merging)
{
	// "p(X, Y)" in csv header will be split into "p(X) and Y)", so we need to merge them
	// assume "" matches otherwise may have bugs
	assert(after_merging.size() == 0);
	bool continuation = false;
	string temp;
	for (const string& str : before_merging)
	{
		if (continuation)
		{
			if (str.back() == '\"')
			{
				temp = temp + "," + str.substr(0, str.size() - 1);
				after_merging.push_back(temp);
				continuation = false;
				temp.clear();
			}
			else
			{
				temp = temp + "," + str;
			}
		}
		else
		{
			if (str[0] != '\"')
			{
				after_merging.push_back(str);
			}
			else
			{
				temp = str.substr(1);
				assert((temp.size() > 0) && (temp.back() != '\"'));
				continuation = true;
			}
		}
	}
}
