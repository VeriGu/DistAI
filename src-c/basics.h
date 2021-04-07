#ifndef BASICS_H
#define BASICS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <map>
#include <set>
#include <unordered_set>
#include <iterator>
#include <ctime>
#include <chrono>
using std::cout; using std::endl; using std::ifstream; using std::ofstream;
using std::string; using std::vector; using std::pair;
using std::stringstream;
using std::map; using std::unordered_set; using std::set;
using std::reverse_iterator;

#define MAX_COMB_K 10

struct Config
{
	int max_literal;
	bool one_to_one_exists;
	map<string, string> one_to_one;
	map<string, string> one_to_one_bidir;
	string one_to_one_f;
	bool total_order_exists;
	vector<string> total_order;
	vector<vector<string>> same_type;
	bool hard;
	vector<string> safety_properties;
};

struct DataMatrix
{
	int** data_ptr;
	int nrow;
	int ncol;
};

typedef vector<string> vars_t;  // e.g., ['N1', 'N2', 'K0']
typedef vector<int> inv_t;      // e.g., [3, 4, 7]

void join_string(const vector<string>& v, char c, string& s);
void join_string(const vector<string>& v, const string& c, string& s);
void split_string(const string& str, char delimeter, vector<string>& splitted_results);
int** contiguous_2d_array(int nrow, int ncol);

// hash function for unordered_set<vector<int>>, from https://stackoverflow.com/questions/29855908/c-unordered-set-of-vectors/29855973#29855973
struct VectorHash {
	size_t operator()(const std::vector<int>& v) const {
		std::hash<int> hasher;
		size_t seed = 0;
		for (int i : v) {
			seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

// auxiliary functions for string trimming, from https://www.techiedelight.com/trim-string-cpp-remove-leading-trailing-spaces/
const std::string WHITESPACE = " \n\r\t\f\v";
inline std::string ltrim(const std::string& s)
{
	size_t start = s.find_first_not_of(WHITESPACE);
	return (start == std::string::npos) ? "" : s.substr(start);
}
inline std::string rtrim(const std::string& s)
{
	size_t end = s.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
inline std::string trim_string(const std::string& s)
{
	return rtrim(ltrim(s));
}

int binomialCoeff(int n, int k);

template <typename T>
void calc_combinations(const vector<T>& input_seq, int k, vector<vector<T>>& output_seq);
template <typename T>
void calc_permutations(const vector<T>& input_seq, int k, vector<vector<T>>& output_seq);
template<typename T>
vector<vector<T>> cart_product(const vector<vector<T>>& v);

#endif
