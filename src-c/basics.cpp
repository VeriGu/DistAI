#include "basics.h"

void split_string(const string& str, char delimeter, vector<string>& splitted_results)
{
	stringstream tokenizer(str);
	string temp;
	while (getline(tokenizer, temp, delimeter))
	{
		temp = trim_string(temp);
		if (temp.size() > 0)
		{
			splitted_results.push_back(temp);
		}
	}
}

void join_string(const vector<string>& v, char c, string& s) {

	s.clear();

	for (vector<string>::const_iterator p = v.begin();
		p != v.end(); ++p) {
		s += *p;
		if (p != v.end() - 1)
			s += c;
	}
}

void join_string(const vector<string>& v, const string& c, string& s) {

	s.clear();

	for (vector<string>::const_iterator p = v.begin();
		p != v.end(); ++p) {
		s += *p;
		if (p != v.end() - 1)
			s += c;
	}
}

int** contiguous_2d_array(int nrow, int ncol)
{
	int** matrix;
	matrix = new int* [nrow];
	matrix[0] = new int[nrow * ncol];

	for (int i = 1; i < nrow; i++) {
		matrix[i] = matrix[i - 1] + ncol;
	}
	return matrix;
}

int binomialCoeff(int n, int k)
{
	assert(k <= MAX_COMB_K);
	// from https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
	int C[MAX_COMB_K + 1];
	memset(C, 0, sizeof(C));

	C[0] = 1; // nC0 is 1

	for (int i = 1; i <= n; i++) {
		// Compute next row of pascal triangle using
		// the previous row
		for (int j = std::min(i, k); j > 0; j--)
			C[j] = C[j] + C[j - 1];
	}
	return C[k];
}


template<typename T>
void calc_combinations(const vector<T>& input_seq, int k, vector<vector<T>>& output_seq)
{
	// TODO: this can be accelerated
	assert(output_seq.size() == 0);
	int n = input_seq.size();
	if (n == 0 || n < k) return;
	assert(k <= MAX_COMB_K);
	if (k == 0) return;
	int output_len = binomialCoeff(n, k);
	output_seq.resize(output_len);
	for (int i = 0; i < output_len; i++) output_seq[i].resize(k);
	int indices[MAX_COMB_K + 1];
	for (int i = 0; i < k; i++) indices[i] = i;
	int comb_count = 0;
	bool next_comb_exists;
	// learned from Python implementation https://docs.python.org/3/library/itertools.html#itertools.combinations
	do
	{
		for (int i = 0; i < k; i++) output_seq[comb_count][i] = input_seq[indices[i]];
		comb_count++;
		next_comb_exists = false;
		int i;
		for (i = k - 1; i >= 0; i--)
		{
			if (indices[i] != i + n - k)
			{
				next_comb_exists = true;
				break;
			}
		}
		if (next_comb_exists)
		{
			indices[i] += 1;
			for (int j = i + 1; j < k; j++)
			{
				indices[j] = indices[j - 1] + 1;
			}
		}
	} while (next_comb_exists);
}

template <typename T>
void calc_permutations(const vector<T>& input_seq, int k, vector<vector<T>>& output_seq)
{
	vector<vector<T>> comb_seq;
	calc_combinations(input_seq, k, comb_seq);
	vector<int> k_permutation;
	vector<vector<int>> all_k_permutations;
	for (int i = 0; i < k; i++) k_permutation.push_back(i);
	do
	{
		all_k_permutations.push_back(k_permutation);
	} while (std::next_permutation(k_permutation.begin(), k_permutation.end()));
	for (const vector<T>& comb : comb_seq)
	{
		// const vector<T>& seq_to_perm = comb;
		for (const vector<int>& k_permutation : all_k_permutations)
		{
			vector<T> perm;
			for (int i = 0; i < k; i++)
			{
				perm.push_back(comb[k_permutation[i]]);
			}
			output_seq.push_back(perm);
		}
	}
}

// Cartesian product, from https://stackoverflow.com/questions/5279051/how-can-i-create-cartesian-product-of-vector-of-vectors/17050528#17050528
template<typename T>
vector<vector<T>> cart_product(const vector<vector<T>>& v) {
	vector<vector<T>> s = { {} };
	for (const auto& u : v) {
		vector<vector<T>> r;
		for (const auto& x : s) {
			for (const auto y : u) {
				r.push_back(x);
				r.back().push_back(y);
			}
		}
		s = move(r);
	}
	return s;
}

template void calc_combinations<string>(const vector<string>& input_seq, int k, vector<vector<string>>& output_seq);
template void calc_combinations<int>(const vector<int>& input_seq, int k, vector<vector<int>>& output_seq);
template void calc_permutations<string>(const vector<string>& input_seq, int k, vector<vector<string>>& output_seq);
template vector<vector<map<string, string>>> cart_product(const vector<vector<map<string, string>>>& v);
