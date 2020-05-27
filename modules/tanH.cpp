#include"utils.h"
vector<vector<double>> tanH::forward(vector<vector<double>> Z)
{
	//Z=WX+b
	int row = Z.size();
	int col = Z[0].size();
	vector<vector<double>> forward_output(row, vector<double>(col));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			forward_output[i][j] = tanh(Z[i][j]);
		}
	}
	return forward_output;
}

vector<vector<double>> tanH::backward(vector<vector<double>> Z, vector<vector<double>> grad)
{
	int row = Z.size();
	int col = Z[0].size();
	vector<vector<double>> backward_output(row, vector<double>(col, 0));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			backward_output[i][j] = (1 - pow(tanh(Z[i][j]), 2)) * grad[i][j];
		}
	}
	return backward_output;
}

