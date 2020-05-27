#include"utils.h"
vector<vector<double>> reLU::forward(vector<vector<double>> Z)
{
	//Z=WX+b
	int row = Z.size();
	int col = Z[0].size();
	vector<vector<double>> forward_output(row, vector<double>(col));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			forward_output[i][j] = max(Z[i][j], 0.0);
		}
	}
	return forward_output;
}

vector<vector<double>> reLU::backward(vector<vector<double>> Z, vector<vector<double>> grad)
{
	int row = Z.size();
	int col = Z[0].size();
	vector<vector<double>> backward_output(row, vector<double>(col,0));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (Z[i][j]>0)
				backward_output[i][j] = grad[i][j];
		}
	}
	return backward_output;
}

