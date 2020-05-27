#include "utils.h"
vector<vector<double>> sigmoid(vector<vector<double>> z)
{
	int row = z.size();
	int col = z[0].size();
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			z[i][j] = tanh(z[i][j]);
		}
	}
	return z;
}
