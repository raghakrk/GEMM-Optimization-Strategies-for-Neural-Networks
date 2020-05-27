#include"utils.h"
softmax_crossentropy::softmax_crossentropy(vector<vector<double>> X)
{
	this->inputlen = X.size();
	this->inputdim = X[0].size();
	prob = make2Dvec(inputlen, inputdim);
	this->X = X;
}

vector<vector<double>> softmax_crossentropy::forward()
{
	double x_max;
	double sum;
	for (size_t i = 0; i < inputlen; i++)
	{
		x_max = *max_element(X[i].begin(), X[i].end());
		sum = 0;
		for (size_t j = 0; j < inputdim; j++)
		{
			sum += expf(X[i][j] - x_max);
		}

		for (size_t j = 0; j < inputdim; j++)
		{
			prob[i][j] = expf(X[i][j] - x_max) / expf(sum);
			X[i][j] = expf(X[i][j] - x_max - log(sum));
		}
	}
	return X;
}
