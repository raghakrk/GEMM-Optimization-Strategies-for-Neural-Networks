#include"utils.h"
softmax_crossentropy::softmax_crossentropy()
{
	this->inputlen = 0;
	this->inputdim = 0;
	this->forward_output = 0;
}

double softmax_crossentropy::forward(vector<vector<double>> X, vector<vector<double>> Y)
{
	this->inputlen = X.size();
	this->inputdim = X[0].size();
	this->prob = make2Dvec(inputlen, inputdim);
	double x_max;
	double sum;
	double temp;
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
			temp = X[i][j] - x_max;
			prob[i][j] =  expf(temp)/ sum;
			this->forward_output+= Y[i][j]*(temp- log(sum));
		}
	}
	this->forward_output /= X.size();
	return this->forward_output;
}

vector<vector<double>>softmax_crossentropy::backward(vector<vector<double>> X, vector<vector<double>> Y)
{

	this->backward_output = make2Dvec(X.size(), X[0].size());
	for (size_t i = 0; i < Y.size(); i++)
	{
		for (size_t j = 0; j < Y[0].size(); j++)
		{
			this->backward_output[i][j] = (Y[i][j] - this->prob[i][j]) / X.size();
		}
	}
	return this->backward_output;
}
