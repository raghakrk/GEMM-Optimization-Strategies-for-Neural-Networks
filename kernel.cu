
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<vector>
#include<algorithm>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <iomanip>
#include "modules/utils.h"
#include <map>


using namespace std;

class hidden_layer 
{
public:
	map<char, vector<vector<double>>> gradient;
	map<char, vector<vector<double>>> params;
	int output_DIM;
	int input_DIM;
	int no_of_input;

	hidden_layer(int no_of_input, int input_DIM, int output_DIM)
	{
		this->input_DIM = input_DIM;
		this->no_of_input = no_of_input;
		this->output_DIM = output_DIM;
		this->params['W']= make2Dvec(input_DIM, output_DIM);
		this->params['b'] = make2Dvec(1, output_DIM);
		this->gradient['W']=make2Dvec(input_DIM, output_DIM);
		this->gradient['b'] = make2Dvec(1, output_DIM);
		weightInitialization();
	}

	vector<vector<double>> forward(vector<vector<double>> X)
	{
		auto forward_output = MATMUL(X, this->params['W']);
		vectoradd(forward_output, this->params['b']);
		return forward_output;
	}

	vector<vector<double>> backward(vector<vector<double>> X, vector<vector<double>> grad)
	{
		auto XT = MatTranspose(X);
		this->gradient['W'] = MATMUL(XT, grad);
		this->gradient['b'] = rowsum(grad);
		auto WT=MatTranspose(this->params['W']);
		auto backward_output = MATMUL(grad,WT);
		return backward_output;
	}
	void weightInitialization()
	{
		int  row = this->input_DIM;
		int col = this->output_DIM;
		float shape = row * col;
		random_device rd{};
		mt19937 gen{ rd() };
		normal_distribution<> d{ 0,1 };
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
				this->params['W'][i][j] = double(d(gen)) * sqrtf(2/shape);
		}
	}
};


void miniBatchStochasticGradientDescent()
{
	;
}
int main()
{
	// model parameters
	int num_epoch = 10;
	int minibatch_size = 128;

	//optimisation alpha for momentum, lambda for weight decay
	float learning_rate = 0.01;
	int step = 10;
	float alpha = 0.99;
	float lambda = 0.01;
	float dropout = 0.5;
	string activation = "relu";
	
	if (activation == "relu")
		reLU act;
	else
		tanH act;

	// create objects for each layer
	int no_of_inputs = 3;
	int input_dimension = 2;
	int layer1_neurons = 3;
	int output_dimension = 1;
	hidden_layer L1(no_of_inputs, input_dimension, layer1_neurons);
	hidden_layer L2(no_of_inputs, layer1_neurons, output_dimension);

	string filename = "C:\\Users\\ragha\\source\\repos\\Neural Networks\\train-images.idx3-ubyte";

	//ReadMNIST(filename, 60000, 784, ar);

;

	return 0;
}

