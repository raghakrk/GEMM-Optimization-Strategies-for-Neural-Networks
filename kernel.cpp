#include<vector>
#include<algorithm>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <iomanip>
#include "modues/utils.h"
#include <map>
#include <functional>
#include <list>
#include <math.h>

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
		this->params['W'] = make2Dvec(input_DIM, output_DIM);
		this->params['b'] = make2Dvec(1, output_DIM);
		this->gradient['W'] = make2Dvec(input_DIM, output_DIM);
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
		auto WT = MatTranspose(this->params['W']);
		auto backward_output = MATMUL(grad, WT);
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
				this->params['W'][i][j] = double(d(gen)) * sqrtf(2 / shape);
		}
	}
};


void miniBatchStochasticGradientDescent(hidden_layer* layer, float learning_rate, float lambda)
{
	double g;
	int N = layer->params['W'].size();
	int D = layer->params['W'][0].size();
	bool flag = true;
	for (size_t i = 0; i < N; i++)
	{

		for (size_t j = 0; j < D; j++)
		{
			if (flag)
			{
				flag = false;
				g = layer->gradient['b'][0][j] + lambda * layer->params['b'][0][j];
				layer->params['b'][0][j] -= learning_rate * g;
			}
			g = layer->gradient['W'][i][j] + lambda * layer->params['W'][i][j];
			layer->params['W'][i][j] -= learning_rate * g;
		}
	}
}

int myrandom(int i) { return std::rand() % i; }

template<typename T>
vector<T> slice(vector<T> const& v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	vector<T> vec(first, last);
	return vec;
}

vector<vector<double>> onehotencoding(vector<int> Y)
{
	vector<vector<double>> label(Y.size(), vector<double>(10, 0));
	for (int i = 0; i < Y.size(); i++)
	{
		label[i][Y[i]] = 1;
	}
	return label;
}

vector<int> predict(vector<vector<double>> v)
{
	vector<int> pred;
	for (int i = 0; i < v.size(); i++)
	{
		int ind = max_element(v[i].begin(), v[i].end()) - v[i].begin();
		pred.push_back(ind);
	}
	return pred;
}
int main()
{
	//read File MNIST
	string filename = "C:\\Users\\ragha\\source\\repos\\Neural Networks\\train-images.idx3-ubyte";
	string label_filename = "C:\\Users\\ragha\\source\\repos\\Neural Networks\\train-labels.idx1-ubyte";
	int imagecount = 60000;
	int imagedim = 28 * 28;
	vector<vector<double>> images(imagecount, vector<double>(imagedim));
	read_mnist_images(filename, imagecount, imagedim, images);
	auto readlabels = read_mnist_labels(label_filename, imagecount);
	auto label = onehotencoding(readlabels);
	/* vector<int> c(10);
	for (int i = 0; i < 10; i++)
		c[i] = rand()%10;

	cout << endl;
	vector<vector<double>> r = onehotencoding(c);
	for (int i = 0; i < r.size(); i++)
	{   
		cout << c[i] << "   ";
		for (int j = 0; j < r[0].size(); j++)
			cout << r[i][j] << " ";
		cout << endl;
	}
	*/


	///*
	// model parameters
	int num_epoch = 10;
	int minibatch_size = 128;

	//optimisation alpha for momentum, lambda for weight decay
	float learning_rate = 0.01;
	int step = 10;
	float alpha = 0;
	float lambda = 0;
	float dropout = 0.5;
	string activation = "relu";
	tanH act;
	if (activation == "relu")
	{
		reLU act;
	}

	// create objects for each layer
	int no_of_inputs = 1000;
	int input_dimension = 784;
	int layer1_neurons = 100;
	int output_dimension = 10;

	vector<vector<double>> x_train = slice(images, 0, no_of_inputs);
	vector<vector<double>> y_train = slice(label, 0, no_of_inputs);
	vector<int> label_train = slice(readlabels, 0, no_of_inputs);

	vector<vector<double>> x_test = slice(images, 50000, 59999);
	vector<vector<double>> y_test = slice(label, 50000, 59999);
	vector<int> label_test = slice(readlabels, 50000, 59999);


	hidden_layer L1(no_of_inputs, input_dimension, layer1_neurons);
	hidden_layer L2(no_of_inputs, layer1_neurons, output_dimension);
	softmax_crossentropy LOSS;
	vector <double> train_acc_record;
	vector <double> val_acc_record;
	vector <double> train_loss_record;
	vector <double> val_loss_record;


	for (int t = 0; t < num_epoch; t++)
	{
		cout << "At epoch " << t + 1 << endl;
		if ((t % step == 0) && (t != 0))
			learning_rate = learning_rate * 0.1;
		//random_shuffle(images.begin(), images.end(), myrandom);

		float train_acc = 0.0;
		float train_loss = 0.0;
		int train_count = 0;
		float val_acc = 0.0;
		int val_count = 0;
		float val_loss = 0.0;

		//training
		for (int i = 0; i<int(floor(no_of_inputs / minibatch_size)); i++)
		{
			cout << "=";
			vector<vector<double>> x = slice(x_train, i * minibatch_size, (i + 1) * minibatch_size);
			vector<vector<double>> y = slice(y_train, i * minibatch_size, (i + 1) * minibatch_size);
			//forward 
			auto a1 = L1.forward(x);
			auto h1 = act.forward(a1);
			auto a2 = L2.forward(h1);
			auto loss = LOSS.forward(a2, y);

			//backward
			auto grad_a2 = LOSS.backward(a2, y);
			auto grad_h1 = L2.backward(h1, grad_a2);
			auto grad_a1 = act.backward(a1, grad_h1);
			auto grad_x = L1.backward(x, grad_a1);
			miniBatchStochasticGradientDescent(&L1, learning_rate, lambda);
			miniBatchStochasticGradientDescent(&L2, learning_rate, lambda);
		}
		cout << ">" << endl;
		// extracting accuracy
		for (int i = 0; i<int(floor(no_of_inputs / minibatch_size)); i++)
		{
			vector<vector<double>> x = slice(x_train, i * minibatch_size, (i + 1) * minibatch_size);
			vector<vector<double>> y = slice(y_train, i * minibatch_size, (i + 1) * minibatch_size);
			vector<int> y_t = slice(label_train, i * minibatch_size, (i + 1) * minibatch_size);
			auto a1 = L1.forward(x);
			auto h1 = act.forward(a1);
			auto a2 = L2.forward(h1);
			auto loss = LOSS.forward(a2, y);

			vector<int> y_pred = predict(a2);
			float sum = 0;
			for (int i = 0; i < y_t.size(); i++)
			{
				if (y_pred[i] == y_t[i])
					sum += 1;
			}

			train_acc += sum;
			train_loss += loss;
			train_count += y_t.size();
		}
		train_acc /= train_count;
		train_acc_record.push_back(train_acc);
		train_loss_record.push_back(train_loss);
		cout << "Training loss at epoch " << t + 1 << " is " << train_loss << endl;
		cout << "Training accuracy at epoch " << t + 1 << " is " << train_acc << endl;
	}
	//*/
	return 0;
}

