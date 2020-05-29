#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>  
#include <random>
#include <cmath>

using namespace std;
void read_mnist_images(string full_path, int& number_of_images, int& image_size, vector<vector<double>>& X);
vector<int> read_mnist_labels(string full_path, int& number_of_labels);
vector<vector<double>> make2Dvec(int row, int col);
vector<vector<double>> MatTranspose(vector<vector<double>>& MAT);
vector<vector<double>> MATMUL(vector<vector<double>> x, vector<vector<double>> w);
void vectoradd(vector<vector<double>>& WX, vector<vector<double>> b);
vector<vector<double>>rowsum(vector<vector<double>> input);
vector<vector<double>> multiply(vector<vector<double>>A, vector<vector<double>>B);


class softmax_crossentropy
{
public:
	int inputlen;
	int inputdim;
	softmax_crossentropy();
	double forward(vector<vector<double>> X, vector<vector<double>> Y);
	vector<vector<double>> backward(vector<vector<double>> X, vector<vector<double>> Y);
	vector<vector<double>> backward_output;
	vector<vector<double>> prob;
	double forward_output;
};

class reLU
{
public:
	//Z=WX+b
	vector<vector<double>> forward(vector<vector<double>> Z);
	vector<vector<double>> backward(vector<vector<double>> Z, vector<vector<double>> grad);
};

class tanH
{
public:
	//Z=WX+b
	vector<vector<double>> forward(vector<vector<double>> Z);
	vector<vector<double>> backward(vector<vector<double>> Z, vector<vector<double>> grad);
};


