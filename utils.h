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
int ReverseInt(int i);
void ReadMNIST(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>>& arr);
vector<vector<double>> make2Dvec(int row, int col);
vector<vector<double>> MatTranspose(vector<vector<double>>& MAT);
vector<vector<double>> MATMUL(vector<vector<double>> x, vector<vector<double>> w);
void vectoradd(vector<vector<double>>& WX, vector<vector<double>> b);
vector<vector<double>>rowsum(vector<vector<double>> input);
vector<vector<double>> multiply(vector<vector<double>>A, vector<vector<double>>B);


class softmax_crossentropy
{
	int inputlen;
	int inputdim;
public:
	softmax_crossentropy(vector<vector<double>> X);
	vector<vector<double>> forward();
	vector<vector<double>> prob;
	vector<vector<double>> X;
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


