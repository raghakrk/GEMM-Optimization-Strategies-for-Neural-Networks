#include "utils.h"

//Matrix Transpose
vector<vector<double>> MatTranspose(vector<vector<double>>& MAT)
{
	int row = MAT.size();
	int col = MAT[0].size();
	vector<vector<double>> MAT_T(col, vector<double>(row));
	for (int i = 0; i < row; ++i)
		for (int j = 0; j < col; ++j) {
			MAT_T[j][i] = MAT[i][j];
		}
	return MAT_T;
}

//Create 2D Matrix
vector<vector<double>> make2Dvec(int row, int col)
{
	return vector<vector<double>>(row, vector<double>(col));
}

//Function to multiply Matrices
vector<vector<double>> MATMUL(vector<vector<double>> x, vector<vector<double>> w)
{
	int Wrow = w.size();
	int Wcol = w[0].size();
	int Xrow = x.size();
	int Xcol = x[0].size();
	//cout << Wcol << " " << Xrow << endl;
	vector<vector<double>> h(Xrow, vector<double>(Wcol));
	try {
		if (Xcol != Wrow)
			throw "Dimension mismatch in MATMUL!";
		for (int i = 0; i < Xrow; i++)
		{
			for (int j = 0; j < Wcol; j++)
			{

				float sum = 0;
				for (int k = 0; k < Xcol; k++)
				{
					sum += float(x[i][k]) * w[k][j];
				}
				h[i][j] = sum;
			}
		}
	}
	catch (const char* msg) {
		cerr << msg << endl;
	}
	return h;
}

// Function to add vector and matrix
void vectoradd(vector<vector<double>> &WX, vector<vector<double>> b)
{
	int row = WX.size();
	int col = WX[0].size();
	//cout << b.size() << " " << row << endl;
	try {
		if (b[0].size() != col)
			throw "Dimension mismatch in vector add!";
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				WX[i][j] += b[0][j];
			}
		}
	}
	catch (const char* msg) {
		cerr << msg << endl;
	}
}

// Function to sum up the rows
vector<vector<double>>rowsum(vector<vector<double>> input)
{
	int row = input.size();
	vector<vector<double>> sum(row, vector<double>(1));
	for (auto i = 0; i < row; ++i) {
		sum[i][0]= accumulate(begin(input[i]), end(input[i]), 0);
	}
	return sum;
}

//function for matrix element product
vector<vector<double>> multiply(vector<vector<double>>A, vector<vector<double>>B)
{
	int row= A.size();
	int col=A[0].size();
	vector<vector<double>> product(row,vector<double>(col));
	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{
			product[i][j] = A[i][j] * B[i][j];
		}
	}
	return product;
}



