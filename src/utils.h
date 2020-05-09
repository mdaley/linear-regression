#ifndef LINEAR_REGRESSION_UTILS_H
#define LINEAR_REGRESSION_UTILS_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

VectorXd standardDeviations(MatrixXd& m);

double computeCost(MatrixXd& X, VectorXd& y, VectorXd& theta, int size);

/*#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <sstream>
#include <math.h>

using namespace std;

typedef vector<vector<float>> Matrix;
typedef vector<float> Vector;

float computeCost(Matrix& X, Vector& y,  Vector& theta, int size);

void vecMul(Vector& a, Vector& b, Vector &c, int size);

float sum(Vector& a, int size);

float sumVecMul(Vector& a, Vector& b, int size);

int rowCount(const Matrix& m);

int columnCount(const Matrix& m);

Matrix transpose(const Matrix& m);

Vector rowMeans(const Matrix& m);

Vector columnMeans(const Matrix& m);

Vector rowStandardDeviations(const Matrix& m);

Vector columnStandardDeviations(const Matrix& m);

void subtract(Matrix& m, const Vector& v);

void divide(Matrix& m, const Vector& v);

void printMatrix(Matrix& m);

void printVector(Vector& v);

Matrix columnsSubMatrix(Matrix& m, int colStart, int colEnd);

Vector columnOfMatrix(Matrix& m, int col);

Matrix multiply(Matrix& m, Matrix& n);

Matrix multiply(Matrix& m, Vector& v);*/

#endif //LINEAR_REGRESSION_UTILS_H