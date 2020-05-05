#ifndef LINEAR_REGRESSION_UTILS_H
#define LINEAR_REGRESSION_UTILS_H

#include <vector>
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

#endif //LINEAR_REGRESSION_UTILS_H