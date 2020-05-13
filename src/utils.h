#ifndef LINEAR_REGRESSION_UTILS_H
#define LINEAR_REGRESSION_UTILS_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string_view>
#include <Eigen/Dense>
#include <dlib/matrix.h>

using namespace std;
using namespace Eigen;\
using namespace dlib;

typedef matrix<double,0,1> column_vector;

MatrixXd parseCsv(string filename);

matrix<double> parseCsvDlib(string filename);

VectorXd standardDeviations(MatrixXd& m);

double computeCost(MatrixXd& X, VectorXd& y, VectorXd& theta, int size);

VectorXd computeCosts(MatrixXd& X, VectorXd& y,  MatrixXd& thetas, int size);

void gradientDescent(MatrixXd& X,  VectorXd& y,  VectorXd &theta, double alpha, int iterations, int size, MatrixXd &thetaHistory);

double sigmoid(double z);

MatrixXd sigmoid(MatrixXd& z);

VectorXd sigmoid(VectorXd& z);

double computeLogisticRegressionCost(MatrixXd& X, VectorXd& y, VectorXd& theta, VectorXd& gradient);

double computeLogisticRegressionCostDlib(matrix<double>& X, column_vector& y, column_vector& theta, column_vector& gradient);

#endif //LINEAR_REGRESSION_UTILS_H