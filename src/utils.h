#ifndef LINEAR_REGRESSION_UTILS_H
#define LINEAR_REGRESSION_UTILS_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string_view>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd parseCsv(string filename);

VectorXd standardDeviations(MatrixXd& m);

double computeCost(MatrixXd& X, VectorXd& y, VectorXd& theta, int size);

VectorXd computeCosts(MatrixXd& X, VectorXd& y,  MatrixXd& thetas, int size);

void gradientDescent(MatrixXd& X,  VectorXd& y,  VectorXd &theta, double alpha, int iterations, int size, MatrixXd &thetaHistory);

#endif //LINEAR_REGRESSION_UTILS_H