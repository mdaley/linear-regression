#ifndef LINEAR_REGRESSION_CSV_H
#define LINEAR_REGRESSION_CSV_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string_view>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd parseCsv(string filename);



#endif //LINEAR_REGRESSION_CSV_H
