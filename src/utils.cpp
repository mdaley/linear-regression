#include "utils.h"

using namespace std;
using namespace Eigen;
using namespace dlib;

std::vector<string_view> splitString(const string_view strv, const string_view delims) {
    std::vector<string_view> output;
    size_t first = 0;

    while (first < strv.size())
    {
        const auto second = strv.find_first_of(delims, first);

        if (first != second)
            output.emplace_back(strv.substr(first, second - first));

        if (second == std::string_view::npos)
            break;

        first = second + 1;
    }

    return output;
}

MatrixXd parseCsv(string filename) {
    ifstream file = ifstream(filename);

    if (!file.good()) {
        throw invalid_argument("file invalid: " + filename);
    }

    string line;
    std::vector<std::vector<double>> vectors;
    while(getline(file, line)) {
        std::vector<string_view> values = splitString(line, ",");
        std::vector<double> v(values.size());
        for (int i = 0; i < values.size(); i++) {
            v[i] = stod(string(values.at(i)));
        }

        vectors.emplace_back(v);
    }

    int rows = vectors.size();
    int cols = vectors[0].size();

    MatrixXd m(rows, cols);

    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++ ) {
            m(i, j) = vectors[i][j];
        }
    }

    return m;
}

matrix<double> parseCsvDlib(string filename) {
    ifstream file = ifstream(filename);

    if (!file.good()) {
        throw invalid_argument("file invalid: " + filename);
    }

    string line;
    std::vector<std::vector<double>> vectors;
    while(getline(file, line)) {
        std::vector<string_view> values = splitString(line, ",");
        std::vector<double> v(values.size());
        for (int i = 0; i < values.size(); i++) {
            v[i] = stod(string(values.at(i)));
        }

        vectors.emplace_back(v);
    }

    int rows = vectors.size();
    int cols = vectors[0].size();

    matrix<double> m(rows, cols);

    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++ ) {
            m(i, j) = vectors[i][j];
        }
    }

    return m;
}

VectorXd standardDeviations(MatrixXd& m) {
    VectorXd means = m.colwise().mean();

    MatrixXd n = m.rowwise() - means.transpose();

    ArrayXd sqns = m.colwise().squaredNorm().array();

    ArrayXd sds = (sqns / m.rows()).sqrt();

    return sds.matrix();
}

double computeCost(MatrixXd& X, VectorXd& y,  VectorXd& theta, int size) {
    VectorXd X_Theta = X * theta;
    VectorXd X_Theta_minus_y = X_Theta - y;
    double vm = X_Theta_minus_y.transpose() * X_Theta_minus_y;
    return vm / (2.0 * size);
}

VectorXd computeCosts(MatrixXd& X, VectorXd& y,  MatrixXd& thetas, int size) {
    MatrixXd X_Theta = X * thetas.transpose();
    MatrixXd X_Theta_minus_y = X_Theta.colwise().operator-=(y);
    VectorXd vm = (X_Theta_minus_y.transpose() * X_Theta_minus_y).diagonal();
    return vm / (2.0 * size);
}

void gradientDescent(MatrixXd& X,  VectorXd& y,  VectorXd &theta, double alpha, int iterations, int size, MatrixXd &thetaHistory) {
    for (int i = 0; i < iterations; i++) {

        VectorXd h(size);
        h = X * theta;

        double f = alpha / size;

        VectorXd h_y(size);
        h_y = h - y;

        float cost = computeCost(X, y, theta, size);

        //cout << "Cost = " << cost << endl;

        for (int j = 0; j < theta.size(); j++) {
            theta[j] -= f * (h_y.array() * X.col(j).array()).sum();
        }

        //cout << "Theta: " << theta.transpose() << endl;
        thetaHistory.row(i) = theta;
    }
}

double _sigmoid(double z) {
    return 1.0 / (1 + exp(-z));
}

MatrixXd sigmoid(MatrixXd& z) {
    return z.unaryExpr(&_sigmoid);
}

VectorXd sigmoid(VectorXd& z) {
    return z.unaryExpr(&_sigmoid);
}

double sigmoid(double z) {
    return _sigmoid(z);
}

double _oneMinus(double d) {
    return 1 - d;
}

double _logOneMinus(double d) {
    return log(1 - d);
}

double _log(double d) {
    return log(d);
}

double computeLogisticRegressionCost(MatrixXd& X, VectorXd& y, VectorXd& theta, VectorXd& gradient) {
    VectorXd Theta_X = X * theta;

    VectorXd H = sigmoid(Theta_X);

    int size = X.rows();

    double s = (-1.0 * (y.transpose() * H.unaryExpr(&_log))(0,0))
            - ((y.transpose().unaryExpr(&_oneMinus)) * (H.unaryExpr(&_logOneMinus)));

    double cost = s / size;

    gradient = ((H - y).transpose() * X) / size;

    return cost;
}

double computeLogisticRegressionCostDlib(matrix<double>& X, column_vector& y, column_vector& theta, column_vector& gradient) {
    column_vector theta_X = X * theta;
    column_vector H = dlib::sigmoid(theta_X);

    long size = X.nr();

    double s = ((trans(y) * -1) * log(H)) - ((1 - trans(y)) * log(1 - H));

    double cost = s / size;

    gradient = (trans(H - y) * X) / size;

    return cost;
}