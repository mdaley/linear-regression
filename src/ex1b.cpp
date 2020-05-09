#include "ex1b.h"

using namespace Eigen;

using namespace std;

int ex1b() {
    cout << "Multiple variable linear regression..." << endl;

    MatrixXd data = parseCsv("/Users/mdaley/workspace/clion/linear-regression/ex1b_data.csv");

    cout <<" DATA" << endl << data << endl;

    MatrixXd _X = data.leftCols(data.cols() - 1);
    VectorXd y = data.rightCols(1);

    VectorXd means = _X.colwise().mean();

    cout << "MEANS" << endl << means.transpose() << endl;

    _X.rowwise() -= means.transpose();

    cout << "MEAN SUBTRACTED" << endl << _X << endl;

    VectorXd sds = standardDeviations(_X);

    cout << "STANDARD DEVIATIONS " << endl << sds.transpose() << endl;

    for(int i = 0; i < _X.cols(); i++) {
        _X.col(i) /= sds[i];
    }

    cout << "NORMALISED " << endl << _X << endl;

    MatrixXd X(_X.rows(), _X.cols() + 1);
    X << VectorXd::Ones(_X.rows()), _X;

    cout << "X " << endl << X << endl;

    cout << "y " << endl << y.transpose() << endl;
    /*
    Matrix x = columnsSubMatrix(data, 0, 1);
    Vector y = columnOfMatrix(data, 2);
    Matrix X(x.size());
    for (int i = 0; i < x.size(); i++) {
        X[i] = Vector(x[i].size() + 1);
        X[i][0] = 1.0f;
        for (int j = 0; j < x[i].size(); j++) {
            X[i][j + 1] = x[i][j];
        }
    }

    cout << "X" << endl;
    printMatrix(X);

    printVector(y);

    Vector theta(3);

    cout << "X_thata" << endl;
    Matrix X_theta = multiply(X, theta);

    printMatrix(X_theta);*/
    return 0;
}

