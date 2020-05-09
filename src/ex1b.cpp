#include "ex1b.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

using namespace std;

VectorXd costsForAlpha(MatrixXd& X, VectorXd& y, double alpha, int iterations, int size) {
    VectorXd theta(3);
    theta << 0.0, 0.0, 0.0;

    MatrixXd thetaHistory(iterations, theta.size());

    gradientDescent(X, y, theta, alpha, iterations, size, thetaHistory);

    VectorXd costs = computeCosts(X, y, thetaHistory, size);

    return costs;
}

int ex1b() {
    cout << "Multiple variable linear regression..." << endl;

    MatrixXd data = parseCsv("/Users/mdaley/workspace/clion/linear-regression/ex1b_data.csv");

    cout << " DATA" << endl << data << endl;

    MatrixXd _X = data.leftCols(data.cols() - 1);
    VectorXd y = data.rightCols(1);
    int size = data.rows();

    VectorXd means = _X.colwise().mean();

    cout << "MEANS" << endl << means.transpose() << endl;

    _X.rowwise() -= means.transpose();

    cout << "MEAN SUBTRACTED" << endl << _X << endl;

    VectorXd sds = standardDeviations(_X);

    cout << "STANDARD DEVIATIONS " << endl << sds.transpose() << endl;

    for (int i = 0; i < _X.cols(); i++) {
        _X.col(i) /= sds[i];
    }

    cout << "NORMALISED " << endl << _X << endl;

    MatrixXd X(_X.rows(), _X.cols() + 1);
    X << VectorXd::Ones(_X.rows()), _X;

    cout << "X " << endl << X << endl;

    cout << "y " << endl << y.transpose() << endl;

    VectorXd theta(3);
    theta << 0.0, 0.0, 0.0;

    cout << "Initial cost = " << computeCost(X, y, theta, size) << endl;

    //double alpha = 0.03;
    int iterations = 100;

    vector<pair<string, vector<double>>> costsByAlpha;

    for (string alpha : {"0.01", "0.03", "0.1", "0.3", "1.0"}) {
        double alpha_d = stod(alpha);
        VectorXd costs = costsForAlpha(X, y, alpha_d, iterations, size);
        pair<string, vector<double>> p;
        p.first = alpha;
        p.second = vector<double>(costs.data(), costs.data() + costs.size());
        costsByAlpha.push_back(p);
    }

    vector<double> iterations_v;
    for (int i = 0; i < iterations; i++) {
        iterations_v.push_back((double) i);
    }

    //vector<double> costs_v = vector<double>(costs.data(), costs.data() + costs.size());

    cout << "iterations count " << iterations_v.size() << endl;
    //cout << "costs count " << costs_v.size() << endl;
    //vector<double> costs_v = costsByAlpha.at(0).second;

    for (pair<string, vector<double>> costByAlpha : costsByAlpha) {
        map<string, string> settings;
        settings.insert(pair<string, string>("label", costByAlpha.first));
        plt::plot(iterations_v, costByAlpha.second);
    }
    plt::ylabel_u(L"J(\u03b8)");
    plt::xlabel("Number of iterations");
    plt::title("Convergence\n");
    plt::show();
    return 0;
}

