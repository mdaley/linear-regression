#include "ex1b.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

using namespace std;

VectorXd costsForAlpha(MatrixXd& X, VectorXd& y, double alpha, int iterations, int size, VectorXd& calcTheta) {
    VectorXd theta(3);
    theta << 0.0, 0.0, 0.0;

    MatrixXd thetaHistory(iterations, theta.size());

    gradientDescent(X, y, theta, alpha, iterations, size, thetaHistory);

    VectorXd costs = computeCosts(X, y, thetaHistory, size);

    calcTheta = theta;
    return costs;
}

int ex1b(const int argc, const char** argv) {
    cout << "Multiple variable linear regression..." << endl;

    MatrixXd data = parseCsv("data/ex1b.csv");

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

    int iterations = 100;

    std::vector<pair<string, std::vector<double>>> costsByAlpha;
    std::vector<VectorXd> calculatedThetas;

    for (string alpha : {"0.01", "0.03", "0.1", "0.3", "1.0"}) {
        double alpha_d = stod(alpha);
        VectorXd calcTheta(3);
        VectorXd costs = costsForAlpha(X, y, alpha_d, iterations, size, calcTheta);
        calculatedThetas.push_back(calcTheta);
        pair<string, std::vector<double>> p;
        p.first = alpha;
        p.second = std::vector<double>(costs.data(), costs.data() + costs.size());
        costsByAlpha.push_back(p);
    }

    std::vector<double> iterations_v;
    for (int i = 0; i < iterations; i++) {
        iterations_v.push_back((double) i);
    }

    cout << "iterations count " << iterations_v.size() << endl;

    for (pair<string, std::vector<double>> costByAlpha : costsByAlpha) {
        plt::named_plot(costByAlpha.first, iterations_v, costByAlpha.second);
    }

    VectorXd scaledTheta = calculatedThetas[4];
    string scaledAlpha = costsByAlpha[4].first;
    VectorXd scaledHouse(3);
    scaledHouse << 1.0, (1650.0 - means[0]) / sds[0], (3.0 - means[1]) / sds[1];
    double scaledPrice = scaledHouse.transpose() * scaledTheta;

    cout << "Scaled theta (alpha " << scaledAlpha << ") = " << scaledTheta.transpose() << endl;
    cout << "House price (scaled approach) " << scaledPrice << endl;

    MatrixXd X_unscaled(data.rows(), data.cols());
    X_unscaled << VectorXd::Ones(data.rows()), data.leftCols(data.cols() - 1);
    VectorXd calcTheta = ((X_unscaled.transpose() * X_unscaled).inverse() * X_unscaled.transpose()) * y;
    VectorXd house(3);
    house << 1.0, 1650.0, 3.0;
    double calcPrice = house.transpose() * calcTheta;

    cout << "Calculated theta " << calcTheta.transpose() << endl;
    cout << "Calculated theta: price = " << calcPrice << endl;

    plt::ylabel_u(L"J(\u03b8)");
    plt::xlabel("Number of iterations");
    plt::title("Convergence\n");
    plt::legend();
    plt::show();
    return 0;
}