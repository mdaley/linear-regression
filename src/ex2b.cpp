#include "ex2b.h"

using namespace std;
using namespace dlib;
namespace plt = matplotlibcpp;

int ex2b() {
    cout << "Multi-variable logistic regression..." << endl;

    matrix<double> data = parseCsvDlib("/Users/mdaley/workspace/clion/ml/data/ex2b.csv");

    std::vector<double> acceptedTest1, acceptedTest2, rejectedTest1, rejectedTest2;
    for (long i = 0; i < data.nr(); i++) {
        if (data(i, 2) == 0) {
            rejectedTest1.push_back(data(i, 0));
            rejectedTest2.push_back(data(i, 1));
        } else {
            acceptedTest1.push_back(data(i, 0));
            acceptedTest2.push_back(data(i, 1));
        }
    }
    int size = data.nr();

    // the raw X values
    matrix<double> _X(colm(data, range(0, 1)));

    // X with column of 1s followed by pairs of columns for X, X^2, X^3, X^4, X^5, X^6.
    matrix<double> X(size, 1 + 6 * _X.nc());
    set_colm(X, 0) = 1;

    for (int i = 0; i < 6; i++) {
        set_colm(X, range(1 + 2 * i, 2 + 2 * i)) = pow(colm(_X, range(0, 1)), i + 1);
    }

    column_vector y(colm(data, 2));

    double lambda = 1;

    auto costFn = [X, y, lambda](const column_vector& theta) -> double {
        column_vector H = dlib::sigmoid(X * theta);
        double s = ((trans(y) * -1) * log(H)) - ((1 - trans(y)) * log(1 - H));
        double cost = s / X.nr();

        // add lambda
        column_vector thetaZero(theta);
        thetaZero(0) = 0;
        cost += (lambda / (2 * X.nr())) * sum(squared(theta));

        // if the log function has zero input, it returns NaN. And H can easily be 1 or zero.
        // So, if that happens return a high cost to warn off the minimisation algorithm!
        if (!is_finite(cost))
            return numeric_limits<double>::max();

        return cost;
    };

    auto derivativeFn = [X, y, lambda] (const column_vector& theta) -> column_vector {
        column_vector H = dlib::sigmoid(X * theta);
        column_vector derivative = trans((trans(H - y) * X) / X.nr());
        ;
        // add lambda
        column_vector thetaZero(theta);
        thetaZero(0) = 0;
        derivative += thetaZero * (lambda / X.nr());

        return derivative;
    };

    column_vector theta(X.nc(), 1);
    theta = 0;

    cout << "Initial theta = " << trans(theta);
    cout << "Initial cost = " << costFn(theta) << endl;

    try {
        cout << "Finding mimimum..." << endl;
        find_min(bfgs_search_strategy(),
                 objective_delta_stop_strategy(1e-7), //.be_verbose(),
                 costFn, derivativeFn, theta, -1);
    } catch (std::exception& e) {
        cout << e.what() << endl;
    }

    cout << "Final theta " << trans(theta);
    cout << "Final cost = " << costFn(theta) << endl;

    double test1min = min(colm(_X, 0));
    double test1max = max(colm(_X, 0));
    double test1range = test1max - test1min;
    double test2min = min(colm(_X, 1));
    double test2max = max(colm(_X, 1));
    double test2range = test2max - test1min;

    // Find all the points on the graph where the probability of acceptance is very close to 0.5
    // by examining a 1000 x 1000 mesh of points across the graph. Joining these up will show the
    // decision boundary. Can't work out how to plot as a smooth curve, so show as a scatter plot -
    // with enough points it looks enough like a line!
    std::vector<double> t1_v, t2_v;
    double delta = 0.0025;
    double test1inc = test1range / 1000;
    double test2inc = test2range / 1000;
    double t1 = test1min;
    double t2 = test2min;
    for (int i = 0; i < 1001; i++) {
        for (int j = 0; j < 1001; j++) {
            column_vector t = {1, t1, t2, pow(t1, 2), pow(t2, 2), pow(t1, 3), pow(t2, 3),
                                       pow(t1, 4), pow(t2, 4), pow(t1, 5), pow(t2, 5),
                                       pow(t1, 6), pow(t2, 6)};
            double v = sigmoid(trans(theta) * t);
            if (v > 0.5 - delta && v < 0.5 + delta) {
                t1_v.push_back(t1);
                t2_v.push_back(t2);
            }
            t2 += test2inc;
        }

        t1 += test1inc;
        t2 = test2min;
    }

    plt::scatter(acceptedTest1, acceptedTest2, 10.0,
                 {{"label", "accepted"}, {"color", "green"}, {"marker", "^"}});
    plt::scatter(rejectedTest1, rejectedTest2, 10.0,
                 {{"label", "rejected"}, {"color", "red"}, {"marker", "o"}});
    plt::scatter(t1_v, t2_v, 1.0, {{"label", "decision boundary"}, {"color", "black"}}); // boundary
    plt::ylabel("Microchip test 2");
    plt::xlabel("Microchip test 1");
    ostringstream os;
    os << "Microchip acceptance (lambda = " << lambda << ")\n";
    plt::title(os.str());
    plt::legend();
    plt::show();

    return 0;
}
