#include "ex2a.h"

using namespace std;
using namespace dlib;
namespace plt = matplotlibcpp;

int ex2a() {

    cout << "Logistic regression..." << endl;

    matrix<double> data = parseCsvDlib("/Users/mdaley/workspace/clion/ml/data/ex2a.csv");

    std::vector<double> admittedExam1, admittedExam2, notAdmittedExam1, notAdmittedExam2;
    for (long i = 0; i < data.nr(); i++) {
        if (data(i, 2) == 0) {
            notAdmittedExam1.push_back(data(i, 0));
            notAdmittedExam2.push_back(data(i, 1));
        } else {
            admittedExam1.push_back(data(i, 0));
            admittedExam2.push_back(data(i, 1));
        }
    }

    plt::scatter(admittedExam1, admittedExam2, 10.0f,
            {{"label", "admitted"}, {"color", "green"}, {"marker", "^"}});
    plt::scatter(notAdmittedExam1, notAdmittedExam2, 10.0f,
            {{"label", "not admitted"}, {"color", "red"}, {"marker", "o"}});
    plt::ylabel("Exam 2 score");
    plt::xlabel("Exam 1 score");
    plt::title("Admissions and exam results\n");
    plt::legend();
    plt::show();

    int size = data.nr();

    matrix<double> X(size, data.nc());
    set_colm(X, 0) = 1;
    set_colm(X, range(1, 2)) = colm(data, range(0,1));

    column_vector y(colm(data, 2));

    auto costFn = [X, y](const column_vector& theta) -> double {
        column_vector H = dlib::sigmoid(X * theta);
        double s = ((trans(y) * -1) * log(H)) - ((1 - trans(y)) * log(1 - H));
        double cost = s / X.nr();

        // if the log function has zero input, it returns NaN. And H can easily be 1 or zero.
        // So, if that happens return a high cost to warn off the minimisation algorithm!
        if (!is_finite(cost))
            return 1000.0;

        return cost;
    };

    auto derivativeFn = [X, y] (const column_vector& theta) -> column_vector {
        column_vector H = dlib::sigmoid(X * theta);
        column_vector derivative = trans((trans(H - y) * X) / X.nr());

        return derivative;
    };

    column_vector theta(3, 1);
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

    column_vector student = {1, 45, 85};
    cout << "Admission probability for student with Exam 1 = 45, Exam 2 = 85 is "
            << sigmoid(trans(theta) * student) << endl;

    return 0;
}