//
// Created by Matthew Daley on 10/05/2020.
//

#include "ex2a.h"

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

int ex2a() {
    cout << "Logistic regression..." << endl;

    MatrixXd data = parseCsv("data/ex2a.csv");

    cout << data << endl;

    vector<double> admittedExam1, admittedExam2, notAdmittedExam1, notAdmittedExam2;
    for (int i = 0; i < data.rows(); i++) {
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

    return 0;
}