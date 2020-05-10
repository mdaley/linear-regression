#include "ex1a.h"

using namespace std;
namespace plt = matplotlibcpp;

int ex1a() {
    cout << "Single variable linear regression..." << endl;

    MatrixXd data = parseCsv("data/ex1a.csv");

    int size = data.rows();

    VectorXd x(size);
    x << data.leftCols(1);

    MatrixXd X(data.rows(), 2);
    X << VectorXd::Ones(size), x;

    VectorXd y(size);
    y << data.rightCols(1);

    cout << "X = " << endl << X << endl;

    cout << "y = " << y.transpose() << endl;

    VectorXd theta(2);
    theta << 0, 0;

    double initialCost = computeCost(X, y, theta, size);
    cout << "Initial cost = " << initialCost << endl;

    int iterations = 1500;
    MatrixXd thetaHistory(iterations, theta.size());

    gradientDescent(X, y, theta, 0.01, iterations, size, thetaHistory);

    cout << "Theta after gradient descent = " << theta.transpose() << endl;

    VectorXd finalY(size);
    finalY = X * theta;

    vector<double> x_v = vector<double>(x.data(), x.data() + x.size());
    vector<double> y_v = vector<double>(y.data(), y.data() + y.size());
    vector<double> finalY_v = vector<double>(finalY.data(), finalY.data() + finalY.size());
    plt::scatter(x_v, y_v, 5.0f);
    plt::plot(x_v, finalY_v, "r-");
    plt::ylabel("Profit in $10,000s");
    plt::xlabel("Population of city in 10,000s");
    plt::title("Linear regression\n");
    plt::show();

    MatrixXd thetaMinMax(2, theta.size());
    thetaMinMax.row(0) = thetaHistory.colwise().minCoeff();
    thetaMinMax.row(1) = thetaHistory.colwise().maxCoeff();

    cout << "Theta min / max =" << endl << thetaMinMax << endl;

    vector<vector<double>> a, b, c;

    for (int i = 0; i < 101; i++) {
        vector<double> a_row, b_row, c_row;
        for (int j = 0; j < 101; j++) {
            VectorXd t(2);
            t << -10 + i * 0.2f, -1.0f + j * 0.05f;
            float cost = computeCost(X, y, t, size);
            a_row.push_back(t(0));
            b_row.push_back(t(1));
            c_row.push_back(cost);
        }
        a.push_back(a_row);
        b.push_back(b_row);
        c.push_back(c_row);
    }

    map<string, plt::SettingValue> settings;

    settings.insert({"edgecolor", plt::SettingValue(string("black"))});
    settings.insert({"linewidth", plt::SettingValue(2.0f)});
    settings.insert({"linestyle", plt::SettingValue(string("--"))});
    settings.insert({"alpha", plt::SettingValue(0.5f)});
    settings.insert({"rstride", plt::SettingValue(10)});
    settings.insert({"cstride", plt::SettingValue(10)});

    settings.insert({"cmap", plt::SettingValue(string("gist_rainbow"))});

    plt::plot_surface(a, b, c, settings);
    plt::xlabel_u(L"\u03b8\u2080");
    plt::ylabel_u(L"\u03b8\u2081");
    plt::set_zlabel_u(L"J(\u03b8)");
    plt::show();

    return 0;
}

