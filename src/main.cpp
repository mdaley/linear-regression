#include <iostream>
#include "CSV.h"
#include "Data.h"
#include "matplotlib.h"

using namespace std;
namespace plt = matplotlibcpp;

typedef vector<vector<float>> Matrix;
typedef vector<float> Vector;

float computeCost(Matrix X, Vector y,  Vector theta, int size) {
    Vector h(size);
    for (int i = 0; i < size; i++) {
        h[i] = theta[0] * X[0][i] + theta[1] * X[1][i];
    }
    float s = 0.0f;
    for (int i = 0; i < size; i++) {
        float d = h[i] - y[i];
        s += d * d;
    }

    return s / (2.0f * size);
}

void vecMul(Vector a, Vector b, Vector &c, const int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

float sum(Vector a, int size) {
    float r = 0.0f;
    for (int i = 0; i < size; i++) {
        r += a[i];
    }

    return r;
}

float sumVecMul(Vector a, Vector b, const int size) {
    Vector c(size);
    vecMul(a, b , c, size);
    return sum(c, size);
}

void gradientDescent(Matrix X,  Vector y,  Vector &theta, float alpha, int iterations, int size, Matrix &thetaHistory) {
    for (int i = 0; i < iterations; i++) {
        Vector h(size);
        for (int i = 0; i < size; i++) {
            h[i] = theta[0] * X[0][i] + theta[1] * X[1][i];
        }

        float f = alpha / size;

        Vector h_y(size);
        for (int i = 0; i < size; i++) {
            h_y[i] = h[i] - y[i];
        }

        float cost = computeCost(X, y, theta, size);

        cout << "Cost = " << cost << endl;

        theta[0] -= f * sumVecMul(h_y, X[0], size);
        theta[1] -= f * sumVecMul(h_y, X[1], size);

        // capture the history of theta and the cost function
        thetaHistory[0][i] = theta[0];
        thetaHistory[1][i] = theta[1];
        thetaHistory[2][i] = cost;
    }
}

int main() {
    cout << "Linear regression..." << endl;

    function<Data(vector<string_view>)> mappingFn = [] (vector<string_view> in) -> Data {
        if (in.size() != 2) {
            throw invalid_argument("Data row wrong size");
        }
        Data d;
        d.setX(stof(string(in.at(0)).c_str()));
        d.setY(stof(string(in.at(1)).c_str()));
        return d;
    };

    vector<Data> data = CSV::parseCsv("/Users/mdaley/workspace/clion/linear-regression/ex1data.csv", mappingFn);

    int size = data.size();
    cout << "Data size = " << size << endl;
    // arrays of x and y values
    Vector x(size);
    Vector y(size);

    for (int i = 0; i < size; i++) {
        Data d = data.at(i);
        x[i] = d.getX();
        y[i] = d.getY();
    }

    cout << "x values = ";
    for (int i = 0; i < size; i++) {
        cout << x[i] << " ";
    }
    cout << endl;

    cout << "y values = ";
    for (int i = 0; i < size; i++) {
        cout << y[i] << " ";
    }
    cout << endl;

    // matrix with col 0 all ones and col 1 being the x values
    Matrix X(2);
    X[0] = Vector(size);
    X[1] = Vector(size);
    for (int i = 0; i < size; i++) {
        X[0][i] = 1.0f;
        X[1][i] = x[i];
    }

    cout << "X matrix values = ";
    for (int i = 0; i < size; i++) {
        cout << "[" << X[0][i] << "," << X[1][i] << "] ";
    }
    cout << endl;

    // array of two initially zero values
    Vector theta = {0.0f, 0.0f};

    float initialCost = computeCost(X, y, theta, size);
    cout << "Initial Cost = " << initialCost << endl;

    cout << "Do gradient descent..." << endl;
    int iterations = 1500;
    Matrix thetaHistory(3);
    thetaHistory[0] = Vector(iterations); // theta 0
    thetaHistory[1] = Vector(iterations); // theta 1
    thetaHistory[2] = Vector(iterations); // cost function

    gradientDescent(X, y, theta, 0.01f, iterations, size, thetaHistory);
    cout << "Theta is " << theta[0] << " " << theta[1] << endl;

    // create y points using final theta and the x points
    Vector yFinal(size);
    for (int i = 0; i < size; i++) {
        yFinal[i] = X[0][i] * theta[0] + X[1][i] * theta[1];
    }

    // theta history
    float theta0min = numeric_limits<float>::max();
    float theta0max = numeric_limits<float>::min();
    float theta1min = numeric_limits<float>::max();
    float theta1max = numeric_limits<float>::min();
    for (int i = 0; i < iterations; i++) {
        theta0min = thetaHistory[0][i] < theta0min ? thetaHistory[0][i] : theta0min;
        theta0max = thetaHistory[0][i] > theta0max ? thetaHistory[0][i] : theta0max;
        theta1min = thetaHistory[1][i] < theta1min ? thetaHistory[1][i] : theta1min;
        theta1max = thetaHistory[1][i] > theta1max ? thetaHistory[1][i] : theta1max;
    }

    cout << theta0min << " <= theta[0] >= " << theta0max << endl;
    cout << theta1min << " <= theta[1] >= " << theta1max << endl;

    // print a graph of the results
    plt::scatter(x, y, 5.0f);
    plt::plot(x, yFinal, "r-");
    plt::ylabel("Profit in $10,000s");
    plt::xlabel("Population of city in 10,000s");
    plt::title("Linear regression\n");
    plt::show();

    // print a surface graph of cost for different values of theta
    Matrix a, b, c;

    for (int i = 0; i < 101; i++) {
        Vector a_row, b_row, c_row;
        for (int j = 0; j < 101; j++) {
            float t0 = -10 + i * 0.2f;
            float t1 = -1.0f + j * 0.05f;
            vector<float> t {t0, t1};
            float cost = computeCost(X, y, t, size);
            a_row.push_back(t0);
            b_row.push_back(t1);
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
    plt::show();

    return 0;
}

