#include <iostream>
#include "CSV.h"
#include "Data.h"

using namespace std;

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

void gradientDescent(Matrix X,  Vector y,  Vector &theta, float alpha, int iterations, int size) {
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

        cout << "Cost = " << computeCost(X, y, theta, size) << endl;

        theta[0] -= f * sumVecMul(h_y, X[0], size);
        theta[1] -= f * sumVecMul(h_y, X[1], size);
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
    gradientDescent(X, y, theta, 0.01f, 15000, size);
    cout << "Theta is " << theta[0] << " " << theta[1] << endl;
    return 0;
}

