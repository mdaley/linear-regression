//
// Created by Matthew Daley on 05/05/2020.
//

#include <cfloat>
#include "utils.h"

using namespace std;

float computeCost(Matrix& X, Vector& y,  Vector& theta, int size) {
    Vector h(size);
    for (int i = 0; i < size; i++) {
        h[i] = theta[0] * X[0][i] + theta[1] * X[1][i];
    }
    float s = 0.0f;
    for (int i = 0; i < size; i++) {
        float d = h[i] - y[i];
        s += d * d;
    }

    return s / (2.0f * (float)size);
}

void vecMul(Vector& a, Vector& b, Vector& c, const int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

float sum(Vector& a, int size) {
    float r = 0.0f;
    for (int i = 0; i < size; i++) {
        r += a[i];
    }

    return r;
}

float sumVecMul(Vector& a, Vector& b, const int size) {
    Vector c(size);
    vecMul(a, b , c, size);
    return sum(c, size);
}

int rowCount(const Matrix& m) {
    return m.size();
}

int columnCount(const Matrix& m) {
    return m[0].size();
}

Matrix transpose(const Matrix& m) {
    int cols = columnCount(m);
    int rows = rowCount(m);
    Matrix result(cols);
    for (int i = 0; i < cols; i++) {
        result[i] = vector<float>(rows);
        for (int j = 0; j < rows; j++) {
            result[i][j] = m[j][i];
        }
    }

    return result;
}

Vector rowMeans(const Matrix& m) {
    Vector rowMeans(m.size());
    for (int i = 0; i < m.size(); i++) {
        Vector row = m[i];
        for (int j = 0; j < row.size(); j++) {
            rowMeans[i] += row[j];
        }

        rowMeans[i] /= row.size();
    }

    return rowMeans;
}

Vector columnMeans(const Matrix& m) {
    return rowMeans(transpose(m));
}

Vector rowStandardDeviations(const Matrix& m) {
    Vector means = rowMeans(m);
    Vector sds(m.size());

    for (int i = 0; i < m.size(); i++) {
        Vector row = m[i];
        for (int j = 0; j < row.size(); j++) {
            float diff = means[i] - row[j];
            sds[i] += diff * diff;
        }

        sds[i] /= row.size();
        sds[i] = sqrt(sds[i]);
    }

    return sds;
}

Vector columnStandardDeviations(const Matrix& m) {
    return rowStandardDeviations(transpose(m));
}

void subtract(Matrix& m, const Vector& v) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < v.size(); j ++) {
            m[i][j] -= v[j];
        }
    }
}

void divide(Matrix& m, const Vector& v) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < v.size(); j++) {
            m[i][j] /= v[j];
        }
    }
}

void printMatrix(Matrix& m) {
    cout.precision(6);
    for (int i = 0; i < m.size(); i++) {
        Vector row = m[i];
        for(int j = 0; j < row.size(); j++) {
            cout << setw(13) << right << row[j] << " ";
        }
        cout << endl;
    }
}