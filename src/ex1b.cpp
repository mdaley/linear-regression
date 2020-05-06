//
// Created by Matthew Daley on 05/05/2020.
//

#include "ex1b.h"

using namespace std;

int ex1b() {
    cout << "Multiple variable linear regression..." << endl;

    function<vector<float>(vector<string_view>)> mappingFn = [] (vector<string_view> in) -> vector<float> {
        
        vector<float> data(in.size());
        
        for (int i = 0; i < in.size(); i++) {
            data[i] = stof(string(in.at(i)).c_str());
        }
        return data;
    };

    Matrix data = CSV::parseCsv("/Users/mdaley/workspace/clion/linear-regression/ex1b_data.csv", mappingFn);

    // normalise
    subtract(data, columnMeans(data));
    divide(data, columnStandardDeviations(data));

    cout << "NORMALISED" << endl;
    printMatrix(data);

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

    printMatrix(X_theta);
    return 0;
}

