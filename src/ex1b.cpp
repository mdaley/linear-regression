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

    return 0;
}

