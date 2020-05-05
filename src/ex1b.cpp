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
    for (int i = 0; i < data.size(); i++) {
        Vector row = data[i];
        for(int j = 0; j < row.size(); j++) {
            cout << row[j] << " ";
        }
        cout << endl;
    }

    /*Matrix transposed = transpose(data);

    cout << "TRANSPOSED" << endl;
    for (int i = 0; i < transposed.size(); i++) {
        Vector row = transposed[i];
        for (int j = 0; j < row.size(); j++) {
            cout << row[j] << " ";
        }
        cout << endl;
    }

    Vector means = columnMeans(data);

    cout << "MEANS" << endl;
    for (int i = 0; i < means.size(); i++) {
      cout << means[i] << " ";
    }
    cout << endl;

    subtract(data, means);

    Vector sds = columnStandardDeviations(data);

    cout << "Standard Deviations" << endl;
    for (int i = 0; i < sds.size(); i++) {
        cout << sds[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < data.size(); i++) {
        Vector row = data[i];
        for(int j = 0; j < row.size(); j++) {
            cout << row[j] << " ";
        }
        cout << endl;
    }*/

    return 0;
}

