#ifndef LINEAR_REGRESSION_CSV_H
#define LINEAR_REGRESSION_CSV_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string_view>

using namespace std;

class CSV {

    static vector<string_view> splitString(string_view strv, string_view delims = " ");
public:

    template <typename T>
    static vector<T> parseCsv(string filename, function<T(vector<string_view>)> mappingFn);
};

template <typename T>
vector<T> CSV::parseCsv(string filename, function<T(vector<string_view>)> mappingFn) {
    ifstream file = ifstream(filename);

    if (!file.good()) {
        throw invalid_argument("file invalid: " + filename);
    }

    vector<T> results;
    string line;

    while(getline(file, line)) {
        vector<string_view> values = splitString(line, ",");
        results.emplace_back(mappingFn(values));
    }

    return results;
}

#endif //LINEAR_REGRESSION_CSV_H
