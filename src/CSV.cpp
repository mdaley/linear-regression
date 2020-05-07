#include "CSV.h"

using namespace std;
using namespace Eigen;

vector<string_view> splitString(const string_view strv, const string_view delims) {
    vector<string_view> output;
    size_t first = 0;

    while (first < strv.size())
    {
        const auto second = strv.find_first_of(delims, first);

        if (first != second)
            output.emplace_back(strv.substr(first, second - first));

        if (second == std::string_view::npos)
            break;

        first = second + 1;
    }

    return output;
}

MatrixXd parseCsv(string filename) {
    ifstream file = ifstream(filename);

    if (!file.good()) {
        throw invalid_argument("file invalid: " + filename);
    }

    string line;
    vector<vector<double>> vectors;
    while(getline(file, line)) {
        vector<string_view> values = splitString(line, ",");
        vector<double> v(values.size());
        for (int i = 0; i < values.size(); i++) {
            v[i] = stod(string(values.at(i)));
        }

        vectors.emplace_back(v);
    }

    int rows = vectors.size();
    int cols = vectors[0].size();

    MatrixXd m(rows, cols);

    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++ ) {
            m(i, j) = vectors[i][j];
        }
    }

    return m;
}