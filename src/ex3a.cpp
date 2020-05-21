#include "ex3a.h"

int ex3a(const int argc, const char** argv) {
    matrix<double> X = openMatLabData("/Users/mdaley/workspace/clion/ml/data/ex3a_data.mat", "X");
    column_vector y = openMatLabData("/Users/mdaley/workspace/clion/ml/data/ex3a_data.mat", "y");

    cout << trans(y) << endl;

    return 0;
}
