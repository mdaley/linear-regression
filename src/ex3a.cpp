#include "ex3a.h"

int ex3a(const int argc, const char** argv) {
    matrix<double> X = openMatLabData("data/ex3a_data.mat", "X");
    column_vector y = openMatLabData("data/ex3a_data.mat", "y");

    cout << trans(y) << endl;

    return 0;
}
