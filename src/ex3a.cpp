#include "ex3a.h"

struct mat_t_deleter {
    void operator()(mat_t *p) {
        Mat_Close(p);
    }
};

int ex3a(const int argc, const char** argv) {
    unique_ptr<mat_t, mat_t_deleter> matfp(Mat_Open("/Users/mdaley/workspace/clion/ml/data/ex3a_data.mat", MAT_ACC_RDONLY));

    if (NULL == matfp.get()) {
        cout << "Couldn't open .mat file" << endl;
        return 1;
    }

    matvar_t *X_mat = Mat_VarRead(matfp.get(), "X");
    Mat_VarPrint(X_mat, 1);

    matvar_t *y_mat = Mat_VarRead(matfp.get(), "y");
    Mat_VarPrint(y_mat, 1);

    int size = X_mat->dims[0];
    int columns = X_mat->dims[1];
    cout << "ROWS " << size << " COLUMNS " << columns << endl;

    matrix<double> X(size, columns);

    const double *X_mat_data = static_cast<const double*>(X_mat->data);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < columns; j++) {
            X(i,j) = X_mat_data[i * columns + j];
        }
    }

    // cout << X << endl;

    column_vector y(size);

    const double *y_mat_data = static_cast<const double*>(y_mat->data);
    for (int i = 0; i < size; i++) {
        y(i) = y_mat_data[i];
    }

    cout << trans(y) << endl;

    return 0;
}
