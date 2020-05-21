#include "ex3a.h"

int ex3a(const int argc, const char** argv) {

    mat_t* matfp;

    matfp = Mat_Open("/Users/mdaley/workspace/clion/ml/data/ex3a_data.mat", MAT_ACC_RDONLY);

    if (NULL == matfp) {
        cout << "Couldn't open .mat file" << endl;
    }

    cout << "Perhaps opening worked!" << endl;

    matvar_t* X = Mat_VarRead(matfp, "X");
    Mat_VarPrint(X, 1);

    matvar_t* y = Mat_VarRead(matfp, "y");
    Mat_VarPrint(y, 1);
    
    Mat_Close(matfp);
    return 0;
}
