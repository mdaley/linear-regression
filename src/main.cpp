#include <iostream>
#include "ex1a.h"
#include "ex1b.h"

using namespace std;

int main(int argc, char** argv) {
    bool ok = false;
    if (argc == 2) {
        if (string("1a").compare(argv[1]) == 0) {
            return ex1a();
        } else if (string("1b").compare(argv[1]) == 0) {
            return ex1b();
        }
    }

    if (!ok) {
        cout << "usage: " << argv[0] << " {problem number}" << endl;
    }
}