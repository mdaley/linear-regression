#include <iostream>
#include "ex1a.h"
#include "ex1b.h"
#include "ex2a.h"
#include "ex2b.h"

using namespace std;

int main(int argc, char** argv) {
    bool ok = false;
    if (argc == 2) {
        if (string("1a").compare(argv[1]) == 0) {
            return ex1a();
        } else if (string("1b").compare(argv[1]) == 0) {
            return ex1b();
        } else if (string("2a").compare(argv[1]) == 0) {
            return ex2a();
        } else if (string("2b").compare(argv[1]) == 0) {
            return ex2b();
        }
    }

    if (!ok) {
        cout << "usage: " << argv[0] << " {problem number}" << endl;
    }
}