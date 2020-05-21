#include <iostream>
#include "ex1a.h"
#include "ex1b.h"
#include "ex2a.h"
#include "ex2b.h"
#include "ex3a.h"

using namespace std;

std::map<string, function<int (const int argc, const char** argv)>> exercises = {
        {"1a", ex1a},
        {"1b", ex1b},
        {"2a", ex2a},
        {"2b", ex2b},
        {"3a", ex3a}
};

int main(const int argc, const char** argv) {
    bool ok = false;
    if (argc >= 2) {
        if (exercises.find(argv[1]) != exercises.end()) {
            ok = true;
            function<int(const int argc, const char **argv)> fn = exercises.at(argv[1]);
            fn(argc, argv);
        } else {
            cout << "Exercise " << argv[1] << " not found!" << endl;
        }
    }

    if (!ok) {
        cout << "usage: " << argv[0] << " {problem number}" << endl;
    }
}