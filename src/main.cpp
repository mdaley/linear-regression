#include <iostream>
#include "CSV.h"
#include "Data.h"

using namespace std;

int main() {
    cout << "Hello, World!" << endl;

    function<Data(vector<string_view>)> mappingFn = [] (vector<string_view> in) -> Data {
        Data d;
        d.setX(stof(string(in.at(0)).c_str()));
        d.setY(stof(string(in.at(1)).c_str()));
        return d;
    };

    vector<Data> results = CSV::parseCsv("ex1data.csv", mappingFn);

    cout << "In a vector" << endl;
    for (Data result : results) {
        cout << result.getX() << " " << result.getY() << endl;
    }

    return 0;
}
