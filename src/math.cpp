#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
namespace py = pybind11;


vector<int> primes(int n){
    // TODO: euler algorithm
    vector<int> res;
    char* flags = new char[n+1]();
    int range = sqrt(n)+1;
    for (int i=2; i<=range; i++) if (flags[i] == 0) [[unlikely]] for (int j=i*2; j<=n; j+=i) flags[j] = 1;
    for (int i=2; i<=n; i++) if(flags[i] == 0) [[unlikely]] res.push_back(i);
    return res;
}


PYBIND11_MODULE(_kzhutil, m) {
    m.def("primes", &primes, "all primes lower than n", py::arg("n"));
}
