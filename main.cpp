#include <pybind/pybind11.h>

namespace py = pybind11;

int testFunc()
{
    return 42;
}

PYBIND11_MODULE(RICH, m)
{
    m.doc() = "RICH";
    m.def("test", &testFunc);
}