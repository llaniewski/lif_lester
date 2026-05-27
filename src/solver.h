//#include "l3ster/l3ster.hpp"
#include <pugixml.hpp>

// using namespace lstr;
// using std::numbers::pi;
// namespace fs = std::filesystem;
// constexpr double infty = std::numeric_limits<double>::infinity();


class SolverBase {
public:
    typedef std::function< double (double, double, double)> space_function;
    virtual void initField( size_t, space_function);
    virtual void initField( size_t, double);
    virtual void solve(size_t, double, double, double, double, double, double, space_function);
    virtual void addMult(size_t,size_t,double);
    virtual void addMult(size_t,size_t,double,size_t);
    virtual void exportVTK(const lstr::ExportDefinition&);
};

SolverBase* makeSolver(
    lstr::dim_t dim,
    lstr::el_o_t mesh_order,
    const lstr::MpiComm& comm,
    size_t nfields,
    double Lx, size_t elx,
    double Ly, size_t ely,
    double Lz = 0.0, size_t elz = 0
);
