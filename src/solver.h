#include "l3ster/l3ster.hpp"
#include <pugixml.hpp>
#include "Factory.h"

struct solver_input {
    lstr::dim_t dim;
    lstr::el_o_t mesh_order;
    const std::shared_ptr< lstr::MpiComm > comm;
    size_t nfields;
    double Lx;
    size_t elx;
    double Ly;
    size_t ely;
    double Lz;
    size_t elz;
};

class SolverBase {
public:
    typedef std::function< double (double, double, double)> space_function;
    virtual void initField( size_t, space_function) = 0;
    virtual void initField( size_t, double) = 0;
    virtual int solve(size_t, size_t, size_t, size_t, double, double, double, double, double, double, space_function) = 0;
    virtual void addMult(size_t,size_t,double) = 0;
    virtual void addMult(size_t,size_t,double,size_t) = 0;
    virtual void exportVTK(const lstr::ExportDefinition&) = 0;
};

typedef Factory< SolverBase, solver_input > SolverFactory;

inline SolverBase* makeSolver(const solver_input& in) {
    return SolverFactory::Produce(in);
};
