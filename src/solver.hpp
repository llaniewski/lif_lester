#include "solver.h"
#include "l3ster/l3ster.hpp"
#include <pugixml.hpp>
#include "Factory.h"

using namespace lstr;
using std::numbers::pi;
namespace fs = std::filesystem;
constexpr double infty = std::numeric_limits<double>::infinity();


template <dim_t dim, el_o_t mesh_order>
class Solver : public SolverBase {
    SolutionManager solution_manager;
    mesh::MeshPartition< mesh_order > mesh;
public:
    virtual void initField( size_t, space_function);
    virtual void initField( size_t, double);
    virtual void solve(size_t, double, double, double, double, double, double, space_function);
    virtual void addMult(size_t,size_t,double);
    virtual void addMult(size_t,size_t,double,size_t);
    virtual void exportVTK(const lstr::ExportDefinition&);
};

typedef std::pair<dim_t, el_o_t> FactoryInput;
typedef Factory< SolverBase, FactoryInput > SolverFactory;

template <dim_t dim, el_o_t mesh_order>
SolverBase * ask_for_solver(const FactoryInput& in) {
    if (in.first == dim && in.second == mesh_order) return Solver<dim, mesh_order>{};
    return NULL;
}


template <dim_t dim, el_o_t mesh_order>
auto makeMesh(const MpiComm& comm, double Lx, size_t elx, double Ly, size_t ely, double Lz = 0.0, size_t elz = 0)
{
    const auto mesh_generator = [&] {
        assert(elx > 0);
        assert(ely > 0);
        const auto       node_dist_x = util::linspace(0., Lx, elx+1); // linspace semantics same as numpy
        const auto       node_dist_y = util::linspace(0., Ly, ely+1); // linspace semantics same as numpy
        if constexpr (dim == 2) {
            return mesh::makeSquareMesh(node_dist_x, node_dist_y);
        } else {
            assert(elz > 0);
            const auto       node_dist_z = util::linspace(0., Lz, elz+1); // linspace semantics same as numpy
            return mesh::makeCubeMesh(node_dist_x, node_dist_y, node_dist_z);
        }
    };
    return generateAndDistributeMesh(comm, mesh_generator, L3STER_WRAP_CTVAL(mesh_order));
}

template <dim_t dim, el_o_t mesh_order>
void Solver<dim, mesh_order>::initField( size_t idx, double v) {
    solution_manager.setFields({idx},v);
}
