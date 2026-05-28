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
    std::shared_ptr< lstr::MpiComm > comm;
    std::shared_ptr< mesh::MeshPartition< mesh_order > > mesh;
    SolutionManager solution_manager;
public:
    int domain_id;
    std::vector<int> boundary_ids;
    Solver(std::shared_ptr< lstr::MpiComm > comm_, size_t nfields, double Lx, size_t elx, double Ly, size_t ely, double Lz, size_t elz);
    virtual void initField( size_t, space_function);
    virtual void initField( size_t, double);
    virtual int solve(size_t, size_t, size_t, size_t, double, double, double, double, double, double, space_function);
    virtual void addMult(size_t,size_t,double);
    virtual void addMult(size_t,size_t,double,size_t);
    virtual void exportVTK(const lstr::ExportDefinition&);
};



template <dim_t dim, el_o_t mesh_order>
SolverBase * ask_for_solver(const solver_input& in) {
    printf("Ask %d %d\n", (int)dim, (int) mesh_order); fflush(stdout);
    if (in.dim == dim && in.mesh_order == mesh_order) {
        printf("Create Solver %p ...\n", (void*) in.comm.get()); fflush(stdout);
        return new
            Solver<dim, mesh_order>{
                in.comm,
                in.nfields,
                in.Lx, in.elx,
                in.Ly, in.ely,
                in.Lz, in.elz
            }
        ;
    }
    return NULL;
}


template <dim_t dim, el_o_t mesh_order>
auto makeMesh(const MpiComm& comm, double Lx, size_t elx, double Ly, size_t ely, double Lz = 0.0, size_t elz = 0)
{
    printf("makeMesh1\n"); fflush(stdout);
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
    printf("makeMesh2\n"); fflush(stdout);
    return generateAndDistributeMesh(comm, mesh_generator, L3STER_WRAP_CTVAL(mesh_order));
}

template <dim_t dim, el_o_t mesh_order>
Solver<dim, mesh_order>::Solver(
    std::shared_ptr< lstr::MpiComm > comm_, size_t nfields,
    double Lx, size_t elx, double Ly, size_t ely, double Lz, size_t elz)
    :   comm(comm_),
        mesh(makeMesh<dim,mesh_order>(*comm, Lx, elx, Ly, ely, Lz, elz)),
        solution_manager{*mesh, nfields} {
    printf("Solver constructor\n"); fflush(stdout);
    domain_id = 0;
    if constexpr (dim == 2) {
        boundary_ids = std::vector{1,2,3,4};
    } else {
        boundary_ids = std::vector{1,2,3,4,5,6};
    }
}

template <dim_t dim, el_o_t mesh_order>
void Solver<dim, mesh_order>::initField( size_t idx, double v) {
    solution_manager.setFields({idx},v);
}


template <dim_t dim, el_o_t mesh_order>
void Solver<dim, mesh_order>::initField( size_t idx, space_function fun) {
    const auto init_fun = wrapDomainResidualKernel< KernelParams{.dimension = dim, .n_equations = 1} >(
        [&](const auto& in, auto& out) {
            const auto x    = in.point.space.x();
            const auto y    = in.point.space.y();
            const auto z    = in.point.space.z();
            out[0] = fun(x,y,z);
        }
    );
    solution_manager.setFields(*comm, *mesh, init_fun, {domain_id}, {idx});

}

template <dim_t dim, el_o_t mesh_order>
int Solver<dim, mesh_order>::solve(
    size_t f_idx, size_t concentration_idx, size_t source_idx, size_t absorption_idx, double vx, double vy, double vz, double kappa, double lambda, double gamma, space_function surf_source_def) {

    constexpr auto algsys_opts = AlgebraicSystemParams{.eval_strategy = OperatorEvaluationStrategy::MatrixFree};

    const auto problem_def = ProblemDefinition< 1 >{{domain_id}}; //1 - number of fields
    auto algebraic_system = makeAlgebraicSystem(comm, mesh, problem_def, {}, L3STER_WRAP_CTVAL(algsys_opts));
    algebraic_system.describe();

    constexpr auto kernel_params = KernelParams{.dimension = dim, .n_equations = 1, .n_unknowns = 1, .n_fields = 3};
    const auto kernel = wrapDomainEquationKernel< kernel_params >([&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto concentration = field_vals[0];
        const auto source = field_vals[1];
        const auto absorbed = field_vals[2];

        auto& [operators, rhs] = out;
        operators[0](0, 0) = kappa*concentration;
        operators[1](0, 0) = vx;
        operators[2](0, 0) = vy;
        if constexpr (dim > 2) operators[3](0, 0) = vz;
        rhs(0, 0) = gamma*source + lambda*absorbed;
    });

    constexpr auto boundary_kernel_params = KernelParams{.dimension = dim, .n_equations = 1, .n_unknowns = 1};
    const auto bc_kernel = wrapBoundaryEquationKernel< boundary_kernel_params >([&](const auto& in, auto& out) {
        // Unit boundary normal (obviously this is only available in boundary kernels)
        const auto  nx     = in.normal[0];
        const auto  ny     = in.normal[1];
        const auto  nz     = in.normal[2];
        const auto x    = in.point.space.x();
        const auto y    = in.point.space.y();
        const auto z    = in.point.space.z();
        
        auto& [operators, rhs] = out;

        if (nx*vx + ny*vy + nz*vz < 0) {
            operators[0](0, 0) = 1;
            rhs[0] = gamma * surf_source_def(x,y,z);
        }
    });

    constexpr auto init_kernel_params = KernelParams{.dimension = dim, .n_equations = 1, .n_fields = 1};
    const auto init_kernel = wrapDomainResidualKernel< init_kernel_params >([&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto init = field_vals[0];
        out[0] = init;
    });

    constexpr auto precond_opts = NativeJacobiOpts{};
    constexpr auto solver_opts = IterSolverOpts{
        .tol = 1e-6,
        .max_iters = 10000,
        .throw_on_fail = false,
        .verbosity = {.summary = false, .iter_details = false, .timing = false},
        .print_freq = 1000
    };
    auto solver = CG{solver_opts, precond_opts};

    const auto field_access = solution_manager.getFieldAccess(std::array{concentration_idx, source_idx, absorption_idx});
    // Zero out the system
    algebraic_system.beginAssembly();
    algebraic_system.assembleProblem(kernel, {domain_id}, field_access);
    algebraic_system.assembleProblem(bc_kernel, boundary_ids);
    algebraic_system.endAssembly();

    algebraic_system.setValues(
        algebraic_system.getSolution(),
        init_kernel,
        {domain_id},
        util::makeIotaArray<size_t,1>(),
        solution_manager.getFieldAccess(std::array{f_idx})
    );
    IterSolveResult res = algebraic_system.solve(solver);
    algebraic_system.updateSolution({0}, solution_manager, {f_idx});

    return res.num_iters;
}

template <dim_t dim, el_o_t mesh_order>
void Solver<dim, mesh_order>::addMult(size_t f_idx, size_t g_idx, double mult) {
    auto f = solution_manager.getFieldView(f_idx);
    const auto g = solution_manager.getFieldView(g_idx);
    for (size_t i=0;i<f.size(); i++) f[i] += g[i] * mult;
}

template <dim_t dim, el_o_t mesh_order>
void Solver<dim, mesh_order>::addMult(size_t f_idx, size_t g_idx, double mult, size_t h_idx) {
    auto f = solution_manager.getFieldView(f_idx);
    const auto g = solution_manager.getFieldView(g_idx);
    const auto h = solution_manager.getFieldView(h_idx);
    for (size_t i=0;i<f.size(); i++) f[i] += g[i] * mult * h[i];
}

template <dim_t dim, el_o_t mesh_order>
void Solver<dim, mesh_order>::exportVTK(const lstr::ExportDefinition& export_def){
    auto exporter = PvtuExporter{comm, *mesh};
    exporter.exportSolution(export_def, solution_manager);
}