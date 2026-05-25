#include "l3ster/l3ster.hpp"
#include <pugixml.hpp>

using namespace lstr;
using std::numbers::pi;
namespace fs = std::filesystem;
constexpr double infty = std::numeric_limits<double>::infinity();

constexpr dim_t dim = 3;
constexpr el_o_t mesh_order  = 4;


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



int main(int argc, char* argv[])
{    

    const auto scope_guard = L3sterScopeGuard{argc, argv};
    const auto comm        = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    std::vector<std::string> args;
    for (size_t i=0;i<argc;i++) {
        args.push_back(argv[i]);
        std::print("{}: {}\n",i,args[i]);
    }

    if (args.size() < 2) {
        std::print("Usage: {} config.xml\n", args[0]);
        return -1;
    }
    std::string config_filename = args[1];
    auto config_path = fs::path(config_filename);
    std::string output_path = "results/";
    std::string output_name = config_path.stem();

    #define TRE(...) throw std::runtime_error(std::format(__VA_ARGS__))
    #define ATRE(x__, ...) if (!(x__)) TRE(__VA_ARGS__)
    #define XATRE(x__, ...) ATRE(x__, "Error in {}: {}",config_filename, std::format(__VA_ARGS__))

    pugi::xml_document doc;
    pugi::xml_attribute attr;
    pugi::xml_parse_result result = doc.load_file(config_filename.c_str());
    XATRE(result, "{}", result.description());
    pugi::xml_node config = doc.child("LIF");
    XATRE(config, "LIF element doesn't exist");
    attr = config.attribute("dim");
    XATRE(attr, "No dim attribute in {}\n", config.name());
    XATRE(dim == attr.as_int(), "Dimension mismatch {} != {}\n", dim, attr.value());
    attr = config.attribute("output");
    if (attr) output_path = attr.value();

    double mesh_Lx, mesh_Ly, mesh_Lz = 0;
    size_t mesh_elx = 1, mesh_ely = 1, mesh_elz = 1;
    pugi::xml_node geom = config.child("Geometry");
    XATRE(config, "Geometry element doesn't exist");
    attr = geom.attribute("order");
    if (attr) {
        XATRE(mesh_order != attr.as_int(),"Mesh order mismatch {} != {}", mesh_order, attr.value());
    }
    attr = geom.attribute("Lx");
    XATRE(attr, "No Lx attribute in {}\n", geom.name());
    mesh_Ly = mesh_Lx = attr.as_double();
    XATRE(mesh_Lx > 0.0, "Assertion failed: Lx > 0\n");
    attr = geom.attribute("Ly");
    if (attr) mesh_Ly = attr.as_double();
    XATRE(mesh_Ly > 0.0, "Assertion failed: Ly > 0\n");
    attr = geom.attribute("Lz");
    if (attr) mesh_Lz = attr.as_double();
    if (dim > 2) XATRE(mesh_Lz > 0.0, "Assertion failed: Lz > 0\n");
    attr = geom.attribute("elx");
    if (attr) mesh_elx = attr.as_int();
    XATRE(mesh_elx > 0, "Assertion failed: elx > 0\n");
    attr = geom.attribute("ely");
    if (attr) mesh_ely = attr.as_int();
    XATRE(mesh_ely > 0, "Assertion failed: ely > 0\n");
    attr = geom.attribute("elz");
    if (attr) mesh_elz = attr.as_int();
    if (dim > 2) XATRE(mesh_elz > 0, "Assertion failed: elz > 0\n");

    struct cut_gauss_t {
        double scale=1;
        double x=0,y=0,z=0;
        double x_sigma=infty,y_sigma=infty,z_sigma=infty;
        double xmin=-infty,ymin=-infty,zmin=-infty;
        double xmax=infty,ymax=infty,zmax=infty;
        inline static double sq(double a, double b) {
            double ret = a/b;
            return ret*ret;
        }
        inline double operator() (double x0,double y0, double z0) const  {
            if (x0 < xmin || x0 > xmax) return 0;
            if (y0 < ymin || y0 > ymax) return 0;
            if (z0 < zmin || z0 > zmax) return 0;
            double ret = sq(x0-x,x_sigma) + sq(y0-y,y_sigma) + sq(z0-z,z_sigma);
            return scale * exp(-ret);
        }
    };
    struct field_def_t : public std::vector<cut_gauss_t> {
        inline double operator() (double x0,double y0, double z0) const {
            double ret = 0;
            for (const auto& fun : *this) {
                ret += fun(x0,y0,z0);
            }
            return ret;
        }
    };
    const auto& from_xml = [](pugi::xml_node node) {
        field_def_t ret;
        for (pugi::xml_node child = node.first_child(); child; child = child.next_sibling()) {
            std::string name = child.name();
            if (name == "Gauss" || name == "Box") {
                cut_gauss_t fun;
                pugi::xml_attribute attr;
                #define load_attr(member) attr = child.attribute(#member); if (attr) fun.member = attr.as_double()
                load_attr(scale);
                load_attr(x);
                load_attr(y);
                load_attr(z);
                load_attr(x_sigma);
                load_attr(y_sigma);
                load_attr(z_sigma);
                load_attr(xmin);
                load_attr(ymin);
                load_attr(zmin);
                load_attr(xmax);
                load_attr(ymax);
                load_attr(zmax);
                #undef load_attr
                ret.push_back(fun);
            } else {
                TRE("Unknown function {} in {}", name, node.name());
            }
        }
        return ret;
    };

    pugi::xml_node rays = config.child("Rays");
    XATRE(rays, "No Rays element in {}", config.name());
    struct sphere_point_t {
        double vx, vy, vz = 0.0;
        double weight;
    };
    std::vector<sphere_point_t> sphere_points;
    if (dim == 2) {
        attr = rays.attribute("n");
        if (attr) {
            size_t n = attr.as_int();
            XATRE(n>2, "Too small {} in {}", attr.name(), rays.name());
            double w = 1.0/n;
            for (size_t i=0; i<n;i++) {
                double an = 2*pi*i/n;
                sphere_point_t sp = {
                    .vx = cos(an),
                    .vy = sin(an),
                    .weight = w
                };
                sphere_points.push_back(sp);
            }
        }
    }
    attr = rays.attribute("set");
    if (attr) {
        std::string data_txt = attr.value();
        FILE* f = fopen(data_txt.c_str(), "r");
        ATRE(f != NULL, "Could not open file {}", data_txt);
        while (! feof(f)) {
            double a1,a2,w;
            int ret = fscanf(f, "%lf %lf %lf", &a1, &a2, &w);
            if (ret <= 0) break;
            ATRE(ret == 3, "Malformed file {} at {} (fscanf returned {})", data_txt, sphere_points.size(), ret);
            a1 = pi*a1/180;
            a2 = pi*a2/180;
            sphere_point_t sp = {
                .vx = sin(a2)*sin(a1),
                .vy = sin(a2)*cos(a1),
                .vz = cos(a2),
                .weight = w
            };
            sphere_points.push_back(sp);
        }
        fclose(f);
    }
    XATRE(sphere_points.size() > 0, "No rays defined in {}",rays.name());

    pugi::xml_node waves = config.child("Waves");
    XATRE(waves, "No Waves element in {}", config.name());
    struct spec_elem_t {
        double absorbtion=0;
        double source=0;
        double emission=0;
    };
    std::vector<spec_elem_t> spectrum;
    for (pugi::xml_node wave : waves.children()) {
        XATRE(std::string(wave.name()) == "Wave", "Uknown element {} in {}", wave.name(), waves.name());
        spec_elem_t spec_elem;
        attr = wave.attribute("absorbtion");
        if (attr) spec_elem.absorbtion = attr.as_double();
        attr = wave.attribute("emission");
        if (attr) spec_elem.emission = attr.as_double();
        attr = wave.attribute("source");
        if (attr) spec_elem.source = attr.as_double();
        spectrum.push_back(spec_elem);
    }
    XATRE(spectrum.size() > 0, "No rays defined in {}",rays.name());


    field_def_t concentration_def, volume_source_def, surf_source_def, empty_def;
    std::map<size_t, field_def_t> surf_source_defs;

    pugi::xml_node fun_node;
    fun_node = config.child("Concentration");
    XATRE(fun_node, "No Concentration element in {}", config.name());
    concentration_def = from_xml(fun_node);
    fun_node = config.child("VolumeSource");
    if (fun_node) volume_source_def = from_xml(fun_node);
    for (pugi::xml_node fun_node : config.children("SurfaceSource")) {
        attr = fun_node.attribute("direction");
        XATRE(attr, "{} needs a direction attribute", fun_node.name());
        size_t dir = attr.as_int();
        surf_source_defs[dir] = from_xml(fun_node);
    }

    constexpr int domain_id = 0;
    constexpr auto boundary_ids = []{
        if constexpr (dim == 2) {
            return std::array{1,2,3,4};
        } else {
            return std::array{1,2,3,4,5,6};
        }
    }();
    const auto mesh = makeMesh(*comm, mesh_Lx, mesh_elx, mesh_Ly, mesh_ely, mesh_Lz, mesh_elz);

    const auto problem_def = ProblemDefinition< 1 >{{domain_id}}; //1 - number of fields

    constexpr auto algsys_opts = AlgebraicSystemParams{.eval_strategy = OperatorEvaluationStrategy::MatrixFree};
    auto algebraic_system = makeAlgebraicSystem(comm, mesh, problem_def, {}, L3STER_WRAP_CTVAL(algsys_opts));
    algebraic_system.describe();

    double vx = 0;
    double vy = 0;
    double vz = 0;
    double kappa = 0;
    double lambda = 0;
    double gamma = 0;
    double theta = 0;

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

    const auto init_fun = [&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto init = field_vals[0];
        out[0] = init;
    };
    constexpr auto init_kernel_params = KernelParams{.dimension = dim, .n_equations = 1, .n_fields = 1};
    const auto init_kernel = wrapDomainResidualKernel< init_kernel_params >(init_fun);

    size_t sol_total = 0;
    std::vector<size_t> field_idx;
    for (size_t j=0;j<spectrum.size();j++) {
        for (size_t i=0;i<sphere_points.size();i++) {
            field_idx.push_back(sol_total); sol_total++;
        }
    }
    size_t concentration_idx = sol_total; sol_total++;
    size_t source_idx = sol_total; sol_total++;
    size_t absorption_idx = sol_total; sol_total++;
    std::vector<size_t> sum_idx;
    for (size_t j=0;j<spectrum.size();j++) {
        sum_idx.push_back(sol_total); sol_total++;
    }
    size_t abs_idx = sol_total; sol_total++;
    size_t buf_idx = sol_total; sol_total++;

    auto solution_manager = SolutionManager{*mesh, sol_total};

    // Analytical solution - used for setting the initial values and computing the error
    const auto analytical_functions = [&](const auto& in, auto& out) {
        const auto x    = in.point.space.x();
        const auto y    = in.point.space.y();
        const auto z    = in.point.space.z();
        out[0] = concentration_def(x,y,z);
        out[1] = volume_source_def(x,y,z);
    };
    constexpr auto functions_kernel_params = KernelParams{.dimension = dim, .n_equations = 2};
    const auto functions_kernel = wrapDomainResidualKernel< functions_kernel_params >(analytical_functions);

    solution_manager.setFields(*comm, *mesh, functions_kernel, {domain_id}, {concentration_idx, source_idx});

    double add_mul_kernel_val = 1;
    const auto add_kernel = wrapDomainResidualKernel< KernelParams{.dimension = dim, .n_equations = 1, .n_fields = 2} >([&add_mul_kernel_val](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto old = field_vals[0];
        const auto val = field_vals[1];
        out[0] = old + val*add_mul_kernel_val;
    });

    
    const auto add_mul_kernel = wrapDomainResidualKernel< KernelParams{.dimension = dim, .n_equations = 1, .n_fields = 3} >([&add_mul_kernel_val](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto old = field_vals[0];
        const auto val = field_vals[1];
        const auto mul = field_vals[2];
        out[0] = old + val*mul*add_mul_kernel_val;
    });


    // L3STER interface to KLU2 direct solver
//    auto solver = Klu2{};

    constexpr auto precond_opts = NativeJacobiOpts{};
    constexpr auto solver_opts = IterSolverOpts{
        .tol = 1e-6,
        .max_iters = 10000,
        .throw_on_fail = false,
        .verbosity = {.summary = false, .iter_details = false, .timing = false},
        .print_freq = 1000
    };
    auto solver = CG{solver_opts, precond_opts};

    auto exporter = PvtuExporter{comm, *mesh};

    for (size_t iter=0; iter<20; iter++) {
        
        solution_manager.setFields({abs_idx},0.0);
        size_t max_cg_iter = 0;
        for (size_t j=0;j<spectrum.size();j++) {
            const auto& wv = spectrum[j];
            size_t& s_idx = sum_idx[j];
            solution_manager.setFields({s_idx},0.0);
            for (size_t i=0;i<sphere_points.size();i++) {
                const auto &sp = sphere_points[i];
                size_t& f_idx = field_idx[i + j*sphere_points.size()];
                //an = 2*pi*i/sphere_points;
                vx = sp.vx;
                vy = sp.vy;
                vz = sp.vz;
                kappa = wv.absorbtion;
                lambda = wv.emission;
                gamma = wv.source;
                {   // surf_source_def = surf_source_defs[i];
                    auto it = surf_source_defs.find(i);
                    if (it == surf_source_defs.end()) {
                        surf_source_def = empty_def;
                    } else {
                        surf_source_def = it->second;
                    }
                }

                printf("Iteration %3d: solving for %3d wavelength, %3d direction (%.3lf,%.3lf,%.3lf) \n", (int) iter, (int) j, (int) i, vx,vy,vz);

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
                if (res.num_iters > max_cg_iter) max_cg_iter = res.num_iters;
                algebraic_system.updateSolution({0}, solution_manager, {f_idx});
                {
                    const auto f_access = solution_manager.getFieldAccess(std::array{s_idx, f_idx});
                    add_mul_kernel_val = sp.weight;
                    solution_manager.setFields(*comm, *mesh, add_kernel, {domain_id}, {buf_idx}, f_access);
                    std::swap(s_idx,buf_idx);
                }
                {
                    const auto f_access = solution_manager.getFieldAccess(std::array{abs_idx, f_idx, concentration_idx});
                    add_mul_kernel_val = kappa * sp.weight;
                    solution_manager.setFields(*comm, *mesh, add_mul_kernel, {domain_id}, {buf_idx}, f_access);
                    std::swap(abs_idx,buf_idx);
                }
            }
        }
        std::swap(abs_idx,absorption_idx);
        
        auto export_def = ExportDefinition{std::format("{}{}_{:04}.pvtu", output_path, output_name, iter)};
        export_def.defineField("concentration", {concentration_idx});
        export_def.defineField("source", {source_idx});
        for (size_t j=0;j<spectrum.size();j++) {
            export_def.defineField(std::format("sum_{}",j), {sum_idx[j]});
        }
        export_def.defineField("absorbed", {absorption_idx});
        int sp_export = sphere_points.size();
        if (sp_export > 6) sp_export = 6;
        for (size_t j=0;j<spectrum.size();j++) {
            for (size_t i=0;i<sp_export;i++) {
                size_t& f_idx = field_idx[i + j*sphere_points.size()];
                export_def.defineField(std::format("f_{}_{:02}", j, i), {f_idx});
            }
        }
        exporter.exportSolution(export_def, solution_manager);  // Write to file
        if (max_cg_iter == 0) break;
    }        
}