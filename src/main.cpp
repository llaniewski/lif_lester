#include "l3ster/l3ster.hpp"
#include <pugixml.hpp>
#include "solver.h"

using namespace lstr;
using std::numbers::pi;
namespace fs = std::filesystem;
constexpr double infty = std::numeric_limits<double>::infinity();





int main(int argc, char* argv[])
{    
    const dim_t dim = 3;
    const el_o_t mesh_order  = 3;

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
            for (const auto& fun : *this) ret += fun(x0,y0,z0);
            return ret;
        }
    };
    const auto& from_xml = [](pugi::xml_node node) {
        field_def_t ret;
        for (pugi::xml_node child : node.children()) {
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
        double absorption=0;
        double source=0;
        double emission=0;
    };
    std::vector<spec_elem_t> spectrum;
    for (pugi::xml_node wave : waves.children()) {
        XATRE(std::string(wave.name()) == "Wave", "Uknown element {} in {}", wave.name(), waves.name());
        spec_elem_t spec_elem;
        attr = wave.attribute("absorption");
        if (attr) spec_elem.absorption = attr.as_double();
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

    printf("making solver\n"); fflush(stdout);
    SolverBase* solver = makeSolver(solver_input{dim, mesh_order, comm, sol_total, mesh_Lx, mesh_elx, mesh_Ly, mesh_ely, mesh_Lz, mesh_elz});
    printf("solver: %p\n", solver); fflush(stdout);
    ATRE(solver != NULL, "Failed to produce solver. Probably unsupported combination of dimension {} and element order {}", dim, mesh_order);
    
    printf("init field...\n"); fflush(stdout);
    solver->initField( concentration_idx, concentration_def);
    printf("init field...\n"); fflush(stdout);
    solver->initField( source_idx, volume_source_def);
    printf("init field...\n"); fflush(stdout);

    for (size_t iter=0; iter<20; iter++) {
        solver->initField( abs_idx, 0.0);

        size_t max_cg_iter = 0;
        for (size_t j=0;j<spectrum.size();j++) {
            const auto& wv = spectrum[j];
            size_t& s_idx = sum_idx[j];
            solver->initField( s_idx, 0.0);
            for (size_t i=0;i<sphere_points.size();i++) {
                const auto &sp = sphere_points[i];
                size_t& f_idx = field_idx[i + j*sphere_points.size()];
                //an = 2*pi*i/sphere_points;
                double vx = sp.vx;
                double vy = sp.vy;
                double vz = sp.vz;
                double kappa = wv.absorption;
                double lambda = wv.emission;
                double gamma = wv.source;
                {   // surf_source_def = surf_source_defs[i];
                    auto it = surf_source_defs.find(i);
                    if (it == surf_source_defs.end()) {
                        surf_source_def = empty_def;
                    } else {
                        surf_source_def = it->second;
                    }
                }

                printf("Iteration %3d: solving for %3d wavelength, %3d direction (%.3lf,%.3lf,%.3lf) \n", (int) iter, (int) j, (int) i, vx,vy,vz);
                int iter = solver->solve(f_idx, concentration_idx, source_idx, absorption_idx, vx, vy, vz, kappa, lambda, gamma, surf_source_def);
                if (iter > max_cg_iter) max_cg_iter = iter;
                solver->addMult(s_idx,f_idx,sp.weight);
                solver->addMult(abs_idx,f_idx,sp.weight,concentration_idx);
            }
        }

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
        solver->exportVTK(export_def);
        if (max_cg_iter == 0) break;
    }        
}
