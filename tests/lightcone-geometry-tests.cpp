// Writer/reader shell consistency: replay both legacy gevolution and gasgrd-v1 writers
// (record and slot gates, per-process periodic copy lists, half-open shells, last write wins) must
// assign every node to exactly the record that lightcone-raytracer assigns to it.
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <random>

using namespace lightcone;
namespace {

struct Case {
    int n, thickness;
    std::array<double, 3> vertex;
    bool cone, staggered, snap;
    GeometryConvention convention;
};

// Independent legacy writer expression, with separate roundings fixed by volatile stores.
// The gasgrd oracle uses the same vendored function as its actual writer.
double writer_node_radius(const std::int64_t xyz[3], const double offset[3], int n,
                          const Geometry& g, bool& cone) {
    if (g.convention == GeometryConvention::GasgrdV1)
        return lcg::node_radius(xyz,offset,n,g.vertex,g.direction,g.angular_limit ? g.opening_cos : -1.,cone);
    double pos[3], vertex[3];
    for (int a=0;a<3;++a) {
        const auto copy=static_cast<std::int64_t>(std::floor(static_cast<double>(xyz[a])/n));
        pos[a]=(static_cast<double>(xyz[a]-copy*n)+offset[a])/n;
        vertex[a]=g.vertex[a]-static_cast<double>(copy);
    }
    const double dx=pos[0]-vertex[0], dy=pos[1]-vertex[1], dz=pos[2]-vertex[2];
    volatile double xx=dx*dx, yy=dy*dy, zz=dz*dz;
    volatile double xy=xx+yy;
    const double r=std::sqrt(xy+zz);
    volatile double px=dx*g.direction[0], py=dy*g.direction[1], pz=dz*g.direction[2];
    volatile double pxy=px+py;
    cone=!g.angular_limit || ((pxy+pz)/r > g.opening_cos);
    return r;
}

// shell radii as computed by lccubeFill for a sequence of time steps (box units)
// Boundaries can be snapped onto node radii (all nodes and copies u in {-1,0,1}^3, given offset), so
// that every record has nodes exactly on its shell edges: any 1-ulp disagreement shows up.
std::vector<InfoRecord> writer_records(int n, int thickness, double d, double distance_outer, double distance_inner,
                                       bool snap, const Geometry& g, const double offset[3], std::mt19937_64& rng) {
    std::vector<InfoRecord> records;
    double tau = 0.0, dtau_old = 0.0, dtau_older = 0.0;
    std::uniform_real_distribution<double> wobble(0.85, 1.15);
    std::vector<double> radii;
    if (snap) {
        for (int u = -1; u <= 1; ++u) for (int w = -1; w <= 1; ++w) for (int t = -1; t <= 1; ++t)
            for (int z = 0; z < n; ++z) for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
                const std::int64_t xyz[3] = {x-u*n, y-w*n, z-t*n};
                bool cone;
                radii.push_back(writer_node_radius(xyz, offset, n, g, cone));
            }
        std::sort(radii.begin(), radii.end());
    }
    auto snap_to_node = [&](double target, double window) {
        const auto it = std::lower_bound(radii.begin(), radii.end(), target);
        double best = target;
        if (it != radii.end() && *it-target < window) best = *it;
        if (it != radii.begin() && target-*(it-1) < window && target-*(it-1) < std::fabs(best-target)) best = *(it-1);
        return best;
    };
    for (int cycle = 0; tau < d+0.2; ++cycle) {
        const double dtau = 0.7/n*wobble(rng);
        std::vector<double> s(thickness+1);
        if (thickness == 1) { s[0] = d-tau-0.5*dtau; s[1] = d-tau+0.5*dtau_old; }
        else if (thickness == 2) { s[0] = d-tau-dtau; s[1] = d-tau; s[2] = d-tau+dtau_old; }
        else { s[0] = d-tau-2.*dtau; s[1] = d-tau-0.5*dtau; s[2] = d-tau+0.5*dtau_old; s[3] = d-tau+dtau_old+0.5*dtau_older; }
        if (snap && cycle > 2)
            for (int j = 0; j <= thickness; j += 2) s[j] = snap_to_node(s[j], 0.2*dtau); // keeps the order
        bool increasing = true;
        for (int j = 1; j <= thickness; ++j) increasing = increasing && s[j] > s[j-1];
        if (cycle > 0 && increasing && lcg::record_written(s.data(), thickness, distance_outer, distance_inner))
            records.push_back(InfoRecord{cycle, tau, 0.5+tau, d-tau-0.5*dtau, d-tau+0.5*dtau+1e-9, s});
        dtau_older = dtau_old; dtau_old = dtau; tau += dtau;
    }
    return records;
}

void consistency_case(const Case& c, std::mt19937_64& rng) {
    const int n = c.n, T = c.thickness;
    Geometry g; g.convention=c.convention;
    for (int a = 0; a < 3; ++a) g.vertex[a] = c.vertex[a];
    if (c.cone) {
        const double dir[3] = {0.3, -0.5, 0.8};
        const double norm = std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
        for (int a = 0; a < 3; ++a) g.direction[a] = dir[a]/norm;
        g.opening_cos = std::cos(50.0*3.14159265358979323846/180.0);
        g.angular_limit = true;
    }
    g.radial_limit = true; g.inner = 0.0; g.outer = 0.5;
    GridView grid;
    if (c.staggered) grid.node_offset[0] = 0.5;
    const auto records = writer_records(n, T, 0.62, g.outer, g.inner, c.snap, g, grid.node_offset, rng);
    check(records.size() > 3, "Enough records");

    // writer replay: four processes (2 x 2 in y and z), each with its own copy list per shell
    const std::size_t cells = static_cast<std::size_t>(n)*n*n;
    std::vector<int> stored(T*cells, -1);
    const int half = n/2;
    for (int k = 0; k < static_cast<int>(records.size()); ++k) {
        const auto& s = records[k].boundaries;
        for (int j = 0; j < T; ++j) {
            if (!lcg::slot_written(s.data(), j, g.outer, g.inner)) continue;
            for (int py = 0; py < 2; ++py) for (int pz = 0; pz < 2; ++pz) {
                const int y0 = py ? half : 0, y1 = py ? n : half, z0 = pz ? half : 0, z1 = pz ? n : half;
                const double lo[3] = {-1.0/n, (y0-1.0)/n, (z0-1.0)/n};
                const double hi[3] = {(n+1.0)/n, (y1+1.0)/n, (z1+1.0)/n};
                std::int64_t copies[lcg::MAX_COPIES][3];
                const int ncopies = lcg::enumerate_copies(g.vertex, s[j+1], s[j], lo, hi, copies, lcg::MAX_COPIES);
                check(ncopies <= lcg::MAX_COPIES, "Copy list capacity");
                for (int z = z0; z < z1; ++z) for (int y = y0; y < y1; ++y) for (int x = 0; x < n; ++x)
                    for (int u = 0; u < ncopies; ++u) {
                        const std::int64_t xyz[3] = {x-copies[u][0]*n, y-copies[u][1]*n, z-copies[u][2]*n};
                        bool cone;
                        const double r = writer_node_radius(xyz, grid.node_offset, n, g, cone);
                        if (r >= 0.0 && cone && lcg::in_shell(r, s[j], s[j+1])) {
                            stored[j*cells+x+n*(y+static_cast<std::size_t>(n)*z)] = k;
                            break;
                        }
                    }
            }
        }
    }

    // reader: newest record over all periodic copies, exact mode (no guard band)
    const auto metadata = build_metadata(records, T, g);
    auto view = host_view(metadata, n, g, nullptr);
    double outer = 0.0, inner = 0.0;
    for (const auto& r : records) { outer = std::max(outer, r.boundaries.back()); inner = std::min(inner, r.boundaries.front()); }
    const double lo[3] = {-1.0/n, -1.0/n, -1.0/n}, hi[3] = {(n+1.0)/n, (n+1.0)/n, (n+1.0)/n};
    std::int64_t copies[lcg::MAX_COPIES][3];
    const int ncopies = lcg::enumerate_copies(g.vertex, outer, inner, lo, hi, copies, lcg::MAX_COPIES);
    long long written = 0, mismatches = 0, overlaps = 0;
    for (int j = 0; j < T; ++j)
        for (int z = 0; z < n; ++z) for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
            int best = -1;
            for (int u = 0; u < ncopies; ++u) {
                const std::int64_t xyz[3] = {x-copies[u][0]*n, y-copies[u][1]*n, z-copies[u][2]*n};
                bool cone;
                const double r = node_radius(view, grid, xyz, cone);
                if (r < 0.0 || !cone) continue;
                const int record = record_at(view, j, r);
                if (record < 0) continue;
                if (best >= 0 && record != best) ++overlaps;
                best = std::max(best, record);
            }
            const int value = stored[j*cells+x+n*(y+static_cast<std::size_t>(n)*z)];
            if (value >= 0) ++written;
            if (value != best) ++mismatches;
        }
    check(written > 0, "Writer replay fills cells");
    check(overlaps == 0, "No periodic overlap within half a box");
    check(mismatches == 0, "Writer and reader assign every node to the same record (n=" + std::to_string(n) +
          ", T=" + std::to_string(T) + ", vertex=" + std::to_string(c.vertex[0]) + ", mismatches=" + std::to_string(mismatches) + ")");
}

void consistency_tests() {
    std::mt19937_64 rng(20261106);
    const std::array<std::array<double, 3>, 4> vertices = {{{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.3, 0.7, 0.1}, {1.0/3.0, 2.0/3.0, 0.2}}};
    for (auto mode : {GeometryConvention::Legacy, GeometryConvention::GasgrdV1})
    for (int n : {16, 25}) for (int T : {1, 2, 3}) for (const auto& v : vertices)
        for (bool cone : {false, true}) for (bool staggered : {false, true})
            consistency_case({n, T, v, cone, staggered, true, mode}, rng);
}

void arithmetic_tests() {
    // the radius is independent of the decomposition into copy shift and node: (X - uN, v) only
    const double offset[3] = {0.5, 0.0, 0.0}, v[3] = {0.3, 0.7, 0.1}, dir[3] = {0, 0, 1};
    bool cone;
    const std::int64_t a[3] = {-7, 40, 3};
    const double r1 = lcg::node_radius(a, offset, 33.0, v, dir, -1.0, cone);
    double d[3];
    lcg::displacement(a, offset, 33.0, v, d);
    check(r1 == std::sqrt(std::fma(d[2], d[2], std::fma(d[1], d[1], d[0]*d[0]))), "Explicit fused radius");
    check(lcg::in_shell(1.0, 1.0, 2.0) && !lcg::in_shell(2.0, 1.0, 2.0), "Half-open shells");
    const double s[4] = {0.1, 0.2, 0.3, 0.4};
    check(lcg::slot_written(s, 0, 0.5, 0.0) && !lcg::slot_written(s, 2, 0.4, 0.0) && !lcg::slot_written(s, 0, 0.5, 0.25),
          "Writer slot gate");
}

void boundary_tests() {
    // The old 8-ULP guard missed this 228-ULP periodic-crossing difference.
    Geometry g; g.vertex[0]=0.; g.vertex[1]=g.vertex[2]=.5;
    GridView grid;
    const std::int64_t xyz[3]={-1,288,288};
    const double b=1.-575./576;
    std::vector<InfoRecord> records{{0,.1,.5,0.,1.,{0.,b}},{1,.2,.6,0.,1.,{b,1.}}};
    const auto metadata=build_metadata(records,1,g);
    auto view=host_view(metadata,576,g,&grid);
    bool cone;
    near(node_radius(view,grid,xyz,cone),b,0,"Legacy periodic radius reproduces writer");
    check(record_at(view,0,node_radius(view,grid,xyz,cone))==1,"Legacy exact boundary owned by outer record");
    check(record_at(view,0,std::nextafter(b,0.))==0,"Previous representable radius owned by inner record");
    check(record_at(view,0,std::nextafter(b,INFINITY))==1,"Next representable radius owned by outer record");
    view.geometry.convention=GeometryConvention::GasgrdV1;
    near(node_radius(view,grid,xyz,cone),1./576,0,"New convention preserves unwrapped division");
    check(record_at(view,0,node_radius(view,grid,xyz,cone))==0,"Conventions are deliberately distinct");
    // This boundary value must be usable, rather than discarded by a guard.
    float payload=123.f,value=0;
    grid.values=&payload;
    for(int a=0;a<3;++a) { grid.origin[a]=wrap(xyz[a],576); grid.extent[a]=1; grid.strides[a]=1; }
    view.geometry=g; view.time_derivatives=TimeDerivativeMode::Zero;
    check(node_at(view,0,xyz,.2,value)==Status::Ok && value==payload,"Boundary sample remains available");
    for(auto mode : {GeometryConvention::Legacy,GeometryConvention::GasgrdV1})
    for(int n : {16,25,576}) for(double v : {0.,.5,.1,1./3.}) for(int field=0;field<4;++field)
    for(int image : {-2,-1,0,1,2}) {
        Geometry geometry; geometry.convention=mode;
        geometry.vertex[0]=v; geometry.vertex[1]=.3; geometry.vertex[2]=.7;
        auto vv=host_view(metadata,n,geometry,&grid);
        const std::int64_t node[3]={image*n-1,2*n+3,-n+5};
        grid.node_offset[0]=grid.node_offset[1]=grid.node_offset[2]=0.;
        if(field) grid.node_offset[field-1]=.5;
        bool expected_cone;
        const double expected=writer_node_radius(node,grid.node_offset,n,geometry,expected_cone);
        near(node_radius(vv,grid,node,cone),expected,0,"Every staggered component and periodic image matches its writer");
        check(cone==expected_cone,"Cone agrees with writer");
    }
}

void geometry_file_tests() {
    TempDirectory temp;
    lcg::GeometryInfo g;
    g.n = 1024; g.thickness = 3; g.b_collocated = 1; g.downgrade = 2; g.fields = lcg::FIELD_PHI|lcg::FIELD_B|lcg::FIELD_WEYL;
    g.periodic_overlap = 0; g.lightcone = 0;
    g.vertex[0] = 1000.0/3500.0; g.vertex[1] = 0.1; g.vertex[2] = 1.0/3.0;
    g.direction[0] = 0.6; g.direction[1] = 0.0; g.direction[2] = 0.8;
    g.opening_cos = std::cos(0.3); g.distance_outer = 0.5; g.distance_inner = 1.0/7.0; g.tauobs = 9028.3/3500.0; g.boxsize = 3500.0;
    const auto path = (temp.path/"geometry.bin").string();
    check(lcg::write_geometry_binary(path, g) && lcg::write_geometry_text((temp.path/"geometry.txt").string(), g), "Write geometry");
    const auto r = read_geometry_info(path);
    bool same = r.n == g.n && r.thickness == g.thickness && r.b_collocated == g.b_collocated && r.downgrade == g.downgrade &&
                r.fields == g.fields && r.periodic_overlap == g.periodic_overlap && r.lightcone == g.lightcone &&
                r.opening_cos == g.opening_cos && r.distance_outer == g.distance_outer && r.distance_inner == g.distance_inner &&
                r.tauobs == g.tauobs && r.boxsize == g.boxsize;
    for (int a = 0; a < 3; ++a) same = same && r.vertex[a] == g.vertex[a] && r.direction[a] == g.direction[a];
    check(same, "Geometry file round trip is exact");
    rejects([&] { require_geometry_grid(r, 512); }, "Reject geometry/grid mismatch");
    require_geometry_grid(r, 1024);
    auto bad=g; bad.convention=2;
    lcg::write_geometry_binary(path,bad);
    rejects([&] { read_geometry_info(path); }, "Reject unsupported arithmetic convention");
    bad=g; bad.vertex[0]=NAN;
    lcg::write_geometry_binary(path,bad);
    rejects([&] { read_geometry_info(path); }, "Reject nonfinite geometry");
    bad=g; bad.b_collocated=2;
    lcg::write_geometry_binary(path,bad);
    rejects([&] { read_geometry_info(path); }, "Reject unknown B layout");
    lcg::write_geometry_binary(path,g);
    std::filesystem::resize_file(path, std::filesystem::file_size(path)-1);
    rejects([&] { read_geometry_info(path); }, "Reject truncated geometry file");
    rejects([&] { read_geometry_info((temp.path/"geometry.txt").string()); }, "Reject non-geometry file");
}
} // namespace

int main() {
    try {
        arithmetic_tests(); boundary_tests(); geometry_file_tests(); consistency_tests();
        std::cout << "Writer/reader shell geometry checks passed (" << checks << ").\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
