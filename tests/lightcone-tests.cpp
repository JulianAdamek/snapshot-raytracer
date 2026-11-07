#include "test_support.hpp"
#include <array>
#include <cstring>

using namespace lightcone;
namespace {
void metadata_tests() {
    TempDirectory temp;
    for (int t : {1, 2, 3, 5}) {
        const auto path = (temp.path/("info"+std::to_string(t))).string();
        const auto original = legacy_records(t);
        write_info(path, original);
        const auto decoded = read_info(path, t);
        check(std::filesystem::file_size(path) == original.size()*(4+8*(t+5)), "Packed record width");
        check(decoded.size() == original.size() && decoded[0].boundaries.size() == static_cast<std::size_t>(t+1), "Decode runtime thickness");
        near(decoded[3].tau, original[3].tau, 0, "Decode tau");
        near(decoded[3].boundaries[1], original[3].boundaries[1], 0, "Decode shell boundary");
        auto truncated = temp.path/("truncated"+std::to_string(t));
        std::filesystem::copy_file(path, truncated);
        std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated)-1);
        rejects([&] { read_info(truncated.string(), t); }, "Reject truncated records");
        rejects([&] { read_info(path, t+1); }, "Reject wrong thickness");
        Geometry g;
        auto m = build_metadata(decoded, t, g);
        auto v = host_view(m, 32, g, nullptr);
        // Compare every interval lookup to literal chronological writer replay.
        for (int slot = 0; slot < t; ++slot) for (int i = 0; i <= 400; ++i) {
            const double radius = i/200.0;
            int expected = -1;
            for (int k = 0; k < static_cast<int>(original.size()); ++k) {
                const auto& s = original[k].boundaries;
                if (s.back() > 0 && s[slot+1] > 0 && radius >= s[slot] && radius < s[slot+1]) expected = k;
            }
            check(record_at(v, slot, radius) == expected, "Latest half-open shell write owns slot");
        }
        for (double outer : {0.2, 0.7, 1.3}) {
            g.radial_limit = true; g.inner = 0.1; g.outer = outer;
            m = build_metadata(decoded, t, g); v = host_view(m, 32, g, nullptr);
            for (int slot = 0; slot < t; ++slot) for (int i = 0; i <= 200; ++i) {
                const double radius = i/200.0;
                int expected = -1;
                for (int k = 0; k < static_cast<int>(original.size()); ++k) {
                    const auto& s = original[k].boundaries;
                    if (g.outer > s[1] && g.inner <= s.back() && s.back() > 0 &&
                        g.outer > s[slot+1] && g.inner <= s[slot+1] && s[slot+1] > 0 &&
                        radius >= s[slot] && radius < s[slot+1]) expected = k;
                }
                check(record_at(v, slot, radius) == expected, "Reproduce writer distance gates");
            }
        }
    }
    auto rs = legacy_records(3);
    Geometry g;
    rs[2].tau = rs[1].tau;
    rejects([&] { build_metadata(rs, 3, g); }, "Reject duplicate times");
    rs = legacy_records(3); rs[2].cycle = rs[1].cycle;
    rejects([&] { build_metadata(rs, 3, g); }, "Reject duplicate cycles");
    rs = legacy_records(3); rs[2].a = -1;
    rejects([&] { build_metadata(rs, 3, g); }, "Reject nonpositive scale factors");
    rs = legacy_records(3); rs[2].boundaries[1] = NAN;
    rejects([&] { build_metadata(rs, 3, g); }, "Reject nonfinite boundaries");
    rs = legacy_records(3); std::swap(rs[2].boundaries[0], rs[2].boundaries[1]);
    rejects([&] { build_metadata(rs, 3, g); }, "Reject unordered boundaries");
    rejects([&] { read_info("/nonexistent/lightcone-info", 1); }, "Reject missing info file");
}
void background_tests() {
    Geometry g;
    auto rs = legacy_records(1);
    for (auto& r : rs) r.a = std::exp(0.3*r.tau);
    auto m = build_metadata(rs, 1, g);
    auto v = host_view(m, 16, g, nullptr);
    double a;
    for (const auto& r : rs) {
        check(background_at(v, r.tau, a) == Status::Ok, "Background knot available");
        near(a, r.a, 0, "Background reproduces exact knots");
    }
    for (int i = 0; i <= 400; ++i) {
        const double tau = i/200.0;
        check(background_at(v, tau, a) == Status::Ok, "Spline interior available");
        near(a, std::exp(0.3*tau), 8e-5, "Non-LCDM background spline");
        check(a > 0, "Positive spline");
    }
    check(background_at(v, -0.001, a) == Status::BackgroundCoverage, "No lower extrapolation");
    check(background_at(v, 2.001, a) == Status::BackgroundCoverage, "No upper extrapolation");
    rs.resize(2); rs[0].a = 0.4; rs[1].a = 0.8;
    m = build_metadata(rs, 1, g); v = host_view(m, 16, g, nullptr);
    check(background_at(v, 0.125, a) == Status::Ok, "Two-record background");
    near(a, 0.6, 1e-15, "Two-record linear interpolation");
    rs = legacy_records(1);
    for (std::size_t i = 0; i < rs.size(); ++i) rs[i].a = i%2 ? 0.01 : 2;
    m = build_metadata(rs, 1, g); v = host_view(m, 16, g, nullptr);
    for (int i = 0; i <= 400; ++i) {
        check(background_at(v, i/200.0, a) == Status::Ok && a >= 0.01-1e-14 && a <= 2+1e-14,
              "Positive shape-preserving interpolation across extrema");
    }
}
void storage_tests() {
    TempDirectory temp;
    for (int t : {1, 2, 3}) {
        Fixture f(8, t);
        f.fill([](int field, double, double x, double y, double z) { return static_cast<float>(1000*field+x+10*y+100*z); });
        const auto base = (temp.path/("lc"+std::to_string(t))).string();
        f.write(base);
        Reader reader(base, t);
        check(reader.grid_size() == 8, "Inspect legacy grid");
        const std::int64_t origin[3] = {-19, 6, 15};
        const int extent[3] = {12, 5, 4};
        for (int field = 0; field < 4; ++field) for (int slot = 0; slot < t; ++slot) {
            const auto values = reader.read_region(field, slot, origin, extent);
            for (int z = 0; z < extent[2]; ++z) for (int y = 0; y < extent[1]; ++y) for (int x = 0; x < extent[0]; ++x) {
                const auto input = wrap(origin[0]+x, 8)+8*(wrap(origin[1]+y, 8)+8*wrap(origin[2]+z, 8));
                const float expected = f.fields[field*t+slot][input];
                const float actual = values[x+extent[0]*(y+extent[1]*z)];
                check((std::isnan(actual) && std::isnan(expected)) || actual == expected, "Periodic XYZ hyperslab, including float64 conversion");
            }
        }
        // Scalar sources: _weyl files, or W = phi - chi/2 from _phi and _chi files.
        rejects([&] { Reader no_chi(base, t, ScalarSource::PhiChi); }, "Reject phi-chi without chi files");
        rejects([&] { Reader no_weyl(base, t, ScalarSource::Weyl); }, "Reject weyl without weyl files");
        std::vector<std::vector<float>> chi(t), weyl(t);
        for (int slot = 0; slot < t; ++slot) {
            chi[slot].resize(512); weyl[slot].resize(512);
            for (int i = 0; i < 512; ++i) {
                const float phi = f.fields[slot][i];
                chi[slot][i] = std::isnan(phi) ? NAN : 0.37f*phi-3.1f+0.001f*i;
                weyl[slot][i] = phi-0.5f*chi[slot][i];
            }
            const std::string suffix = t == 1 ? "" : "_"+std::to_string(slot);
            write_grid(base+"_chi"+suffix+".h5", 8, chi[slot], slot%2 == 1); // float64 chi exercises conversion
            write_grid(base+"_weyl"+suffix+".h5", 8, weyl[slot]);
        }
        Reader combined(base, t, ScalarSource::PhiChi), stored(base, t, ScalarSource::Weyl);
        check(combined.filename(0, 0) == reader.filename(0, 0) && combined.filename(CHI_FILE, 0).find("_chi") != std::string::npos &&
              stored.filename(0, 0).find("_weyl") != std::string::npos, "Scalar source file names");
        rejects([&] { reader.filename(CHI_FILE, 0); }, "No chi file without phi-chi");
        for (int slot = 0; slot < t; ++slot) {
            const auto from_phi_chi = combined.read_region(0, slot, origin, extent);
            const auto from_weyl = stored.read_region(0, slot, origin, extent);
            const auto b = combined.read_region(1, slot, origin, extent);
            const auto b_reference = reader.read_region(1, slot, origin, extent);
            for (std::size_t i = 0; i < from_phi_chi.size(); ++i) {
                const float a = from_phi_chi[i], w = from_weyl[i];
                check((std::isnan(a) && std::isnan(w)) || a == w, "phi - chi/2 matches stored weyl bit for bit");
                check((std::isnan(b[i]) && std::isnan(b_reference[i])) || b[i] == b_reference[i], "B unaffected by scalar source");
            }
        }
        std::filesystem::remove(reader.filename(3, t-1));
        rejects([&] { Reader missing(base, t); }, "Reject missing component");
        write_grid(reader.filename(3, t-1), 7, std::vector<float>(343));
        rejects([&] { Reader mismatch(base, t); }, "Reject mismatched grid");
    }
    const std::uint16_t halves[] = {0, 0x3c00, 0xbc00, 1, 0x7c00, 0x7e00};
    near(decode_half(halves[1]), 1, 0, "Half decode normal");
    near(decode_half(halves[2]), -1, 0, "Half decode negative");
    near(decode_half(halves[3]), std::ldexp(1.0, -24), 0, "Half decode subnormal");
    check(!lightcone::finite(decode_half(halves[4])) && !lightcone::finite(decode_half(halves[5])), "Nonfinite half decode");
}
void sampling_tests() {
    for (int t : {2, 3, 5}) for (bool staggered : {false, true}) {
        Fixture f(32, t, staggered);
        f.fill([](int field, double tau, double x, double y, double z) {
            return static_cast<float>((field+1)*1e-5*(tau+1)*(x*x+2*y*y+3*z*z+x*y));
        });
        auto v = f.view();
        for (int field = 0; field < 4; ++field) {
            const double x = 17.25, y = 16.375, z = 15.125, tau = 0.93, c = (field+1)*1e-5*(tau+1);
            float value, grad[3], hessian[3][3];
            check(sample(v, field, tau, x, y, z, value, grad, hessian) == Status::Ok, "Fixed-time interpolation available");
            near(value, c*(x*x+2*y*y+3*z*z+x*y), 8e-7, "Linear time before cubic space");
            near(grad[0], c*(2*x+y), 5e-7, "Fixed-time X gradient");
            near(grad[1], c*(4*y+x), 5e-7, "Fixed-time Y gradient");
            near(grad[2], c*6*z, 5e-7, "Fixed-time Z gradient");
            near(hessian[0][0], 2*c, 8e-7, "Fixed-time XX Hessian");
            near(hessian[1][1], 4*c, 8e-7, "Fixed-time YY Hessian");
            near(hessian[2][2], 6*c, 8e-7, "Fixed-time ZZ Hessian");
            near(hessian[0][1], c, 8e-7, "Fixed-time mixed Hessian");
        }
        const std::int64_t xyz[3] = {16, 16, 16};
        float value;
        check(node_at(v, 0, xyz, 3.0, value) == Status::TimeCoverage, "Strict missing time bracket");
        // Ignore poisoned storage in a slot that is not selected at an exact time.
        std::fill(f.fields[0].begin(), f.fields[0].end(), NAN);
        check(node_at(v, 0, xyz, 1.0, value) == Status::Ok, "Exact-time sample needs no second slot");
    }
    Fixture single(16, 1);
    single.fill([](int, double tau, double, double, double) { return static_cast<float>(tau); });
    float value, grad[3], hessian[3][3];
    std::int64_t xyz[3] = {8, 8, 8};
    check(node_at(single.view(), 0, xyz, 1.8, value) == Status::Ok, "Thickness one uses available nearest-time record");
    near(value, 1.0, 0, "Nearest-time record belongs to background shell");
    single.geometry.angular_limit = true;
    single.geometry.opening_cos = 0.9;
    check(node_at(single.view(), 0, xyz, 1.0, value) == Status::SpatialCoverage, "Angular vertex has no cap stencil");
    single.geometry.angular_limit = false;
    xyz[0] = 16*40+8;
    check(node_at(single.view(), 0, xyz, 1.0, value) == Status::SpatialCoverage, "Radius is unwrapped, not minimum-image");

    Fixture full(16, 1);
    full.fill([](int, double, double x, double y, double z) { return static_cast<float>(x+y+z); });
    TempDirectory temp;
    const auto base = (temp.path/"tiles").string(); full.write(base);
    Reader reader(base, 1);
    const std::int64_t origin[3] = {6, 6, 6}; const int extent[3] = {5, 5, 5};
    auto tile = reader.read_region(0, 0, origin, extent);
    auto tile_grids = full.grids;
    auto& grid = tile_grids[0]; grid.values = tile.data();
    for (int i = 0; i < 3; ++i) { grid.origin[i] = origin[i]; grid.extent[i] = extent[i]; }
    grid.strides[1] = 5; grid.strides[2] = 25;
    auto tiled = full.view(); tiled.grids = tile_grids.data();
    float tile_value, tile_grad[3], tile_hessian[3][3];
    check(sample(full.view(), 0, 1, 8.2, 8.3, 8.4, value, grad, hessian) == Status::Ok, "Full grid sample");
    check(sample(tiled, 0, 1, 8.2, 8.3, 8.4, tile_value, tile_grad, tile_hessian) == Status::Ok, "Resident tile plus stencil halo");
    near(tile_value, value, 0, "Tile/full identical value");
    for (int i = 0; i < 3; ++i) {
        near(tile_grad[i], grad[i], 0, "Tile/full identical gradient");
        for (int j = 0; j < 3; ++j) near(tile_hessian[i][j], hessian[i][j], 0, "Tile/full identical Hessian");
    }
    grid.extent[0] = 3;
    check(sample(tiled, 0, 1, 8.2, 8.3, 8.4, tile_value, tile_grad, tile_hessian) == Status::NotResident, "Residency differs from coverage");
    const std::int64_t seam_origin[3] = {-1, 7, 7}; const int seam_extent[3] = {4, 4, 4};
    auto seam = reader.read_region(0, 0, seam_origin, seam_extent);
    grid.values = seam.data(); grid.strides[1] = 4; grid.strides[2] = 16;
    for (int i = 0; i < 3; ++i) { grid.origin[i] = seam_origin[i]; grid.extent[i] = 4; }
    check(sample(full.view(), 0, 1, 0.2, 8.3, 8.4, value, grad, hessian) == Status::Ok, "Periodic full grid stencil");
    check(sample(tiled, 0, 1, 0.2, 8.3, 8.4, tile_value, tile_grad, tile_hessian) == Status::Ok, "Periodic resident tile stencil");
    near(tile_value, value, 0, "Periodic tile/full identical value");
    for (int i = 0; i < 3; ++i) {
        near(tile_grad[i], grad[i], 0, "Periodic tile/full identical gradient");
        for (int j = 0; j < 3; ++j) near(tile_hessian[i][j], hessian[i][j], 0, "Periodic tile/full identical Hessian");
    }
    full.fields[0][8+16*(8+16*8)] = NAN;
    check(sample(full.view(), 0, 1, 8.2, 8.3, 8.4, value, grad, hessian) == Status::InvalidValue, "Selected nonfinite fields rejected");
}
void integration_tests() {
    Fixture f(32, 3);
    f.fill([](int, double, double, double, double) { return 0.0f; });
    auto v = f.view();
    double y[STATE_SIZE], z;
    check(initialize(v, 1.0, 1.2, 0.3, y) == Status::Ok, "Observer initialization");
    check(observed_redshift(v, y, z) == Status::Ok, "Observer redshift available");
    near(z, 0, 1e-15, "Nonunit a_obs redshift normalization");
    check(trace(v, 0.03, y, z) == Status::Ok, "Trace to target with tabulated background");
    check(z >= 0.03 && z-0.03 <= FINAL_Z_TOL, "Final-step target tolerance");
    double a; check(background_at(v, y[STATE_TAU], a) == Status::Ok, "Endpoint background");
    near(z, 0.7/a-1, 1e-14, "Homogeneous redshift a_obs/a_source");
    check(initialize(v, 1.0, 1.2, 0.3, y) == Status::Ok, "Reinitialize");
    const auto before = std::vector<double>(y, y+STATE_SIZE);
    check(rk4_step(v, y, 40.0) != Status::Ok, "Invalid RK stage rejects entire step");
    check(std::memcmp(y, before.data(), sizeof(y)) == 0, "Rejected step does not change state");
    // A short background domain forces a fractional last step without extrapolation.
    Fixture short_domain(16, 1);
    short_domain.fill([](int, double, double, double, double) { return 0.0f; });
    auto records = short_domain.metadata.records;
    records.resize(2); records[0].tau = 0.97; records[1].tau = 1.5;
    for (auto& r : records) r.boundaries = {-1.0, 2.0};
    short_domain.metadata = build_metadata(records, 1, short_domain.geometry);
    v = short_domain.view();
    v.time_derivatives = TimeDerivativeMode::Zero;
    check(initialize(v, 1.0, 1.2, 0.3, y) == Status::Ok, "Short domain initialization");
    check(trace(v, 10.0, y, z) == Status::TargetNotReached, "Time domain bounds unsuccessful tracing");
    near(y[STATE_TAU], 0.97, 0, "Fractional step lands on earliest time");
}
}
int main() {
    try {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        metadata_tests(); background_tests(); storage_tests(); sampling_tests(); integration_tests();
        std::cout << "CPU light-cone checks passed (" << checks << ").\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
