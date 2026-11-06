#pragma once
#include "lightcone_ray.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <hdf5.h>

inline int checks = 0;
inline void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
inline void near(double actual, double expected, double tolerance, const std::string& message) {
    check(std::isfinite(actual) && std::fabs(actual-expected) <= tolerance,
          message+": got "+std::to_string(actual)+", expected "+std::to_string(expected));
}
template<class Function> void rejects(Function function, const std::string& message) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
struct TempDirectory {
    std::filesystem::path path;
    TempDirectory() {
        path = std::filesystem::temp_directory_path()/
               ("lightcone-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(path), "Create fixture directory");
    }
    ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
inline void write_info(const std::string& path, const std::vector<lightcone::InfoRecord>& records) {
    std::ofstream out(path, std::ios::binary);
    for (const auto& r : records) {
        auto put = [&](const auto& x) { out.write(reinterpret_cast<const char*>(&x), sizeof(x)); };
        put(r.cycle); put(r.tau); put(r.a); put(r.particle_inner); put(r.particle_outer);
        for (double x : r.boundaries) put(x);
    }
    check(static_cast<bool>(out), "Write binary fixture");
}
inline void write_grid(const std::string& path, int n, const std::vector<float>& values, bool doubles = false) {
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    const hsize_t dimensions[3] = {static_cast<hsize_t>(n), static_cast<hsize_t>(n), static_cast<hsize_t>(n)};
    const hid_t space = H5Screate_simple(3, dimensions, nullptr);
    const hid_t data = H5Dcreate2(file, "/data", doubles ? H5T_IEEE_F64LE : H5T_IEEE_F32LE, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check(data >= 0 && H5Dwrite(data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0,
          "Write grid fixture");
    H5Dclose(data); H5Sclose(space); H5Fclose(file);
}
inline std::vector<lightcone::InfoRecord> legacy_records(int thickness, double dt = 0.25) {
    std::vector<lightcone::InfoRecord> records;
    for (int k = 0; k <= static_cast<int>(2.0/dt); ++k) {
        const double tau = k*dt;
        lightcone::InfoRecord r{k, tau, 0.5+0.2*tau, 1-tau-0.5*dt, 1-tau+0.5*dt, {}};
        if (thickness == 1) r.boundaries = {r.particle_inner, r.particle_outer};
        else if (thickness == 2) r.boundaries = {1-tau-dt, 1-tau, 1-tau+dt};
        else if (thickness == 3) r.boundaries = {1-tau-2*dt, 1-tau-0.5*dt, 1-tau+0.5*dt, 1-tau+1.5*dt};
        else for (int j = 0; j <= thickness; ++j) r.boundaries.push_back(1-tau+(j-0.5*thickness)*dt);
        records.push_back(r);
    }
    return records;
}
struct Fixture {
    int n, thickness;
    lightcone::Geometry geometry;
    lightcone::Metadata metadata;
    std::vector<std::vector<float>> fields;
    std::vector<lightcone::GridView> grids;
    Fixture(int size, int t, bool staggered = true, double dt = 0.5) : n(size), thickness(t) {
        for (double& x : geometry.vertex) x = 0.5;
        metadata = lightcone::build_metadata(legacy_records(t, dt), t, geometry);
        fields.resize(4*t, std::vector<float>(static_cast<std::size_t>(n)*n*n, NAN));
        for (int field = 0; field < 4; ++field) for (int slot = 0; slot < t; ++slot)
            grids.push_back(lightcone::float_grid(fields[field*t+slot].data(), n, field, staggered));
    }
    lightcone::View view() const { return lightcone::host_view(metadata, n, geometry, grids.data()); }
    void fill(const std::function<float(int,double,double,double,double)>& function) {
        const auto v = view();
        for (int field = 0; field < 4; ++field) for (int slot = 0; slot < thickness; ++slot)
            for (int z = 0; z < n; ++z) for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
                const auto& grid = grids[field*thickness+slot];
                const double px = x+grid.node_offset[0], py = y+grid.node_offset[1], pz = z+grid.node_offset[2];
                const double dx = px/n-geometry.vertex[0], dy = py/n-geometry.vertex[1], dz = pz/n-geometry.vertex[2];
                const int record = lightcone::record_at(v, slot, std::sqrt(dx*dx+dy*dy+dz*dz));
                fields[field*thickness+slot][x+n*(y+n*z)] = record < 0 ? NAN :
                    function(field, metadata.times[record], px, py, pz);
            }
    }
    void write(const std::string& base) const {
        for (int field = 0; field < 4; ++field) for (int slot = 0; slot < thickness; ++slot) {
            const auto path = base+(field ? "_B"+std::to_string(field) : "_phi")+
                              (thickness == 1 ? "" : "_"+std::to_string(slot))+".h5";
            write_grid(path, n, fields[field*thickness+slot], field == 2);
        }
        write_info(base+"_info.bin", metadata.records);
    }
};
