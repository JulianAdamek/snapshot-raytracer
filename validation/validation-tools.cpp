#include "lightcone_ray.hpp"
#include "lightcone_maps.hpp"
#include <cfloat>
#include <chealpix.h>
#include <hdf5.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
struct Handle {
    hid_t id; herr_t (*close)(hid_t);
    Handle(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer) { require(id >= 0, "HDF5 operation failed"); }
    ~Handle() { close(id); }
    Handle(const Handle&) = delete;
};
int grid_size(hid_t data) {
    Handle space(H5Dget_space(data), H5Sclose);
    require(H5Sget_simple_extent_ndims(space.id) == 3, "Expected rank-3 snapshot");
    hsize_t dims[3]; H5Sget_simple_extent_dims(space.id, dims, nullptr);
    require(dims[0] > 0 && dims[0] <= 100000 && dims[0] == dims[1] && dims[1] == dims[2], "Expected cubic snapshot");
    return static_cast<int>(dims[0]);
}
void prepare(const fs::path& potential, const fs::path& vector, const fs::path& output, int expected_n) {
    require(!fs::exists(output), "Fixture directory already exists; choose a new output directory");
    Handle phi_file(H5Fopen(potential.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    Handle phi(H5Dopen2(phi_file.id, "/data", H5P_DEFAULT), H5Dclose);
    Handle phi_type(H5Dget_type(phi.id), H5Tclose);
    require(H5Tget_class(phi_type.id) == H5T_FLOAT, "Snapshot potential must be scalar floating point");
    const int n = grid_size(phi.id);
    require(n == expected_n, "Snapshot Ngrid disagrees with settings");
    Handle b_file(H5Fopen(vector.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    Handle b(H5Dopen2(b_file.id, "/data", H5P_DEFAULT), H5Dclose);
    Handle b_type(H5Dget_type(b.id), H5Tclose);
    require(grid_size(b.id) == n && H5Tget_class(b_type.id) == H5T_ARRAY && H5Tget_array_ndims(b_type.id) == 1,
            "Expected matching packed array[3] B snapshot");
    hsize_t components; H5Tget_array_dims2(b_type.id, &components);
    require(components == 3, "Expected three B components");
    Handle base_type(H5Tget_super(b_type.id), H5Tclose);
    require(H5Tget_class(base_type.id) == H5T_FLOAT, "Expected floating-point B components");
    Handle mem_type(H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &components), H5Tclose);
    const std::size_t size = static_cast<std::size_t>(n)*n*n;
    std::vector<float> packed(3*size);
    require(H5Dread(b.id, mem_type.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, packed.data()) >= 0, "Cannot read B snapshot");
    fs::create_directories(output);
    const hsize_t dims[3] = {static_cast<hsize_t>(n), static_cast<hsize_t>(n), static_cast<hsize_t>(n)};
    std::vector<float> component(size);
    for (int c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < size; ++i) component[i] = packed[3*i+c];
        const auto path = output/("B"+std::to_string(c+1)+".h5");
        Handle file(H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        Handle space(H5Screate_simple(3, dims, nullptr), H5Sclose);
        Handle data(H5Dcreate2(file.id, "/data", H5T_IEEE_F32LE, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
        require(H5Dwrite(data.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, component.data()) >= 0, "Cannot write B component");
        for (int slot = 0; slot < 3; ++slot)
            fs::create_symlink(path.filename(), output/("lc_B"+std::to_string(c+1)+"_"+std::to_string(slot)+".h5"));
    }
    for (int slot = 0; slot < 3; ++slot)
        fs::create_symlink(fs::absolute(potential), output/("lc_phi_"+std::to_string(slot)+".h5"));
    lightcone::Reader reader((output/"lc").string(), 3);
    require(reader.grid_size() == n, "Prepared fixture grid mismatch");
    std::cout << "Prepared three identical slots; source snapshots are read-only. Ngrid=" << n << '\n';
}
void check_fixture(const fs::path& info, int n, double observer_tau, double depth) {
    lightcone::Geometry geometry;
    for (double& x : geometry.vertex) x = 0.5;
    auto metadata = lightcone::build_metadata(lightcone::read_info(info.string(), 3), 3, geometry);
    std::array<lightcone::GridView, 12> grids;
    for (int field = 0; field < 4; ++field) for (int slot = 0; slot < 3; ++slot)
        grids[field*3+slot] = lightcone::float_grid(nullptr, n, field, true);
    const auto view = lightcone::host_view(metadata, n, geometry, grids.data());
    std::uint64_t checked = 0;
    // Cover directions, time stages and the complete staggered stencil without loading payloads.
    for (int pixel = 0; pixel < 12*4*4; ++pixel) {
        double theta, phi; pix2ang_ring64(4, pixel, &theta, &phi);
        const double direction[3] = {std::sin(theta)*std::cos(phi), std::sin(theta)*std::sin(phi), std::cos(theta)};
        for (double s = 0; s <= depth+1; s += 0.5) for (int field = 0; field < 4; ++field) {
            std::int64_t cell[3];
            for (int axis = 0; axis < 3; ++axis)
                cell[axis] = static_cast<std::int64_t>(std::floor(0.5*n+s*direction[axis]-grids[field*3].node_offset[axis]));
            for (int z = -1; z <= 2; ++z) for (int y = -1; y <= 2; ++y) for (int x = -1; x <= 2; ++x) {
                const std::int64_t xyz[3] = {cell[0]+x, cell[1]+y, cell[2]+z};
                float ignored, slope;
                const auto status = lightcone::node_at(view, field, xyz, observer_tau-s/n, ignored, &slope);
                require(status == lightcone::Status::NotResident,
                        std::string("Synthetic shell coverage failed: ")+lightcone::status_name(status));
                ++checked;
            }
        }
    }
    std::cout << "Synthetic full-stencil time-coverage preflight passed: " << checked << " nodes\n";
}
struct Map {
    long nside = 0;
    std::vector<float> data;
    std::string ordering;
    explicit Map(long n) : nside(n), data(12*n*n,0.0f), ordering("ZERO (frozen reference)") {}
    explicit Map(const fs::path& path) {
        require(fs::is_regular_file(path), "Missing FITS map: "+path.string());
        char coords[32] = {}, scheme[32] = {};
        std::unique_ptr<float, decltype(&std::free)> values(read_healpix_map(path.c_str(), &nside, coords, scheme), &std::free);
        ordering = scheme;
        require(values && lightcone::valid_nside(nside) && (ordering == "RING" || ordering == "NESTED"), "Invalid FITS map or ordering");
        data.assign(values.get(), values.get()+12*nside*nside);
        if (ordering == "RING") data = lightcone::ring_to_nested(data, nside);
    }
};
void compare(const fs::path& snapshot, const fs::path& cone, int nside, int batch, double target,
             double distance_cells, double beta, double om, double ol, bool strict = false, bool frozen_reference = false, bool require_zero_isw = false) {
    const auto& names = lightcone::MAP_NAMES;
    const double floors[] = {1e-4, 1e-7, 1e-7, 1e-10, 1e-8, 1e-8, 3e-7, 0};
    const double endpoint_budget = strict ? 0.0 : 2e-4/(beta*std::sqrt(om*std::pow(1+target,3)+ol));
    const double relative_budget = strict ? 4*FLT_EPSILON : 1e-4+4*endpoint_budget/std::max(1.0, distance_cells);
    const std::int64_t count = batch < 0 ? 12LL*nside*nside : 1LL*nside*nside;
    const std::int64_t first = batch < 0 ? 0 : batch*count;
    bool all_ok = true;
    std::cout << std::setprecision(17) << "{\"target_z\":" << target << ",\"batch\":" << batch
              << ",\"nside\":" << nside << ",\"strict\":" << (strict ? "true" : "false") << ",\"canonical_ordering\":\"NESTED\",\"endpoint_budget_cells\":" << endpoint_budget
              << ",\"shape_relative_budget\":" << relative_budget << ",\"maps\":[";
    for (int map = 0; map < lightcone::MAP_COUNT; ++map) {
        // Logical batch -1 means full sky, but the legacy output label is batch0.
        const std::string filename = std::string(names[map])+"_batch"+std::to_string(batch < 0 ? 0 : batch)+".fits";
        // References are always full sky; the snapshot executable's batches are RING ranges.
        const Map a = map == 7 && frozen_reference ? Map(nside) : Map(snapshot/(std::string(names[map])+"_batch0.fits"));
        const Map b(cone/filename);
        require(a.nside == nside && b.nside == nside, "FITS Nside mismatch");
        double maximum = 0, reference_max = 0, squared = 0, reference_squared = 0, z_error = 0;
        std::uint64_t invalid = 0, mask_mismatch = 0, outside_nonzero = 0, rounding_failures = 0, nonzero_isw = 0;
        for (std::size_t i = 0; i < a.data.size(); ++i) {
            if (i < static_cast<std::size_t>(first) || i >= static_cast<std::size_t>(first+count)) {
                if (b.data[i] != 0) ++outside_nonzero;
                continue;
            }
            // chealpix/CFITSIO replaces FITS nulls (including stored NaNs) with
            // HEALPix's finite UNSEEN sentinel when reading the map.
            const auto valid = [](float value) { return std::isfinite(value) && value != -1.6375e30f; };
            const bool valid_a = valid(a.data[i]), valid_b = valid(b.data[i]);
            if (valid_a != valid_b) ++mask_mismatch;
            if (!valid_a || !valid_b) { ++invalid; continue; }
            if (map == 7 && (frozen_reference || require_zero_isw) && (a.data[i] != 0 || b.data[i] != 0)) ++nonzero_isw;
            const double delta = static_cast<double>(b.data[i])-a.data[i];
            if (strict && std::fabs(delta) > 4*FLT_EPSILON*std::max(std::fabs(a.data[i]),std::fabs(b.data[i]))+4*std::numeric_limits<float>::denorm_min()) ++rounding_failures;
            maximum = std::max(maximum, std::fabs(delta)); reference_max = std::max(reference_max, std::fabs(static_cast<double>(a.data[i])));
            squared += delta*delta; reference_squared += static_cast<double>(a.data[i])*a.data[i];
            if (map == 6) z_error = std::max({z_error, std::fabs(a.data[i]-target), std::fabs(b.data[i]-target)});
        }
        const double tolerance = map == 7 && frozen_reference ? 0.0 : strict ? relative_budget*reference_max+4*std::numeric_limits<float>::denorm_min() :
                                 map == 0 ? endpoint_budget+floors[map] : map == 6 ? 1.003e-4 : floors[map]+relative_budget*reference_max;
        const bool ok = nonzero_isw == 0 && invalid == 0 && mask_mismatch == 0 && outside_nonzero == 0 &&
                        (strict ? rounding_failures == 0 : maximum <= tolerance) &&
                        (map != 6 || z_error <= 1.003e-4);
        all_ok &= ok;
        if (map) std::cout << ',';
        std::cout << "{\"name\":\"" << names[map] << "\",\"max_abs_difference\":" << maximum
                  << ",\"rms_difference\":" << std::sqrt(squared/count)
                  << ",\"relative_l2_difference\":";
        if (reference_squared) std::cout << std::sqrt(squared/reference_squared);
        else std::cout << (squared == 0 ? "0" : "null");
        std::cout << ",\"reference_max_abs\":" << reference_max << ",\"absolute_tolerance\":" << tolerance
                  << ",\"reference_ordering\":\"" << a.ordering << "\",\"candidate_ordering\":\"" << b.ordering << "\",\"rounding_failures\":" << rounding_failures << ",\"invalid_pixels\":" << invalid << ",\"mask_mismatches\":" << mask_mismatch
                  << ",\"nonzero_isw_pixels\":" << nonzero_isw << ",\"outside_batch_nonzero\":" << outside_nonzero << ",\"max_target_z_error\":" << z_error
                  << ",\"pass\":" << (ok ? "true" : "false") << '}';
    }
    std::cout << "],\"pass\":" << (all_ok ? "true" : "false") << "}\n";
}
void self_test(const fs::path& root, const std::string& fault) {
    fs::create_directories(root/"snapshot"); fs::create_directories(root/"lightcone");
    std::vector<float> data(12*4*4);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = 0.001f+i*1e-7f;
    const bool batched = fault == "batch" || fault == "batch-ring" || fault == "outside";
    const bool strict = fault == "strict" || fault == "strict-difference";
    const auto& names = lightcone::MAP_NAMES;
    for (int m = 0; m < lightcone::MAP_COUNT; ++m) {
        if (m == 6) std::fill(data.begin(), data.end(), 0.1f);
        if (m == 7) std::fill(data.begin(), data.end(), 0.0f);
        for (const char* dir : {"snapshot", "lightcone"}) {
            const bool candidate = std::string(dir) == "lightcone";
            if (m == 7 && !candidate && !strict) continue; // actual snapshot format has seven maps
            auto path = root/dir/(std::string(names[m])+"_batch"+(candidate && batched ? "7" : "0")+".fits");
            require(!fs::exists(path), "Self-test file exists");
            auto values = data;
            if (candidate && batched) for (std::size_t i = 0; i < values.size(); ++i)
                if (i < 7*16 || i >= 8*16) values[i] = 0;
            if (m == 7 && candidate) {
                if (fault == "isw-nonzero") values[0] = 1e-8f;
                if (fault == "isw-nan") values[0] = NAN;
            }
            if (m == 0 && candidate) {
                if (fault == "nan") values[0] = std::numeric_limits<float>::quiet_NaN();
                if (fault == "difference" || fault == "outside") values[0] = 1.0f;
                if (fault == "strict-difference") values[0] += 1e-7f;
            }
            lightcone::write_nested_map(values, 4, !candidate || fault == "batch-ring" ? lightcone::Ordering::Ring : lightcone::Ordering::Nested, path.string());
        }
    }
    // Exercise conversion of actual array[3] float64 datasets and slot links.
    const int n = 8;
    const std::size_t size = n*n*n;
    std::vector<double> source(3*size);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i+1)*1.23456789123e-9;
    const hsize_t dims[3] = {n, n, n}, three = 3;
    {
        Handle file(H5Fcreate((root/"source_B.h5").c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        Handle space(H5Screate_simple(3, dims, nullptr), H5Sclose);
        Handle type(H5Tarray_create2(H5T_IEEE_F64LE, 1, &three), H5Tclose);
        Handle memory(H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, &three), H5Tclose);
        Handle data(H5Dcreate2(file.id, "/data", type.id, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
        require(H5Dwrite(data.id, memory.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, source.data()) >= 0, "Write test packed vector");
    }
    {
        Handle file(H5Fcreate((root/"source_phi.h5").c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        Handle space(H5Screate_simple(3, dims, nullptr), H5Sclose);
        Handle data(H5Dcreate2(file.id, "/data", H5T_IEEE_F64LE, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
        require(H5Dwrite(data.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, source.data()) >= 0, "Write test potential");
    }
    // Keep stdout machine-readable JSON for the caller.
    auto* saved = std::cout.rdbuf(std::cerr.rdbuf());
    prepare(root/"source_phi.h5", root/"source_B.h5", root/"fixture", n);
    std::cout.rdbuf(saved);
    lightcone::Reader reader((root/"fixture"/"lc").string(), 3);
    const std::int64_t origin[3] = {0, 0, 0}; const int extent[3] = {n, n, n};
    for (int field = 0; field < 4; ++field) for (int slot = 0; slot < 3; ++slot) {
        const auto values = reader.read_region(field, slot, origin, extent);
        for (std::size_t i = 0; i < size; ++i)
            require(values[i] == static_cast<float>(source[field ? 3*i+field-1 : i]), "Prepared slot differs from snapshot float32 conversion");
    }
    require(fs::equivalent(root/"fixture"/"lc_phi_0.h5", root/"source_phi.h5"), "Potential link does not preserve source");
    require(fs::equivalent(root/"fixture"/"lc_B1_0.h5", root/"fixture"/"lc_B1_2.h5"), "Slots are not identical");
    compare(root/"snapshot", root/"lightcone", 4, batched ? 7 : -1, 0.1, 20, 0.005, 0.3, 0.7, strict, !strict);
}
void assembly_control(const fs::path& root, bool outside) {
    const int nside = 2, face_size = nside*nside;
    fs::create_directories(root/"reference"); fs::create_directories(root/"faces");
    const auto& names = lightcone::MAP_NAMES;
    for (int m = 0; m < lightcone::MAP_COUNT; ++m) {
        std::vector<float> full(12*face_size);
        for (std::size_t i = 0; i < full.size(); ++i) full[i] = m == 6 ? 0.1f : 0.001f+i*1e-6f;
        lightcone::write_nested_map(full,nside,lightcone::Ordering::Ring,(root/"reference"/(std::string(names[m])+"_batch0.fits")).string());
        for (int face = 0; face < 12; ++face) {
            const auto directory = root/"faces"/("face-"+(face < 10 ? std::string("0") : std::string())+std::to_string(face));
            fs::create_directories(directory);
            std::vector<float> part(full.size(),0.0f);
            std::copy_n(full.data()+face*face_size,face_size,part.data()+face*face_size);
            if (outside && face == 1 && m == 0) part[0] = 1;
            lightcone::write_nested_map(part,nside,face%2 ? lightcone::Ordering::Ring : lightcone::Ordering::Nested,
                (directory/(std::string(names[m])+"_batch"+std::to_string(face)+".fits")).string());
        }
    }
}
void assemble(const fs::path& reference, const fs::path& faces, int nside, double target, bool require_zero_isw = false) {
    require(lightcone::valid_nside(nside), "Invalid assembly Nside");
    const auto output = faces/"assembled";
    require(!fs::exists(output), "Assembly output already exists");
    fs::create_directory(output);
    const auto& names = lightcone::MAP_NAMES;
    const std::size_t face_size = static_cast<std::size_t>(nside)*nside;
    for (const char* name : names) {
        std::vector<float> combined(12*face_size,0.0f);
        for (int face = 0; face < 12; ++face) {
            const auto directory = faces/("face-"+(face < 10 ? std::string("0") : std::string())+std::to_string(face));
            const Map map(directory/(std::string(name)+"_batch"+std::to_string(face)+".fits"));
            require(map.nside == nside, "Assembly Nside mismatch");
            for (std::size_t i = 0; i < map.data.size(); ++i) {
                if (i/face_size == static_cast<std::size_t>(face)) combined[i] = map.data[i];
                else require(map.data[i] == 0, "Nonzero value outside declared base face");
            }
        }
        lightcone::write_nested_map(combined,nside,lightcone::Ordering::Nested,(output/(std::string(name)+"_batch0.fits")).string());
    }
    compare(reference,output,nside,-1,target,1,1,0.3,0.7,true,false,require_zero_isw);
}
}
int main(int argc, char** argv) {
    try {
        require(argc >= 2, "Modes: prepare, check-fixture, compare, compare-strict, compare-frozen, assemble, self-test");
        const std::string mode = argv[1];
        if (mode == "prepare" && argc == 6) prepare(argv[2], argv[3], argv[4], std::stoi(argv[5]));
        else if (mode == "check-fixture" && argc == 6) check_fixture(argv[2], std::stoi(argv[3]), std::stod(argv[4]), std::stod(argv[5]));
        else if (mode == "compare" && argc == 11) compare(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]),
                std::stod(argv[6]), std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]),false,true);
        else if (mode == "compare-frozen" && argc == 7) compare(argv[2],argv[3],std::stoi(argv[4]),std::stoi(argv[5]),std::stod(argv[6]),1,1,0.3,0.7,true,true);
        else if (mode == "compare-strict" && (argc == 7 || (argc == 8 && std::string(argv[7]) == "--require-zero-isw")))
            compare(argv[2],argv[3],std::stoi(argv[4]),std::stoi(argv[5]),std::stod(argv[6]),1,1,0.3,0.7,true,false,argc == 8);
        else if (mode == "assembly-self-test" && (argc == 3 || argc == 4)) {
            assembly_control(argv[2],argc == 4);
            assemble(fs::path(argv[2])/"reference",fs::path(argv[2])/"faces",2,0.1);
        }
        else if (mode == "assemble" && (argc == 6 || (argc == 7 && std::string(argv[6]) == "--require-zero-isw")))
            assemble(argv[2],argv[3],std::stoi(argv[4]),std::stod(argv[5]),argc == 7);
        else if (mode == "self-test" && (argc == 3 || argc == 4)) self_test(argv[2], argc == 4 ? argv[3] : "");
        else throw std::runtime_error("Invalid validation-tools arguments");
        return 0;
    } catch (const std::exception& error) { std::cerr << "validation-tools: " << error.what() << '\n'; return 1; }
}
