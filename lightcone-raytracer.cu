// Light-cone fork of snapshot-raytracer: the optical equations live in lightcone_ray.hpp.
#include "lightcone_ray.hpp"
#include "lightcone_maps.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chealpix.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max()/sizeof(T)) throw std::runtime_error("Device allocation overflows");
        if (count) cuda_check(cudaMalloc(&data, count*sizeof(T)));
    }
    explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(values.size()) {
        if (!values.empty()) {
            const auto result = cudaMemcpy(data, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice);
            if (result != cudaSuccess) { cudaFree(data); data = nullptr; cuda_check(result); }
        }
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data(other.data) { other.data = nullptr; }
};
struct Options {
    std::string base, info, geometry_file;
    lcg::GeometryInfo writer;
    lightcone::ScalarSource scalar = lightcone::ScalarSource::Phi;
    std::int64_t nside = 0;
    int thickness = 0, batch = -1;
    double observer_tau = NAN, target_z = NAN;
    lightcone::Geometry geometry;
    bool staggered = true, ziggurat = false;
    lightcone::Ordering ordering = lightcone::Ordering::Nested;
    int slab_cells = 16, padding_cells = 8;
    double max_lookback = INFINITY;
    lightcone::TimeDerivativeMode time_derivatives = lightcone::TimeDerivativeMode::FiniteDifference;
};
const char* usage =
    "Usage: lightcone-raytracer <lightcone-base> <Nside> --info <info.bin> --target-z <z>\n"
    "  (--geometry <lccube_geometry.bin> | --thickness <T> --vertex <x/L> <y/L> <z/L> --observer-tau <tau/L>)\n"
    "  [--batch <0..11>]\n"
    "  [--b-layout staggered|collocated] [--direction <dx> <dy> <dz> --opening-deg <angle>]\n"
    "  [--radial-range <rmin/L> <rmax/L>] [--ordering nested|ring]\n"
    "  [--storage full|ziggurat] [--slab-cells <N>] [--padding-cells <N>]\n"
    "  [--max-lookback-cells <S>] [--time-derivatives finite-difference|zero]\n"
    "  [--scalar phi|weyl|phi-chi]\n"
    "With --geometry, vertex, direction, opening, radial range, thickness, B layout and\n"
    "the default observer time come from the binary file. Otherwise legacy geometry is used.\n";
Options parse(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error(usage);
    Options o;
    o.base = argv[1];
    auto number = [](const std::string& s) {
        std::size_t end;
        const double value = std::stod(s, &end);
        if (end != s.size() || !std::isfinite(value)) throw std::runtime_error("Expected finite number: " + s);
        return value;
    };
    auto integer = [](const std::string& s) {
        std::size_t end;
        const auto value = std::stoll(s, &end);
        if (end != s.size()) throw std::runtime_error("Expected integer: " + s);
        return value;
    };
    o.nside = integer(argv[2]);
    bool vertex = false, direction = false, opening = false, slab_option = false;
    bool radial = false, layout_set = false, observer_set = false, scalar_set = false;
    double opening_deg = 180;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i+1 >= argc) throw std::runtime_error("Missing argument for " + arg);
            return argv[++i];
        };
        if (arg == "--info") o.info = next();
        else if (arg == "--thickness") {
            const auto value = integer(next());
            if (value <= 0 || value > std::numeric_limits<int>::max()/4) throw std::runtime_error("Invalid thickness");
            o.thickness = static_cast<int>(value);
        }
        else if (arg == "--vertex") { for (double& x : o.geometry.vertex) x = number(next()); vertex = true; }
        else if (arg == "--observer-tau") { o.observer_tau = number(next()); observer_set = true; }
        else if (arg == "--geometry") o.geometry_file = next();
        else if (arg == "--scalar") {
            const auto source = next();
            if (source == "phi") o.scalar = lightcone::ScalarSource::Phi;
            else if (source == "weyl") o.scalar = lightcone::ScalarSource::Weyl;
            else if (source == "phi-chi") o.scalar = lightcone::ScalarSource::PhiChi;
            else throw std::runtime_error("Scalar source must be phi, weyl or phi-chi");
            scalar_set = true;
        }
        else if (arg == "--target-z") o.target_z = number(next());
        else if (arg == "--time-derivatives") {
            const auto mode = next();
            if (mode != "finite-difference" && mode != "zero")
                throw std::runtime_error("Time derivatives must be finite-difference or zero");
            o.time_derivatives = mode == "zero" ? lightcone::TimeDerivativeMode::Zero : lightcone::TimeDerivativeMode::FiniteDifference;
        }
        else if (arg == "--batch") {
            const auto value = integer(next());
            if (value < 0 || value > 11) throw std::runtime_error("Batch must be in 0..11");
            o.batch = static_cast<int>(value);
        }
        else if (arg == "--ordering") {
            const auto value = next();
            if (value != "nested" && value != "ring") throw std::runtime_error("Ordering must be nested or ring");
            o.ordering = value == "nested" ? lightcone::Ordering::Nested : lightcone::Ordering::Ring;
        }
        else if (arg == "--storage") {
            const auto value = next();
            if (value != "full" && value != "ziggurat") throw std::runtime_error("Storage must be full or ziggurat");
            o.ziggurat = value == "ziggurat";
        }
        else if (arg == "--slab-cells" || arg == "--padding-cells") {
            const auto value = integer(next());
            if (value < (arg == "--slab-cells" ? 1 : 0) || value > std::numeric_limits<int>::max()-4)
                throw std::runtime_error("Invalid slab thickness or padding");
            if (arg == "--slab-cells") o.slab_cells = static_cast<int>(value);
            else o.padding_cells = static_cast<int>(value);
            slab_option = true;
        }
        else if (arg == "--max-lookback-cells") {
            o.max_lookback = number(next());
            if (o.max_lookback <= 0) throw std::runtime_error("Look-back cap must be positive");
        }
        else if (arg == "--b-layout") {
            const auto layout = next();
            if (layout != "staggered" && layout != "collocated") throw std::runtime_error("Unknown B layout: " + layout);
            o.staggered = layout == "staggered";
            layout_set = true;
        }
        else if (arg == "--direction") { for (double& x : o.geometry.direction) x = number(next()); direction = true; }
        else if (arg == "--opening-deg") { opening_deg = number(next()); opening = true; }
        else if (arg == "--radial-range") {
            o.geometry.inner = number(next()); o.geometry.outer = number(next()); o.geometry.radial_limit = true; radial = true;
            if (o.geometry.inner < 0 || o.geometry.outer <= o.geometry.inner) throw std::runtime_error("Require 0 <= rmin < rmax");
        }
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (!o.geometry_file.empty()) {
        const auto& g = o.writer = lightcone::read_geometry_info(o.geometry_file);
        if (vertex || direction || opening || radial)
            throw std::runtime_error("--vertex, --direction, --opening-deg and --radial-range are taken from the geometry file");
        if (o.thickness && o.thickness != g.thickness) throw std::runtime_error("--thickness conflicts with the geometry file");
        if (layout_set && o.staggered != (g.b_collocated == 0)) throw std::runtime_error("--b-layout conflicts with the geometry file");
        if (!scalar_set)
            o.scalar = g.fields & lcg::FIELD_WEYL ? lightcone::ScalarSource::Weyl : lightcone::ScalarSource::PhiChi;
        if (o.scalar == lightcone::ScalarSource::Phi)
            throw std::runtime_error("gasgrd geometry describes raw phi; select --scalar weyl or --scalar phi-chi");
        if (o.scalar == lightcone::ScalarSource::Weyl && !(g.fields & lcg::FIELD_WEYL)) throw std::runtime_error("The light cone has no weyl output");
        if (o.scalar == lightcone::ScalarSource::PhiChi && (g.fields & (lcg::FIELD_PHI|lcg::FIELD_CHI)) != (lcg::FIELD_PHI|lcg::FIELD_CHI))
            throw std::runtime_error("The light cone lacks phi or chi output");
        if (!(g.fields & lcg::FIELD_B)) throw std::runtime_error("The light cone has no B output");
        o.thickness = g.thickness;
        o.staggered = g.b_collocated == 0;
        for (int axis = 0; axis < 3; ++axis) {
            o.geometry.vertex[axis] = g.vertex[axis]; o.geometry.direction[axis] = g.direction[axis];
        }
        o.geometry.opening_cos = g.opening_cos;
        o.geometry.angular_limit = g.opening_cos > -1.0;
        o.geometry.inner = g.distance_inner; o.geometry.outer = g.distance_outer; o.geometry.radial_limit = true;
        o.geometry.convention = lightcone::GeometryConvention::GasgrdV1;
        if (!observer_set) o.observer_tau = g.tauobs;
        else if (o.observer_tau != g.tauobs)
            std::cerr << "Warning: --observer-tau differs from the writer's tauobs " << std::setprecision(17) << g.tauobs << '\n';
        if (g.periodic_overlap || g.distance_outer > 0.5)
            std::cerr << "Warning: the footprint may overlap periodic copies; overwritten data cannot be recovered. Continuing.\n";
        vertex = true;
    }
    if (!lightcone::valid_nside(o.nside)) throw std::runtime_error("Nside must be a power of two in 1..2^29");
    if (o.ziggurat && o.batch < 0) throw std::runtime_error("Ziggurat storage requires --batch");
    if (!o.ziggurat && slab_option) throw std::runtime_error("Slab-specific options require --storage ziggurat");
    if (o.info.empty() || !o.thickness || !vertex || !std::isfinite(o.observer_tau) ||
        !std::isfinite(o.target_z) || o.target_z <= 0) throw std::runtime_error(usage);
    if (o.thickness == 1 && o.time_derivatives != lightcone::TimeDerivativeMode::Zero)
        throw std::runtime_error("Thickness 1 requires explicit --time-derivatives zero");
    if (opening != direction) throw std::runtime_error("Supply --direction and --opening-deg together");
    if (opening_deg <= 0 || opening_deg > 180) throw std::runtime_error("Opening angle must be in (0,180]");
    if (direction) {
        const auto& d = o.geometry.direction;
        const double norm = lcg::sqrt(lcg::add(lcg::add(lcg::mul(d[0],d[0]),lcg::mul(d[1],d[1])),lcg::mul(d[2],d[2])));
        if (!std::isfinite(norm) || norm == 0.0) throw std::runtime_error("Direction must be a finite nonzero vector");
        for (double& x : o.geometry.direction) x /= norm;
        o.geometry.angular_limit = opening_deg < 180;
        o.geometry.opening_cos = std::cos(opening_deg*3.14159265358979323846/180.0);
    }
    return o;
}

__global__ void trace_kernel(lightcone::View view, double observer_tau, double target_z,
                             std::int64_t count, const double* theta, const double* phi,
                             float* maps, int* statuses, std::int64_t start, int* first_miss, lightcone::MissDiagnostic* missed) {
    const auto i = static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= count) return;
    lightcone::MissDiagnostic diagnostic;
    view.miss = &diagnostic;
    double y[lightcone::STATE_SIZE], reached_z = NAN;
    auto status = lightcone::initialize(view, observer_tau, theta[i], phi[i], y);
    if (status == lightcone::Status::Ok) status = lightcone::trace(view, target_z, y, reached_z);
    float result[6];
    if (status == lightcone::Status::Ok) lightcone::observables(view, y, theta[i], phi[i], result);
    else for (float& value : result) value = nanf("");
    for (int map = 0; map < 6; ++map) maps[static_cast<std::int64_t>(map)*count+i] = result[map];
    maps[6*count+i] = static_cast<float>(reached_z);
    maps[7*count+i] = status == lightcone::Status::Ok ? static_cast<float>(y[lightcone::STATE_ISW]) : nanf("");
    statuses[i] = static_cast<int>(status);
    if (status == lightcone::Status::NotResident && atomicCAS(first_miss, 0, 1) == 0) {
        diagnostic.nested_pixel = start+i;
        *missed = diagnostic;
    }
}

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }

int run(const Options& o) {
    std::cout << "Shell geometry: " << lightcone::geometry_convention_name(o.geometry.convention) << '\n';
    if (o.geometry_file.empty())
        std::cerr << "Warning: no geometry sidecar supplied; using the legacy writer's wrapped-position/shifted-vertex convention "
                     "with separately rounded products and sums. Geometry parameters and original compiler arithmetic cannot be fully verified. "
                     "Supply the original geometry at full precision.\n";
    const auto metadata = lightcone::build_metadata(lightcone::read_info(o.info, o.thickness), o.thickness, o.geometry);
    auto view = lightcone::host_view(metadata, 1, o.geometry, nullptr);
    double a_obs;
    if (lightcone::background_at(view, o.observer_tau, a_obs) != lightcone::Status::Ok)
        throw std::runtime_error("Observation time is outside the sidecar background table; extrapolation is disabled");
    lightcone::Reader reader(o.base, o.thickness, o.scalar);
    const int n = reader.grid_size();
    if (!o.geometry_file.empty()) lightcone::require_geometry_grid(o.writer, n);
    const int extent[3] = {n, n, n};
    const std::size_t full_voxels = lightcone::checked_volume(extent);
    const double lookback = std::min((o.observer_tau-metadata.times.front())*n, o.max_lookback);
    const double lower_tau = std::max(metadata.times.front(), o.observer_tau-lookback/n);
    const double effective_lookback = (o.observer_tau-lower_tau)*n;
    std::cout << "Grid: " << n << "^3; thickness: " << o.thickness << "; records: " << metadata.records.size()
              << "; tau range: " << metadata.times.front() << " .. " << metadata.times.back()
              << "; a_obs: " << a_obs << "; effective look-back cells: " << effective_lookback << '\n';
    std::cout << "Time derivatives: " << lightcone::time_derivative_name(o.time_derivatives)
              << "; scalar source: " << lightcone::scalar_source_name(o.scalar) << '\n';
    if (!o.geometry.angular_limit || !o.geometry.radial_limit)
        std::cerr << "Warning: undeclared footprint limits are assumed covered; unrecorded holes cannot be detected.\n";
    double max_radius = 0;
    for (const auto& r : metadata.records) max_radius = std::max(max_radius, r.boundaries.back());
    double nearest_face = 1;
    for (double x : o.geometry.vertex) nearest_face = std::min(nearest_face, std::min(x, 1-x));
    if (max_radius+4.0/n > nearest_face)
        std::cerr << "Warning: the footprint/stencil may cross periodic boundaries. Self-overlap can overwrite data and cannot be repaired.\n";
    const std::int64_t npix = 12*o.nside*o.nside;
    const std::int64_t count = o.batch < 0 ? npix : o.nside*o.nside;
    const std::int64_t start = o.batch < 0 ? 0 : o.batch*count;
    lightcone::checked_multiply(static_cast<std::size_t>(npix), sizeof(float));
    const auto geometry_start = Clock::now();
    std::vector<double> theta, phi;
    lightcone::nested_angles(o.nside, o.batch, theta, phi);
    lightcone::SlabLayout layout;
    if (o.ziggurat) layout = lightcone::build_ziggurat(n, o.geometry, o.batch, theta, phi,
                                                     effective_lookback, o.slab_cells, o.padding_cells);
    const auto geometry_seconds = seconds(geometry_start);
    const std::size_t voxels = o.ziggurat ? layout.voxels : full_voxels;
    const std::size_t largest = o.ziggurat ? layout.largest_slab : full_voxels;
    std::size_t auxiliary = lightcone::checked_multiply(metadata.times.size(), sizeof(double));
    auxiliary = lightcone::checked_add(auxiliary, lightcone::checked_multiply(metadata.background.size(), sizeof(lightcone::BackgroundSegment)));
    auxiliary = lightcone::checked_add(auxiliary, lightcone::checked_multiply(metadata.shells.size(), sizeof(lightcone::RadialSegment)));
    auxiliary = lightcone::checked_add(auxiliary, lightcone::checked_multiply(metadata.slot_offsets.size(), sizeof(int)));
    auxiliary = lightcone::checked_add(auxiliary, lightcone::checked_multiply(4ULL*o.thickness, sizeof(lightcone::GridView)));
    auxiliary = lightcone::checked_add(auxiliary, lightcone::checked_multiply(layout.slabs.size(), sizeof(lightcone::SlabDescriptor)));
    auxiliary = lightcone::checked_add(auxiliary, sizeof(int)+sizeof(lightcone::MissDiagnostic));
    auto estimate = lightcone::estimate_storage(voxels, largest, o.thickness, count, auxiliary);
    if (o.scalar == lightcone::ScalarSource::PhiChi)
        estimate.field_staging_bytes = lightcone::checked_multiply(largest, 2*sizeof(float));
    std::size_t free_bytes, total_bytes;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    lightcone::require_storage_memory(estimate, free_bytes);
    std::cout << "Storage: " << (o.ziggurat ? "ziggurat" : "full") << "; full-volume voxels: " << full_voxels
              << "; resident voxels: " << voxels << "; slabs: " << layout.slabs.size()
              << "; field bytes: " << estimate.field_bytes << "; estimated device bytes: " << estimate.device_bytes
              << "; peak field-staging bytes: " << estimate.field_staging_bytes << '\n';
    const auto loading_start = Clock::now();
    DeviceBuffer<double> times(metadata.times);
    DeviceBuffer<lightcone::BackgroundSegment> background(metadata.background);
    DeviceBuffer<lightcone::RadialSegment> shells(metadata.shells);
    DeviceBuffer<int> offsets(metadata.slot_offsets);
    DeviceBuffer<lightcone::SlabDescriptor> descriptors(layout.slabs);
    std::vector<DeviceBuffer<float>> potentials;
    std::vector<DeviceBuffer<std::uint16_t>> vectors;
    std::vector<lightcone::GridView> grids;
    const std::int64_t origin[3] = {0, 0, 0};
    for (int field = 0; field < 4; ++field) for (int slot = 0; slot < o.thickness; ++slot) {
        std::cout << "Loading " << reader.filename(field, slot) << '\n';
        auto grid = lightcone::float_grid(nullptr, n, field, o.staggered);
        if (field == 0) { potentials.emplace_back(voxels); grid.values = potentials.back().data; }
        else { vectors.emplace_back(voxels); grid.half_values = vectors.back().data; }
        const std::size_t regions = o.ziggurat ? layout.slabs.size() : 1;
        for (std::size_t region = 0; region < regions; ++region) {
            const auto* read_origin = o.ziggurat ? layout.slabs[region].origin : origin;
            const auto* read_extent = o.ziggurat ? layout.slabs[region].extent : extent;
            const auto offset = o.ziggurat ? layout.slabs[region].offset : 0;
            auto values = reader.read_region(field, slot, read_origin, read_extent);
            if (field == 0) cuda_check(cudaMemcpy(potentials.back().data+offset, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice));
            else {
                std::vector<std::uint16_t> packed(values.size());
                #pragma omp parallel for
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const __half_raw value = __float2half(values[i]*134217728.0f);
                    packed[i] = value.x;
                }
                cuda_check(cudaMemcpy(vectors.back().data+offset, packed.data(), packed.size()*sizeof(std::uint16_t), cudaMemcpyHostToDevice));
            }
        }
        grids.push_back(grid);
    }
    DeviceBuffer<lightcone::GridView> device_grids(grids);
    view = {n, o.thickness, static_cast<int>(metadata.times.size()), o.geometry,
            times.data, background.data, shells.data, offsets.data, device_grids.data};
    view.layout = layout.view(); view.layout.slabs = descriptors.data;
    view.trace_tau_min = lower_tau;
    view.time_derivatives = o.time_derivatives;
    DeviceBuffer<double> device_theta(theta), device_phi(phi);
    DeviceBuffer<float> maps(lightcone::checked_multiply(count, lightcone::MAP_COUNT));
    DeviceBuffer<int> device_status(count), first_miss(1);
    DeviceBuffer<lightcone::MissDiagnostic> missed(1);
    cuda_check(cudaMemset(first_miss.data, 0, sizeof(int)));
    const auto loading_seconds = seconds(loading_start);
    const std::int64_t blocks = (count+63)/64;
    if (blocks > std::numeric_limits<int>::max()) throw std::runtime_error("Too many rays for a single launch; use batching");
    const auto tracing_start = Clock::now();
    trace_kernel<<<static_cast<unsigned>(blocks), 64>>>(view, o.observer_tau, o.target_z, count,
            device_theta.data, device_phi.data, maps.data, device_status.data, start, first_miss.data, missed.data);
    cuda_check(cudaGetLastError()); cuda_check(cudaDeviceSynchronize());
    const auto tracing_seconds = seconds(tracing_start);
    std::vector<int> statuses(count);
    cuda_check(cudaMemcpy(statuses.data(), device_status.data, statuses.size()*sizeof(int), cudaMemcpyDeviceToHost));
    std::array<std::uint64_t, static_cast<int>(lightcone::Status::Count)> totals{};
    for (int status : statuses) {
        if (status < 0 || status >= static_cast<int>(totals.size())) throw std::runtime_error("Invalid device status");
        ++totals[status];
    }
    for (int s = 0; s < static_cast<int>(totals.size()); ++s)
        std::cout << lightcone::status_name(static_cast<lightcone::Status>(s)) << ": " << totals[s] << '\n';
    std::cout << "Requested source payload bytes: " << reader.bytes_read() << "; geometry seconds: " << geometry_seconds
              << "; loading seconds: " << loading_seconds << "; tracing seconds: " << tracing_seconds << '\n';
    std::ofstream metrics("run-metrics.json");
    if (!metrics) throw std::runtime_error("Cannot write run-metrics.json");
    metrics << std::setprecision(17) << "{\"storage\":\"" << (o.ziggurat ? "ziggurat" : "full")
            << "\",\"static_shear_transport\":\"negative"
            << "\",\"time_derivatives\":\"" << lightcone::time_derivative_name(o.time_derivatives)
            << "\",\"shell_geometry\":\"" << lightcone::geometry_convention_name(o.geometry.convention)
            << "\",\"scalar_source\":\"" << lightcone::scalar_source_name(o.scalar)
            << "\",\"ordering\":\"" << (o.ordering == lightcone::Ordering::Nested ? "NESTED" : "RING")
            << "\",\"full_voxels\":" << full_voxels << ",\"resident_voxels\":" << voxels
            << ",\"slabs\":" << layout.slabs.size() << ",\"field_bytes\":" << estimate.field_bytes
            << ",\"device_bytes\":" << estimate.device_bytes << ",\"field_staging_bytes\":" << estimate.field_staging_bytes
            << ",\"source_payload_bytes\":" << reader.bytes_read() << ",\"geometry_seconds\":" << geometry_seconds
            << ",\"loading_seconds\":" << loading_seconds << ",\"tracing_seconds\":" << tracing_seconds
            << ",\"effective_lookback_cells\":" << effective_lookback << ",\"residency_misses\":"
            << totals[static_cast<int>(lightcone::Status::NotResident)] << "}\n";
    metrics.close();
    if (!metrics) throw std::runtime_error("Cannot finish run-metrics.json");
    if (o.ziggurat && totals[static_cast<int>(lightcone::Status::NotResident)]) {
        lightcone::MissDiagnostic miss;
        cuda_check(cudaMemcpy(&miss, missed.data, sizeof(miss), cudaMemcpyDeviceToHost));
        std::cerr << "Missing resident node: (" << miss.node[0] << ',' << miss.node[1] << ',' << miss.node[2]
                  << "), NESTED pixel=" << miss.nested_pixel << ", field=" << miss.field << ", slot=" << miss.slot
                  << ", tau=" << std::setprecision(17) << miss.tau << '\n';
        throw std::runtime_error("Ziggurat residency failure; no FITS maps written. Increase --padding-cells or check the look-back cap. No automatic retry was performed.");
    }
    std::vector<float> output(npix, 0.0f);
    for (int map = 0; map < lightcone::MAP_COUNT; ++map) {
        cuda_check(cudaMemcpy(output.data()+start, maps.data+map*count, count*sizeof(float), cudaMemcpyDeviceToHost));
        const int output_batch = o.batch < 0 ? 0 : o.batch;
        const std::string filename = std::string(lightcone::MAP_NAMES[map])+"_batch"+std::to_string(output_batch)+".fits";
        lightcone::write_nested_map(output, o.nside, o.ordering, filename);
    }
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { std::cout << usage; return 0; }
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "lightcone-raytracer: " << error.what() << '\n';
        return 1;
    }
}
