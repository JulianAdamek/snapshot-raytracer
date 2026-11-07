#pragma once

#include "lccube_geometry.h" // Vendored gasgrd convention 1; no external checkout required.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define LC_HD __host__ __device__
#else
#define LC_HD
#endif

namespace lightcone {

enum class TimeDerivativeMode { FiniteDifference, Zero };
const char* time_derivative_name(TimeDerivativeMode mode);
constexpr int MAP_COUNT = 8;
inline constexpr const char* MAP_NAMES[MAP_COUNT] = {
    "dAmap", "ellipticity1map", "ellipticity2map", "rotationmap",
    "deflection1map", "deflection2map", "zmap", "iswmap"
};
struct TimeSample {
    float partial_time = 0; // partial / partial eta, eta = N * tau/L
    float gradient[3] = {0, 0, 0}; // partial_q partial_eta, q = N * x/L
};

enum class Status : int {
    Ok, SpatialCoverage, TimeCoverage, BackgroundCoverage, NotResident,
    InvalidValue, TargetNotReached, Count
};
const char* status_name(Status status);

enum class GeometryConvention { Legacy, GasgrdV1 };
const char* geometry_convention_name(GeometryConvention convention);
struct Geometry {
    double vertex[3] = {0.0, 0.0, 0.0};
    double direction[3] = {0.0, 0.0, 1.0};
    double opening_cos = -1.0;
    bool angular_limit = false;
    bool radial_limit = false;
    double inner = 0.0;
    double outer = 0.0;
    GeometryConvention convention = GeometryConvention::Legacy;
};
lcg::GeometryInfo read_geometry_info(const std::string& path);
void require_geometry_grid(const lcg::GeometryInfo& info, int n);

struct InfoRecord {
    std::int32_t cycle;
    double tau, a, particle_inner, particle_outer;
    std::vector<double> boundaries;
};
struct RadialSegment {
    double inner, outer;
    int record;
};
struct BackgroundSegment {
    double tau0, tau1, a0, a1, c1, c2, c3;
};
struct Metadata {
    int thickness = 0;
    std::vector<InfoRecord> records;
    std::vector<double> times;
    std::vector<BackgroundSegment> background;
    std::vector<RadialSegment> shells;
    std::vector<int> slot_offsets;
};

std::vector<InfoRecord> read_info(const std::string& path, int thickness);
Metadata build_metadata(std::vector<InfoRecord> records, int thickness,
                        const Geometry& geometry);

// Field zero is always the Weyl potential; Phi preserves the legacy filename.
enum class ScalarSource { Phi, Weyl, PhiChi };
const char* scalar_source_name(ScalarSource source);
constexpr int CHI_FILE = 4;

// Logical XYZ coordinates. The result is contiguous with X varying fastest.
// An origin outside the cube is allowed; periodic reads are split into hyperslabs.
class Reader {
public:
    Reader(std::string base, int thickness, ScalarSource scalar = ScalarSource::Phi);
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    int grid_size() const { return n_; }
    int thickness() const { return thickness_; }
    std::string filename(int field, int slot) const;
    std::vector<float> read_region(int field, int slot,
                                  const std::int64_t origin[3], const int extent[3]) const;
    std::uint64_t bytes_read() const;
private:
    std::vector<float> read_file_region(int file, int slot,
                                        const std::int64_t origin[3], const int extent[3]) const;
    int file_count() const { return scalar_ == ScalarSource::PhiChi ? 5 : 4; }
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string base_;
    int thickness_, n_ = 0;
    ScalarSource scalar_ = ScalarSource::Phi;
};

// Origin/extent describe residency independently of the physical periodic cube.
// Half values encode B * 2^27, matching snapshot-raytracer's device storage.
struct GridView {
    const float* values = nullptr;
    const std::uint16_t* half_values = nullptr;
    std::int64_t origin[3] = {0, 0, 0};
    int extent[3] = {0, 0, 0};
    std::size_t strides[3] = {1, 0, 0};
    double node_offset[3] = {0.0, 0.0, 0.0};
};
struct SlabDescriptor {
    std::int64_t core_begin = 0, core_end = 0;
    std::int64_t origin[3] = {0, 0, 0};
    int extent[3] = {0, 0, 0};
    std::size_t strides[3] = {1, 0, 0};
    std::size_t offset = 0;
};
struct SlabLayoutView {
    int axis = 0, slab_cells = 0, count = 0;
    std::int64_t begin = 0, end = 0;
    const SlabDescriptor* slabs = nullptr;
};
struct SlabLayout {
    int axis = 0, slab_cells = 0;
    std::int64_t begin = 0, end = 0;
    std::size_t voxels = 0, largest_slab = 0;
    std::vector<SlabDescriptor> slabs;
    SlabLayoutView view() const;
};
struct MissDiagnostic {
    bool valid = false;
    int field = -1, slot = -1;
    std::int64_t node[3] = {0, 0, 0};
    std::int64_t nested_pixel = -1;
    double tau = 0;
};
struct View {
    int n = 0, thickness = 0, record_count = 0;
    Geometry geometry;
    const double* times = nullptr;
    const BackgroundSegment* background = nullptr;
    const RadialSegment* shells = nullptr;
    const int* slot_offsets = nullptr;
    const GridView* grids = nullptr; // field * thickness + slot; phi, B1, B2, B3
    SlabLayoutView layout;
    double trace_tau_min = -1.7976931348623157e308;
    MissDiagnostic* miss = nullptr; // thread-local diagnostic; never shared between rays
    TimeDerivativeMode time_derivatives = TimeDerivativeMode::FiniteDifference;
};
View host_view(const Metadata& metadata, int n, const Geometry& geometry,
               const GridView* grids);
GridView float_grid(const float* values, int n, int field, bool staggered);

SlabLayout build_ziggurat(int n, const Geometry& geometry, int face,
                         const std::vector<double>& theta, const std::vector<double>& phi,
                         double max_lookback, int slab_cells = 16, int padding_cells = 8);
std::size_t checked_add(std::size_t a, std::size_t b);
std::size_t checked_multiply(std::size_t a, std::size_t b);
std::size_t checked_volume(const int extent[3]);
struct StorageEstimate {
    std::size_t field_bytes, device_bytes, field_staging_bytes;
};
StorageEstimate estimate_storage(std::size_t voxels, std::size_t largest_slab,
                                 int thickness, std::size_t rays, std::size_t auxiliary_bytes);
void require_storage_memory(const StorageEstimate& estimate, std::size_t available_bytes);

} // namespace lightcone
