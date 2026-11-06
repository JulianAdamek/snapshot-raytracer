#include "lightcone.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <hdf5.h>

namespace lightcone {
const char* time_derivative_name(TimeDerivativeMode mode) {
    return mode == TimeDerivativeMode::Zero ? "zero" : "finite-difference";
}
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
struct H5Handle {
    hid_t id;
    herr_t (*close)(hid_t);
    H5Handle(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer) {
        require(id >= 0, "HDF5 operation failed");
    }
    ~H5Handle() { close(id); }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
};
void validate_records(const std::vector<InfoRecord>& records, int thickness) {
    require(thickness > 0 && !records.empty(), "Info table needs records and positive thickness");
    require(records.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Too many info records");
    for (std::size_t k = 0; k < records.size(); ++k) {
        const auto& r = records[k];
        require(std::isfinite(r.tau) && std::isfinite(r.a) && r.a > 0.0 &&
                std::isfinite(r.particle_inner) && std::isfinite(r.particle_outer) &&
                r.particle_inner < r.particle_outer, "Invalid info values at record " + std::to_string(k));
        require(r.boundaries.size() == static_cast<std::size_t>(thickness) + 1,
                "Info thickness does not match boundaries");
        if (k) require(r.cycle > records[k-1].cycle && r.tau > records[k-1].tau,
                       "Info cycles and times must increase strictly (native byte order required)");
        for (std::size_t j = 0; j < r.boundaries.size(); ++j)
            require(std::isfinite(r.boundaries[j]) && (!j || r.boundaries[j] > r.boundaries[j-1]),
                    "Metric shell boundaries must be finite and strictly ordered");
    }
}
double endpoint_slope(double h0, double h1, double d0, double d1) {
    double m = ((2*h0+h1)*d0-h0*d1)/(h0+h1);
    if (m*d0 <= 0.0) return 0.0;
    if (d0*d1 <= 0.0 && std::fabs(m) > 3*std::fabs(d0)) return 3*d0;
    return m;
}
std::size_t volume(const int extent[3]) {
    std::size_t size = 1;
    for (int i = 0; i < 3; ++i) {
        require(extent[i] > 0 && size <= std::numeric_limits<std::size_t>::max()/extent[i],
                "Invalid or overflowing region dimensions");
        size *= extent[i];
    }
    require(size <= std::numeric_limits<std::size_t>::max()/sizeof(float), "Region byte count overflows");
    return size;
}
}

const char* status_name(Status s) {
    switch (s) {
        case Status::Ok: return "success";
        case Status::SpatialCoverage: return "missing spatial coverage";
        case Status::TimeCoverage: return "missing time bracket";
        case Status::BackgroundCoverage: return "outside background table";
        case Status::NotResident: return "nonresident storage";
        case Status::InvalidValue: return "nonfinite field or ray state";
        case Status::TargetNotReached: return "target not reached within time/refinement limits";
        default: return "unknown status";
    }
}

std::vector<InfoRecord> read_info(const std::string& path, int thickness) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 && sizeof(std::int32_t) == 4, "Legacy info requires IEEE doubles and int32");
    require(thickness > 0, "--thickness must be positive");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(file), "Cannot open info file: " + path);
    const auto bytes = file.tellg();
    const std::uint64_t width = 4 + 8*(static_cast<std::uint64_t>(thickness)+5);
    require(bytes > 0 && static_cast<std::uint64_t>(bytes) % width == 0,
            "Empty/truncated info file or wrong thickness: " + path);
    require(static_cast<std::uint64_t>(bytes)/width < static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            "Too many info records");
    file.seekg(0);
    std::vector<InfoRecord> records;
    for (std::uint64_t k = 0; k < static_cast<std::uint64_t>(bytes)/width; ++k) {
        InfoRecord r{};
        auto read = [&](auto& value) { file.read(reinterpret_cast<char*>(&value), sizeof(value)); };
        read(r.cycle); read(r.tau); read(r.a); read(r.particle_inner); read(r.particle_outer);
        r.boundaries.resize(static_cast<std::size_t>(thickness)+1);
        for (double& boundary : r.boundaries) read(boundary);
        require(static_cast<bool>(file), "Read error in info file: " + path);
        records.push_back(std::move(r));
    }
    validate_records(records, thickness);
    return records;
}

Metadata build_metadata(std::vector<InfoRecord> records, int thickness, const Geometry& geometry) {
    validate_records(records, thickness);
    require(records.size() >= 2, "Background interpolation requires at least two info records");
    Metadata result;
    result.thickness = thickness;
    result.records = std::move(records);
    const auto& rs = result.records;
    const int count = static_cast<int>(rs.size());
    std::vector<double> h(count-1), d(count-1), slopes(count);
    for (const auto& r : rs) result.times.push_back(r.tau);
    for (int k = 0; k < count-1; ++k) {
        h[k] = rs[k+1].tau-rs[k].tau;
        d[k] = (rs[k+1].a-rs[k].a)/h[k];
        require(std::isfinite(d[k]), "Background slope overflows");
    }
    if (count == 2) slopes[0] = slopes[1] = d[0];
    else {
        slopes[0] = endpoint_slope(h[0], h[1], d[0], d[1]);
        slopes[count-1] = endpoint_slope(h[count-2], h[count-3], d[count-2], d[count-3]);
        for (int k = 1; k < count-1; ++k) {
            if (d[k-1]*d[k] <= 0.0) slopes[k] = 0.0;
            else {
                const double w1 = 2*h[k]+h[k-1], w2 = h[k]+2*h[k-1];
                slopes[k] = (w1+w2)/(w1/d[k-1]+w2/d[k]);
            }
        }
    }
    for (int k = 0; k < count-1; ++k) {
        const double delta = rs[k+1].a-rs[k].a;
        BackgroundSegment b{rs[k].tau, rs[k+1].tau, rs[k].a, rs[k+1].a,
                            h[k]*slopes[k], 3*delta-h[k]*(2*slopes[k]+slopes[k+1]),
                            -2*delta+h[k]*(slopes[k]+slopes[k+1])};
        require(std::isfinite(b.c1) && std::isfinite(b.c2) && std::isfinite(b.c3),
                "Background spline coefficients overflow");
        result.background.push_back(b);
    }
    struct Event { double radius; int record; bool begin; };
    for (int slot = 0; slot < thickness; ++slot) {
        require(result.shells.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()), "Too many shell intervals");
        result.slot_offsets.push_back(static_cast<int>(result.shells.size()));
        std::vector<Event> events;
        for (int k = 0; k < count; ++k) {
            const auto& s = rs[k].boundaries;
            // Exact legacy writer gates: limits are applied to shell outer edges,
            // rather than clipping every voxel to the declared distance interval.
            if (s.back() <= 0.0 || s[slot+1] <= 0.0) continue;
            if (geometry.radial_limit &&
                (geometry.outer <= s[1] || geometry.inner > s.back() ||
                 geometry.outer <= s[slot+1] || geometry.inner > s[slot+1])) continue;
            events.push_back({s[slot], k, true});
            events.push_back({s[slot+1], k, false});
        }
        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) { return a.radius < b.radius; });
        std::vector<bool> active(count, false);
        std::priority_queue<int> newest;
        for (std::size_t p = 0; p < events.size();) {
            const double radius = events[p].radius;
            do {
                active[events[p].record] = events[p].begin;
                if (events[p].begin) newest.push(events[p].record);
                ++p;
            } while (p < events.size() && events[p].radius == radius);
            while (!newest.empty() && !active[newest.top()]) newest.pop();
            if (p == events.size() || newest.empty()) continue;
            const RadialSegment segment{radius, events[p].radius, newest.top()};
            if (result.shells.size() > static_cast<std::size_t>(result.slot_offsets.back()) &&
                result.shells.back().record == segment.record && result.shells.back().outer == radius)
                result.shells.back().outer = segment.outer;
            else result.shells.push_back(segment);
        }
    }
    require(result.shells.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()), "Too many shell intervals");
    result.slot_offsets.push_back(static_cast<int>(result.shells.size()));
    return result;
}

struct Reader::Impl {
    hid_t file = -1, data = -1, space = -1;
    int field = -1, slot = -1;
    std::size_t source_bytes = 0;
    std::uint64_t bytes = 0;
    void reset() {
        if (space >= 0) H5Sclose(space);
        if (data >= 0) H5Dclose(data);
        if (file >= 0) H5Fclose(file);
        file = data = space = -1;
        field = slot = -1;
    }
    ~Impl() { reset(); }
};
Reader::~Reader() = default;
std::uint64_t Reader::bytes_read() const { return impl_->bytes; }
Reader::Reader(std::string base, int thickness) : impl_(std::make_unique<Impl>()), base_(std::move(base)), thickness_(thickness) {
    require(thickness > 0, "--thickness must be positive");
    for (int field = 0; field < 4; ++field) for (int slot = 0; slot < thickness; ++slot) {
        const auto path = filename(field, slot);
        H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
        H5Handle data(H5Dopen2(file.id, "/data", H5P_DEFAULT), H5Dclose);
        H5Handle space(H5Dget_space(data.id), H5Sclose);
        H5Handle type(H5Dget_type(data.id), H5Tclose);
        require(H5Tget_class(type.id) == H5T_FLOAT && (H5Tget_size(type.id) == 4 || H5Tget_size(type.id) == 8),
                "Expected scalar float32/float64 /data: " + path);
        require(H5Sget_simple_extent_ndims(space.id) == 3, "Expected rank-3 /data: " + path);
        hsize_t shape[3];
        require(H5Sget_simple_extent_dims(space.id, shape, nullptr) >= 0, "Cannot inspect /data dimensions");
        require(shape[0] > 0 && shape[0] <= static_cast<hsize_t>(std::numeric_limits<int>::max()) &&
                shape[0] == shape[1] && shape[1] == shape[2], "Expected a nonempty cubic grid: " + path);
        if (!n_) n_ = static_cast<int>(shape[0]);
        require(shape[0] == static_cast<hsize_t>(n_), "Inconsistent grid sizes: " + path);
    }
    const int extent[3] = {n_, n_, n_};
    volume(extent);
}
std::string Reader::filename(int field, int slot) const {
    require(field >= 0 && field < 4 && slot >= 0 && slot < thickness_, "Invalid field or slot");
    return base_ + (field == 0 ? "_phi" : "_B" + std::to_string(field)) +
           (thickness_ == 1 ? "" : "_" + std::to_string(slot)) + ".h5";
}
std::vector<float> Reader::read_region(int field, int slot, const std::int64_t origin[3], const int extent[3]) const {
    std::vector<float> result(volume(extent));
    if (impl_->field != field || impl_->slot != slot) {
        impl_->reset();
        impl_->file = H5Fopen(filename(field, slot).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        require(impl_->file >= 0, "Cannot open HDF5 input");
        impl_->data = H5Dopen2(impl_->file, "/data", H5P_DEFAULT);
        require(impl_->data >= 0, "Cannot open HDF5 /data");
        impl_->space = H5Dget_space(impl_->data);
        require(impl_->space >= 0, "Cannot inspect HDF5 /data");
        H5Handle type(H5Dget_type(impl_->data), H5Tclose);
        impl_->source_bytes = H5Tget_size(type.id);
        impl_->field = field; impl_->slot = slot;
    }
    const hsize_t memory_shape[3] = {static_cast<hsize_t>(extent[2]), static_cast<hsize_t>(extent[1]), static_cast<hsize_t>(extent[0])};
    H5Handle memory_space(H5Screate_simple(3, memory_shape, nullptr), H5Sclose);
    struct Span { hsize_t file, memory, length; };
    std::vector<Span> spans[3];
    for (int axis = 0; axis < 3; ++axis) {
        std::int64_t wrapped = (origin[axis] % n_ + n_) % n_;
        for (int offset = 0; offset < extent[axis];) {
            const int length = std::min(n_-static_cast<int>(wrapped), extent[axis]-offset);
            spans[axis].push_back({static_cast<hsize_t>(wrapped), static_cast<hsize_t>(offset), static_cast<hsize_t>(length)});
            offset += length;
            wrapped = 0;
        }
    }
    for (const auto& z : spans[2]) for (const auto& y : spans[1]) for (const auto& x : spans[0]) {
        const hsize_t start[3] = {z.file, y.file, x.file};
        const hsize_t output[3] = {z.memory, y.memory, x.memory};
        const hsize_t count[3] = {z.length, y.length, x.length};
        require(H5Sselect_hyperslab(impl_->space, H5S_SELECT_SET, start, nullptr, count, nullptr) >= 0 &&
                H5Sselect_hyperslab(memory_space.id, H5S_SELECT_SET, output, nullptr, count, nullptr) >= 0,
                "Cannot select HDF5 subarray");
        require(H5Dread(impl_->data, H5T_NATIVE_FLOAT, memory_space.id, impl_->space, H5P_DEFAULT, result.data()) >= 0,
                "Cannot read HDF5 subarray");
    }
    const auto bytes = checked_multiply(result.size(), impl_->source_bytes);
    impl_->bytes = checked_add(impl_->bytes, bytes);
    return result;
}
View host_view(const Metadata& m, int n, const Geometry& geometry, const GridView* grids) {
    View view{n, m.thickness, static_cast<int>(m.times.size()), geometry, m.times.data(),
              m.background.data(), m.shells.data(), m.slot_offsets.data(), grids, {}};
    view.trace_tau_min = m.times.front();
    return view;
}
GridView float_grid(const float* values, int n, int field, bool staggered) {
    GridView grid;
    grid.values = values;
    for (int axis = 0; axis < 3; ++axis) grid.extent[axis] = n;
    grid.strides[1] = n;
    grid.strides[2] = static_cast<std::size_t>(n)*n;
    if (staggered && field) grid.node_offset[field-1] = 0.5;
    return grid;
}
} // namespace lightcone
