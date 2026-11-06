#include "lightcone.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightcone {
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max()-b) throw std::runtime_error("Storage size addition overflows");
    return a+b;
}
std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b) throw std::runtime_error("Storage size multiplication overflows");
    return a*b;
}
std::size_t checked_volume(const int extent[3]) {
    std::size_t size = 1;
    for (int i = 0; i < 3; ++i) {
        if (extent[i] <= 0) throw std::runtime_error("Nonpositive storage extent");
        size = checked_multiply(size, static_cast<std::size_t>(extent[i]));
    }
    return size;
}
StorageEstimate estimate_storage(std::size_t voxels, std::size_t largest, int thickness,
                                 std::size_t rays, std::size_t auxiliary) {
    if (thickness <= 0) throw std::runtime_error("Invalid storage thickness");
    StorageEstimate result{};
    result.field_bytes = checked_multiply(checked_multiply(voxels, thickness), 10); // float phi + three half B
    const auto ray_bytes = checked_multiply(rays, 2*sizeof(double)+MAP_COUNT*sizeof(float)+sizeof(int));
    result.device_bytes = checked_add(checked_add(result.field_bytes, ray_bytes), auxiliary);
    result.field_staging_bytes = checked_multiply(largest, sizeof(float)+sizeof(std::uint16_t));
    return result;
}
void require_storage_memory(const StorageEstimate& estimate, std::size_t available_bytes) {
    if (estimate.device_bytes > available_bytes)
        throw std::runtime_error("Resident storage requires "+std::to_string(estimate.device_bytes)+
                                 " bytes; GPU has "+std::to_string(available_bytes)+
                                 " bytes free. Reduce the look-back cap or resolution.");
}
SlabLayoutView SlabLayout::view() const {
    return {axis, slab_cells, static_cast<int>(slabs.size()), begin, end, slabs.data()};
}
namespace {
struct Bounds {
    double lo[3] = {INFINITY, INFINITY, INFINITY};
    double hi[3] = {-INFINITY, -INFINITY, -INFINITY};
};
std::int64_t rounded_down(double value) {
    if (!std::isfinite(value) || std::fabs(value) > 4e18) throw std::runtime_error("Slab coordinate outside supported integer range");
    return static_cast<std::int64_t>(std::floor(value));
}
}
SlabLayout build_ziggurat(int n, const Geometry& geometry, int face,
                         const std::vector<double>& theta, const std::vector<double>& phi,
                         double max_lookback, int slab_cells, int padding_cells) {
    if (n <= 0 || face < 0 || face > 11 || theta.empty() || theta.size() != phi.size() ||
        !std::isfinite(max_lookback) || max_lookback < 0 || slab_cells <= 0 || padding_cells < 0)
        throw std::runtime_error("Invalid ziggurat geometry or dimensions");
    SlabLayout result;
    result.axis = face < 4 || face >= 8 ? 2 : (face % 2 == 0 ? 0 : 1);
    result.slab_cells = slab_cells;
    const int axis = result.axis;
    const double margin = 4.0+padding_cells;
    double vertex[3];
    for (int a = 0; a < 3; ++a) {
        vertex[a] = n*geometry.vertex[a];
        rounded_down(vertex[a]);
    }
    double low = vertex[axis], high = vertex[axis];
    for (std::size_t i = 0; i < theta.size(); ++i) {
        if (!std::isfinite(theta[i]) || !std::isfinite(phi[i])) throw std::runtime_error("Nonfinite launch direction");
        const double u[3] = {std::sin(theta[i])*std::cos(phi[i]), std::sin(theta[i])*std::sin(phi[i]), std::cos(theta[i])};
        low = std::min(low, vertex[axis]+max_lookback*u[axis]);
        high = std::max(high, vertex[axis]+max_lookback*u[axis]);
    }
    result.begin = rounded_down(low-margin);
    result.end = rounded_down(high+margin)+1;
    const auto span = result.end-result.begin;
    const auto count64 = span/slab_cells+(span%slab_cells != 0);
    if (count64 <= 0 || count64 > std::numeric_limits<int>::max()) throw std::runtime_error("Too many ziggurat slabs");
    const int count = static_cast<int>(count64);
    result.slabs.resize(count);
    for (int s = 0; s < count; ++s) {
        auto& slab = result.slabs[s];
        slab.core_begin = result.begin+static_cast<std::int64_t>(s)*slab_cells;
        slab.core_end = std::min(result.end, slab.core_begin+slab_cells);
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    checked_multiply(checked_multiply(threads, count), sizeof(Bounds));
    std::vector<std::vector<Bounds>> partial(threads, std::vector<Bounds>(count));
    // A separate reduction per thread avoids atomics in the ray/slab loop.
    #pragma omp parallel
    {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        auto& bounds = partial[thread];
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < theta.size(); ++i) {
            const double u[3] = {std::sin(theta[i])*std::cos(phi[i]), std::sin(theta[i])*std::sin(phi[i]), std::cos(theta[i])};
            for (int s = 0; s < count; ++s) {
                const auto& slab = result.slabs[s];
                double first = 0, last = max_lookback;
                if (u[axis] == 0) {
                    if (vertex[axis] < slab.core_begin-margin || vertex[axis] > slab.core_end+margin) continue;
                } else {
                    double a = (slab.core_begin-margin-vertex[axis])/u[axis];
                    double b = (slab.core_end+margin-vertex[axis])/u[axis];
                    if (a > b) std::swap(a, b);
                    first = std::max(first, a); last = std::min(last, b);
                    if (first > last) continue;
                }
                for (int a = 0; a < 3; ++a) {
                    const double p = vertex[a]+first*u[a], q = vertex[a]+last*u[a];
                    bounds[s].lo[a] = std::min(bounds[s].lo[a], std::min(p, q));
                    bounds[s].hi[a] = std::max(bounds[s].hi[a], std::max(p, q));
                }
            }
        }
    }
    for (int s = 0; s < count; ++s) {
        Bounds bounds;
        for (const auto& thread : partial) for (int a = 0; a < 3; ++a) {
            bounds.lo[a] = std::min(bounds.lo[a], thread[s].lo[a]);
            bounds.hi[a] = std::max(bounds.hi[a], thread[s].hi[a]);
        }
        auto& slab = result.slabs[s];
        for (int a = 0; a < 3; ++a) {
            const auto first = a == axis ? slab.core_begin : rounded_down(bounds.lo[a]-margin);
            const auto last = a == axis ? slab.core_end : rounded_down(bounds.hi[a]+margin)+1;
            if (last <= first || last-first > std::numeric_limits<int>::max()) throw std::runtime_error("Slab extent exceeds supported range");
            slab.origin[a] = first; slab.extent[a] = static_cast<int>(last-first);
        }
        slab.strides[0] = 1;
        slab.strides[1] = slab.extent[0];
        slab.strides[2] = checked_multiply(slab.extent[0], slab.extent[1]);
        slab.offset = result.voxels;
        const auto size = checked_volume(slab.extent);
        result.voxels = checked_add(result.voxels, size);
        result.largest_slab = std::max(result.largest_slab, size);
    }
    return result;
}
} // namespace lightcone
