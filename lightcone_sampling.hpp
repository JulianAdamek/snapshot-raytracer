#pragma once
#include "lightcone.hpp"
#include "lightcone_geometry.hpp"
#include <cmath>
#include <cfloat>

namespace lightcone {

LC_HD inline bool finite(double x) { return x == x && x <= DBL_MAX && x >= -DBL_MAX; }
LC_HD inline std::int64_t wrap(std::int64_t x, int n) { return (x % n + n) % n; }
LC_HD inline Status background_at(const View& view, double tau, double& a) {
    if (!finite(tau)) return Status::InvalidValue;
    if (tau < view.times[0] || tau > view.times[view.record_count-1]) return Status::BackgroundCoverage;
    int low = 0, high = view.record_count-1;
    while (high-low > 1) {
        const int mid = low+(high-low)/2;
        if (tau < view.times[mid]) high = mid; else low = mid;
    }
    const auto& b = view.background[low];
    if (tau == b.tau0) a = b.a0;
    else if (tau == b.tau1) a = b.a1;
    else {
        const double t = (tau-b.tau0)/(b.tau1-b.tau0);
        a = b.a0+t*(b.c1+t*(b.c2+t*b.c3));
    }
    return finite(a) && a > 0.0 ? Status::Ok : Status::InvalidValue;
}
LC_HD inline int record_at(const View& view, int slot, double radius) {
    int low = view.slot_offsets[slot], high = view.slot_offsets[slot+1];
    const int end = high;
    while (low < high) {
        const int mid = low+(high-low)/2;
        if (view.shells[mid].outer <= radius) low = mid+1; else high = mid;
    }
    if (low == end || radius < view.shells[low].inner) return -1;
    return view.shells[low].record;
}
LC_HD inline float decode_half(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    if (exponent == 31) return mantissa ? NAN : (bits & 32768 ? -INFINITY : INFINITY);
    const float value = exponent ? std::ldexp(static_cast<float>(1024+mantissa), exponent-25)
                                 : std::ldexp(static_cast<float>(mantissa), -24);
    return bits & 32768 ? -value : value;
}
LC_HD inline double integration_lower_bound(const View& view) {
    return std::fmax(view.times[0], view.trace_tau_min);
}
LC_HD inline Status load_node(const GridView& grid, int n, const std::int64_t xyz[3], float& value,
                              const SlabLayoutView& layout = {}) {
    std::size_t index = 0;
    if (layout.count) {
        const auto layer = xyz[layout.axis];
        if (layer < layout.begin || layer >= layout.end) return Status::NotResident;
        const auto owner = (layer-layout.begin)/layout.slab_cells;
        if (owner < 0 || owner >= layout.count) return Status::NotResident;
        const auto& slab = layout.slabs[owner];
        index = slab.offset;
        for (int axis = 0; axis < 3; ++axis) {
            // No periodic aliasing here: only the HDF5 reader wraps source indices.
            if (xyz[axis] < slab.origin[axis]) return Status::NotResident;
            const auto relative = xyz[axis]-slab.origin[axis];
            if (relative >= slab.extent[axis]) return Status::NotResident;
            index += static_cast<std::size_t>(relative)*slab.strides[axis];
        }
    } else {
        for (int axis = 0; axis < 3; ++axis) {
            const std::int64_t relative = wrap(wrap(xyz[axis], n)-wrap(grid.origin[axis], n), n);
            if (relative >= grid.extent[axis]) return Status::NotResident;
            index += static_cast<std::size_t>(relative)*grid.strides[axis];
        }
    }
    if (grid.values) value = grid.values[index];
    else if (grid.half_values) value = decode_half(grid.half_values[index])/134217728.0f;
    else return Status::NotResident;
    return finite(value) ? Status::Ok : Status::InvalidValue;
}
LC_HD inline Status load_sample(const View& view, int field, int slot, const std::int64_t xyz[3],
                               double tau, float& value) {
    const auto status = load_node(view.grids[field*view.thickness+slot], view.n, xyz, value, view.layout);
    if (status == Status::NotResident && view.miss && !view.miss->valid) {
        view.miss->valid = true; view.miss->field = field; view.miss->slot = slot; view.miss->tau = tau;
        for (int axis = 0; axis < 3; ++axis) view.miss->node[axis] = xyz[axis];
    }
    return status;
}

// Resolve times at each physical stencil node before taking spatial derivatives.
LC_HD inline Status node_at(const View& view, int field, const std::int64_t xyz[3],
                            double tau, float& value, float* partial_time = nullptr) {
    if (partial_time) *partial_time = 0;
    const bool derivative = partial_time && view.time_derivatives == TimeDerivativeMode::FiniteDifference;
    if (derivative && view.thickness < 2) return Status::TimeCoverage;
    bool cone;
    const double radius = node_radius(view, view.grids[field*view.thickness], xyz, cone);
    if (radius < 0.0) return Status::InvalidValue;
    if (!cone) return Status::SpatialCoverage;
    int below = -1, above = -1, exact = -1;
    double low_tau = -DBL_MAX, high_tau = DBL_MAX;
    bool any = false;
    for (int slot = 0; slot < view.thickness; ++slot) {
        const int record = record_at(view, slot, radius);
        if (record < 0) continue;
        any = true;
        const double time = view.times[record];
        if (view.thickness == 1 || time == tau) {
            if (!derivative) return load_sample(view, field, slot, xyz, tau, value);
            exact = slot;
        }
        if (time < tau && time > low_tau) { low_tau = time; below = slot; }
        if (time > tau && time < high_tau) { high_tau = time; above = slot; }
    }
    if (!any) return Status::SpatialCoverage;
    if (exact >= 0) {
        // Preserve knot values exactly; prefer the interval behind a backward ray.
        const int neighbor = below >= 0 ? below : above;
        if (neighbor < 0) return Status::TimeCoverage;
        auto status = load_sample(view, field, exact, xyz, tau, value);
        if (status != Status::Ok) return status;
        float other;
        status = load_sample(view, field, neighbor, xyz, tau, other);
        if (status != Status::Ok) return status;
        *partial_time = static_cast<float>(below >= 0 ?
            (static_cast<double>(value)-other)/(view.n*(tau-low_tau)) :
            (static_cast<double>(other)-value)/(view.n*(high_tau-tau)));
        return finite(*partial_time) ? Status::Ok : Status::InvalidValue;
    }
    if (below < 0 || above < 0) return Status::TimeCoverage;
    float lo, hi;
    auto status = load_sample(view, field, below, xyz, tau, lo);
    if (status != Status::Ok) return status;
    status = load_sample(view, field, above, xyz, tau, hi);
    if (status != Status::Ok) return status;
    value = static_cast<float>(static_cast<double>(lo)+(tau-low_tau)/(high_tau-low_tau)*(static_cast<double>(hi)-lo));
    if (derivative) {
        *partial_time = static_cast<float>((static_cast<double>(hi)-lo)/(view.n*(high_tau-low_tau)));
        if (!finite(*partial_time)) return Status::InvalidValue;
    }
    return finite(value) ? Status::Ok : Status::InvalidValue;
}
// 1D Catmull-Rom spline interpolation
LC_HD inline float cubic_cr(const float p[4], float t) noexcept {
    return 0.5f * ((2 * p[1]) + (-p[0] + p[2]) * t +
        (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]) * t * t +
        (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) * t * t * t);
}

// Gradient of the 1D Catmull-Rom spline
LC_HD inline float cubic_cr_grad(const float p[4], float t) noexcept {
    return 0.5f * ((-p[0] + p[2]) +
        (4 * p[0] - 10 * p[1] + 8 * p[2] - 2 * p[3]) * t +
        (-3 * p[0] + 9 * p[1] - 9 * p[2] + 3 * p[3]) * t * t);
}

// Second derivative of the 1D Catmull-Rom spline
LC_HD inline float cubic_cr_grad2(const float p[4], float t) noexcept {
    return (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3] +
        (-3 * p[0] + 9 * p[1] - 9 * p[2] + 3 * p[3]) * t);
}


LC_HD inline Status sample(const View& view, int field, double tau, double x, double y, double z,
                           float& value, float grad[3], float hessian[3][3], TimeSample* time = nullptr) {
    if (time) *time = TimeSample{};
    const bool derivative = time && view.time_derivatives == TimeDerivativeMode::FiniteDifference;
    if (!finite(x) || !finite(y) || !finite(z) || !finite(tau) ||
        std::fabs(x) > 4e18 || std::fabs(y) > 4e18 || std::fabs(z) > 4e18) return Status::InvalidValue;
    const auto& grid = view.grids[field*view.thickness];
    x -= grid.node_offset[0]; y -= grid.node_offset[1]; z -= grid.node_offset[2];
    const std::int64_t ix = static_cast<std::int64_t>(std::floor(x));
    const std::int64_t iy = static_cast<std::int64_t>(std::floor(y));
    const std::int64_t iz = static_cast<std::int64_t>(std::floor(z));
    const float dx = static_cast<float>(x-ix), dy = static_cast<float>(y-iy), dz = static_cast<float>(z-iz);
    float tmp_yz[4][4], tmp_gradx_yz[4][4], tmp_grad2x_yz[4][4];
    float time_yz[4][4], time_gradx_yz[4][4];
    for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) {
        float p[4], slopes[4];
        for (int l = 0; l < 4; ++l) {
            const std::int64_t xyz[3] = {ix+l-1, iy+j-1, iz+k-1};
            const auto status = node_at(view, field, xyz, tau, p[l], derivative ? &slopes[l] : nullptr);
            if (status != Status::Ok) return status;
        }
        tmp_yz[j][k] = cubic_cr(p, dx);
        tmp_gradx_yz[j][k] = cubic_cr_grad(p, dx);
        tmp_grad2x_yz[j][k] = cubic_cr_grad2(p, dx);
        if (derivative) {
            time_yz[j][k] = cubic_cr(slopes, dx);
            time_gradx_yz[j][k] = cubic_cr_grad(slopes, dx);
        }
    }
    // 2) along y
    float tmp_z[4];
    float tmp_gradx_z[4];
    float tmp_grady_z[4];
    float tmp_grad2x_z[4];
    float tmp_grad2y_z[4];
    float tmp_gradxy_z[4];
    for (int k = 0; k < 4; ++k) {
        float p[4] = {
            tmp_yz[0][k],
            tmp_yz[1][k],
            tmp_yz[2][k],
            tmp_yz[3][k]
        };
        tmp_z[k] = cubic_cr(p, dy);
        tmp_grady_z[k] = cubic_cr_grad(p, dy);
        tmp_grad2y_z[k] = cubic_cr_grad2(p, dy);
        p[0] = tmp_gradx_yz[0][k];
        p[1] = tmp_gradx_yz[1][k];
        p[2] = tmp_gradx_yz[2][k];
        p[3] = tmp_gradx_yz[3][k];
        tmp_gradx_z[k] = cubic_cr(p, dy);
        tmp_gradxy_z[k] = cubic_cr_grad(p, dy);
        p[0] = tmp_grad2x_yz[0][k];
        p[1] = tmp_grad2x_yz[1][k];
        p[2] = tmp_grad2x_yz[2][k];
        p[3] = tmp_grad2x_yz[3][k];
        tmp_grad2x_z[k] = cubic_cr(p, dy);
    }

    // 3) along z
    grad[0] = cubic_cr(tmp_gradx_z, dz);
    grad[1] = cubic_cr(tmp_grady_z, dz);
    grad[2] = cubic_cr_grad(tmp_z, dz);


    hessian[0][0] = cubic_cr(tmp_grad2x_z, dz);
    hessian[0][1] = cubic_cr(tmp_gradxy_z, dz);
    hessian[0][2] = cubic_cr_grad(tmp_gradx_z, dz);
    hessian[1][0] = hessian[0][1];
    hessian[1][1] = cubic_cr(tmp_grad2y_z, dz);
    hessian[1][2] = cubic_cr_grad(tmp_grady_z, dz);
    hessian[2][0] = hessian[0][2];
    hessian[2][1] = hessian[1][2];
    hessian[2][2] = cubic_cr_grad2(tmp_z, dz);

    value = cubic_cr(tmp_z, dz);
    if (derivative) {
        float v[4], gx[4], gy[4];
        for (int k = 0; k < 4; ++k) {
            const float p[4] = {time_yz[0][k],time_yz[1][k],time_yz[2][k],time_yz[3][k]};
            const float px[4] = {time_gradx_yz[0][k],time_gradx_yz[1][k],time_gradx_yz[2][k],time_gradx_yz[3][k]};
            v[k] = cubic_cr(p,dy); gy[k] = cubic_cr_grad(p,dy); gx[k] = cubic_cr(px,dy);
        }
        time->partial_time = cubic_cr(v,dz);
        time->gradient[0] = cubic_cr(gx,dz);
        time->gradient[1] = cubic_cr(gy,dz);
        time->gradient[2] = cubic_cr_grad(v,dz);
        if (!finite(time->partial_time)) return Status::InvalidValue;
        for (float component : time->gradient) if (!finite(component)) return Status::InvalidValue;
    }
    for (int i = 0; i < 3; ++i) {
        if (!finite(grad[i])) return Status::InvalidValue;
        for (int j = 0; j < 3; ++j) if (!finite(hessian[i][j])) return Status::InvalidValue;
    }
    return finite(value) ? Status::Ok : Status::InvalidValue;
}


} // namespace lightcone
