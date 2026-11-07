#pragma once
#include "lightcone.hpp"

namespace lightcone {

// gevolution's legacy pointInShell: pos = (wrapped index + offset)/N,
// vertex_image = vertex + integer copy shift. Preserve these subtractions;
// replacing them by unwrapped_index/N - vertex changes boundary ownership.
// Products and left-associated sums are separately rounded (unfused legacy mode).
LC_HD inline double legacy_node_radius(const std::int64_t xyz[3], const double offset[3],
                                       int n, const Geometry& geometry, bool& cone) {
    double d[3];
    for (int axis = 0; axis < 3; ++axis) {
        auto image = xyz[axis]/n;
        auto index = xyz[axis]%n;
        if (index < 0) { index += n; --image; }
        const double pos = lcg::div(lcg::add(static_cast<double>(index), offset[axis]), n);
        const double vertex = lcg::add(geometry.vertex[axis], -static_cast<double>(image));
        d[axis] = lcg::sub(pos, vertex);
    }
    const double r = lcg::sqrt(lcg::add(lcg::add(lcg::mul(d[0], d[0]), lcg::mul(d[1], d[1])),
                                      lcg::mul(d[2], d[2])));
    if (!(r >= 0.0) || r > 1.7976931348623157e308) { cone = false; return -1.0; }
    cone = true;
    if (geometry.angular_limit) {
        const double dot = lcg::add(lcg::add(lcg::mul(d[0], geometry.direction[0]),
                                            lcg::mul(d[1], geometry.direction[1])),
                                            lcg::mul(d[2], geometry.direction[2]));
        cone = r > 0.0 && lcg::div(dot, r) > geometry.opening_cos;
    }
    return r;
}

LC_HD inline double node_radius(const View& view, const GridView& grid,
                                const std::int64_t xyz[3], bool& cone) {
    const auto& g = view.geometry;
    if (g.convention == GeometryConvention::GasgrdV1)
        return lcg::node_radius(xyz, grid.node_offset, view.n, g.vertex, g.direction,
                                g.angular_limit ? g.opening_cos : -1.0, cone);
    return legacy_node_radius(xyz, grid.node_offset, view.n, g, cone);
}

} // namespace lightcone
