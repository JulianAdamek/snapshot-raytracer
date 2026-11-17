#pragma once
#include <cmath>

// Independent experimental controls. Production defaults enable every correction.
#ifndef RT_CORRECT_SHEAR_TRANSPORT
#define RT_CORRECT_SHEAR_TRANSPORT 1
#endif
#ifndef RT_CORRECT_SACHS_B
#define RT_CORRECT_SACHS_B 1
#endif
#ifndef RT_NONLINEAR_SCALAR
#define RT_NONLINEAR_SCALAR 1
#endif
#ifndef RT_CORRECT_OBSERVER_DISTANCE
#define RT_CORRECT_OBSERVER_DISTANCE 1
#endif
#if (RT_CORRECT_SHEAR_TRANSPORT != 0 && RT_CORRECT_SHEAR_TRANSPORT != 1) || \
    (RT_CORRECT_SACHS_B != 0 && RT_CORRECT_SACHS_B != 1) || \
    (RT_NONLINEAR_SCALAR != 0 && RT_NONLINEAR_SCALAR != 1) || \
    (RT_CORRECT_OBSERVER_DISTANCE != 0 && RT_CORRECT_OBSERVER_DISTANCE != 1)
#error "Raytracer correction flags must be 0 or 1"
#endif
#ifdef __CUDACC__
#define RT_HD __host__ __device__
#else
#define RT_HD
#endif
#define RT_STRINGIFY_IMPL(x) #x
#define RT_STRINGIFY(x) RT_STRINGIFY_IMPL(x)
namespace raytracer {
constexpr double DEFAULT_REDSHIFT_TOL = 1e-8;
constexpr char PHYSICS_JSON[] =
    "{\"correct_shear_transport\":" RT_STRINGIFY(RT_CORRECT_SHEAR_TRANSPORT)
    ",\"correct_sachs_b\":" RT_STRINGIFY(RT_CORRECT_SACHS_B)
    ",\"nonlinear_scalar\":" RT_STRINGIFY(RT_NONLINEAR_SCALAR)
    ",\"correct_observer_distance\":" RT_STRINGIFY(RT_CORRECT_OBSERVER_DISTANCE) "}";
constexpr const char* SHEAR_TRANSPORT_NAME = RT_CORRECT_SHEAR_TRANSPORT ? "negative" : "positive";
RT_HD inline double observer_distance_prime(double potential) {
    return RT_CORRECT_OBSERVER_DISTANCE ? -std::exp(2.0*potential) : -1.0;
}
// G_i = (1/2) partial_i E and T_ij = (1/2) E partial_i partial_j E,
// E=exp(2W). Vector terms retain the paper's leading-vector approximation.
RT_HD inline void scalar_jets(double e, const float grad[3], const float hessian[3][3],
                             double g[3], double tidal[3][3]) {
    for (int i=0;i<3;++i) {
        g[i] = RT_NONLINEAR_SCALAR ? e*grad[i] : grad[i];
        for (int j=0;j<3;++j)
            tidal[i][j] = RT_NONLINEAR_SCALAR ?
                e*e*(hessian[i][j]+2.0*static_cast<double>(grad[i])*grad[j]) : hessian[i][j];
    }
}
} // namespace raytracer
