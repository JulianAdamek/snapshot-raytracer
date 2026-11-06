#pragma once
#include "lightcone_sampling.hpp"

namespace lightcone {

constexpr int RK4_REFINEMENT_STEPS = 24;
constexpr double FINAL_Z_TOL = 1e-4;

enum RayStateIndex : int {
    STATE_X = 0,
    STATE_Y,
    STATE_Z,
    STATE_NX,
    STATE_NY,
    STATE_NZ,
    STATE_E1X,
    STATE_E1Y,
    STATE_E1Z,
    STATE_E2X,
    STATE_E2Y,
    STATE_E2Z,
    STATE_DA,
    STATE_DA_PRIME,
    STATE_SIGMA1,
    STATE_SIGMA2,
    STATE_ELLIPTICITY1,
    STATE_ELLIPTICITY2,
    STATE_ROTATION,
    STATE_LOG_KTILDE0,
    STATE_TAU,
    STATE_ISW,
    STATE_SIZE
};


LC_HD inline Status ode(const View& view, const double y[STATE_SIZE], double dydt[STATE_SIZE]) {
    double a;
    auto status = background_at(view, y[STATE_TAU], a);
    if (status != Status::Ok) return status;
    const double x_pos = y[STATE_X];
    const double y_pos = y[STATE_Y];
    const double z_pos = y[STATE_Z];
    double nx = y[STATE_NX];
    double ny = y[STATE_NY];
    double nz = y[STATE_NZ];
    double e1x = y[STATE_E1X];
    double e1y = y[STATE_E1Y];
    double e1z = y[STATE_E1Z];
    double e2x = y[STATE_E2X];
    double e2y = y[STATE_E2Y];
    double e2z = y[STATE_E2Z];
    const double dA = y[STATE_DA];
    const double dAprime = y[STATE_DA_PRIME];
    const double sigma1 = y[STATE_SIGMA1];
    const double sigma2 = y[STATE_SIGMA2];
    const double ellipticity1 = y[STATE_ELLIPTICITY1];
    const double ellipticity2 = y[STATE_ELLIPTICITY2];

    double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    nx /= norm;
    ny /= norm;
    nz /= norm;
    norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    e1x /= norm;
    e1y /= norm;
    e1z /= norm;
    norm = std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);
    e2x /= norm;
    e2y /= norm;
    e2z /= norm;

    double transverse_projector[3][3] = {
        {nx * nx - 1, nx * ny, nx * nz},
        {nx * ny, ny * ny - 1, ny * nz},
        {nx * nz, ny * nz, nz * nz - 1}
    };

    double shear_projector1[3][3] = {
        {e1x * e1x - e2x * e2x, e1x * e1y - e2x * e2y, e1x * e1z - e2x * e2z},
        {e1x * e1y - e2x * e2y, e1y * e1y - e2y * e2y, e1y * e1z - e2y * e2z},
        {e1x * e1z - e2x * e2z, e1y * e1z - e2y * e2z, e1z * e1z - e2z * e2z}
    };

    double shear_projector2[3][3] = {
        {2 * e1x * e2x, e1x * e2y + e1y * e2x, e1x * e2z + e1z * e2x},
        {e1x * e2y + e1y * e2x, 2 * e1y * e2y, e1y * e2z + e1z * e2y},
        {e1x * e2z + e1z * e2x, e1y * e2z + e1z * e2y, 2 * e1z * e2z}
    };

    float grad[3];
    float ngradB[3];
    float hessian[3][3];
    float tidalBterm[3][3];
    float e1gradB[3] = {0,0,0};
    float e2gradB[3] = {0,0,0};
    TimeSample b_time[3], w_time;

    float Bx;
    status = sample(view, 1, y[STATE_TAU], x_pos, y_pos, z_pos, Bx, grad, hessian, &b_time[0]);
    if (status != Status::Ok) return status;

    ngradB[0] = grad[0] * nx;
    ngradB[1] = grad[1] * nx;
    ngradB[2] = grad[2] * nx;

    float e1ngradB = 0.5 * (grad[0] * nx + grad[1] * ny + grad[2] * nz);
    float e2ngradB = e1ngradB;

    e1ngradB = e1ngradB * e1x + 0.5 * (ngradB[0] * e1x + ngradB[1] * e1y + ngradB[2] * e1z);
    e2ngradB = e2ngradB * e2x + 0.5 * (ngradB[0] * e2x + ngradB[1] * e2y + ngradB[2] * e2z);

    tidalBterm[0][0] = hessian[0][0] * nx - (nx * hessian[0][0] + ny * hessian[1][0] + nz * hessian[2][0]);
    tidalBterm[0][1] = hessian[0][1] * nx - 0.5 * (nx * hessian[0][1] + ny * hessian[1][1] + nz * hessian[2][1]);
    tidalBterm[0][2] = hessian[0][2] * nx - 0.5 * (nx * hessian[0][2] + ny * hessian[1][2] + nz * hessian[2][2]);
    tidalBterm[1][1] = hessian[1][1] * nx;
    tidalBterm[1][2] = hessian[1][2] * nx;
    tidalBterm[2][2] = hessian[2][2] * nx;

    float nLaplaceB = nx * (hessian[0][0] + hessian[1][1] + hessian[2][2]);

    e1gradB[0] = 0.5 * (e1y * grad[1] + e1z * grad[2]);
    e1gradB[1] = -0.5 * e1x * grad[1];
    e1gradB[2] = -0.5 * e1x * grad[2];

    e2gradB[0] = 0.5 * (e2y * grad[1] + e2z * grad[2]);
    e2gradB[1] = -0.5 * e2x * grad[1];
    e2gradB[2] = -0.5 * e2x * grad[2];

    float By;
    status = sample(view, 2, y[STATE_TAU], x_pos, y_pos, z_pos, By, grad, hessian, &b_time[1]);
    if (status != Status::Ok) return status;

    ngradB[0] += grad[0] * ny;
    ngradB[1] += grad[1] * ny;
    ngradB[2] += grad[2] * ny;

    e1ngradB += 0.5 * e1y * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * ny * (grad[0] * e1x + grad[1] * e1y + grad[2] * e1z);
    e2ngradB += 0.5 * e2y * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * ny * (grad[0] * e2x + grad[1] * e2y + grad[2] * e2z);

    tidalBterm[0][0] += hessian[0][0] * ny;
    tidalBterm[0][1] += hessian[0][1] * ny - 0.5 * (nx * hessian[0][0] + ny * hessian[1][0] + nz * hessian[2][0]);
    tidalBterm[0][2] += hessian[0][2] * ny;
    tidalBterm[1][1] += hessian[1][1] * ny - (nx * hessian[0][1] + ny * hessian[1][1] + nz * hessian[2][1]);
    tidalBterm[1][2] += hessian[1][2] * ny - 0.5 * (nx * hessian[0][2] + ny * hessian[1][2] + nz * hessian[2][2]);
    tidalBterm[2][2] += hessian[2][2] * ny;

    nLaplaceB += (hessian[0][0] + hessian[1][1] + hessian[2][2]) * ny;

    e1gradB[0] += -0.5 * e1y * grad[0];
    e1gradB[1] += 0.5 * (e1x * grad[0] + e1z * grad[2]);
    e1gradB[2] += -0.5 * e1y * grad[2];

    e2gradB[0] += -0.5 * e2y * grad[0];
    e2gradB[1] += 0.5 * (e2x * grad[0] + e2z * grad[2]);
    e2gradB[2] += -0.5 * e2y * grad[2];

    float Bz;
    status = sample(view, 3, y[STATE_TAU], x_pos, y_pos, z_pos, Bz, grad, hessian, &b_time[2]);
    if (status != Status::Ok) return status;

    ngradB[0] += grad[0] * nz;
    ngradB[1] += grad[1] * nz;
    ngradB[2] += grad[2] * nz;

    e1ngradB += 0.5 * e1z * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * nz * (grad[0] * e1x + grad[1] * e1y + grad[2] * e1z);
    e2ngradB += 0.5 * e2z * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * nz * (grad[0] * e2x + grad[1] * e2y + grad[2] * e2z);

    tidalBterm[0][0] += hessian[0][0] * nz;
    tidalBterm[0][1] += hessian[0][1] * nz;
    tidalBterm[0][2] += hessian[0][2] * nz - 0.5 * (nx * hessian[0][0] + ny * hessian[1][0] + nz * hessian[2][0]);
    tidalBterm[1][1] += hessian[1][1] * nz;
    tidalBterm[1][2] += hessian[1][2] * nz - 0.5 * (nx * hessian[0][1] + ny * hessian[1][1] + nz * hessian[2][1]);
    tidalBterm[2][2] += hessian[2][2] * nz - (nx * hessian[0][2] + ny * hessian[1][2] + nz * hessian[2][2]);

    tidalBterm[1][0] = tidalBterm[0][1];
    tidalBterm[2][0] = tidalBterm[0][2];
    tidalBterm[2][1] = tidalBterm[1][2];

    nLaplaceB += (hessian[0][0] + hessian[1][1] + hessian[2][2]) * nz;

    e1gradB[0] += -0.5 * e1z * grad[0];
    e1gradB[1] += -0.5 * e1z * grad[1];
    e1gradB[2] += 0.5 * (e1x * grad[0] + e1y * grad[1]);

    e2gradB[0] += -0.5 * e2z * grad[0];
    e2gradB[1] += -0.5 * e2z * grad[1];
    e2gradB[2] += 0.5 * (e2x * grad[0] + e2y * grad[1]);

    float phi;
    status = sample(view, 0, y[STATE_TAU], x_pos, y_pos, z_pos, phi, grad, hessian, &w_time);
    if (status != Status::Ok) return status;
    const double exp_2phi = std::exp(2.0 * static_cast<double>(phi));
    const double n_dot_grad_phi = nx * grad[0] + ny * grad[1] + nz * grad[2];
    const double dlogktilde_dt = 4.0 * exp_2phi * n_dot_grad_phi + nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2];

    dydt[STATE_X] = -nx * exp_2phi - Bx;
    dydt[STATE_Y] = -ny * exp_2phi - By;
    dydt[STATE_Z] = -nz * exp_2phi - Bz;

    dydt[STATE_NX] = -(transverse_projector[0][0] * (2 * grad[0] + ngradB[0]) + transverse_projector[0][1] * (2 * grad[1] + ngradB[1]) + transverse_projector[0][2] * (2 * grad[2] + ngradB[2]));
    dydt[STATE_NY] = -(transverse_projector[1][0] * (2 * grad[0] + ngradB[0]) + transverse_projector[1][1] * (2 * grad[1] + ngradB[1]) + transverse_projector[1][2] * (2 * grad[2] + ngradB[2]));
    dydt[STATE_NZ] = -(transverse_projector[2][0] * (2 * grad[0] + ngradB[0]) + transverse_projector[2][1] * (2 * grad[1] + ngradB[1]) + transverse_projector[2][2] * (2 * grad[2] + ngradB[2]));

    dydt[STATE_E1X] = -2 * nx * (e1x * grad[0] + e1y * grad[1] + e1z * grad[2]) - nx * e1ngradB;
    dydt[STATE_E1Y] = -2 * ny * (e1x * grad[0] + e1y * grad[1] + e1z * grad[2]) - ny * e1ngradB;
    dydt[STATE_E1Z] = -2 * nz * (e1x * grad[0] + e1y * grad[1] + e1z * grad[2]) - nz * e1ngradB;

    dydt[STATE_E2X] = -2 * nx * (e2x * grad[0] + e2y * grad[1] + e2z * grad[2]) - nx * e2ngradB;
    dydt[STATE_E2Y] = -2 * ny * (e2x * grad[0] + e2y * grad[1] + e2z * grad[2]) - ny * e2ngradB;
    dydt[STATE_E2Z] = -2 * nz * (e2x * grad[0] + e2y * grad[1] + e2z * grad[2]) - nz * e2ngradB;

    dydt[STATE_DA] = -dAprime;
    dydt[STATE_DA_PRIME] = -4 * dAprime * (nx * grad[0] + ny * grad[1] + nz * grad[2] + 0.25 * (nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2])) - (transverse_projector[0][0] * hessian[0][0] + transverse_projector[0][1] * hessian[0][1] + transverse_projector[0][2] * hessian[0][2] + transverse_projector[1][0] * hessian[1][0] + transverse_projector[1][1] * hessian[1][1] + transverse_projector[1][2] * hessian[1][2] + transverse_projector[2][0] * hessian[2][0] + transverse_projector[2][1] * hessian[2][1] + transverse_projector[2][2] * hessian[2][2]) * dA + 0.5 * nLaplaceB * dA;

    hessian[0][0] += 0.5 * tidalBterm[0][0];
    hessian[0][1] += 0.5 * tidalBterm[0][1];
    hessian[0][2] += 0.5 * tidalBterm[0][2];
    hessian[1][0] += 0.5 * tidalBterm[1][0];
    hessian[1][1] += 0.5 * tidalBterm[1][1];
    hessian[1][2] += 0.5 * tidalBterm[1][2];
    hessian[2][0] += 0.5 * tidalBterm[2][0];
    hessian[2][1] += 0.5 * tidalBterm[2][1];
    hessian[2][2] += 0.5 * tidalBterm[2][2];

    // Backward transport: d(sigma)/ds = -d(log ktilde0)/ds * sigma + tidal source.
    // Ellipticity sign conventions do not change this homogeneous coefficient.
    dydt[STATE_SIGMA1] = -4 * sigma1 * (nx * grad[0] + ny * grad[1] + nz * grad[2] + 0.25 * (nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2])) + (shear_projector1[0][0] * hessian[0][0] + shear_projector1[0][1] * hessian[0][1] + shear_projector1[0][2] * hessian[0][2] + shear_projector1[1][0] * hessian[1][0] + shear_projector1[1][1] * hessian[1][1] + shear_projector1[1][2] * hessian[1][2] + shear_projector1[2][0] * hessian[2][0] + shear_projector1[2][1] * hessian[2][1] + shear_projector1[2][2] * hessian[2][2]) * dA * dA;
    dydt[STATE_SIGMA2] = -4 * sigma2 * (nx * grad[0] + ny * grad[1] + nz * grad[2] + 0.25 * (nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2])) + (shear_projector2[0][0] * hessian[0][0] + shear_projector2[0][1] * hessian[0][1] + shear_projector2[0][2] * hessian[0][2] + shear_projector2[1][0] * hessian[1][0] + shear_projector2[1][1] * hessian[1][1] + shear_projector2[1][2] * hessian[1][2] + shear_projector2[2][0] * hessian[2][0] + shear_projector2[2][1] * hessian[2][1] + shear_projector2[2][2] * hessian[2][2]) * dA * dA;

    if (dA > 0.5)
    {
        dydt[STATE_DA_PRIME] += (sigma1 * sigma1 + sigma2 * sigma2) / (dA * dA * dA);
        dydt[STATE_ELLIPTICITY1] = -2 * sigma1 * std::sqrt(4 + ellipticity1 * ellipticity1 + ellipticity2 * ellipticity2) / (dA * dA);
        dydt[STATE_ELLIPTICITY2] = -2 * sigma2 * std::sqrt(4 + ellipticity1 * ellipticity1 + ellipticity2 * ellipticity2) / (dA * dA);
        dydt[STATE_ROTATION] = -(sigma1 * ellipticity2 - sigma2 * ellipticity1) / (dA * dA * (2 + std::sqrt(4 + ellipticity1 * ellipticity1 + ellipticity2 * ellipticity2)));
    }
    else
    {
        dydt[STATE_ELLIPTICITY1] = 0;
        dydt[STATE_ELLIPTICITY2] = 0;
        dydt[STATE_ROTATION] = 0;
    }

    dydt[STATE_LOG_KTILDE0] = dlogktilde_dt;
    // arXiv:2002.04024v2 (23), (26), (30), in backward cell time s.
    // The existing tidalBterm already contains the convective part of d/dtau.
    // b_time[j].gradient[i] is the additional partial_i partial_eta B_j.
    const double isw_source = 2.0*w_time.partial_time;
    if (isw_source != 0) {
        dydt[STATE_LOG_KTILDE0] += isw_source;
        dydt[STATE_DA_PRIME] -= isw_source*dAprime;
        // The time-dependent contribution has the same negative transport sign.
        dydt[STATE_SIGMA1] -= isw_source*sigma1;
        dydt[STATE_SIGMA2] -= isw_source*sigma2;
    }
    const double direction[3] = {nx,ny,nz};
    double longitudinal = 0, tidal1 = 0, tidal2 = 0;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        const double mixed = 0.5*(static_cast<double>(b_time[j].gradient[i])+b_time[i].gradient[j]);
        longitudinal += direction[i]*direction[j]*mixed;
        tidal1 += shear_projector1[i][j]*mixed;
        tidal2 += shear_projector2[i][j]*mixed;
    }
    if (longitudinal != 0) dydt[STATE_DA_PRIME] += 0.5*longitudinal*dA;
    if (tidal1 != 0) dydt[STATE_SIGMA1] -= 0.5*dA*dA*tidal1;
    if (tidal2 != 0) dydt[STATE_SIGMA2] -= 0.5*dA*dA*tidal2;
    // Positive source-to-observer integral, despite tracing toward the past.
    dydt[STATE_ISW] = isw_source;
    dydt[STATE_TAU] = -1.0/view.n;
    for (int i = 0; i < STATE_SIZE; ++i) if (!finite(dydt[i])) return Status::InvalidValue;
    return Status::Ok;
}


LC_HD inline void copy_state(double* dst, const double* src) {
    for (int i = 0; i < STATE_SIZE; ++i) dst[i] = src[i];
}
LC_HD inline Status observed_redshift(const View& view, const double y[STATE_SIZE], double& z) {
    double a;
    auto status = background_at(view, y[STATE_TAU], a);
    if (status != Status::Ok) return status;
    float phi, grad[3], hessian[3][3];
    status = sample(view, 0, y[STATE_TAU], y[STATE_X], y[STATE_Y], y[STATE_Z], phi, grad, hessian);
    if (status != Status::Ok) return status;
    z = std::exp(3.0*static_cast<double>(phi)+y[STATE_LOG_KTILDE0])/a-1.0;
    return finite(z) ? Status::Ok : Status::InvalidValue;
}
LC_HD inline Status initialize(const View& view, double tau, double theta, double phi, double y[STATE_SIZE]) {
    for (int i = 0; i < STATE_SIZE; ++i) y[i] = 0.0;
    for (int axis = 0; axis < 3; ++axis) y[STATE_X+axis] = view.geometry.vertex[axis]*view.n;
    y[STATE_NX] = -std::sin(theta)*std::cos(phi);
    y[STATE_NY] = -std::sin(theta)*std::sin(phi);
    y[STATE_NZ] = -std::cos(theta);
    y[STATE_E1X] = std::cos(theta)*std::cos(phi);
    y[STATE_E1Y] = std::cos(theta)*std::sin(phi);
    y[STATE_E1Z] = -std::sin(theta);
    y[STATE_E2X] = std::sin(phi);
    y[STATE_E2Y] = -std::cos(phi);
    y[STATE_DA_PRIME] = -1.0;
    y[STATE_TAU] = tau;
    double a;
    auto status = background_at(view, tau, a);
    if (status != Status::Ok) return status;
    float potential, grad[3], hessian[3][3];
    status = sample(view, 0, tau, y[STATE_X], y[STATE_Y], y[STATE_Z], potential, grad, hessian);
    if (status == Status::Ok) y[STATE_LOG_KTILDE0] = std::log(a)-3.0*potential;
    return status;
}

// Clamp only roundoff at the last background knot, never a genuine extrapolation.
LC_HD inline double stage_tau(const View& view, double tau) {
    const double minimum = integration_lower_bound(view);
    const double tolerance = 8*DBL_EPSILON*std::fmax(1.0, std::fabs(minimum));
    return tau < minimum && minimum-tau <= tolerance ? minimum : tau;
}

// Transactional: no ray state changes unless all four stages succeed.
LC_HD inline Status rk4_step(const View& view, double y[STATE_SIZE], double dt) {
    if (!finite(dt) || dt <= 0.0) return Status::InvalidValue;
    double temp[STATE_SIZE], k1[STATE_SIZE], k2[STATE_SIZE], k3[STATE_SIZE], k4[STATE_SIZE];
    auto status = ode(view, y, k1);
    if (status != Status::Ok) return status;
    if (dt > (y[STATE_TAU]-integration_lower_bound(view))*view.n) return Status::BackgroundCoverage;
    for (int i = 0; i < STATE_SIZE; ++i) temp[i] = y[i]+0.5*dt*k1[i];
    temp[STATE_TAU] = stage_tau(view, y[STATE_TAU]-0.5*dt/view.n);
    status = ode(view, temp, k2);
    if (status != Status::Ok) return status;
    for (int i = 0; i < STATE_SIZE; ++i) temp[i] = y[i]+0.5*dt*k2[i];
    temp[STATE_TAU] = stage_tau(view, y[STATE_TAU]-0.5*dt/view.n);
    status = ode(view, temp, k3);
    if (status != Status::Ok) return status;
    for (int i = 0; i < STATE_SIZE; ++i) temp[i] = y[i]+dt*k3[i];
    temp[STATE_TAU] = stage_tau(view, y[STATE_TAU]-dt/view.n);
    status = ode(view, temp, k4);
    if (status != Status::Ok) return status;
    for (int i = 0; i < STATE_SIZE; ++i) {
        temp[i] = y[i]+(dt/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
        if (!finite(temp[i])) return Status::InvalidValue;
    }
    temp[STATE_TAU] = stage_tau(view, y[STATE_TAU]-dt/view.n);
    copy_state(y, temp);
    return Status::Ok;
}
LC_HD inline Status trace(const View& view, double target_z, double y[STATE_SIZE], double& reached_z) {
    reached_z = NAN;
    if (!finite(target_z) || target_z <= 0.0) return Status::InvalidValue;
    auto status = observed_redshift(view, y, reached_z);
    if (status != Status::Ok) return status;
    const double span = (y[STATE_TAU]-integration_lower_bound(view))*view.n;
    if (!finite(span) || span < 0.0 || span > 4e18) return Status::InvalidValue;
    const auto max_steps = static_cast<std::int64_t>(std::ceil(span))+1;
    for (std::int64_t step = 0; step < max_steps; ++step) {
        if (reached_z >= target_z) return Status::Ok;
        const double remaining = (y[STATE_TAU]-integration_lower_bound(view))*view.n;
        if (remaining <= 0.0) break;
        const double dt = std::fmin(1.0, remaining);
        double candidate[STATE_SIZE];
        copy_state(candidate, y);
        status = rk4_step(view, candidate, dt);
        if (status != Status::Ok) return status;
        if (candidate[STATE_TAU] >= y[STATE_TAU]) return Status::InvalidValue;
        double next_z;
        status = observed_redshift(view, candidate, next_z);
        if (status != Status::Ok) return status;
        if (next_z >= target_z) {
            double low = 0.0, high = dt;
            for (int iter = 0; iter < RK4_REFINEMENT_STEPS && std::fabs(next_z-target_z) > FINAL_Z_TOL; ++iter) {
                const double mid = 0.5*(low+high);
                double refined[STATE_SIZE];
                copy_state(refined, y);
                status = rk4_step(view, refined, mid);
                if (status != Status::Ok) return status;
                double mid_z;
                status = observed_redshift(view, refined, mid_z);
                if (status != Status::Ok) return status;
                if (mid_z >= target_z) { high = mid; copy_state(candidate, refined); next_z = mid_z; }
                else low = mid;
            }
            if (std::fabs(next_z-target_z) > FINAL_Z_TOL) return Status::TargetNotReached;
        }
        copy_state(y, candidate);
        reached_z = next_z;
    }
    return reached_z >= target_z ? Status::Ok : Status::TargetNotReached;
}
LC_HD inline void observables(const View& view, const double y[STATE_SIZE], double theta, double phi, float maps[6]) {
    double r[3];
    for (int axis = 0; axis < 3; ++axis) r[axis] = y[STATE_X+axis]-view.geometry.vertex[axis]*view.n;
    const double norm = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    r[0] = r[0]/norm-std::sin(theta)*std::cos(phi);
    r[1] = r[1]/norm-std::sin(theta)*std::sin(phi);
    r[2] = r[2]/norm-std::cos(theta);
    maps[0] = static_cast<float>(y[STATE_DA]);
    maps[1] = static_cast<float>(y[STATE_ELLIPTICITY1]);
    maps[2] = static_cast<float>(y[STATE_ELLIPTICITY2]);
    maps[3] = static_cast<float>(y[STATE_ROTATION]);
    maps[4] = static_cast<float>(r[0]*std::cos(theta)*std::cos(phi)+r[1]*std::cos(theta)*std::sin(phi)-r[2]*std::sin(theta));
    maps[5] = static_cast<float>(r[0]*std::sin(phi)-r[1]*std::cos(phi));
}
} // namespace lightcone
