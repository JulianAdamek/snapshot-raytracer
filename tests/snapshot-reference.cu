// Compile the snapshot source directly as a host-callable reference.
#define main snapshot_cli_main
#include "../snapshot-raytracer.cu"
#undef main
#include "snapshot-reference.hpp"

void snapshot_pack_b(const float* values, std::uint16_t* packed, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const __half_raw h = __float2half(values[i]*134217728.0f);
        packed[i] = h.x;
    }
}
namespace {
std::vector<__half> half_vectors(const float* data, int n) {
    const std::size_t size = 3ULL*n*n*n;
    std::vector<__half> result(size);
    for (std::size_t i = 0; i < size; ++i) result[i] = __float2half(data[i]*134217728.0f);
    return result;
}
}
void snapshot_reference_ode(const float* potential, const float* vectors, int n, const double* state,
                            double* rhs, double cell_size_mpc, double h0, double omega_m, double omega_lambda) {
    auto b = half_vectors(vectors, n);
    const std::size_t size = static_cast<std::size_t>(n)*n*n;
    const int dims[3] = {n, n, n};
    RayIntegrationConfig config{1, cell_size_mpc, h0/SPEED_OF_LIGHT_KM_PER_S, omega_m, omega_lambda, 100};
    raytracing_ode(state, rhs, potential, b.data(), b.data()+size, b.data()+2*size, dims, config);
}
void snapshot_reference_step(const float* potential, const float* vectors, int n, double* state,
                             double dt, double cell_size_mpc, double h0, double omega_m, double omega_lambda) {
    auto b = half_vectors(vectors, n);
    const std::size_t size = static_cast<std::size_t>(n)*n*n;
    const int dims[3] = {n, n, n};
    RayIntegrationConfig config{1, cell_size_mpc, h0/SPEED_OF_LIGHT_KM_PER_S, omega_m, omega_lambda, 100};
    rk4_step(potential, b.data(), b.data()+size, b.data()+2*size, dims, state, dt, config);
}
bool snapshot_reference_trace(const float* potential, const float* vectors, int n, double* state,
                              double target_z, int max_steps, double cell_size_mpc, double h0,
                              double omega_m, double omega_lambda, double* reached_z) {
    auto b = half_vectors(vectors, n);
    const std::size_t size = static_cast<std::size_t>(n)*n*n;
    const int dims[3] = {n, n, n};
    RayIntegrationConfig config{target_z, cell_size_mpc, h0/SPEED_OF_LIGHT_KM_PER_S, omega_m, omega_lambda, max_steps};
    return raytracing(potential, b.data(), b.data()+size, b.data()+2*size, dims, config, state, reached_z);
}
