#pragma once
#include <cstddef>
#include <cstdint>
void snapshot_pack_b(const float* values, std::uint16_t* packed, std::size_t count);
void snapshot_reference_ode(const float* potential, const float* vectors, int n, const double* state,
                            double* rhs, double cell_size_mpc, double h0, double omega_m, double omega_lambda);
void snapshot_reference_step(const float* potential, const float* vectors, int n, double* state,
                             double dt, double cell_size_mpc, double h0, double omega_m, double omega_lambda);
bool snapshot_reference_trace(const float* potential, const float* vectors, int n, double* state,
                              double target_z, int max_steps, double cell_size_mpc, double h0,
                              double omega_m, double omega_lambda, double* reached_z);
