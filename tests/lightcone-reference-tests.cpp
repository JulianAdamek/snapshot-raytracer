#include "test_support.hpp"
#include "snapshot-reference.hpp"
#include <array>
using namespace lightcone;
namespace {
constexpr double cell = 100, h0 = 70, om = 0.3, ol = 0.7;
double scale_factor(double tau, int n) {
    double a = 1;
    const int steps = std::max(1, static_cast<int>(std::ceil(std::fabs(tau-1)/0.0001)));
    const double dt = (tau-1)/steps, factor = n*cell*h0/299792.458;
    auto f = [&](double value) { return factor*std::sqrt(om*value+ol*std::pow(value, 4)); };
    for (int i = 0; i < steps; ++i) {
        const double k1 = f(a), k2 = f(a+dt*k1/2), k3 = f(a+dt*k2/2), k4 = f(a+dt*k3);
        a += dt*(k1+2*k2+2*k3+k4)/6;
    }
    return a;
}
void set_background(Fixture& f, double spacing) {
    auto records = legacy_records(f.thickness, spacing);
    for (auto& r : records) r.a = scale_factor(r.tau, f.n);
    f.metadata = build_metadata(records, f.thickness, f.geometry);
}
void static_shear_transport() {
    // Independent Jacobi-map check: J_ss + F J_s = T J implies
    // H_s = -F H + D^2 T_TF for H = D^2 (J_s J^-1)_TF.
    // With zero anisotropic tide, the weighted optical shear transports as -F sigma.
    // Test both implementations separately so a shared sign error cannot pass.
    for (bool vector : {false, true}) {
        Fixture f(32, 3);
        set_background(f, 0.5);
        const double alpha = std::ldexp(1.0, -20), beta = std::ldexp(1.0, -18);
        f.fill([&](int field, double, double x, double y, double z) {
            if (!vector) return field ? 0.0f : static_cast<float>(alpha*z);
            // Divergence-free, with rotational symmetry around the launched z ray.
            if (field == 1) return static_cast<float>(-0.5*beta*(x-16));
            if (field == 2) return static_cast<float>(-0.5*beta*(y-16));
            if (field == 3) return static_cast<float>(beta*(z-16));
            return 0.0f;
        });
        const std::size_t size = static_cast<std::size_t>(f.n)*f.n*f.n;
        std::vector<float> vectors;
        std::vector<std::vector<std::uint16_t>> packed(9, std::vector<std::uint16_t>(size));
        for (int field = 1; field < 4; ++field) {
            vectors.insert(vectors.end(), f.fields[field*3].begin(), f.fields[field*3].end());
            for (int slot = 0; slot < 3; ++slot) {
                auto& data = packed[(field-1)*3+slot];
                snapshot_pack_b(f.fields[field*3+slot].data(), data.data(), size);
                f.grids[field*3+slot].values = nullptr;
                f.grids[field*3+slot].half_values = data.data();
            }
        }
        for (auto mode : {TimeDerivativeMode::FiniteDifference, TimeDerivativeMode::Zero}) {
            auto view = f.view(); view.time_derivatives = mode;
            double state[STATE_SIZE], snapshot[STATE_SIZE], rhs[STATE_SIZE], snapshot_rhs[STATE_SIZE] = {};
            check(initialize(view, 1.0, 0.0, 0.0, state) == Status::Ok, "Shear transport observer");
            state[STATE_DA] = 2.0;
            state[STATE_SIGMA1] = 1e-4; state[STATE_SIGMA2] = -2e-4;
            copy_state(snapshot, state); snapshot[20] = 0;
            check(ode(view, state, rhs) == Status::Ok, "Shear transport RHS");
            snapshot_reference_ode(f.fields[0].data(), vectors.data(), f.n, snapshot, snapshot_rhs, cell, h0, om, ol);
            // Scalar F uses the existing weak-field spatial prefactor approximation.
            // In the pure vector case F = d(log ktilde0)/ds = beta exactly.
            const double transport = vector ? beta : -4*alpha;
            for (int component : {STATE_SIGMA1, STATE_SIGMA2}) {
                const double expected = -transport*state[component];
                near(rhs[component], expected, 1e-16, "Light-cone shear has Jacobi transport sign");
                near(snapshot_rhs[component], expected, 1e-16, "Snapshot shear has Jacobi transport sign");
            }
            if (vector) {
                near(rhs[STATE_LOG_KTILDE0], beta, 1e-16, "Analytic constant vector energy transport");
                const double initial_log_k = state[STATE_LOG_KTILDE0];
                const double initial[2] = {state[STATE_SIGMA1], state[STATE_SIGMA2]};
                for (int step = 0; step < 16; ++step) {
                    check(rk4_step(view, state, 0.25) == Status::Ok, "Source-free vector shear step");
                    snapshot_reference_step(f.fields[0].data(), vectors.data(), f.n, snapshot, 0.25, cell, h0, om, ol);
                    const double elapsed = (step+1)*0.25;
                    for (int m = 0; m < 2; ++m) {
                        const int component = STATE_SIGMA1+m;
                        const double exact = initial[m]*std::exp(-beta*elapsed);
                        near(state[component], exact, 5e-15, "Light-cone exponential transport");
                        near(snapshot[component], exact, 5e-15, "Snapshot exponential transport");
                        near(std::exp(state[STATE_LOG_KTILDE0]-initial_log_k)*state[component], initial[m], 5e-15,
                             "Light-cone ktilde0 times weighted shear is conserved");
                        near(std::exp(snapshot[STATE_LOG_KTILDE0]-initial_log_k)*snapshot[component], initial[m], 5e-15,
                             "Snapshot ktilde0 times weighted shear is conserved");
                    }
                    near(state[STATE_ISW], 0, 0, "Static vector transport produces zero ISW");
                }
            }
        }
    }
}
void snapshot_reduction() {
    for (int t : {1, 2, 3}) {
        Fixture f(16, t);
        set_background(f, 0.5);
        // All cubes contain the same snapshot, including regions outside recorded support.
        for (int field = 0; field < 4; ++field) for (int slot = 0; slot < t; ++slot)
            for (int z = 0; z < f.n; ++z) for (int y = 0; y < f.n; ++y) for (int x = 0; x < f.n; ++x) {
                const float value = field == 0 ? 5e-7f*(std::sin(0.4*x)+std::cos(0.3*y)+std::sin(0.2*z))
                                                : 1e-8f*std::cos(0.3*(x+field*y+z));
                f.fields[field*t+slot][x+f.n*(y+f.n*z)] = value;
            }
        const std::size_t size = static_cast<std::size_t>(f.n)*f.n*f.n;
        std::vector<float> reference_vectors;
        std::vector<std::vector<std::uint16_t>> packed(3*t, std::vector<std::uint16_t>(size));
        for (int field = 1; field < 4; ++field) {
            reference_vectors.insert(reference_vectors.end(), f.fields[field*t].begin(), f.fields[field*t].end());
            for (int slot = 0; slot < t; ++slot) {
                auto& data = packed[(field-1)*t+slot];
                snapshot_pack_b(f.fields[field*t+slot].data(), data.data(), size);
                f.grids[field*t+slot].values = nullptr;
                f.grids[field*t+slot].half_values = data.data();
            }
        }
        auto v = f.view();
        if (f.thickness == 1) v.time_derivatives = TimeDerivativeMode::Zero;
        for (const auto& angles : {std::array<double,2>{1.2,0.3}, {0.7,2.1}, {2.3,4.5}}) {
            double current[STATE_SIZE], original[STATE_SIZE];
            check(initialize(v, 1.0, angles[0], angles[1], current) == Status::Ok, "Static observer initialization");
            copy_state(original, current); original[20] = 0;
            for (int k = 0; k < 5; ++k) {
                check(rk4_step(v, current, 0.5) == Status::Ok, "Static light-cone RK4");
                near(current[STATE_ISW],0,0,"Frozen snapshot ISW is exactly zero");
                snapshot_reference_step(f.fields[0].data(), reference_vectors.data(), f.n, original, 0.5, cell, h0, om, ol);
                for (int component = 0; component < 20; ++component)
                    near(current[component], original[component], 8e-10*std::max(1.0,std::fabs(original[component])),
                         "Snapshot reduction at matched integration time, component "+std::to_string(component));
            }
            float maps[6], expected[6];
            observables(v, current, angles[0], angles[1], maps);
            observables(v, original, angles[0], angles[1], expected);
            for (int map = 0; map < 6; ++map) near(maps[map], expected[map], 2e-9, "Static observable reduction");
            check(initialize(v, 1.0, angles[0], angles[1], current) == Status::Ok, "Endpoint initialization");
            copy_state(original, current); original[20] = 0;
            double reached, reference_z;
            check(trace(v, 0.06, current, reached) == Status::Ok, "Static light-cone endpoint");
            check(snapshot_reference_trace(f.fields[0].data(), reference_vectors.data(), f.n, original,
                                           0.06, 30, cell, h0, om, ol, &reference_z), "Snapshot endpoint");
            check(reached >= 0.06 && reached-0.06 <= FINAL_Z_TOL && reference_z >= 0.06 && reference_z-0.06 <= FINAL_Z_TOL,
                  "Both endpoint redshifts meet tolerance");
            near(current[STATE_DA], original[STATE_DA], 0.012, "Endpoint distance accounts for table and stopping error");
        }
    }
}
void background_convergence() {
    Fixture f(16, 1);
    for (auto& field : f.fields) std::fill(field.begin(), field.end(), 0.0f);
    double previous_error = 1, coarse_error = 0;
    for (double spacing : {0.5, 0.25, 0.125, 0.0625}) {
        set_background(f, spacing);
        auto v = f.view();
        if (f.thickness == 1) v.time_derivatives = TimeDerivativeMode::Zero;
        double error = 0;
        for (int i = 0; i <= 100; ++i) {
            const double tau = 0.7+i*0.004;
            double a;
            check(background_at(v, tau, a) == Status::Ok, "LCDM table interpolation");
            error = std::max(error, std::fabs(a-scale_factor(tau, f.n)));
        }
        if (spacing == 0.5) coarse_error = error;
        check(error < previous_error, "LCDM background converges with table refinement");
        previous_error = error;
        double y[STATE_SIZE], reference[STATE_SIZE], z, reference_z;
        check(initialize(v, 1.0, 1.2, 0.3, y) == Status::Ok, "Convergence observer");
        copy_state(reference, y); reference[20] = 0;
        check(trace(v, 0.1, y, z) == Status::Ok, "Convergence endpoint");
        std::vector<float> vectors(3ULL*f.n*f.n*f.n, 0.0f);
        check(snapshot_reference_trace(f.fields[0].data(), vectors.data(), f.n, reference,
                                      0.1, 30, cell, h0, om, ol, &reference_z), "Convergence snapshot endpoint");
        near(y[STATE_DA], reference[STATE_DA], 0.015, "Converged endpoint maps within redshift stopping uncertainty");
        std::cout << "LCDM spacing " << spacing << ": background error=" << error
                  << ", endpoint distance difference=" << std::fabs(y[STATE_DA]-reference[STATE_DA]) << '\n';
    }
    check(previous_error < coarse_error/100, "Background error decreases by more than 100x");
}
}
int main() {
    try {
        static_shear_transport(); snapshot_reduction(); background_convergence();
        std::cout << "Snapshot reference checks passed (" << checks << "). No GPU calls made.\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
