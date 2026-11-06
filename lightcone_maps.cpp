#include "lightcone_maps.hpp"
#include "lightcone.hpp"
#include <chealpix.h>
#include <stdexcept>

namespace lightcone {
bool valid_nside(std::int64_t nside) {
    return nside > 0 && nside <= (1LL<<29) && (nside & (nside-1)) == 0;
}
namespace {
std::size_t pixel_count(std::int64_t nside) {
    if (!valid_nside(nside)) throw std::runtime_error("Nside must be a power of two in 1..2^29");
    return checked_multiply(12, checked_multiply(nside, nside));
}
void check_map(const std::vector<float>& data, std::int64_t nside) {
    if (data.size() != pixel_count(nside)) throw std::runtime_error("Map size does not match Nside");
}
}
void nested_angles(std::int64_t nside, int batch, std::vector<double>& theta, std::vector<double>& phi) {
    const auto all = pixel_count(nside);
    if (batch < -1 || batch > 11) throw std::runtime_error("Invalid base face");
    const auto count = batch < 0 ? all : all/12;
    const auto start = batch < 0 ? 0 : batch*count;
    checked_multiply(count, sizeof(double));
    theta.resize(count); phi.resize(count);
    #pragma omp parallel for
    for (std::size_t i = 0; i < count; ++i) pix2ang_nest64(nside, start+i, &theta[i], &phi[i]);
}
std::vector<float> ring_to_nested(const std::vector<float>& data, std::int64_t nside) {
    check_map(data, nside);
    std::vector<float> output(data.size());
    for (std::size_t nested = 0; nested < data.size(); ++nested) {
        std::int64_t ring; nest2ring64(nside, nested, &ring);
        output[nested] = data[ring];
    }
    return output;
}
std::vector<float> nested_to_ring(const std::vector<float>& data, std::int64_t nside) {
    check_map(data, nside);
    std::vector<float> output(data.size());
    for (std::size_t nested = 0; nested < data.size(); ++nested) {
        std::int64_t ring; nest2ring64(nside, nested, &ring);
        output[ring] = data[nested];
    }
    return output;
}
void write_nested_map(const std::vector<float>& nested, std::int64_t nside,
                      Ordering ordering, const std::string& path) {
    check_map(nested, nside);
    if (ordering == Ordering::Nested) write_healpix_map(nested.data(), nside, path.c_str(), 1, "C");
    else {
        const auto ring = nested_to_ring(nested, nside);
        write_healpix_map(ring.data(), nside, path.c_str(), 0, "C");
    }
}
} // namespace lightcone
