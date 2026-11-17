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
    range_angles(nside,pixel_range(nside,batch),theta,phi);
}
PixelRange pixel_range(std::int64_t nside,int batch,int level,std::int64_t part) {
    const auto all=pixel_count(nside);
    if(batch < -1 || batch > 11 || level<0 || part<0) throw std::runtime_error("Invalid base face or subdivision");
    int order=0; for(auto n=nside;n>1;n/=2) ++order;
    if(level>order || (batch<0 && (level || part))) throw std::runtime_error("Subdivision requires an explicit face and level <= log2(Nside)");
    const std::int64_t parts=1LL<<(2*level);
    if(part>=parts) throw std::runtime_error("Part is outside the subdivision");
    const auto count=batch<0 ? static_cast<std::int64_t>(all) : nside*nside/parts;
    return {batch<0 ? 0 : batch*nside*nside+part*count,count};
}
void range_angles(std::int64_t nside,PixelRange range,std::vector<double>& theta,std::vector<double>& phi) {
    const auto all=pixel_count(nside);
    if(range.first<0 || range.count<=0 || static_cast<std::uint64_t>(range.first)>all || static_cast<std::uint64_t>(range.count)>all-range.first)
        throw std::runtime_error("Invalid NESTED interval");
    const auto count=range.count, start=range.first;
    checked_multiply(count, sizeof(double));
    theta.resize(count); phi.resize(count);
    #pragma omp parallel for
    for (std::int64_t i = 0; i < count; ++i) pix2ang_nest64(nside, start+i, &theta[i], &phi[i]);
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
