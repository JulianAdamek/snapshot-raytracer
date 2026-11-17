#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lightcone {
enum class Ordering { Nested, Ring };
bool valid_nside(std::int64_t nside);
struct PixelRange { std::int64_t first, count; };
PixelRange pixel_range(std::int64_t nside, int batch, int level=0, std::int64_t part=0);
void range_angles(std::int64_t nside, PixelRange range, std::vector<double>& theta, std::vector<double>& phi);
void nested_angles(std::int64_t nside, int batch, std::vector<double>& theta, std::vector<double>& phi);
std::vector<float> ring_to_nested(const std::vector<float>& data, std::int64_t nside);
std::vector<float> nested_to_ring(const std::vector<float>& data, std::int64_t nside);
void write_nested_map(const std::vector<float>& nested, std::int64_t nside,
                      Ordering ordering, const std::string& path);
} // namespace lightcone
