#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lightcone {
enum class Ordering { Nested, Ring };
bool valid_nside(std::int64_t nside);
void nested_angles(std::int64_t nside, int batch, std::vector<double>& theta, std::vector<double>& phi);
std::vector<float> ring_to_nested(const std::vector<float>& data, std::int64_t nside);
std::vector<float> nested_to_ring(const std::vector<float>& data, std::int64_t nside);
void write_nested_map(const std::vector<float>& nested, std::int64_t nside,
                      Ordering ordering, const std::string& path);
} // namespace lightcone
