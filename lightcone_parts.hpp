#pragma once
#include "lightcone_maps.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lightcone {
constexpr std::size_t OUTPUT_BLOCK = 1048576;
struct PartInfo {
    std::int64_t nside=0, first=0, count=0, part=0;
    int face=-1, level=0;
    double target_z=0, redshift_tolerance=0;
    std::string run_id, physics;
};
class PartWriter {
public:
    PartWriter(const std::string& path,const PartInfo& info);
    ~PartWriter();
    void map(int map,std::int64_t offset,std::size_t count,const float* values);
    void statuses(std::int64_t offset,std::size_t count,const int* values);
    void finish();
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
PartInfo inspect_part(const std::string& path);
void check_part(const std::string& path,bool frozen=true);
void compare_part(const std::string& path,const std::string& reference);
void import_face(const std::string& reference,const std::string& path,const PartInfo& info);
void assemble_parts(const std::vector<std::string>& paths,const std::string& directory);
} // namespace lightcone
