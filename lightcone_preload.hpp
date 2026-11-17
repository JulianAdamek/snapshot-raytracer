#pragma once
#include "lightcone.hpp"
#include <vector>

namespace lightcone {
// Coordinates in the source band and destination slab, respectively. Both are
// local XYZ coordinates; periodic images are resolved by the host planner.
struct PreloadCopy {
    std::size_t slab;
    int source[3], destination[3], extent[3];
};
struct PreloadBand {
    std::int64_t origin[3];
    int extent[3];
    std::vector<PreloadCopy> copies;
};
struct PreloadPlan {
    std::vector<PreloadBand> bands;
    std::size_t peak_voxels = 0;
};
int preload_tile_height(int n, std::size_t budget, std::size_t bytes_per_voxel);
PreloadPlan tiled_preload_plan(int n, const SlabLayout& layout, int tile_height);
PreloadPlan regional_preload_plan(const SlabLayout& layout);
} // namespace lightcone
