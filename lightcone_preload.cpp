#include "lightcone_preload.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace lightcone {
int preload_tile_height(int n, std::size_t budget, std::size_t bytes_per_voxel) {
    if (n <= 0 || !bytes_per_voxel) throw std::runtime_error("Invalid tiled preload dimensions");
    const auto plane = checked_multiply(checked_multiply(n,n),bytes_per_voxel);
    if (budget < plane) throw std::runtime_error("I/O buffer cannot hold one source XY plane (requires "+std::to_string(plane)+" bytes)");
    return static_cast<int>(std::min<std::size_t>(n,budget/plane));
}
namespace {
struct Span { int file, local, length; };
struct Piece { std::size_t slab; int source[3], destination[3], extent[3]; };
std::vector<Span> split(std::int64_t origin,int extent,int n) {
    if (extent <= 0) throw std::runtime_error("Invalid preload extent");
    std::vector<Span> result;
    int file = static_cast<int>((origin%n+n)%n);
    for (int local=0;local<extent;) {
        const int length=std::min(n-file,extent-local);
        result.push_back({file,local,length}); local+=length; file=0;
    }
    return result;
}
}
PreloadPlan regional_preload_plan(const SlabLayout& layout) {
    PreloadPlan result;
    for (std::size_t i=0;i<layout.slabs.size();++i) {
        const auto& s=layout.slabs[i]; PreloadBand b{};
        PreloadCopy c{}; c.slab=i;
        for(int a=0;a<3;++a) { b.origin[a]=s.origin[a]; b.extent[a]=c.extent[a]=s.extent[a]; }
        b.copies.push_back(c);
        result.peak_voxels=std::max(result.peak_voxels,checked_volume(b.extent));
        result.bands.push_back(std::move(b));
    }
    return result;
}
PreloadPlan tiled_preload_plan(int n,const SlabLayout& layout,int tile_height) {
    if (n<=0 || tile_height<=0 || layout.slabs.empty()) throw std::runtime_error("Invalid tiled preload plan");
    std::vector<Piece> pieces;
    for (std::size_t i=0;i<layout.slabs.size();++i) {
        const auto& s=layout.slabs[i];
        const auto xs=split(s.origin[0],s.extent[0],n), ys=split(s.origin[1],s.extent[1],n), zs=split(s.origin[2],s.extent[2],n);
        for (auto z:zs) for(auto y:ys) for(auto x:xs)
            pieces.push_back({i,{x.file,y.file,z.file},{x.local,y.local,z.local},{x.length,y.length,z.length}});
    }
    PreloadPlan result;
    for (int z=0;z<n;) {
        const int depth=std::min(tile_height,n-z), end=z+depth;
        std::vector<std::pair<int,int>> ys;
        for(const auto& p:pieces) if(p.source[2]<end && p.source[2]+p.extent[2]>z)
            ys.emplace_back(p.source[1],p.source[1]+p.extent[1]);
        std::sort(ys.begin(),ys.end());
        std::vector<std::pair<int,int>> merged;
        for(auto y:ys) {
            if(!merged.empty() && y.first<=merged.back().second) merged.back().second=std::max(merged.back().second,y.second);
            else merged.push_back(y);
        }
        for(auto y:merged) {
            PreloadBand b{{0,y.first,z},{n,y.second-y.first,depth},{}};
            for(const auto& p:pieces) {
                const int first[3]={p.source[0],std::max(p.source[1],y.first),std::max(p.source[2],z)};
                const int last[3]={p.source[0]+p.extent[0],std::min(p.source[1]+p.extent[1],y.second),std::min(p.source[2]+p.extent[2],end)};
                if(first[1]>=last[1] || first[2]>=last[2]) continue;
                PreloadCopy c{}; c.slab=p.slab;
                for(int a=0;a<3;++a) {
                    c.source[a]=first[a]-static_cast<int>(b.origin[a]);
                    c.destination[a]=p.destination[a]+first[a]-p.source[a];
                    c.extent[a]=last[a]-first[a];
                }
                b.copies.push_back(c);
            }
            result.peak_voxels=std::max(result.peak_voxels,checked_volume(b.extent));
            result.bands.push_back(std::move(b));
        }
        z=end;
    }
    return result;
}
} // namespace lightcone
