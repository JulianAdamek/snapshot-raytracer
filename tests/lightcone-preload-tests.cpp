#include "test_support.hpp"
#include "lightcone_preload.hpp"
#include "lightcone_parts.hpp"
#include <chealpix.h>
#include <cstring>
#include <set>
using namespace lightcone;
namespace {
std::vector<float> apply(Reader& reader,int field,int slot,const SlabLayout& layout,const PreloadPlan& plan,int n,bool tiled) {
    std::vector<float> out(layout.voxels,NAN); std::vector<int> writes(layout.voxels),reads(n*n*n);
    for(const auto& b:plan.bands) {
        if(tiled) check(b.origin[0]==0 && b.extent[0]==n,"Tiled reads use full x rows");
        auto values=reader.read_region(field,slot,b.origin,b.extent);
        if(tiled) for(int z=0;z<b.extent[2];++z) for(int y=0;y<b.extent[1];++y) for(int x=0;x<n;++x)
            check(++reads[x+n*(b.origin[1]+y+n*(b.origin[2]+z))]==1,"Canonical source voxel read at most once");
        for(const auto& c:b.copies) {
            const auto& d=layout.slabs[c.slab];
            for(int z=0;z<c.extent[2];++z) for(int y=0;y<c.extent[1];++y) for(int x=0;x<c.extent[0];++x) {
                auto src=c.source[0]+x+b.extent[0]*(c.source[1]+y+b.extent[1]*(c.source[2]+z));
                auto dst=d.offset+c.destination[0]+x+d.strides[1]*(c.destination[1]+y)+d.strides[2]*(c.destination[2]+z);
                check(dst<out.size() && ++writes[dst]==1,"Every slab voxel receives exactly one write"); out[dst]=values[src];
            }
        }
    }
    for(auto v:writes) check(v==1,"No destination voxel omitted");
    return out;
}
void ranges() {
    for(int n:{1,2,4,16}) for(int level=0;(1<<level)<=n;++level) {
        std::int64_t next=0;
        for(int f=0;f<12;++f) for(int p=0;p<(1<<(2*level));++p) {
            auto r=pixel_range(n,f,level,p); check(r.first==next,"Subregions exactly tile sky"); next+=r.count;
            std::vector<double>a,b; range_angles(n,r,a,b);
            for(std::int64_t j=0;j<r.count;++j) { double t,ph;pix2ang_nest64(n,r.first+j,&t,&ph);check(t==a[j] && ph==b[j],"Global NESTED directions unchanged"); }
        }
        check(next==12*n*n,"Full twelve-face coverage");
    }
    rejects([]{pixel_range(4,-1,1,0);},"No implicit face subdivision");
    rejects([]{pixel_range(4,0,3,0);},"No level beyond Nside order");
    rejects([]{pixel_range(4,0,1,4);},"No out-of-range part");
    rejects([]{preload_tile_height(32,1024,8);},"Reject insufficient plane budget");
    near(preload_tile_height(13,13*13*8*3,8),3,0,"Tile budget accounts for simultaneous buffers");
}
void preload() {
    TempDirectory temp; Fixture f(13,3,true); f.geometry.vertex[0]=0.98; f.geometry.vertex[1]=-0.04;
    f.fill([](int field,double t,double x,double y,double z){return float(1e-6*(field+1)*(t+1)*(x*x+2*y*y+z*z));});
    const auto base=(temp.path/"lc").string(); f.write(base);
    // Chunked sources must retain the same value semantics as contiguous files.
    {
        const auto path=base+"_B1_0.h5"; const hsize_t dims[3]={13,13,13},chunk[3]={3,3,3};
        const auto file=H5Fcreate(path.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
        const auto space=H5Screate_simple(3,dims,nullptr),prop=H5Pcreate(H5P_DATASET_CREATE);
        check(H5Pset_chunk(prop,3,chunk)>=0,"Chunked source properties");
        const auto data=H5Dcreate2(file,"/data",H5T_IEEE_F32LE,space,H5P_DEFAULT,prop,H5P_DEFAULT);
        check(H5Dwrite(data,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,f.fields[3].data())>=0,"Write chunked source");
        H5Dclose(data);H5Pclose(prop);H5Sclose(space);H5Fclose(file);
    }
    for(int face=0;face<12;++face) {
        std::vector<double>a,b; range_angles(4,pixel_range(4,face,1,face%4),a,b);
        auto layout=build_ziggurat(13,f.geometry,face,a,b,18.5,3,2);
        const auto regional=regional_preload_plan(layout), tiled=tiled_preload_plan(13,layout,3);
        for(int field=0;field<4;++field) for(int slot=0;slot<3;++slot) {
            Reader reader(base,3);
            auto x=apply(reader,field,slot,layout,regional,13,false),y=apply(reader,field,slot,layout,tiled,13,true);
            check(std::memcmp(x.data(),y.data(),x.size()*sizeof(float))==0,"Float32/64 staged values, including NaNs and aliases, are bit identical");
        }
        check(tiled.peak_voxels<=13*13*3,"Partial final tiles respect budget");
    }
    // Evolving data, both offsets and both geometry conventions: use actual sampler
    // outputs to catch a physically misplaced copy even if data counts agree.
    for(bool staggered:{false,true}) for(auto geometry:{GeometryConvention::Legacy,GeometryConvention::GasgrdV1}) {
        Fixture g(32,3,staggered); g.geometry.convention=geometry;
        g.fill([](int field,double t,double x,double y,double z){return float((field+1)*1e-7*(t+1)*(x*x+2*y*y+z*z));});
        const auto path=(temp.path/(std::to_string(staggered)+std::to_string(int(geometry)))).string(); g.write(path);
        // phi-chi combines in float32, without changing a single bit of W.
        for(int slot=0;slot<3;++slot) {
            auto raw=g.fields[slot];for(auto& v:raw)v*=2;
            write_grid(path+"_phi_"+std::to_string(slot)+".h5",32,raw);
            write_grid(path+"_chi_"+std::to_string(slot)+".h5",32,raw);
            write_grid(path+"_weyl_"+std::to_string(slot)+".h5",32,g.fields[slot]);
        }
        for(auto scalar:{ScalarSource::PhiChi,ScalarSource::Weyl}) {
            Reader reader(path,3,scalar); std::vector<double>a,b; nested_angles(2,4,a,b);
            auto layout=build_ziggurat(32,g.geometry,4,a,b,5,3,2); auto plan=tiled_preload_plan(32,layout,3);
            std::vector<std::vector<float>> values; auto grids=g.grids;
            for(int field=0;field<4;++field) for(int slot=0;slot<3;++slot) { values.push_back(apply(reader,field,slot,layout,plan,32,true)); grids[field*3+slot].values=values.back().data(); }
            auto full=g.view(),sliced=full;sliced.layout=layout.view();sliced.grids=grids.data();
            for(int field=0;field<4;++field) {
                float v,w,gv[3],gw[3],hv[3][3],hw[3][3];TimeSample tv,tw;
                check(sample(full,field,.95,17.3,16.2,16.8,v,gv,hv,&tv)==Status::Ok,"Full sample available");
                check(sample(sliced,field,.95,17.3,16.2,16.8,w,gw,hw,&tw)==Status::Ok,"Tiled sample available");
                check(v==w && !std::memcmp(gv,gw,sizeof(gv)) && !std::memcmp(hv,hw,sizeof(hv)) && !std::memcmp(&tv,&tw,sizeof(tv)),"Values, jets and partial-time derivatives identical");
            }
        }
    }
}
PartInfo info(int face,int level,int part) {
    const auto r=pixel_range(4,face,level,part);
    return {4,r.first,r.count,part,face,level,.5,1e-8,std::string(64,'a'),raytracer::PHYSICS_JSON};
}
void write_part(const std::string& path,PartInfo i,int fault=0) {
    PartWriter w(path,i); std::vector<float> v(i.count);std::vector<int> st(i.count,0);
    for(int m=0;m<8;++m) {
        for(int p=0;p<i.count;++p) v[p]=m==7?0:(m==6?.5f:float(m+1+(i.first+p)*1e-5));
        if(fault==1 && m==7) v[0]=1e-5;
        if(fault==2 && m==0) v[0]=NAN;
        w.map(m,0,v.size(),v.data());
    }
    if(fault==2) st[0]=int(Status::InvalidValue);
    w.statuses(0,st.size(),st.data());w.finish();
}
void parts() {
    TempDirectory tmp; std::vector<std::string> paths;
    for(int face=0;face<12;++face) for(int p=0;p<4;++p) {
        auto path=(tmp.path/(std::to_string(face)+"-"+std::to_string(p)+".h5")).string();write_part(path,info(face,1,p));check_part(path); paths.push_back(path);
    }
    const auto parent=(tmp.path/"parent.h5").string();write_part(parent,info(0,0,0));
    for(int p=0;p<4;++p) compare_part(paths[p],parent);
    auto missing=paths;missing.pop_back();rejects([&]{assemble_parts(missing,(tmp.path/"missing").string());},"Reject missing intervals");
    auto duplicate=paths;duplicate.push_back(paths[0]);rejects([&]{assemble_parts(duplicate,(tmp.path/"duplicates").string());},"Reject overlaps");
    auto mixed=info(11,1,3);mixed.run_id=std::string(64,'b'); const auto bad=(tmp.path/"mixed.h5").string();write_part(bad,mixed);auto mismatched=paths;mismatched.back()=bad;
    rejects([&]{assemble_parts(mismatched,(tmp.path/"mixed").string());},"Reject configurations");
    const auto isw=(tmp.path/"isw.h5").string();write_part(isw,info(0,1,0),1);rejects([&]{check_part(isw);},"Reject nonzero frozen ISW");
    const auto invalid=(tmp.path/"invalid.h5").string();write_part(invalid,info(0,1,0),2);rejects([&]{check_part(invalid);},"Reject failed ray status");
    auto high=info(11,0,0);high.nside=16384;high.level=14;high.part=268435455;
    const auto high_range=pixel_range(high.nside,high.face,high.level,high.part); high.first=high_range.first;high.count=high_range.count;
    const auto high_path=(tmp.path/"high.h5").string();write_part(high_path,high);check_part(high_path);
    check(inspect_part(high_path).first==3221225471LL,"Compact metadata preserves pixel indices above INT32_MAX");
    const auto incomplete=(tmp.path/"incomplete.h5").string();
    { PartWriter writer(incomplete,info(0,1,0));rejects([&]{writer.finish();},"Reject incomplete compact writes"); }
    check(!std::filesystem::exists(incomplete) && std::filesystem::exists(incomplete+".partial"),"Interrupted output is not published");
    const auto truncated=tmp.path/"truncated.h5";std::filesystem::copy_file(paths[0],truncated);std::filesystem::resize_file(truncated,64);
    H5E_BEGIN_TRY { rejects([&]{inspect_part(truncated.string());},"Reject truncated compact files"); } H5E_END_TRY;
    assemble_parts(paths,(tmp.path/"full").string());
    for(auto& path:paths) compare_part(path,(tmp.path/"full").string());
    const auto imported=(tmp.path/"imported.h5").string();import_face((tmp.path/"full").string(),imported,info(0,0,0));
    for(int p=0;p<4;++p) compare_part(paths[p],imported);
    const auto seven=tmp.path/"face-seven";std::filesystem::create_directory(seven);
    for(int m=0;m<MAP_COUNT;++m) {
        std::vector<float> data(12*4*4,0);
        for(int k=7*16;k<8*16;++k) data[k]=m==7?0:(m==6?.5f:float(m+1+k*1e-5));
        write_nested_map(data,4,Ordering::Nested,(seven/(std::string(MAP_NAMES[m])+"_batch7.fits")).string());
    }
    const auto imported7=(tmp.path/"imported7.h5").string();import_face(seven.string(),imported7,info(7,0,0));
    for(int p=0;p<4;++p) compare_part(paths[7*4+p],imported7);
}
}
int main() {
    try { ranges();preload();parts();std::cout<<"Preload/range/compact checks passed ("<<checks<<").\n";return 0; }
    catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
