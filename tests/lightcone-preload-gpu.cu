// Run the production executable's host orchestration and tracing kernel directly.
#define main lightcone_program_main
#include "../lightcone-raytracer.cu"
#undef main
#include "test_support.hpp"

int main() {
    const auto original=std::filesystem::current_path();
    try {
        TempDirectory temp;
        const int n=32; const auto base=(temp.path/"lc").string();
        for(int field=0;field<4;++field) {
            std::vector<float> values(n*n*n);
            for(int z=0;z<n;++z) for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
                const double pi=3.14159265358979323846;
                const double q[3]={double(x),double(y),double(z)};
                values[x+n*(y+n*z)]=field?float(1e-6*std::sin(2*pi*q[field%3]/n)):float(1e-5*(std::sin(2*pi*x/n)+std::cos(2*pi*y/n)));
            }
            write_grid(base+(field?"_B"+std::to_string(field):"_phi")+".h5",n,values,field==2);
        }
        std::vector<lightcone::InfoRecord> records;
        for(int k=0;k<=31;++k) {const double tau=.7+k*.01;records.push_back({k,tau,1/(1+2*(1-tau)),-1,2,{-1,2}});}
        write_info(base+"_info.bin",records);
        for(bool staggered:{false,true}) for(int face=0;face<12;++face) {
            Options o;o.base=base;o.info=base+"_info.bin";o.nside=4;o.batch=face;o.thickness=1;o.observer_tau=1;o.target_z=.1;
            o.geometry.vertex[0]=.98;o.geometry.vertex[1]=-.04;o.geometry.vertex[2]=.5;
            o.time_derivatives=lightcone::TimeDerivativeMode::Zero;o.compact=true;o.run_id=std::string(64,'a');o.max_lookback=5;
            o.staggered=staggered;o.slab_cells=3;o.padding_cells=8;o.io_budget=32768;
            auto execute=[&](const std::string& label) {
                const auto directory=temp.path/(std::to_string(staggered)+"-"+std::to_string(face)+"-"+label);
                std::filesystem::create_directory(directory);std::filesystem::current_path(directory);
                check(run(o)==0,"Production GPU run succeeds");lightcone::check_part((directory/"ray-part.h5").string());
                return (directory/"ray-part.h5").string();
            };
            const auto full=execute("full");
            o.ziggurat=true;o.io_mode="regions";lightcone::compare_part(execute("regions"),full);
            o.io_mode="tiles";lightcone::compare_part(execute("tiles"),full);
            o.subdivision_level=1;
            for(int part=0;part<4;++part) {o.part=part;lightcone::compare_part(execute("part"+std::to_string(part)),full);}
        }
        std::filesystem::current_path(original);
        std::cout<<"Production preload GPU controls passed; physics="<<raytracer::PHYSICS_JSON<<std::endl;return 0;
    } catch(const std::exception& e) {std::filesystem::current_path(original);std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}
}
