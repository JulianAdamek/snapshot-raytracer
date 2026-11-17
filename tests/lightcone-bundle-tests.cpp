#include "test_support.hpp"
#include "analytic_fields.hpp"
using namespace lightcone;
namespace {
void analytic_bundles() {
    Fixture f(64,3);
    for(const auto parameters : {std::array<double,3>{.002,0.,0.}, {.0002,3e-7,1.2}, {0.,1e-6,.4}}) {
        AnalyticFields fields{64,parameters[0],parameters[1],parameters[2]};
        const auto view=f.view(); double observer[STATE_SIZE],endpoint[STATE_SIZE];
        check(initialize_with_sampler(view,1.,1.2,.3,observer,fields)==Status::Ok,"Analytic observer");
        copy_state(endpoint,observer);
        const double depth=16.;
        for(double s=0;s<depth;) {const double dt=std::min(.05,depth-s);check(rk4_step_with_sampler(view,endpoint,dt,fields)==Status::Ok,"Analytic optical step");s+=dt;}
        const auto bundle=bundle_result(fields,observer,endpoint,depth,.025,2e-5);
        const auto finer=bundle_result(fields,observer,endpoint,depth,.0125,1e-5);
        near(bundle.distance,finer.distance,2e-7,"Neighbor-ray angular and step convergence");
        near(bundle.epsilon1,finer.epsilon1,1e-7,"Bundle shape convergence");
        std::cout<<"Bundle residuals: D="<<endpoint[STATE_DA]-finer.distance
                 <<" eps="<<endpoint[STATE_ELLIPTICITY1]-finer.epsilon1<<','<<endpoint[STATE_ELLIPTICITY2]-finer.epsilon2
                 <<" omega="<<endpoint[STATE_ROTATION]-finer.rotation<<'\n';
        near(endpoint[STATE_DA],finer.distance,2e-6,"Optical distance agrees with independent geodesic bundle");
        near(endpoint[STATE_ELLIPTICITY1],finer.epsilon1,8e-7,"Optical ellipticity 1 agrees with bundle");
        near(endpoint[STATE_ELLIPTICITY2],finer.epsilon2,8e-7,"Optical ellipticity 2 agrees with bundle");
        near(endpoint[STATE_ROTATION],finer.rotation,2e-7,"Optical rotation agrees with bundle");
        near(endpoint[STATE_ISW],finer.isw,2e-10,"ISW agrees with independent geodesic quadrature");
        for(int screen=0;screen<2;++screen) {
            double dot=0.,norm=0.;
            for(int i=0;i<3;++i) {dot+=endpoint[STATE_NX+i]*endpoint[STATE_E1X+3*screen+i];norm+=std::pow(endpoint[STATE_E1X+3*screen+i],2);}
            near(dot,0.,1e-9,"Sachs screen remains orthogonal to ray");
            near(norm,1.,1e-9,"Sachs screen norm is preserved");
        }
        double mutual=0.;for(int i=0;i<3;++i)mutual+=endpoint[STATE_E1X+i]*endpoint[STATE_E2X+i];
        near(mutual,0.,1e-9,"Screen basis vectors remain mutually orthogonal");
    }
}
double writer_radius(int x,int y,int z,int n,const double offset[3]) {
    // Literal unfused writer geometry, independent of sampler/index construction.
    const double dx=(x+offset[0])/n-.5,dy=(y+offset[1])/n-.5,dz=(z+offset[2])/n-.5;
    volatile double xx=dx*dx,yy=dy*dy,zz=dz*dz;
    volatile double xy=xx+yy;
    return std::sqrt(xy+zz);
}
void gridded_evolution() {
    TempDirectory tmp;
    for(bool staggered : {false,true}) {
        double previous=1.;
        for(int n : {24,48,96}) {
            Fixture f(n,3,staggered,8./n); // Half-step coverage exceeds the complete stencil radius.
            AnalyticFields fields{n,.0002,3e-7,1.2};
            // Replay time-labelled writer samples. Do not use record_at/node_at
            // to construct the expected input cubes.
            for(int field=0;field<4;++field) for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x) {
                const auto& grid=f.grids[field*3];
                const double r=writer_radius(x,y,z,n,grid.node_offset);
                int record[3]={-1,-1,-1};
                for(int k=0;k<static_cast<int>(f.metadata.records.size());++k)for(int slot=0;slot<3;++slot) {
                    const auto& b=f.metadata.records[k].boundaries;
                    if(r>=b[slot] && r<b[slot+1]) record[slot]=k;
                }
                const double q[3]={x+grid.node_offset[0],y+grid.node_offset[1],z+grid.node_offset[2]};
                for(int slot=0;slot<3;++slot) if(record[slot]>=0)
                    f.fields[field*3+slot][x+n*(y+n*z)]=static_cast<float>(fields.jet(field,f.metadata.times[record[slot]],q).value);
            }
            const auto base=(tmp.path/(std::to_string(n)+(staggered ? "s":"c"))).string();f.write(base);
            const auto metadata=build_metadata(read_info(base+"_info.bin",3),3,f.geometry);
            Reader reader(base,3);
            std::vector<std::vector<float>> resident(12);auto grids=f.grids;
            const std::int64_t origin[3]={0,0,0};const int extent[3]={n,n,n};
            for(int field=0;field<4;++field)for(int slot=0;slot<3;++slot) {
                resident[field*3+slot]=reader.read_region(field,slot,origin,extent);
                grids[field*3+slot].values=resident[field*3+slot].data();
            }
            const auto view=host_view(metadata,n,f.geometry,grids.data());
            double observed[STATE_SIZE],grid_state[STATE_SIZE],analytic_state[STATE_SIZE];
            check(initialize_with_sampler(view,1.,1.2,.3,observed,fields)==Status::Ok,"Wave bundle observer");
            check(initialize(view,1.,1.2,.3,grid_state)==Status::Ok,"HDF5 evolving observer");
            copy_state(analytic_state,observed);
            const double depth=.25*n;
            for(double s=0;s<depth;) {
                const double dt=std::min(.1,depth-s);
                const auto status=rk4_step(view,grid_state,dt);
                check(status==Status::Ok,"Evolving HDF5 integration n="+std::to_string(n)+" s="+std::to_string(s)+" status="+status_name(status));
                check(rk4_step_with_sampler(view,analytic_state,dt,fields)==Status::Ok,"Analytic central screen");s+=dt;
            }
            const auto bundle=bundle_result(fields,observed,analytic_state,depth,.05,1e-5);
            const double error=std::max({std::fabs(grid_state[STATE_DA]-bundle.distance)/n,
                std::fabs(grid_state[STATE_ELLIPTICITY1]-bundle.epsilon1),std::fabs(grid_state[STATE_ELLIPTICITY2]-bundle.epsilon2)});
            std::cout<<"Evolving HDF5 bundle: n="<<n<<" staggered="<<staggered<<" error="<<error<<'\n';
            check(error<previous*.8,"Real sampling converges toward independent bundle with resolution");
            previous=error;
            near(grid_state[STATE_ISW],bundle.isw,3e-7,"Evolving HDF5 ISW integral");
            if(n==96) check(error<5e-6,"Resolved evolving plane-wave optical accuracy");
        }
    }
}
}
int main(){try{analytic_bundles();gridded_evolution();std::cout<<"Independent evolving bundle checks passed ("<<checks<<").\n";return 0;}
catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
