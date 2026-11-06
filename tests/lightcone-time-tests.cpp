#include "test_support.hpp"
#include <array>
using namespace lightcone;
namespace {
// Every slot contains a constant recording time throughout this small fixture.
// Nonuniform times isolate temporal differencing from legacy shell construction.
void set_times(Fixture& f, const std::vector<double>& times) {
    check(times.size() == static_cast<std::size_t>(f.thickness),"Fixture slot times");
    std::vector<InfoRecord> records;
    for (int k = 0; k < f.thickness; ++k) {
        InfoRecord r{k,times[k],0.7+0.1*times[k],-1,1,{}};
        for (int j = 0; j <= f.thickness; ++j) r.boundaries.push_back(10.0*(j-k-0.5));
        records.push_back(r);
    }
    f.metadata = build_metadata(records,f.thickness,f.geometry);
}
void sampling_tests() {
    for (bool staggered : {false,true}) {
        Fixture f(32,3,staggered); set_times(f,{0.5,0.75,1.25});
        const double alpha = std::ldexp(1.0,-20);
        f.fill([&](int field,double tau,double x,double y,double z) {
            return static_cast<float>((field+1)*alpha*(f.n*tau+1)*(x*x+2*y+z*z));
        });
        auto v=f.view();
        for (int field=0;field<4;++field) for (double tau : {0.5,0.7,0.75,1.0,1.25}) {
            float value,g[3],h[3][3]; TimeSample t;
            const double x=16.3,y=15.7,z=17.2, c=(field+1)*alpha;
            check(sample(v,field,tau,x,y,z,value,g,h,&t)==Status::Ok,"Spacetime sample at irregular times");
            near(t.partial_time,c*(x*x+2*y+z*z),3e-8,"Fixed-position partial_eta f");
            near(t.gradient[0],c*2*x,2e-8,"Mixed x/time derivative");
            near(t.gradient[1],c*2,2e-8,"Mixed y/time derivative");
            near(t.gradient[2],c*2*z,2e-8,"Mixed z/time derivative");
            near(g[0],c*(f.n*tau+1)*2*x,8e-7,"Common-time spatial gradient");
            near(h[0][0],2*c*(f.n*tau+1),2e-6,"Common-time Hessian preserved");
        }
        // Exact knots select the past interval; the earliest knot uses the future.
        f.fill([](int,double tau,double,double,double){return static_cast<float>(tau*tau);});
        const std::int64_t node[3]={16,16,16};
        for (int k=0;k<3;++k) {
            float value,slope; const double tau=f.metadata.times[k];
            check(node_at(v,0,node,tau,value,&slope)==Status::Ok,"Knot derivative available");
            const int lo=k ? k-1 : 0, hi=k ? k : 1;
            const double a=f.metadata.times[lo],b=f.metadata.times[hi];
            near(value,static_cast<float>(tau*tau),0,"Knot value exact");
            near(slope,(static_cast<float>(b*b)-static_cast<float>(a*a))/(32*(b-a)),0,"One-sided knot slope");
        }
        f.fill([](int field,double,double x,double y,double z){return static_cast<float>(1e-6*(field+1)*(x*x+y+z));});
        float value,g[3],h[3][3];TimeSample t;
        check(sample(v,0,0.9,16.4,16.3,16.1,value,g,h,&t)==Status::Ok,"Frozen spatially varying sample");
        near(t.partial_time,0,0,"Frozen partial time is exactly zero");
        for(float a:t.gradient) near(a,0,0,"Frozen mixed derivative is exactly zero");
        check(g[0]!=0,"Frozen convective derivative need not vanish");
        double state[STATE_SIZE],reference[STATE_SIZE]; check(initialize(v,1,1.2,0.3,state)==Status::Ok,"Frozen observer");
        copy_state(reference,state);auto zero=v;zero.time_derivatives=TimeDerivativeMode::Zero;
        for(int i=0;i<5;++i) {
            check(rk4_step(v,state,0.5)==Status::Ok,"Frozen step");
            check(rk4_step(zero,reference,0.5)==Status::Ok,"Frozen zero-mode step");
            for(int j=0;j<STATE_SIZE;++j) near(state[j],reference[j],0,"Frozen RHS and ray states unchanged");
        }
        near(state[STATE_ISW],0,0,"Frozen ISW exactly zero");
        // Here neighboring nodes belong to different legacy shell slots/times.
        Fixture shells(32,3,staggered);
        shells.fill([&](int field,double tau,double x,double y,double z) {
            return static_cast<float>((field+1)*alpha*(32*tau+1)*(x*x+2*y+z*z));
        });
        for(int field=0;field<4;++field) {
            check(sample(shells.view(),field,0.7,23.9,16.1,16.2,value,g,h,&t)==Status::Ok,"Derivative stencil crosses recording-shell boundary");
            near(t.partial_time,(field+1)*alpha*(23.9*23.9+2*16.1+16.2*16.2),4e-8,"Shell-dependent time pairs reconstruct common partial derivative");
            near(t.gradient[0],(field+1)*alpha*2*23.9,3e-8,"Mixed derivative across shell interface");
        }
    }
    Fixture f(16,1);f.fill([](int,double,double,double,double){return 1e-6f;});
    auto v=f.view(); float value,g[3],h[3][3];TimeSample t;
    check(sample(v,0,1,8,8,8,value,g,h,&t)==Status::TimeCoverage,"Thickness one cannot supply time differences");
    v.time_derivatives=TimeDerivativeMode::Zero;
    check(sample(v,0,1,8,8,8,value,g,h,&t)==Status::Ok,"Explicit zero mode preserves thickness one");
    near(t.partial_time,0,0,"Zero mode supplies literal zero");
    // An exact value alone is insufficient for derivatives, but works in zero mode.
    Fixture missing(16,2);set_times(missing,{0.5,1.25});
    missing.fill([](int,double,double,double,double){return 0.0f;});
    v=missing.view();int offsets[3]={0,1,1};RadialSegment segment{-1,2,0};
    v.slot_offsets=offsets;v.shells=&segment;
    double state[STATE_SIZE];check(initialize(v,0.5,1.1,0.2,state)==Status::Ok,"Exact-only observer value");
    state[STATE_ISW]=123;const std::vector<double> before(state,state+STATE_SIZE);
    check(rk4_step(v,state,0.1)==Status::TimeCoverage,"Missing derivative bracket rejects RK4");
    check(std::equal(before.begin(),before.end(),state),"Rejected step preserves ISW and all existing state");
    const std::int64_t node[3]={8,8,8};
    check(node_at(v,0,node,0.5,value)==Status::Ok,"Value-only exact sample remains usable");
}
void rhs_tests() {
    for(bool staggered:{false,true}) {
        Fixture f(32,3,staggered);set_times(f,{0.5,0.75,1.25});
        const double alpha=std::ldexp(1.0,-22), beta=std::ldexp(1.0,-30);
        const double matrix[3][3]={{1,2,-1},{-3,2,4},{2,-2,-3}}; // trace zero
        f.fill([&](int field,double tau,double x,double y,double z){
            if(!field) return static_cast<float>(alpha*f.n*tau);
            return static_cast<float>(beta*f.n*tau*(matrix[0][field-1]*(x-16)+matrix[1][field-1]*(y-16)+matrix[2][field-1]*(z-16)));
        });
        auto v=f.view(),zero=v;zero.time_derivatives=TimeDerivativeMode::Zero;
        double y[STATE_SIZE],a[STATE_SIZE],b[STATE_SIZE];
        check(initialize(v,1,1.2,0.3,y)==Status::Ok,"Evolving observer");
        y[STATE_DA]=2.3;y[STATE_DA_PRIME]=-0.9;y[STATE_SIGMA1]=0.2;y[STATE_SIGMA2]=-0.13;
        check(ode(v,y,a)==Status::Ok && ode(zero,y,b)==Status::Ok,"Both RHS modes available");
        double expected[STATE_SIZE]={};const double source=2*alpha;
        expected[STATE_LOG_KTILDE0]=source;expected[STATE_ISW]=source;
        expected[STATE_DA_PRIME]=-source*y[STATE_DA_PRIME];
        expected[STATE_SIGMA1]=-source*y[STATE_SIGMA1];expected[STATE_SIGMA2]=-source*y[STATE_SIGMA2];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            const double c=beta*(matrix[i][j]+matrix[j][i])/2;
            const double n_i=y[STATE_NX+i],n_j=y[STATE_NX+j];
            const double q1=y[STATE_E1X+i]*y[STATE_E1X+j]-y[STATE_E2X+i]*y[STATE_E2X+j];
            const double q2=y[STATE_E1X+i]*y[STATE_E2X+j]+y[STATE_E2X+i]*y[STATE_E1X+j];
            expected[STATE_DA_PRIME]+=0.5*n_i*n_j*c*y[STATE_DA];
            expected[STATE_SIGMA1]-=0.5*y[STATE_DA]*y[STATE_DA]*q1*c;
            expected[STATE_SIGMA2]-=0.5*y[STATE_DA]*y[STATE_DA]*q2*c;
        }
        for(int i=0;i<STATE_SIZE;++i) near(a[i]-b[i],expected[i],2e-13,"Independent tensor RHS increment "+std::to_string(i));
    }
}
void homogeneous_tests() {
    Fixture f(32,3);set_times(f,{0.5,0.75,1.25});const double alpha=std::ldexp(1.0,-20);
    f.fill([&](int field,double tau,double,double,double){return field ? 0.0f : static_cast<float>(alpha*f.n*tau);});
    auto v=f.view();double y[STATE_SIZE],z;
    check(initialize(v,1,1.2,0.3,y)==Status::Ok,"Homogeneous time-varying observer");
    const double initial_energy=y[STATE_LOG_KTILDE0];
    for(int i=0;i<16;++i)check(rk4_step(v,y,0.5)==Status::Ok,"Homogeneous RK4");
    near(y[STATE_ISW],16*alpha,2e-12,"ISW equals 2 alpha lookback");
    near(y[STATE_LOG_KTILDE0]-initial_energy,y[STATE_ISW],2e-12,"ISW and photon-energy contributions agree");
    near(y[STATE_DA],-std::expm1(-16*alpha)/(2*alpha),2e-10,"Scalar time damping of distance");
    check(observed_redshift(v,y,z)==Status::Ok,"Homogeneous redshift");
    double a;check(background_at(v,y[STATE_TAU],a)==Status::Ok,"Source background");
    near(std::log1p(z),std::log(0.8/a)+alpha*f.n*(y[STATE_TAU]-1),2e-12,"Analytic homogeneous lapse redshift");
    check(initialize(v,1,1.2,0.3,y)==Status::Ok,"Refinement observer");
    check(trace(v,0.01,y,z)==Status::Ok,"Target refinement with ISW");
    near(y[STATE_ISW],2*alpha*f.n*(1-y[STATE_TAU]),2e-12,"ISW uses accepted refined endpoint");
    // Piecewise constant slopes: check convergence, without asserting RK4 order.
    set_times(f,{0.5,0.8,1.25});
    f.fill([](int field,double tau,double,double,double){return field ? 0.0f : static_cast<float>(0.001*tau*tau);});
    v=f.view();const double depth=9.3;
    auto at=[&](double tau){float w,g[3],h[3][3];check(sample(v,0,tau,16,16,16,w,g,h)==Status::Ok,"Piecewise endpoint");return w;};
    const double exact=2*(at(1)-at(1-depth/32));
    double errors[2];int k=0;
    for(double step:{1.3,0.017}){
        check(initialize(v,1,1.2,0.3,y)==Status::Ok,"Piecewise observer");
        for(double distance=0;distance<depth;){double dt=std::min(step,depth-distance);check(rk4_step(v,y,dt)==Status::Ok,"Step across slope knot");distance+=dt;}
        errors[k++]=std::fabs(y[STATE_ISW]-exact);
    }
    std::cout << "ISW slope-knot integration errors: " << errors[0] << " -> " << errors[1] << '\n';
    check(errors[1]<errors[0]/10,"ISW converges across derivative discontinuities");
}
}
int main(){try{sampling_tests();rhs_tests();homogeneous_tests();std::cout<<"Time-derivative/ISW CPU checks passed ("<<checks<<").\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
