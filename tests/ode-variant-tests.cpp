#include "test_support.hpp"
#include "snapshot-reference.hpp"
using namespace lightcone;
int main(){try {
    Fixture f(32,3,false); // Pack exactly representable affine B into snapshot half storage.
    const double alpha=std::ldexp(1.,-18),beta=std::ldexp(1.,-24),offset=std::ldexp(1.,-10);
    f.fill([&](int field,double,double x,double,double z){
        return static_cast<float>(field==0 ? offset+alpha*z : field==2 ? beta*(x-16):0.);
    });
    std::vector<float> vectors;
    const std::size_t count=32*32*32;
    for(int field=1;field<4;++field) vectors.insert(vectors.end(),f.fields[field*3].begin(),f.fields[field*3].end());
    auto view=f.view(); double y[STATE_SIZE],rhs[STATE_SIZE],snap_rhs[STATE_SIZE]={};
    check(initialize(view,1.,1.2,.3,y)==Status::Ok,"Variant observer");
    const double w=offset+16*alpha,e=std::exp(2*w);
    near(y[STATE_DA_PRIME],RT_CORRECT_OBSERVER_DISTANCE ? -e:-1.,1e-15,"Observer correction switch");
    y[STATE_DA]=2.; y[STATE_DA_PRIME]=-.7; y[STATE_SIGMA1]=1e-4; y[STATE_SIGMA2]=-2e-4;
    check(ode(view,y,rhs)==Status::Ok,"Variant RHS");
    double snapshot[STATE_SIZE];copy_state(snapshot,y);snapshot[20]=0.;
    snapshot_reference_ode(f.fields[0].data(),vectors.data(),32,snapshot,snap_rhs,100.,70.,.3,.7);
    const double* n=y+STATE_NX;
    const double gradient=(RT_NONLINEAR_SCALAR ? e:1.)*alpha;
    const double tidal=RT_NONLINEAR_SCALAR ? 2*e*e*alpha*alpha:0.;
    const double transport=4*n[2]*gradient+n[0]*n[1]*beta;
    for(int screen=0;screen<2;++screen) {
        const double* v=y+STATE_E1X+3*screen;
        const double sym=.5*beta*(v[0]*n[1]+v[1]*n[0]);
        const double antisym[3]={-.5*beta*v[1],.5*beta*v[0],0.};
        for(int i=0;i<3;++i) {
            const double expected=-n[i]*(2*v[2]*gradient+sym)-(RT_CORRECT_SACHS_B ? antisym[i]:0.);
            near(rhs[STATE_E1X+3*screen+i],expected,2e-13,"Light-cone Sachs switch");
            near(snap_rhs[STATE_E1X+3*screen+i],expected,2e-13,"Snapshot Sachs switch");
        }
        double constraint=0.;for(int i=0;i<3;++i) constraint+=rhs[STATE_NX+i]*v[i]+n[i]*rhs[STATE_E1X+3*screen+i];
        double missing=0.;for(int i=0;i<3;++i) missing+=n[i]*antisym[i];
        near(constraint,RT_CORRECT_SACHS_B ? 0.:missing,2e-13,"Screen orthogonality diagnoses omitted term");
    }
    const double q[2]={y[STATE_E1Z]*y[STATE_E1Z]-y[STATE_E2Z]*y[STATE_E2Z],2*y[STATE_E1Z]*y[STATE_E2Z]};
    for(int m=0;m<2;++m) {
        const int index=STATE_SIGMA1+m;
        const double expected=(RT_CORRECT_SHEAR_TRANSPORT ? -transport:transport)*y[index]+q[m]*tidal*y[STATE_DA]*y[STATE_DA];
        near(rhs[index],expected,2e-13,"Light-cone shear switch");
        near(snap_rhs[index],expected,2e-13,"Snapshot shear switch");
    }
    const double expected_distance=-transport*y[STATE_DA_PRIME]-(n[2]*n[2]-1)*tidal*y[STATE_DA]+
        (y[STATE_SIGMA1]*y[STATE_SIGMA1]+y[STATE_SIGMA2]*y[STATE_SIGMA2])/std::pow(y[STATE_DA],3);
    near(rhs[STATE_DA_PRIME],expected_distance,2e-13,"Scalar transport and tidal correction");
    near(snap_rhs[STATE_DA_PRIME],expected_distance,2e-13,"Snapshot scalar transport and tide");
    // Scalar interpolation must be invariant under large periodic translations.
    std::vector<float> data(count);
    for(int z=0;z<32;++z)for(int j=0;j<32;++j)for(int i=0;i<32;++i) data[i+32*(j+32*z)]=std::sin(.2*i)+std::cos(.3*j)+.01*z;
    float base_gradient[3],base_hessian[3][3],grad[3],hessian[3][3];
    const float base=snapshot_reference_sample(data.data(),32,16.125,16.25,16.375,base_gradient,base_hessian);
    for(double shift : {-3200000.,-8000.,8000.,3200000.}) {
        const float value=snapshot_reference_sample(data.data(),32,16.125+shift,16.25,16.375,grad,hessian);
        near(value,base,0.,"Snapshot double subtraction preserves fractional coordinates");
        for(int i=0;i<3;++i) {near(grad[i],base_gradient[i],0.,"Translated gradient");for(int j=0;j<3;++j)near(hessian[i][j],base_hessian[i][j],0.,"Translated Hessian");}
    }
    std::cout<<"ODE variant "<<raytracer::PHYSICS_JSON<<" passed ("<<checks<<")\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
