#pragma once
#include "../lightcone_ray.hpp"
#include <algorithm>
#include <array>

// Independent analytic fields, linear in conformal time, with transverse B modes.
// Lengths are grid cells; tau is in box units. No optical-equation helpers used.
struct AnalyticFields {
    int n=64;
    double scalar=0.002, vector=0.0, growth=0.0;
    struct Jet { double value=0, gradient[3]={}, hessian[3][3]={}, time=0, mixed[3]={}; };
    Jet jet(int field,double tau,const double x[3]) const {
        Jet out;
        if(!field) out.value=0.35*scalar;
        const double modes[4][3]={{1,2,1},{-2,1,3},{2,-3,1},{3,1,-2}};
        for(int m=0;m<4;++m) {
            double k[3],phase=0;
            for(int i=0;i<3;++i) { k[i]=modes[m][i]*6.283185307179586/n; phase+=k[i]*(x[i]-.5*n); }
            double amplitude=scalar/4;
            if(field) {
                const double polar[3]={-modes[m][1],modes[m][0],0};
                amplitude=vector*polar[field-1]/(4*std::hypot(polar[0],polar[1]));
            }
            const double wave=amplitude*std::sin(phase), scale=1+growth*(tau-1);
            out.value+=scale*wave; out.time+=growth*wave/n;
            for(int i=0;i<3;++i) {
                out.gradient[i]+=scale*amplitude*k[i]*std::cos(phase);
                out.mixed[i]+=growth*amplitude*k[i]*std::cos(phase)/n;
                for(int j=0;j<3;++j) out.hessian[i][j]-=scale*k[i]*k[j]*wave;
            }
        }
        return out;
    }
    lightcone::Status operator()(int field,double tau,double x,double y,double z,float& value,
                                float grad[3],float hessian[3][3],lightcone::TimeSample* time=nullptr) const {
        const double q[3]={x,y,z}; const auto data=jet(field,tau,q);
        value=static_cast<float>(data.value);
        for(int i=0;i<3;++i) {
            grad[i]=static_cast<float>(data.gradient[i]);
            for(int j=0;j<3;++j) hessian[i][j]=static_cast<float>(data.hessian[i][j]);
        }
        if(time) {
            time->partial_time=static_cast<float>(data.time);
            for(int i=0;i<3;++i) time->gradient[i]=static_cast<float>(data.mixed[i]);
        }
        return lightcone::Status::Ok;
    }
};

using BundleRay=std::array<double,7>; // position, forward direction, independent ISW quadrature
inline BundleRay geodesic_rhs(const AnalyticFields& fields,const BundleRay& y,double s) {
    BundleRay rhs{}; double n[3],bn[3]={};
    const double norm=std::hypot(y[3],y[4],y[5]);
    for(int i=0;i<3;++i) n[i]=y[3+i]/norm;
    const auto w=fields.jet(0,1-s/fields.n,y.data()); const double e=std::exp(2*w.value);
    for(int j=0;j<3;++j) {
        const auto b=fields.jet(j+1,1-s/fields.n,y.data());
        rhs[j]=-e*n[j]-b.value;
        for(int i=0;i<3;++i) bn[i]+=n[j]*b.gradient[i];
    }
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        rhs[3+i]+=((i==j ? 1.:0.)-n[i]*n[j])*(2*e*w.gradient[j]+bn[j]);
    rhs[6]=2*w.time;
    return rhs;
}
inline void geodesic_step(const AnalyticFields& fields,BundleRay& y,double s,double dt) {
    const auto a=geodesic_rhs(fields,y,s); auto tmp=y;
    for(int i=0;i<7;++i) tmp[i]=y[i]+.5*dt*a[i];
    const auto b=geodesic_rhs(fields,tmp,s+.5*dt);
    for(int i=0;i<7;++i) tmp[i]=y[i]+.5*dt*b[i];
    const auto c=geodesic_rhs(fields,tmp,s+.5*dt);
    for(int i=0;i<7;++i) tmp[i]=y[i]+dt*c[i];
    const auto d=geodesic_rhs(fields,tmp,s+dt);
    for(int i=0;i<7;++i) y[i]+=dt*(a[i]+2*b[i]+2*c[i]+d[i])/6;
}
struct BundleResult { double distance,epsilon1,epsilon2,rotation,isw; };
inline BundleResult bundle_result(const AnalyticFields& fields,const double* observer,const double* endpoint,
                                   double depth,double step,double angle) {
    BundleRay center{};
    double jacobi[2][2]={};
    for(int launch=-1;launch<4;++launch) {
        BundleRay ray{};
        for(int i=0;i<3;++i) {
            ray[i]=observer[lightcone::STATE_X+i]; ray[3+i]=observer[lightcone::STATE_NX+i];
            if(launch>=0) ray[3+i]-=(launch%2 ? -angle:angle)*observer[lightcone::STATE_E1X+3*(launch/2)+i];
        }
        const double norm=std::hypot(ray[3],ray[4],ray[5]); for(int i=3;i<6;++i) ray[i]/=norm;
        for(double s=0;s<depth;) {const double dt=std::min(step,depth-s);geodesic_step(fields,ray,s,dt);s+=dt;}
        if(launch<0) center=ray;
        else for(int a=0;a<2;++a) for(int i=0;i<3;++i)
            jacobi[a][launch/2]+=(launch%2 ? -1.:1.)*endpoint[lightcone::STATE_E1X+3*a+i]*ray[i]/(2*angle);
    }
    const double det=jacobi[0][0]*jacobi[1][1]-jacobi[0][1]*jacobi[1][0];
    const double c11=jacobi[0][0]*jacobi[0][0]+jacobi[0][1]*jacobi[0][1];
    const double c22=jacobi[1][0]*jacobi[1][0]+jacobi[1][1]*jacobi[1][1];
    const double c12=jacobi[0][0]*jacobi[1][0]+jacobi[0][1]*jacobi[1][1];
    return {std::sqrt(det),(c11-c22)/det,2*c12/det,
            std::atan2(jacobi[0][1]-jacobi[1][0],jacobi[0][0]+jacobi[1][1]),center[6]};
}
