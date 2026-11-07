// Run on an allocated GPU node: make test-gpu. Both conventions must produce
// identical host/device radii, cone decisions and exact-boundary shell ownership.
#include "lightcone_sampling.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

struct Input {
    std::int64_t xyz[3];
    double offset[3], boundary;
    int n;
    lightcone::Geometry geometry;
};
struct Result { double radius; int cone, record; };
LC_HD Result evaluate_one(const Input& input) {
    lightcone::View view; view.n=input.n; view.geometry=input.geometry;
    lightcone::GridView grid;
    for(int a=0;a<3;++a) grid.node_offset[a]=input.offset[a];
    bool cone;
    const double r=lightcone::node_radius(view,grid,input.xyz,cone);
    const lightcone::RadialSegment shells[2]={{0.,input.boundary,0},{input.boundary,input.boundary+10.,1}};
    const int offsets[2]={0,2};
    view.shells=shells; view.slot_offsets=offsets;
    return {r,static_cast<int>(cone),lightcone::record_at(view,0,r)};
}
__global__ void evaluate(const Input* input, Result* results, int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) results[i]=evaluate_one(input[i]);
}
void checked(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
int main() {
    try {
        const int count=1<<19;
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int> grids(8,4096), shift(-2,2), bit(0,1);
        std::uniform_real_distribution<double> unit(0.,1.), cosine(-1.,1.);
        std::vector<Input> inputs(count);
        for(int i=0;i<count;++i) {
            auto& s=inputs[i]; s.n=grids(rng);
            s.geometry.convention=i%2 ? lightcone::GeometryConvention::Legacy : lightcone::GeometryConvention::GasgrdV1;
            for(int a=0;a<3;++a) {
                s.xyz[a]=static_cast<std::int64_t>(unit(rng)*s.n)+static_cast<std::int64_t>(shift(rng))*s.n;
                s.offset[a]=.5*bit(rng);
                s.geometry.vertex[a]=bit(rng) ? unit(rng) : .5*bit(rng);
                s.geometry.direction[a]=cosine(rng);
            }
            const double norm=std::hypot(s.geometry.direction[0],s.geometry.direction[1],s.geometry.direction[2]);
            for(double& x:s.geometry.direction) x/=norm;
            s.geometry.angular_limit=bit(rng); s.geometry.opening_cos=cosine(rng);
            if(i<32) {
                s.n=576; s.xyz[0]=-1; s.xyz[1]=s.xyz[2]=288;
                s.geometry.vertex[0]=0.; s.geometry.vertex[1]=s.geometry.vertex[2]=.5;
                for(double& x:s.offset) x=0.;
                const int field=(i/2)%4;
                if(field) s.offset[field-1]=.5;
            }
            const auto reference=evaluate_one(s);
            s.boundary=i%3==0 ? std::nextafter(reference.radius,0.) :
                       i%3==1 ? reference.radius : std::nextafter(reference.radius,INFINITY);
        }
        Input* device_input=nullptr; Result* device_result=nullptr;
        checked(cudaMalloc(&device_input,count*sizeof(Input)));
        checked(cudaMalloc(&device_result,count*sizeof(Result)));
        checked(cudaMemcpy(device_input,inputs.data(),count*sizeof(Input),cudaMemcpyHostToDevice));
        evaluate<<<(count+255)/256,256>>>(device_input,device_result,count);
        checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
        std::vector<Result> output(count);
        checked(cudaMemcpy(output.data(),device_result,count*sizeof(Result),cudaMemcpyDeviceToHost));
        checked(cudaFree(device_input)); checked(cudaFree(device_result));
        std::size_t differences=0;
        for(int i=0;i<count;++i) {
            const auto host=evaluate_one(inputs[i]); const auto& gpu=output[i];
            if(std::memcmp(&host.radius,&gpu.radius,sizeof(double)) || host.cone!=gpu.cone || host.record!=gpu.record) ++differences;
        }
        std::cout<<"Host/device geometry: "<<count<<" nodes, both conventions, "<<differences<<" differences\n";
        return differences ? 1 : 0;
    } catch(const std::exception& error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}
