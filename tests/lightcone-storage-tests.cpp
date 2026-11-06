#include "test_support.hpp"
#include "lightcone_maps.hpp"
#include <chealpix.h>
#include <array>
#include <limits>
#include <set>

using namespace lightcone;
namespace {
void face_and_ordering_tests() {
    check(valid_nside(1) && valid_nside(16) && !valid_nside(0) && !valid_nside(3), "NESTED Nside validation");
    for (int nside : {1, 2, 4, 8}) {
        const int npix = 12*nside*nside;
        std::vector<int> seen(npix);
        std::vector<float> data(npix);
        for (int i = 0; i < npix; ++i) data[i] = i+0.25f;
        check(ring_to_nested(nested_to_ring(data, nside), nside) == data, "RING/NESTED exact round trip");
        for (int face = 0; face < 12; ++face) {
            std::vector<double> theta, phi;
            nested_angles(nside, face, theta, phi);
            check(theta.size() == static_cast<std::size_t>(nside*nside), "Face pixel count");
            for (int i = 0; i < nside*nside; ++i) {
                std::int64_t pixel, ring;
                ang2pix_nest64(nside, theta[i], phi[i], &pixel);
                check(pixel == face*nside*nside+i, "Launch direction belongs to exact base face");
                nest2ring64(nside, pixel, &ring);
                check(++seen[ring] == 1, "Disjoint face pixels");
            }
        }
        for (int count : seen) check(count == 1, "Twelve faces cover sphere");
        TempDirectory temp;
        for (int face : {-1, 0, 7, 11}) for (auto ordering : {Ordering::Nested, Ordering::Ring}) {
            std::vector<float> expected(npix, 0.0f);
            const int first = face < 0 ? 0 : face*nside*nside;
            const int count = face < 0 ? npix : nside*nside;
            std::copy_n(data.data()+first, count, expected.data()+first);
            const auto path = temp.path/("map-"+std::to_string(face)+"-"+std::to_string(static_cast<int>(ordering))+".fits");
            write_nested_map(expected, nside, ordering, path.string());
            long actual_n; char coords[32] = {}, scheme[32] = {};
            float* raw = read_healpix_map(path.c_str(), &actual_n, coords, scheme);
            check(raw && actual_n == nside, "FITS Nside");
            std::vector<float> actual(raw, raw+npix); std::free(raw);
            check(std::string(scheme) == (ordering == Ordering::Nested ? "NESTED" : "RING"), "FITS ordering header");
            if (ordering == Ordering::Ring) actual = ring_to_nested(actual, nside);
            check(actual == expected, "FITS ordering retains values and face zero mask");
        }
    }
    rejects([] { std::vector<double> a,b; nested_angles(3, 0, a,b); }, "Reject nonhierarchical Nside");
}
std::vector<float> pack(const std::vector<float>& full, int n, const SlabLayout& layout) {
    std::vector<float> output(layout.voxels);
    for (const auto& slab : layout.slabs)
        for (int z = 0; z < slab.extent[2]; ++z) for (int y = 0; y < slab.extent[1]; ++y) for (int x = 0; x < slab.extent[0]; ++x) {
            const auto source = wrap(slab.origin[0]+x,n)+n*(wrap(slab.origin[1]+y,n)+n*wrap(slab.origin[2]+z,n));
            output[slab.offset+x+slab.strides[1]*y+slab.strides[2]*z] = full[source];
        }
    return output;
}
void addressing_tests() {
    const int n = 32;
    Geometry g; g.vertex[0] = 0.98; g.vertex[1] = -0.04; g.vertex[2] = 0.37;
    std::vector<float> full(n*n*n);
    for (std::size_t i = 0; i < full.size(); ++i) full[i] = static_cast<float>(i);
    const auto original = float_grid(full.data(),n,0,false);
    for (int face = 0; face < 12; ++face) {
        std::vector<double> theta, phi; nested_angles(4,face,theta,phi);
        const auto layout = build_ziggurat(n,g,face,theta,phi,18.5,7,8);
        check(layout.axis == (face < 4 || face >= 8 ? 2 : face%2 == 0 ? 0 : 1), "Face slab axis");
        auto data = pack(full,n,layout);
        auto grid = original; grid.values = data.data();
        for (const auto& slab : layout.slabs) {
            check(slab.core_begin == slab.origin[layout.axis] && slab.core_end-slab.core_begin == slab.extent[layout.axis], "Core ownership without duplicate axial halo");
            check(slab.strides[0] == 1 && slab.strides[1] == static_cast<std::size_t>(slab.extent[0]), "XYZ strides stay X contiguous");
        }
        for (std::size_t i = 0; i < theta.size(); ++i) for (double s = 0; s <= 18.5; s += 2.5) {
            const double u[3] = {std::sin(theta[i])*std::cos(phi[i]),std::sin(theta[i])*std::sin(phi[i]),std::cos(theta[i])};
            // Exercise opposite eight-cell physical displacements and B staggering.
            for (int side : {-1,1}) for (int field = 0; field < 4; ++field) {
                std::int64_t cell[3];
                for (int a = 0; a < 3; ++a) cell[a] = static_cast<std::int64_t>(std::floor(g.vertex[a]*n+s*u[a]+side*8-(field == a+1 ? 0.5 : 0)));
                for (int z = -1; z <= 2; ++z) for (int y = -1; y <= 2; ++y) for (int x = -1; x <= 2; ++x) {
                    const std::int64_t xyz[3] = {cell[0]+x,cell[1]+y,cell[2]+z};
                    float expected,actual;
                    check(load_node(original,n,xyz,expected) == Status::Ok, "Periodic full node");
                    check(load_node(grid,n,xyz,actual,layout.view()) == Status::Ok, "Padded slab contains entire staggered stencil");
                    near(actual,expected,0,"Packed slab value equals periodic full volume");
                }
            }
        }
        // Every core boundary has exactly one owning slab, including its first layer.
        for (const auto& slab : layout.slabs) {
            std::int64_t xyz[3];
            for (int a = 0; a < 3; ++a) xyz[a] = slab.origin[a]+slab.extent[a]/2;
            xyz[layout.axis] = slab.core_begin;
            float expected,actual;
            check(load_node(grid,n,xyz,actual,layout.view()) == Status::Ok, "Half-open slab interface");
            check(load_node(original,n,xyz,expected) == Status::Ok, "Interface full reference");
            near(actual,expected,0,"Interface value");
        }
        std::int64_t outside[3] = {0,0,0}; outside[layout.axis] = layout.end+n;
        float value;
        check(load_node(grid,n,outside,value,layout.view()) == Status::NotResident, "No periodic alias for nonresident world coordinates");
        const auto unpadded = build_ziggurat(n,g,face,theta,phi,18.5,7,0);
        // A physical excursion past the four-cell halo is a configuration miss,
        // even when the full periodic cube can provide the same node.
        std::int64_t excursion[3];
        for (int a = 0; a < 3; ++a) excursion[a] = static_cast<std::int64_t>(std::floor(g.vertex[a]*n));
        excursion[unpadded.axis] = unpadded.end+1;
        check(load_node(original,n,excursion,value) == Status::Ok,"Full cube permits excessive excursion");
        check(load_node(grid,n,excursion,value,unpadded.view()) == Status::NotResident,"Insufficient padding reports residency failure");
        // Test scaled-half addressing with the same offsets/strides.
        std::vector<std::uint16_t> half(layout.voxels);
        for (std::size_t i = 0; i < half.size(); ++i) half[i] = static_cast<std::uint16_t>(0x3c00+(i%512));
        grid.values = nullptr; grid.half_values = half.data();
        for (const auto& slab : layout.slabs) {
            const std::int64_t xyz[3] = {slab.origin[0],slab.origin[1],slab.origin[2]};
            check(load_node(grid,n,xyz,value,layout.view()) == Status::Ok, "Scaled-half slab node");
            near(value,decode_half(half[slab.offset])/134217728.0f,0,"Scaled-half offset is correct");
        }
    }
}
void interpolation_tests() {
    TempDirectory temp;
    for (bool staggered : {false,true}) {
        Fixture f(32,3,staggered);
        f.fill([](int field,double tau,double x,double y,double z) { return static_cast<float>((field+1)*1e-6*(tau+1)*(x*x+2*y*y+z*z)); });
        const auto base = (temp.path/(staggered ? "staggered" : "collocated")).string(); f.write(base);
        Reader reader(base,3);
        std::vector<double> theta,phi; nested_angles(2,0,theta,phi);
        auto layout = build_ziggurat(f.n,f.geometry,0,theta,phi,5,3,2);
        std::vector<std::vector<float>> packed(12,std::vector<float>(layout.voxels));
        auto grids = f.grids;
        std::uint64_t expected_bytes = 0;
        for (int field = 0; field < 4; ++field) for (int slot = 0; slot < 3; ++slot) {
            for (const auto& slab : layout.slabs) {
                const auto values = reader.read_region(field,slot,slab.origin,slab.extent);
                std::copy(values.begin(),values.end(),packed[field*3+slot].begin()+slab.offset);
                expected_bytes += values.size()*(field == 2 ? 8 : 4); // fixture B2 is float64
            }
            grids[field*3+slot].values = packed[field*3+slot].data();
        }
        check(reader.bytes_read() == expected_bytes,"Source payload accounting includes datatype width and repeated slabs");
        auto full = f.view(), sliced = full; sliced.grids = grids.data(); sliced.layout = layout.view();
        for (int field = 0; field < 4; ++field) for (double x : {15.9,16.0,17.3,18.0}) {
            float a,b,ga[3],gb[3],ha[3][3],hb[3][3];
            TimeSample ta,tb;
            check(sample(full,field,0.95,x,16.2,16.8,a,ga,ha,&ta) == Status::Ok,"Full fixed-time stencil");
            check(sample(sliced,field,0.95,x,16.2,16.8,b,gb,hb,&tb) == Status::Ok,"Slab fixed-time stencil");
            near(ta.partial_time,tb.partial_time,0,"Slab partial-time derivative is bit identical");
            near(a,b,0,"Fixed-time value is bit identical");
            for (int i = 0; i < 3; ++i) {
                near(ta.gradient[i],tb.gradient[i],0,"Slab mixed derivative is bit identical");
                near(ga[i],gb[i],0,"Fixed-time gradient is bit identical");
                for (int j = 0; j < 3; ++j) near(ha[i][j],hb[i][j],0,"Fixed-time Hessian is bit identical");
            }
        }
        double full_state[STATE_SIZE], slab_state[STATE_SIZE];
        check(initialize(full,1.0,theta[0],phi[0],full_state) == Status::Ok,"Full observer for storage ray comparison");
        check(initialize(sliced,1.0,theta[0],phi[0],slab_state) == Status::Ok,"Slab observer for storage ray comparison");
        for (int step = 0; step < 3; ++step) {
            check(rk4_step(full,full_state,1.0) == Status::Ok,"Full ray step");
            check(rk4_step(sliced,slab_state,1.0) == Status::Ok,"Slab ray step");
            for (int i = 0; i < STATE_SIZE; ++i) near(full_state[i],slab_state[i],0,"Storage paths have identical ray states");
        }
        // Deliberately move the resident domain away: verify diagnostic and transactional state.
        MissDiagnostic diagnostic;
        sliced.miss = &diagnostic;
        auto bad_layout = layout.view(); bad_layout.begin += 100; bad_layout.end += 100; sliced.layout = bad_layout;
        double y[STATE_SIZE];
        check(initialize(full,1.0,theta[0],phi[0],y) == Status::Ok,"Miss test observer");
        const std::vector<double> before(y,y+STATE_SIZE);
        check(rk4_step(sliced,y,0.25) == Status::NotResident,"Missing slab rejects RK stage");
        check(std::equal(before.begin(),before.end(),y),"Residency failure leaves ray intact");
        check(diagnostic.valid && diagnostic.field >= 0 && diagnostic.slot >= 0 && diagnostic.tau == 1.0,"Representative miss has field, slot, node and time");
        sliced = full; sliced.trace_tau_min = 0.97;
        double z;
        // Force a high target so both storage paths reach the exact shared cap.
        for (auto& field : f.fields) std::fill(field.begin(),field.end(),0.0f);
        for (auto& field : packed) std::fill(field.begin(),field.end(),0.0f);
        check(initialize(sliced,1.0,theta[0],phi[0],y) == Status::Ok,"Capped observer");
        check(trace(sliced,10,y,z) == Status::TargetNotReached,"Look-back cap bounds tracing");
        near(y[STATE_TAU],0.97,0,"Fractional last step lands on cap");
        auto slab_cap = sliced; slab_cap.grids = grids.data(); slab_cap.layout = layout.view();
        check(initialize(slab_cap,1.0,theta[0],phi[0],slab_state) == Status::Ok,"Capped slab observer");
        check(trace(slab_cap,10,slab_state,z) == Status::TargetNotReached,"Slab look-back cap");
        for (int i = 0; i < STATE_SIZE; ++i) near(y[i],slab_state[i],0,"Full and slab cap identical");
    }
}
void size_tests() {
    const auto estimate = estimate_storage(100,20,3,10,64);
    check(estimate.field_bytes == 3000 && estimate.field_staging_bytes == 120,"Storage byte estimate");
    check(estimate.device_bytes == 3000+10*(2*sizeof(double)+MAP_COUNT*sizeof(float)+sizeof(int))+64,"Device estimate includes maps and status");
    require_storage_memory(estimate,estimate.device_bytes);
    rejects([&] { require_storage_memory(estimate,estimate.device_bytes-1); },"Reject insufficient available memory before payload reads");
    rejects([] { checked_add(std::numeric_limits<std::size_t>::max(),1); },"Reject offset overflow");
    rejects([] { checked_multiply(std::numeric_limits<std::size_t>::max(),2); },"Reject payload overflow");
    rejects([] { estimate_storage(std::numeric_limits<std::size_t>::max(),1,3,1,0); },"Reject overflowing memory estimate");
    rejects([] { int extent[3]={1,0,2}; checked_volume(extent); },"Reject invalid extent");
}
}
int main() {
    try {
        face_and_ordering_tests(); addressing_tests(); interpolation_tests(); size_tests();
        std::cout << "NESTED/slab CPU checks passed (" << checks << ").\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
