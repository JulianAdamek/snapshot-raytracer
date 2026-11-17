#include "lightcone_parts.hpp"
#include "lightcone.hpp"
#include <hdf5.h>
#include <fitsio.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace fs=std::filesystem;
namespace lightcone {
namespace {
void need(bool b,const std::string& m) { if(!b) throw std::runtime_error(m); }
struct H {
    hid_t id=-1; herr_t(*close)(hid_t)=nullptr;
    H(hid_t v,herr_t(*c)(hid_t)):id(v),close(c) { need(v>=0,"HDF5 compact-file operation failed"); }
    ~H() { if(id>=0) close(id); }
    void finish() { auto v=id; id=-1; need(close(v)>=0,"Cannot close compact-file handle"); }
    H(const H&)=delete;
};
void attr(hid_t f,const char* key,std::int64_t value) {
    H s(H5Screate(H5S_SCALAR),H5Sclose), a(H5Acreate2(f,key,H5T_STD_I64LE,s.id,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    need(H5Awrite(a.id,H5T_NATIVE_INT64,&value)>=0,"Write compact integer attribute");
}
void attr(hid_t f,const char* key,double value) {
    H s(H5Screate(H5S_SCALAR),H5Sclose), a(H5Acreate2(f,key,H5T_IEEE_F64LE,s.id,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    need(H5Awrite(a.id,H5T_NATIVE_DOUBLE,&value)>=0,"Write compact real attribute");
}
void attr(hid_t f,const char* key,const std::string& value) {
    H t(H5Tcopy(H5T_C_S1),H5Tclose); need(H5Tset_size(t.id,value.size()+1)>=0,"String attribute size");
    H s(H5Screate(H5S_SCALAR),H5Sclose), a(H5Acreate2(f,key,t.id,s.id,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    need(H5Awrite(a.id,t.id,value.c_str())>=0,"Write compact string attribute");
}
template<class T> T number(hid_t f,const char* key,hid_t type) {
    H a(H5Aopen(f,key,H5P_DEFAULT),H5Aclose), s(H5Aget_space(a.id),H5Sclose);
    need(H5Sget_simple_extent_npoints(s.id)==1,"Expected scalar compact attribute");
    T value; need(H5Aread(a.id,type,&value)>=0,"Read compact attribute"); return value;
}
std::string string(hid_t f,const char* key) {
    H a(H5Aopen(f,key,H5P_DEFAULT),H5Aclose), t(H5Aget_type(a.id),H5Tclose), s(H5Aget_space(a.id),H5Sclose);
    need(H5Tget_class(t.id)==H5T_STRING && !H5Tis_variable_str(t.id) && H5Tget_size(t.id)<=65536 && H5Sget_simple_extent_npoints(s.id)==1,"Invalid compact string attribute");
    std::vector<char> v(H5Tget_size(t.id)+1,0); need(H5Aread(a.id,t.id,v.data())>=0,"Read compact string"); return v.data();
}
void info_valid(const PartInfo& i) {
    auto r=pixel_range(i.nside,i.face,i.level,i.part);
    need(i.face>=0 && r.first==i.first && r.count==i.count,"Compact interval disagrees with subdivision");
    need(i.run_id.size()==64 && i.run_id.find_first_not_of("0123456789abcdef")==std::string::npos,"Compact run ID must be a lowercase SHA256");
    need(!i.physics.empty() && std::isfinite(i.target_z) && i.target_z>0 && std::isfinite(i.redshift_tolerance) && i.redshift_tolerance>0,"Invalid compact scientific metadata");
    bool valid_physics=false;
    for(int bits=0;bits<16;++bits) {
        const std::string expected="{\"correct_shear_transport\":"+std::to_string(bits&1)+
            ",\"correct_sachs_b\":"+std::to_string((bits>>1)&1)+",\"nonlinear_scalar\":"+std::to_string((bits>>2)&1)+
            ",\"correct_observer_distance\":"+std::to_string((bits>>3)&1)+"}";
        valid_physics=valid_physics || i.physics==expected;
    }
    need(valid_physics,"Invalid/noncanonical compact physics flags");
}
void check_shape(hid_t f,const char* name,std::int64_t count,bool status) {
    H d(H5Dopen2(f,name,H5P_DEFAULT),H5Dclose), s(H5Dget_space(d.id),H5Sclose), t(H5Dget_type(d.id),H5Tclose);
    hsize_t n=0; need(H5Sget_simple_extent_ndims(s.id)==1,"Compact dataset must have rank one"); H5Sget_simple_extent_dims(s.id,&n,nullptr);
    need(n==static_cast<hsize_t>(count) && H5Tget_size(t.id)==4 && H5Tget_class(t.id)==(status?H5T_INTEGER:H5T_FLOAT) && (!status || H5Tget_sign(t.id)==H5T_SGN_2),"Wrong compact dataset shape/type");
}
void transfer(hid_t d,hid_t type,std::int64_t offset,std::size_t count,void* p,bool write) {
    hsize_t start=offset,n=count; H file(H5Dget_space(d),H5Sclose), memory(H5Screate_simple(1,&n,nullptr),H5Sclose);
    need(H5Sselect_hyperslab(file.id,H5S_SELECT_SET,&start,nullptr,&n,nullptr)>=0,"Select compact block");
    need((write?H5Dwrite(d,type,memory.id,file.id,H5P_DEFAULT,p):H5Dread(d,type,memory.id,file.id,H5P_DEFAULT,p))>=0,"Transfer compact block");
}
struct Fits {
    fitsfile* file=nullptr; int status=0, hdutype=0;
    ~Fits() { if(file) { int s=0; fits_close_file(file,&s); } }
    void checked() { if(status) { char s[FLEN_STATUS]; fits_get_errstatus(status,s); throw std::runtime_error(std::string("FITS: ")+s); } }
    void finish() { fits_close_file(file,&status); file=nullptr; checked(); }
};
void same(const PartInfo& a,const PartInfo& b) {
    need(a.nside==b.nside && a.run_id==b.run_id && a.physics==b.physics && a.target_z==b.target_z && a.redshift_tolerance==b.redshift_tolerance,"Incompatible compact scientific configurations");
}
}
struct PartWriter::Impl {
    std::string path,temp; PartInfo info; H file;
    std::array<std::unique_ptr<H>,MAP_COUNT+1> datasets;
    std::array<std::int64_t,MAP_COUNT+1> next{};
    Impl(const std::string& p,const PartInfo& i):path(p),temp(p+".partial"),info(i),file((info_valid(i),need(!fs::exists(p),"Compact output exists"),H5Fcreate(temp.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT)),H5Fclose) {
        attr(file.id,"FORMAT_VERSION",std::int64_t(1)); attr(file.id,"NSIDE",i.nside); attr(file.id,"FIRSTPIX",i.first); attr(file.id,"NPIX",i.count);
        attr(file.id,"FACE",std::int64_t(i.face)); attr(file.id,"LEVEL",std::int64_t(i.level)); attr(file.id,"PART",i.part);
        attr(file.id,"ORDERING",std::string("NESTED")); attr(file.id,"RUN_ID",i.run_id); attr(file.id,"PHYSICS",i.physics);
        attr(file.id,"TARGET_Z",i.target_z); attr(file.id,"REDSHIFT_TOL",i.redshift_tolerance);
        hsize_t n=i.count; H space(H5Screate_simple(1,&n,nullptr),H5Sclose);
        for(int m=0;m<=MAP_COUNT;++m) datasets[m].reset(new H(H5Dcreate2(file.id,m==MAP_COUNT?"status":MAP_NAMES[m],m==MAP_COUNT?H5T_STD_I32LE:H5T_IEEE_F32LE,space.id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),H5Dclose));
    }
    void put(int m,std::int64_t offset,std::size_t count,const void* values) {
        need(m>=0 && m<=MAP_COUNT && offset==next[m] && count>0 && count<=static_cast<std::uint64_t>(info.count-offset),"Out-of-order compact write");
        transfer(datasets[m]->id,m==MAP_COUNT?H5T_NATIVE_INT:H5T_NATIVE_FLOAT,offset,count,const_cast<void*>(values),true); next[m]+=count;
    }
};
PartWriter::PartWriter(const std::string& path,const PartInfo& info):impl_(new Impl(path,info)) {}
PartWriter::~PartWriter()=default;
void PartWriter::map(int m,std::int64_t offset,std::size_t n,const float* v) { need(m>=0 && m<MAP_COUNT,"Invalid map"); impl_->put(m,offset,n,v); }
void PartWriter::statuses(std::int64_t offset,std::size_t n,const int* v) { impl_->put(MAP_COUNT,offset,n,v); }
void PartWriter::finish() {
    for(int m=0;m<=MAP_COUNT;++m) { need(impl_->next[m]==impl_->info.count,"Incomplete compact output"); impl_->datasets[m]->finish(); }
    need(H5Fflush(impl_->file.id,H5F_SCOPE_GLOBAL)>=0,"Flush compact output"); impl_->file.finish();
    check_part(impl_->temp,false);
    need(!fs::exists(impl_->path),"Compact output appeared during write"); fs::rename(impl_->temp,impl_->path);
}
PartInfo inspect_part(const std::string& path) {
    H file(H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose); PartInfo i;
    auto integer=[&](const char* s){ return number<std::int64_t>(file.id,s,H5T_NATIVE_INT64); };
    need(integer("FORMAT_VERSION")==1 && string(file.id,"ORDERING")=="NESTED","Unsupported compact format/order");
    i.nside=integer("NSIDE"); i.first=integer("FIRSTPIX"); i.count=integer("NPIX"); i.part=integer("PART");
    const auto face=integer("FACE"),level=integer("LEVEL"); need(face>=0 && face<12 && level>=0 && level<=29,"Invalid compact face/level");
    i.face=static_cast<int>(face); i.level=static_cast<int>(level);
    i.target_z=number<double>(file.id,"TARGET_Z",H5T_NATIVE_DOUBLE); i.redshift_tolerance=number<double>(file.id,"REDSHIFT_TOL",H5T_NATIVE_DOUBLE);
    i.run_id=string(file.id,"RUN_ID"); i.physics=string(file.id,"PHYSICS"); info_valid(i);
    for(auto name:MAP_NAMES) check_shape(file.id,name,i.count,false);
    check_shape(file.id,"status",i.count,true);
    return i;
}
void check_part(const std::string& path,bool frozen) {
    const auto i=inspect_part(path); H f(H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose), st(H5Dopen2(f.id,"status",H5P_DEFAULT),H5Dclose);
    std::vector<int> statuses(OUTPUT_BLOCK); std::vector<float> values(OUTPUT_BLOCK);
    for(std::int64_t o=0;o<i.count;o+=OUTPUT_BLOCK) {
        const auto n=static_cast<std::size_t>(std::min<std::int64_t>(OUTPUT_BLOCK,i.count-o)); transfer(st.id,H5T_NATIVE_INT,o,n,statuses.data(),false);
        for(std::size_t k=0;k<n;++k) need(statuses[k]>=0 && statuses[k]<static_cast<int>(Status::Count) && (!frozen || statuses[k]==0),"Invalid/failed compact ray");
        for(int m=0;m<MAP_COUNT;++m) {
            H d(H5Dopen2(f.id,MAP_NAMES[m],H5P_DEFAULT),H5Dclose); transfer(d.id,H5T_NATIVE_FLOAT,o,n,values.data(),false);
            for(std::size_t k=0;k<n;++k) if(statuses[k]==0) {
                need(std::isfinite(values[k]) && values[k]!=-1.6375e30f,"Invalid compact observable");
                if(frozen && m==7) need(values[k]==0,"Nonzero frozen ISW");
                if(m==6) need(std::fabs(values[k]-i.target_z)<=i.redshift_tolerance+2*FLT_EPSILON*std::fabs(i.target_z),"Compact redshift outside tolerance");
            }
        }
    }
}
void compare_part(const std::string& path,const std::string& reference) {
    const auto i=inspect_part(path); H f(H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
    const bool hdf=fs::is_regular_file(reference);
    PartInfo r;
    std::unique_ptr<H> other;
    if(hdf) { r=inspect_part(reference); same(i,r); need(i.first>=r.first && i.first+i.count<=r.first+r.count,"Reference does not contain part"); other.reset(new H(H5Fopen(reference.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose)); }
    std::vector<float> a(OUTPUT_BLOCK),b(OUTPUT_BLOCK);
    if(hdf) {
        H sa(H5Dopen2(f.id,"status",H5P_DEFAULT),H5Dclose), sb(H5Dopen2(other->id,"status",H5P_DEFAULT),H5Dclose);
        std::vector<int> va(OUTPUT_BLOCK),vb(OUTPUT_BLOCK);
        for(std::int64_t o=0;o<i.count;o+=OUTPUT_BLOCK) {
            const auto n=static_cast<std::size_t>(std::min<std::int64_t>(OUTPUT_BLOCK,i.count-o));
            transfer(sa.id,H5T_NATIVE_INT,o,n,va.data(),false);transfer(sb.id,H5T_NATIVE_INT,i.first-r.first+o,n,vb.data(),false);
            need(std::equal(va.begin(),va.begin()+n,vb.begin()),"Compact/reference status mismatch");
        }
    }
    for(int m=0;m<MAP_COUNT;++m) {
        H d(H5Dopen2(f.id,MAP_NAMES[m],H5P_DEFAULT),H5Dclose);
        std::unique_ptr<H> rd; Fits fits; LONGLONG repeat=0;
        if(hdf) rd.reset(new H(H5Dopen2(other->id,MAP_NAMES[m],H5P_DEFAULT),H5Dclose));
        else {
            auto p=fs::path(reference)/(std::string(MAP_NAMES[m])+"_batch"+std::to_string(i.face)+".fits");
            const bool full=!fs::exists(p);
            if(full) p=fs::path(reference)/(std::string(MAP_NAMES[m])+"_batch0.fits");
            fits_open_file(&fits.file,p.c_str(),READONLY,&fits.status); fits.checked(); fits_movabs_hdu(fits.file,2,&fits.hdutype,&fits.status);
            LONGLONG nside=0,rows=0,width=0; int type=0; char order[FLEN_VALUE]={};
            fits_read_key(fits.file,TLONGLONG,"NSIDE",&nside,nullptr,&fits.status); fits_read_key(fits.file,TSTRING,"ORDERING",order,nullptr,&fits.status);
            fits_get_num_rowsll(fits.file,&rows,&fits.status); fits_get_coltypell(fits.file,1,&type,&repeat,&width,&fits.status); fits.checked();
            if(full) { char object[FLEN_VALUE]={}; fits_read_key(fits.file,TSTRING,"OBJECT",object,nullptr,&fits.status); fits.checked(); need(std::string(object)=="FULLSKY","Reference is not a declared full sky"); }
            need(nside==i.nside && std::string(order)=="NESTED" && repeat>0 && rows==12*i.nside*i.nside/repeat && (12*i.nside*i.nside)%repeat==0,"Invalid saved FITS reference");
        }
        for(std::int64_t o=0;o<i.count;o+=OUTPUT_BLOCK) {
            const auto n=static_cast<std::size_t>(std::min<std::int64_t>(OUTPUT_BLOCK,i.count-o)); transfer(d.id,H5T_NATIVE_FLOAT,o,n,a.data(),false);
            if(hdf) transfer(rd->id,H5T_NATIVE_FLOAT,i.first-r.first+o,n,b.data(),false);
            else { int any=0; float null=NAN; const auto first=i.first+o; fits_read_col(fits.file,TFLOAT,1,first/repeat+1,first%repeat+1,n,&null,b.data(),&any,&fits.status); fits.checked(); }
            for(std::size_t k=0;k<n;++k) {
                const bool va=std::isfinite(a[k]) && a[k]!=-1.6375e30f,vb=std::isfinite(b[k]) && b[k]!=-1.6375e30f;
                need(va==vb,"Compact/reference validity mismatch");
                if(va) need(std::fabs(double(a[k])-b[k])<=4*FLT_EPSILON*std::max(std::fabs(a[k]),std::fabs(b[k]))+4*std::numeric_limits<float>::denorm_min(),"Compact/reference float32 difference");
            }
        }
    }
}
void import_face(const std::string& reference,const std::string& path,const PartInfo& i) {
    info_valid(i); need(i.level==0,"Saved FITS imports must be complete base faces");
    PartWriter writer(path,i); std::vector<float> values(OUTPUT_BLOCK); std::vector<int> statuses(OUTPUT_BLOCK,0);
    for(int m=0;m<MAP_COUNT;++m) {
        Fits f; const auto p=fs::path(reference)/(std::string(MAP_NAMES[m])+"_batch"+std::to_string(i.face)+".fits");
        fits_open_file(&f.file,p.c_str(),READONLY,&f.status); f.checked(); fits_movabs_hdu(f.file,2,&f.hdutype,&f.status);
        LONGLONG side=0,rows=0,repeat=0,width=0; int type=0;char order[FLEN_VALUE]={};
        fits_read_key(f.file,TLONGLONG,"NSIDE",&side,nullptr,&f.status); fits_read_key(f.file,TSTRING,"ORDERING",order,nullptr,&f.status);
        fits_get_num_rowsll(f.file,&rows,&f.status);fits_get_coltypell(f.file,1,&type,&repeat,&width,&f.status); f.checked();
        need(side==i.nside && std::string(order)=="NESTED" && type==TFLOAT && repeat>0 && rows==12*i.nside*i.nside/repeat && (12*i.nside*i.nside)%repeat==0,"Invalid saved FITS import");
        for(std::int64_t o=0;o<i.count;o+=OUTPUT_BLOCK) {
            auto n=static_cast<std::size_t>(std::min<std::int64_t>(OUTPUT_BLOCK,i.count-o)); const auto first=i.first+o;
            int any=0;float null=NAN;fits_read_col(f.file,TFLOAT,1,first/repeat+1,first%repeat+1,n,&null,values.data(),&any,&f.status); f.checked();
            need(!any,"Nulls in saved FITS import"); writer.map(m,o,n,values.data());
        }
    }
    for(std::int64_t o=0;o<i.count;o+=OUTPUT_BLOCK) writer.statuses(o,std::min<std::int64_t>(OUTPUT_BLOCK,i.count-o),statuses.data());
    writer.finish(); check_part(path,true);
}
void assemble_parts(const std::vector<std::string>& paths,const std::string& directory) {
    need(!paths.empty(),"No compact parts to assemble");
    std::vector<std::pair<PartInfo,std::string>> parts;
    for(const auto& p:paths) { auto i=inspect_part(p); if(!parts.empty()) same(i,parts.front().first); check_part(p,true); parts.emplace_back(i,p); }
    std::sort(parts.begin(),parts.end(),[](const auto& a,const auto& b){return a.first.first<b.first.first;});
    const auto& model=parts.front().first; const std::int64_t npix=12*model.nside*model.nside;
    std::int64_t next=0; for(const auto& p:parts) { need(p.first.first==next,"Missing or overlapping compact intervals"); next+=p.first.count; } need(next==npix,"Incomplete full sky");
    fs::path output(directory),temp(directory+".partial"); need(!fs::exists(output) && !fs::exists(temp),"Assembly output/temporary directory already exists"); fs::create_directories(temp);
    std::vector<float> buffer(OUTPUT_BLOCK);
    for(int m=0;m<MAP_COUNT;++m) {
        Fits f; const auto path=temp/(std::string(MAP_NAMES[m])+"_batch0.fits");
        fits_create_file(&f.file,path.c_str(),&f.status); fits_create_img(f.file,BYTE_IMG,0,nullptr,&f.status);
        char signal[]="SIGNAL",form[]="1E",unit[]=""; char* names[]={signal}; char* forms[]={form}; char* units[]={unit};
        fits_create_tbl(f.file,BINARY_TBL,npix,1,names,forms,units,nullptr,&f.status);
        auto key=[&](const char* k,const char* v){ fits_update_key(f.file,TSTRING,k,const_cast<char*>(v),nullptr,&f.status); };
        key("PIXTYPE","HEALPIX"); key("ORDERING","NESTED"); key("INDXSCHM","IMPLICIT"); key("OBJECT","FULLSKY"); key("COORDSYS","C"); key("RUNID",model.run_id.c_str());
        LONGLONG side=model.nside,first=0,last=npix-1;
        fits_update_key(f.file,TLONGLONG,"NSIDE",&side,nullptr,&f.status); fits_update_key(f.file,TLONGLONG,"FIRSTPIX",&first,nullptr,&f.status); fits_update_key(f.file,TLONGLONG,"LASTPIX",&last,nullptr,&f.status); f.checked();
        for(const auto& part:parts) {
            H file(H5Fopen(part.second.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose),d(H5Dopen2(file.id,MAP_NAMES[m],H5P_DEFAULT),H5Dclose);
            for(std::int64_t o=0;o<part.first.count;o+=OUTPUT_BLOCK) {
                const auto n=static_cast<std::size_t>(std::min<std::int64_t>(OUTPUT_BLOCK,part.first.count-o)); transfer(d.id,H5T_NATIVE_FLOAT,o,n,buffer.data(),false);
                fits_write_col(f.file,TFLOAT,1,part.first.first+o+1,1,n,buffer.data(),&f.status); f.checked();
            }
        }
        fits_write_chksum(f.file,&f.status); f.checked(); f.finish();
        // Validate every published value against its source, using bounded reads.
        Fits verify; fits_open_file(&verify.file,path.c_str(),READONLY,&verify.status); fits_movabs_hdu(verify.file,2,&verify.hdutype,&verify.status); verify.checked();
        std::vector<float> actual(OUTPUT_BLOCK);
        for(const auto& part:parts) {
            H file(H5Fopen(part.second.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose),d(H5Dopen2(file.id,MAP_NAMES[m],H5P_DEFAULT),H5Dclose);
            for(std::int64_t o=0;o<part.first.count;o+=OUTPUT_BLOCK) {
                const auto n=static_cast<std::size_t>(std::min<std::int64_t>(OUTPUT_BLOCK,part.first.count-o)); transfer(d.id,H5T_NATIVE_FLOAT,o,n,buffer.data(),false);
                int any=0; float null=NAN; fits_read_col(verify.file,TFLOAT,1,part.first.first+o+1,1,n,&null,actual.data(),&any,&verify.status); verify.checked();
                need(!any && std::equal(buffer.begin(),buffer.begin()+n,actual.begin()),"Assembled FITS data mismatch");
            }
        }
    }
    fs::rename(temp,output);
}
} // namespace lightcone
