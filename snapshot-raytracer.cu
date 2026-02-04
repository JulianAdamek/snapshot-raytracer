#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <chealpix.h>
#include <cuda_fp16.h>
#include <cstdint>

// CUDA error checking helper
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)
#endif

// HDF5 Array Loader
std::vector<float> load_hdf5_array(const std::string& filename, const std::string& datasetname,
    std::vector<hsize_t>& dims) {
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // Open the HDF5 file
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Failed to open HDF5 file: " << filename << std::endl;
        return {};
    }

    // Open the dataset
    dataset_id = H5Dopen2(file_id, datasetname.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Failed to open dataset: " << datasetname << std::endl;
        H5Fclose(file_id);
        return {};
    }

    // Get the dataspace
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Failed to get dataspace" << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return {};
    }

    // Get the dimensions of the dataset
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    dims.resize(rank);
    status = H5Sget_simple_extent_dims(dataspace_id, dims.data(), NULL);
    if (status < 0) {
        std::cerr << "Failed to get dimensions" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return {};
    }

    // Calculate the total number of elements
    hsize_t total_elements = 1;
    for (hsize_t dim : dims) {
        total_elements *= dim;
    }

    // Read the data
    std::vector<float> data_float(total_elements);
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_float.data());
    if (status < 0) {
        std::cerr << "Failed to read data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return {};
    }

    // Close resources
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return data_float;
}

// Load an HDF5 dataset whose base datatype is an array[3] of float (vector field per voxel).
// The reported dataspace dimensions are the 3D grid dims (nx, ny, nz). Each grid cell stores 3 floats.
// Returned layout: data[((i * ny + j) * nz + k) * 3 + c] with c in {0,1,2}.
std::vector<float> load_hdf5_vector_field(const std::string& filename, const std::string& datasetname,
                                          std::vector<hsize_t>& dims) {
    hid_t file_id = -1, dataset_id = -1, dataspace_id = -1, dtype_id = -1, memtype_id = -1;
    std::vector<float> buffer;

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Failed to open HDF5 file: " << filename << std::endl;
        return buffer;
    }
    dataset_id = H5Dopen2(file_id, datasetname.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Failed to open dataset: " << datasetname << std::endl;
        H5Fclose(file_id);
        return buffer;
    }
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Failed to get dataspace" << std::endl;
        H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 3) {
        std::cerr << "Expected 3D dataspace, got rank=" << rank << std::endl;
        H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    dims.resize(rank);
    if (H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr) < 0) {
        std::cerr << "Failed to get dimensions" << std::endl;
        H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }

    // Inspect datatype (should be array[3] of IEEE_F32)
    dtype_id = H5Dget_type(dataset_id);
    if (dtype_id < 0) {
        std::cerr << "Failed to get datatype" << std::endl;
        H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    if (H5Tget_class(dtype_id) != H5T_ARRAY) {
        std::cerr << "Dataset is not an array type (expected array[3] of float)." << std::endl;
        H5Tclose(dtype_id); H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    int array_ndims = H5Tget_array_ndims(dtype_id);
    hsize_t array_dims[8];
    if (H5Tget_array_dims2(dtype_id, array_dims) < 0) {
        std::cerr << "Failed to get array dimensions." << std::endl;
        H5Tclose(dtype_id); H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    if (array_ndims != 1 || array_dims[0] != 3) {
        std::cerr << "Expected per-voxel array length 3, got length=" << array_dims[0] << std::endl;
        H5Tclose(dtype_id); H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }

    // Prepare memory array type (array[3] of native float)
    hsize_t mdim = 3;
    memtype_id = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &mdim);
    if (memtype_id < 0) {
        std::cerr << "Failed to create memory array type" << std::endl;
        H5Tclose(dtype_id); H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    size_t total_voxels = static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]);
    size_t total_floats = total_voxels * 3;
    size_t total_bytes = total_floats * sizeof(float);
    if (total_bytes > static_cast<size_t>(-1)) {
        std::cerr << "Size overflow computing allocation for vector field." << std::endl;
        H5Tclose(memtype_id); H5Tclose(dtype_id); H5Sclose(dataspace_id); H5Dclose(dataset_id); H5Fclose(file_id); return buffer;
    }
    if (total_bytes > 8ULL * 1024 * 1024 * 1024) {
        std::cerr << "Warning: allocating ~" << (static_cast<double>(total_bytes) / (1024*1024*1024)) << " GiB for vector field." << std::endl;
    }

    buffer.resize(static_cast<size_t>(total_floats));
    if (buffer.size() != static_cast<size_t>(total_floats)) {
        std::cerr << "Failed to allocate memory for vector field buffer." << std::endl;
        H5Tclose(memtype_id);
        H5Tclose(dtype_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        buffer.clear();
        return buffer;
    }
    // Single read (the previous version performed this twice, doubling I/O and memory pressure for large grids)
    if (H5Dread(dataset_id, memtype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0) {
        std::cerr << "Failed to read vector field dataset." << std::endl;
        buffer.clear();
    }

    H5Tclose(memtype_id);
    H5Tclose(dtype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return buffer;
}

// Helper accessor for flattened vector field layout.
inline float vector_field_at(const std::vector<float>& vec, const std::vector<hsize_t>& dims,
                             hsize_t i, hsize_t j, hsize_t k, int comp) {
    return vec[(((i * dims[1] + j) * dims[2] + k) * 3) + comp];
}

// 1D Catmull-Rom spline interpolation
__host__ __device__ inline float cubic_cr(const float p[4], float t) noexcept {
    return 0.5f * ((2 * p[1]) + (-p[0] + p[2]) * t +
        (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]) * t * t +
        (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) * t * t * t);
}

// Gradient of the 1D Catmull-Rom spline
__host__ __device__ inline float cubic_cr_grad(const float p[4], float t) noexcept {
    return 0.5f * ((-p[0] + p[2]) +
        (4 * p[0] - 10 * p[1] + 8 * p[2] - 2 * p[3]) * t +
        (-3 * p[0] + 9 * p[1] - 9 * p[2] + 3 * p[3]) * t * t);
}

// Second derivative of the 1D Catmull-Rom spline
__host__ __device__ inline float cubic_cr_grad2(const float p[4], float t) noexcept {
    return (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3] +
        (-3 * p[0] + 9 * p[1] - 9 * p[2] + 3 * p[3]) * t);
}

// Tricubic grid interpolation
template <typename T>
__host__ __device__ float tricubic_interpolation(const T * data, const int * dims,
    double x, double y, double z, float* grad, float hessian[3][3]) {
    
    // Extract grid dimensions
    long long nx = dims[0], ny = dims[1], nz = dims[2];

    // Determine the indices of the grid cell containing the point
    long long ix = static_cast<long long>(std::floor(x));
    long long iy = static_cast<long long>(std::floor(y));
    long long iz = static_cast<long long>(std::floor(z));

    // Compute local coordinates within the cell
    float dx = static_cast<float>(x) - static_cast<float>(ix);
    float dy = static_cast<float>(y) - static_cast<float>(iy);
    float dz = static_cast<float>(z) - static_cast<float>(iz);

    // 1) along x
    float tmp_yz[4][4];
    float tmp_gradx_yz[4][4];
    float tmp_grad2x_yz[4][4];
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            if constexpr (std::is_same_v<T, __half>)
            {
                float p[4] = { // x-direction is contiguous in memory
                    __half2float(data[((ix - 1L + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny]) / 134217728.0f,
                    __half2float(data[((ix     + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny]) / 134217728.0f,
                    __half2float(data[((ix + 1L + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny]) / 134217728.0f,
                    __half2float(data[((ix + 2L + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny]) / 134217728.0f
                };
                tmp_yz[j][k] = cubic_cr(p, dx);
                tmp_gradx_yz[j][k] = cubic_cr_grad(p, dx);
                tmp_grad2x_yz[j][k] = cubic_cr_grad2(p, dx);
            }
            else
            {
                float p[4] = { // x-direction is contiguous in memory
                    data[((ix - 1L + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny],
                    data[((ix     + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny],
                    data[((ix + 1L + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny],
                    data[((ix + 2L + 20L * nx) % nx) + ((iy + j - 1L + 20L * ny) % ny) * nx + ((iz + k - 1L + 20L * nz) % nz) * nx * ny]
                };
                tmp_yz[j][k] = cubic_cr(p, dx);
                tmp_gradx_yz[j][k] = cubic_cr_grad(p, dx);
                tmp_grad2x_yz[j][k] = cubic_cr_grad2(p, dx);
            }
        }
    }

    // 2) along y
    float tmp_z[4];
    float tmp_gradx_z[4];
    float tmp_grady_z[4];
    float tmp_grad2x_z[4];
    float tmp_grad2y_z[4];
    float tmp_gradxy_z[4];
    for (int k = 0; k < 4; ++k) {
        float p[4] = {
            tmp_yz[0][k],
            tmp_yz[1][k],
            tmp_yz[2][k],
            tmp_yz[3][k]
        };
        tmp_z[k] = cubic_cr(p, dy);
        tmp_grady_z[k] = cubic_cr_grad(p, dy);
        tmp_grad2y_z[k] = cubic_cr_grad2(p, dy);
        p[0] = tmp_gradx_yz[0][k];
        p[1] = tmp_gradx_yz[1][k];
        p[2] = tmp_gradx_yz[2][k];
        p[3] = tmp_gradx_yz[3][k];
        tmp_gradx_z[k] = cubic_cr(p, dy);
        tmp_gradxy_z[k] = cubic_cr_grad(p, dy);
        p[0] = tmp_grad2x_yz[0][k];
        p[1] = tmp_grad2x_yz[1][k];
        p[2] = tmp_grad2x_yz[2][k];
        p[3] = tmp_grad2x_yz[3][k];
        tmp_grad2x_z[k] = cubic_cr(p, dy);
    }

    // 3) along z
    grad[0] = cubic_cr(tmp_gradx_z, dz);
    grad[1] = cubic_cr(tmp_grady_z, dz);
    grad[2] = cubic_cr_grad(tmp_z, dz);

    
    hessian[0][0] = cubic_cr(tmp_grad2x_z, dz);
    hessian[0][1] = cubic_cr(tmp_gradxy_z, dz);
    hessian[0][2] = cubic_cr_grad(tmp_gradx_z, dz);
    hessian[1][0] = hessian[0][1];
    hessian[1][1] = cubic_cr(tmp_grad2y_z, dz);
    hessian[1][2] = cubic_cr_grad(tmp_grady_z, dz);
    hessian[2][0] = hessian[0][2];
    hessian[2][1] = hessian[1][2];
    hessian[2][2] = cubic_cr_grad2(tmp_z, dz);

    return cubic_cr(tmp_z, dz);
}

// Helper function to do the raytracing with GSL ODE solver
__host__ __device__ void raytracing_ode(double t, const double y[], double dydt[], const float* data, const __half * B1data, const __half * B2data, const __half * B3data, const int * dims) {
    // Unpack the state vector
    const double x_pos = y[0];
    const double y_pos = y[1];
    const double z_pos = y[2];
    double nx = y[3];
    double ny = y[4];
    double nz = y[5];
    double e1x = y[6];
    double e1y = y[7];
    double e1z = y[8];
    double e2x = y[9];
    double e2y = y[10];
    double e2z = y[11];
    const double dA = y[12];
    const double dAprime = y[13];
    const double sigma1 = y[14];
    const double sigma2 = y[15];
    const double ellipticity1 = y[16];
    const double ellipticity2 = y[17];
    //const double rotation = y[18];

    // Normalize n, e1 and e2
    double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    nx /= norm;
    ny /= norm;
    nz /= norm;
    norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    e1x /= norm;
    e1y /= norm;
    e1z /= norm;
    norm = std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);
    e2x /= norm;
    e2y /= norm;
    e2z /= norm;

    double transverse_projector[3][3] = {
        {nx * nx - 1, nx * ny, nx * nz},
        {nx * ny, ny * ny - 1, ny * nz},
        {nx * nz, ny * nz, nz * nz - 1}
    };

    double shear_projector1[3][3] = {
        {e1x * e1x - e2x * e2x, e1x * e1y - e2x * e2y, e1x * e1z - e2x * e2z},
        {e1x * e1y - e2x * e2y, e1y * e1y - e2y * e2y, e1y * e1z - e2y * e2z},
        {e1x * e1z - e2x * e2z, e1y * e1z - e2y * e2z, e1z * e1z - e2z * e2z}
    };

    double shear_projector2[3][3] = {
        {2 * e1x * e2x, e1x * e2y + e1y * e2x, e1x * e2z + e1z * e2x},
        {e1x * e2y + e1y * e2x, 2 * e1y * e2y, e1y * e2z + e1z * e2y},
        {e1x * e2z + e1z * e2x, e1y * e2z + e1z * e2y, 2 * e1z * e2z}
    };

    // Compute the gradient and hessian
    float grad[3];
    float ngradB[3];
    float hessian[3][3];
    float tidalBterm[3][3];
    float e1gradB[3] = {0,0,0};
    float e2gradB[3] = {0,0,0};

    float Bx = tricubic_interpolation(B1data, dims, x_pos - 0.5, y_pos, z_pos, grad, hessian); // B1 is offset by -0.5 in x

    ngradB[0] = grad[0] * nx;
    ngradB[1] = grad[1] * nx;
    ngradB[2] = grad[2] * nx;

    float e1ngradB = 0.5 * (grad[0] * nx + grad[1] * ny + grad[2] * nz);
    float e2ngradB = e1ngradB;

    e1ngradB = e1ngradB * e1x + 0.5 * (ngradB[0] * e1x + ngradB[1] * e1y + ngradB[2] * e1z);
    e2ngradB = e2ngradB * e2x + 0.5 * (ngradB[0] * e2x + ngradB[1] * e2y + ngradB[2] * e2z);

    tidalBterm[0][0] = hessian[0][0] * nx - (nx * hessian[0][0] + ny * hessian[1][0] + nz * hessian[2][0]);
    tidalBterm[0][1] = hessian[0][1] * nx - 0.5 * (nx * hessian[0][1] + ny * hessian[1][1] + nz * hessian[2][1]);
    tidalBterm[0][2] = hessian[0][2] * nx - 0.5 * (nx * hessian[0][2] + ny * hessian[1][2] + nz * hessian[2][2]);
    tidalBterm[1][1] = hessian[1][1] * nx;
    tidalBterm[1][2] = hessian[1][2] * nx;
    tidalBterm[2][2] = hessian[2][2] * nx;

    float nLaplaceB = nx * (hessian[0][0] + hessian[1][1] + hessian[2][2]);

    e1gradB[0] = 0.5 * (e1y * grad[1] + e1z * grad[2]);
    e1gradB[1] = -0.5 * e1x * grad[1];
    e1gradB[2] = -0.5 * e1x * grad[2];

    e2gradB[0] = 0.5 * (e2y * grad[1] + e2z * grad[2]);
    e2gradB[1] = -0.5 * e2x * grad[1];
    e2gradB[2] = -0.5 * e2x * grad[2];

    float By = tricubic_interpolation(B2data, dims, x_pos, y_pos - 0.5, z_pos, grad, hessian); // B2 is offset by -0.5 in y

    ngradB[0] += grad[0] * ny;
    ngradB[1] += grad[1] * ny;
    ngradB[2] += grad[2] * ny;

    e1ngradB += 0.5 * e1y * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * ny * (grad[0] * e1x + grad[1] * e1y + grad[2] * e1z);
    e2ngradB += 0.5 * e2y * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * ny * (grad[0] * e2x + grad[1] * e2y + grad[2] * e2z);

    tidalBterm[0][0] += hessian[0][0] * ny;
    tidalBterm[0][1] += hessian[0][1] * ny - 0.5 * (nx * hessian[0][0] + ny * hessian[1][0] + nz * hessian[2][0]);
    tidalBterm[0][2] += hessian[0][2] * ny;
    tidalBterm[1][1] += hessian[1][1] * ny - (nx * hessian[0][1] + ny * hessian[1][1] + nz * hessian[2][1]);
    tidalBterm[1][2] += hessian[1][2] * ny - 0.5 * (nx * hessian[0][2] + ny * hessian[1][2] + nz * hessian[2][2]);
    tidalBterm[2][2] += hessian[2][2] * ny;

    nLaplaceB += (hessian[0][0] + hessian[1][1] + hessian[2][2]) * ny;

    e1gradB[0] += -0.5 * e1y * grad[0];
    e1gradB[1] += 0.5 * (e1x * grad[0] + e1z * grad[2]);
    e1gradB[2] += -0.5 * e1y * grad[2];

    e2gradB[0] += -0.5 * e2y * grad[0];
    e2gradB[1] += 0.5 * (e2x * grad[0] + e2z * grad[2]);
    e2gradB[2] += -0.5 * e2y * grad[2];

    float Bz = tricubic_interpolation(B3data, dims, x_pos, y_pos, z_pos - 0.5, grad, hessian); // B3 is offset by -0.5 in z

    ngradB[0] += grad[0] * nz;
    ngradB[1] += grad[1] * nz;
    ngradB[2] += grad[2] * nz;

    e1ngradB += 0.5 * e1z * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * nz * (grad[0] * e1x + grad[1] * e1y + grad[2] * e1z);
    e2ngradB += 0.5 * e2z * (grad[0] * nx + grad[1] * ny + grad[2] * nz) + 0.5 * nz * (grad[0] * e2x + grad[1] * e2y + grad[2] * e2z);

    tidalBterm[0][0] += hessian[0][0] * nz;
    tidalBterm[0][1] += hessian[0][1] * nz;
    tidalBterm[0][2] += hessian[0][2] * nz - 0.5 * (nx * hessian[0][0] + ny * hessian[1][0] + nz * hessian[2][0]);
    tidalBterm[1][1] += hessian[1][1] * nz;
    tidalBterm[1][2] += hessian[1][2] * nz - 0.5 * (nx * hessian[0][1] + ny * hessian[1][1] + nz * hessian[2][1]);
    tidalBterm[2][2] += hessian[2][2] * nz - (nx * hessian[0][2] + ny * hessian[1][2] + nz * hessian[2][2]);

    tidalBterm[1][0] = tidalBterm[0][1];
    tidalBterm[2][0] = tidalBterm[0][2];
    tidalBterm[2][1] = tidalBterm[1][2];

    nLaplaceB += (hessian[0][0] + hessian[1][1] + hessian[2][2]) * nz;

    e1gradB[0] += -0.5 * e1z * grad[0];
    e1gradB[1] += -0.5 * e1z * grad[1];
    e1gradB[2] += 0.5 * (e1x * grad[0] + e1y * grad[1]);

    e2gradB[0] += -0.5 * e2z * grad[0];
    e2gradB[1] += -0.5 * e2z * grad[1];
    e2gradB[2] += 0.5 * (e2x * grad[0] + e2y * grad[1]);

    float phi = tricubic_interpolation(data, dims, x_pos, y_pos, z_pos, grad, hessian);

    // Compute the derivatives
    dydt[0] = -nx * (1 + 2 * phi) - Bx;
    dydt[1] = -ny * (1 + 2 * phi) - By;
    dydt[2] = -nz * (1 + 2 * phi) - Bz;

    dydt[3] = -(transverse_projector[0][0] * (2 * grad[0] + ngradB[0]) + transverse_projector[0][1] * (2 * grad[1] + ngradB[1]) + transverse_projector[0][2] * (2 * grad[2] + ngradB[2]));
    dydt[4] = -(transverse_projector[1][0] * (2 * grad[0] + ngradB[0]) + transverse_projector[1][1] * (2 * grad[1] + ngradB[1]) + transverse_projector[1][2] * (2 * grad[2] + ngradB[2]));
    dydt[5] = -(transverse_projector[2][0] * (2 * grad[0] + ngradB[0]) + transverse_projector[2][1] * (2 * grad[1] + ngradB[1]) + transverse_projector[2][2] * (2 * grad[2] + ngradB[2]));

    dydt[6] = -2 * nx * (e1x * grad[0] + e1y * grad[1] + e1z * grad[2]) - nx * e1ngradB;
    dydt[7] = -2 * ny * (e1x * grad[0] + e1y * grad[1] + e1z * grad[2]) - ny * e1ngradB;
    dydt[8] = -2 * nz * (e1x * grad[0] + e1y * grad[1] + e1z * grad[2]) - nz * e1ngradB;

    dydt[9] = -2 * nx * (e2x * grad[0] + e2y * grad[1] + e2z * grad[2]) - nx * e2ngradB;
    dydt[10] = -2 * ny * (e2x * grad[0] + e2y * grad[1] + e2z * grad[2]) - ny * e2ngradB;
    dydt[11] = -2 * nz * (e2x * grad[0] + e2y * grad[1] + e2z * grad[2]) - nz * e2ngradB;

    dydt[12] = -dAprime;
    dydt[13] = -4 * dAprime * (nx * grad[0] + ny * grad[1] + nz * grad[2] + 0.25 * (nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2])) - (transverse_projector[0][0] * hessian[0][0] + transverse_projector[0][1] * hessian[0][1] + transverse_projector[0][2] * hessian[0][2] + transverse_projector[1][0] * hessian[1][0] + transverse_projector[1][1] * hessian[1][1] + transverse_projector[1][2] * hessian[1][2] + transverse_projector[2][0] * hessian[2][0] + transverse_projector[2][1] * hessian[2][1] + transverse_projector[2][2] * hessian[2][2]) * dA + 0.5 * nLaplaceB * dA;

    hessian[0][0] += 0.5 * tidalBterm[0][0];
    hessian[0][1] += 0.5 * tidalBterm[0][1];
    hessian[0][2] += 0.5 * tidalBterm[0][2];
    hessian[1][0] += 0.5 * tidalBterm[1][0];
    hessian[1][1] += 0.5 * tidalBterm[1][1];
    hessian[1][2] += 0.5 * tidalBterm[1][2];
    hessian[2][0] += 0.5 * tidalBterm[2][0];
    hessian[2][1] += 0.5 * tidalBterm[2][1];
    hessian[2][2] += 0.5 * tidalBterm[2][2];

    dydt[14] = 4 * sigma1 * (nx * grad[0] + ny * grad[1] + nz * grad[2] + 0.25 * (nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2])) + (shear_projector1[0][0] * hessian[0][0] + shear_projector1[0][1] * hessian[0][1] + shear_projector1[0][2] * hessian[0][2] + shear_projector1[1][0] * hessian[1][0] + shear_projector1[1][1] * hessian[1][1] + shear_projector1[1][2] * hessian[1][2] + shear_projector1[2][0] * hessian[2][0] + shear_projector1[2][1] * hessian[2][1] + shear_projector1[2][2] * hessian[2][2]) * dA * dA;
    dydt[15] = 4 * sigma2 * (nx * grad[0] + ny * grad[1] + nz * grad[2] + 0.25 * (nx * ngradB[0] + ny * ngradB[1] + nz * ngradB[2])) + (shear_projector2[0][0] * hessian[0][0] + shear_projector2[0][1] * hessian[0][1] + shear_projector2[0][2] * hessian[0][2] + shear_projector2[1][0] * hessian[1][0] + shear_projector2[1][1] * hessian[1][1] + shear_projector2[1][2] * hessian[1][2] + shear_projector2[2][0] * hessian[2][0] + shear_projector2[2][1] * hessian[2][1] + shear_projector2[2][2] * hessian[2][2]) * dA * dA;

    if (dA > 0.5)
    {
        dydt[13] += (sigma1 * sigma1 + sigma2 * sigma2) / (dA * dA * dA);
        dydt[16] = -2 * sigma1 * std::sqrt(4 + ellipticity1 * ellipticity1 + ellipticity2 * ellipticity2) / (dA * dA);
        dydt[17] = -2 * sigma2 * std::sqrt(4 + ellipticity1 * ellipticity1 + ellipticity2 * ellipticity2) / (dA * dA);
        dydt[18] = -(sigma1 * ellipticity2 - sigma2 * ellipticity1) / (dA * dA * (2 + std::sqrt(4 + ellipticity1 * ellipticity1 + ellipticity2 * ellipticity2)));
    }
    else
    {
        dydt[16] = 0;
        dydt[17] = 0;
        dydt[18] = 0;
    }
}

// Raytracing function
__host__ __device__ void raytracing(const float * data, const __half * B1data, const __half * B2data, const __half * B3data, const int * dims, double y[19], int n_steps) {

    // Initial state vector

    // temporary arrays for RK4 integration
    double temp[19], k[19], kacc[19];

    // Time variable
    double t = 0.0;

    // Perform the RK4 integration
    for (int i = 0; i < n_steps; ++i) {
        //int status = gsl_odeiv2_driver_apply(d, &t, t + 1.0, y);
        raytracing_ode(t, y, kacc, data, B1data, B2data, B3data, dims);
        #pragma unroll
        for (int j = 0; j < 19; ++j) {
            temp[j] = y[j] + 0.5 * kacc[j];
        }
        raytracing_ode(t + 0.5, temp, k, data, B1data, B2data, B3data, dims);
        #pragma unroll
        for (int j = 0; j < 19; ++j) {
            temp[j] = y[j] + 0.5 * k[j];
            kacc[j] += 2 * k[j];
        }
        raytracing_ode(t + 1.0, temp, k, data, B1data, B2data, B3data, dims);
        #pragma unroll
        for (int j = 0; j < 19; ++j) {
            temp[j] = y[j] + k[j];
            kacc[j] += 2 * k[j];
        }
        raytracing_ode(t + 1.0, temp, k, data, B1data, B2data, B3data, dims);
        #pragma unroll
        for (int j = 0; j < 19; ++j) {
            y[j] += (kacc[j] + k[j]) / 6.0;
        }
        t += 1.0; // advance time (system is autonomous, but keeps semantics clear)
    }
}

__global__ void raytracing_kernel(const float * data, const __half * B1data, const __half * B2data, const __half * B3data, const int * dims, const int64_t Nside,
    const int64_t iend, const int n_steps, double * thetamap, double * phimap,
    float * dAmap, float * ellipticity1map, float * ellipticity2map,
    float * rotationmap, float * deflection1map, float * deflection2map) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= iend) return;

    double y[19] = { static_cast<double>(dims[0]) * 0.5, static_cast<double>(dims[1]) * 0.5, static_cast<double>(dims[2]) * 0.5, -std::sin(thetamap[i]) * std::cos(phimap[i]), -std::sin(thetamap[i]) * std::sin(phimap[i]), -std::cos(thetamap[i]), std::cos(thetamap[i]) * std::cos(phimap[i]), std::cos(thetamap[i]) * std::sin(phimap[i]), -std::sin(thetamap[i]), std::sin(phimap[i]), -std::cos(phimap[i]), 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    
    // Perform raytracing
    raytracing(data, B1data, B2data, B3data, dims, y, n_steps);

    // compute deflection angle
    y[0] -= static_cast<double>(dims[0]) * 0.5;
    y[1] -= static_cast<double>(dims[1]) * 0.5;
    y[2] -= static_cast<double>(dims[2]) * 0.5;

    double norm = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);

    y[0] = y[0] / norm - std::sin(thetamap[i]) * std::cos(phimap[i]);
    y[1] = y[1] / norm - std::sin(thetamap[i]) * std::sin(phimap[i]);
    y[2] = y[2] / norm - std::cos(thetamap[i]);

    // Store the results
    dAmap[i] = static_cast<float>(y[12]);
    ellipticity1map[i] = static_cast<float>(y[16]);
    ellipticity2map[i] = static_cast<float>(y[17]);
    rotationmap[i] = static_cast<float>(y[18]);
    deflection1map[i] = static_cast<float>(y[0] * std::cos(thetamap[i]) * std::cos(phimap[i]) + y[1] * std::cos(thetamap[i]) * std::sin(phimap[i]) - y[2] * std::sin(thetamap[i]));
    deflection2map[i] = static_cast<float>(y[0] * std::sin(phimap[i]) - y[1] * std::cos(phimap[i]));
}

__global__ void vector3_to_half(const float * vec3, __half * h1, __half * h2, __half * h3, int64_t n, int64_t xy, int64_t offset) {
    int64_t i = static_cast<int64_t>(blockIdx.y) * xy + static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    if ((offset + i) % 3 == 0) {
        h1[(offset + i) / 3] = __float2half(vec3[i]*134217728.0f); // scale by 2^27 to preserve precision
    }
    else if ((offset + i) % 3 == 1) {
        h2[(offset + i) / 3] = __float2half(vec3[i]*134217728.0f);
    }
    else {
        h3[(offset + i) / 3] = __float2half(vec3[i]*134217728.0f);
    }
}

// Main function
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <hdf5_file_potential> <hdf5_file_B> <Nside> <n_steps> [<batch>]" << std::endl;
        std::cerr << "  <hdf5_file_potential>: HDF5 file containing a snapshot of the Weyl potential" << std::endl;
        std::cerr << "  <hdf5_file_B>: HDF5 file containing a snapshot of the frame-dragging potential (shift)" << std::endl;
        std::cerr << "  <Nside>: HEALPix Nside parameter for output maps" << std::endl;
        std::cerr << "  <n_steps>: Number of integration steps for raytracing (1 step = 1 grid unit)" << std::endl;
        std::cerr << "  [<batch>]: Optional batch index for processing a subset of pixels" << std::endl;
        return -1;
    }

    int64_t Nside = std::stol(argv[3]);
    if (Nside <= 0) {
        std::cerr << "Invalid Nside value: " << Nside << std::endl;
        return -1;
    }

    int n_steps = std::stoi(argv[4]);
    if (n_steps <= 0) {
        std::cerr << "Invalid n_steps value: " << n_steps << std::endl;
        return -1;
    }

    int batch = -1;
    if (argc > 5) {
        batch = std::stoi(argv[5]);
        if (batch < 0) {
            std::cerr << "Invalid batch value: " << batch << std::endl;
            return -1;
        }
    }

    std::cout << "Nside: " << Nside << std::endl;
    std::cout << "n_steps: " << n_steps << std::endl;
    std::cout << "Loading potential data from HDF5 file " << argv[1] << " ..." << std::endl;

    // Load the HDF5 file
    std::string filename = argv[1];
    std::vector<hsize_t> dims;
    std::vector<float> data = load_hdf5_array(filename, "/data", dims);
    if (data.empty()) {
        return -1;
    }

    int dims_int[3] = { static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]) };

    std::cout << "Data dimensions: " << dims_int[0] << " x " << dims_int[1] << " x " << dims_int[2] << std::endl;

    // Basic sanity/overflow check for dimension products (guard against silent overflow on size_t casts elsewhere)
    {
        long double voxels_ld = static_cast<long double>(dims[0]) * static_cast<long double>(dims[1]) * static_cast<long double>(dims[2]);
        long double bytes_potential = voxels_ld * sizeof(float);
        if (voxels_ld > static_cast<long double>(std::numeric_limits<size_t>::max())) {
            std::cerr << "Voxel count exceeds size_t range; aborting." << std::endl;
            return -1;
        }
        if (bytes_potential > 0.85L * static_cast<long double>(std::numeric_limits<size_t>::max())) {
            std::cerr << "Potential data size dangerously close to size_t limit; aborting." << std::endl;
            return -1;
        }
    }

#ifdef RESCALE_DATA
    // rescale the data by constant factor RESCALE_DATA
    std::cout << "Rescaling potential data by factor " << RESCALE_DATA << std::endl;
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= RESCALE_DATA;
    }
#endif

    std::cout << "Loading B field data from HDF5 file " << argv[2] << " ..." << std::endl;

    // Load the HDF5 file
    filename = argv[2];
    std::vector<float> Bdata = load_hdf5_vector_field(filename, "/data", dims);
    if (Bdata.empty()) {
        return -1;
    }

#ifdef RESCALE_BDATA
    // rescale the Bdata by constant factor RESCALE_BDATA
    std::cout << "Rescaling B field data by factor " << RESCALE_BDATA << std::endl;
    #pragma omp parallel for
    for (size_t i = 0; i < Bdata.size(); ++i) {
        Bdata[i] *= RESCALE_BDATA;
    }
#endif

    double * thetamap = nullptr;
    double * phimap = nullptr;

    float * dAmap_device = nullptr;
    float * ellipticity1map_device = nullptr;
    float * ellipticity2map_device = nullptr;
    float * rotationmap_device = nullptr;
    float * deflection1map_device = nullptr;
    float * deflection2map_device = nullptr;

    float * data_device = nullptr;
    int * dims_device = nullptr;
    __half * B1_device = nullptr;
    __half * B2_device = nullptr;
    __half * B3_device = nullptr;

    cudaError_t success;

    int64_t istart = 0;
    int64_t iend = 12L * Nside * Nside;
    if (batch >= 0) {
        istart = static_cast<int64_t>(batch) * Nside * Nside;
        iend = Nside * Nside;
        if (istart >= 12L * Nside * Nside) {
            std::cerr << "Batch exceeds total number of pixels" << std::endl;
            return -1;
        }
    }

    std::cout << "Allocating device memory and copying data to device..." << std::endl;

    success = cudaMalloc(&dAmap_device, iend * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for dAmap: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&ellipticity1map_device, iend * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for ellipticity1map: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&ellipticity2map_device, iend * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for ellipticity2map: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&rotationmap_device, iend * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for rotationmap: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&deflection1map_device, iend * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for deflection1map: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&deflection2map_device, iend * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for deflection2map: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaMalloc(&data_device, (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(float));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for data_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaMalloc(&B1_device, (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(__half));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for B1_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&B2_device, (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(__half));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for B2_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&B3_device, (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(__half));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for B3_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    // first, copy the vector field to device in three batches
    success = cudaMemcpy(data_device, Bdata.data(), (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(float), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy data to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    int threads_per_block = 128;
    int blocks = ((long) dims[1] * (long) dims[2] + threads_per_block - 1) / threads_per_block;

    vector3_to_half<<<dim3(blocks, dims[0]), dim3(threads_per_block, 1)>>>(data_device, B1_device, B2_device, B3_device, (long) dims[0] * (long) dims[1] * (long) dims[2], (long) dims[1] * (long) dims[2], 0);

    success = cudaGetLastError();
    if (success != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaDeviceSynchronize();
    if (success != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaMemcpy(data_device, Bdata.data() + (long) dims[0] * (long) dims[1] * (long) dims[2], (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(float), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy data to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    vector3_to_half<<<dim3(blocks, dims[0]), dim3(threads_per_block, 1)>>>(data_device, B1_device, B2_device, B3_device, (long) dims[0] * (long) dims[1] * (long) dims[2], (long) dims[1] * (long) dims[2], (long) dims[0] * (long) dims[1] * (long) dims[2]);

    success = cudaGetLastError();
    if (success != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaDeviceSynchronize();
    if (success != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaMemcpy(data_device, Bdata.data() + 2L * (long) dims[0] * (long) dims[1] * (long) dims[2], (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(float), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy data to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    vector3_to_half<<<dim3(blocks, dims[0]), dim3(threads_per_block, 1)>>>(data_device, B1_device, B2_device, B3_device, (long) dims[0] * (long) dims[1] * (long) dims[2], (long) dims[1] * (long) dims[2], 2L * (long) dims[0] * (long) dims[1] * (long) dims[2]);

    success = cudaGetLastError();
    if (success != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaDeviceSynchronize();
    if (success != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    Bdata.clear();

    success = cudaMemcpy(data_device, data.data(), (long) dims[0] * (long) dims[1] * (long) dims[2] * sizeof(float), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy data to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaMalloc(&dims_device, 3 * sizeof(int));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for dims_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMemcpy(dims_device, dims_int, 3 * sizeof(int), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy dims to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    thetamap = (double *) malloc(iend * sizeof(double));
    phimap = (double *) malloc(iend * sizeof(double));

    std::cout << "Computing theta and phi angles..." << std::endl;

    // compute theta and phi
#pragma omp parallel for shared(thetamap, phimap)
    for (int64_t i = 0; i < iend; ++i) {
        pix2ang_ring64(static_cast<int64_t>(Nside), static_cast<int64_t>(i+istart), &thetamap[i], &phimap[i]);
    }

    double * thetamap_device = nullptr;
    double * phimap_device = nullptr;
    success = cudaMalloc(&thetamap_device, iend * sizeof(double));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for thetamap_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMalloc(&phimap_device, iend * sizeof(double));
    if (success != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for phimap_device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMemcpy(thetamap_device, thetamap, iend * sizeof(double), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy thetamap to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    success = cudaMemcpy(phimap_device, phimap, iend * sizeof(double), cudaMemcpyHostToDevice);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy phimap to device: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    free(thetamap);
    free(phimap);

    std::cout << "Launching raytracing kernel..." << std::endl;

    // Launch the kernel
    blocks = (iend + threads_per_block - 1) / threads_per_block;
    raytracing_kernel<<<blocks, threads_per_block>>>(data_device, B1_device, B2_device, B3_device, dims_device, Nside, iend, n_steps,
        thetamap_device, phimap_device,
        dAmap_device, ellipticity1map_device, ellipticity2map_device,
        rotationmap_device, deflection1map_device, deflection2map_device);

    success = cudaGetLastError();
    if (success != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    success = cudaDeviceSynchronize();
    if (success != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }

    std::cout << "Raytracing completed." << std::endl;

    cudaFree(thetamap_device);
    cudaFree(phimap_device);
    cudaFree(data_device);
    cudaFree(dims_device);

    // Save the results to FITS files
    
    char filename_buffer[256];
    float * outmap = nullptr;
    outmap = (float *) malloc(12L * Nside * Nside * sizeof(float));
    if (outmap == nullptr) {
        std::cerr << "Failed to allocate memory for output map" << std::endl;
        return -1;
    }

#pragma omp parallel for shared(outmap)
    for (int64_t i = 0; i < 12L * Nside * Nside; ++i) {
        outmap[i] = 0.0f;
    }

    std::cout << "Saving results to FITS files..." << std::endl;

    if (batch < 0) batch = 0;

    snprintf(filename_buffer, sizeof(filename_buffer), "dAmap_batch%d.fits", batch);
    success = cudaMemcpy(outmap + istart, dAmap_device, iend * sizeof(float), cudaMemcpyDeviceToHost);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy dAmap to host: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    write_healpix_map(outmap, Nside, filename_buffer, 0, "C");

    snprintf(filename_buffer, sizeof(filename_buffer), "ellipticity1map_batch%d.fits", batch);
    success = cudaMemcpy(outmap + istart, ellipticity1map_device, iend * sizeof(float), cudaMemcpyDeviceToHost);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy ellipticity1map to host: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    write_healpix_map(outmap, Nside, filename_buffer, 0, "C");

    snprintf(filename_buffer, sizeof(filename_buffer), "ellipticity2map_batch%d.fits", batch);
    success = cudaMemcpy(outmap + istart, ellipticity2map_device, iend * sizeof(float), cudaMemcpyDeviceToHost);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy ellipticity2map to host: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    write_healpix_map(outmap, Nside, filename_buffer, 0, "C");

    snprintf(filename_buffer, sizeof(filename_buffer), "rotationmap_batch%d.fits", batch);
    success = cudaMemcpy(outmap + istart, rotationmap_device, iend * sizeof(float), cudaMemcpyDeviceToHost);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy rotationmap to host: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    write_healpix_map(outmap, Nside, filename_buffer, 0, "C");

    snprintf(filename_buffer, sizeof(filename_buffer), "deflection1map_batch%d.fits", batch);
    success = cudaMemcpy(outmap + istart, deflection1map_device, iend * sizeof(float), cudaMemcpyDeviceToHost);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy deflection1map to host: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    write_healpix_map(outmap, Nside, filename_buffer, 0, "C");

    snprintf(filename_buffer, sizeof(filename_buffer), "deflection2map_batch%d.fits", batch);
    success = cudaMemcpy(outmap + istart, deflection2map_device, iend * sizeof(float), cudaMemcpyDeviceToHost);
    if (success != cudaSuccess) {
        std::cerr << "Failed to copy deflection2map to host: " << cudaGetErrorString(success) << std::endl;
        return -1;
    }
    write_healpix_map(outmap, Nside, filename_buffer, 0, "C");

    // Free the output map
    free(outmap);

    // Free the device memory
    cudaFree(dAmap_device);
    cudaFree(ellipticity1map_device);
    cudaFree(ellipticity2map_device);
    cudaFree(rotationmap_device);
    cudaFree(deflection1map_device);
    cudaFree(deflection2map_device);

    // Free the data array
    data.clear();
    dims.clear();

    std::cout << "All done!" << std::endl;

    return 0;
}
        
