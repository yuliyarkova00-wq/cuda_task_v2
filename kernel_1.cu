#include "kernel_01.cuh"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <cmath>

#define IDX(i,j,k) (((k)*(ny+2) + (j))*(nx+2) + (i))

const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;

__device__ double d_an_sol(double x, double y, double z, double t) {
    const double at = 0.5 * sqrt(4.0 / (Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));
    const double pi = M_PI;
    return sin(2.0 * pi * x / Lx) * sin(pi * y / Ly) * sin(pi * z / Lz) * cos(at * t + 2.0 * pi);
}

__device__ double d_compute_p(double x, double y, double z) {
    return d_an_sol(x, y, z, 0.0);
}

__global__ void it0_kernel(
    double* u0,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double* err)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;

    if (ii >= nx || jj >= ny || kk >= nz) return;

    int i = ii + 1;
    int j = jj + 1;
    int k = kk + 1;

    int gi = sx + ii;
    int gj = sy + jj;
    int gk = sz + kk;

    double x = gi * dx;
    double y = gj * dy;
    double z = gk * dz;

    int id = IDX(i,j,k);
    double val = d_compute_p(x,y,z);
    u0[id] = val;

    int tid = ii + nx * (jj + ny * kk);
    err[tid] = fabs(val - d_an_sol(x,y,z,0.0));
}

__global__ void it1_kernel(
    double* u0,
    double* u1,
    double *d_bx0, double *d_bx1, double *d_by0, double *d_by1,
    double *d_bz0, double *d_bz1,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau,
    double* err, bool write_bounds
){
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;

    if (ii >= nx || jj >= ny || kk >= nz) return;

    int i = ii + 1;
    int j = jj + 1;
    int k = kk + 1;

    int gi = sx + ii;
    int gj = sy + jj;
    int gk = sz + kk;

    double x = gi * dx;
    double y = gj * dy;
    double z = gk * dz;

    double d2x = (d_compute_p(x+dx,y,z) - 2*d_compute_p(x,y,z) + d_compute_p(x-dx,y,z))/(dx*dx);
    double d2y = (d_compute_p(x,y+dy,z) - 2*d_compute_p(x,y,z) + d_compute_p(x,y-dy,z))/(dy*dy);
    double d2z = (d_compute_p(x,y,z+dz) - 2*d_compute_p(x,y,z) + d_compute_p(x,y,z-dz))/(dz*dz);

    int id = IDX(i,j,k);
    double val = u0[id] + 0.5 * a * tau * tau * (d2x+d2y+d2z);
    u1[id] = val;

    if (write_bounds){
        if (ii == 0)      d_bx0[jj*nz + kk] = val;
        if (ii == nx-1)   d_bx1[jj*nz + kk] = val;
        if (jj == 0)      d_by0[ii*nz + kk] = val;
        if (jj == ny-1)   d_by1[ii*nz + kk] = val;
        if (kk == 0)      d_bz0[ii*ny + jj] = val;
        if (kk == nz-1)   d_bz1[ii*ny + jj] = val;
    }

    int tid = ii + nx * (jj + ny * kk);
    err[tid] = fabs(val - d_an_sol(x,y,z,tau));
}

__global__ void main_kernel(
    double* u0,
    double* u1,
    double* u2,
    double *d_bx0, double *d_bx1,
    double *d_by0, double *d_by1,
    double *d_bz0, double *d_bz1,
    int nx, int ny, int nz,
    int gx, int gy, int gz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau,
    double t,
    double* err, bool write_bounds
){
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;

    if (ii >= nx || jj >= ny || kk >= nz) return;

    int i = ii + 1;
    int j = jj + 1;
    int k = kk + 1;

    if (write_bounds){
        if (ii == 0)      u1[IDX(0,j,k)]     = d_bx0[jj*nz + kk];
        if (ii == nx-1)   u1[IDX(nx+1,j,k)]  = d_bx1[jj*nz + kk];
        if (jj == 0)      u1[IDX(i,0,k)]     = d_by0[ii*nz + kk];
        if (jj == ny-1)   u1[IDX(i,ny+1,k)]  = d_by1[ii*nz + kk];
        if (kk == 0)      u1[IDX(i,j,0)]     = d_bz0[ii*ny + jj];
        if (kk == nz-1)   u1[IDX(i,j,nz+1)]  = d_bz1[ii*ny + jj];
    }

    int gi = sx + ii;
    int gj = sy + jj;
    int gk = sz + kk;

    int id = IDX(i,j,k);

    if (gi==0 || gi==gx-1 || gj==0 || gj==gy-1 || gk==0 || gk==gz-1) {
        u2[id] = 0.0;
        err[ii + nx * (jj + ny * kk)] = 0.0;
        return;
    }

    double lap =
        (u1[IDX(i+1,j,k)] - 2*u1[id] + u1[IDX(i-1,j,k)])/(dx*dx) +
        (u1[IDX(i,j+1,k)] - 2*u1[id] + u1[IDX(i,j-1,k)])/(dy*dy) +
        (u1[IDX(i,j,k+1)] - 2*u1[id] + u1[IDX(i,j,k-1)])/(dz*dz);

    double val = 2*u1[id] - u0[id] + a*tau*tau*lap;
    u2[id] = val;

    if (write_bounds){
        if (ii == 0)      d_bx0[jj*nz + kk] = val;
        if (ii == nx-1)   d_bx1[jj*nz + kk] = val;
        if (jj == 0)      d_by0[ii*nz + kk] = val;
        if (jj == ny-1)   d_by1[ii*nz + kk] = val;
        if (kk == 0)      d_bz0[ii*ny + jj] = val;
        if (kk == nz-1)   d_bz1[ii*ny + jj] = val;
    }

    double x = gi * dx;
    double y = gj * dy;
    double z = gk * dz;

    err[ii + nx * (jj + ny * kk)] = fabs(val - d_an_sol(x,y,z,t));
}

void it0_kernel_launcher(
    double* u0, double* d_err,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double& err)
{
    dim3 block(32,4,4);
    dim3 grid((nx+block.x-1)/block.x,
              (ny+block.y-1)/block.y,
              (nz+block.z-1)/block.z);

    int n = nx*ny*nz;

    it0_kernel<<<grid, block>>>(
        u0, nx, ny, nz,
        sx, sy, sz,
        dx, dy, dz,
        d_err
    );

    cudaDeviceSynchronize();

    err = *thrust::max_element(
        thrust::device_pointer_cast(d_err),
        thrust::device_pointer_cast(d_err + n)
    );
}

void it1_kernel_launcher(
    double *u0,
    double *u1,
    double *d_bx0, double *d_bx1,
    double *d_by0, double *d_by1,
    double *d_bz0, double *d_bz1,
    double *d_err,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau,
    double &err, bool write_bounds
)
{
    dim3 block(32,4,4);
    dim3 grid((nx+block.x-1)/block.x,
              (ny+block.y-1)/block.y,
              (nz+block.z-1)/block.z);

    int n = nx*ny*nz;

    it1_kernel<<<grid, block>>>(
        u0, u1,
        d_bx0, d_bx1,
        d_by0, d_by1,
        d_bz0, d_bz1,
        nx, ny, nz,
        sx, sy, sz,
        dx, dy, dz,
        a, tau,
        d_err, write_bounds
    );

    cudaDeviceSynchronize();

    err = *thrust::max_element(
        thrust::device_pointer_cast(d_err),
        thrust::device_pointer_cast(d_err + n)
    );
}

void main_kernel_launcher(
    double *u0,
    double *u1,
    double *u2,
    double *d_bx0, double * d_bx1,
    double *d_by0, double *d_by1,
    double *d_bz0, double *d_bz1,
    double *d_err,
    int nx, int ny, int nz,
    int gx, int gy, int gz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau, double t,
    double &err, bool write_bounds
)
{
    dim3 block(32,4,4);
    dim3 grid((nx+block.x-1)/block.x,
              (ny+block.y-1)/block.y,
              (nz+block.z-1)/block.z);

    int n = nx*ny*nz;

    main_kernel<<<grid, block>>>(
        u0, u1, u2,
        d_bx0, d_bx1,
        d_by0, d_by1,
        d_bz0, d_bz1,
        nx, ny, nz,
        gx, gy, gz,
        sx, sy, sz,
        dx, dy, dz,
        a, tau, t,
        d_err, write_bounds
    );

    cudaDeviceSynchronize();

    err = *thrust::max_element(
        thrust::device_pointer_cast(d_err),
        thrust::device_pointer_cast(d_err + n)
    );
}

