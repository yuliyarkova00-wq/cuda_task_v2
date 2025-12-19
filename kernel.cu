#include "kernel.cuh"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <cmath>

#define IDX(i,j,k) (((i)*(ny+2) + (j))*(nz+2) + (k))

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
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > nx || j > ny || k > nz) return;

    int gi = sx + (i-1);
    int gj = sy + (j-1);
    int gk = sz + (k-1);

    double x = gi * dx;
    double y = gj * dy;
    double z = gk * dz;

    int id = IDX(i,j,k);
    double val = d_compute_p(x,y,z);
    u0[id] = val;

    int tid = ((i-1)*ny + (j-1))*nz + (k-1);
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
    double* err
){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > nx || j > ny || k > nz) return;

    int gi = sx + (i-1);
    int gj = sy + (j-1);
    int gk = sz + (k-1);

    double x = gi * dx;
    double y = gj * dy;
    double z = gk * dz;

    double d2x = (d_compute_p(x+dx,y,z) - 2*d_compute_p(x,y,z) + d_compute_p(x-dx,y,z))/(dx*dx);
    double d2y = (d_compute_p(x,y+dy,z) - 2*d_compute_p(x,y,z) + d_compute_p(x,y-dy,z))/(dy*dy);
    double d2z = (d_compute_p(x,y,z+dz) - 2*d_compute_p(x,y,z) + d_compute_p(x,y,z-dz))/(dz*dz);

    int id = IDX(i,j,k);
    double val = u0[id] + 0.5 * a * tau * tau * (d2x+d2y+d2z);
    u1[id] = val;
    if (i == 1) d_bx0[(j-1)*nz + k -1] = val;
if (i == nx) d_bx1[(j-1)*nz + k-1] = val;
if (j == 1) d_by0[(i-1)*nz + k-1] = val;
if (j == ny) d_by1[(i-1)*nz + k-1] = val;
if (k == 1) d_bz0[(i-1)*ny+ j-1] = val;
if (k == nz) d_bz1[(i-1)*ny + j-1] = val;
   
    int tid = ((i-1)*ny + (j-1))*nz + (k-1);
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
    double* err
){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > nx || j > ny || k > nz) return;
    if (i == 1) u1[(0 * (ny+2) + j)*(nz+2) + k] =  d_bx0[(j-1)*nz + k -1];
    if (i == nx) u1[((nx+1) * (ny+2) + j)*(nz+2) + k] =  d_bx1[(j-1)*nz + k -1];
if (j == 1) u1[(i * (ny+2) + 0)*(nz+2) + k] =  d_by0[(i-1)*nz + k -1];
if (j == ny) u1[(i * (ny+2) + ny+1)*(nz+2) + k] =  d_by1[(i-1)*nz + k -1];
if (k == 1) u1[(i * (ny+2) + j)*(nz+2) + 0] =  d_bz0[(i-1)*ny + j -1];
if (k == nz) u1[(i * (ny+2) + 0)*(nz+2) + nz+1] =  d_bz1[(i-1)*ny + j -1];
    int gi = sx + (i-1);
    int gj = sy + (j-1);
    int gk = sz + (k-1);

    int id = IDX(i,j,k);

    if (gi==0 || gi==gx-1 || gj==0 || gj==gy-1 || gk==0 || gk==gz-1) {
        u2[id] = 0.0;
        err[((i-1)*ny + (j-1))*nz + (k-1)] = 0.0;
        return;
    }

    double lap =
        (u1[IDX(i+1,j,k)] - 2*u1[id] + u1[IDX(i-1,j,k)])/(dx*dx) +
        (u1[IDX(i,j+1,k)] - 2*u1[id] + u1[IDX(i,j-1,k)])/(dy*dy) +
        (u1[IDX(i,j,k+1)] - 2*u1[id] + u1[IDX(i,j,k-1)])/(dz*dz);

    double val = 2*u1[id] - u0[id] + a*tau*tau*lap;
    u2[id] = val;
if (i == 1) d_bx0[(j-1)*nz + k-1] = val;
    if (i == nx) d_bx1[(j-1)*nz + k-1] = val;
if (j == 1) d_by0[(i-1)*nz + k-1] = val;
if (j == ny) d_by1[(i-1)*nz + k-1] = val;
if (k == 1) d_bz0[(i-1)*ny + j-1] = val;
if (k == nz) d_bz1[(i-1)*ny + j-1] = val;

    double x = gi * dx;
    double y = gj * dy;
    double z = gk * dz;

    err[((i-1)*ny + (j-1))*nz + (k-1)] =
        fabs(val - d_an_sol(x,y,z,t));
}

void it0_kernel_launcher( 
    double* u0, int nx, int ny, int nz,
    int sx, int sy, int sz, 
    double dx, double dy, double dz, double& err) 
{
    dim3 block(8,8,8);
    dim3 grid((nx-1+block.x)/block.x, (ny-1+block.y)/block.y, (nz-1+block.z)/block.z);
    int num_threads = nx * ny * nz;
    thrust::device_vector<double> err0(num_threads, 0.0);
    it0_kernel<<<grid, block>>>(
        u0, nx, ny, nz,
        sx, sy, sz, 
        dx, dy, dz,
        thrust::raw_pointer_cast(err0.data())
    );
    
    cudaDeviceSynchronize();
    err = *thrust::max_element(err0.begin(), err0.end()); 
}

void it1_kernel_launcher(
    double *u0,
    double *u1,
    double *d_bx0, double *d_bx1,
    double *d_by0, double *d_by1,
    double *d_bz0, double *d_bz1,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau,
    double &err
)
{
    dim3 block(8, 8, 8);
    dim3 grid((nx-1+block.x)/block.x, (ny-1+block.y)/block.y, (nz-1+block.z)/block.z);
    int num_threads = nx * ny * nz;
    thrust::device_vector<double> err1(num_threads, 0.0);

    it1_kernel<<<grid, block>>>(
        u0, u1, d_bx0, d_bx1, d_by0, d_by1, d_bz0, d_bz1, nx, ny, nz,
        sx, sy, sz,
        dx, dy, dz, a, tau,
        thrust::raw_pointer_cast(err1.data())
    );

    cudaDeviceSynchronize();
    err = *thrust::max_element(err1.begin(), err1.end());
}


void main_kernel_launcher(
    double *u0,
    double *u1,
    double *u2,
    double *d_bx0, double * d_bx1, double *d_by0, double *d_by1,
    double *d_bz0, double *d_bz1,
    int nx, int ny, int nz,
    int gx, int gy, int gz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau, double t, 
    double &err
)
{
    dim3 block(8, 8, 8);
    dim3 grid((nx-1+block.x)/block.x, (ny-1+block.y)/block.y, (nz-1+block.z)/block.z);

    int num_threads = nx * ny * nz;
    thrust::device_vector<double> err_main(num_threads, 0.0);

    main_kernel<<<grid, block>>>(
        u0, u1, u2, 
        d_bx0, d_bx1, d_by0, d_by1, 
        d_bz0, d_bz1,
        nx, ny, nz,
        gx, gy, gz,
        sx, sy, sz,
        dx, dy, dz,
        a, tau, t,
        thrust::raw_pointer_cast(err_main.data())
    );

    cudaDeviceSynchronize();

    err = *thrust::max_element(err_main.begin(), err_main.end());
}
