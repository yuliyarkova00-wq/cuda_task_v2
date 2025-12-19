#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <chrono>


const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;


double an_sol(double x, double y, double z, double t) {
    const double at = 0.5 * sqrt(4.0 / (Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));
    const double pi = M_PI;
    return sin(2.0 * pi * x / Lx) * sin(pi * y / Ly) * sin(pi * z / Lz) * cos(at * t + 2.0 * pi);
}

double compute_p(double x, double y, double z) {
    return an_sol(x, y, z, 0);
}

double laplacian(double (*func)(double, double, double),
                            double x, double y, double z,
                            double dx, double dy, double dz) {
    double d2f_dx2 = (func(x + dx, y, z) - 2.0 * func(x, y, z) + func(x - dx, y, z)) / (dx * dx);
    double d2f_dy2 = (func(x, y + dy, z) - 2.0 * func(x, y, z) + func(x, y - dy, z)) / (dy * dy);
    double d2f_dz2 = (func(x, y, z + dz) - 2.0 * func(x, y, z) + func(x, y, z - dz)) / (dz * dz);

    return d2f_dx2 + d2f_dy2 + d2f_dz2;
}

double laplace_array(const std::vector<std::vector<std::vector<double> > >& u,
                     int i, int j, int k,
                     double dx, double dy, double dz) {

    int Nx = u.size();
    int Ny = u[0].size();
    int Nz = u[0][0].size();

    
    int ip = i == Nx - 1 ? 0 : i + 1;      
    int im = i == 0 ? Nx - 1 : i - 1;     

    double d2x = (u[ip][j][k] - 2*u[i][j][k] + u[im][j][k]) / (dx*dx);
    double d2y = (u[i][j+1][k] - 2*u[i][j][k] + u[i][j-1][k]) / (dy*dy);
    double d2z = (u[i][j][k+1] - 2*u[i][j][k] + u[i][j][k-1]) / (dz*dz);
    return d2x + d2y + d2z;
}


int main() {
    
    auto t_beg = std::chrono::high_resolution_clock::now();
    
    int Nx = 32;
    int Ny = 32;
    int Nz = 32;

    double dx = Lx / (Nx - 1);
    double dy = Ly / (Ny - 1);
    double dz = Lz / (Nz - 1);

    std::vector<std::vector<std::vector<double> > > u_0(Nx, std::vector<std::vector<double> >(Ny, std::vector<double>(Nz, 0.0)));
    std::vector<std::vector<std::vector<double> > > u_1(Nx, std::vector<std::vector<double> >(Ny, std::vector<double>(Nz, 0.0)));
    std::vector<std::vector<std::vector<double> > > u_2(Nx, std::vector<std::vector<double> >(Ny, std::vector<double>(Nz, 0.0)));

    double a = 1 / (4 * M_PI * M_PI);
    double tau = 0.002;
    int Nt = 50;

    double max_error_0 = 0.0;

    for (int i = 1; i < Nx - 1; ++i) {
        double x = i * dx;
        for (int j = 1; j < Ny - 1; ++j) {
            double y = j * dy;
            for (int k = 1; k < Nz - 1; ++k) {
                double z = k * dz;
                u_0[i][j][k] = compute_p(x, y, z);
                double err = std::abs(u_0[i][j][k] - an_sol(x, y, z, 0));
                if (err > max_error_0) max_error_0 = err;
            }
        }
    }
    std::cout << "Max error at timestep " << 0 << " = " << max_error_0 << "\n";

    double max_error_1 = 0.0;
    for (int i = 1; i < Nx - 1; ++i) {
        double x = i * dx;
        for (int j = 1; j < Ny - 1; ++j) {
            double y = j * dy;
            for (int k = 1; k < Nz - 1; ++k) {
                double z = k * dz;
                u_1[i][j][k] = u_0[i][j][k] + 0.5 * a * tau * tau * laplace_array(u_0, i, j, k, dx, dy, dz);
                double err = std::abs(u_1[i][j][k] - an_sol(x, y, z, tau));
                if (err > max_error_1) max_error_1 = err;
            }
        }
    }

    std::cout << "Max error at timestep " << 1 << " = " << max_error_1 << "\n";

    for (int n = 2; n < Nt; ++n) {
        double max_error = 0.0;
        int im, jm, km;
        double true_val, pred_val;
        for (int i = 1; i < Nx-1; ++i) {
            double x = i * dx;
            for (int j = 1; j < Ny - 1; ++j) {
                double y = j * dy;
                for (int k = 1; k < Nz - 1; ++k) {
                    double z = k * dz;
                    double lap = laplace_array(u_1, i, j, k, dx, dy, dz);
                    u_2[i][j][k] = 2.0 * u_1[i][j][k] - u_0[i][j][k] + a * tau * tau * lap;
                    double t = n * tau;
                    double diff = std::fabs(an_sol(x, y, z, t) - u_2[i][j][k]);
                    if (diff > max_error){
                        max_error = diff;
                        
                    } 
                }
            }
        }

        std::cout << "Max error at timestep " << n << " = " << max_error << "\n";
        std::swap(u_0, u_1);
        std::swap(u_2, u_1);
    }
    double t_end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    std::cout << "Execution time: " << duration.count() / 1000000.0f << " seconds " << "\n";
    return 0;
}

