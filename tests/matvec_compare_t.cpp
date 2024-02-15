#include <functional>
#include <omp.h>
#include <cmath>

#include "../src/include/matrices.hpp"
#include "../src/include/vector.hpp"

using namespace Poisson;
using T = double;

int main ()
{

	int p = 4; int q = p;
	Grid grid(p,q);
	std::cout << "Grid is " << grid.x << " x " << grid.y << std::endl;
	GPUVector<T> v(grid);
	GPUVector<T> x(grid);
	std::function<void(const GPUVector<T>&, GPUVector<T>&, const T&)> A = opPBCGPU<T, GPUVector<T>>;
	//std::function<void(const T*, T*, const T&, const int&, const int&)> A = opPBCGPU2<T>;
	std::function<void(const Vector<T>&, Vector<T>&, const T&)> Acpu = opPBC<T>;
	v.ones();
	Vector<T> xcpu(grid);
	Vector<T> vcpu(grid);
	vcpu.ones();

	T shift = 1e-3;
	int ns = 10000;
	const int N = grid.x*grid.y;
	T startg;
	T endg;
	std::cout << "Before matvec on GPU: ||x||_2 = " << std::sqrt(x.norm2()) << std::endl;
        std::cout << "Before matvec on CPU: ||x||_2 = " << std::sqrt(xcpu.norm2()) << std::endl;
	startg = omp_get_wtime();
	#pragma omp target enter data map(to:v.data[0:N]) map(alloc:x.data[0:N])
	for (int i = 0; i < ns; i++) {
	A(v,x,shift);
	}
	#pragma omp target exit data map(from:x.data[0:N]) map(release:v.data[0:N])
	std::cout << "Norm of solution vector on GPU is: ||x||_2 = " << std::sqrt(x.norm2()) << std::endl;
	endg = omp_get_wtime();
	std::cout << "Time for " << ns << " matrix-vector products on GPU is " << endg - startg << " seconds " << std::endl;

	T start;
	T endt;
	start = omp_get_wtime();
        for (int i = 0; i < ns; i++) {
        Acpu(vcpu,xcpu,shift);
        }
	endt = omp_get_wtime();
	std::cout << "Norm of solution vector on CPU is: ||x||_2 = " << std::sqrt(xcpu.norm2()) << std::endl;
        std::cout << "Time for " << ns << " matrix-vector products on CPU is " << endt - start << " seconds " << std::endl;


	return EXIT_SUCCESS;
}
