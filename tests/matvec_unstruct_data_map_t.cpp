#include <functional>
#include <omp.h>
#include <cmath>

#include "../src/include/matrices.hpp"
#include "../src/include/vector.hpp"

using namespace Poisson;
using T = double;

int main ()
{

	int p = 8; int q = 8;
	Grid grid(p,q);
	std::cout << "Grid is " << grid.x << " x " << grid.y << std::endl;
	GPUVector<T> v(grid);
	GPUVector<T> x(grid);
	std::function<void(const GPUVector<T>&, GPUVector<T>&, const T&)> A = opPBCGPU<T, GPUVector<T>>;
	v.randn();
	x.ones();

	T shift = 1e-3;
	int ns = 10000;
	const int N = grid.x*grid.y-1;
	T start;
	T end;
	start = omp_get_wtime();

	//#pragma omp target enter data map(to: v.data[0:N]) map(from:x.data[0:N])
//#pragma omp target enter data map(alloc: x.data[0:N], v.data[0:N])
	#pragma omp target data map(to:v.data[0:N]) map(tofrom:x.data[0:N])
	for (int i = 0; i < ns; i++) {
	A(v,x,shift);
	}
	//#pragma omp target update from(x.data[0:N])
	end = omp_get_wtime();
	//#pragma omp target exit data map(release: x.data[0:N], v.data[0:N])
//#pragma omp target exit data map(delete: x.data[0:N], v.data[0:N])
	//std::cout << "After loop" << std::endl;
	std::cout << "Norm of solution vector is: ||x||_2 = " << std::sqrt(x.norm2()) << std::endl;
	std::cout << "Time for " << ns << " matrix-vector products is " << end - start << " seconds " << std::endl;
	return EXIT_SUCCESS;
}
