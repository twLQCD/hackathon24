#include <functional>
#include <omp.h>
#include <cmath>

#include "../src/include/matrices.hpp"
#include "../src/include/vector.hpp"
#include "../src/include/smoothers.hpp"

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
	std::function<void(const GPUVector<T>&, GPUVector<T>&)> LU = LUPBCGPU<T, GPUVector<T>>;
	std::function<void(const GPUVector<T>&, GPUVector<T>&, const T&)> DINV = DinvPBCGPU<T, GPUVector<T>>;
	v.randn();
	x.ones();
	std::cout << std::sqrt(v.norm2()) << std::endl;
	setEqualTo<T, GPUVector<T>>(x,v,v.size);
	std::cout << std::sqrt(v.norm2()) << std::endl;
	v.randn();
	std::cout << std::sqrt(v.norm2()) << std::endl;
	v = x;
	std::cout << std::sqrt(v.norm2()) << std::endl;
/*	T shift = 1e-3;
	GPUSmootherPBC<T> s(LU,DINV,A,shift,100,1e-6);

	T start;
	T end;
	const int N = grid.x*grid.y;
	std::cout << "Norm of solution vector before smoothing is: ||x||_2 = " << std::sqrt(x.norm2()) << std::endl;
	start = omp_get_wtime();
#pragma omp target enter data map(to:v.data[0]) map(alloc:x.data[0:N])
		s(v,x);
#pragma omp target exit data map(from: x.data[0:N]) map(release: v.data[0:N])
	end = omp_get_wtime();
	std::cout << "Norm of solution vector after smoothing is: ||x||_2 = " << std::sqrt(x.norm2()) << std::endl; */
	return EXIT_SUCCESS;
}
