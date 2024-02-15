#include <functional>
#include <omp.h>
#include <cmath>

#include "../src/include/matrices.hpp"
#include "../src/include/vector.hpp"

using namespace Poisson;
using T = double;

void prim_ones(T* in, const int N)
{
	for (int i = 0; i < N; i++) in[i] = 1.0;
}

void prim_zeros(T* in, const int N)
{
        for (int i = 0; i < N; i++) in[i] = 0.0;
}

void print(T* in, const int N)
{
	for (int i = 0; i < N; i++) std::cout << in[i] << std::endl;
}

T norm(T* in, const int N)
{
	T out = 0.0;
	for (int i = 0; i < N; i++) out += in[i]*in[i];
	return out;
}

int main ()
{

	int p = 4; int q = 4;
	Grid grid(p,q);
	std::cout << "Grid is " << grid.x << " x " << grid.y << std::endl;
	std::function<void(const T*, T*, const T&, const int&, const int&)> A = opPBCGPU2<T>;
	std::function<void(const Vector<T>&, Vector<T>&, const T&)> Acpu = opPBC<T>;
	Vector<T> xcpu(grid);
	Vector<T> vcpu(grid);
	vcpu.ones();

	T shift = 1e-3;
	int ns = 1;
	const int N = grid.x*grid.y;
	T startg;
	T endg;

	T* datain = new T[N];
	T* dataout = new T[N];
	prim_ones(datain, N);
	prim_zeros(dataout,N);
	startg = omp_get_wtime();
	std::cout << "Before matvec on GPU: ||x||_2 = " << std::sqrt(norm(datain,N)) << std::endl;
	std::cout << "Before matvec on CPU: ||x||_2 = " << std::sqrt(xcpu.norm2()) << std::endl;
	#pragma omp target enter data map(to:datain[0:N]) map(alloc:dataout[0:N])
	for (int i = 0; i < ns; i++) {
	A(datain,dataout,shift,grid.x,grid.y);
	}
	#pragma omp target exit data map(from:dataout[0:N]) map(release:datain[0:N])
	endg = omp_get_wtime();
	std::cout << "Norm of solution vector on GPU is: ||x||_2 = " << std::sqrt(norm(dataout,N)) << std::endl;
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

	delete[] datain;
	delete[] dataout;
	return EXIT_SUCCESS;
}
