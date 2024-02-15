#include <functional>
#include <omp.h>
#include <cmath>

#include "../src/include/vector.hpp"

using namespace Poisson;
using T = double;

int main ()
{

	int p = 8; int q = 8;
	Grid grid(p,q);
	std::cout << "Grid is " << grid.x << " x " << grid.y << std::endl;
	GPUVector<T> x(grid);
	GPUVector<T> y(grid);
	//Vector<T> xcpu(grid);
	x.ones();
	y.ones();
	T norm = vecnorm3<T>(x.data,y.data,x.size);
	std::cout << norm << std::endl;
	//xcpu.ones();
	//std::cout << "GPU ||x||_2 = " << std::sqrt(x.norm2()) << std::endl; 
	//std::cout << "CPU ||x||_2 = " << std::sqrt(xcpu.norm2()) << std::endl;


	return EXIT_SUCCESS;
}
