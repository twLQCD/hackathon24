#pragma once

#include "../include/smoothers.hpp"

//template<class T>
//using std::function<void(const Vector<T>&,Vector<T>&)> = Func;
//using Func = std::function;

namespace Poisson {


template<class T>
void jacobi(Vector<T>& in, Vector<T>& out, int iters, std::function<void(const Vector<T>&,Vector<T>&)>& LU, std::function<void(const Vector<T>&,Vector<T>&)>& Dinv)
{
	Vector<T> tmp1(out.grid);
	Vector<T> tmp2(out.grid);
	Vector<T> x0(out.grid);

	for (int i = 0; i < iters; i++) 
	{
		LU(x0,tmp1);
		tmp2 = in - tmp1;
		Dinv(tmp2,out);
		x0 = out;
	}

}
template void jacobi<float>(Vector<float>& in, Vector<float>& out, int iters, std::function<void(const Vector<float>&, Vector<float>&)>& LU, std::function<void(const Vector<float>&,Vector<float>&)>& Dinv);
template void jacobi<double>(Vector<double>& in, Vector<double>& out, int iters, std::function<void(const Vector<double>&, Vector<double>&)>& LU, std::function<void(const Vector<double>&, Vector<double>&)>& Dinv);
} //namespace
