#pragma once

#include <functional>
#include <cmath>
#include <cassert>
#include <omp.h>

#include "vector.hpp"
#include "matrices.hpp"

//using Func = std::function;

namespace Poisson {

template<class T>
class Smoother {
        public:
                int iters;
		T target;
                std::function<void(const Vector<T>&,Vector<T>&)> lu;
                std::function<void(const Vector<T>&,Vector<T>&)> dinv;
		std::function<void(const Vector<T>&,Vector<T>&)> A;

                //default constructor
                Smoother<T>(){};

                //parameterized constructor
                Smoother<T>(std::function<void(const Vector<T>&,Vector<T>&)>& lu, std::function<void(const Vector<T>&,Vector<T>&)>& dinv, std::function<void(const Vector<T>&,Vector<T>&)>& A, int iters, T target) :
                        lu(lu),
                        dinv(dinv),
			A(A),
                        iters(iters),
			target(target)
                {}

                void operator()(const Vector<T>& in, Vector<T>& out)
                {
                        //temporaries. Maybe a better way?
			assert(in.size == out.size);
                        Vector<T> tmp1(out.grid);
                        Vector<T> tmp2(out.grid);
			Vector<T> tmp3(out.grid);
			Vector<T> tmp4(out.grid);
			Vector<T> tmp5(out.grid);
                        Vector<T> x0(out.grid);
			Vector<T> bnew(out.grid);
			Vector<T> bin(out.grid);
			Vector<T> xin(out.grid);
			Vector<T> r(out.grid);
			xin = out;
			bin = in;
			//compute the initial residual. If out is zeros, it will be the rhs
			A(xin,tmp1);
			bnew = bin - tmp1;
			T rn = std::sqrt(bin.norm2());
			int i = 0;
			while (i <= iters && rn >= target)
                        {
                                lu(x0,tmp2);
                                tmp3 = bnew - tmp2;
                                dinv(tmp3,tmp4);
                                x0 = tmp4;

				xin = x0 + out;
				A(xin,tmp5);
				r = bin - tmp5;
				rn = std::sqrt(r.norm2());
				i++;
                        }
			out = xin;

                }

                ~Smoother<T>(){}

};

template<class T>
class SmootherPBC {
        public:
                int iters;
                T target;
		T eps;
                std::function<void(const Vector<T>&,Vector<T>&)> lu;
                std::function<void(const Vector<T>&,Vector<T>&, const T&)> dinv;
                std::function<void(const Vector<T>&,Vector<T>&,const T&)> A;

                //default constructor
                SmootherPBC<T>(){};

                //parameterized constructor
                SmootherPBC<T>(std::function<void(const Vector<T>&,Vector<T>&)>& lu, std::function<void(const Vector<T>&,Vector<T>&,const T&)>& dinv, std::function<void(const Vector<T>&,Vector<T>&,const T&)>& A, const T& eps, int iters, T target) :
                        lu(lu),
                        dinv(dinv),
                        A(A),
			eps(eps),
                        iters(iters),
                        target(target)
                {}

                void operator()(const Vector<T>& in, Vector<T>& out)
                {
                        //temporaries. Maybe a better way?
			assert(in.size == out.size);
                        Vector<T> tmp1(out.grid);
                        Vector<T> tmp2(out.grid);
                        Vector<T> tmp3(out.grid);
                        Vector<T> tmp4(out.grid);
                        Vector<T> tmp5(out.grid);
                        Vector<T> x0(out.grid);
                        Vector<T> bnew(out.grid);
                        Vector<T> bin(out.grid);
                        Vector<T> xin(out.grid);
                        Vector<T> r(out.grid);
                        xin = out;
                        bin = in;
                        //compute the initial residual. If out is zeros, it will be the rhs
                        A(xin,tmp1,eps);
                        bnew = bin - tmp1;
                        T rn = std::sqrt(bin.norm2());
                        //std::cout << "Initial residual norm is : " << bnew.norm2() << std::endl;
                        //std::cout << "Norm of rhs in Jacobi is : " << xin.norm2() << std::endl;
                        int i = 0;
                        while (i <= iters && rn >= target)
                        {
                                lu(x0,tmp2);
                                tmp3 = bnew - tmp2;
                                dinv(tmp3,tmp4,eps);
                                x0 = tmp4;

                                xin = x0 + out;
                                A(xin,tmp5,eps);
                                r = bin - tmp5;
                                rn = std::sqrt(r.norm2());
                                std::cout << "Jacobi Iteration: " << i << ", ||r||_2 = " << rn << std::endl;
                                i++;
                        }
                        out = xin;

                }

                ~SmootherPBC<T>(){}

};

template<class T>
class CoarseSmoother {
        public:
                int iters;
                T target;
		std::vector<T> weights;
                std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>& weights)> lu;
                std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>& weights)> dinv;
                std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>& weights)> A;

                //default constructor
                CoarseSmoother<T>(){};

                //parameterized constructor
                CoarseSmoother<T>(std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& lu, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& dinv, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& A, std::vector<T>& weights, int iters, T target) :
                        lu(lu),
                        dinv(dinv),
                        A(A),
			weights(weights),
                        iters(iters),
                        target(target)
                {}

		//for use with default constructor
		void create(std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& lu_t, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& dinv_t, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& A_t, std::vector<T>& weights_t, int iters_t, T target_t)
		{

			lu = lu_t;
			dinv = dinv_t;
			A = A_t;
			weights = weights_t;
			iters = iters_t;
			target = target_t;

		}

                void operator()(const Vector<T>& in, Vector<T>& out)
                {
                        //temporaries. Maybe a better way?
			assert(in.size == out.size);
                        Vector<T> tmp1(out.grid);
                        Vector<T> tmp2(out.grid);
                        Vector<T> tmp3(out.grid);
                        Vector<T> tmp4(out.grid);
                        Vector<T> tmp5(out.grid);
                        Vector<T> x0(out.grid);
                        Vector<T> bnew(out.grid);
                        Vector<T> bin(out.grid);
                        Vector<T> xin(out.grid);
                        Vector<T> r(out.grid);
                        xin = out;
                        bin = in;
                        //compute the initial residual. If out is zeros, it will be the rhs
                        A(xin,tmp1,weights);
                        bnew = bin - tmp1;
                        T rn = std::sqrt(bin.norm2());
                        //std::cout << "Initial residual norm is : " << bnew.norm2() << std::endl;
                        //std::cout << "Norm of rhs in Jacobi is : " << xin.norm2() << std::endl;
                        int i = 0;
                        while (i <= iters && rn >= target)
                        {
                                lu(x0,tmp2,weights);
                                tmp3 = bnew - tmp2;
                                dinv(tmp3,tmp4,weights);
                                x0 = tmp4;

                                xin = x0 + out;
                                A(xin,tmp5,weights);
                                r = bin - tmp5;
                                rn = std::sqrt(r.norm2());
				//std::cout << "Coarse Jacobi: iter = " << i << " with residual norm " << rn << std::endl;
                                i++;
                        }
                        out = xin;

                }

                ~CoarseSmoother<T>(){}

};

template<class T>
class CoarseSmootherPBC {
        public:
                int iters;
                T target;
                std::vector<T> weights;
                std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>& weights)> lu;
                std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>& weights)> dinv;
                std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>& weights)> A;

                //default constructor
                CoarseSmootherPBC<T>(){};

                //parameterized constructor
                CoarseSmootherPBC<T>(std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& lu, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& dinv, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& A, std::vector<T>& weights, int iters, T target) :
                        lu(lu),
                        dinv(dinv),
                        A(A),
                        weights(weights),
                        iters(iters),
                        target(target)
                {}

                //for use with default constructor
                void create(std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& lu_t, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& dinv_t, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& A_t, std::vector<T>& weights_t, int iters_t, T target_t)
                {

                        lu = lu_t;
                        dinv = dinv_t;
                        A = A_t;
                        weights = weights_t;
                        iters = iters_t;
                        target = target_t;

                }

                void operator()(const Vector<T>& in, Vector<T>& out)
                {
                        //temporaries. Maybe a better way?
			assert(in.size == out.size);
                        Vector<T> tmp1(out.grid);
                        Vector<T> tmp2(out.grid);
                        Vector<T> tmp3(out.grid);
                        Vector<T> tmp4(out.grid);
                        Vector<T> tmp5(out.grid);
                        Vector<T> x0(out.grid);
                        Vector<T> bnew(out.grid);
                        Vector<T> bin(out.grid);
                        Vector<T> xin(out.grid);
                        Vector<T> r(out.grid);
                        xin = out;
                        bin = in;
                        //compute the initial residual. If out is zeros, it will be the rhs
                        A(xin,tmp1,weights);
                        bnew = bin - tmp1;
                        T rn = std::sqrt(bin.norm2());
                        //std::cout << "Initial residual norm is : " << bnew.norm2() << std::endl;
                        //std::cout << "Norm of rhs in Jacobi is : " << xin.norm2() << std::endl;
                        int i = 0;
                        while (i <= iters && rn >= target)
                        {
                                lu(x0,tmp2,weights);
                                tmp3 = bnew - tmp2;
                                dinv(tmp3,tmp4,weights);
                                x0 = tmp4;

                                xin = x0 + out;
                                A(xin,tmp5,weights);
                                r = bin - tmp5;
                                rn = std::sqrt(r.norm2());
                                std::cout << "Coarse Jacobi: iter = " << i << " with residual norm " << rn << std::endl;
                                i++;
                        }
                        out = xin;

                }

                ~CoarseSmootherPBC<T>(){}

};

template<class T>
class GPUSmootherPBC {
        public:
                int iters;
                T target;
                T eps;
                std::function<void(const GPUVector<T>&,GPUVector<T>&)> lu;
                std::function<void(const GPUVector<T>&,GPUVector<T>&, const T&)> dinv;
                std::function<void(const GPUVector<T>&,GPUVector<T>&,const T&)> A;

                //default constructor
                GPUSmootherPBC<T>(){};

                //parameterized constructor
                GPUSmootherPBC<T>(std::function<void(const GPUVector<T>&,GPUVector<T>&)>& lu, std::function<void(const GPUVector<T>&,GPUVector<T>&,const T&)>& dinv, std::function<void(const GPUVector<T>&,GPUVector<T>&,const T&)>& A, const T& eps, int iters, T target) :
                        lu(lu),
                        dinv(dinv),
                        A(A),
                        eps(eps),
                        iters(iters),
                        target(target)
                {}

                void operator()(const GPUVector<T>& in, GPUVector<T>& out)
                {
			assert(in.size == out.size);
			const int N = out.size;
                        //temporaries. Maybe a better way?
                        GPUVector<T> tmp1(out.grid);
                        GPUVector<T> tmp2(out.grid);
                        //GPUVector<T> tmp3(out.grid);
                        //GPUVector<T> tmp4(out.grid);
                        //GPUVector<T> tmp5(out.grid);
                        GPUVector<T> x0(out.grid);
                        GPUVector<T> bnew(out.grid);
                        GPUVector<T> bin(out.grid);
                        GPUVector<T> xin(out.grid);
                        GPUVector<T> r(out.grid);
			std::cout << "In the smoother" << std::endl;
//#pragma omp target enter data map(alloc: x0.data[0:N], xin.data[0:N], bin.data[0:N], tmp1.data[0:N], tmp2.data[0:N], bnew.data[0:N], r.data[0:N])
			#pragma omp target enter data map(to: xin.data[0:N], bin.data[0:N]) map(to: tmp1.data[0:N])
                        xin = out;
			bin = in;
			A(xin,tmp1,eps);
			out = tmp1; //FOR TESTING ONLY
			#pragma omp target exit data map(release: xin.data[0:N], bin.data[0:N], tmp1.data[0:N])
			std::cout << "Norm of xin is: " << std::sqrt(xin.norm2()) << std::endl;
			std::cout << "Norm of bin is: " << std::sqrt(bin.norm2()) << std::endl;
			std::cout << "Norm of tmp1 is: " << std::sqrt(tmp1.norm2()) << std::endl;
			//compute the initial residual. If out is zeros, it will be the rhs
			/*A(xin,tmp1,eps);
			bnew = bin - tmp1;
                        T rn = std::sqrt(bin.norm2());
                        int i = 0;
                        while (i <= iters && rn >= target)
                        {
                                //lu(x0,tmp2);
				lu(x0,tmp1);
                                //tmp3 = bnew - tmp2;
				tmp2 = bnew - tmp1;
                                //dinv(tmp3,tmp4,eps);
				dinv(tmp2,x0,eps);
                                //x0 = tmp4;

                                xin = x0 + out;
                                //A(xin,tmp5,eps);
				A(xin,tmp1,eps);
                                //r = bin - tmp5;
				r = bin - tmp1;
                                rn = std::sqrt(r.norm2());
                                std::cout << "Jacobi Iteration: " << i << ", ||r||_2 = " << rn << std::endl;
                                i++;
                        }
                        out = xin;
#pragma omp target exit data map(release: x0.data[0:N], xin.data[0:N], bin.data[0:N], tmp1.data[0:N], tmp2.data[0:N], bnew.data[0:N], r.data[0:N])			

		*/
                } 

                ~GPUSmootherPBC<T>(){}

};

} //namespace
