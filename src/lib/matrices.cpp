//#pragma once


#include "../include/matrices.hpp"

namespace Poisson {

int mod(int i, int dim)
{
	int a;
	return a = (i%dim == -1) ? dim - 1 : i%dim;
}

//there must be a better way to do this!
template<class T>
std::vector<T> calc_weights(Grid& grid, std::function<void(const Vector<T>&,Vector<T>&)>& op, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R)
{
	Vector<T> x(grid);
	std::vector<int> nodes = {0,1,x.grid.x+1};
	std::vector<T> weights(nodes.size());
	x.unit(0);

	Vector<T> tmp1(x.grid);
	P(x,tmp1);
	op(tmp1,x);
	R(x,tmp1);
	for (int i = 0; i < nodes.size(); i++)
	{
		weights[i] = tmp1(nodes[i]);
	}

	return weights;
}
template std::vector<double> calc_weights<double>(Grid& grid, std::function<void(const Vector<double>&,Vector<double>&)>& op, std::function<void(const Vector<double>&,Vector<double>&)>& P,  std::function<void(const Vector<double>&,Vector<double>&)>& R);

template std::vector<float> calc_weights<float>(Grid& grid, std::function<void(const Vector<float>&,Vector<float>&)>& op, std::function<void(const Vector<float>&,Vector<float>&)>& P, std::function<void(const Vector<float>&,Vector<float>&)>& R);

template<class T>
std::vector<T> calc_weightsPBC(Grid& grid, std::function<void(const Vector<T>&,Vector<T>&,const T&)>& op, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R, const T& eps)
{
        Vector<T> x(grid);
        std::vector<int> nodes = {0,1,x.grid.x+1};
        std::vector<T> weights(nodes.size());
        x.unit(0);

        Vector<T> tmp1(x.grid);
        P(x,tmp1);
        opPBC(tmp1,x,eps);
        R(x,tmp1);
        for (int i = 0; i < nodes.size(); i++)
        {
                weights[i] = tmp1(nodes[i]);
        }

        return weights;
}
template std::vector<double> calc_weightsPBC<double>(Grid& grid, std::function<void(const Vector<double>&,Vector<double>&,const double&)>& op, std::function<void(const Vector<double>&,Vector<double>&)>& P,  std::function<void(const Vector<double>&,Vector<double>&)>& R, const double& eps);

template std::vector<float> calc_weightsPBC<float>(Grid& grid, std::function<void(const Vector<float>&,Vector<float>&,const float&)>& op, std::function<void(const Vector<float>&,Vector<float>&)>& P, std::function<void(const Vector<float>&,Vector<float>&)>& R, const float& eps);


template<class T>
std::vector<T> calc_coarse_weights(Grid& grid, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& op, std::vector<T>& prev_coeffs, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R)
{
        Vector<T> x(grid);
        std::vector<int> nodes = {0,1,x.grid.x+1};
        std::vector<T> weights(nodes.size());
        x.unit(0);

        Vector<T> tmp1(x.grid);
        P(x,tmp1);
        op(tmp1,x,prev_coeffs);
        R(x,tmp1);
        for (int i = 0; i < nodes.size(); i++)
        {
                weights[i] = tmp1(nodes[i]);
        }

        return weights;
}
template std::vector<double> calc_coarse_weights<double>(Grid& grid, std::function<void(const Vector<double>&,Vector<double>&,std::vector<double>&)>& op, std::vector<double>& prev_coeffs, std::function<void(const Vector<double>&,Vector<double>&)>& P,  std::function<void(const Vector<double>&,Vector<double>&)>& R);

template std::vector<float> calc_coarse_weights<float>(Grid& grid, std::function<void(const Vector<float>&,Vector<float>&,std::vector<float>&)>& op, std::vector<float>& prev_coeffs, std::function<void(const Vector<float>&,Vector<float>&)>& P, std::function<void(const Vector<float>&,Vector<float>&)>& R);

//the coarse matvec
template<class T>
void coarse_op(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights)
{
	assert(in.size == out.size);
	int i; int j;

	//the origin
	out.data[0] = weights[0]*in(0) + weights[1]*in(1) + weights[1]*in(in.grid.y) + weights[2]*in(in.grid.y+1);
	//corner opposite the origin
	out.data[out.grid.x*out.grid.y-1] = weights[0]*in(in.grid.x * in.grid.y - 1) + weights[1]*in( in.grid.x * in.grid.y - 2 ) +
						weights[1]*in(in.grid.x * in.grid.y - in.grid.x - 1) +
						weights[2]*in(in.grid.x * in.grid.y - in.grid.x - 2);
	//corner adjacent to origin
	out.data[out.grid.x-1] = weights[0]*in(in.grid.x-1) + weights[1]*in(in.grid.x-2) + weights[1]*in(in.grid.x + in.grid.y - 1) +
					weights[2]*in(in.grid.x + in.grid.y - 2);
	//corner above origin
	out.data[out.grid.x*out.grid.y - out.grid.x] = weights[0]*in(in.grid.x*in.grid.y - in.grid.x) + 
							weights[1]*in( (in.grid.x-1)*in.grid.y - in.grid.x ) + 
							weights[1]*in(in.grid.x*in.grid.y - in.grid.x + 1) + 
							weights[2]*in( (in.grid.x-1)*in.grid.y - in.grid.x + 1);
	//the bottom/top edge
#pragma omp parallel for
	for (i = 1; i < out.grid.x - 1; i++) {

		out.data[i] = weights[1]*in(i-1) + weights[1]*in(i+1) + weights[1]*in(i + in.grid.x) +  weights[0]*in(i) + 
				weights[2]*in(i + in.grid.x - 1) + weights[2]*in(i + in.grid.x + 1);
		out.data[i + (out.grid.y-1)*out.grid.x] = weights[1]*in(i-1 + (in.grid.y-1)*in.grid.x) + weights[1]*in(i+1 + (in.grid.y-1)*in.grid.x) + weights[1]*in(i + (in.grid.y-2)*in.grid.x) + weights[0]*in(i + (in.grid.y-1)*in.grid.x) + 
			weights[2]*in(i-1 + (in.grid.y-2)*in.grid.x) + weights[2]*in(i+1 + (in.grid.y-2)*in.grid.x);

	}

	//the left/right edge
#pragma omp parallel for
	for (i = 1; i < out.grid.y - 1; i++) {

		 out.data[(i+1)*out.grid.x-1] = weights[1]*in(i*in.grid.x - 1) + weights[1]*in((i+1)*in.grid.x - 2) + weights[1]*in((i+2)*in.grid.x - 1) + weights[0]*in((i+1)*in.grid.x - 1) + weights[2]*in(i*in.grid.x - 2) + weights[2]*in((i+2)*in.grid.x - 2);
		 
		out.data[(i-1)*out.grid.x + out.grid.x] = weights[1]*in(i*in.grid.x + in.grid.x) + weights[1]*in( (i-2)*in.grid.x + in.grid.x) + weights[1]*in( (i-1)*in.grid.x + in.grid.x + 1) + weights[0]*in((i-1)*in.grid.x + in.grid.x) + weights[2]*in( (i-2)*in.grid.x + in.grid.x + 1) + weights[2]*in(i*in.grid.x + in.grid.x + 1);
	}

	//now the interior points
#pragma omp parallel for collapse(2)
	        for (j = 1; j < in.grid.y - 1; j++){
			for (i = 1; i < in.grid.x - 1; i++) {
				out.data[i + j*out.grid.x] = weights[1]*in(i + (j-1)*in.grid.x) + weights[1]*in( (i-1) + j*in.grid.x) + weights[0]*in(i + j*in.grid.x) + weights[1]*in( (i+1) + j*in.grid.x ) + weights[1]*in(i + (j+1)*in.grid.x) +
					weights[2]*in(i + (j+1)*in.grid.x + 1) + weights[2]*in(i + (j+1)*in.grid.x - 1) +
					weights[2]*in(i + (j-1)*in.grid.x - 1) +  weights[2]*in(i + (j-1)*in.grid.x + 1);
			}

		}

	
}
template void coarse_op<float>(const Vector<float>& in, Vector<float>& out, std::vector<float>& weights);
template void coarse_op<double>(const Vector<double>& in, Vector<double>& out, std::vector<double>& weights);

//the matvec: applies A*x
template<class T>
void op(const Vector<T>& in, Vector<T>& out)
{
	assert(in.size == out.size);
	int i; int j;
	T four = static_cast<T>(4.0);
	T mone = static_cast<T>(-1.0);

	//the origin
	out.data[0] = mone*(in(1) + in(in.grid.y) - four*in(0));
	//corner opposite the origin
	out.data[out.grid.x*out.grid.y-1] = mone*( in(in.grid.x * in.grid.y - in.grid.x - 1) + in( in.grid.x * in.grid.y - 2 ) - four*in(in.grid.x * in.grid.y - 1));
	//corner adjacent to origin
	out.data[out.grid.x-1] = mone*(in(in.grid.x-2) + in(in.grid.x + in.grid.y - 1) - four*in(in.grid.x-1));
	//corner above the origin
	out.data[out.grid.x*out.grid.y - out.grid.x] = mone*(in( (in.grid.x-1)*in.grid.y - in.grid.x ) + in(in.grid.x*in.grid.y - in.grid.x + 1) -four*in(in.grid.x*in.grid.y - in.grid.x));
	//the bottom/top edge
#pragma omp parallel for
	for (i = 1; i < in.grid.x-1; i++)
	{
	out.data[i] = mone*(in(i-1) + in(i+1) + in(i + in.grid.x) - four*in(i));
	out.data[i + (out.grid.y-1)*out.grid.x] = mone*(in(i-1 + (in.grid.y-1)*in.grid.x) + in(i+1 + (in.grid.y-1)*in.grid.x) + in(i + (in.grid.y-2)*in.grid.x) - four*in(i + (in.grid.y-1)*in.grid.x));
	}

	//the right/left edge
#pragma omp parallel for
	for (i = 1; i < in.grid.y-1; i++)
	{
	out.data[(i+1)*out.grid.x-1] = mone*(in(i*in.grid.x - 1) + in((i+1)*in.grid.x-2) + in((i+2)*in.grid.x-1) - four*in((i+1)*in.grid.x-1));
	out.data[(i-1)*out.grid.x + out.grid.x] = mone*(in(i*in.grid.x + in.grid.x) + in( (i-2)*in.grid.x + in.grid.x) + in( (i-1)*in.grid.x + in.grid.x + 1) - four*in((i-1)*in.grid.x + in.grid.x));
	}

	//now the interior nodes
#pragma omp parallel for collapse(2)
	for (j = 1; j < in.grid.y - 1; j++){
		for (i = 1; i < in.grid.x - 1; i++) {
			out.data[i + j*out.grid.x] = (mone)*(in(i + (j-1)*in.grid.x) + in( (i-1) + j*in.grid.x) - four*in(i + j*in.grid.x) + in( (i+1) + j*in.grid.x ) + in(i + (j+1)*in.grid.x));
		}

	}
}
template void op<float>(const Vector<float>& in, Vector<float>& out);
template void op<double>(const Vector<double>& in, Vector<double>& out);

template<class T>
void opPBC(const Vector<T>& in, Vector<T>& out, const T& eps)
{
	assert(in.size == out.size);
        int i; int j;
        T four = static_cast<T>(4.0) + eps;
        T mone = static_cast<T>(-1.0);
#pragma omp parallel for collapse(2)
        for (j = 0; j < in.grid.y; j++){
                for (i = 0; i < in.grid.x; i++) {
                        out.data[mod(i,out.grid.x) + mod(j,out.grid.y)*out.grid.x] = (mone)*(in( mod(i,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x) + in( mod(i-1,in.grid.x) + mod(j,in.grid.y)*in.grid.x) - four*in( mod(i,in.grid.x) + mod(j,in.grid.y)*in.grid.x) + in( mod(i+1,in.grid.x) + mod(j,in.grid.y)*in.grid.x ) + in( mod(i,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x));
                }

        }
}
template void opPBC<float>(const Vector<float>& in, Vector<float>& out, const float& eps);
template void opPBC<double>(const Vector<double>& in, Vector<double>& out, const double& eps);


template<class T, class V>
void opPBCGPU(const V& in, V& out, const T& eps)
{
        assert(in.size == out.size);
        int i; int j;
	const int x = in.grid.x;
	const int y = in.grid.y;
        T four = static_cast<T>(4.0) + eps;
        T mone = static_cast<T>(-1.0);
	#pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j < y; j++){
                for (i = 0; i < x; i++) {
                        out.data[mod(i,x) + mod(j,y)*x] = (mone)*(in( mod(i,x) + mod(j-1,y)*x) + in( mod(i-1,x) + mod(j,y)*x) - four*in( mod(i,x) + mod(j,y)*x) + in( mod(i+1,x) + mod(j,y)*x ) + in( mod(i,x) + mod(j+1,y)*x));
                }

        }
}
template void opPBCGPU<float, Vector<float>>(const Vector<float>& in, Vector<float>& out, const float& eps);
template void opPBCGPU<double, Vector<double>>(const Vector<double>& in, Vector<double>& out, const double& eps);
template void opPBCGPU<float, GPUVector<float>>(const GPUVector<float>& in, GPUVector<float>& out, const float& eps);
template void opPBCGPU<double, GPUVector<double>>(const GPUVector<double>& in, GPUVector<double>& out, const double& eps);

template<class T>
void opPBCGPU2(const T* in, T* out, const T& eps, const int& x, const int& y)
{
        int i; int j;
        const int N = x*y-1;
        T four = static_cast<T>(4.0) + eps;
        T mone = static_cast<T>(-1.0);
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j < y; j++){
                for (i = 0; i < x; i++) {
                        out[mod(i,x) + mod(j,y)*x] = (mone)*(in[ mod(i,x) + mod(j-1,y)*x] + in[ mod(i-1,x) + mod(j,y)*x] - four*in[ mod(i,x) + mod(j,y)*x] + in[ mod(i+1,x) + mod(j,y)*x ] + in[ mod(i,x) + mod(j+1,y)*x]);
                }

        }
}
template void opPBCGPU2<float>(const float* in, float* out, const float& eps, const int& x, const int& y);
template void opPBCGPU2<double>(const double* in, double* out, const double& eps, const int& x, const int& y);

template<class T>
void LUPBC(const Vector<T>& in, Vector<T>& out)
{
        assert(in.size == out.size);
        int i; int j;
        T mone = static_cast<T>(-1.0);
#pragma omp parallel for collapse(2)
        for (j = 0; j < in.grid.y; j++){
                for (i = 0; i < in.grid.x; i++) {
                        out.data[mod(i,out.grid.x) + mod(j,out.grid.y)*out.grid.x] = (mone)*(in( mod(i,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x) + in( mod(i-1,in.grid.x) + mod(j,in.grid.y)*in.grid.x) + in( mod(i+1,in.grid.x) + mod(j,in.grid.y)*in.grid.x ) + in( mod(i,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x));
                }

        }
}
template void LUPBC<float>(const Vector<float>& in, Vector<float>& out);
template void LUPBC<double>(const Vector<double>& in, Vector<double>& out);

template<class T, class V>
void LUPBCGPU(const V& in, V& out)
{
        assert(in.size == out.size);
	const int x = in.grid.x;
	const int y = in.grid.y;
        int i; int j;
        T mone = static_cast<T>(-1.0);
	#pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j < y; j++){
                for (i = 0; i < x; i++) {
                        out.data[mod(i,x) + mod(j,y)*x] = (mone)*(in( mod(i,x) + mod(j-1,in.grid.y)*x) + in( mod(i-1,x) + mod(j,y)*x) + in( mod(i+1,x) + mod(j,y)*x ) + in( mod(i,x) + mod(j+1,y)*x));
                }

        }
}
template void LUPBCGPU<float, Vector<float>>(const Vector<float>& in, Vector<float>& out);
template void LUPBCGPU<double, Vector<double>>(const Vector<double>& in, Vector<double>& out);
template void LUPBCGPU<float, GPUVector<float>>(const GPUVector<float>& in, GPUVector<float>& out);
template void LUPBCGPU<double, GPUVector<double>>(const GPUVector<double>& in, GPUVector<double>& out);


template<class T>
void coarse_opPBC(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights)
{
        assert(in.size == out.size);
        int i; int j;
#pragma omp parallel for collapse(2)
                for (j = 0; j < in.grid.y; j++){
                        for (i = 0; i < in.grid.x; i++) {
                                out.data[mod(i,out.grid.x) + mod(j,out.grid.y)*out.grid.x] = weights[1]*in( mod(i,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x) + 
							weights[1]*in( mod(i-1,in.grid.x) + mod(j,in.grid.y)*in.grid.x) + 
							weights[0]*in( mod(i,in.grid.x) + mod(j,in.grid.y)*in.grid.x) + 
							weights[1]*in( mod(i+1,in.grid.x) + mod(j,in.grid.y)*in.grid.x ) + 
							weights[1]*in(mod(i,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x) +
                                        		weights[2]*in( mod(i-1,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x) + 
							weights[2]*in( mod(i+1,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x) +
                                        		weights[2]*in( mod(i-1,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x) +  
							weights[2]*in(mod(i+1,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x);
                        }

                }
}
template void coarse_opPBC<float>(const Vector<float>& in, Vector<float>& out, std::vector<float>& weights);
template void coarse_opPBC<double>(const Vector<double>& in, Vector<double>& out, std::vector<double>& weights);


template<class T>
void coarse_LUPBC(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights)
{
        assert(in.size == out.size);
        int i; int j;
#pragma omp parallel for collapse(2)
                for (j = 0; j < in.grid.y; j++){
                        for (i = 0; i < in.grid.x; i++) {
                                out.data[mod(i,out.grid.x) + mod(j,out.grid.y)*out.grid.x] = weights[1]*in( mod(i,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x) +
                                                        weights[1]*in( mod(i-1,in.grid.x) + mod(j,in.grid.y)*in.grid.x) +
                                                        weights[1]*in( mod(i+1,in.grid.x) + mod(j,in.grid.y)*in.grid.x ) +
                                                        weights[1]*in(mod(i,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x) +
                                                        weights[2]*in( mod(i-1,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x) +
                                                        weights[2]*in( mod(i+1,in.grid.x) + mod(j+1,in.grid.y)*in.grid.x) +
                                                        weights[2]*in( mod(i-1,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x) +
                                                        weights[2]*in(mod(i+1,in.grid.x) + mod(j-1,in.grid.y)*in.grid.x);
                        }

                }
}
template void coarse_LUPBC<float>(const Vector<float>& in, Vector<float>& out, std::vector<float>& weights);
template void coarse_LUPBC<double>(const Vector<double>& in, Vector<double>& out, std::vector<double>& weights);

//applies the lower plus upper part of the matrix A
template<class T>
void LU(const Vector<T>& in, Vector<T>& out)
{
        assert(in.size == out.size);
        int i; int j;
        T mone = static_cast<T>(-1.0);

        //the origin
        out.data[0] = mone*(in(1) + in(in.grid.y) );
        //corner opposite the origin
        out.data[out.grid.x*out.grid.y-1] = mone*( in(in.grid.x * in.grid.y - in.grid.x - 1) + in( in.grid.x * in.grid.y - 2 ) );
        //corner adjacent to origin
        out.data[out.grid.x-1] = mone*(in(in.grid.x-2) + in(in.grid.x + in.grid.y - 1));
        //corner above the origin
        out.data[out.grid.x*out.grid.y - out.grid.x] = mone*(in( (in.grid.x-1)*in.grid.y - in.grid.x ) + in(in.grid.x*in.grid.y - in.grid.x + 1) );
        //the bottom/top edge
#pragma omp parallel for
        for (i = 1; i < in.grid.x-1; i++)
        {
        out.data[i] = mone*(in(i-1) + in(i+1) + in(i + in.grid.x));
        out.data[i + (out.grid.y-1)*out.grid.x] = mone*(in(i-1 + (in.grid.y-1)*in.grid.x) + in(i+1 + (in.grid.y-1)*in.grid.x) + in(i + (in.grid.y-2)*in.grid.x) );
        }

        //the right/left edge
#pragma omp parallel for
        for (i = 1; i < in.grid.y-1; i++)
        {
        out.data[(i+1)*out.grid.x-1] = mone*(in(i*in.grid.x - 1) + in((i+1)*in.grid.x-2) + in((i+2)*in.grid.x-1) );
        out.data[(i-1)*out.grid.x + out.grid.x] = mone*(in(i*in.grid.x + in.grid.x) + in( (i-2)*in.grid.x + in.grid.x) + in( (i-1)*in.grid.x + in.grid.x + 1) );
        }

        //now the interior nodes
#pragma omp parallel for collapse(2)
        for (j = 1; j < in.grid.y - 1; j++){
                for (i = 1; i < in.grid.x - 1; i++) {
                        out.data[i + j*out.grid.x] = (mone)*(in(i + (j-1)*in.grid.x) + in( (i-1) + j*in.grid.x) + in( (i+1) + j*in.grid.x ) + in(i + (j+1)*in.grid.x));
                }

        }
}

template void LU<float>(const Vector<float>& in, Vector<float>& out);
template void LU<double>(const Vector<double>& in, Vector<double>& out);

//applies (L+U) to the vector in for a coarse grid matrix
template<class T>
void coarse_LU(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights)
{
        assert(in.size == out.size);
        int i; int j;

        //the origin
        out.data[0] = weights[1]*in(1) + weights[1]*in(in.grid.y) + weights[2]*in(in.grid.y+1);
        //corner opposite the origin
        out.data[out.grid.x*out.grid.y-1] = weights[1]*in( in.grid.x * in.grid.y - 2 ) +
                                                weights[1]*in(in.grid.x * in.grid.y - in.grid.x - 1) +
                                                weights[2]*in(in.grid.x * in.grid.y - in.grid.x - 2);
        //corner adjacent to origin
        out.data[out.grid.x-1] =  weights[1]*in(in.grid.x-2) + weights[1]*in(in.grid.x + in.grid.y - 1) +
                                        weights[2]*in(in.grid.x + in.grid.y - 2);
        //corner above origin
        out.data[out.grid.x*out.grid.y - out.grid.x] =  weights[1]*in( (in.grid.x-1)*in.grid.y - in.grid.x ) +
                                                        weights[1]*in(in.grid.x*in.grid.y - in.grid.x + 1) +
                                                        weights[2]*in( (in.grid.x-1)*in.grid.y - in.grid.x + 1);
        //the bottom/top edge
#pragma omp parallel for
        for (i = 1; i < out.grid.x - 1; i++) {

                out.data[i] = weights[1]*in(i-1) + weights[1]*in(i+1) + weights[1]*in(i + in.grid.x) +
                                weights[2]*in(i + in.grid.x - 1) + weights[2]*in(i + in.grid.x + 1);
                out.data[i + (out.grid.y-1)*out.grid.x] = weights[1]*in(i-1 + (in.grid.y-1)*in.grid.x) + weights[1]*in(i+1 + (in.grid.y-1)*in.grid.x) + weights[1]*in(i + (in.grid.y-2)*in.grid.x) +
                        weights[2]*in(i-1 + (in.grid.y-2)*in.grid.x) + weights[2]*in(i+1 + (in.grid.y-2)*in.grid.x);

        }

        //the left/right edge
#pragma omp parallel for
        for (i = 1; i < out.grid.y - 1; i++) {

                 out.data[(i+1)*out.grid.x-1] = weights[1]*in(i*in.grid.x - 1) + weights[1]*in((i+1)*in.grid.x - 2) + weights[1]*in((i+2)*in.grid.x - 1) + weights[2]*in(i*in.grid.x - 2) + weights[2]*in((i+2)*in.grid.x - 2);

                out.data[(i-1)*out.grid.x + out.grid.x] = weights[1]*in(i*in.grid.x + in.grid.x) + weights[1]*in( (i-2)*in.grid.x + in.grid.x) + weights[1]*in( (i-1)*in.grid.x + in.grid.x + 1) + weights[2]*in( (i-2)*in.grid.x + in.grid.x + 1) + weights[2]*in(i*in.grid.x + in.grid.x + 1);
        }

        //now the interior points
#pragma omp parallel for collapse(2)
                for (j = 1; j < in.grid.y - 1; j++){
                        for (i = 1; i < in.grid.x - 1; i++) {
                                out.data[i + j*out.grid.x] = weights[1]*in(i + (j-1)*in.grid.x) + weights[1]*in( (i-1) + j*in.grid.x) + weights[1]*in( (i+1) + j*in.grid.x ) + weights[1]*in(i + (j+1)*in.grid.x) +
                                        weights[2]*in(i + (j+1)*in.grid.x + 1) + weights[2]*in(i + (j+1)*in.grid.x - 1) +
                                        weights[2]*in(i + (j-1)*in.grid.x - 1) +  weights[2]*in(i + (j-1)*in.grid.x + 1);
                        }

                }
}
template void coarse_LU<float>(const Vector<float>& in, Vector<float>& out, std::vector<float>& weights);
template void coarse_LU<double>(const Vector<double>& in, Vector<double>& out, std::vector<double>& weights);

//applies the inv of the diagonal of the matrix
//this can be reused for the coarse grid matrices
template<class T>
void Dinv(const Vector<T>& in, Vector<T>& out)
{
	assert(in.size == out.size);
	int i; //int j;
	T quart = static_cast<T>(0.25);
#pragma omp parallel for
	for (i = 0; i < out.size; i++) { out.data[i] = quart*in(i); }
}
template void Dinv<float>(const Vector<float>& in, Vector<float>& out);
template void Dinv<double>(const Vector<double>& in, Vector<double>& out);

template<class T>
void DinvPBC(const Vector<T>& in, Vector<T>& out, const T& eps)
{
        assert(in.size == out.size);
        int i; //int j;
        T quart = static_cast<T>(1.0/(4 + eps));
#pragma omp parallel for
        for (i = 0; i < out.size; i++) { out.data[i] = quart*in(i); }
}
template void DinvPBC<float>(const Vector<float>& in, Vector<float>& out, const float& eps);
template void DinvPBC<double>(const Vector<double>& in, Vector<double>& out, const double& eps);

template<class T, class V>
void DinvPBCGPU(const V& in, V& out, const T& eps)
{
        assert(in.size == out.size);
	const int N = out.size;
        int i;
        T quart = static_cast<T>(1.0/(4 + eps));
	#pragma omp target teams distribute parallel for
        for (i = 0; i < N; i++) { out.data[i] = quart*in(i); }
}
template void DinvPBCGPU<float, Vector<float>>(const Vector<float>& in, Vector<float>& out, const float& eps);
template void DinvPBCGPU<double, Vector<double>>(const Vector<double>& in, Vector<double>& out, const double& eps);
template void DinvPBCGPU<float, GPUVector<float>>(const GPUVector<float>& in, GPUVector<float>& out, const float& eps);
template void DinvPBCGPU<double, GPUVector<double>>(const GPUVector<double>& in, GPUVector<double>& out, const double& eps);

template<class T>
void coarse_Dinv(const Vector<T>& in, Vector<T>& out, std::vector<T>& diag)
{
	assert(in.size == out.size);
	int i;
#pragma omp parallel for
	for (i = 0; i < out.size; i++) { out.data[i] = (1.0/diag[0])*in(i); }
}
template void coarse_Dinv<float>(const Vector<float>& in, Vector<float>& out, std::vector<float>& diag);
template void coarse_Dinv<double>(const Vector<double>& in, Vector<double>& out, std::vector<double>& diag);

//prolongator
template <class T>
void prolong(const Vector<T>& in, Vector<T>& out)
{
	int i; int j;

	//do the corners first
	out.data[0] = 0.25*in(0);
        //out.data[out.grid.x*out.grid.y-1] = 0.25*in(in.grid.x*in.grid.y-1);
        //out.data[out.grid.x-1] = 0.25*in(in.grid.x-1);
        //out.data[out.grid.x*out.grid.y - out.grid.x] = 0.25*in(in.grid.x*in.grid.y - in.grid.x);

	//bottom/top edges. using the fact that ints will be rounded down
#pragma omp parallel for
	for (i = 1; i < out.grid.x - 1; i+=2){ //was out.grid.x - 1
		//bottom
		out.data[i] = 0.5*in(i/2);
		if (i+1 == out.grid.x-1) {
		out.data[i+1] = 0.25*(in(i/2)); //problem is here I think
		} else {
		out.data[i+1] = 0.25*(in(i/2) + in(i/2 + 1));
		}
		//top
		out.data[i + (out.grid.y-1)*out.grid.x] = 0.5*in( (i/2) + (in.grid.y-1)*in.grid.x);
		out.data[i+1 + (out.grid.y-1)*out.grid.x] = 0.25*(in( (i/2) + (in.grid.y-1)*in.grid.x) + in(i/2 + 1 + (in.grid.y-1)*in.grid.x));
	}

	//left/right edge
#pragma omp parallel for
	for (j = 1; j < out.grid.y - 1; j+=2){ //was  out.grid.y - 1
		//left edge
		out.data[j*out.grid.x] = 0.5*in((j/2)*in.grid.x);
		out.data[(j+1)*out.grid.x] = 0.25*(in((j/2)*in.grid.x) + in( ((j/2)+1)*in.grid.x));
		//right edge
		out.data[(j+1)*out.grid.x-1] = 0.5*in( (j+1)/2*in.grid.x-1);
		out.data[(j+2)*out.grid.x-1] = 0.25*(in( (j+1)/2*in.grid.x-1) + in( (((j+1)/2)+1)*in.grid.x-1));
	}
#pragma omp parallel for collapse(2)
	for (j = 1; j < out.grid.y-1; j++){
		for (i = 1; i < out.grid.x-1; i++) {
			if (i%2 == 1 && j%2 == 1) {out.data[i+j*out.grid.x] = in(i/2 + (j/2)*in.grid.x);} //theres a one to one correspondence
			if (i%2 == 1 && j%2 == 0) {out.data[i+j*out.grid.x] = 0.5*(in(i/2 + (j/2)*in.grid.x) + in(i/2 + ((j/2)-1)*in.grid.x));} //average over verticals
			if (i%2 == 0 && j%2 == 1) {out.data[i+j*out.grid.x] = 0.5*(in(i/2 + (j/2)*in.grid.x) + in(i/2-1 + (j/2)*in.grid.x));}//average over horizontals
			if(i%2 == 0 && j%2 == 0) {out.data[i+j*out.grid.x] = 0.25*(in(i/2 + (j/2)*in.grid.x) + in(i/2-1 + (j/2-1)*in.grid.x) + in(i/2-1 + (j/2)*in.grid.x) + in(i/2 + (j/2-1)*in.grid.x));}
		}
	}
}
template void prolong<float>(const Vector<float>& in, Vector<float>& out);
template void prolong<double>(const Vector<double>& in, Vector<double>& out);

template<class T>
void prolongPBC(const Vector<T>& in, Vector<T>& out)
{
	//first the corners
	//the origin
	int i = 0; int j = 0;
	//std::cout << i + j * out.grid.x << std::endl;
	out.data[i + j * out.grid.x] = in( mod( i/2, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
	       			       0.5*in( mod( i/2-1, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
				       0.5*in( mod( i/2, in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x ) +
			               0.25*in( mod( i/2-1, in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x );
	//lower right corner
	i = out.grid.x - 1; j = 0;
	//std::cout << i + j * out.grid.x << std::endl;
	out.data[i + j * out.grid.x] = in( mod( i/2, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
				       0.5*in( mod( i/2+1, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
				        0.5*in( mod( i/2, in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x ) +
				       0.25*in( mod( i/2+1, in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x );

	//upper left corner
	i = 0; j = out.grid.y - 1;
	//std::cout << i + j * out.grid.x << std::endl;
	out.data[i + j * out.grid.x] = in( mod( i/2, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
		                       0.5*in( mod( i/2-1, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
				       0.5*in( mod( i/2, in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x ) +
				       0.25*in( mod( i/2-1, in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x );

	//upper right corner (opposite origin)
	i = out.grid.x - 1; j = out.grid.y - 1;
	//std::cout << i + j * out.grid.x << std::endl;
	out.data[i + j * out.grid.x] = in( mod( i/2, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
				       0.5*in( mod( i/2+1, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
				       0.5*in( mod( i/2, in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x ) +
				       0.25*in( mod( i/2+1, in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x );

	//now do the edges
	//bottom edge, odd case
	j = 0;
#pragma omp parallel for
	for (i = 1; i < out.grid.x - 1; i+=2 ) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j * out.grid.x] = 0.5*in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x) + 
					       0.5*in( mod(i/2+1,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x) + 
					       0.25*in( mod(i/2,in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x) + 
					       0.25*in( mod(i/2+1,in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x);

	}

	//bottom edge, even case
#pragma omp parallel for
	for (i = 2; i < out.grid.x - 1; i+=2) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j*out.grid.x] = in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) + 
					     0.5*in( mod(i/2,in.grid.x) + mod(j/2-1,in.grid.y)*in.grid.x );

	}

	//top edge, odd case
	j = out.grid.y - 1;
#pragma omp parallel for
	for (i = 1; i < out.grid.x - 1; i+=2 ) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j * out.grid.x] = 0.5*in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x) +
					       0.5*in( mod(i/2+1,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x) +
					       0.25*in( mod(i/2,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x) +
					       0.25*in( mod(i/2+1,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x);

	}

	//top edge, even case
#pragma omp parallel for
	for (i = 2; i < out.grid.x - 1; i +=2 ) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j*out.grid.x] = in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
					    0.5*in( mod(i/2,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x );
	}


	//the left edge, odd case
	i = 0;
#pragma omp parallel for
	for (j = 1; j < out.grid.y - 1; j += 2) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j*out.grid.x] = 0.5*in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) + 
					     0.5*in( mod(i/2,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x ) + 
					     0.25*in( mod(i/2 - 1, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
					     0.25*in( mod(i/2 - 1, in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x );		

	}

	//the left edge, even case
#pragma omp parallel for
	for ( j = 2; j < out.grid.y - 1; j +=2) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j*out.grid.x] = in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
					    0.5*in( mod(i/2 -1, in.grid.x) + mod(j/2,in.grid.y)*in.grid.x );

	}

	//the right edge, odd case (a little sus)
	i = out.grid.x - 1;
#pragma omp parallel for
	for ( j = 1; j < out.grid.y - 1; j += 2) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j*out.grid.x] = 0.5*in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
			                     0.5*in( mod(i/2,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x ) +
					     0.25*in( mod(i/2+1,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x) + 
					     0.25*in( mod(i/2+1,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x);
	}
	
	//the right edge, even case
#pragma omp parallel for
	for (j = 2; j < out.grid.y - 1; j +=2) {
		//std::cout << i + j * out.grid.x << std::endl;
		out.data[i + j*out.grid.x] = in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
					    0.5*in( mod(i/2+1,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x );

	}

	//now the interior points. breaking it up once again
	
	//odd,odd -> 4 neighbors (this is a problem here)
#pragma omp parallel for
	for (j = 1; j < out.grid.y - 1; j += 2) {
		for (i = 1; i < out.grid.x - 1; i += 2) {
			//std::cout << i + j * out.grid.x << std::endl;
			out.data[i + j*out.grid.x] = 0.25*( in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) + 
							   in( mod(i/2+1,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
							   in( mod(i/2,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x ) +
							   in( mod(i/2+1,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x) );

		}
	}

	//even,odd (also here)
#pragma omp parallel for
	for (j = 1; j < out.grid.y - 1; j += 2) {
		for (i = 2; i < out.grid.x - 1; i += 2) {
			//std::cout << i + j * out.grid.x << std::endl;
			out.data[i + j*out.grid.x] = 0.5*( in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
							  in( mod(i/2,in.grid.x) + mod(j/2+1,in.grid.y)*in.grid.x ) );

		}
	}
	
	//odd,even
#pragma omp parallel for
	for (j = 2; j < out.grid.y - 1; j += 2) {
		for (i = 1; i < out.grid.x - 1; i += 2) {
			//std::cout << i + j * out.grid.x << std::endl;
			out.data[i + j*out.grid.x] = 0.5*( in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) +
							  in( mod(i/2+1,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x ) );

		}
	}
	
	//even,even. direct injection only?
#pragma omp parallel for
	for (j = 2; j < out.grid.y - 1; j += 2) {
		for (i = 2; i < out.grid.x - 1; i += 2) {
			//std::cout << i + j * out.grid.x << std::endl;
			out.data[i + j*out.grid.x] = in( mod(i/2,in.grid.x) + mod(j/2,in.grid.y)*in.grid.x );

		}
	}


} 
template void prolongPBC<float>(const Vector<float>& in, Vector<float>& out);
template void prolongPBC<double>(const Vector<double>& in, Vector<double>& out);

//restrictor
template<class T>
void restrict(const Vector<T>& in, Vector<T>& out)
{
	//right now this only does nearest neighbor on the edges
	//might need to implement next nearest neighbor
	//keeping it this way for now
	int i; int j;

	//loop over the COARSE interior nodes
#pragma omp parallel for collapse(2)
        for (j = 0; j < out.grid.y; j++){
                for (i = 0; i < out.grid.x; i++) {
                        out.data[i + j*out.grid.x] = 0.25*in((2*i+1) + (2*j+1)*in.grid.x) //central node
							+ 0.125*in( (2*i) +(2*j+1)*in.grid.x) // -x dir
							+ 0.125*in( (2*i+2) + (2*j+1)*in.grid.x) // +x dir
							+ 0.125*in( (2*i+1) + (2*j)*in.grid.x) // -y dir
							+ 0.125*in( (2*i+1) + (2*j+2)*in.grid.x) // +y dir
							+ 0.0625*in( (2*i) + (2*j)*in.grid.x) // SW corner
							+ 0.0625*in( (2*i) + (2*j+2)*in.grid.x) //NW corner
							+ 0.0625*in( (2*i+2) + (2*j+2)*in.grid.x) //NE corner
							+ 0.0625*in( (2*i+2) + (2*j)*in.grid.x); //SE corner
                }
        }


}
template void restrict<float>(const Vector<float>& in, Vector<float>& out);
template void restrict<double>(const Vector<double>& in, Vector<double>& out);

template<class T>
void restrictPBC(const Vector<T>& in, Vector<T>& out)
{
        //right now this only does nearest neighbor on the edges
        //might need to implement next nearest neighbor
        //keeping it this way for now
        int i; int j;

        //loop over the COARSE interior nodes
#pragma omp parallel for collapse(2)
        for (j = 0; j < out.grid.y; j++){
                for (i = 0; i < out.grid.x; i++) {
                        out.data[i + j*out.grid.x] = 0.25*in(mod(2*i,in.grid.x) + mod(2*j,in.grid.y)*in.grid.x) //central node
                                                        + 0.125*in( mod(2*i+1,in.grid.x) + mod(2*j,in.grid.y)*in.grid.x) // -x dir
                                                        + 0.125*in( mod(2*i,in.grid.x) + mod(2*j+1,in.grid.y)*in.grid.x) // +x dir
                                                        + 0.125*in( mod(2*i-1,in.grid.x) + mod(2*j,in.grid.y)*in.grid.x) // -y dir
                                                        + 0.125*in( mod(2*i,in.grid.x) + mod(2*j-1,in.grid.y)*in.grid.x) // +y dir
                                                        + 0.0625*in( mod(2*i+1,in.grid.x) + mod(2*j+1,in.grid.y)*in.grid.x) // SW corner
                                                        + 0.0625*in( mod(2*i-1,in.grid.x) + mod(2*j+1,in.grid.y)*in.grid.x) //NW corner
                                                        + 0.0625*in( mod(2*i-1,in.grid.x) + mod(2*j-1,in.grid.x)*in.grid.x) //NE corner
                                                        + 0.0625*in( mod(2*i+1,in.grid.x) + mod(2*j-1,in.grid.x)*in.grid.x); //SE corner
                }
        }


}
template void restrictPBC<float>(const Vector<float>& in, Vector<float>& out);
template void restrictPBC<double>(const Vector<double>& in, Vector<double>& out);

//set up the vcycle
template<class T>
Vcycle<T> setup(int p, int q, int smooth_iters, T smooth_target, int coarse_iters, T coarse_target)
{
        int min_level = 3;
        int num_levels = p - min_level + 1;

        //set up the fine grid independently
        Grid grid_f(p,q);

        std::cout << "Fine grid is " << grid_f.x << " x " << grid_f.y << std::endl;


        std::function<void(const Vector<T>&, Vector<T>&)> A = op<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> P = prolong<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> R = restrict<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> lu = LU<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> dinv = Dinv<T>;

        int el = 0;
        Smoother<T> s_f(lu,dinv,A,smooth_iters,smooth_target);
        std::vector<T> foo(3);

        auto fine_level = std::make_shared<Level<T,std::function<void(const Vector<T>&, Vector<T>&)>,Smoother<T>>>(el, grid_f, A, foo, P, R, s_f);
        auto coarse_levels = std::make_shared<std::vector<Level<T,std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)>,CoarseSmoother<T>>>>(num_levels - 1);

        for (int i = 0; i < num_levels - 1; i++) {

                p -= 1; q -= 1;
                el += 1;

                Grid grid_c(p,q);
                std::vector<T> weights(3);
                if ( i == 0) {
                        weights = calc_weights<T>(grid_c, A, P, R);
                } else {
                        weights = calc_coarse_weights<T>(grid_c, (*coarse_levels)[i-1].A, (*coarse_levels)[i-1].weights, P, R);
                }

                std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)> Ac = coarse_op<T>;
                std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)> luc = coarse_LU<T>;
                std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)> dinvc = coarse_Dinv<T>;
                CoarseSmoother<T> s_c;

                if ( i == num_levels - 2) {
                        s_c.create(luc,dinvc,Ac,weights,coarse_iters,coarse_target);
                } else {
                        s_c.create(luc,dinvc,Ac,weights,smooth_iters,smooth_target);
                }
                (*coarse_levels)[i].create(el, grid_c, Ac, weights, P, R, s_c);

        }

        Vcycle<T> vcycle;
        vcycle.num_levels = num_levels;
        vcycle.fine_level = fine_level;
        vcycle.coarse_levels = coarse_levels;
        return vcycle;

}

template Vcycle<double> setup<double>(int p, int q, int smooth_iters, double smooth_target, int coarse_iters, double coarse_target);
template Vcycle<float> setup<float>(int p, int q, int smoother_iters, float smooth_target, int coarse_iters, float coarse_target);

template<class T>
VcyclePBC<T> setupPBC(int p, int q, const T& eps, int smooth_iters, T smooth_target, int coarse_iters, T coarse_target)
{
        int min_level = 2;
        int num_levels = p - min_level + 1;

        //set up the fine grid independently
        Grid grid_f(p,q);

        //std::cout << "Fine grid is " << grid_f.x << " x " << grid_f.y << std::endl;


        std::function<void(const Vector<T>&, Vector<T>&,const T&)> A = opPBC<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> P = prolongPBC<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> R = restrictPBC<T>;
        std::function<void(const Vector<T>&, Vector<T>&)> lu = LUPBC<T>;
        std::function<void(const Vector<T>&, Vector<T>&,const T&)> dinv = DinvPBC<T>;

        int el = 0;
        SmootherPBC<T> s_f(lu,dinv,A,eps,smooth_iters,smooth_target);
        std::vector<T> foo(3);

        auto fine_level = std::make_shared<Level<T,std::function<void(const Vector<T>&, Vector<T>&,const T&)>,SmootherPBC<T>>>(el, grid_f, A, foo, P, R, s_f);
        auto coarse_levels = std::make_shared<std::vector<Level<T,std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)>,CoarseSmootherPBC<T>>>>(num_levels - 1);

        for (int i = 0; i < num_levels - 1; i++) {

                p -= 1; q -= 1;
                el += 1;

                Grid grid_c(p,q);
                std::vector<T> weights(3);
                if ( i == 0) {
                        weights = calc_weightsPBC<T>(grid_c, A, P, R, eps);
			for ( auto w : weights ) { std::cout << w << std::endl; }
                } else {
                        weights = calc_coarse_weights<T>(grid_c, (*coarse_levels)[i-1].A, (*coarse_levels)[i-1].weights, P, R);
                }

                std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)> Ac = coarse_opPBC<T>;
                std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)> luc = coarse_LUPBC<T>;
                std::function<void(const Vector<T>&, Vector<T>&,std::vector<T>&)> dinvc = coarse_Dinv<T>;
                CoarseSmootherPBC<T> s_c;

                if ( i == num_levels - 2) {
                        s_c.create(luc,dinvc,Ac,weights,coarse_iters,coarse_target);
                } else {
                        s_c.create(luc,dinvc,Ac,weights,smooth_iters,smooth_target);
                }
                (*coarse_levels)[i].create(el, grid_c, Ac, weights, P, R, s_c);

        }

        VcyclePBC<T> vcycle;
        vcycle.num_levels = num_levels;
        vcycle.fine_level = fine_level;
        vcycle.coarse_levels = coarse_levels;
        return vcycle;

}

template VcyclePBC<double> setupPBC<double>(int p, int q, const double& eps, int smooth_iters, double smooth_target, int coarse_iters, double coarse_target);
template VcyclePBC<float> setupPBC<float>(int p, int q, const float& eps, int smoother_iters, float smooth_target, int coarse_iters, float coarse_target);


} //namespace
