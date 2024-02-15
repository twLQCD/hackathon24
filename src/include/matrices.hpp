#pragma once

#include <iostream>
#include <vector>
#include <functional>

#include "vector.hpp"
#include "vcycle.hpp"

namespace Poisson {

//calculates the coefficients of the coarse grid matrices if transfering from fine to coarse
template<class T>
extern std::vector<T> calc_weights(Grid& grid, std::function<void(const Vector<T>&,Vector<T>&)>& op, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R);

template<class T>
extern std::vector<T> calc_weightsPBC(Grid& grid, std::function<void(const Vector<T>&,Vector<T>&)>& op, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R, const T& eps);

//calculates the coefficients of the coarse grid matrix if transfering from coarse to coarse
template<class T>
extern std::vector<T> calc_coarse_weights(Grid& grid, std::function<void(const Vector<T>&,Vector<T>&,std::vector<T>&)>& op, std::vector<T>& prev_coeffs, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R);

//the fine grid matvec
template<class T>
extern void op(const Vector<T>& in, Vector<T>& out);

template<class T>
extern void opPBC(const Vector<T>& in, Vector<T>& out, const T& eps);

template<class T, class V>
extern void opPBCGPU(const V& in, V& out, const T& eps);

template<class T>
extern void opPBCGPU2(const T* in, T* out, const T& eps, const int& x, const int& y);

//applies the coarse grid matrix matvec
template<class T>
extern void coarse_op(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights);

template<class T>
extern void coarse_opPBC(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights);

//applies (L+U) of A
template<class T>
extern void LU(const Vector<T>& in, Vector<T>& out);

template<class T>
extern void LUPBC(const Vector<T>& in, Vector<T>& out);

template<class T, class V>
extern void LUPBCGPU(const V& in, V& out);

//applies (L+U) of a coarse grid matrix
template<class T>
extern void coarse_LU(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights);

//applies Dinv, the inverse of the diagonal of coarse operator A_c
template<class T>
extern void coarse_Dinv(const Vector<T>& in, Vector<T>& out, std::vector<T>& weights);

//applies Dinv, the inverse of the diagonal of A
template<class T>
extern void Dinv(const Vector<T>& in, Vector<T>& out);

template<class T>
extern void DinvPBC(const Vector<T>& in, Vector<T>& out, const T& eps);

template<class T, class V>
extern void DinvPBCGPU(const V& in, V& out, const T& eps);

//prolongate
template<class T>
extern void prolong(const Vector<T>& in, Vector<T>& out);

template<class T>
extern void prolongPBC(const Vector<T>& in, Vector<T>& out);

//restrict
template<class T>
extern void restrict(const Vector<T>& in, Vector<T>& out);

template<class T>
extern void restrictPBC(const Vector<T>& in, Vector<T>& out);

//the multigrid setup
template<class T>
extern Vcycle<T> setup(int p, int q, int smooth_iters, T smooth_target, int coarse_iters, T coarse_target);

template<class T>
extern VcyclePBC<T> setupPBC(int p, int q, const T& eps, int smooth_iters, T smooth_target, int coarse_iters, T coarse_target);

}//namespace

