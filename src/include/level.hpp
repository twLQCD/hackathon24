#pragma once

#include "matrices.hpp"
#include "smoothers.hpp"
#include "vector.hpp"

namespace Poisson {


template<class T, class M, class U>
struct Level {

	//the level
	int el;
	//the grid at the level el
	Grid grid;
	//the matrix
	M A;
	//the coeffs of the matrix
	std::vector<T> weights;
	//the prolongator
	std::function<void(const Vector<T>&,Vector<T>&)> P;
	//the restrictor
	std::function<void(const Vector<T>&,Vector<T>&)> R;
	//the smoother
	U S;
	//the solution at the level el
	Vector<T> x;
	//the right hand side at level el
	Vector<T> b;
	//the residual vector at level el
	Vector<T> r;
	//the error vector at level el
	Vector<T> e;
	//temporary for holding Ax
	Vector<T> Ax;

	Level(){};
	Level(int el, Grid& grid, M& A, std::vector<T>& weights, std::function<void(const Vector<T>&,Vector<T>&)>& P, std::function<void(const Vector<T>&,Vector<T>&)>& R, U& S) :
		el(el),
		grid(grid),
		A(A),
		weights(weights),
		P(P),
		R(R),
		S(S)
	{
		x.create(grid);
		b.create(grid);
		r.create(grid);
		e.create(grid);
		Ax.create(grid);
	
	}
	void create(int el_t, Grid& grid_t, M& A_t, std::vector<T>& weights_t, std::function<void(const Vector<T>&,Vector<T>&)>& P_t, std::function<void(const Vector<T>&,Vector<T>&)>& R_t, U& S_t)
	{
	
		el = el_t;
		grid = grid_t;
		weights = weights_t;
		A = A_t;
		P = P_t;
		R = R_t;
		S = S_t;

                x.create(grid);
                b.create(grid);
                r.create(grid);
                e.create(grid);
                Ax.create(grid);

	}

	void operator=(const Level<T,M,U>& level)
	{
		el = level.el;
		grid = level.grid;
		weights = level.weights;
		A = level.A;
		P = level.P;
		R = level.R;
		S = level.S;

		x.create(level.grid);
		x = level.x;
		b.create(level.grid);
		b = level.b;
		r.create(level.grid);
		r = level.r;
		e.create(level.grid);
		e = level.e;
		Ax.create(level.grid);
		Ax = level.Ax; 

	}

	~Level(){};
}; //Level

} // namespace
