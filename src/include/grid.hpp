#pragma once

#include <cassert>
#include <cmath>

namespace Poisson {

struct Grid {

	//power to make dims in the x direction
	int p;

	//dim in x direction
	int x;

	//power to make dims in the y direction 
	int q;

	//dim in y direction
	int y;

	//default constructor
	Grid(){};

	//parameterized constructor
	Grid(int a, int b) : p(a), q(b)
	{
		assert( a >= 2 && b >= 2);
		//this was previously pow(2.0, a) - 1
		//but I have changed it to pow(2.0, a) + 1
		//as this is easier for periodic boundary conditions
		x = static_cast<int>(pow(2.0, a) + 1);
		y = static_cast<int>(pow(2.0, b) + 1);
	//keeping it isotropic
	//on the boundary [0,1] x [0,1]
	}

	void operator=(const Grid& grid_t)
	{ 	p = grid_t.p;
		q = grid_t.q;
		x = grid_t.x;
		y = grid_t.y;
	}
	//destructor
	~Grid(){};
};

} //namespace
