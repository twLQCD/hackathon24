#pragma once

#include <iomanip>
#include <iostream>
#include <random>
#include <cassert>
#include <omp.h>

#include "grid.hpp" 

namespace Poisson {

template<class T, class V>
void setEqualTo(V& x, V& y, const int& N)
{
	#pragma omp target data map(to:x.data[0:N]) map(tofrom:y.data[0:N])
	{
		#pragma omp target teams distribute parallel for
		for (int i = 0; i < N; i++) {y.data[i] = x.data[i];}
	}
}

template<class T, class V>
T vecnorm2(V& x, V& y) {
	assert(x.size == y.size);
	const int N = x.size;
	T out = 0.0;
	#pragma omp target teams distribute parallel for reduction(+:out) map(to: x.data[0:N],y.data[0:N])
	{
	for (int i = 0; i < N; i++) { out += x.data[i]*y.data[i]; }
	}
	return out;
}	

template<class T>
T vecnorm3(T* x, T* y, const int N)
{
	T out = 0.0;
#pragma omp target teams distribute parallel for reduction(+:out) map(to: x[0:N],y[0:N])
	{
		for (int i = 0; i < N; i++) { out += x[i]*y[i]; }
	}
	return out;
}

template<class T>
class Vector {
	public:
		//Grid sets up the 2D grid/mesh. Contains the dimensions
		//grid.x and grid.y
		Grid grid;
		
		//for convenience
		//size = grid.x * grid.y
		int size;

		//the data of the vector
		T* data;

		//default constructor
		Vector<T>(){};

		//copy constructor
		Vector<T>(Vector<T>& v) : grid(v.grid)
       		{ 
			size = v.size;
			data = new T[v.size];
		#pragma omp parallel for
			for (int i = 0; i < v.size; i++) data[i] = (v.data)[i];
		}

		//parameterized constructor
		Vector<T>(Grid& grid) : grid(grid) { size = grid.x * grid.y; data = new T[size]; zeros(); }

		//parameterize constructor
		Vector<T>(Grid& grid, T* data) : grid(grid), data(data) {size = grid.x * grid.y;};

		//to be used when default constructor is called
		void create(const Grid& grid_t)
		{
		 grid = grid_t;
		 size = grid.x*grid.y;
		 data = new T[size];
		 zeros();
		}

		//prints out the elements of the vector
		void print() { for (int i = 0; i < size; i++) std::cout << std::setprecision(std::numeric_limits<T>::max_digits10) << data[i] << std::endl; }

		//set to zero
		void zeros()
		{
		#pragma omp parallel for
			for (int i = 0; i < size; i++) data[i] = static_cast<T>(0);
		}

		//set to ones
		void ones()
		{
		#pragma omp parallel for
			for (int i = 0; i < size; i++) data[i] = static_cast<T>(1);
		}

		void unit(int j)
		{
		this->zeros();
		this->data[j] = 1.0;
		}

		//set to random normal with mean 0
		//and std 1
		void randn()
		{
			std::random_device dev;
			std::mt19937 gen(dev());
			std::normal_distribution<T> d(static_cast<T>(0.0), static_cast<T>(1.0));
		#pragma omp parallel for
			for (int i = 0; i < size; i++) data[i] = d(gen);
		}

		//return the 2-norm of the vector
		T norm2()
		{
			T tmp = static_cast<T>(0);
		#pragma omp parallel for reduction(+ : tmp)
			for (int i = 0; i < size; i++) { tmp += data[i]*data[i]; }
			return tmp;
		}

		//return the sum of a vector
		T sum()
		{
			T tmp = static_cast<T>(0);
		#pragma omp parallel for reduction(+ : tmp)
			for (int i = 0; i < size; i++) {tmp += data[i]; }
			return tmp;
		}

		//overloaded * op -> returns element wise
		//multiplication, not a norm
		Vector<T> operator*(const Vector<T>& v)
		{
			assert(size == v.size);
			T* tmp = new T[v.size];
			Grid grid_t(v.grid.p,v.grid.q);
		#pragma omp parallel for
			for (int i = 0; i < size; i++) {tmp[i] = data[i] * v.data[i];}
			Vector<T> newvec(grid_t, tmp);
			return newvec;
		//#pragma omp parallel for
			//for (int i = 0; i < size; i++) {this->data[i] * v.data[i];}
			//return *this;
		}


		/*friend Vector<T> operator*(Vector<T> lhs, const Vector<T>& rhs)
		{
			lhs *= rhs;
			return lhs;
		}*/

		//overloaded + op
		Vector<T> operator+(const Vector<T>& v) 
                {
                        assert(size == v.size);
                        T* tmp = new T[v.size];
			Grid grid_t(v.grid.p,v.grid.q);
                #pragma omp parallel for
                        for (int i = 0; i < size; i++) {tmp[i] = data[i] + v.data[i];}
                        Vector<T> newvec(grid_t, tmp);
                        return newvec;
		//#pragma omp parallel for
			//for (int i = 0; i < size; i++) {this->data[i] + v.data[i];}
			//return *this;
                }

		/*friend Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
		{
			lhs += rhs;
			return lhs;
		}*/

		//overloaded - op
		Vector<T> operator-(const Vector<T>& v)
                {
                        assert(size == v.size);
                        T* tmp = new T[v.size];
			Grid grid_t(v.grid.p,v.grid.q);
                #pragma omp parallel for
                        for (int i = 0; i < size; i++) {tmp[i] = data[i] - v.data[i];}
                        Vector<T> newvec(grid_t, tmp);
                        return newvec;
		//#pragma omp parallel for
		//	for (int i = 0; i < size; i++) {this->data[i] - v.data[i];}
		//	return *this;
                }

		/*friend Vector<T>& operator-(Vector<T> lhs, const Vector<T>& rhs)
		{
			lhs += rhs;
			return lhs;
		}*/

		//assignment operator
		void operator=(const Vector<T>& v)
		{
			//grid = v.grid;
			//size = grid.x * grid.y;
			//size = v.size;
			//grid = v.grid;
			assert(this->size == v.size);
		#pragma omp parallel for
			for (int i = 0; i < size; i++) {data[i] = v.data[i];}
			//return *this;
			/*if (this == &v) {return *this;}
			assert(this -> size == v.size);
			std::copy(v.data, v.data + v.size, this -> data);
			return *this;*/
		}

		//return the element of the vector given a linear index
		T operator()(int i) const
		{
			return data[i];
		}

		//return the element of the vector given a two dimensional
		//index
		/*T operator()(int i, int j)
		{
			return data[i + j*grid.y];
		} */

		//destructor
		~Vector<T>(){ delete[] data; }


}; //class

template<class T>
class GPUVector {
        public:
                //Grid sets up the 2D grid/mesh. Contains the dimensions
                //grid.x and grid.y
                Grid grid;

                //for convenience
                //size = grid.x * grid.y
                int size;

                //the data of the vector
                T* data;

                //default constructor
                GPUVector<T>(){};

                //copy constructor
                GPUVector<T>(GPUVector<T>& v) : grid(v.grid)
                {
                        size = v.size;
                        data = new T[v.size];
		#pragma omp target teams distribute parallel for map(to: data[0:size]) map(from: v.data[0:size])
                        for (int i = 0; i < v.size; i++) data[i] = (v.data)[i];
                }

                //parameterized constructor
                GPUVector<T>(Grid& grid) : grid(grid) { size = grid.x * grid.y; data = new T[size]; zeros(); }

                //parameterize constructor
                GPUVector<T>(Grid& grid, T* data) : grid(grid), data(data) {size = grid.x * grid.y;};

                //to be used when default constructor is called
                void create(const Grid& grid_t)
                {
                 grid = grid_t;
                 size = grid.x*grid.y;
                 data = new T[size];
                 zeros();
                }

                //prints out the elements of the vector
                void print() { for (int i = 0; i < size; i++) std::cout << std::setprecision(std::numeric_limits<T>::max_digits10) << data[i] << std::endl; }


		void setdata(const T* in) 
		{
		#pragma omp parallel for
			for (int i = 0; i < size; i++) data[i] = in[i];
		}

                //set to zero
                void zeros()
                {
                #pragma omp parallel for
                        for (int i = 0; i < size; i++) data[i] = 0.0;
                }

                //set to ones
                void ones()
                {
                #pragma omp parallel for
                        for (int i = 0; i < size; i++) data[i] = 1.0;
                }

                void unit(int j)
                {
                this->zeros();
                this->data[j] = 1.0;
                }

                //set to random normal with mean 0
                //and std 1
                void randn()
                {
                        std::random_device dev;
                        std::mt19937 gen(dev());
                        std::normal_distribution<T> d(static_cast<T>(0.0), static_cast<T>(1.0));
                #pragma omp parallel for
                        for (int i = 0; i < size; i++) data[i] = d(gen);
                }

                //return the 2-norm of the vector
                T norm2()
                {
                        T tmp = static_cast<T>(0);
		#pragma omp target teams distribute parallel for reduction(+ : tmp) map(to: data[0:size])
                        for (int i = 0; i < size; i++) { tmp += data[i]*data[i]; }
                        return tmp;
                }

                //return the sum of a vector
                T sum()
                {
                        T tmp = static_cast<T>(0);
                #pragma omp target teams distribute parallel for reduction(+ : tmp) map(to: data[0:size])
                        for (int i = 0; i < size; i++) {tmp += data[i]; }
                        return tmp;
                }

                //overloaded * op -> returns element wise
                //multiplication, not a norm
                GPUVector<T> operator*(const GPUVector<T>& v)
                {
                        assert(size == v.size);
                        T* tmp = new T[v.size];
                        Grid grid_t(v.grid.p,v.grid.q);
                #pragma omp target teams distribute parallel for
                        for (int i = 0; i < size; i++) {tmp[i] = data[i] * v.data[i];}
                        GPUVector<T> newvec(grid_t, tmp);
                        return newvec;
                }


                /*friend Vector<T> operator*(Vector<T> lhs, const Vector<T>& rhs)
                {
                        lhs *= rhs;
                        return lhs;
                }*/

                //overloaded + op
                GPUVector<T> operator+(const GPUVector<T>& v)
                {
                        assert(size == v.size);
                        T* tmp = new T[v.size];
                        Grid grid_t(v.grid.p,v.grid.q);
                #pragma omp target teams distribute parallel for
                        for (int i = 0; i < size; i++) {tmp[i] = data[i] + v.data[i];}
                        GPUVector<T> newvec(grid_t, tmp);
                        return newvec;
                //#pragma omp parallel for
                        //for (int i = 0; i < size; i++) {this->data[i] + v.data[i];}
                        //return *this;
                }

                /*friend Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
                {
                        lhs += rhs;
                        return lhs;
                }*/

                //overloaded - op
                GPUVector<T> operator-(const GPUVector<T>& v)
                {
                        assert(size == v.size);
                        T* tmp = new T[v.size];
                        Grid grid_t(v.grid.p,v.grid.q);
		#pragma omp target teams distribute parallel for
                        for (int i = 0; i < size; i++) {tmp[i] = data[i] - v.data[i];}
                        GPUVector<T> newvec(grid_t, tmp);
                        return newvec;
                //#pragma omp parallel for
                //      for (int i = 0; i < size; i++) {this->data[i] - v.data[i];}
                //      return *this;
                }

                /*friend Vector<T>& operator-(Vector<T> lhs, const Vector<T>& rhs)
                {
                        lhs += rhs;
                        return lhs;
                }*/

                //assignment operator
                void operator=(const GPUVector<T>& v)
                {

                          assert(this->size == v.size);
			  const int N = v.size;
		#pragma omp target data map(to:v.data[0:N]) map(tofrom:this->data[0:N])
		{
                #pragma omp target teams distribute parallel for
                        for (int i = 0; i < N; i++) {this->data[i] = v.data[i];}
		}
                        /*assert(this->size == v.size);
                #pragma omp parallel for
                        for (int i = 0; i < size; i++) {data[i] = v.data[i];} */
                }

                //return the element of the vector given a linear index
                T operator()(int i) const
                {
                        return data[i];
                }

                //return the element of the vector given a two dimensional
                //index
                /*T operator()(int i, int j)
                {
                        return data[i + j*grid.y];
                } */

                //destructor
                ~GPUVector<T>(){ delete[] data; }


}; //class

} //namespace
