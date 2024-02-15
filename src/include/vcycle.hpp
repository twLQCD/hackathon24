#pragma once

#include <memory>

#include "smoothers.hpp"
#include "vector.hpp"
#include "matrices.hpp"
#include "level.hpp"

namespace Poisson {

template <class T>
class Vcycle {
	public:
		int num_levels;
		std::shared_ptr<Level<T, std::function<void(const Vector<T>&,Vector<T>&)>, Smoother<T>>> fine_level{nullptr};
		std::shared_ptr<std::vector<Level<T, std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>&)>, CoarseSmoother<T>>>> coarse_levels{nullptr};

		//default constructor only
		Vcycle(){}

		//default destructor
		~Vcycle(){}

	
		void cycle(int level) 
		{
			if (level == num_levels - 1)
			{
				(*coarse_levels)[num_levels-2].S((*coarse_levels)[num_levels-2].b,(*coarse_levels)[num_levels-2].x);
				return;

			}

			if (level == 0) { //on the finest level

				//pre smooth
				fine_level->S(fine_level->b,fine_level->x);

				//compute residual
				fine_level->A(fine_level->x,fine_level->Ax);
				
				fine_level->r = fine_level->b - fine_level->Ax;
				

				//restrict
				fine_level->R(fine_level->r,(*coarse_levels)[level].b);

				//recursive call
				(*coarse_levels)[level].x.zeros();
				cycle(level+1);

				//prolong the error to the fine level
				fine_level->P((*coarse_levels)[level].x,fine_level->e);


				//add it to the current solution on the fine level
				fine_level->x = fine_level->x + fine_level->e;

				//post smoothing
				fine_level->S(fine_level->b,fine_level->x);


			} else {

				//pre smooth
				(*coarse_levels)[level-1].S((*coarse_levels)[level-1].b,(*coarse_levels)[level-1].x);

				//compute residual
				(*coarse_levels)[level-1].A((*coarse_levels)[level-1].x,(*coarse_levels)[level-1].Ax,(*coarse_levels)[level-1].weights);
				(*coarse_levels)[level-1].r = (*coarse_levels)[level-1].b - (*coarse_levels)[level-1].Ax;


				//restrict to the next coarsest level
				(*coarse_levels)[level].b.zeros();
				(*coarse_levels)[level-1].R((*coarse_levels)[level-1].r,(*coarse_levels)[level].b);

				//recursive call
				(*coarse_levels)[level].x.zeros();
				cycle(level+1);

				//prolong the error to the next level
				(*coarse_levels)[level-1].P((*coarse_levels)[level].x,(*coarse_levels)[level-1].e);


				//add it to the current solution
				(*coarse_levels)[level-1].x = (*coarse_levels)[level-1].x + (*coarse_levels)[level-1].e;


				//post smoothing
				(*coarse_levels)[level-1].S((*coarse_levels)[level-1].b,(*coarse_levels)[level-1].x);


			}

		}

		void operator()(int max_iters, T target)
		{

			int iter = 1;
			int level = 0;
			T rn = std::sqrt(fine_level->b.norm2());
			while (iter <= max_iters && rn >= target) {
				cycle(level);
				iter += 1;
				fine_level->A(fine_level->x,fine_level->Ax);
				fine_level->r = fine_level->b - fine_level->Ax;
				rn = std::sqrt(fine_level->r.norm2());
			}
			std::cout << "Vcycle finished in " << iter-1 << " iterations with ||r||_2 = " << rn << std::endl;

		}



};

template <class T>
class VcyclePBC {
        public:
                int num_levels;
                std::shared_ptr<Level<T, std::function<void(const Vector<T>&,Vector<T>&, const T&)>, SmootherPBC<T>>> fine_level{nullptr};
                std::shared_ptr<std::vector<Level<T, std::function<void(const Vector<T>&,Vector<T>&, std::vector<T>&)>, CoarseSmootherPBC<T>>>> coarse_levels{nullptr};

                //default constructor only
                VcyclePBC(){}

                //default destructor
                ~VcyclePBC(){}


                void cycle(int level)
                {
                        if (level == num_levels - 1)
                        {
                                (*coarse_levels)[num_levels-2].S((*coarse_levels)[num_levels-2].b,(*coarse_levels)[num_levels-2].x);
                                return;

                        }

                        if (level == 0) { //on the finest level

                                //pre smooth
                                fine_level->S(fine_level->b,fine_level->x);

                                //compute residual
                                fine_level->A(fine_level->x,fine_level->Ax,fine_level->S.eps);

                                fine_level->r = fine_level->b - fine_level->Ax;


                                //restrict
                                fine_level->R(fine_level->r,(*coarse_levels)[level].b);

                                //recursive call
                                (*coarse_levels)[level].x.zeros();
                                cycle(level+1);

                                //prolong the error to the fine level
                                fine_level->P((*coarse_levels)[level].x,fine_level->e);


                                //add it to the current solution on the fine level
                                fine_level->x = fine_level->x + fine_level->e;

                                //post smoothing
                                fine_level->S(fine_level->b,fine_level->x);


                        } else {

                                //pre smooth
                                (*coarse_levels)[level-1].S((*coarse_levels)[level-1].b,(*coarse_levels)[level-1].x);

                                //compute residual
                                (*coarse_levels)[level-1].A((*coarse_levels)[level-1].x,(*coarse_levels)[level-1].Ax,(*coarse_levels)[level-1].weights);
                                (*coarse_levels)[level-1].r = (*coarse_levels)[level-1].b - (*coarse_levels)[level-1].Ax;


                                //restrict to the next coarsest level
                                (*coarse_levels)[level].b.zeros();
                                (*coarse_levels)[level-1].R((*coarse_levels)[level-1].r,(*coarse_levels)[level].b);

                                //recursive call
                                (*coarse_levels)[level].x.zeros();
                                cycle(level+1);

                                //prolong the error to the next level
                                (*coarse_levels)[level-1].P((*coarse_levels)[level].x,(*coarse_levels)[level-1].e);


                                //add it to the current solution
                                (*coarse_levels)[level-1].x = (*coarse_levels)[level-1].x + (*coarse_levels)[level-1].e;


                                //post smoothing
                                (*coarse_levels)[level-1].S((*coarse_levels)[level-1].b,(*coarse_levels)[level-1].x);


                        }

                }

                void operator()(int max_iters, T target)
                {

                        int iter = 1;
                        int level = 0;
                        T rn = std::sqrt(fine_level->b.norm2());
                        while (iter <= max_iters && rn >= target) {
                                cycle(level);
                                iter += 1;
                                fine_level->A(fine_level->x,fine_level->Ax,fine_level->S.eps);
                                fine_level->r = fine_level->b - fine_level->Ax;
                                rn = std::sqrt(fine_level->r.norm2());
                        }
                        std::cout << "Vcycle finished in " << iter-1 << " iterations with ||r||_2 = " << rn << std::endl;

                }



};

}
