#ifndef OPTIMIZER_H_INCLUDED
#define OPTIMIZER_H_INCLUDED


#include "basic.h"

class optimizer
{

private:
    double lr;
    double beta_1;
    double beta_2;
    double epsilon;
    int count_;

public:
    optimizer();

    friend class ParametricEst;

};


#endif // OPTIMIZER_H_INCLUDED
