#ifndef OPTIMIZER_H_INCLUDED
#define OPTIMIZER_H_INCLUDED


#include "Basic.h"

class Optimizer
{

private:
    double lr;
    double beta_1;
    double beta_2;
    double epsilon;
    int count_;

public:
    Optimizer();

    friend class HawkesExpEstimation;

};


#endif // OPTIMIZER_H_INCLUDED
