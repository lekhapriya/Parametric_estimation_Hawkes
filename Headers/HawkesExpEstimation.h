#ifndef PARAMETRICEST_H_INCLUDED
#define PARAMETRICEST_H_INCLUDED

#include "DataPreprocess.h"
#include "Optimizer.h"

class HawkesExpEstimation: public DataPreprocess
{
private:
    int epochs;
    double bestll;
    double likelihood;

public:
    HawkesExpEstimation(vector<vector <double> >  events, int b);

    void getAdjacentKernel(ArrayXXi &adjacentKernel);

    ArrayXd exponentialIntegratedKernel(ArrayXd tend, int kernel);

    ArrayXd exponentialKernel(ArrayXd tp, int kernel);

    ArrayXd gradientIntegratedKernel(double tend_i, int p);

    ArrayXd gradientExpKernel(ArrayXd temp, int p);

    void gradientLogLikelihood(ArrayXXi iArray,ArrayXXi &adjacentKernel,ArrayXXd &grad);

    void computeLoss(ArrayXXi &adjacentKernel);

    void fit();

    double score();

    int nodes();

    ArrayXXd baseline();

    ArrayXXd adjacency();

    ArrayXXd decays();
};


#endif // PARAMETRICEST_H_INCLUDED
