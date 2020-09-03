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

    ArrayXd integratedKernelsOfAllExcitations(ArrayXd tend, int kernel);

    ArrayXd exponentialKernel(ArrayXd tp, int kernel_row, int kernel_col);

    ArrayXXd gradientIntegratedKernel(double tend_i, int p);

    ArrayXd gradientExpKernel(ArrayXd temp,int kernel_row, int kernel_col);

    void gradientLogLikelihood(ArrayXXi iArray,ArrayXXd &grad_alpha,ArrayXXd &grad_beta,ArrayXd &grad_mu);

    void computeLoss();

    void fit();

    double score();

    int nodes();

    ArrayXd baseline_intensity();

    ArrayXXd adjacency();

    ArrayXXd decays();
};


#endif // PARAMETRICEST_H_INCLUDED
