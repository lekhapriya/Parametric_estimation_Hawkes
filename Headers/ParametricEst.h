#ifndef PARAMETRICEST_H_INCLUDED
#define PARAMETRICEST_H_INCLUDED

#include "DataPreprocess.h"
#include "optimizer.h"

class ParametricEst: public DataPreprocess
{
private:
    int epochs;
    double bestll;
    ArrayXXi adjacentKernel;

public:
    //ParametricEst():DataPreprocess(){};
    ParametricEst(vector<vector <double> >  events, int b);

    ArrayXXi get_adjacentKernel();

    ArrayXd exponentialIntegratedKernel(ArrayXd tend, int kernel);

    ArrayXd exponentialKernel(ArrayXd tp, int kernel);

    ArrayXd gradientIntegratedKernel(double tend_i, int p);

    ArrayXd gradientExpKernel(ArrayXd temp, int p);

    ArrayXXd gradientll(ArrayXXi iArray);

    double loss();

    void fit();
};


#endif // PARAMETRICEST_H_INCLUDED
