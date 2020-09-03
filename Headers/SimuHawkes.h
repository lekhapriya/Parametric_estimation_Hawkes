#ifndef SIMUHAWKES_H_INCLUDED
#define SIMUHAWKES_H_INCLUDED

#include "Basic.h"

class SimuHawkes: public Basic
{
private:
    ArrayXXd adjacency;
    ArrayXXd decays;
    VectorXd baseline;
    int end_time;
    vector<vector<double> > tau;
    int nodes;

public:
//set
    SimuHawkes(ArrayXXd alpha,ArrayXXd beta, VectorXd mu, int T);

    void getNodes(int *result);

    void getIntensity(double s,double *result);

    void simulate();

    void reset();

//get
    vector<vector<double> > timestamp();
};
#endif // SIMUHAWKES_H_INCLUDED
