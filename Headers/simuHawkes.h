#ifndef SIMUHAWKES_H_INCLUDED
#define SIMUHAWKES_H_INCLUDED

#include "basic.h"

class simuHawkes: public basic
{
private:
    ArrayXXd alpha;
    ArrayXXd beta;
    VectorXd mu;
    int T;
    vector<vector<double> > timestamp;

public:
//set
    simuHawkes(ArrayXXd x,ArrayXXd y, VectorXd z, int w);
//get
    vector<vector<double> > get_timestamp();
};
#endif // SIMUHAWKES_H_INCLUDED
