#include "simuHawkes.h"
#include "ParametricEst.h"

int main()
{
    //input
    VectorXd mu(2);
    ArrayXXd alpha(2,2);
    ArrayXXd beta(2,2);
    int T = 10000;
    mu<<0.12, 0.07;
    alpha<< .3, 0.,.6, .21;
    beta<<4., 1., 2., 2.;

    simuHawkes hawkes(alpha, beta, mu, T);
    vector<vector<double> > t = hawkes.get_timestamp();

    ParametricEst Obj2(t,30);
    Obj2.fit();

    return 0;
}
