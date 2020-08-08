#include "SimuHawkes.h"
#include "HawkesExpEstimation.h"

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

    SimuHawkes hawkes(alpha, beta, mu, T);
    hawkes.simulate();
    //Hawkes.reset();

    HawkesExpEstimation hawkesExpKernel(hawkes.timestamp(),30);

    hawkesExpKernel.fit();
    //HawkesExpKernel.score();
    hawkesExpKernel.nodes();
    hawkesExpKernel.baseline();
    hawkesExpKernel.adjacency();
    hawkesExpKernel.decays();

    return 0;
}
