#include "SimuHawkes.h"

SimuHawkes::SimuHawkes(ArrayXXd alpha,ArrayXXd beta, VectorXd mu, int T) : adjacency(alpha),decays(beta),baseline(mu),end_time(T){}

void SimuHawkes::getNodes(int *result)
{
   *result = baseline.size();
}

void SimuHawkes::getIntensity(double s,double *result)
{
    for (int i=0; i< nodes; i++)
    {
        double sum = 0;
        for (int j=0; j<nodes; j++)
        {//vector<double> tau_n( tau[j].rbegin(), tau[j].rend()); //for the std case, generating reverse tau
            if (tau[j].empty())
                sum = 0;
            else
            {
                Map<ArrayXd> tau_n(tau[j].data(),tau[j].size());
                ArrayXd temp = (-decays(i,j)*(s - tau_n.reverse())).exp() * adjacency(i,j); //a_(m,n) e^b_(m,n)(s-tau)
                sum += temp.sum();
            }
        }
        *result += baseline(i)+sum;
    }

}

void SimuHawkes::simulate()
{
    //initialise tau, n ,s
    getNodes(&nodes);
    VectorXd n = VectorXd::Zero(nodes);
    double s = 0.0;

    std::vector<double> v{}; //append 0
    for (int i=0; i< nodes; i++)
        tau.push_back(v);

    double lambda_bar;
    double lambda_s;

    while (s<end_time)
    {
        lambda_bar = 0;
        getIntensity(s,&lambda_bar);

        double u = getRandomNum();
        double w= -log(u)/lambda_bar;
        s=s+w;
        double d = getRandomNum();

        //find lambda_s
        lambda_s = 0;
        getIntensity(s,&lambda_s);

        //validate s
        if ((d*lambda_bar)<=lambda_s)
        {
            int k = 1;
            double lambda_dim = 0.0;
            // searching for the first k,such that d*lambda_ <= lambda_dimension k
            for (int i=0; i< k; i++)
            {
                double sum = 0;
                for (int j=0; j<nodes; j++)
                {
                    if (tau[j].empty())
                        sum = 0;
                    else
                    {
                        Map<ArrayXd> tau_n(tau[j].data(),tau[j].size());
                        ArrayXd temp = (-decays(i,j)*(s - tau_n.reverse())).exp() * adjacency(i,j); //a_(m,n) e^b_(m,n)(s-tau)
                        sum += temp.sum();
                    }
                }
                lambda_dim = lambda_dim + baseline(i)+sum;
                if ((d*lambda_bar) > lambda_dim)
                    k = k+1;
            }
            n(k-1) += 1; // updating the number of points in dimension k
            tau[k-1].push_back(s); //adding s to the ordered set in tau
            //for(auto e : tau[k-1]) std::cout << e;//to print vector
        }
    }

}

void SimuHawkes::reset()
{
    tau.clear();
    nodes = 0;
}

vector<vector<double> > SimuHawkes::timestamp()

{
    //print tau
    //for (int i = 0; i < tau.size(); i++)
    //{
     //   for (int j = 0; j < tau[i].size(); j++)
       //     cout << tau[i][j] << " ";
        //cout << endl;
    //}
   return tau;
}
