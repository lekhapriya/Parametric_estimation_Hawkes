#include "ParametricEst.h"

ParametricEst::ParametricEst(vector<vector <double> >  events, int b): DataPreprocess(events),epochs(b), bestll (1e8) {}

ArrayXXi ParametricEst::get_adjacentKernel()
{
    ArrayXXi adjacentKernel(2,2);
    adjacentKernel(0,0) = 0; adjacentKernel(0,1) = 1;
    adjacentKernel(1,0) = 3; adjacentKernel(1,1) = 2;
    return adjacentKernel;
}

ArrayXd ParametricEst::exponentialIntegratedKernel(ArrayXd tend, int kernel)
{
    double alpha = params.coeffRef(kernel,0);
    double beta = params.coeffRef(kernel,1);
    ArrayXd res = (alpha/beta)*(1-exp(-beta*tend));
    return res;
}

ArrayXd ParametricEst::exponentialKernel(ArrayXd tp, int kernel)
{
    double alpha = params.coeffRef(kernel,0);
    double beta = params.coeffRef(kernel,1);
    ArrayXd res = alpha*exp(-beta*tp);
    return res;
}

ArrayXd ParametricEst::gradientIntegratedKernel(double tend_i, int p)
{
    double alpha = params.coeffRef(p,0);
    double beta = params.coeffRef(p,1);
    double fac1 = exp(-beta*tend_i);
    ArrayXd val = ArrayXd::Zero(2);
    val(0)=(1/beta)*(1-fac1);
    val(1) = (alpha/beta)*(-(1/beta)+fac1*((1/beta)+tend_i));
    return val;
}

ArrayXd ParametricEst::gradientExpKernel(ArrayXd temp, int p)
{
    double alpha = params.coeffRef(p,0);
    double beta = params.coeffRef(p,1);
    ArrayXd fac1 = exp(-beta*temp);
    ArrayXd val = ArrayXd::Zero(2);
    val(0)= fac1.sum();
    val(1)= -alpha * (temp.cwiseProduct(fac1)).sum();
    return val;
}

ArrayXXd ParametricEst::gradientll(ArrayXXi iArray)
{
    ArrayXXd grad = ArrayXXd::Zero(no_of_params, no_of_nodes);
    for (int i = 0; i < iArray.cols(); ++i)   // arr.cols() number of columns
    {
        ArrayXd factor1 = ArrayXd::Zero(2);
        ArrayXd factor2 = ArrayXd::Zero(2);
        int p = iArray(0,i);
        int index = iArray(1,i);
        double tend = T - events[p][index];
        grad.row(p) = grad.row(p) + gradientIntegratedKernel(tend,p).transpose();
        grad.row(p+2) = grad.row(p+2) + gradientIntegratedKernel(tend,p+2).transpose();
        int li = max(index-30,0);
        Map<ArrayXd> tp(events[p].data(),events[p].size());
        ArrayXd temp1 = tp(index)-tp.segment(li, index-li);
        double decayFactor = exponentialKernel(temp1,adjacentKernel(p,0)).sum();
        factor1 =  gradientExpKernel(temp1,adjacentKernel(p,0));
        int jT = MapT1T2[p][index];
        int otherP = (p==0)*1;
        Map<ArrayXd> tOtherP(events[otherP].data(),events[otherP].size());
        if(jT!= -1){
            int lj = max(jT-30,0);
            ArrayXd temp2 = tp(index)-tOtherP.segment(lj, (jT+1)-lj);
            decayFactor += exponentialKernel(temp2,adjacentKernel(p,1)).sum();
            factor2 = gradientExpKernel(temp2,adjacentKernel(p,1));
        }
        double mu_p = params.coeffRef(no_of_params-1,p);
        double lam = mu_p + decayFactor;
        grad.row(adjacentKernel(p,0)) = grad.row(adjacentKernel(p,0)) - (1/lam)*factor1.transpose();
        grad.row(adjacentKernel(p,1)) = grad.row(adjacentKernel(p,1)) - (1/lam)*factor2.transpose();

        //to calculate mu
        if (index>0)
            grad(grad.rows()-1,p)  = grad(grad.rows()-1,p) + (tp(index)-tp(index-1))- (1/lam);
    }

    ArrayXXd res_grad = grad / iArray.cols();
    return res_grad;
}


double ParametricEst::loss()
{
    double ll = 0;
    for (int p = 0; p < no_of_nodes; ++p)
    {
        Map<ArrayXd> tp(events[p].data(),events[p].size());
        ArrayXd tend = T - tp;
        double a = exponentialIntegratedKernel(tend, p).sum();
        a = a + exponentialKernel(tp, p+2).sum();

        double mu_p = params.coeffRef(no_of_params-1,p);
        ll = ll + mu_p * T + a;
        ll = ll - log(mu_p);
        int otherP = (p==0)*1;
        Map<ArrayXd> tOtherP(events[otherP].data(),events[otherP].size());

        for (int i = 1; i < tp.size(); ++i)
        {
            int li = max(i-30,0);
            ArrayXd temp1 = tp.coeff(i)-tp.segment(li, i-li);
            double decayFactor = exponentialKernel(temp1,adjacentKernel(p,0)).sum();
            int jT = MapT1T2[p][i];

            if(jT!= -1){
                int lj = max(jT-30,0);
                ArrayXd temp2 = tp.coeff(i)-tOtherP.segment(lj, (jT+1)-lj);
                decayFactor += exponentialKernel(temp2,adjacentKernel(p,1)).sum();
            }
            double logLam = - log(mu_p+decayFactor);
            ll = ll + logLam;
        }
    }

    return ll;
}

void ParametricEst::fit()
{
    optimizer Adam;
    params = initialize_para();
    no_of_nodes = get_nodes();
    no_of_params = get_no_of_params();
    T = max_event();
    adjacentKernel = get_adjacentKernel();
    MapT1T2 =  map_index();
    int totalLength = total_events();

    ArrayXXd bestpara = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd m_t = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd v_t = ArrayXXd::Zero(no_of_params, no_of_nodes);

    double error;

    ArrayXXi tcompressed = compressed_array();

    int batch_size = 30;

    for (int m = 1; m<epochs; ++m)
    {
        VectorXd rsample = random_sample(totalLength);
        int range = rsample.size() - (rsample.size() % batch_size); //range for for-loop below
        for (int i = 0; i < range; i = i + batch_size)
        {
            Adam.count_ +=1;
            ArrayXXd grad = gradientll(tcompressed(all,rsample.segment(i, batch_size))); //'all' is only available in eigen 3.3.90 (unstable version)
            m_t = Adam.beta_1*m_t + (1-Adam.beta_1)*grad;
            v_t = Adam.beta_2*v_t + (1-Adam.beta_2)*(grad.cwiseProduct(grad));
            ArrayXXd m_cap = m_t/(1 - pow(Adam.beta_1,Adam.count_));
            ArrayXXd v_cap = v_t/(1 - pow(Adam.beta_2,Adam.count_));
            params = params-(Adam.lr*m_cap) / (v_cap.cwiseSqrt() + Adam.epsilon);

            error = loss();
            bestpara = params*(bestll>=error)+bestpara*(bestll<error);
            params = params.max(0);
            bestll = std::min(bestll,error);

        }
        cout<<"epoch:"<<m<<endl;
        //cout<<"bestpara:"<<bestpara<<endl;
        cout<<"error:"<<error<<endl;
        cout<<"bestll:"<<bestll<<endl;
        cout<<"para:"<<params<<endl;
    }


}
