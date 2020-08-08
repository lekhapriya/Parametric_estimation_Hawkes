#include "HawkesExpEstimation.h"

HawkesExpEstimation::HawkesExpEstimation(vector<vector <double> >  events, int b): DataPreprocess(events),epochs(b), bestll (1e8) {}

void HawkesExpEstimation::getAdjacentKernel(ArrayXXi &adjacentKernel)
{
    adjacentKernel(0,0) = 0; adjacentKernel(0,1) = 1;
    adjacentKernel(1,0) = 3; adjacentKernel(1,1) = 2;
    return;
}

ArrayXd HawkesExpEstimation::exponentialIntegratedKernel(ArrayXd tend, int kernel)
{
    double alpha = params.coeffRef(kernel,0);
    double beta = params.coeffRef(kernel,1);
    ArrayXd res = (alpha/beta)*(1-exp(-beta*tend));
    return res;
}

ArrayXd HawkesExpEstimation::exponentialKernel(ArrayXd tp, int kernel)
{
    double alpha = params.coeffRef(kernel,0);
    double beta = params.coeffRef(kernel,1);
    ArrayXd res = alpha*exp(-beta*tp);
    return res;
}

ArrayXd HawkesExpEstimation::gradientIntegratedKernel(double tend_i, int p)
{
    double alpha = params.coeffRef(p,0);
    double beta = params.coeffRef(p,1);
    double fac1 = exp(-beta*tend_i);
    ArrayXd val = ArrayXd::Zero(2);
    val(0)=(1/beta)*(1-fac1);
    val(1) = (alpha/beta)*(-(1/beta)+fac1*((1/beta)+tend_i));
    return val;
}

ArrayXd HawkesExpEstimation::gradientExpKernel(ArrayXd temp, int p)
{
    double alpha = params.coeffRef(p,0);
    double beta = params.coeffRef(p,1);
    ArrayXd fac1 = exp(-beta*temp);
    ArrayXd val = ArrayXd::Zero(2);
    val(0)= fac1.sum();
    val(1)= -alpha * (temp.cwiseProduct(fac1)).sum();
    return val;
}

void HawkesExpEstimation::gradientLogLikelihood(ArrayXXi iArray,ArrayXXi &adjacentKernel,ArrayXXd &grad)
{
    resetArray(grad);
    int p;
    int index;
    double tend;
    for (int i = 0; i < iArray.cols(); ++i)   // arr.cols() number of columns
    {
        ArrayXd factor1 = ArrayXd::Zero(2);
        ArrayXd factor2 = ArrayXd::Zero(2);
        p = iArray(0,i);
        index = iArray(1,i);
        tend = T - events[p][index];
        grad.row(p) = grad.row(p) + gradientIntegratedKernel(tend,p).transpose();
        grad.row(p+2) = grad.row(p+2) + gradientIntegratedKernel(tend,p+2).transpose();
        int li = max(index-30,0);
        Map<ArrayXd> tp(events[p].data(),events[p].size());
        ArrayXd temp1 = tp(index)-tp.segment(li, index-li);
        double decayFactor = exponentialKernel(temp1,adjacentKernel(p,0)).sum();
        factor1 =  gradientExpKernel(temp1,adjacentKernel(p,0));
        int jT = mapT1T2[p][index];
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

    grad = grad / iArray.cols();
    return;
}


void HawkesExpEstimation::computeLoss(ArrayXXi &adjacentKernel)
{
    likelihood = 0;
    for (int p = 0; p < no_of_nodes; ++p)
    {
        Map<ArrayXd> tp(events[p].data(),events[p].size());
        ArrayXd tend = T - tp;
        double a = exponentialIntegratedKernel(tend, p).sum();
        a = a + exponentialKernel(tp, p+2).sum();

        double mu_p = params.coeffRef(no_of_params-1,p);
        likelihood = likelihood + mu_p * T + a;
        likelihood = likelihood - log(mu_p);
        int otherP = (p==0)*1;
        Map<ArrayXd> tOtherP(events[otherP].data(),events[otherP].size());

        for (int i = 1; i < tp.size(); ++i)
        {
            int li = max(i-30,0);
            ArrayXd temp1 = tp.coeff(i)-tp.segment(li, i-li);
            double decayFactor = exponentialKernel(temp1,adjacentKernel(p,0)).sum();
            int jT = mapT1T2[p][i];

            if(jT!= -1){
                int lj = max(jT-30,0);
                ArrayXd temp2 = tp.coeff(i)-tOtherP.segment(lj, (jT+1)-lj);
                decayFactor += exponentialKernel(temp2,adjacentKernel(p,1)).sum();
            }
            double logLam = - log(mu_p+decayFactor);
            likelihood = likelihood + logLam;
        }
    }

    return;
}

void HawkesExpEstimation::fit()
{
    Optimizer Adam;

    getNodesSize(&no_of_nodes);
    getParameterSize(&no_of_params);
    initializePara(params);
    T = maxEvent();
    ArrayXXi adjacentKernel(no_of_nodes,no_of_nodes);
    getAdjacentKernel(adjacentKernel);
    mapIndex(mapT1T2);
    int totalLength = totalEventSize();

    ArrayXXd grad = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd bestpara = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd m_t = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd v_t = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd m_cap = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd v_cap = ArrayXXd::Zero(no_of_params, no_of_nodes);

    ArrayXXi tcompressed = compressedArray();

    int batch_size = 30;

    for (int m = 1; m<epochs; ++m)
    {
        VectorXd rsample = getRandomSamples(totalLength);
        int range = rsample.size() - (rsample.size() % batch_size); //range for for-loop below
        for (int i = 0; i < range; i = i + batch_size)
        {
            Adam.count_ +=1;
            gradientLogLikelihood(tcompressed(all,rsample.segment(i, batch_size)),adjacentKernel,grad); //'all' is only available in eigen 3.3.90 (unstable version)
            m_t = Adam.beta_1*m_t + (1-Adam.beta_1)*grad;
            v_t = Adam.beta_2*v_t + (1-Adam.beta_2)*(grad.cwiseProduct(grad));
            m_cap = m_t/(1 - pow(Adam.beta_1,Adam.count_));
            v_cap = v_t/(1 - pow(Adam.beta_2,Adam.count_));
            params = params-(Adam.lr*m_cap) / (v_cap.cwiseSqrt() + Adam.epsilon);

            computeLoss(adjacentKernel);
            bestpara = params*(bestll>=likelihood)+bestpara*(bestll<likelihood);
            params = params.max(0);
            bestll = std::min(bestll,likelihood);

        }
        //cout<<"epoch:"<<m<<endl;
        //cout<<"bestpara:"<<bestpara<<endl;
        //cout<<"error:"<<likelihood<<endl;
        //cout<<"bestll:"<<bestll<<endl;
        //cout<<"para:"<<params<<endl;
    }


}

double HawkesExpEstimation::score()
{
    cout<<"score(Neg-Loglikelihood):"<<likelihood<<endl;
    return likelihood;
}

int HawkesExpEstimation::nodes()
{
    cout<<"Number of nodes :"<< no_of_nodes<<endl;
    return no_of_nodes;
}

ArrayXXd HawkesExpEstimation::baseline()
{
    cout<< "Mu :"<< params.bottomRows(1)<<endl;
    return params.bottomRows(1);
}

ArrayXXd HawkesExpEstimation::adjacency()
{

    Map<ArrayXXd> alpha(params.block(0, 0, no_of_params-1, 1).data(),no_of_nodes,no_of_nodes);
    cout<<"alpha:"<<alpha<<endl;;
    return alpha;
}

ArrayXXd HawkesExpEstimation::decays()
{

    Map<ArrayXXd> beta(params.block(0, 1, no_of_params-1, 1).data(),no_of_nodes,no_of_nodes);
    cout<<"beta:"<<beta<<endl;;
    return beta;
}

