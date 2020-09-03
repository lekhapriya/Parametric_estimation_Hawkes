#include "HawkesExpEstimation.h"

HawkesExpEstimation::HawkesExpEstimation(vector<vector <double> >  events, int b): DataPreprocess(events),epochs(b), bestll (1e8) {}


ArrayXd HawkesExpEstimation::integratedKernelsOfAllExcitations(ArrayXd tend, int kernel)
{
    ArrayXd alpha_current_node = alpha.row(kernel);
    ArrayXd beta_current_node = beta.row(kernel);
    ArrayXd temp;
    temp = ArrayXd::Zero(tend.size());
    for (int i = 0; i < alpha_current_node.size(); ++i)
        temp = temp + (alpha_current_node(i)/beta_current_node(i))*(1-exp(-beta_current_node(i)*tend));
    return temp;
}



ArrayXd HawkesExpEstimation::exponentialKernel(ArrayXd tp, int kernel_row, int kernel_col )
{
    double alpha_current_node = alpha(kernel_row,kernel_col);
    double beta_current_node = beta(kernel_row,kernel_col);
    ArrayXd res = alpha_current_node*exp(-beta_current_node*tp);
    return res;
}

ArrayXXd HawkesExpEstimation::gradientIntegratedKernel(double tend_i, int kernel)
{
    ArrayXd alpha_current_node = alpha.row(kernel);
    ArrayXd beta_current_node = beta.row(kernel);
    ArrayXd fac1 = exp(-beta_current_node*tend_i);
    ArrayXXd val = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    val.row(0)=(1/beta_current_node).cwiseProduct(1-fac1);
    val.row(1) = (alpha_current_node.array()/beta_current_node.array()).cwiseProduct(-(1/beta_current_node)+fac1.array()*((1/beta_current_node)+tend_i).array());
    return val;
}

ArrayXd HawkesExpEstimation::gradientExpKernel(ArrayXd temp,int kernel_row, int kernel_col)
{
    double alpha_current_node = alpha(kernel_row,kernel_col);
    double beta_current_node = beta(kernel_row,kernel_col);
    ArrayXd fac1 = exp(-beta_current_node*temp);
    ArrayXd val = ArrayXd::Zero(2,1);
    val(0)= fac1.sum();
    val(1)= -alpha_current_node * (temp.cwiseProduct(fac1)).sum();
    return val;
}

void HawkesExpEstimation::gradientLogLikelihood(ArrayXXi iArray,ArrayXXd &grad_alpha,ArrayXXd &grad_beta,ArrayXd &grad_mu)
{
    resetArray(grad_alpha);
    resetArray(grad_beta);
    grad_mu.setZero();
    int p;
    int index;
    double tend;
    for (int i = 0; i < iArray.cols(); ++i)   // arr.cols() number of columns
    {
        ArrayXd factor1 = ArrayXd::Zero(2);
        p = iArray(0,i);
        index = iArray(1,i);
        tend = T - events[p][index];
        ArrayXXd val = gradientIntegratedKernel(tend,p);
        grad_alpha.row(p) += val.row(0);
        grad_beta.row(p) += val.row(1) ;
        int li = max(index-30,0);
        Map<ArrayXd> tp(events[p].data(),events[p].size());
        ArrayXd temp1 = tp(index)-tp.segment(li, index-li);
        double decayFactor = exponentialKernel(temp1,p,p).sum();
        factor1 =  gradientExpKernel(temp1,p,p);
        VectorXi other_nodes = getOtherNodes(p);
        ArrayXXd factor2 = ArrayXXd::Zero(other_nodes.size(),2);
        for (int j = 0; j < other_nodes.size(); ++j)
        {
            int jT = mapT1T2[p].coeff(index,j);
            if(jT!= -1)
            {
                int lj = max(jT-30,0);
                Map<ArrayXd> tOtherP(events[other_nodes[j]].data(),events[other_nodes[j]].size());
                ArrayXd temp2 = tp.coeff(index)-tOtherP.segment(lj, (jT+1)-lj);
                decayFactor += exponentialKernel(temp2,other_nodes[j],p).sum();
                factor2.row(j) = gradientExpKernel(temp2,other_nodes[j],p);
            }
        }

        double mu_p = baseline(p);
        double lam = mu_p + decayFactor;
        factor1 = (1/lam)*factor1;
        factor2 = (1/lam)*factor2;
        grad_alpha(p,p) = grad_alpha(p,p) - factor1(0);
        grad_beta(p,p) = grad_beta(p,p) - factor1(1);

        for (int j = 0; j < other_nodes.size(); ++j)
        {
            grad_alpha(other_nodes[j],p) = grad_alpha(other_nodes[j],p) - factor2(j,0);
            grad_beta(other_nodes[j],p) = grad_beta(other_nodes[j],p) - factor2(j,1);
        }


        //to calculate mu
        if (index>0)
            grad_mu(p)  = grad_mu(p) + (tp(index)-tp(index-1))- (1/lam);
    }

    grad_alpha = grad_alpha / iArray.cols();
    grad_beta = grad_beta / iArray.cols();
    grad_mu = grad_mu / iArray.cols();
    return;
}


void HawkesExpEstimation::computeLoss()
{
    likelihood = 0;
    for (int p = 0; p < no_of_nodes; ++p)
    {
        Map<ArrayXd> tp(events[p].data(),events[p].size());
        ArrayXd tend = T - tp;
        double exp_integrated_kernel= integratedKernelsOfAllExcitations(tend, p).sum();

        double mu_p = baseline(p);
        likelihood = likelihood + mu_p * T + exp_integrated_kernel;
        likelihood = likelihood - log(mu_p);

        for (int i = 1; i < tp.size(); ++i)
        {
            int li = max(i-30,0);
            ArrayXd temp1 = tp.coeff(i)-tp.segment(li, i-li);
            double decayFactor = exponentialKernel(temp1,p,p).sum();
            VectorXi other_nodes = getOtherNodes(p);
            for (int j = 0; j < other_nodes.size(); ++j)
            {
                int jT = mapT1T2[p].coeff(i,j);
                if(jT!= -1)
                {
                    int lj = max(jT-30,0);
                    Map<ArrayXd> tOtherP(events[other_nodes[j]].data(),events[other_nodes[j]].size());
                    ArrayXd temp2 = tp.coeff(i)-tOtherP.segment(lj, (jT+1)-lj);
                    decayFactor += exponentialKernel(temp2,other_nodes[j],p).sum();
                }
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
    initializeParams(alpha);
    initializeParams(beta);
    initializeBaseline(baseline);
    maxEvent(&T);

    mapIndex(mapT1T2);
    int totalLength = totalEventSize();

    ArrayXXd grad_alpha = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd grad_beta = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXd grad_mu = ArrayXd::Zero(no_of_nodes);
    //ArrayXXd bestpara = ArrayXXd::Zero(no_of_params, no_of_nodes);
    ArrayXXd m_t_alpha = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd v_t_alpha = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd m_cap_alpha = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd v_cap_alpha = ArrayXXd::Zero(no_of_nodes, no_of_nodes);

    ArrayXXd m_t_beta = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd v_t_beta = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd m_cap_beta = ArrayXXd::Zero(no_of_nodes, no_of_nodes);
    ArrayXXd v_cap_beta = ArrayXXd::Zero(no_of_nodes, no_of_nodes);

    ArrayXd m_t_mu = ArrayXd::Zero(no_of_nodes);
    ArrayXd v_t_mu = ArrayXd::Zero(no_of_nodes);
    ArrayXd m_cap_mu = ArrayXd::Zero(no_of_nodes);
    ArrayXd v_cap_mu = ArrayXd::Zero(no_of_nodes);

    ArrayXXi tcompressed = compressedArray();

    int batch_size = 30;

    for (int m = 1; m<epochs; ++m)
    {
        VectorXd rsample = getRandomSamples(totalLength);
        int range = rsample.size() - (rsample.size() % batch_size); //range for for-loop below
        for (int i = 0; i < range; i = i + batch_size)
        {
            Adam.count_ +=1;
            gradientLogLikelihood(tcompressed(all,rsample.segment(i, batch_size)),grad_alpha,grad_beta, grad_mu); //'all' is only available in eigen 3.3.90 (unstable version)
            m_t_alpha = Adam.beta_1*m_t_alpha + (1-Adam.beta_1)*grad_alpha;
            v_t_alpha = Adam.beta_2*v_t_alpha + (1-Adam.beta_2)*(grad_alpha.cwiseProduct(grad_alpha));
            m_cap_alpha = m_t_alpha/(1 - pow(Adam.beta_1,Adam.count_));
            v_cap_alpha = v_t_alpha/(1 - pow(Adam.beta_2,Adam.count_));
            alpha = alpha-(Adam.lr*m_cap_alpha) / (v_cap_alpha.cwiseSqrt() + Adam.epsilon);

            m_t_beta = Adam.beta_1*m_t_beta + (1-Adam.beta_1)*grad_beta;
            v_t_beta = Adam.beta_2*v_t_beta + (1-Adam.beta_2)*(grad_beta.cwiseProduct(grad_beta));
            m_cap_beta = m_t_beta/(1 - pow(Adam.beta_1,Adam.count_));
            v_cap_beta = v_t_beta/(1 - pow(Adam.beta_2,Adam.count_));
            beta = beta-(Adam.lr*m_cap_beta) / (v_cap_beta.cwiseSqrt() + Adam.epsilon);

            m_t_mu = Adam.beta_1*m_t_mu + (1-Adam.beta_1)*grad_mu;
            v_t_mu = Adam.beta_2*v_t_mu + (1-Adam.beta_2)*(grad_mu.cwiseProduct(grad_mu));
            m_cap_mu = m_t_mu/(1 - pow(Adam.beta_1,Adam.count_));
            v_cap_mu = v_t_mu/(1 - pow(Adam.beta_2,Adam.count_));
            baseline = baseline-(Adam.lr*m_cap_mu) / (v_cap_mu.cwiseSqrt() + Adam.epsilon);

            computeLoss();
            //bestpara = params*(bestll>=likelihood)+bestpara*(bestll<likelihood);
            alpha = alpha.max(0);
            beta = beta.max(0);
            baseline = baseline.max(0);
            bestll = std::min(bestll,likelihood);

        }
        //cout<<"epoch:"<<m<<endl;
        //cout<<"bestpara:"<<bestpara<<endl;
        //cout<<"error:"<<likelihood<<endl;
        //cout<<"bestll:"<<bestll<<endl;
        //cout<<"alpha:"<<alpha<<endl;
        //cout<<"beta:"<<beta<<endl;
        //cout<<"baseline:"<<baseline<<endl;
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

ArrayXd HawkesExpEstimation::baseline_intensity()
{
    cout<< "Mu :"<< baseline<<endl;
    return baseline;
}

ArrayXXd HawkesExpEstimation::adjacency()
{

    cout<<"alpha:"<<alpha.transpose()<<endl;;
    return alpha.transpose();
}

ArrayXXd HawkesExpEstimation::decays()
{
    cout<<"beta:"<<beta.transpose()<<endl;;
    return beta.transpose();
}

