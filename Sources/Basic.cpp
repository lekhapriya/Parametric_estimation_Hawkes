#include "Basic.h"

double Basic::getRandomNum()
{
    static std::default_random_engine rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<double> dis(0, 1); // range 0 - 1
    return dis(e2);

}


std::vector<int> Basic::getRandomSamples(std::vector<double> t)
{
    std::vector<int> samples;
    int length_t = t.size();
    int sample_size = length_t * 50;
    std::default_random_engine rd;
    std::uniform_int_distribution<int> dis(0, (length_t-1));
    for (int i = 0; i < sample_size; ++i)
    {
        samples.push_back(dis(rd));
    }
    return samples;
}

void Basic::initializePara(ArrayXXd &arr)
{
    //params.setRandom(no_of_params, no_of_nodes); //setRandom [-1,1]
    arr = 0+(ArrayXXd::Random(no_of_params, no_of_nodes)*0.5+0.5)*(1-0); //[0,1] [low high]
    //cout<<params<<endl;
    return;
}

void Basic::resetArray(ArrayXXd &arr)
{
    arr.setZero();
}

VectorXd Basic::getRandomSamples(int length)
{
    // This method generates repetitions
    //std::vector<int> samples;
    //std::default_random_engine rd;
    //std::uniform_int_distribution<int> dis(0, (length-1));
    //for (int i = 0; i < length; ++i)
    //{
        //samples.push_back(dis(rd));
    //}

//No repetitions, but unnecessary eigen to std conversions
    //srand(time(0)); //set seed

    VectorXd samples = VectorXd::LinSpaced(length,0, length-1);
    vector<double> v2;
    v2.resize(samples.size());
    VectorXd::Map(&v2[0], samples.size()) = samples; //eigen to std
    std::random_shuffle ( v2.begin(), v2.end() );
    Map<VectorXd> v1(v2.data(),v2.size()); //std to eigen
    return v1;
}
