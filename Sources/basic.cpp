#include "basic.h"

double basic::get_random()
{
    static std::default_random_engine rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<double> dis(0, 1); // range 0 - 1
    return dis(e2);

}


std::vector<int> basic::random_sample(std::vector<double> t)
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

std::vector<double> basic::init_to_zero()
{
    int no_of_param = 3; //based on no. of nodes for multivariate (n+n*2)
    std::vector<double> init_param;
    for (int i = 0; i < no_of_param; ++i)
    {
        init_param.push_back(0);
    }
    return init_param;
}

