#ifndef BASIC_H_INCLUDED
#define BASIC_H_INCLUDED

#include <iostream>
#include <cmath>
#include <random>
#include <list>
#include <vector>
#include<algorithm>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class basic
{
public:
//uniform random number
//if you don't want the numbers to be the same all the time, set the engine and distribution to be static
    double get_random();

//generate randome samples from a given timestamp
    std::vector<int> random_sample(std::vector<double> t);

//initialize to zero
    std::vector<double> init_to_zero();

};

#endif // BASIC_H_INCLUDED

