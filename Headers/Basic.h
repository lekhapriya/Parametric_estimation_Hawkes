#ifndef BASIC_H_INCLUDED
#define BASIC_H_INCLUDED

#include <iostream>
#include <cmath>
#include <random>
#include <list>
#include <vector>
#include<bits/stdc++.h>
#include<algorithm>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Basic
{
public:
//uniform random number
//if you don't want the numbers to be the same all the time, set the engine and distribution to be static
    double getRandomNum();
    int no_of_nodes;
    int no_of_params;

//generate randome samples from a given timestamp
    std::vector<int> getRandomSamples(std::vector<double> t);

    void initializePara(ArrayXXd &arr);

    void resetArray(ArrayXXd &arr);

    //np.random.choice in python
    VectorXd getRandomSamples(int length);

};

#endif // BASIC_H_INCLUDED

