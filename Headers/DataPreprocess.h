#ifndef DATAPREPROCESS_H_INCLUDED
#define DATAPREPROCESS_H_INCLUDED

#include "Basic.h"

class DataPreprocess: public Basic
{
protected:
    vector<vector <double> > events;
    ArrayXXd params;
    double T;
    vector<ArrayXi> mapT1T2;

public:
    DataPreprocess(vector<vector <double> > a);

    // number of dimensions
    void getNodesSize(int *nodes);

    //value of m in (mxn) matrix
    void getParameterSize(int *no_params);

    //total number of events in all dimensions
    int totalEventSize();

    //value of T
    double maxEvent();

    //array of 2 rows: one row for type of event, one row for the index
    ArrayXXi compressedArray();

    vector<int> mapAtoBIndex1(VectorXd a, VectorXd b); //couldn't find range based loop solution for arrayXd and vectorXd

    ArrayXi mapAtoBIndex2(VectorXd a, VectorXd b);

    vector<int> mapAtoBIndex3(vector<double> a, vector<double> b);

    void mapIndex(vector<ArrayXi > &rmapT1T2); //map index for given events

};

#endif // DATAPREPROCESS_H_INCLUDED
