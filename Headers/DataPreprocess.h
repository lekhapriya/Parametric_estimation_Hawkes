#ifndef DATAPREPROCESS_H_INCLUDED
#define DATAPREPROCESS_H_INCLUDED

#include "Basic.h"

class DataPreprocess: public Basic
{
protected:
    vector<vector <double> > events;
    ArrayXXd alpha;
    ArrayXXd beta;
    ArrayXd baseline;
    double T;
    vector<ArrayXXi> mapT1T2;

public:
    DataPreprocess(vector<vector <double> > a);

    // number of dimensions
    void getNodesSize(int *nodes);

    //value of m in (mxn) matrix
    void getParameterSize(int *no_params);

    VectorXi getOtherNodes(int current_node);

    //total number of events in all dimensions
    int totalEventSize();

    //value of T
    void maxEvent(double *T);

    //array of 2 rows: one row for type of event, one row for the index
    ArrayXXi compressedArray();

    vector<int> mapAtoBIndex1(VectorXd a, VectorXd b); //couldn't find range based loop solution for arrayXd and vectorXd

    ArrayXi mapAtoBIndex2(VectorXd a, VectorXd b);

    vector<int> mapAtoBIndex3(vector<double> a, vector<double> b);

    void mapIndex(vector<ArrayXXi> &rmapT1T2); //map index for given events

};

#endif // DATAPREPROCESS_H_INCLUDED
