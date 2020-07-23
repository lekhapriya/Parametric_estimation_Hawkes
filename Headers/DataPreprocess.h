#ifndef DATAPREPROCESS_H_INCLUDED
#define DATAPREPROCESS_H_INCLUDED

#include "basic.h"

class DataPreprocess: public basic
{
protected:
    vector<vector <double> > events;
    ArrayXXd params;
    int no_of_nodes;
    int no_of_params;
    double T;
    vector<vector<int> > MapT1T2;

public:
    DataPreprocess(vector<vector <double> > a);

    // number of dimensions
    int get_nodes();

    //value of m in (mxn) matrix
    int get_no_of_params();

    //total number of events in all dimensions
    int total_events();

    //value of T
    double max_event();

    //array of 2 rows: one row for type of event, one row for the index
    ArrayXXi compressed_array();

    vector<int> MapAtoBIndex(VectorXd a, VectorXd b); //couldn't find range based loop solution for arrayXd and vectorXd

    vector<vector<int> > map_index(); //map index for given events

    ArrayXXd initialize_para();

    //np.random.choice in python
    VectorXd random_sample(int length);

};

#endif // DATAPREPROCESS_H_INCLUDED
