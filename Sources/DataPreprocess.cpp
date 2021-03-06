#include "DataPreprocess.h"


DataPreprocess::DataPreprocess(vector<vector <double> > a) : events(a) {}

void DataPreprocess::getNodesSize(int *nodes)
{
    *nodes =  events.size();
}

void DataPreprocess::getParameterSize(int *no_params)
{
    *no_params= 1+(no_of_nodes*2);
}

VectorXi DataPreprocess::getOtherNodes(int current_node)
{
    int dim[no_of_nodes];
    for (int i = 0; i < no_of_nodes; i++)
        dim[i] = i;
    std::vector<int> other_nodes (no_of_nodes-1);
    //std::remove_copy(std::begin(dim), std::end(dim),other_nodes.begin(), current_node);
    std::remove_copy(dim, dim+no_of_nodes,other_nodes.begin(), current_node);
    Map<VectorXi> res(other_nodes.data(),other_nodes.size());
    return res;
}

int DataPreprocess::totalEventSize()
{
    //int no_of_nodes = events.size();
    vector<int> length_of_t;

    for (int i = 0; i < no_of_nodes; i++)
    {
        length_of_t.push_back(events[i].size());
    }

    int total_length = accumulate(length_of_t.begin(),length_of_t.end(),0);

    //cout<<"total no. of events:"<<total_length<<endl;
    return total_length;
}

void DataPreprocess::maxEvent(double *T)
{
    VectorXd temp(no_of_nodes);
    for (int i = 0; i < no_of_nodes; i++)
    {
         temp[i] = events[i][events[i].size()-1];
    }
    *T =temp.maxCoeff();
    //Map<VectorXd> t1(events[0].data(),events[0].size()); //convert std vector to eigen vector
    //Map<VectorXd> t2(events[1].data(),events[1].size());
    //T = (t1.tail(1)).cwiseMax(t2.tail(1)).coeff(0);
    //T = ((t1.tail(1)).max(t2.tail(1))).coeff(0); //array1.max(array2), if array
    return;
}

ArrayXXi DataPreprocess::compressedArray()
{
    //int no_of_nodes =  events.size();
    int total_length = totalEventSize();

    vector<int> length_of_t;

    for (int i = 0; i < no_of_nodes; i++)
    {
        length_of_t.push_back(events[i].size());
    }


    ArrayXXi tCompressed = ArrayXXi::Zero(2, total_length);

    int temp = 0;
    for (int i = 0; i < no_of_nodes; i++)
    {
        tCompressed.middleCols(temp, length_of_t[i]) = i; //P(:, j+1:j+cols)
        temp += length_of_t[i];
    }

    // LinSpaced(size, low, high)
    ArrayXi index = tCompressed.row(1);
    temp = 0;
    for (int i = 0; i < no_of_nodes; i++)
    {
       index.segment(temp, length_of_t[i]) = ArrayXi::LinSpaced(length_of_t[i], 0, length_of_t[i]-1);
       temp += length_of_t[i];
    }
    tCompressed.row(1) = index;
    //cout << tCompressed << "\n"<<endl;

    return tCompressed;
}

vector<int> DataPreprocess::mapAtoBIndex1(VectorXd a, VectorXd b) //unnecessary std to eigen and eigen to std conversions. change the code
{
    vector <int> v; //need std::vector for nested vector
    ArrayXi::Index max_index;
    ArrayXi mapAtoBIndex = ArrayXi::Zero(a.size());
    int n = 0;
    for(auto x : a)
    {
        ArrayXd result = (b.array() < x).select(b,-1);
        if (result.maxCoeff() == -1)
            mapAtoBIndex(n) = -1;
        else
        {
            result.maxCoeff(&max_index);
            mapAtoBIndex(n) = max_index;
        }
        v.push_back(mapAtoBIndex.coeff(n));
        n += 1;
    }
    return v;
}

ArrayXi DataPreprocess::mapAtoBIndex2(VectorXd a, VectorXd b)
{

    ArrayXi::Index max_index;
    ArrayXi mapAtoBIndex = ArrayXi::Constant(a.size(), -1);
    int n = a.size() - 1;
    for(auto x : a.reverse())
    {
        ArrayXd result = (b.array() < x).select(b,-1);
        if (result.maxCoeff() == -1)
            continue;
        else
        {
            result.maxCoeff(&max_index);
            mapAtoBIndex(n) = max_index;
            b.conservativeResize(max_index+1);
        }
        n--;
    }
    return mapAtoBIndex;
}

vector<int> DataPreprocess::mapAtoBIndex3(vector<double> a, vector<double> b)
{
    vector<int> vect(a.size(), -1);
    for (int i = a.size()-1; i >= 0; i--)
    {
        for (int j = b.size()-1; j >= 0; j--)
        {
            if(a[i] < b[j])
                b.pop_back();
            else
            {
                vect[i]=j;
                break;
            }
        }
    }
    return vect;
}

void DataPreprocess::mapIndex(vector<ArrayXXi> &rmapT1T2)
{

    for (int p = 0; p < no_of_nodes; p++)
    {
        VectorXi other_nodes = getOtherNodes(p);
        Map<VectorXd> t_current(events[p].data(),events[p].size());
        ArrayXXi index_map(t_current.size(),no_of_nodes-1);

        for(int i=0; i<no_of_nodes-1; i++)
        {
            Map<VectorXd> t_other(events[other_nodes[i]].data(),events[other_nodes[i]].size());
            index_map.col(i) = mapAtoBIndex2(t_current,t_other);
        }

        rmapT1T2.push_back(index_map);
    }
    return;
}

