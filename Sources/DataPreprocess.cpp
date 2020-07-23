#include "DataPreprocess.h"


DataPreprocess::DataPreprocess(vector<vector <double> > a) : events(a) {}

int DataPreprocess::get_nodes()
{
    int no_of_nodes =  events.size();
    cout<<"number of nodes:"<<no_of_nodes<<endl;
    return no_of_nodes;
}

int DataPreprocess::get_no_of_params()
{
    int temp = events.size();
    no_of_params= 1+(temp*2);
    cout<<"number of parameters:"<<no_of_params<<endl;
    return no_of_params;
}


int DataPreprocess::total_events()
{
    int no_of_nodes = events.size();
    vector<int> length_of_t;

    for (int i = 0; i < no_of_nodes; i++)
    {
        length_of_t.push_back(events[i].size());
    }

    int total_length = accumulate(length_of_t.begin(),length_of_t.end(),0);

    //cout<<"total no. of events:"<<total_length<<endl;
    return total_length;
}

double DataPreprocess::max_event()
{
    Map<VectorXd> t1(events[0].data(),events[0].size()); //convert std vector to eigen vector
    Map<VectorXd> t2(events[1].data(),events[1].size());
    T = (t1.tail(1)).cwiseMax(t2.tail(1)).coeff(0);
    //T = ((t1.tail(1)).max(t2.tail(1))).coeff(0); //array1.max(array2), if array
    return T;
}

ArrayXXi DataPreprocess::compressed_array()
{
    int no_of_nodes =  events.size();
    int total_length = total_events();

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

vector<int> DataPreprocess::MapAtoBIndex(VectorXd a, VectorXd b) //unnecessary std to eigen and eigen to std conversions. change the code
{
    vector <int> v; //need std::vector for nested vector
    ArrayXi::Index max_index;
    ArrayXi mapAtoBIndex = ArrayXi::Zero(a.size());
    int n = 0;
    for(auto x : vector<double>(a.data(), a.data()+a.size()))
    {
        ArrayXd result = (b.array() < x).select(b,-1);
        if (result.maxCoeff() == -1)
            mapAtoBIndex(n) = -1;
        else
        {
            double temp = result.maxCoeff(&max_index);
            mapAtoBIndex(n) = max_index;
        }
        v.push_back(mapAtoBIndex.coeff(n));
        n += 1;
    }
    return v;
}

vector<vector<int> > DataPreprocess::map_index()
{
    //if events is ArrayXXd then change this, for now its vector<vector<double>>

    Map<VectorXd> t1(events[0].data(),events[0].size()); //convert std vector to eigen vector
    Map<VectorXd> t2(events[1].data(),events[1].size());
    vector<vector<int> > MapT1T2; //using 2-D vector
    MapT1T2.push_back(MapAtoBIndex(t1, t2));
    MapT1T2.push_back(MapAtoBIndex(t2, t1));
    return  MapT1T2;
}

ArrayXXd DataPreprocess::initialize_para()
{
    int no_of_nodes =  events.size();
    int no_of_params= 1+(no_of_nodes*2);
    //params.setRandom(no_of_params, no_of_nodes); //setRandom [-1,1]
    params = 0+(ArrayXXd::Random(no_of_params, no_of_nodes)*0.5+0.5)*(1-0); //[0,1] [low high]
    cout<<params<<endl;
    return params;
}

VectorXd DataPreprocess::random_sample(int length)
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
