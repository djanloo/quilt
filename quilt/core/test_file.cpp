/**
 * This file is necessary only to profile the C++ part of the code
 * through callgrind & massif. It implements a dummy system and it's not part of the quilt's code.
*/

#include <iostream>
#include <fstream>
#include <string>
// #include <boost/math/special_functions/erf.hpp>

#include "include/base_objects.hpp"
#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/neuron_models.hpp"
#include "include/network.hpp"
#include "include/oscillators.hpp"

using namespace std;
// ulimit -c unlimited
// sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t

float rand_01(){
    return ((float)rand())/RAND_MAX;
}

vector<vector<float>> get_rand_proj_mat(int N, int M, double min, double max){
    
    vector<vector<float>> matrix(N, vector<float>(M, 0));

    for (int i=0;i<N;i++){
        for (int j=0; j< M; j++){
            matrix[i][j] = min + (max-min)*rand_01();
        }
    }
    return matrix;
}


void test_spiking()
{

    int Na = 1000;
    int Nb = 1000;

    SpikingNetwork sn = SpikingNetwork();


    map<string, float> map_of_params = {{"neuron_type", (float)neuron_type::aeif},
                                        {"C_m", 40.1},
                                        {"G_L",2.0},
                                        {"E_l", -70.0},
                                        {"V_reset", -55.0},
                                        {"V_peak",0.1},
                                        {"tau_refrac",0.0},
                                        {"delta_T",1.7},
                                        {"V_th", -40.0},
                                        {"ada_a", 0.0},
                                        {"ada_b",5.0},
                                        {"ada_tau_w",100.0},
                                        {"tau_ex", 10.},
                                        {"tau_in", 5.5},
                                        {"E_ex", 0.0},
                                        {"E_in",-65}
                                        };

    ParaMap paramap = ParaMap(map_of_params);
    
    Population a = Population(Na, &paramap, &sn);
    Population b = Population(Nb, &paramap, &sn);

    cout << "size of neuron is " << sizeof(*(a.neurons[0])) << " bytes" << endl ;
    cout << "size of population is " << sizeof(a) << " bytes" << endl;

    vector<vector<float>>  weights, delays;

    weights = get_rand_proj_mat(Na,Nb, -0.02,0.1);
    delays = get_rand_proj_mat(Na,Nb, 0.5, 1.0);

    for (int i = 0; i < Na; i ++){
        for (int j=0; j < Nb; j++){
            if (rand_01() > 0.1){
                weights[i][j] = 0.0;
                delays[i][j] = 0.0;
            }
            if (abs(weights[i][j]) < WEIGHT_EPS){
                weights[i][j] = 0.0;
                delays[i][j] = 0.0;
            }
        }
    }

    Projection projection = Projection(weights, delays);

    a.project(&projection, &b);
    b.project(&projection, &a);


    // PopCurrentInjector stimulus_a = PopCurrentInjector(&a, 500.0, 0.0, 10.0);
    // PopCurrentInjector stimulus_b = PopCurrentInjector(&b, 500.0, 0.0, 10.0);
    PoissonSpikeSource stimulus_a = PoissonSpikeSource(&a, 900, 10.0, 2, 0.1, 2);
    PoissonSpikeSource stimulus_b = PoissonSpikeSource(&b, 900, 10.0, 2,0.1, 2);

    sn.add_injector(&stimulus_a);
    sn.add_injector(&stimulus_b);


    sn.add_spike_monitor(&a);

    EvolutionContext evo = EvolutionContext(0.1);
    
    sn.run(&evo, 5);

    for (auto val : sn.population_spike_monitors[0]->get_history()){
        cout << val << " "; 
    }
}

void test_poisson(){
    int Na = 5;
    int Nb = 5;

    SpikingNetwork sn = SpikingNetwork();


    map<string, float> map_of_params = {{"neuron_type", (float)neuron_type::aeif},
                                        {"C_m", 40.1},
                                        {"G_L",2.0},
                                        {"E_l", -70.0},
                                        {"V_reset", -55.0},
                                        {"V_peak",0.1},
                                        {"tau_refrac",0.0},
                                        {"delta_T",1.7},
                                        {"V_th", -40.0},
                                        {"ada_a", 0.0},
                                        {"ada_b",5.0},
                                        {"ada_tau_w",100.0},
                                        {"tau_ex", 10.},
                                        {"tau_in", 5.5},
                                        {"E_ex", 0.0},
                                        {"E_in",-65}
                                        };

    ParaMap paramap = ParaMap(map_of_params);
    
    Population a = Population(Na, &paramap, &sn);
    Population b = Population(Nb, &paramap, &sn);

    SparseLognormProjection btoa(0.05, 0, 
                                Nb, Na, 
                                1.2, 0.0, 
                                1.5, 0.0 );

    SparseLognormProjection atob(0.05, 0, 
                                Na, Nb, 
                                3, 1, 
                                3, 1 );
    a.project(&atob, &b);
    b.project(&btoa, &a);

    PoissonSpikeSource stimulus_a = PoissonSpikeSource(&a, 6000, 10.0, 2, 0.1, -0.1);
    PoissonSpikeSource stimulus_b = PoissonSpikeSource(&b, 6000, 10.0, 2, 0.1, -0.1);

    sn.add_injector(&stimulus_a);
    sn.add_injector(&stimulus_b);

    EvolutionContext evo = EvolutionContext(0.1);
    sn.run(&evo, 5000);
}

void test_oscill(){

    int N = 2;
    vector<vector<float>> weights, delays;


    // cout << "making weights" << endl;
    // weights = get_rand_proj_mat(N,N, 0,0);
    // delays = get_rand_proj_mat(N,N, 0,0);

    // for (int i = 0; i< N;i++){
    //     for (int j=0; j< N; j++){
    //         weights[i][j] = 0.5;
    //         delays[i][j] = 1;
    //     }
    // }

    // for (int i =0; i< N; i++){
    //     weights[i][i] = 0.0;
    // }
    // cout << "making projection" << endl;
    // Projection proj = Projection(weights, delays);

    cout << "Preparing params" << endl;
    vector<ParaMap*> params(N);

    params[0] = new ParaMap();
    params[1] = new ParaMap();

    cout << "params done "<< endl;
    EvolutionContext evo = EvolutionContext(1);

    OscillatorNetwork osc_net = OscillatorNetwork(&evo);

    vector<dynamical_state> init_cond;
    for (int i=0; i< N; i++){
        new jansen_rit_oscillator(params[i], &osc_net, &evo);
        vector<double> initstate(6, 0);

        initstate[0] = 0.13 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[1] = 23.9 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[2] = 16.2 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[3] = -0.14/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[4] = 5.68/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[5] = 108.2/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);

        init_cond.push_back(initstate);
    }    
    osc_net.oscillators[1]-> connect(osc_net.oscillators[0], 1, 100);
    osc_net.oscillators[0]-> connect(osc_net.oscillators[1], 1, 100);

    osc_net.init_oscillators(init_cond);

    ofstream file("output.txt");
    osc_net.run(&evo, 10000);

    for (int i=0; i < osc_net.oscillators[0]->memory_integrator.state_history.size(); i++){
        for (auto osc : osc_net.oscillators){
            for (auto val : osc->memory_integrator.state_history[i]){
                file << val << " ";
            }
        }
        file << endl;
    }
}


void test_sparse(){
    int Na = 4000;
    int Nb = 4000;

    SpikingNetwork sn = SpikingNetwork();


    map<string, float> map_of_params = {{"neuron_type", (float)neuron_type::aeif},
                                        {"C_m", 40.1},
                                        {"G_L",2.0},
                                        {"E_l", -70.0},
                                        {"V_reset", -55.0},
                                        {"V_peak",0.1},
                                        {"tau_refrac",0.0},
                                        {"delta_T",1.7},
                                        {"V_th", -40.0},
                                        {"ada_a", 0.0},
                                        {"ada_b",5.0},
                                        {"ada_tau_w",100.0},
                                        {"tau_ex", 10.},
                                        {"tau_in", 5.5},
                                        {"E_ex", 0.0},
                                        {"E_in",-65}
                                        };

    ParaMap paramap = ParaMap(map_of_params);
    
    Population a = Population(Na, &paramap, &sn);
    Population b = Population(Nb, &paramap, &sn);

    cout << "size of neuron is " << sizeof(*(a.neurons[0])) << " bytes" << endl ;
    cout << "size of population is " << sizeof(a) << " bytes" << endl;

    SparseLognormProjection btoa(0.05, 0, 
                                Nb, Na, 
                                1.2, 0.0, 
                                1.5, 0.0 );
    auto start = std::chrono::high_resolution_clock::now();
    SparseLognormProjection atob(0.05, 0, 
                                Na, Nb, 
                                3, 1, 
                                3, 1 );
    a.project(&atob, &b);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() ;
    cout << "MAIN: projection took "<< duration/1000/1000 << "ms ";
    cout << duration/atob.n_connections << " ns/syn" <<endl;
    b.project(&btoa, &a);

    std::ofstream out_file("output.txt");
    for (auto sector : atob.weights_delays){
        for (auto pair : sector){
            out_file << pair.first.first <<" "<< pair.first.second <<" "<< pair.second.first <<" "<<pair.second.second<<endl;
        }
    }
}
void test_erf_overflow(){
    // float u = 1.0;
    // float val = std::exp(-2.0 + 1.0 * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));

}

int main(){
    // test_sparse();
    // test_poisson();
    test_oscill();
}

