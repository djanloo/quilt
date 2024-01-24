/**
 * This file is necessary only to profile the C++ part of the code
 * through callgrind & massif. It implements a dummy system and it's not part of the quilt's code.
*/

#include <iostream>
#include <fstream>
#include <string>

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

float ** get_rand_proj_mat(int N, int M, double min, double max){
    float** matrix = new float*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new float[M];
    }

    for (int i=0;i<N;i++){
        for (int j=0; j< M; j++){
            matrix[i][j] = min + (max-min)*rand_01();
        }
    }
    return matrix;
}

void free_proj_mat(float** matrix, int N) {
    for (int i = 0; i < N; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
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

    float ** weights, **delays;

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

    Projection projection = Projection(weights, delays, Na, Nb);

    a.project(&projection, &b);
    b.project(&projection, &a);


    free_proj_mat(weights, Na);
    free_proj_mat(delays, Nb);


    // PopCurrentInjector stimulus_a = PopCurrentInjector(&a, 500.0, 0.0, 10.0);
    // PopCurrentInjector stimulus_b = PopCurrentInjector(&b, 500.0, 0.0, 10.0);
    PoissonSpikeSource stimulus_a = PoissonSpikeSource(&a, 900, 10.0, 0.1, 2);
    PoissonSpikeSource stimulus_b = PoissonSpikeSource(&b, 900, 10.0, 0.1, 2);

    sn.add_injector(&stimulus_a);
    sn.add_injector(&stimulus_b);


    sn.add_spike_monitor(&a);

    EvolutionContext evo = EvolutionContext(0.1);
    
    sn.run(&evo, 5);

    for (auto val : sn.population_spike_monitors[0]->get_history()){
        cout << val << " "; 
    }
}

void test_oscill(){

    int N = 4;
    float ** weights, **delays;

    weights = get_rand_proj_mat(N, N, 0.01, 0.05);
    delays = get_rand_proj_mat(N, N, 10.0, 20.0);
    Projection proj = Projection(weights, delays, N, N);

    cout << "Preparing params" << endl;
    vector<ParaMap*> params(4);

    params[0] = new ParaMap();
    params[0]->add("x0", 0.0);
    params[0]->add("v0", 1.0);
    params[0]->add("k", 1.0);

    params[1] = new ParaMap();
    params[1]->add("x0", 0.0);
    params[1]->add("v0", 0.0);
    params[1]->add("k", 5.0);

    params[2] = new ParaMap();
    params[2]->add("x0", 0.0);
    params[2]->add("v0", 0.0);
    params[2]->add("k", 2.0);

    params[3] = new ParaMap();
    params[3]->add("x0", 0.0);
    params[3]->add("v0", 0.0);
    params[3]->add("k", 2.5);
    
    cout << "params done "<< endl;
    OscillatorNetwork osc_net = OscillatorNetwork(oscillator_type::harmonic, params, proj);    

    EvolutionContext evo = EvolutionContext(0.1);

    ofstream file("output.txt");
    osc_net.run(&evo, 900.0);

    for (int i=0; i < osc_net.oscillators[0]->history.size(); i++){
        for (auto osc : osc_net.oscillators){
            for (auto val : osc->history[i]){
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

int main(){
    random_utils::rng.seed(1234);
    test_sparse();
}

