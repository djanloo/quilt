/**
 * This file is necessary only to profile the C++ part of the code
 * through callgrind. It implements a dummy system and it's not part of the quilt's code.
*/

#include <iostream>
#include <fstream>

#include "include/base_objects.hpp"
#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/neuron_models.hpp"
#include "include/network.hpp"
#include "include/oscillators.hpp"

using namespace std;
// ulimit -c unlimited
// sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t

double rand_01(){
    return ((double)rand())/RAND_MAX;
}

double ** get_rand_proj_mat(int N, int M, double min, double max){
    double** matrix = new double*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[M];
    }

    for (int i=0;i<N;i++){
        for (int j=0; j< M; j++){
            matrix[i][j] = min + (max-min)*rand_01();
        }
    }
    return matrix;
}

void free_proj_mat(double** matrix, int N) {
    for (int i = 0; i < N; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

int main(){
    // int Na = 2000;
    // int Nb = 2000;

    // SpikingNetwork sn = SpikingNetwork();
    // Population a = Population(Na, neuron_type::aeif, &sn);
    // Population b = Population(Nb, neuron_type::aeif, &sn);

    // std::cout << "size of neuron is " << sizeof(*(a.neurons[0])) << " bytes" << std::endl ;
    // std::cout << "size of population is " << sizeof(a) << " bytes" << std::endl;

    // double ** weights, **delays;

    // weights = get_rand_proj_mat(Na,Nb, -0.02,0.1);
    // delays = get_rand_proj_mat(Na,Nb, 0.5, 1.0);

    // for (int i = 0; i < Na; i ++){
    //     for (int j=0; j < Nb; j++){
    //         if (rand_01() > 0.1){
    //             weights[i][j] = 0.0;
    //             delays[i][j] = 0.0;
    //         }
    //         if (std::abs(weights[i][j]) < WEIGHT_EPS){
    //             weights[i][j] = 0.0;
    //             delays[i][j] = 0.0;
    //         }
    //     }
    // }

    // Projection * projection = new Projection(weights, delays, Na, Nb);

    // a.project(projection, &b);
    // b.project(projection, &a);

    // delete projection;
    // free_proj_mat(weights, Na);
    // free_proj_mat(delays, Nb);


    // PopCurrentInjector stimulus_a = PopCurrentInjector(&a, 500.0, 0.0, 15.0);
    // PopCurrentInjector stimulus_b = PopCurrentInjector(&b, 500.0, 0.0, 15.0);

    // sn.add_injector(&stimulus_a);
    // sn.add_injector(&stimulus_b);


    // sn.add_spike_monitor(&a);

    // EvolutionContext evo = EvolutionContext(0.1);
    
    // sn.run(&evo, 10);

    // for (auto val : sn.population_spike_monitors[0]->get_history()){
    //     std::cout << val << " "; 
    // }
    int N = 4;
    double ** weights, **delays;

    weights = get_rand_proj_mat(N, N, 0.01, 0.05);
    delays = get_rand_proj_mat(N, N, 10.0, 20.0);
    Projection proj = Projection(weights, delays, N, N);

    cout << "Preparing params" << endl;
    std::vector<ParaMap*> params(4);

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

    std::ofstream file("output.txt");
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

