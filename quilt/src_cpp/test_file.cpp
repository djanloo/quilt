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

    EvolutionContext evo = EvolutionContext(0.1);
    dummy_osc a = dummy_osc(1.0, 0.5, 0.0);
    dummy_osc b = dummy_osc(1.5, 0.0, 0.0);

    b.connect(&a, 0.01, 6.51);
    a.connect(&b, 0.1, 3.24);

    std::ofstream outputFile("output.txt");
    
    for (int i = 0; i < 1000; i++){
        std::cout <<"----- Time: "<< evo.now << std::endl;
        a.evolve(&evo);
        b.evolve(&evo);
        evo.do_step();
        for (auto val : a.state){
            outputFile << val << " ";
        }
        for (auto val : b.state){
            outputFile << val << " ";
        }
        outputFile << std::endl;
    }

}

