#include <iostream>
#include <fstream>

#include "include/base_objects.hpp"
#include "include/devices.hpp"
#include "include/neurons.hpp"
#include "include/network.hpp"

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
    int Na = 5000;
    int Nb = 5000;

    SpikingNetwork sn = SpikingNetwork();
    Population a = Population(Na, neuron_type::aqif, &sn);
    Population b = Population(Nb, neuron_type::aqif, &sn);

    cout << "size of neuron is " << sizeof(*(a.neurons[0])) << " bytes" <<endl ;
    cout << "size of population is " << sizeof(a) << " bytes" << endl;

    double ** weights, **delays;

    weights = get_rand_proj_mat(Na,Nb, -0.02,0.08);
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


    Projection * projection = new Projection(weights, delays, Na, Nb);

    a.project(projection, &b);
    b.project(projection, &a);

    delete projection;
    free_proj_mat(weights, Na);
    free_proj_mat(delays, Nb);


    PopCurrentInjector stimulus = PopCurrentInjector(&a, 15, 2);
    sn.add_injector(&stimulus);

    EvolutionContext evo = EvolutionContext(0.1);
    
    sn.run(&evo, 5);
}

