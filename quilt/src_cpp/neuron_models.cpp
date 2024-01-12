#include "include/neuron_models.hpp"
#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <boost/numeric/odeint.hpp>

// The menu:
class EvolutionContext;
class HierarchicalID;
class Spike;
class Synapse;
class Neuron;
class Population;
class Projection;


aqif_neuron::aqif_neuron(Population * population): Neuron(population){
    nt = neuron_type::aqif;
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    state = neuron_state {p->E_rest ,0.,0.,0.};
}

void aqif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);

    dxdt[0] = 1/p->C_m * ( p->k*(x[0]- p->E_rest)*(x[0]- p->E_thr) - x[1]*(x[0]- p->E_exc) - x[2]*(x[0]- p->E_inh) \
                          - x[3] + p->I_ext + p->I_osc*std::sin(p->omega_I*t) );                              
    dxdt[1] = -x[1]/p->tau_e;                                                                       
    dxdt[2] = -x[2]/p->tau_i;                                                                       
    dxdt[3] = -x[3]/p->ada_tau_w + p->ada_a/p->ada_tau_w * (x[0] - p->E_rest);  
}

void aqif_neuron::on_spike(EvolutionContext * /*evo*/){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    state[0]  = p->E_reset;
    state[3] += p->ada_b;
}

izhikevich_neuron::izhikevich_neuron(Population * population): Neuron(population){
    nt = neuron_type::izhikevich;
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam);

    state = neuron_state { 
                            p->E_rest + ((double)rand())/RAND_MAX, // V
                            0.0, // g_syn_exc
                            0.0, // g_syn_inh
                            0.0 // u
                            };
}

void izhikevich_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double /*t*/ ){
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam);

    dxdt[0] =  0.04*x[0]*x[0] + 5*x[0] + 140 \
                    - x[3] \
                    + x[1] * (x[0] - p->E_exc)\
                    - x[2] * (x[0] - p->E_inh);

    dxdt[3] =  p->a * ( p->b * x[0] - x[3]); 

    dxdt[1] = - x[1]/p->tau_e;
    dxdt[2] = - x[2]/p->tau_i;
}

void izhikevich_neuron::on_spike(EvolutionContext * /*evo*/){
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam); 
    state[0]  = p->E_rest;
    state[3] += p->d;
}

aeif_neuron::aeif_neuron(Population * population): Neuron(population){
    state = {population->neuroparam->E_rest + 10*(((double)rand())/RAND_MAX - 0.5 ), 0.0, 0.0, 0.0};
}

void aeif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    // auto start = std::chrono::high_resolution_clock::now();
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);
    // auto end = std::chrono::high_resolution_clock::now();

    // std::cout << (double) std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() << std::endl;

    if(p->Delta > 0){dxdt[0] = 0;}
    if(p->exp_threshold > 0){dxdt[0] = 0;}
    
    if (t > last_spike_time + p->tau_refrac){
        dxdt[0] = 1.0/p->tau_m * ( - (x[0]-p->E_rest) +\
                                     p->Delta*exp((x[0] - p->exp_threshold)/p->Delta)) \
                + 1.0/p->C_m * ( - x[1]*(x[0]-p->E_exc) - x[2]*(x[0]-p->E_inh) - x[3])\
                + 1.0/p->C_m * ( p->I_ext  + p->I_osc*std::sin(p->omega_I*t)); 
    }else{
        dxdt[0] = 0.0;
    }

    // This neuron suffers from slope divergence
    // and this way is the easiest to solve the issue
    if (dxdt[0] > MAX_POTENTIAL_SLOPE){dxdt[0] = MAX_POTENTIAL_SLOPE;}

    dxdt[1] = -x[1]/p->tau_e;                                                                       
    dxdt[2] = -x[2]/p->tau_i;                                                                      
    dxdt[3] = -x[3]/p->ada_tau_w + p->ada_a/p->ada_tau_w*(x[0]-p->E_rest);       
    
    // These two lines take time
    // but are necessary for now
    // utilities::nan_check_vect(x, "NaN in x");
    // utilities::nan_check_vect(dxdt, "NaN in dxdt");                                     
}

void aeif_neuron::on_spike(EvolutionContext * /*evo*/){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);
    state[0]  = p->E_reset;
    state[3] += p->ada_b;
}

