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
    state = neuron_state {E_rest ,0.,0.,0.};
}

void aqif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    float C_m     = p->C_m;
    float E_rest  = p->E_rest;
    float E_thr   = p->E_thr;
    float E_exc   = p->E_exc;
    float E_inh   = p->E_inh;
    float I_ext   = p->I_ext;
    float I_osc   = p->I_osc;
    float omega_I = p->omega_I;

    float ada_a = p->ada_a;
    float ada_tau_w = p->ada_tau_w;
    float k = p->k;

    dxdt[0] = 1/C_m * ( k*(x[0]- E_rest)*(x[0]- E_thr) - x[1]*(x[0]- E_exc) - x[2]*(x[0]- E_inh) \
                          - x[3] + I_ext + I_osc*std::sin(omega_I*t) );                              
    dxdt[1] = -x[1]/tau_e;                                                                       
    dxdt[2] = -x[2]/tau_i;                                                                       
    dxdt[3] = -x[3]/ada_tau_w + ada_a/ada_tau_w * (x[0] - E_rest);  
}

void aqif_neuron::on_spike(EvolutionContext * /*evo*/){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    state[0]  = p->E_reset;
    state[3] += p->ada_b;
}

izhikevich_neuron::izhikevich_neuron(Population * population): Neuron(population){
    nt = neuron_type::izhikevich;

    state = neuron_state { 
                            population->neuroparam->E_rest + ((double)rand())/RAND_MAX, // V
                            0.0, // g_syn_exc
                            0.0, // g_syn_inh
                            0.0 // u
                            };
    
}

void izhikevich_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
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

    // this->C_m = 40.;
    // this->tau_m = 200.0;

    // this->E_rest = -70.0;
    // this->E_reset = -55.0;
    // this->E_thr = 0.1;
    // this->tau_refrac = 0.0;

    // // Exp pars
    // this->Delta =  1.7;
    // this->exp_threshold = -40;

    // // Adapting pars
    // this->ada_a = 0.0;
    // this->ada_b = 5.0;
    // this->ada_tau_w = 100.0;

    // //  Syn pars
    // this->tau_e= 10.;
    // this->tau_i= 5.5;
    // this->E_exc = 0.;
    // this->E_inh = -65.;

    state = {population->neuroparam->E_rest + 10*(((double)rand())/RAND_MAX - 0.5 ), 0.0, 0.0, 0.0};

}

void aeif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);

    if (t > last_spike_time + p->tau_refrac){
        dxdt[0] = 1.0/p->tau_m * ( - (x[0]-p->E_rest) + p->Delta*std::exp((x[0] - p->exp_threshold)/p->Delta)) \
                + 1.0/p->C_m * ( - x[1]*(x[0]-p->E_exc) - x[2]*(x[0]-E_inh) - x[3])\
                + 1.0/p->C_m * (p->I_ext  + p->I_osc*std::sin(p->omega_I*t)); 
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

void aeif_neuron::on_spike(EvolutionContext * evo){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);
    state[0]  = p->E_reset;
    state[3] += p->ada_b;
}

