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
    state = neuron_state {p->E_l ,0.,0.,0.};
}

void aqif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);

    dxdt[0] = 1/p->C_m * ( p->k*(x[0] - p->E_l)*(x[0] - p->V_th) - x[1]*(x[0]- p->E_ex) - x[2]*(x[0]- p->E_inh) \
                          - x[3] + p->I_e + p->I_osc*std::sin(p->omega_I*t) );                              
    dxdt[1] = -x[1]/p->tau_ex;                                                                       
    dxdt[2] = -x[2]/p->tau_in;                                                                       
    dxdt[3] = -x[3]/p->ada_tau_w + p->ada_a/p->ada_tau_w * (x[0] - p->E_l);  
}

void aqif_neuron::on_spike(EvolutionContext * /*evo*/){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    state[0]  = p->V_reset;
    state[3] += p->ada_b;
}

izhikevich_neuron::izhikevich_neuron(Population * population): Neuron(population){
    nt = neuron_type::izhikevich;
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam);

    state = neuron_state { 
                            p->E_l + ((double)rand())/RAND_MAX, // V
                            0.0, // g_syn_exc
                            0.0, // g_syn_inh
                            0.0 // u
                            };
}

void izhikevich_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double /*t*/ ){
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam);

    dxdt[0] =  0.04*x[0]*x[0] + 5*x[0] + 140 \
                    - x[3] \
                    + x[1] * (x[0] - p->E_ex)\
                    - x[2] * (x[0] - p->E_inh);

    dxdt[3] =  p->a * ( p->b * x[0] - x[3]); 

    dxdt[1] = - x[1]/p->tau_ex;
    dxdt[2] = - x[2]/p->tau_in;
}

void izhikevich_neuron::on_spike(EvolutionContext * /*evo*/){
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam); 
    state[0]  = p->E_l;
    state[3] += p->d;
}

aeif_neuron::aeif_neuron(Population * population): Neuron(population){
    state = {population->neuroparam->E_l + 10*(((double)rand())/RAND_MAX - 0.5 ), 0.0, 0.0, 0.0};
}

void aeif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);

    // if(p->delta_T > 0){dxdt[0] = 0;}
    // if(p->V_th > 0){dxdt[0] = 0;}
    
    if (t > last_spike_time + p->tau_refrac){
        dxdt[0] = 1.0/p->C_m * (\
                                - p->G_L * (x[0]-p->E_l)                                    +\
                                + p->G_L * p->delta_T * exp((x[0] - p->V_th)/p->delta_T)    +\
                              
                                + x[1]*(p->E_ex - x[0]) + x[2]*(p->E_inh - x[0]) - x[3]    +\
                                +  ( p->I_e  + p->I_osc*std::sin(p->omega_I*t))
                                ); 
    }else{
        dxdt[0] = 0.0;
    }

    // This neuron suffers from slope divergence
    // and this way is the easiest to solve the issue
    if (dxdt[0] > MAX_POTENTIAL_SLOPE){dxdt[0] = MAX_POTENTIAL_SLOPE;}

    dxdt[1] = -x[1]/p->tau_ex;                                                                       
    dxdt[2] = -x[2]/p->tau_in;                                                                      
    dxdt[3] = -x[3]/p->ada_tau_w + p->ada_a/p->ada_tau_w*(x[0]-p->E_l);                                           
}

void aeif_neuron::on_spike(EvolutionContext * /*evo*/){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);
    state[0]  = p->V_reset;
    state[3] += p->ada_b;
}


// void poisson_neuron::
