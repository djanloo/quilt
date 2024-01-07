#include "include/neurons_base.hpp"
#include "include/neuron_models.hpp"

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
    
    C_m = 15.2;
    k = 0.2;

    E_rest = -70.0;
    E_reset = -55.0;
    E_thr = 0.1;
    tau_refrac = 0.0;

    // Adapting pars
    ada_a = -20;
    ada_b =  91;
    ada_tau_w = 100.0;

    //  Syn pars
    tau_e= 10.;
    tau_i= 5.5;
    E_exc = 0.;
    E_inh = -74.;

}

void aqif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){

    dxdt[0] = 1/C_m * ( k*(x[0]- E_rest)*(x[0]- E_thr) - x[1]*(x[0]- E_exc) - x[2]*(x[0]- E_inh) \
                          - x[3] + I_ext + I_osc*std::sin(omega_I*t) );                              
    dxdt[1] = -x[1]/tau_e;                                                                       
    dxdt[2] = -x[2]/tau_i;                                                                       
    dxdt[3] = -x[3]/ada_tau_w + ada_a/ada_tau_w * (x[0] - E_rest);  
}

void aqif_neuron::on_spike(EvolutionContext * /*evo*/){
    state[0]  = E_reset;
    state[3] += ada_b;
}

izhikevich_neuron::izhikevich_neuron(Population * population): Neuron(population){
    nt = neuron_type::izhikevich;

    state = neuron_state { 
                            E_rest + ((double)rand())/RAND_MAX, // V
                            0.0, // g_syn_exc
                            0.0, // g_syn_inh
                            0.0 // u
                            };
    
    a = 0.02;
    b = 0.2;
    c = -65;
    d = 8;
}

void izhikevich_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){

    dxdt[0] =  0.04*x[0]*x[0] + 5*x[0] + 140 \
                    - x[3] \
                    + x[1] * (x[0] - E_exc)\
                    - x[2] * (x[0] - E_inh);

    dxdt[3] =  a * ( b * x[0] - x[3]); 

    dxdt[1] = - x[1]/tau_e;
    dxdt[2] = - x[2]/tau_i;
}

void izhikevich_neuron::on_spike(EvolutionContext * /*evo*/){ 
    state[0]  = E_rest;
    state[3] += d;
}

aeif_neuron::aeif_neuron(Population * population): Neuron(population){

    this->C_m = 40.;
    this->tau_m = 200.0;

    this->E_rest = -70.0;
    this->E_reset = -55.0;
    this->E_thr = 0.1;
    this->tau_refrac = 0.0;

    // Exp pars
    this->Delta =  1.7;
    this->exp_threshold = -40;

    // Adapting pars
    this->ada_a = 0.0;
    this->ada_b = 5.0;
    this->ada_tau_w = 100.0;

    //  Syn pars
    this->tau_e= 10.;
    this->tau_i= 5.5;
    this->E_exc = 0.;
    this->E_inh = -65.;

    state = {E_rest + 10*(((double)rand())/RAND_MAX - 0.5 ), 0.0, 0.0, 0.0};

}

void aeif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){

    if (t > last_spike_time + tau_refrac){
        dxdt[0] = 1.0/tau_m * ( - (x[0]-E_rest) + Delta*std::exp((x[0] - exp_threshold)/Delta)) \
                + 1.0/C_m * ( - x[1]*(x[0]-E_exc) - x[2]*(x[0]-E_inh) - x[3])\
                + 1.0/C_m * (I_ext  + I_osc*std::sin(omega_I*t)); 
    }else{
        dxdt[0] = 0.0;
    }

    // This neuron suffers from slope divergence
    // and this way is the easiest to solve the issue
    if (dxdt[0] > MAX_POTENTIAL_SLOPE){dxdt[0] = MAX_POTENTIAL_SLOPE;}

    dxdt[1] = -x[1]/tau_e;                                                                       
    dxdt[2] = -x[2]/tau_i;                                                                      
    dxdt[3] = -x[3]/ada_tau_w + ada_a/ada_tau_w*(x[0]-E_rest);       
    
    // These two lines take time
    // but are necessary for now
    // utilities::nan_check_vect(x, "NaN in x");
    // utilities::nan_check_vect(dxdt, "NaN in dxdt");                                     
}

void aeif_neuron::on_spike(EvolutionContext * evo){
    state[0]  = E_reset;
    state[3] += ada_b;
}

