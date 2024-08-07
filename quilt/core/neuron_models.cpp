#include "include/neuron_models.hpp"
#include "include/base.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <chrono>
#include <cmath>
#include <string>
#include <boost/numeric/odeint.hpp>


// *********************************** AQIF **************************************************//

aqif_neuron::aqif_neuron(Population * population): Neuron(population){
    nt = "aqif";
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    state = dynamical_state {p->E_l + 20*(static_cast<double>(rand())/RAND_MAX - 0.5 ) ,0.,0.,0.};
}

void aqif_neuron::evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);

    // Division by membrane capacity is later
    dxdt[0] = p->k*(x[0] - p->E_l)*(x[0] - p->V_th) - x[1]*(x[0]- p->E_ex) - x[2]*(x[0]- p->E_in) \
                          - x[3] + p->I_e;

    // Trig functions are expensive
    if (p->I_osc > OSCILLATORY_AMPLITUDE_MIN) dxdt[0] += p->I_osc*std::sin(p->omega_I*t);
    dxdt[0] /= p->C_m;  // Just one division for membrane capacity

    dxdt[1] = -x[1]/p->tau_ex;                                                                       
    dxdt[2] = -x[2]/p->tau_in;

    dxdt[3] = -x[3]+ p->ada_a * (x[0] - p->E_l);  
    dxdt[3] /= p->ada_tau_w;
}

void aqif_neuron::on_spike(){
    aqif_param * p = static_cast<aqif_param*>(population->neuroparam);
    state[0]  = p->V_reset;
    state[3] += p->ada_b;
}

// ******************************** IZHIKEVICH **************************************************//


izhikevich_neuron::izhikevich_neuron(Population * population): Neuron(population){
    nt = "izhikevich";
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam);

    state = dynamical_state { 
                            p->E_l + ((double)rand())/RAND_MAX, // V
                            0.0, // g_syn_exc
                            0.0, // g_syn_inh
                            0.0 // u
                            };
}

void izhikevich_neuron::evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double /*t*/ ){
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam);

    dxdt[0] =  0.04*x[0]*x[0] + 5*x[0] + 140 \
                    - x[3] \
                    + x[1] * (x[0] - p->E_ex)\
                    - x[2] * (x[0] - p->E_in);

    dxdt[3] =  p->a * ( p->b * x[0] - x[3]); 

    dxdt[1] = - x[1]/p->tau_ex;
    dxdt[2] = - x[2]/p->tau_in;
}

void izhikevich_neuron::on_spike(){
    izhikevich_param * p = static_cast<izhikevich_param*>(population->neuroparam); 
    state[3] += p->d;
}

// *********************************** AEIF **************************************************//

aeif_neuron::aeif_neuron(Population * population): Neuron(population){
    state = {population->neuroparam->E_l - 20*(static_cast<double>(rand())/RAND_MAX ), 0.0, 0.0, 0.0};
}

void aeif_neuron::evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);
    
    if (t > last_spike_time + p->tau_refrac){
        dxdt[0] =   - p->G_L * (x[0]-p->E_l)                                    +\
                    + p->G_L * p->delta_T * exp((x[0] - p->V_th)/p->delta_T)    +\
                    
                    + x[1]*(p->E_ex - x[0]) + x[2]*(p->E_in - x[0]) - x[3]    +\
                    +  p->I_e;
        // Trig functions are expensive
        if (p->I_osc > OSCILLATORY_AMPLITUDE_MIN) dxdt[0] += p->I_osc*std::sin(p->omega_I*t);
        
        dxdt[0] /= p->C_m;
    }else{
        dxdt[0] = 0.0;
    }

    // This neuron suffers from slope divergence
    // and this way is the easiest to solve the issue
    if (dxdt[0] > MAX_POTENTIAL_SLOPE){dxdt[0] = MAX_POTENTIAL_SLOPE;}

    dxdt[1] = -x[1]/p->tau_ex;                                                                       
    dxdt[2] = -x[2]/p->tau_in;


    dxdt[3] = -x[3] + p->ada_a*(x[0]-p->E_l);  
    dxdt[3] /= p->ada_tau_w;                                         
}

void aeif_neuron::on_spike(){
    aeif_param * p = static_cast<aeif_param*>(population->neuroparam);
    state[0]  = p->V_reset;
    state[3] += p->ada_b;
}

// *********************************** AQIF2 **************************************************//

aqif2_neuron::aqif2_neuron(Population * population): Neuron(population){
    nt = "aqif2";
    aqif2_param * p = static_cast<aqif2_param*>(population->neuroparam);
    state = dynamical_state {p->E_l + 20*(static_cast<double>(rand())/RAND_MAX - 0.5 ) ,0.,0.,0.};
}

void aqif2_neuron::evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ){
    aqif2_param * p = static_cast<aqif2_param*>(population->neuroparam);
    dxdt[0] =   p->k*(x[0]-p->E_l)*(x[0]-p->V_th) - x[1]*(x[0]-p->E_ex) \
                - x[2]*(x[0]-p->E_in) - x[3] + p->I_e ;

     // Trig functions are expensive
    if (p->I_osc > OSCILLATORY_AMPLITUDE_MIN) dxdt[0] += p->I_osc*std::sin(p->omega_I*t);
    dxdt[0] /= p->C_m;     

    dxdt[1] = -x[1]/p->tau_ex;                                                                       
    dxdt[2] = -x[2]/p->tau_in;                                                                 
    if (x[0] < p->V_b)  dxdt[3] = -x[3]/p->ada_tau_w + p->ada_a/p->ada_tau_w * std::pow((x[0]-p->V_b),3);
    else                dxdt[3] = -x[3]/p->ada_tau_w;                                                
}

void aqif2_neuron::on_spike(){
    aqif2_param * p = static_cast<aqif2_param*>(population->neuroparam);
    state[0]  = p->V_reset;
    state[3] += p->ada_b;
}

/**************************** SIOF workaround *******************/
/**
 * Registers the Neuron models in the Neurofactory
 */
struct NeuronModelRegistrar {
    NeuronModelRegistrar() {
        NeuroFactory::get_neuro_factory()->add_neuron("aeif", NeuroFactory::neuron_maker<aeif_neuron>, NeuroFactory::neuroparam_maker<aeif_param>);
        NeuroFactory::get_neuro_factory()->add_neuron("aqif", NeuroFactory::neuron_maker<aqif_neuron>, NeuroFactory::neuroparam_maker<aqif_param>);
        NeuroFactory::get_neuro_factory()->add_neuron("aqif2", NeuroFactory::neuron_maker<aqif2_neuron>, NeuroFactory::neuroparam_maker<aqif2_param>);
        NeuroFactory::get_neuro_factory()->add_neuron("izhikevich", NeuroFactory::neuron_maker<izhikevich_neuron>, NeuroFactory::neuroparam_maker<izhikevich_param>);
    }
} neuron_model_registrar;
