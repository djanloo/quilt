#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <chrono>
#include <cmath>

#include "include/base_objects.hpp"
#include "include/devices.hpp"
#include "include/neurons.hpp"
#include "include/network.hpp"

#include <boost/numeric/odeint.hpp>

#define MAX_POTENTIAL_INCREMENT 10 // mV

void Synapse::fire(EvolutionContext * evo){
    this->postsynaptic->incoming_spikes.emplace(this->weight, evo->now + this->delay);
    return;
}

Neuron::Neuron(Population * population){
    // TODO: too much redundancy in parameters. 
    // If parameter is the same for the population
    // use a this -> population -> value
    this->E_exc = 0.0;      // mV
    this->E_inh = -80.0;    // mV
    this->E_rest = -60.0;   // mV
    this->E_thr = -40.0;     // mV
    
    this->tau_refrac = 1;  // ms
    this->tau_i = 10;
    this->tau_e = 5;
    this->tau_m = 15;

    this -> state = vector<double> { 
                                    this -> E_rest + ((double)rand())/RAND_MAX, 
                                    0.0, 
                                    0.0
                                    };


    this -> id = new HierarchicalID( population -> id);
    this -> population = population;

    this-> last_spike_time = - 1000;

    // Adds to the population
    population -> neurons.push_back(this);
};

void Neuron::connect(Neuron * neuron, double weight, double delay){
    (this -> efferent_synapses).push_back(new Synapse(this, neuron, weight, delay));
}

void Neuron::handle_incoming_spikes(EvolutionContext * evo){

    while (!(this->incoming_spikes.empty())){
        
        auto spike = this->incoming_spikes.top();

        if ((spike.arrival_time < evo->now)&(!spike.processed)){cout << "ERROR: spike missed" << endl;} 

        if (!(spike.processed)){

            if ((spike.arrival_time >= evo->now ) && (spike.arrival_time < evo->now + evo->dt)){
                // Excitatory
                if (spike.weight > 0.0){ this->state[1] += spike.weight;} 
                // Inhibitory
                else if (spike.weight < 0.0){ this->state[2] -= spike.weight;}
                // Spurious zero-weight
                else{
                    cout << "Warning: a zero-weighted spike was received" << endl;
                    cout << "\tweight is " << spike.weight<< endl; 
                }
                spike.processed = true;

                // Removes the spike from the incoming spikes
                this->incoming_spikes.pop();
            } else {
                // If a spike is not to process, neither the rest will be
                break;
            }
        }else{
            cout << "spike already processed" << endl;
        }
    }
}

void Neuron::evolve(EvolutionContext * evo){
    // Gather spikes
    this-> handle_incoming_spikes(evo);

    // Evolve
    this->evolve_state(evo);
    
    // Spike generation
    if ((this -> state[0]) > this->E_thr){ this -> emit_spike(evo);}
}

void Neuron::emit_spike(EvolutionContext * evo){
    for (auto synapse : this->efferent_synapses){ (*synapse).fire(evo); }

    this -> last_spike_time = evo -> now;
    ((this->population)->n_spikes_last_step) ++;

    this-> on_spike(evo);
}

void Neuron::on_spike(EvolutionContext * evo){
    this->state[0] = this->E_rest;
}


// *************************** More detailed models ************************ //

void aqif_neuron::evolve_state(EvolutionContext * evo){
    double V_m = this->state[0];

    if ( (evo->now) > (this->last_spike_time) + (this->tau_refrac) ){
        // Membrane decay
        this->state[0] -= ( V_m - this->E_rest) * evo->dt / this->tau_m;
        // Synaptic currents
        this -> state [0] -= evo -> dt * (this -> state [1])*(  V_m - this -> E_exc);
        this -> state [0] -= evo -> dt * (this -> state [2])*(  V_m - this -> E_inh);
    }
    // Conductances
    this->state[1] -= (this->state[1]) * (evo->dt) / (this->tau_e);
    this->state[2] -= (this->state[2]) * (evo->dt) / (this->tau_i);
}

izhikevich_neuron::izhikevich_neuron(Population * population): Neuron(population){
    this->nt = neuron_type::izhikevich;

    this -> state = vector<double> { 
                                    this -> E_rest + ((double)rand())/RAND_MAX, // V
                                    0.0, // g_syn_exc
                                    0.0, // g_syn_inh
                                    0.0 // u
                                    };
    
    this-> a = 0.02;
    this-> b = 0.2;
    this-> c = -65;
    this-> d = 8;
}

void izhikevich_neuron::evolve_state(EvolutionContext * evo){
    double V_m = this->state[0];
    double u = this->state[3];

    this->state[0] += evo->dt * ( 0.04*V_m*V_m + 5*V_m + 140 - u); // V
    this->state[3] += evo->dt * this->a * ( this->b * V_m - u);  // u

    // Synaptic currents
    this->state[0] -= evo->dt * (this->state[1])*(  V_m - this->E_exc);
    this->state[0] -= evo->dt * (this->state[2])*(  V_m - this->E_inh);

    // Conductances
    this->state[1] -= (this->state[1]) * (evo->dt) / (this->tau_e);
    this->state[2] -= (this->state[2]) * (evo->dt) / (this->tau_i);

    if (this->state[1] > MAX_GSYN_EXC){this->state[1] = MAX_GSYN_EXC;}
    if (this->state[2] > MAX_GSYN_INH){this->state[2] = MAX_GSYN_INH;}

    if (abs(this->state[0] - V_m)> MAX_POTENTIAL_INCREMENT){std::cout << "WARNING: membrane potential increment over maximum"<< std::endl;}

}

void izhikevich_neuron::on_spike(EvolutionContext * evo){ 
    // Does not depend on evolution context
    this->state[0]  = this->E_rest;
    this->state[3] += this->d;
}

aeif_neuron::aeif_neuron(Population * population): Neuron(population){

    this->C_m = 40.;
    this->E_rest = -55.1;
    this->E_exc = 0.;
    this->E_inh = -65.;
    this->E_reset = -60.;
    this->E_thr = -54.7;
    this->g_L = 1.;
    this->tau_e= 10.;
    this->tau_i= 5.5;
    this-> a = 2.5;
    this->b = 70.;
    this->tau_w = 20.;
    this->Delta =  1.7;

    this->state = {this->E_rest, 0.0, 0.0, 0.0};

}

void aeif_neuron::evolve_state(EvolutionContext * evo){
    double V_m = this->state[0];
    double u = this->state[3];

    this->state[0] += evo->dt/C_m * (-g_L*((V_m - E_rest) + Delta * std::exp((V_m - E_rest)/Delta )));
    this->state[0] -= evo->dt/C_m * u;

    // Synaptic currents
    this->state[0] -= evo->dt * (this->state[1])*(  V_m - this->E_exc);
    this->state[0] -= evo->dt * (this->state[2])*(  V_m - this->E_inh);

    // Conductances
    this->state[1] -= (this->state[1]) * (evo->dt) / (this->tau_e);
    this->state[2] -= (this->state[2]) * (evo->dt) / (this->tau_i);

    this->state[3] += - evo->dt * ( u / tau_w + a/tau_w*(state[0] - E_rest));
}

void aeif_neuron::on_spike(EvolutionContext * evo){
    this->state[0] = this->E_rest;
    this->state[3] += this->b * evo->dt;
}