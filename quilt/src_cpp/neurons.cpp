#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "include/base_objects.hpp"
#include "include/devices.hpp"
#include "include/neurons.hpp"
#include "include/network.hpp"

#include <boost/numeric/odeint.hpp>

#define MAX_POTENTIAL_INCREMENT 10 // mV

namespace utilities{

    void nan_check(double value, const string& str){
        if (std::isnan(value)){
            throw std::runtime_error(str);
        }
    }

    void nan_check_vect(std::vector<double> vect, const string& str){
        vector<bool> are_nan;
        bool somebody_is_nan = false;

        for (auto value : vect){
            somebody_is_nan = somebody_is_nan || std::isnan(value) || std::isinf(value);
            are_nan.push_back( std::isnan(value) || std::isinf(value));
        }
        if (somebody_is_nan){
            std::cerr << "vector is nan: [" ;
            for (auto val : are_nan) {cerr << val <<" ";} cerr << " ]" << endl;
            
            throw std::runtime_error(str);
        }
    }
}

void Synapse::fire(EvolutionContext * evo){
    this->postsynaptic->incoming_spikes.emplace(this->weight, evo->now + this->delay);
}

Neuron::Neuron(Population * population){
    // TODO: too much redundancy in parameters. 
    // If parameter is the same for the population
    // use a this -> population -> value
    // or use static attributes
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
            utilities::nan_check(spike.weight, "NaN in spike weight");
            if ((spike.arrival_time >= evo->now ) && (spike.arrival_time < evo->now + evo->dt)){
                // Excitatory
                if (spike.weight > 0.0){ this->state[1] += spike.weight;} 
                // Inhibitory
                else if (spike.weight < 0.0){ this->state[2] -= spike.weight;}
                // Spurious zero-weight
                else{
                    cout << "Warning: a zero-weighted spike was received" << endl;
                    cout << "\tweight is " << spike.weight << endl; 
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
    boost::numeric::odeint::runge_kutta4<neuron_state> stepper;
    auto lambda = [this](const neuron_state & state, neuron_state & dxdt, double t) {
                                    this->evolve_state(state, dxdt, t);
                                };

    auto before_step = this->state;

    try{
        stepper.do_step(lambda, this->state, evo->now, evo->dt);
        utilities::nan_check_vect(this->state, "NaN in neuron state");
    }catch (const std::runtime_error &e){
        cerr << "State before step: ";
        for (auto val : before_step){
            cerr << val << " ";
        }
        cerr << endl;
        throw e;
    }

    // Spike generation
    if ((this -> state[0]) > this->E_thr){ this -> emit_spike(evo);}
}

void Neuron::emit_spike(EvolutionContext * evo){
    for (auto synapse : this->efferent_synapses){ (*synapse).fire(evo); }

    this -> last_spike_time = evo -> now;
    ((this->population)->n_spikes_last_step) ++;

    this-> on_spike(evo);
}

void Neuron::on_spike(EvolutionContext * /*evo*/){
    this->state[0] = this->E_rest;
}

/**
 * Real models
 * -----------
 * 
 * Each model must override these functions:
 * - the constructor
 * - evolve_state
 * 
 * Each model can override these functions:
 * - on_spike
*/


void aqif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){

    dxdt[0] = - (x[0] - E_rest)/tau_m - x[1]*(x[0] - E_exc) - x[2*(x[0] - E_inh)];
    dxdt[1] = - x[1]/tau_e;
    dxdt[2] = - x[2]/tau_i;
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
    this->state[0]  = this->E_rest;
    this->state[3] += this->d;
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
    this->exp_threshold = -50;

    // Adapting pars
    this->a = 0.0;
    this->b = 5.0;
    this->tau_w = 100.0;

    //  Syn pars
    this->tau_e= 10.;
    this->tau_i= 5.5;
    this->E_exc = 0.;
    this->E_inh = -65.;

    this->state = {this->E_rest + ((double)rand())/RAND_MAX , 0.0, 0.0, 0.0};

}

void aeif_neuron::evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){
    double piece1, piece2, piece3, piece4;

    if (t - last_spike_time > tau_refrac ){

        piece1 =  1.0/tau_m * ( - (x[0]-E_rest) );
        piece2 =  1.0/tau_m * ( Delta*std::exp((x[0] - exp_threshold)/Delta) );
        piece3 =  1.0/C_m * ( - x[1]*(x[0]-E_exc) - x[2]*(x[0]-E_inh));
        piece4 =  1.0/C_m * ( - x[3] + 300.0); 

        // dxdt[0] = 1.0/tau_m * ( - (x[0]-E_rest) + Delta*std::exp((x[0] - exp_threshold)/Delta)) \
        //         + 1.0/C_m * ( - x[1]*(x[0]-E_exc) - x[2]*(x[0]-E_inh) - x[3] + 300.0);      
        dxdt[0] = piece1 + piece2 + piece3 + piece4;  
    }else{
        dxdt[0] = 0.0;
    }

    dxdt[1] = -x[1]/tau_e;                                                                       
    dxdt[2] = -x[2]/tau_i;                                                                      
    dxdt[3] = -x[3]/tau_w + a/tau_w*(x[0]-E_rest);       
    
    utilities::nan_check_vect(x, "NaN in x");
    try{
        utilities::nan_check_vect(dxdt, "NaN in dxdt");
    } catch (const std::runtime_error& e){
        cerr << e.what() << endl;
        cerr << "linear: " << piece1 <<endl;
        cerr << "exp: " << piece2 << endl;
        cerr << "synaptic: " << piece3<<endl;
        cerr << "adapt+extern: " << piece4 << endl;

        cerr << "linear is due to: " <<endl;
        cerr << "1/tau_m " << 1.0/tau_m << endl;
        cerr << "x[0] " << x[0] << endl;
        cerr << "E_rest " << E_rest << endl;
        throw std::runtime_error("AAAA");
    }
                                            
}

void aeif_neuron::on_spike(EvolutionContext * evo){
    this->state[0]  = this->E_reset;
    this->state[3] += this->b;
}