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
#include "include/neurons_base.hpp"
#include "include/network.hpp"

#include <boost/numeric/odeint.hpp>

namespace utilities{

    void nan_check(double value, const std::string& str){
        if (std::isnan(value)){
            throw std::runtime_error(str);
        }
    }

    void nan_check_vect(std::vector<double> vect, const std::string& str){
        std::vector<bool> are_nan;
        bool somebody_is_nan = false;

        for (auto value : vect){
            somebody_is_nan = somebody_is_nan || std::isnan(value) || std::isinf(value);
            are_nan.push_back( std::isnan(value) || std::isinf(value));
        }
        if (somebody_is_nan){
            std::cerr << "vector is nan: [" ;
            for (auto val : are_nan) {std::cerr << val <<" ";} std::cerr << " ]" << std::endl;
            
            throw std::runtime_error(str);
        }
    }
}
double Synapse::min_delay = std::numeric_limits<double>::infinity();

void Synapse::fire(EvolutionContext * evo){
    // Adds a (weight, delay) spike in the spike queue of the postsynaptic neuron
    if (this->delay < evo->dt){throw std::runtime_error("Synapse has delay less than timestep");}
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
    this->E_reset = -55.0;  // mV
    this->E_thr = -40.0;     // mV
    
    this->tau_refrac = 1;  // ms
    this->tau_i = 10;
    this->tau_e = 5;
    this->tau_m = 15;
    
    I_ext = 0.0;
    I_osc = 0.0;
    omega_I = 0.0;

    this -> state = neuron_state { E_rest + ((double)rand())/RAND_MAX, 0.0, 0.0};

    this -> id = new HierarchicalID( population -> id);
    this -> population = population;

    this-> last_spike_time = - 1000;

    population -> neurons.push_back(this);        
};

void Neuron::connect(Neuron * neuron, double weight, double delay){
    (this -> efferent_synapses).push_back(new Synapse(this, neuron, weight, delay));
}


/**
 * 
 * This is the single most important function of the code.
 * 
*/
void Neuron::handle_incoming_spikes(EvolutionContext * evo){
    // if (incoming_spikes.empty()) {
    //     std::cout << population->id->get_id() << " - " << id->get_id() << ") -- " ;
    //     std::cout << "\tnow: "<<evo->now<<std::endl;
    // }
    while (!(incoming_spikes.empty())){

        auto spike = incoming_spikes.top();

        // Check for missed spikes
        if ((spike.arrival_time < evo->now)&(!spike.processed)){
            std::string message = "Spike Missed\n";
            message += "Spike arrival time: " + std::to_string(spike.arrival_time) + "\n";
            message += "t: " + std::to_string(evo->now) + "\n";
            message += "Please reduce the timestep or increase the delays.\n";
            throw std::runtime_error(message);
        } 

        if (!(spike.processed)){
            // utilities::nan_check(spike.weight, "NaN in spike weight"); // This might be removed in future

            if ((spike.arrival_time >= evo->now ) && (spike.arrival_time < evo->now + evo->dt)){

                // Excitatory
                if (spike.weight > 0.0){ state[1] += spike.weight;} 
                // Inhibitory
                else if (spike.weight < 0.0){ state[2] -= spike.weight;}

                else{ 
                    throw std::runtime_error("A zero-weighted spike was received. \
                                              Value is " + std::to_string(spike.weight));
                }
                spike.processed = true;

                // std::cout << population->id->get_id() << " - " << id->get_id() << ") processed spike with t = " << spike.arrival_time;
                // std::cout << "\tnow: "<<evo->now<<std::endl;

                // Removes the spike from the incoming spikes
                incoming_spikes.pop();
            } else {
                // std::cout << population->id->get_id() << " - " << id->get_id()  << ") stopped at spike with t = " << spike.arrival_time;
                // std::cout << "\tnow: "<<evo->now<<std::endl;

                // If a spike is not to process, neither the rest will be
                break;
            }
        }else{
            std::cout << "spike already processed" << std::endl;
        }
    }
}

void Neuron::evolve(EvolutionContext * evo){

    // Process incoming spikes
    handle_incoming_spikes(evo);

    // Evolve
    boost::numeric::odeint::runge_kutta4<neuron_state> stepper;
    auto lambda = [this](const neuron_state & state, neuron_state & dxdt, double t) {
                                    this->evolve_state(state, dxdt, t);
                                };

    auto before_step = state;

    // Checks for NaNs after the step
    try{
        stepper.do_step(lambda, this->state, evo->now, evo->dt);
        utilities::nan_check_vect(this->state, "NaN in neuron state");
    }catch (const std::runtime_error &e){
        std::cerr << "State before step: ";
        for (auto val : before_step){ std::cerr << val << " ";}
        std::cerr << std::endl;
        throw e;
    }

    // Spike generation
    if ((state[0]) > E_thr){ emit_spike(evo);}
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


