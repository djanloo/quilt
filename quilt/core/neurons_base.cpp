#include "include/neurons_base.hpp"
#include "include/base_objects.hpp"
#include "include/devices.hpp"
#include "include/network.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>

#include <boost/numeric/odeint.hpp>

namespace utilities{

    void nan_check(double value, const std::string& str){
        if (std::isnan(value)){
            throw std::runtime_error(str);
        }
    }

    // This one costs a lot of time
    void nan_check_vect(const std::vector<double>& vect, const std::string& str){
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

// Synapse::min_delay is used to check if the timestep is small enough
// Sets the minimim delay to infinity, take smaller values when building the network
float Synapse::min_delay = std::numeric_limits<float>::infinity();

void Synapse::fire(EvolutionContext * evo){
    // Adds a (weight, delay) spike in the spike queue of the postsynaptic neuron
    if (this->delay < evo->dt){throw std::runtime_error("Synapse has delay less than timestep");}
    this->postsynaptic->incoming_spikes.emplace(this->weight, evo->now + this->delay);
}

Neuron::Neuron(Population * population):population(population){
    
    id = HierarchicalID( population -> id);
    state = neuron_state { population->neuroparam->E_l + ((double)rand())/RAND_MAX, 0.0, 0.0};
    last_spike_time = - std::numeric_limits<float>::infinity();
    population -> neurons.push_back(this);        
};

void Neuron::connect(Neuron * neuron, double weight, double delay){
    (this -> efferent_synapses).push_back(Synapse(this, neuron, weight, delay));
}


void Neuron::handle_incoming_spikes(EvolutionContext * evo){

    // if (incoming_spikes.size() > MAX_SPIKE_QUEUE_LENGTH){
    //     std::cerr << ("WARNING - spike queue: reached length of " + std::to_string(MAX_SPIKE_QUEUE_LENGTH)) << std::endl;
    // }
    while (!(incoming_spikes.empty())){

        auto spike = incoming_spikes.top();

        // Check for missed spikes
        if ((spike.arrival_time < static_cast<float>(evo->now))&(!spike.processed)){
            std::string message = "Spike Missed\n";
            message += "Spike arrival time: " + std::to_string(spike.arrival_time) + "\n";
            message += "now t is: " + std::to_string(evo->now) + "\n";
            message += "Please reduce the timestep or increase the delays.\n";
            throw std::runtime_error(message);
        } 

        if (!(spike.processed)){

            if ((spike.arrival_time >= static_cast<float>(evo->now) ) && (spike.arrival_time < static_cast<float>(evo->now + evo->dt) )){

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
    if (spike_flag){
        on_spike(evo);
        spike_flag = false;
    }
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
}

void Neuron::emit_spike(EvolutionContext * evo){
    for (auto synapse : this->efferent_synapses){ synapse.fire(evo); }

    this -> last_spike_time = evo -> now;
    ((this->population)->n_spikes_last_step) ++;
    state[0] = population->neuroparam->V_peak;

    spike_flag = true;
    // This is done at the beginning of the next evolution
    // this-> on_spike(evo);
}

void Neuron::on_spike(EvolutionContext * /*evo*/){
    this->state[0] = this->population->neuroparam->V_reset;
}

NeuroParam::NeuroParam(){
    this->neur_type = neuron_type::base_neuron;
    std::map<std::string, float> defaults = {{"I_e", 0.0}, {"I_osc", 0.0}, {"omega_I", 0.0}};
    this->paramap = ParaMap(defaults);
    }

NeuroParam::NeuroParam(const ParaMap & paramap):NeuroParam(){

    this->paramap.update(paramap);
    this->neur_type = static_cast<neuron_type>(this->paramap.get("neuron_type"));

    std::string last = "";
    try{
        last = "E_l";
        this->E_l = this->paramap.get(last);
        last = "V_reset";
        this->V_reset = this->paramap.get(last);
        last = "V_peak";
        this->V_peak = this->paramap.get(last);

        last = "C_m";
        this->C_m = this->paramap.get(last);

        last = "tau_ex";
        this->tau_ex = this->paramap.get(last);
        last = "tau_in";
        this->tau_in = this->paramap.get(last);
        last = "E_ex";
        this->E_ex = this->paramap.get(last);
        last = "E_in";
        this->E_in = this->paramap.get(last);
        
        last = "tau_refrac";
        this->tau_refrac = this->paramap.get(last);
        last = "I_e";
        this->I_e = this->paramap.get(last);
        last = "I_osc";
        this->I_osc = this->paramap.get(last);
        last = "omega_I";
        this->omega_I = this->paramap.get(last);
    } catch (const std::out_of_range & e){
        throw std::out_of_range("Missing parameter for NeuroParam: " + last);
    }
}

void NeuroParam::add(const std::string & key, float value){paramap.add(key, value);}

