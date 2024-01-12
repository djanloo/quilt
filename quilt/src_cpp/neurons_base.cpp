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

#define MAX_SPIKE_QUEUE_LENGTH 1000

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
    state = neuron_state { population->neuroparam->E_rest + ((double)rand())/RAND_MAX, 0.0, 0.0};
    last_spike_time = - std::numeric_limits<float>::infinity();
    population -> neurons.push_back(this);        
};

void Neuron::connect(Neuron * neuron, double weight, double delay){
    (this -> efferent_synapses).push_back(Synapse(this, neuron, weight, delay));
}


void Neuron::handle_incoming_spikes(EvolutionContext * evo){

    if (incoming_spikes.size() > MAX_SPIKE_QUEUE_LENGTH){
        throw std::runtime_error("Max number of spikes reached. Something must have gone wrong.");
    }
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

    // Spike generation
    if ((state[0]) >= this->population->neuroparam->E_thr){ emit_spike(evo);}
}

void Neuron::emit_spike(EvolutionContext * evo){
    for (auto synapse : this->efferent_synapses){ synapse.fire(evo); }

    this -> last_spike_time = evo -> now;
    ((this->population)->n_spikes_last_step) ++;
    state[0] = population->neuroparam->E_thr;

    spike_flag = true;
    // This is done at the beginning of the next evolution
    // this-> on_spike(evo);
}

void Neuron::on_spike(EvolutionContext * /*evo*/){
    this->state[0] = this->population->neuroparam->E_reset;
}

NeuroParam::NeuroParam(){
                this->neur_type = neuron_type::base_neuron;
                std::map<std::string, float> defaults = {{"I_ext", 0.0}, {"I_osc", 0.0}, {"omega_I", 0.0}};
                this->paramap = ParaMap( defaults);
                std::cout << "done"<<std::endl;
                }

NeuroParam::NeuroParam(const ParaMap & paramap):NeuroParam(){

    this->paramap.update(paramap);
    this->neur_type = static_cast<neuron_type>(this->paramap.get("neuron_type"));

    std::string last = "";
    try{
        last = "E_rest";
        this->E_rest = this->paramap.get(last);
        last = "E_reset";
        this->E_reset = this->paramap.get(last);
        last = "E_thr";
        this->E_thr = this->paramap.get(last);

        last = "C_m";
        this->C_m = this->paramap.get(last);
        last = "tau_m";
        this->tau_m = this->paramap.get(last);

        last = "tau_e";
        this->tau_e = this->paramap.get(last);
        last = "tau_i";
        this->tau_i = this->paramap.get(last);
        last = "E_exc";
        this->E_exc = this->paramap.get(last);
        last = "E_inh";
        this->E_inh = this->paramap.get(last);
        
        last = "tau_refrac";
        this->tau_refrac = this->paramap.get(last);
        last = "I_ext";
        this->I_ext = this->paramap.get(last);
        last = "I_osc";
        this->I_osc = this->paramap.get(last);
        last = "omega_I";
        this->omega_I = this->paramap.get(last);
    } catch (const std::out_of_range & e){
        throw std::out_of_range("Missing parameter for NeuroParam: " + last);
    }
}

void NeuroParam::add(const std::string & key, float value){paramap.add(key, value);}

