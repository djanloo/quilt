#include "include/neurons_base.hpp"
#include "include/devices.hpp"
#include "include/network.hpp"

#include <map>
#include <chrono>
#include <cmath>
#include <stdexcept>


#include <boost/numeric/odeint.hpp>

// Synapse::min_delay is used to check if the timestep is small enough
// Sets the minimim delay to infinity, take smaller values when building the network
float Synapse::min_delay = std::numeric_limits<float>::infinity();

void Synapse::fire(EvolutionContext * evo){
    // Adds a (weight, now + delay) spike in the spike queue of the postsynaptic neuron
    if (this->delay < evo->dt){throw std::runtime_error("Synapse has delay less than timestep: " + std::to_string(delay));}
    this->postsynaptic->incoming_spikes.emplace(weight, evo->now + delay);
}

int Synapse::get_efferent_pop_id(){ return postsynaptic->population->id.get_id();}


Neuron::Neuron(Population * population):population(population){
    id = HierarchicalID(population->id);
    state = dynamical_state { population->neuroparam->E_l + ((double)rand())/RAND_MAX, 0.0, 0.0};
    last_spike_time = - std::numeric_limits<float>::infinity();
    spike_flag = false;
    population -> neurons.push_back(this);        
};

void Neuron::connect(Neuron * neuron, double weight, double delay){
    (this->efferent_synapses).push_back(Synapse(this, neuron, weight, delay));
}


void Neuron::handle_incoming_spikes(){

    while (!(incoming_spikes.empty())){ // This loop will be broken later

        auto spike = incoming_spikes.top();

        // Check for missed spikes
        if ((spike.arrival_time < static_cast<float>(evo->now))&(!spike.processed)){
            std::string message = "Spike Missed  in neuron (" + std::to_string(this->population->id.get_id()) + "," + std::to_string(this->id.get_id()) + ") ";
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

                // Removes the spike from the incoming spikes
                incoming_spikes.pop();
            } else {

                // If a spike is not to process, neither the rest will be
                break;
            }
        }else{
            throw std::runtime_error("Spike was processed two timesss");
        }
    }
}


void Neuron::evolve(){
    if (spike_flag){
        on_spike();
        spike_flag = false;
    }

    // Evolve
    boost::numeric::odeint::runge_kutta4<dynamical_state> stepper;
    auto lambda = [this](const dynamical_state & state, dynamical_state & dxdt, double t) {
                                    this->evolve_state(state, dxdt, t);
                                };

    stepper.do_step(lambda, this->state, evo->now, evo->dt);
}

void Neuron::emit_spike(){
    for (auto synapse : this->efferent_synapses){ synapse.fire(evo); }

    this -> last_spike_time = evo -> now;
    ((this->population)->n_spikes_last_step) ++;
    state[0] = population->neuroparam->V_peak;

    spike_flag = true;

}

void Neuron::on_spike(){
    this->state[0] = this->population->neuroparam->V_reset;
}

NeuroParam::NeuroParam(){
    this->neur_type = "base_neuron";
    std::map<std::string, ParaMap::param_t> defaults{{"I_e", 0.0f}, {"I_osc", 0.0f}, {"omega_I", 0.0f}};
    this->paramap = ParaMap(defaults);
    }

NeuroParam::NeuroParam(ParaMap & paramap) : NeuroParam(){

    this->paramap.update(paramap);
    neur_type = paramap.get<string>("neuron_type");

    // Soma
    E_l = paramap.get("E_l", -70.0f);
    C_m = paramap.get("C_m", 15.0f);
    V_reset = paramap.get("V_reset", -60.0f);
    V_peak = paramap.get("V_peak", 5.0f);
    tau_refrac = paramap.get("tau_refrac", 0.0f);

    // Synapses
    tau_ex = paramap.get("tau_ex", 12.0f);
    tau_in = paramap.get("tau_in", 10.0f);
    E_ex = paramap.get("E_ex", 0.0f);
    E_in = paramap.get("E_in", -80.0f);
    
    // External inputs (default is zero)
    // Note: omitting the float 'f' will cause runtime errors
    I_e = paramap.get("I_e", 0.0f);
    I_osc = paramap.get("I_osc", 0.0f);
    omega_I = paramap.get("omega_I", 0.0f);

}

// 
void NeuroParam::add(const std::string & key, float value){paramap.add(key, value);}

// Initialises the Neurofactory instance to null pointer
NeuroFactory * NeuroFactory::instance = nullptr;
