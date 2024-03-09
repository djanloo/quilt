#include "include/neurons_base.hpp"
#include "include/devices.hpp"
#include "include/network.hpp"

#include <map>
#include <chrono>
#include <cmath>
#include <stdexcept>


#include <boost/numeric/odeint.hpp>


// namespace utilities{

//     void nan_check(double value, const std::string& str){
//         if (std::isnan(value)){
//             throw std::runtime_error(str);
//         }
//     }

//     // This one costs a lot of time
//     void nan_check_vect(const std::vector<double>& vect, const std::string& str){
//         std::vector<bool> are_nan;
//         bool somebody_is_nan = false;

//         for (auto value : vect){
//             somebody_is_nan = somebody_is_nan || std::isnan(value) || std::isinf(value);
//             are_nan.push_back( std::isnan(value) || std::isinf(value));
//         }
//         if (somebody_is_nan){
//             std::cerr << "vector is nan: [" ;
//             for (auto val : are_nan) {std::cerr << val <<" ";} std::cerr << " ]" << std::endl;
            
//             throw std::runtime_error(str);
//         }
//     }
// }

// Synapse::min_delay is used to check if the timestep is small enough
// Sets the minimim delay to infinity, take smaller values when building the network
float Synapse::min_delay = std::numeric_limits<float>::infinity();

void Synapse::fire(EvolutionContext * evo){
    // Adds a (weight, now + delay) spike in the spike queue of the postsynaptic neuron
    if (this->delay < evo->dt){throw std::runtime_error("Synapse has delay less than timestep: " + std::to_string(delay));}
    this->postsynaptic->incoming_spikes.emplace(weight, evo->now + delay);
}

Neuron::Neuron(Population * population):population(population){
    id = HierarchicalID(population->id);
    state = dynamical_state { population->neuroparam->E_l + ((double)rand())/RAND_MAX, 0.0, 0.0};
    last_spike_time = - std::numeric_limits<float>::infinity();
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

    // Process incoming spikes from t and t+dt
    // Note: it is conceptually wrong to first evaluate incoming spikes and then do the step
    // because EXACTLY at time t the spikes are not arrived yet
    // The other way of doing this is to evaluate at the beginning of the evolution the spikes
    // that arrived from t-dt and t
    handle_incoming_spikes();

    // THIS CHECKS NANS
    // auto before_step = state;

    // Checks for NaNs after the step
    // try{
    //     stepper.do_step(lambda, this->state, evo->now, evo->dt);
    //     utilities::nan_check_vect(this->state, "NaN in neuron state");

    // }catch (const std::runtime_error &e){
    //     std::cerr << "State before step: ";
    //     for (auto val : before_step){ std::cerr << val << " ";}
    //     std::cerr << std::endl;
    //     throw e;
    // }
}

void Neuron::emit_spike(){
    for (auto synapse : this->efferent_synapses){ synapse.fire(evo); }

    this -> last_spike_time = evo -> now;
    ((this->population)->n_spikes_last_step) ++;
    state[0] = population->neuroparam->V_peak;

    spike_flag = true;
    // This is done at the beginning of the next evolution
    // this-> on_spike(evo);
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
    E_l = paramap.get<float>("E_l");
    C_m = paramap.get<float>("C_m");
    V_reset = paramap.get<float>("V_reset");
    V_peak = paramap.get<float>("V_peak");
    tau_refrac = paramap.get<float>("tau_refrac");

    // Synapses
    tau_ex = paramap.get<float>("tau_ex");
    tau_in = paramap.get<float>("tau_in");
    E_ex = paramap.get<float>("E_ex");
    E_in = paramap.get<float>("E_in");
    
    // External inputs (default is zero)
    // Note: omitting the float 'f' will cause runtime errors
    I_e = paramap.get("I_e", 0.0f);
    I_osc = paramap.get("I_osc", 0.0f);
    omega_I = paramap.get("omega_I", 0.0f);

}

void NeuroParam::add(const std::string & key, float value){paramap.add(key, value);}

