#include <stdexcept>
#include "include/neurons.hpp"
#include "include/network.hpp"
#include "include/base_objects.hpp"
#include "include/devices.hpp"

void PopCurrentInjector::inject(EvolutionContext * evo){
    throw std::runtime_error("PopCurrentInjector is now deprecated!");
    // if (evo->now < this->t_max){
    //     for (auto neuron : this-> pop->neurons){
    //         neuron->state[0] += (this->I) * (evo->dt);
    //     }
    // }
}

void PopulationSpikeMonitor::gather(){
    this->history.push_back(this->monitored_pop->n_spikes_last_step);
}

void PopulationStateMonitor::gather(){
    std::vector<neuron_state> current_state;

    for (auto neuron : this->monitored_pop->neurons){
        current_state.push_back(neuron->get_state());
    }
    this->history.push_back(current_state);
}  
