#include <stdexcept>
#include "include/neurons.hpp"
#include "include/network.hpp"
#include "include/base_objects.hpp"
#include "include/devices.hpp"


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

void PopCurrentInjector::inject(EvolutionContext * evo){
    if (!activated & (evo->now > t_min)){
        for (auto neuron : pop->neurons){
            neuron->I = this->I;
        }
        activated = true;
        // std::cout << "ACTIVATED CURRENT" << std::endl;
    }

    if (!deactivated & (evo->now >= t_max)){
        for (auto neuron : pop->neurons){
            neuron->I = 0;
        }
        deactivated = true;
        // std::cout << "DEACTIVATED CURRENT" << std::endl;
    }
}
