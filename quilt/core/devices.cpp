#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"
#include "include/base.hpp"

#include <stdexcept>
#include <limits>


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
        pop->neuroparam->I_e = I;
        activated = true;
        // std::cout << "ACTIVATED CURRENT" << std::endl;
    }

    if (!deactivated & (evo->now >= t_max)){
        pop->neuroparam->I_e = 0.0;
        deactivated = true;
        // std::cout << "DEACTIVATED CURRENT" << std::endl;
    }
}

PoissonSpikeSource::PoissonSpikeSource(Population * pop, float rate, float weight, float weight_delta, double t_min, double t_max): 
                                                PopInjector(pop),
                                                rate(rate), weight(weight), weight_delta(weight_delta), 
                                                t_min(t_min){
    next_spike_times = std::vector<double> (pop->n_neurons, t_min);
    if (t_max<t_min){
        this->t_max = std::numeric_limits<double>::infinity();
    }else{
        this->t_max = t_max;
    }
    RNG rng(8);
    weights = std::vector<float>(pop->n_neurons, 0);

    for (int i = 0; i < pop->n_neurons; i++){
        weights[i] = weight + weight_delta * (rng.get_uniform() - 0.5);
        if (weights[i] < 0) throw std::runtime_error("Poisson spikesource weight is < 0");
    }

};

// std::ofstream PoissonSpikeSource::outfile = std::ofstream("spikes.txt");

void PoissonSpikeSource::inject(EvolutionContext * evo){
    float delta;

    if (evo->now > this->t_max){return;}

    for (int i = 0; i < pop->n_neurons; i++){
        
        while (next_spike_times[i] < evo->now + evo->dt){
            // TODO: use pcg
            delta = -std::log(static_cast<float>(rand())/RAND_MAX)/this->rate * 1000;
            next_spike_times[i] += delta;
            if (next_spike_times[i] < evo->now) {throw std::runtime_error("Poisson spike produced in past");}
            pop->neurons[i]->incoming_spikes.emplace(this->weights[i], next_spike_times[i]);
        }
    }
}