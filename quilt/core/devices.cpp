#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"
#include "include/base_objects.hpp"

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


PoissonSpikeSource::PoissonSpikeSource(Population * pop, float rate, float weight, double t_min, double t_max): 
                                                PopInjector(pop),
                                                rate(rate), weight(weight), 
                                                t_min(t_min){
    next_spike_times = std::vector<double> (pop->n_neurons, t_min);
    if (t_max<t_min){
        this->t_max = std::numeric_limits<double>::infinity();
    }else{
        this->t_max = t_max;
    }
};

// std::ofstream PoissonSpikeSource::outfile = std::ofstream("spikes.txt");

void PoissonSpikeSource::inject(EvolutionContext * evo){
    float delta;
    float avg_delta = 0;
    float generated_spikes = 0;

    if (evo->now > this->t_max){return;}

    // std::cout << "Generating spikes" << std::endl;
    for (int i = 0; i < pop->n_neurons; i++){
        
        while (next_spike_times[i] < evo->now + evo->dt){ // If the last emitted spike was received, emit a new one

            // std::cout << "t: "<<evo->now<<  " <--> Adding poisson spike at neuron " << i << " of pop " << pop->id.get_id();
            delta = -std::log(static_cast<float>(rand())/RAND_MAX)/this->rate * 1000;
            // if (delta < evo->dt){
            //     // std::cout << "PoissonSpikeSource generated a time smaller than timestep" << std::endl;
            //     delta = evo->dt;
            // }
            // std::cout << " -- delta: "<< delta; 
            avg_delta += delta;
            generated_spikes ++;

            next_spike_times[i] += delta;
            // std::cout << " -- next t: " << next_spike_times[i] << std::endl;
            // outfile << pop->id.get_id() << " " << i << " " << next_spike_times[i] << std::endl;
            if (next_spike_times[i] < evo->now) {std::cerr << "Spike produced in past" << std::endl;}
            pop->neurons[i]->incoming_spikes.emplace(this->weight, next_spike_times[i]);
        }
    }
    // std::cout << "Done generating spikes" << std::endl;
    // std::cout << "AVG delta T: "<< avg_delta/generated_spikes << std::endl;
}