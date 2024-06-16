#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"
#include "include/base.hpp"

#include <stdexcept>
#include <limits>
/**
 * Disclaimer: this section is cumbersome due to the redundancy of the 'get_history()' methods.
 * For now this is somewhat a patch, in future I will make it more clean 
 * (for the procrastination demon: today is feb 29 - 2024 )
*/

// ***************************************** MONITORS ***********************************************
void PopulationRateMonitor::gather()
{
    double rate = monitored_population->n_spikes_last_step;
    rate /= evo->dt;
    rate /= monitored_population->n_neurons;
    history.push_back(rate);
}
vector<float> PopulationRateMonitor::get_history()
{
    return history;
}

void PopulationSpikeMonitor::gather()
{
    this->history.push_back(this->monitored_population->n_spikes_last_step);
}
vector<int> PopulationSpikeMonitor::get_history()
{
    return history;
}

void PopulationStateMonitor::gather()
{
    std::vector<dynamical_state> current_state;

    for (auto neuron : this->monitored_population->neurons){
        current_state.push_back(neuron->get_state());
    }
    this->history.push_back(current_state);
}  
vector<vector<dynamical_state>> PopulationStateMonitor::get_history()
{
    return history;
}
/***************************************** INJECTORS *******************************************/
void PopCurrentInjector::inject(EvolutionContext * evo)
{
    if (!activated & (evo->now > t_min)){
        pop->neuroparam->I_e = I;
        activated = true;
    }

    if (!deactivated & (evo->now >= t_max)){
        pop->neuroparam->I_e = 0.0;
        deactivated = true;
    }
}

PoissonSpikeSource::PoissonSpikeSource(Population * pop, float rate, float weight, float weight_delta, double t_min, double t_max)
    :   PopInjector(pop),
        rate(rate), 
        weight(weight), 
        weight_delta(weight_delta), 
        t_min(t_min)
{
    if (t_min < 0){
        throw std::invalid_argument("t_min was set to negative ( " + std::to_string(t_min) + ") in PoissonSpikeSource");
    }

    next_spike_times = std::vector<double> (pop->n_neurons, t_min);
    
    if (t_max<t_min)
    {
        this->t_max = std::numeric_limits<double>::infinity();
    }
    else
    {
        this->t_max = t_max;
    }
    RNG rng(8);
    weights = std::vector<float>(pop->n_neurons, 0);

    for (int i = 0; i < pop->n_neurons; i++)
    {
        weights[i] = weight + weight_delta * (rng.get_uniform() - 0.5);
        if (weights[i] < 0) throw std::runtime_error("Poisson spikesource weight is < 0");
    }

};

void PoissonSpikeSource::inject(EvolutionContext * evo)
{
    float delta;

    if (evo->now > this->t_max)
    {
        return;
    }

    // cout << "Check PoissonSpikeSource next_spike_time. Now is " <<evo->now << endl;
    // for (int i = 0 ; i < pop->n_neurons; i++){
    //     cout << next_spike_times[i] << " ";
    // }
    // cout << endl;

    for (int i = 0; i < pop->n_neurons; i++)
    {
        
        while (next_spike_times[i] < evo->now + evo->dt)
        {
            // TODO: use pcg
            delta = -std::log(static_cast<float>(rand())/RAND_MAX)/this->rate * 1000;
            if (delta < 0){
                string msg = "Poisson time increment was < 0.\n";
                msg += "\tdelta_t = " + std::to_string(delta) + "\n";
                msg += "\trate = " + std::to_string(rate) + "\n"; 

                throw std::runtime_error("");
            }
            next_spike_times[i] += delta;
            if (next_spike_times[i] < evo->now) {
                string msg = "Poisson spike produced in past.\n";
                msg += "Now is " + std::to_string(evo->now) + " while t_next_spike is " + std::to_string(next_spike_times[i]);
                throw std::runtime_error(msg);
                }
            pop->neurons[i]->incoming_spikes.emplace(this->weights[i], next_spike_times[i]);
        }
    }
}