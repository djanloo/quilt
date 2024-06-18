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
        t_min(t_min),
        rng() // Uses a random device to initialize the generator
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
    weights = std::vector<float>(pop->n_neurons, 0);

    for (int i = 0; i < pop->n_neurons; i++)
    {
        weights[i] = weight + weight_delta * (rng.get_uniform() - 0.5);
        if (weights[i] < 0) throw std::runtime_error("Poisson spikesource weight is < 0");
    }

};

void PoissonSpikeSource::set_rate(float new_rate){ 
    if (new_rate < 0){
        throw std::invalid_argument("Setting a negative rate of a PoissonSpikeSource");
    }
    rate = new_rate; 
}


void PoissonSpikeSource::inject(EvolutionContext * evo)
{
    if (evo->dt * rate >= 1.0){
        string msg = "Error in PoissonSpikeSource. Poisson assumptions failed: rate * dt >= 1.\n";
        msg += "\trate = " + std::to_string(rate) + "\n\tdt = " + std::to_string(evo->dt) + "\n";
        throw std::runtime_error(msg);
    }
    float delta;

    // Quit if the system is after maximum time
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

            delta = -std::log(rng.get_uniform())/this->rate * 1000;
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


InhomPoissonSpikeSource::InhomPoissonSpikeSource( Population * pop, 
                                                std::function<double(double)> rate_function, 
                                                float weight, float weight_delta, double generation_window_length)
    :   PopInjector(pop),
        rate_function(rate_function),
        weight(weight), 
        weight_delta(weight_delta),
        generation_window_length(generation_window_length)
{
    // Spike time initialization
    next_spike_times = std::vector<double> (pop->n_neurons, 0);

    // weights initialization
    weights = std::vector<float>(pop->n_neurons, 0);
    for (int i = 0; i < pop->n_neurons; i++)
    {
        weights[i] = weight + weight_delta * (rng.get_uniform() - 0.5);
        if (weights[i] < 0) throw std::runtime_error("Poisson spikesource weight is < 0");
    }

    cout << "Inhomogeneous Poisson SpikeSource initialized"<< endl; 
}

std::ofstream InhomPoissonSpikeSource::outfile("test_inh_poiss.txt");

void InhomPoissonSpikeSource::inject(EvolutionContext * evo){
    
    double y; // The threshold for spike generation
    double Y; // The cumulative variable 
    double avg_rate_in_timestep = 0;

    int last_spike_time_index, timestep_passed;

    bool abort_generation = false;

    for (int i = 0; i < pop->n_neurons; i++)
    {
        cout << "InhomPoiss: generating spikes for neuron "<< i << endl;
        cout << "current last spike time is "<< next_spike_times[i]<<endl;
        last_spike_time_index = evo->index_of(next_spike_times[i]);
        cout << "index of the last spike time is "<< last_spike_time_index << endl;

        while (next_spike_times[i] < evo->now + generation_window_length)
        {
            y = -std::log(rng.get_uniform());
            Y = 0; 
            timestep_passed = 0;
            cout << "-------------------------"<<endl;
            cout << "\tneuron "<< i <<endl;
            cout << "\tlast spike time = "<<next_spike_times[i]<<endl; 
            cout << "\ty = "<< y << endl;
            cout << "\tY = ";

            while (true){ // This cycle continues until Y reaches y
                // Cumulative integral of the rate function
                // adds int_{now}^{now+dt} to the sum
                // using trapezoidal rule
                avg_rate_in_timestep = 0.5*(rate_function(evo->now + (last_spike_time_index + timestep_passed)*evo->dt) + \
                                        rate_function(evo->now + (last_spike_time_index + timestep_passed + 1)*evo->dt)\
                                        );

                if (avg_rate_in_timestep < 0.0){
                    string msg = "Negative rate in InhomogeneousPoissonSpikeSource.\n";
                    msg += "\tt = " +std::to_string( evo->now) + "\n";
                    throw runtime_error(msg);
                }
                // cout << "avg_rate is "<< avg_rate_in_timestep << "Hz (";
                // Conversion to ms^(-1)
                avg_rate_in_timestep /= 1e3;
                // cout << avg_rate_in_timestep << " ms-1)" <<endl;
                Y += avg_rate_in_timestep * evo->dt; // Trapezoidal rule                

                timestep_passed ++;
                cout << Y << " -> ";
                // If the cumulative integral overcomes the y-value, a spike is generated ad that time
                if (Y >= y) break;
                if ( (last_spike_time_index + timestep_passed)*evo->dt > generation_window_length){
                    abort_generation = true;
                    cout << "Aborting generation"<<endl;
                    break;
                }
            }
            if (!abort_generation){
                cout << "thr overcomed in " << timestep_passed << " timesteps"<< endl;

                last_spike_time_index += timestep_passed;

                // Adds a spike to the neuron
                next_spike_times[i] = evo->now + last_spike_time_index * evo->dt;
                cout << "Adding a spike to neuron "<< i << " at time " << next_spike_times[i]<<endl;
                outfile << i << " " << next_spike_times[i] << endl;
                pop->neurons[i]->incoming_spikes.emplace(this->weights[i], next_spike_times[i]);
            }else{
                break;
            }
        }

        cout << "Reached generating window end"<<endl;
    }
}