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
        generation_window_length(generation_window_length),
        currently_generated_time(0)
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

/**
 * 
 * Disclaimer: this function is messy as hell.
 * 
 * If you, poor thing, are here to understand what I wrote on jun 19 2024, It might be a good idea
 * to look at Cox & Gabbiani - Mathematics for Neuroscients (chap 16.4).
 * 
 * I'll give you (and me) a brief description of my implementation.
 * 
 * Assumptions: 
 *  1- you have a rate function r(t)
 *  2- you want to generate a train of inhomogeneous poisson spikes that follow that rate function
 *  3- (most tricky) you can't generate too much spikes
 * 
 * Assumption 3 comes from the fact that spiking populations hate to have a long spike queue and its the 
 * most bastard thing to implement. In first approximation I decide to do the thing "in fits and starts".
 * 
 * Let's call T_gen the window for generating spikes. When the spiking network is started you have t=0.
 * 
 * Automatically, during evolution, `inject()` is called for each injector, so on the first evolution 
 * step this method is called for the first time.
 * At this point, here we generate spikes that have rate r(t) up to T_gen.
 * 
 * Generation recipe (Cox and Gabbiani):
 *  - get a exp-distributed (avg = 1) number
 *  - integrate r(t) until the area reaches that number
 *  - that time is the spike
 *  - do it again
 * 
*/
void InhomPoissonSpikeSource::inject(EvolutionContext * evo){
    

    Logger &logger = get_global_logger();
    double avg_last_spike_time = 0;

    // If the spikes have already been generated, does nothing
    // This can cause discontinuities when time crosses a window
    // TODO: check how much or set generation_window = dt
    if (evo->now < currently_generated_time){
        // get_global_logger().log(INFO, "InhomPoisson did not inject  because spikes are already generated up to " + std::to_string(currently_generated_time) + " ms (now is " + std::to_string(evo->now) + " ms");
        return;
    }else{
        logger.log(INFO, "Generating inhomogeneous poisson spikes:");
        logger.log(INFO, "\tnow:" + std::to_string(evo->now));
        logger.log(INFO, "\tcurrently_generated_time:" + std::to_string(currently_generated_time));
        logger.log(INFO, "\tgeneration window:" + std::to_string(generation_window_length));

        for (int i =0 ;i< pop->n_neurons; i++){
            avg_last_spike_time += next_spike_times[i];
        }
        avg_last_spike_time/=pop->n_neurons;
        
        logger.log(INFO, "\taverage last spike time generated:" + std::to_string(avg_last_spike_time));
    }


    double y; // The threshold for spike generation
    double Y; // The cumulative variable 
    double avg_rate_in_timestep = 0;

    int last_spike_time_index, timestep_passed, proposed_next_spike_time_index;
    double proposed_next_spike_time;

    bool abort_neuron = false;
    int generated_spikes = 0;

    logger.log(INFO, "Generating inhomogeneous spikes for " + std::to_string(pop->n_neurons) + " neurons for " + std::to_string(generation_window_length*1e-3) + " seconds");
    
    for (int i = 0; i < pop->n_neurons; i++)
    {
        last_spike_time_index = evo->index_of(next_spike_times[i]);

        // Resets the abort flag
        abort_neuron = false;

        if (next_spike_times[i] > currently_generated_time + generation_window_length){
            logger.log(WARNING, "This warning should not be here. Something is wrong!");
        }

        while (true) // Adjust this: it's ugly
        {
            y = -std::log(rng.get_uniform());
            Y = 0; 
            timestep_passed = 0;
            // logger.log(INFO, "\tlast_spike_time_index:" + std::to_string(last_spike_time_index) + "\n");


            // This loop goes on until the integral of the rate overcomes the exp-distributed random variable y
            while (Y <= y){
                
                // Cumulative integral of the rate function
                // adds int_{now}^{now+dt} rate(t) dt to the sum
                // using trapezoidal rule
                avg_rate_in_timestep = 0.5*(
                                            rate_function(evo->now + (last_spike_time_index + timestep_passed)*evo->dt) + \
                                            rate_function(evo->now + (last_spike_time_index + timestep_passed + 1)*evo->dt)\
                                           );

                // A negative rate clearly does not make any sense.... Or does it? (vsauce reference)
                if (avg_rate_in_timestep < 0.0){
                    string msg = "Negative rate in InhomogeneousPoissonSpikeSource.\n";
                    msg += "\tt = " +std::to_string( evo->now) + "\n";
                    throw runtime_error(msg);
                }

                // Does a check on the values of the average instantaneous rate
                // A typical interval of rates should be [10, 2000] Hz
                if ( (avg_rate_in_timestep < 10)|(avg_rate_in_timestep > 2000) ){
                    string msg = "Unusual value for instanteneous rate in InhomogeneousPoissonSpikeSource::inject\n";
                    msg += "\trate = " + std::to_string(avg_rate_in_timestep)  + "Hz\n";
                    logger.log(WARNING, msg);
                }

                // Conversion to ms^(-1)
                avg_rate_in_timestep /= 1e3;

                // Trapezoidal rule
                Y += avg_rate_in_timestep * evo->dt;                

                timestep_passed ++;
            }

            // At this point Y has overcomed y
            proposed_next_spike_time_index = last_spike_time_index + timestep_passed;
            proposed_next_spike_time = proposed_next_spike_time_index * evo->dt;
            
            last_spike_time_index = proposed_next_spike_time_index;
            next_spike_times[i] = proposed_next_spike_time;

            // logger.log(INFO, "\tproposed_next_spike_time_index:" + std::to_string(proposed_next_spike_time_index) + "\n");

            // Adds a spike to the neuron's queue
            pop->neurons[i]->incoming_spikes.emplace(this->weights[i], next_spike_times[i]);
            outfile << i << " " << next_spike_times[i] << endl;
            generated_spikes++; 
            // logger.log(INFO, "Generated spike for neuron " + std::to_string(i) + " at t = " + std::to_string(next_spike_times[i]) + " ms");

            /**
             * The first spike outside the generation window is accepted
             * This is necessary because of the random nature of the process
             * 
             * Imagine that the window is 50ms and the last valid spike is generated at 45ms.
             * This function will do nothing until evo->now is over 50ms, and it may happen that
             * a spike is generated at, say, 48ms while the spiking network is at 50.
             * 
             * This will raise a "Spike in past" error. 
            */
            if (proposed_next_spike_time > currently_generated_time + generation_window_length){
                abort_neuron = true;
            }

            // If the neuron is done, stops and does the next
            if (abort_neuron){
                // get_global_logger().log(INFO, "Aborting generation for neuron " + std::to_string(i));
                break;
            }

        }

        // cout << "Reached generating window end"<<endl;
    }

    // If the window generation is finished, updates the generated time
    currently_generated_time += generation_window_length;
    logger.log(INFO, "Generated " + std::to_string(generated_spikes) + " spikes");

    avg_last_spike_time = 0;
    for (int i =0 ;i< pop->n_neurons; i++){
            avg_last_spike_time += next_spike_times[i];
        }
    avg_last_spike_time/=pop->n_neurons;
    logger.log(INFO, "Last spike generated is on average at t = " + std::to_string(avg_last_spike_time) + " ms");

}