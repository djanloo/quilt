#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"
#include "include/base.hpp"

#include <stdexcept>
#include <limits>

#define N_THREADS_INHOM_POISS_INJECT 4

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
    if (evo->dt * (rate*1e-3)  >= 1.0){
        string msg = "Error in PoissonSpikeSource. Poisson assumptions failed: rate * dt >= 1.\n";
        msg += "\trate = " + std::to_string(rate*1e-3) + "ms^{-1}\n\tdt = " + std::to_string(evo->dt) + "ms\n";
        throw std::runtime_error(msg);
    }
    float delta;

    // Quit if the system is after maximum time
    if (evo->now > this->t_max)
    {
        return;
    }

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
        currently_generated_time(0),
        perf_mgr({"injection"})
{
    // Spike time initialization
    integration_start = std::vector<double> (pop->n_neurons, 0);

    // No integration leftovers at the beginning
    integration_leftovers = std::vector<double> (pop->n_neurons, 0);

    // Initializes stochastic integration thresholds
    RNG rng;
    integration_thresholds = std::vector<double> (pop->n_neurons, 0);
    for (int i=0; i < pop->n_neurons; i++){
        integration_thresholds[i] = -std::log(rng.get_uniform());
    }

    // weights initialization
    weights = std::vector<float>(pop->n_neurons, 0);
    for (int i = 0; i < pop->n_neurons; i++)
    {
        weights[i] = weight + weight_delta * (rng.get_uniform() - 0.5);
        if (weights[i] < 0) throw std::runtime_error("Poisson spikesource weight is < 0");
    }

    // Sets the label for the performance manager
    perf_mgr.set_label("transducer of pop " + to_string(pop->id.get_id()));
}


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

// these are for debug 
static int nullcalls = 0;
static int generation = 0;
using std::to_string;

ThreadSafeFile InhomPoissonSpikeSource::outfile("test_inh_poiss.txt"); //DEBUG
Logger inhomlog("ihompoisson.log");

/**
 * Injects a partition of the target population. For multithreading.
*/
void InhomPoissonSpikeSource::_inject_partition(double now, double dt, int start_id, int end_id, RNGDispatcher * rng_disp){
    
    // DECOMMENT IF THIS METHOD CREATES TROUBLE
    // inhomlog.set_level(INFO);

    double avg_rate_in_timestep = 0; // This is for the trapezoidal rule and for checks

    bool add_spike = false;
    int seek_buffer_index;
    int last_spike_time_index;  // Time index of the last produced spike
    int timestep_done;         // Timesteps of the integration done
    int proposed_next_spike_time_index; // Time index of the newly proposed spike
    double proposed_next_spike_time;    // Time of the newly proposed spike

    int generated_spikes = 0;
    
    // vector<double> produced_spikes; //DEBUG

    // Gets an independent random number generator
    RNG * thread_rng = rng_disp->get_rng();

    inhomlog.log(DEBUG, "Starting generation " + to_string(generation));

    for (int i = start_id; i < end_id; i++)
    {
        // produced_spikes = vector<double>(0); //DEBUG
        stringstream ss;

        ss << "Starting neuron " << i <<". Current state:"<< endl;
        ss << "last_integration_extrema: " << integration_start[i] << endl;
        ss << "integration_leftovers: "<< integration_leftovers[i] << endl;
        ss << "integration threshold: "<< integration_thresholds[i] << endl; 

        last_spike_time_index = static_cast<int>(integration_start[i]/dt);
        inhomlog.log(DEBUG, ss.str());

        // If the neuron has a spike OVER this generation window, it must be skipped
        if (integration_start[i] > currently_generated_time + generation_window_length){
            inhomlog.log(DEBUG, "Neuron " + to_string(i) + " had a spike over the generation window");
            inhomlog.log(DEBUG, "\tcurrently_generated_time:" + std::to_string(currently_generated_time));
            inhomlog.log(DEBUG, "\tgeneration window:" + std::to_string(generation_window_length));
            inhomlog.log(DEBUG, "\tlast spike time:" + std::to_string(integration_start[i]));

            // Skip the neuron, the appropriate generation will take care of him
            continue;
        }

        // 
        /**
         * This loops goes on until all the window is generated.
         * 
         * The first spike outside the generation window is accepted.
         * This is necessary because of the random nature of the process.
         * 
         * Imagine that the window is 50ms and the last valid spike is generated at 45ms.
         * This function will do nothing until evo->now is over 50ms, and it may happen that
         * a spike is generated at, say, 48ms while the spiking network is at 50.
         * 
         * This will raise a "Spike in past" error. 
        */
        do{
            timestep_done = 0;
            stringstream ss;
            ss << "neuron " << i <<  ": integration of r(t) started at t = " << (last_spike_time_index + timestep_done)*dt;
            inhomlog.log(DEBUG, ss.str());
            
            // This loop goes on until the integral of the rate overcomes the exp-distributed random variable y
            // OR if the loop reaches the end of the buffer.
            // In the first case a spike is added.
            // In the latter case the integration extrema is set to the end of the window, another generation will take care of this

            while (true){

                // Remember that, for each generation, r(t) ~ rate_buffer[<int>((t-currently_generated_time)/dt)]
                // So the rate when the last spike happened is
                // r(last_spike_time) = rate_buffer[<int>((last_spike_time - currently_generated_time)/dt )]
                // so to carry out the integration from last_spike_time you have to request
                // r(last_spike_time + timesteps_done*dt) = rate_buffer[<int>((last_spike_time - currently_generated_time)/dt ) + timesteps_done]
                seek_buffer_index = last_spike_time_index + timestep_done - static_cast<int>(currently_generated_time/dt);
                if (seek_buffer_index >= rate_function_buffer.size()){        
                    // Exits with no spike generated but with modified integration extrema
                    add_spike = false;
                    break;
                }

                avg_rate_in_timestep = 0.5*(
                                            rate_function_buffer[seek_buffer_index]     +   \
                                            rate_function_buffer[seek_buffer_index + 1]     \
                                           );

                // A negative rate clearly does not make any sense.... Or does it? (vsauce reference)
                if (avg_rate_in_timestep < 0.0){
                    string msg = "Negative rate in InhomogeneousPoissonSpikeSource.\n";
                    msg += "\tt = " +std::to_string(now) + "\n";
                    throw runtime_error(msg);
                }

                // Does a check on the values of the average instantaneous rate
                // A typical interval of rates should be [10, 2000] Hz
                // if ( (avg_rate_in_timestep < 10)|(avg_rate_in_timestep > 2000) ){
                //     std::stringstream msg;
                //     msg << "Unusual value for instanteneous rate in InhomogeneousPoissonSpikeSource::inject - "
                //         << "rate = " + std::to_string(avg_rate_in_timestep)  + "Hz";
                //     get_global_logger().log(WARNING, msg.str());
                // }

                // Conversion to ms^(-1)
                avg_rate_in_timestep /= 1e3;

                // Trapezoidal rule
                integration_leftovers[i] += avg_rate_in_timestep * dt;                

                timestep_done ++;

                if(integration_leftovers[i] >= integration_thresholds[i]){
                    add_spike = true;
                    break;
                }
            }

            if (add_spike){

                // At this point Y has overcomed y
                proposed_next_spike_time_index = last_spike_time_index + timestep_done;
                proposed_next_spike_time = proposed_next_spike_time_index * dt;

                // This is real bad
                if (proposed_next_spike_time < now){
                    string msg = "A spike was produced in the past from InhomogeneousPoissonSS.\n";
                    msg+= "now : " + to_string(now) + "\n";
                    msg += "neuron: " + to_string(i) + "\n";
                    msg += "last spike produced: " + to_string(integration_start[i]) + "\n";
                    msg += "proposed spike: " + to_string(proposed_next_spike_time) + "\n";
                    inhomlog.log(ERROR, msg);
                    throw runtime_error(msg);
                }
                
                // Accepts the spike and set the integration extrema as the last produced spike
                last_spike_time_index = proposed_next_spike_time_index;
                integration_start[i] = proposed_next_spike_time;

                // produced_spikes.push_back(proposed_next_spike_time); // DEBUG

                // Adds a spike to the neuron's queue
                pop->neurons[i]->incoming_spikes.emplace(this->weights[i], proposed_next_spike_time);
                outfile.write( to_string(i) +  " "  + to_string(proposed_next_spike_time)); // DEBUG
                generated_spikes++;

                // Resets threshold and integration leftover
                integration_thresholds[i] = -std::log(thread_rng->get_uniform());
                integration_leftovers[i] = 0.0;

                stringstream ss;
                ss << "Emitted a spike at t = " << proposed_next_spike_time;
                inhomlog.log(DEBUG, ss.str());

            }else{

                // Tells the next generation that for the i-th neuron the integration
                // was already carried out up to the end of the window
                integration_start[i] = currently_generated_time + generation_window_length;

                stringstream ss;
                ss << "Window ended without spike emission. Setting integration extrema for neuron " << i << " as " << integration_start[i] 
                    << " (currently generated time: " << currently_generated_time << " and window: "<< generation_window_length << " )";
                inhomlog.log(DEBUG, ss.str());
            }


        } while (integration_start[i] < currently_generated_time + generation_window_length);

        ss.str("");ss.clear();

        ss << "Ended neuron " << i <<". Final state:"<< endl;
        ss << "last_integration_extrema: " << integration_start[i] << endl;
        ss << "integration_leftovers: "<< integration_leftovers[i] << endl;
        ss << "integration threshold: "<< integration_thresholds[i] << endl; 
        inhomlog.log(DEBUG, ss.str());
    } 

    // Frees the RNG dispatcher
    rng_disp->free();
}
/**
 * Creates (easy) population partitions and start threads to generate spikes using `InhomogeneousPoissonSPikeSource::_inject_partition()`
*/
void InhomPoissonSpikeSource::inject(EvolutionContext * evo){
    // inhomlog.set_level(INFO);
    // If we are in a time window that was already generated, do nothing

    // Measure the time from the parent thread
    perf_mgr.start_recording("injection");

    if (evo->now < currently_generated_time){
        nullcalls ++;
        return;
    }else{

        stringstream ss;
        ss <<  "InhomPoiss is injecting - now is " << evo->now << " with dt "<<evo->dt; 
        get_global_logger().log(DEBUG,ss.str());

        inhomlog.log(DEBUG, "Before this generation, " + std::to_string(nullcalls) + " null calls were performed");
        inhomlog.log(DEBUG, "\tnow:" + std::to_string(evo->now));
        inhomlog.log(DEBUG, "\tcurrently_generated_time:" + std::to_string(currently_generated_time));
        inhomlog.log(DEBUG, "Starting generation " + std::to_string(generation) + " - [ " + to_string(currently_generated_time) + " -> " + to_string(currently_generated_time + generation_window_length) + "]");
    }

    // Evaluates the rate function and stores it in the buffer
    int rate_func_buf_size = static_cast<int>(generation_window_length/evo->dt);
    rate_function_buffer = vector<double>(rate_func_buf_size, -1.0);

    // Copies the discretised rate function. The interval that will be used is [now, now + generation_window_length]
    // Since the generation is called just after now > currently_generated_time
    // the interval is [currently_generated_time, currently_generated_time + 2*generation_window_length].
    // For each generation, r(t) ~ rate_buf[<int>((t-currently_generated_time)/dt)]
    for (int i = 0; i < rate_func_buf_size; i++){
        rate_function_buffer[i] = rate_function(currently_generated_time + i*evo->dt);
        // cout << rate_function_buffer[i] << " ";
    }
    stringstream ss;
    ss << "Getting rate of incoming oscillators from t= "<<currently_generated_time << " to t= "<<currently_generated_time + rate_func_buf_size*evo->dt;
    get_global_logger().log(DEBUG, ss.str());
    // cout << endl;
    
    int n_threads = N_THREADS_INHOM_POISS_INJECT;
    if (pop->n_neurons < 50){
        n_threads = 1;
    }

    std::vector<std::thread> threads;

    // NOTE: seeding is disabled for now, use random source
    // TODO: add a global management of seed
    RNGDispatcher rng_dispatcher(n_threads);

    for (int i=0; i < n_threads; i++){
        inhomlog.log(DEBUG, "starting thread " + to_string(i) + " || neurons [" + to_string(i*pop->n_neurons/n_threads) + "," + to_string((i+1)*pop->n_neurons/n_threads-1));
        threads.emplace_back(&InhomPoissonSpikeSource::_inject_partition, this ,
                            evo->now, evo->dt,
                            i*pop->n_neurons/n_threads, (i+1)*pop->n_neurons/n_threads-1,
                            &rng_dispatcher);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // SINGLE THREAD FOR DEBUG
    // _inject_partition(evo->now, evo->dt, 0, pop->n_neurons);

    // If the window generation is finished, updates the generated time
    currently_generated_time += generation_window_length;
    generation++;

    // Ends time measure
    perf_mgr.end_recording("injection");
}