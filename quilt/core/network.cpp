#include "include/network.hpp"
#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/devices.hpp"
#include "include/neuron_models.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>
#include <string>
#include <thread>
#include <boost/timer/progress_display.hpp>

using std::cout;
using std::endl;

Population::Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network):
    n_neurons(n_neurons),n_spikes_last_step(0){
    
    // Add itself to the hierarchical structure
    id = HierarchicalID(spiking_network->id);

    // Adds itself to the spiking network populations
    (spiking_network->populations).push_back(this);

    try{ 
        params->get("neuron_type");
    }catch (const std::out_of_range & e) {
        throw( std::out_of_range("Neuron params must havce field neuron_type"));
    }
    
    switch(static_cast<neuron_type> (static_cast<int>(params->get("neuron_type")))){
        case neuron_type::aqif: 
            this->neuroparam = new aqif_param(*params);

            break;
        case neuron_type::aqif2: 
            this->neuroparam = new aqif2_param(*params);
            break;
        case neuron_type::izhikevich:
            this->neuroparam = new izhikevich_param(*params);
            break;
        case neuron_type::aeif: 
            this->neuroparam = new aeif_param(*params);
            break;
        default:
            throw std::runtime_error("Invalid neuron type when building population:" + std::to_string(static_cast<int>(params->get("neuron_type"))));
            break;
    };

    neuron_type neur_type = neuroparam->get_neuron_type();

    for ( int i = 0; i < n_neurons; i++){
        // This can be avoided, probably using <variant>
        switch(neur_type){
        case neuron_type::base_neuron:  new Neuron(this);           break;   // remember not to push_back here
        case neuron_type::aqif:         new aqif_neuron(this);      break;   // calling the constructor is enough
        case neuron_type::izhikevich:   new izhikevich_neuron(this);break;
        case neuron_type::aeif:         new aeif_neuron(this);      break;
        case neuron_type::aqif2:        new aqif2_neuron(this);     break;
        default:
            throw std::runtime_error("Invalid neuron type");
        };
    }

    // this->print_info();
    }

Projection::Projection(float ** weights, float ** delays, unsigned int start_dimension, unsigned int end_dimension):
    weights(weights), delays(delays), start_dimension(start_dimension), end_dimension(end_dimension){

    int n_links = 0;

    for (unsigned int i = 0; i < start_dimension; i++){
        for (unsigned int j =0 ; j< end_dimension; j++){
            if (std::abs(weights[i][j]) >= WEIGHT_EPS){ 
                n_links ++;
            }
        }
    }
    // std::cout << "Projection has density " << ((float)n_links)/start_dimension/end_dimension * 100 << "%" << std::endl;
}

void Population::project(Projection * projection, Population * efferent_population){
    int connections = 0;
    // auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < projection -> start_dimension; i++){
        for (unsigned int j = 0; j < projection -> end_dimension; j++){
            if (std::abs((projection -> weights)[i][j]) > WEIGHT_EPS){
                connections ++;
                (this -> neurons)[i] -> connect(efferent_population -> neurons[j], (projection -> weights)[i][j], (projection -> delays)[i][j]);
            }
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Performing " << connections << " connections took ";
    // std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end -start)).count() << " ms   (";
    // std::cout << ((double)(std::chrono::duration_cast<std::chrono::microseconds>(end -start)).count())/connections << " us/link)" << std::endl;
}

/**
 * 
 * This function evolves a bunch of neurons, from <from> to <to>.
 * It must be thread safe.
 * 
*/
void Population::evolve_bunch(EvolutionContext * evo, unsigned int from, unsigned int to){
    for (unsigned int i = from; i< to; i++){
        this->neurons[i]->evolve(evo);
    }
}

void Population::evolve(EvolutionContext * evo){

    // Splits the work in equal parts using 4 threads
    std::vector<unsigned int> bunch_starts(4), bunch_ends(4);
    for (int i = 0; i < 4; i++){
        bunch_starts[i] = i*static_cast<unsigned int>(this->n_neurons)/4;
        bunch_ends[i] = (i + 1)*static_cast<unsigned int>(this->n_neurons)/4 - 1;
        // std::cout << "bunch start[i]: " << bunch_starts[i]<<std::endl;
        // std::cout << "bunch end[i]:" << bunch_ends[i] << std::endl;
    }

    // Ensures that all neurons are covered
    bunch_ends[3] = this->n_neurons-1;
    
    // Starts four threads
    std::vector<std::thread> evolver_threads(4);
    for (int i = 0; i < 4; i++){
        evolver_threads[i] = std::thread(&Population::evolve_bunch, this, evo, bunch_starts[i], bunch_ends[i] );
    }
    // Waits four threads
    for (int i = 0; i < 4; i++){
        evolver_threads[i].join();
    }

    // TODO: spike emission is moved here in the population evolution because 
    // it's not thread safe. Accessing members of other instances requires
    // a memory access control.
    this->n_spikes_last_step = 0;
    
    for (auto neuron : this->neurons){
        if ((neuron->getV()) >= neuroparam->V_peak){neuron->emit_spike(evo);}
    }
}

void Population::print_info(){
    cout << "Population "<< this->id.get_id() << " infos:"<< endl;
    cout << "\tN:" << this->n_neurons << endl;
    cout << "\tparams:" << endl;
    for (auto couple : this->neuroparam->paramap.value_map){
        cout << "\t\t" << couple.first << "\t" << couple.second << endl;
    }
 }

SpikingNetwork::SpikingNetwork(){
    this->id = new HierarchicalID();
}

PopulationSpikeMonitor * SpikingNetwork::add_spike_monitor(Population * population){
            PopulationSpikeMonitor * new_monitor = new PopulationSpikeMonitor(population);
            this->population_spike_monitors.push_back(new_monitor);
            return new_monitor;
            };

PopulationStateMonitor * SpikingNetwork::add_state_monitor(Population * population){
    PopulationStateMonitor * new_monitor = new PopulationStateMonitor(population);
    this->population_state_monitors.push_back(new_monitor);
    return new_monitor;
    };

void SpikingNetwork::run(EvolutionContext * evo, double time){  

    auto start = std::chrono::high_resolution_clock::now();
    int n_steps_done  = 0;
    int n_steps_total = static_cast<int>(time / evo->dt) ;

    auto gather_time = std::chrono::duration_cast<std::chrono::microseconds>(start-start).count();
    auto inject_time = std::chrono::duration_cast<std::chrono::microseconds>(start-start).count();

    int n_neurons_total = 0;
    for (auto pop : populations){n_neurons_total += pop->n_neurons;}
        
    // std::cout << "Minimum synaptic delay is " << Synapse::min_delay << std::endl;

    // A check on minimum delays
    if (Synapse::min_delay < evo->dt){
        std::string message = "Globally minimum synaptic delay is " + std::to_string(Synapse::min_delay);
        message += " while dt is " + std::to_string(evo->dt);
        throw std::runtime_error(message);
    }

    std::cout << "Running network consisting of " << n_neurons_total << " neurons for " << n_steps_total <<" timesteps"<<std::endl;

    // Evolve
    boost::timer::progress_display progress(n_steps_total);

    while (evo -> now < time){

        // Gathering of spikes
        auto start_gather = std::chrono::high_resolution_clock::now();
        for (const auto& population_monitor : this->population_spike_monitors){
            population_monitor->gather();
        }
        // Gathering of states
        for (const auto& population_monitor : this->population_state_monitors){
            population_monitor->gather();
        }
        auto end_gather = std::chrono::high_resolution_clock::now();
        gather_time += std::chrono::duration_cast<std::chrono::microseconds>(end_gather-start_gather).count();

        // Injection of currents
        auto start_inject = std::chrono::high_resolution_clock::now();
        for (auto injector : this->injectors){
            injector->inject(evo);
        }
        auto end_inject = std::chrono::high_resolution_clock::now();
        inject_time += std::chrono::duration_cast<std::chrono::microseconds>(end_inject-start_inject).count();

        // Evolution of each population
        for (auto population : this -> populations){
            population -> evolve(evo);
        }
        evo -> do_step();

        n_steps_done++;
        ++progress;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Simulation took " << (std::chrono::duration_cast<std::chrono::seconds>(end -start)).count() << " s";
    std::cout << "\t(" << ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(end -start)).count())/n_steps_done << " ms/step)" << std::endl;

    std::cout << "\tGathering time avg: " << static_cast<double>(gather_time)/n_steps_done << " us/step" << std::endl;
    std::cout << "\tInject time avg: " << static_cast<double>(inject_time)/n_steps_done << " us/step" << std::endl;

}
