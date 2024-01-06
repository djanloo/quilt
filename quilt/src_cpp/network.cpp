#include <iostream>
#include <vector>
#include <map>
#include "include/base_objects.hpp"
#include "include/neurons.hpp"
#include "include/network.hpp"
#include "include/devices.hpp"

#include <iomanip>
#include <chrono>
#include <boost/timer/progress_display.hpp>

Population::Population(int n_neurons, neuron_type nt, SpikingNetwork * spiking_network){
    this -> n_neurons = n_neurons;
    this -> n_spikes_last_step = 0;
    this -> id = new HierarchicalID(spiking_network->id);
    
    auto start = std::chrono::high_resolution_clock::now();

    for ( int i = 0; i < n_neurons; i++){
        // This can be avoided, probably using <variant>
        switch(nt){
        case neuron_type::dummy: new Neuron(this); break;       // remember not to push_back here
        case neuron_type::aqif: new aqif_neuron(this); break;   // calling the constructor is enough
        case neuron_type::izhikevich: new izhikevich_neuron(this); break;
        case neuron_type::aeif: new aeif_neuron(this); break;
        };
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Building population "<< this->id->get_id() << " took " << (std::chrono::duration_cast<std::chrono::milliseconds>(end -start)).count() << " ms    (";
    std::cout << ((double)(std::chrono::duration_cast<std::chrono::microseconds>(end-start)).count())/n_neurons << " us/neur)" << std::endl;
    
    // Adds itself to the spiking network populations
    (spiking_network->populations).push_back(this);
    }

Projection::Projection(double ** _weights, double ** _delays, int _start_dimension, int _end_dimension){
    this -> weights = _weights;
    this -> delays = _delays;
    this -> start_dimension = _start_dimension;
    this -> end_dimension = _end_dimension;

    int n_links = 0;
    for (int i = 0; i < _start_dimension; i++){
        for (int j =0 ; j< _end_dimension; j++){
            if (weights[i][j] != 0.0){ //Mhh, dangerous
                n_links ++;
            }
        }
    }
    std::cout << "Projection has density " << ((float)n_links)/_start_dimension/_end_dimension * 100 << "%" << std::endl;
}

void Population::project(Projection * projection, Population * efferent_population){
    int connections = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < projection -> start_dimension; i++){
        for (int j = 0; j < projection -> end_dimension; j++){
            if (std::abs((projection -> weights)[i][j]) > WEIGHT_EPS){
                connections ++;
                (this -> neurons)[i] -> connect(efferent_population -> neurons[j], (projection -> weights)[i][j], (projection -> delays)[i][j]);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Performing " << connections << " connections took ";
    std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end -start)).count() << " ms   (";
    std::cout << ((double)(std::chrono::duration_cast<std::chrono::microseconds>(end -start)).count())/connections << " us/link)" << std::endl;
}

void Population::evolve(EvolutionContext * evo){

    double avg_synaptic_queue_size = 0;
    for (auto neuron : this->neurons){
        avg_synaptic_queue_size += neuron -> incoming_spikes.size();
    }
    avg_synaptic_queue_size /= this->n_neurons;
    // std::cout << "pop " << this->id->get_id() << ") average synaptic queue is long " << avg_synaptic_queue_size << std::endl;

    this->n_spikes_last_step = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (auto neuron : this -> neurons){
        neuron -> evolve(evo);
    }
    auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "pop " <<this->id->get_id() << ") evolving took " << ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(end-start)).count()) << " ms (";
    // std::cout << ((double)(std::chrono::duration_cast<std::chrono::microseconds>(end-start)).count())/this->n_neurons;
    // std::cout << " us/neur )" << std::endl;
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

void  SpikingNetwork::evolve(EvolutionContext * evo){
    for (auto population : this -> populations){
        population -> evolve(evo);
    }
    evo -> do_step();
}

void SpikingNetwork::run(EvolutionContext * evo, double time){  

    /**
     * Gets the values form monitors. 
     * This may be a little too formal, but 
     *      - cycles on auto references of variant (auto&)
     *      - avoid modification of reference (const)
     * 
     * Also maybe too pythonic. I just have to quit heterogeneous iterables.
     * 
     * TODO: check timing. This is a lot of overhead I see here.
     * 
     */

    auto start = std::chrono::high_resolution_clock::now();
    int n_steps_done  = 0;
    int n_steps_total = static_cast<int>(time / evo->dt) ;


    auto gather_time = std::chrono::duration_cast<std::chrono::microseconds>(start-start).count();
    auto inject_time = std::chrono::duration_cast<std::chrono::microseconds>(start-start).count();

    int n_neurons_total = 0;
    for (auto pop : populations){n_neurons_total += pop->n_neurons;}
    std::cout << "Running network consisting of " << n_neurons_total << " neurons for " << n_steps_total <<" timesteps";

    // Evolve
    boost::timer::progress_display progress(n_steps_total);

    while (evo -> now < time){
        auto start_gather = std::chrono::high_resolution_clock::now();
        for (const auto& population_monitor : this->population_spike_monitors){
            population_monitor->gather();
        }
        for (const auto& population_monitor : this->population_state_monitors){
            population_monitor->gather();
        }
        auto end_gather = std::chrono::high_resolution_clock::now();
        gather_time += std::chrono::duration_cast<std::chrono::microseconds>(end_gather-start_gather).count();

        auto start_inject = std::chrono::high_resolution_clock::now();
        for (auto injector : this->injectors){
            injector->inject(evo);
        }
        auto end_inject = std::chrono::high_resolution_clock::now();
        inject_time += std::chrono::duration_cast<std::chrono::microseconds>(end_inject-start_inject).count();

        for (auto population : this -> populations){
            population -> evolve(evo);
        }
        evo -> do_step();
        n_steps_done++;
        ++progress;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Simulation took " << (std::chrono::duration_cast<std::chrono::seconds>(end -start)).count() << " s";
    std::cout << "\t(" << ((double)(std::chrono::duration_cast<std::chrono::seconds>(end -start)).count())/n_steps_done << " s/step)" << std::endl;

    std::cout << "\tGathering time avg: " << static_cast<double>(gather_time)/n_steps_done << " us/step" << std::endl;
    std::cout << "\tInject time avg: " << static_cast<double>(inject_time)/n_steps_done << " us/step" << std::endl;


}
