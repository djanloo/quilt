#include "include/network.hpp"
#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/devices.hpp"
#include "include/neuron_models.hpp"

#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>
#include <string>
#include <thread>

using std::cout;
using std::endl;

pcg32 random_utils::rng;

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
}

SparseLognormProjection::SparseLognormProjection( float connectivity, int type,
                                    float weight, float weight_delta,
                                    float delay, float delay_delta,
                                    unsigned int start_dimension, unsigned int end_dimension):
    connectivity(connectivity),type(type), weight(weight), weight_delta(weight_delta), delay(delay), delay_delta(delay_delta),
    start_dimension(start_dimension), end_dimension(end_dimension){
    auto start = std::chrono::high_resolution_clock::now();

    weights = boost::numeric::ublas::coordinate_matrix<float>(start_dimension, end_dimension);
    delays = boost::numeric::ublas::coordinate_matrix<float>(start_dimension, end_dimension);


    uint32_t i, j;
    int N = static_cast<int>(connectivity*start_dimension*end_dimension);
    bool is_empty;
    float u, lognorm;
    
    progress bar(N, 1);
    int checks = 0;

    for (int t = 0; t < N; t++){
        
        // Finds an empty slot in the sparse matrices
        is_empty = false;
        do{
            checks++;
            i = static_cast<int>(random_utils::rng()) % start_dimension;
            j = static_cast<int>(random_utils::rng()) % end_dimension;
            is_empty = (weights(i,j) == 0)&&(delays(i,j) == 0);
        } while (!is_empty);

        // TODO: ABSOLUTELY TRANSFORM MEAN AND VARIANCE
        // Weights
        u = static_cast<float>(random_utils::rng()) / UINT32_MAX;
        lognorm = std::exp(weight + weight_delta * std::sqrt(-2.0 * std::log(1.0 - u)));
        
        if (type == 0){ // Excitatory
            weights.insert_element(i, j, lognorm);
        }
        else{ // Inhibitory
            weights.insert_element(i,j, -lognorm);
        }

        // Delays
        u = static_cast<float>(random_utils::rng()) / UINT32_MAX;
        delays.insert_element(i,j, std::exp(delay + delay_delta * std::sqrt(-2.0 * std::log(1.0 - u))));
        ++bar;
    }

    cout << N << endl;
    cout << "checks: " << checks << endl; 
    auto end = std::chrono::high_resolution_clock::now();
    double a = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count())/N;
    cout << "Cost: "<<a << " us/synapse"<<endl;

}


Population::Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network):
    n_neurons(n_neurons),n_spikes_last_step(0), 
    timestats_evo(0), timestats_spike_emission(0){
    
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
    }

void Population::project(const Projection & projection, Population * efferent_population){
    int connections = 0;
    for (unsigned int i = 0; i < projection.start_dimension; i++){
        for (unsigned int j = 0; j < projection.end_dimension; j++){
            if (std::abs((projection.weights)[i][j]) > WEIGHT_EPS){
                connections ++;
                (this -> neurons)[i] -> connect(efferent_population -> neurons[j], (projection.weights)[i][j], (projection.delays)[i][j]);
            }
        }
    }
}

void Population::project(const SparseLognormProjection & projection, Population * efferent_population ){

    boost::numeric::ublas::compressed_matrix<double> weights = projection.weights;
    boost::numeric::ublas::compressed_matrix<double> delays = projection.delays;

    int i,j;
    for (auto iter = weights.begin1(); iter != weights.end1(); ++iter) {
        for (auto nz_iter = iter.begin(); nz_iter != iter.end(); ++nz_iter) {
            std::cout << "element (" << nz_iter.index1() << ", " << nz_iter.index2() << ") = " << *nz_iter << std::endl;
            i = nz_iter.index1();
            j = nz_iter.index2();
            this->neurons[i]->connect(efferent_population->neurons[j], weights(i,j) , delays(i,j));
        }
    }
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
    auto start = std::chrono::high_resolution_clock::now();

    // Splits the work in equal parts using Nthreads threads
    unsigned int n_threads = 4;

    std::vector<unsigned int> bunch_starts(n_threads), bunch_ends(n_threads);

    // std::cout << "Bunches for pop "<<this->id.get_id()<< ":";
    for (int i = 0; i < n_threads; i++){
        bunch_starts[i] = i*static_cast<unsigned int>(this->n_neurons)/n_threads;
        bunch_ends[i] = (i + 1)*static_cast<unsigned int>(this->n_neurons)/n_threads - 1;
        // std::cout << "[" << bunch_starts[i] <<",";
        // std::cout << bunch_ends[i]  << "]";
    }

    // Ensures that all neurons are covered
    bunch_ends[3] = this->n_neurons-1;
    // std::cout << "--corrected "<<bunch_ends[3]<<std::endl;

    // Starts four threads
    // NOTE: spawning threads costs roughly 10 us/thread
    // it is a non-negligible overhead
    // auto start_spawn = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> evolver_threads(n_threads);
    for (int i = 0; i < n_threads; i++){
        evolver_threads[i] = std::thread(&Population::evolve_bunch, this, evo, bunch_starts[i], bunch_ends[i] );
    }
    // auto end_spawn = std::chrono::high_resolution_clock::now();
    // std::cout << "Spawn took "<< ((double)std::chrono::duration_cast<std::chrono::microseconds>(end_spawn-start_spawn).count()) <<"us" <<std::endl;
    // Waits four threads
    for (int i = 0; i < n_threads; i++){
        evolver_threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    timestats_evo += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());

    // TODO: spike emission is moved here in the population evolution because 
    // it's not thread safe. Accessing members of other instances requires
    // a memory access control.
    start = std::chrono::high_resolution_clock::now();
    this->n_spikes_last_step = 0;
    
    for (auto neuron : this->neurons){
        if ((neuron->getV()) >= neuroparam->V_peak){neuron->emit_spike(evo);}
    }

    end = std::chrono::high_resolution_clock::now();
    timestats_spike_emission += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
}

void Population::print_info(){
    cout << "Population "<< this->id.get_id() << " infos:"<< endl;
    cout << "\tN:" << this->n_neurons << endl;
    cout << "\tparams:" << endl;
    for (auto couple : this->neuroparam->paramap.value_map){
        cout << "\t\t" << couple.first << "\t" << couple.second << endl;
    }
 }

SpikingNetwork::SpikingNetwork():verbosity(1){
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
        
    // A check on minimum delays
    if (Synapse::min_delay < evo->dt){
        std::string message = "Globally minimum synaptic delay is " + std::to_string(Synapse::min_delay);
        message += " while dt is " + std::to_string(evo->dt);
        throw std::runtime_error(message);
    }

    if (verbosity > 0){
        std::cout << "Running network consisting of " << n_neurons_total << " neurons for " << n_steps_total <<" timesteps"<<std::endl;
    }    
    // Evolve
    progress bar(n_steps_total, verbosity);

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
        ++bar;
    }
    auto end = std::chrono::high_resolution_clock::now();

    if (verbosity > 0){
        std::cout << "Simulation took " << (std::chrono::duration_cast<std::chrono::seconds>(end -start)).count() << " s";
        std::cout << "\t(" << ((double)(std::chrono::duration_cast<std::chrono::milliseconds>(end -start)).count())/n_steps_done << " ms/step)" << std::endl;

        std::cout << "\tGathering time avg: " << static_cast<double>(gather_time)/n_steps_done << " us/step" << std::endl;
        std::cout << "\tInject time avg: " << static_cast<double>(inject_time)/n_steps_done << " us/step" << std::endl;

        std::cout << "Population evolution stats:" << std::endl;
        for (auto pop : populations){
            std::cout << "\t" << pop->id.get_id() << ":"<<std::endl;
            std::cout << "\t\tevolution:\t" << pop->timestats_evo/n_steps_done << " us/step";
            std::cout << "\t---\t" << static_cast<int>(pop->timestats_evo/n_steps_done/pop->n_neurons*1000) << " ns/step/neuron" << std::endl;
            std::cout << "\t\tspike emission:\t" << pop->timestats_spike_emission/n_steps_done << " us/step";
            std::cout << "\t---\t" << static_cast<int>(pop->timestats_spike_emission/n_steps_done/pop->n_neurons*1000) << " ns/step/neuron" << std::endl;
        }
    }
}
