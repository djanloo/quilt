#include "include/network.hpp"
#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/devices.hpp"
#include "include/neuron_models.hpp"

#include <boost/math/special_functions/erf.hpp>
// #include <boost/thread/thread.hpp>

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
void SparseProjection::build_sector(sparse_t * sector, unsigned int sector_nconn, 
                                    unsigned int start_index_1, unsigned int end_index_1, 
                                    unsigned int start_index_2, unsigned int end_index_2){
    
    // cout << "SparseProjection::build_sector PID "<< std::this_thread::get_id();
    // cout << "\t 1_indexes: "<< start_index_1 << " " << end_index_1;
    // cout << "\t 2_indexes: "<< start_index_2 << " " << end_index_2;
    // cout << endl;

    if (start_index_1 > end_index_1) throw std::runtime_error("SparseProjection::build : End index is before start index (efferent)");
    if (start_index_2 > end_index_2) throw std::runtime_error("SparseProjection::build : End index is before start index (afferent)");

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t i, j;  
    int checks = 0;
 
    std::pair<int,int> coordinates;
    bool is_empty;

    while ((*sector).size() < sector_nconn){
        // Finds an empty slot in the sparse matrix
        is_empty = false;
        do{
            checks++;
            i = start_index_1 + static_cast<int>(random_utils::rng()) % (end_index_1 - start_index_1);
            j = start_index_2 + static_cast<int>(random_utils::rng()) % (end_index_2 - start_index_2);
            coordinates = std::make_pair(i,j);
            is_empty = ((*sector)[coordinates].first == 0)&&((*sector)[coordinates].second == 0);
        } while (!is_empty);

        // Insert weight and delay
        (*sector)[coordinates] = this->get_weight_delay(i, j);
    }
    auto end = std::chrono::high_resolution_clock::now();
    // cout << "SparseProjection::build_sector: took " << ((float)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count())/n_connections << " us/syn" << endl; 
    // cout << "SparseProjection::build_sector:  extra checks: " << checks - n_connections << endl;
}

void SparseProjection::build_multithreaded(){
    const int n_threads = 8; 

    weights_delays = std::vector<sparse_t>(n_threads);
    std::vector<std::thread> threads;

    for (int i=0; i < n_threads; i++){
        weights_delays[i].reserve(n_connections/n_threads);
        threads.emplace_back(&SparseProjection::build_sector, this , 
                                    &(weights_delays[i]), n_connections/n_threads,
                                    i*start_dimension/n_threads, (i+1)*start_dimension/n_threads,
                                    0, end_dimension-1);
    }
    
    // cout << "SparseProjection::build_multithreaded: Started ALL" << endl;

    for (auto& thread : threads) {
        thread.join();
    }
    // cout << "SparseProjection::build_multithreaded: Joined ALL" << endl;
}

const std::pair<float, float> SparseLognormProjection::get_weight_delay(unsigned int /*i*/, unsigned int /*j*/){
    double u;
    float new_weight, new_delay;

    u = static_cast<double>(random_utils::rng()) / UINT32_MAX;
    new_weight = std::exp(weight_mu + weight_sigma * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));
    
    u = static_cast<double>(random_utils::rng()) / UINT32_MAX;
    new_delay = std::exp(delay_mu + delay_sigma * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));
    
    // Inhibitory 
    if (type == 1) new_weight *=  -1;

    return std::make_pair(new_weight, new_delay);
}


SparseLognormProjection::SparseLognormProjection(double connectivity, int type,
                                unsigned int start_dimension, unsigned int end_dimension,
                                float weight, float weight_delta,
                                float delay, float delay_delta):
                                SparseProjection(connectivity, type, start_dimension, end_dimension){
       
                                    weight_sigma = std::sqrt(std::log( (weight_delta*weight_delta)/(weight*weight)  + 1.0));
                                    delay_sigma  = std::sqrt(std::log( (delay_delta*delay_delta)/(delay*delay)      + 1.0));

                                    weight_mu   = std::log(weight) - 0.5 * weight_sigma * weight_sigma;
                                    delay_mu    = std::log(delay)  - 0.5 * delay_sigma * delay_sigma;

                                    // cout << "weight_mu "<< weight_mu <<endl;
                                    // cout << "weight_sigma "<< weight_sigma<<endl;
                                    // std::cout << "Starting sparselognorm constructor from PID " <<std::this_thread::get_id()<< std::endl;
                                    auto start = std::chrono::high_resolution_clock::now();
                                    build_multithreaded();
                                    // build(static_cast<int>(connectivity*start_dimension*end_dimension), 0, start_dimension, 0, end_dimension);
                                    auto end = std::chrono::high_resolution_clock::now();

                                    // std::cout << "Ended sparselognorm constructor" << std::endl;
                                    // std::cout<< "LogNorm: number of connections: " << weights_delays.size() ;// << std::endl;
                                    // std::cout << " (should be " << n_connections << " )" <<std::endl;
                                    // std::cout << "LogNorm: Check time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() <<" ms ";
                                    // cout << ((float)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count())/n_connections << " us/syn" << endl; 
                                    
                                    // cout << "Check iter: start" << endl;
                                    // int iterations = 0;
                                    // for (auto it = this->weights_delays.begin(); it != this->weights_delays.end();){
                                    //     iterations++;
                                    //     it++;
                                    // }
                                    // cout << "Check iter: done" << endl;

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
        case neuron_type::aqif:         this->neuroparam = new aqif_param(*params);         break;
        case neuron_type::aqif2:        this->neuroparam = new aqif2_param(*params);        break;
        case neuron_type::izhikevich:   this->neuroparam = new izhikevich_param(*params);   break;
        case neuron_type::aeif:         this->neuroparam = new aeif_param(*params);         break;
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

void Population::project(const Projection * projection, Population * efferent_population){
    int connections = 0;
    for (unsigned int i = 0; i < projection->start_dimension; i++){
        for (unsigned int j = 0; j < projection->end_dimension; j++){
            if (std::abs((projection->weights)[i][j]) > WEIGHT_EPS){
                connections ++;
                neurons[i]->connect(efferent_population->neurons[j], projection->weights[i][j], projection->delays[i][j]);
            }
        }
    }
}

void Population::project(const SparseProjection * projection, Population * efferent_population ){
    for (auto sector : projection->weights_delays){
        for (auto connection : sector){
            neurons[connection.first.first]->connect(efferent_population->neurons[connection.first.second], connection.second.first, connection.second.second);
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

    for (unsigned int i = 0; i < n_threads; i++){
        bunch_starts[i] = i*static_cast<unsigned int>(this->n_neurons)/n_threads;
        bunch_ends[i] = (i + 1)*static_cast<unsigned int>(this->n_neurons)/n_threads - 1;
    }

    // Ensures that all neurons are covered
    bunch_ends[3] = this->n_neurons-1;

    // Starts the threads
    // NOTE: spawning threads costs roughly 10 us/thread
    // it is a non-negligible overhead
    std::vector<std::thread> evolver_threads(n_threads);
    for (unsigned int i = 0; i < n_threads; i++){
        evolver_threads[i] = std::thread(&Population::evolve_bunch, this, evo, bunch_starts[i], bunch_ends[i] );
    }

    // Waits four threads
    for (unsigned int i = 0; i < n_threads; i++){
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
