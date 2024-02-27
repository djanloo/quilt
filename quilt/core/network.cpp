#include "include/network.hpp"
#include "include/neurons_base.hpp"
#include "include/devices.hpp"
#include "include/neuron_models.hpp"

#include <boost/math/special_functions/erf.hpp>

#include <string>
#include <thread>

#define MAX_N_1_THREADS 150
#define MAX_N_2_THREADS 300
#define MAX_N_3_THREADS 600
#define MAX_N_4_THREADS 1000

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

Projection::Projection(vector<vector<float>> weights, vector<vector<float>> delays):
    weights(weights), delays(delays){
    
    start_dimension = weights.size();
    if (start_dimension == 0) throw std::runtime_error("start dimension of projection is zero");
    
    end_dimension = weights[0].size();
    if (end_dimension == 0) throw std::runtime_error("end dimension of projection is zero");

    for (unsigned int i = 0; i < start_dimension; i++){
        for (unsigned int j =0 ; j< end_dimension; j++){
            if (std::abs(weights[i][j]) >= WEIGHT_EPS){ 
                n_links ++;
            }
        }
    }
}
void SparseProjection::build_sector(sparse_t * sector, RNGDispatcher * rng_dispatch,
                                    unsigned int sector_nconn, 
                                    unsigned int start_index_1, unsigned int end_index_1, 
                                    unsigned int start_index_2, unsigned int end_index_2){
    

    if (start_index_1 > end_index_1) throw std::runtime_error("SparseProjection::build : End index is before start index (efferent)");
    if (start_index_2 > end_index_2) throw std::runtime_error("SparseProjection::build : End index is before start index (afferent)");

    // auto start = std::chrono::high_resolution_clock::now();
    
    RNG * rng = rng_dispatch->get_rng();

    uint32_t i, j;  
    int checks = 0;
 
    std::pair<int,int> coordinates;
    bool is_empty;

    while ((*sector).size() < sector_nconn){
        // Finds an empty slot in the sparse matrix
        is_empty = false;
        do{
            checks++;
            i = start_index_1 + rng->get_int() % (end_index_1 - start_index_1);
            j = start_index_2 + rng->get_int() % (end_index_2 - start_index_2);
            coordinates = std::make_pair(i,j);
            is_empty = ((*sector)[coordinates].first == 0)&&((*sector)[coordinates].second == 0);
        } while (!is_empty);

        // Insert weight and delay
        (*sector)[coordinates] = this->get_weight_delay(rng, i, j);
    }
    // auto end = std::chrono::high_resolution_clock::now();
    rng_dispatch->free();
}

void SparseProjection::build_multithreaded(){
    const int n_threads = 8; 

    weights_delays = std::vector<sparse_t>(n_threads);
    std::vector<std::thread> threads;

    RNGDispatcher rng_dispatcher(n_threads, 1997);

    for (int i=0; i < n_threads; i++){
        weights_delays[i].reserve(n_connections/n_threads);
        threads.emplace_back(&SparseProjection::build_sector, this , 
                                    &(weights_delays[i]), &rng_dispatcher,
                                    n_connections/n_threads,
                                    i*start_dimension/n_threads, (i+1)*start_dimension/n_threads,
                                    0, end_dimension-1);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

const std::pair<float, float> SparseLognormProjection::get_weight_delay(RNG* rng, int /*i*/, unsigned int /*j*/){
    double u;
    float new_weight, new_delay;

    try{
        u = rng->get_uniform();
        new_weight = std::exp(weight_mu + weight_sigma * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));
    }catch (const boost::wrapexcept<std::overflow_error>& e){
        cerr << "overflow in erf_inv:" << endl;
        cerr << "u: " << u <<endl;
        cerr << "weight mu: " << weight_mu << endl;
        cerr << "weight sigma:"<< weight_sigma <<endl;
        throw(e);
    }
    try{
        u = rng->get_uniform();
        new_delay = std::exp(delay_mu + delay_sigma * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));
    }catch (const boost::wrapexcept<std::overflow_error>& e){
        cerr << "overflow:" << endl;
        cerr << "u: " << u <<endl;
        cerr << "delay mu: " << delay_mu << endl;
        cerr << "delay sigma:"<< delay_sigma << endl;
        throw(e);
    }
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
                                    
                                    build_multithreaded();
                                }


Population::Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network)
    :   n_neurons(n_neurons),
        n_spikes_last_step(0), 
        spiking_network(spiking_network),
        timestats_evo(0), 
        timestats_spike_emission(0)
{
    // Adds itself to the hierarchical structure
    id = HierarchicalID(&(spiking_network->id));

    // Adds itself to the spiking network populations
    (spiking_network->populations).push_back(this);

    try{ 
        params->get("neuron_type");
    }catch (const std::out_of_range & e) {
        throw( std::out_of_range("Neuron params must have field neuron_type"));
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

void Population::evolve(EvolutionContext * evo){
    auto start = std::chrono::high_resolution_clock::now();

    // Splits the work in equal parts using Nthreads threads
    unsigned int n_threads;

    if (n_neurons < MAX_N_1_THREADS)        n_threads = 0;
    else if (n_neurons < MAX_N_2_THREADS)   n_threads = 2;
    else if (n_neurons < MAX_N_3_THREADS)   n_threads = 3;
    else if (n_neurons < MAX_N_4_THREADS)   n_threads = 4;
    else                                    n_threads = std::thread::hardware_concurrency();

    auto evolve_bunch = [this](EvolutionContext * evo, unsigned int from, unsigned int to){
                            for (unsigned int i = from; i< to; i++){
                                this->neurons[i]->evolve(evo);
                            }
                        };

    // In case few neurons are present, do not use multithreading
    // to avoid overhead
    if (n_threads == 0){
        evolve_bunch(evo, 0, n_neurons);
    }
    else{// multithreading case begin

        std::vector<unsigned int> bunch_starts(n_threads), bunch_ends(n_threads);

        for (unsigned int i = 0; i < n_threads; i++){
            bunch_starts[i] = i*static_cast<unsigned int>(this->n_neurons)/n_threads;
            bunch_ends[i] = (i + 1)*static_cast<unsigned int>(this->n_neurons)/n_threads - 1;
        }

        // Ensures that all neurons are covered
        bunch_ends[n_threads-1] = this->n_neurons-1;

        // Starts the threads
        // NOTE: spawning threads costs roughly 10 us/thread
        // it is a non-negligible overhead
        std::vector<std::thread> evolver_threads(n_threads);
        for (unsigned int i = 0; i < n_threads; i++){
            evolver_threads[i] = std::thread(evolve_bunch, evo, bunch_starts[i], bunch_ends[i] );
        }

        // Waits the threads
        for (unsigned int i = 0; i < n_threads; i++){
            evolver_threads[i].join();
        }
    } // multithreading case end

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

Population::~Population(){
    delete neuroparam;
    for (Neuron * neur : neurons){
        delete neur;
    }
}

SpikingNetwork::SpikingNetwork(){
    id = HierarchicalID();
}

PopulationSpikeMonitor * SpikingNetwork::add_spike_monitor(Population * population)
{
    PopulationSpikeMonitor * new_monitor = new PopulationSpikeMonitor(population);
    this->population_spike_monitors.push_back(new_monitor);
    return new_monitor;
};

PopulationStateMonitor * SpikingNetwork::add_state_monitor(Population * population)
{
    PopulationStateMonitor * new_monitor = new PopulationStateMonitor(population);
    this->population_state_monitors.push_back(new_monitor);
    return new_monitor;
};

void SpikingNetwork::run(EvolutionContext * evo, double time, int verbosity){  

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
        message += " while dt is " + std::to_string(evo->dt) + ".\n";
        int n_cutoff_synapses = 0;
        for (Population * pop : populations){
            for (Neuron * neur : pop->neurons){
                for (Synapse & syn : neur->efferent_synapses){
                    if (syn.get_delay() < evo->dt){
                        syn.set_delay(evo->dt);
                        n_cutoff_synapses ++;
                        }
                }
            }
        }
        message += std::to_string(n_cutoff_synapses) +" synaptic delays were rounded to " + std::to_string(evo->dt);
        std::cerr << message << std::endl;
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

SpikingNetwork::~SpikingNetwork(){
    
    for (auto monitor : population_spike_monitors){
        delete monitor;
    }
    for (auto monitor : population_state_monitors){
        delete monitor;
    }
}