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

#define N_THREADS_BUILD 8

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

void SparseProjection::build_sector(sparse_t * sector, RNGDispatcher * rng_dispatch,
                                    float connectivity, 
                                    unsigned int start_index_1, unsigned int end_index_1, 
                                    unsigned int start_index_2, unsigned int end_index_2)
{
    // NOTE: indices of sector are extrema included [start, end]

    if (start_index_1 > end_index_1) throw std::runtime_error("SparseProjection::build : End index is before start index (efferent)");
    if (start_index_2 > end_index_2) throw std::runtime_error("SparseProjection::build : End index is before start index (afferent)");
    // cout << "Started building projection sector with connectivity:"<<connectivity << endl; 

    // Maximum number of connection in a rectangular sector:
    // prevents from looping over a full matrix
    // This should not be actually used unless the connectivity is 1
    const int sector_max_connections = (end_index_1 - start_index_1 + 1)*(end_index_2 - start_index_2 + 1);
    // cout << "\tSector indexes are  " << start_index_1 << "," << end_index_1<<"-"<<start_index_2<< "," <<end_index_2<<endl;
    // cout << "\tMax connection is " << sector_max_connections <<endl;
    // auto start = std::chrono::high_resolution_clock::now();

    unsigned int sector_nconn = static_cast<unsigned int>(sector_max_connections*connectivity);
    // cout << "\tconnections to be made: "<<sector_nconn<<endl;
    
    RNG * rng = rng_dispatch->get_rng();

    uint32_t i, j;  
    int checks = 0;
 
    std::pair<int,int> coordinates;
    bool is_empty;

    while ((*sector).size() < sector_nconn){
        if ((*sector).size() == static_cast<unsigned int>(sector_max_connections)){
            cerr << "Sparse sturture was used to perform a all-to-all connection" << endl;
            break;
        }
        // Finds an empty slot in the sparse matrix
        is_empty = false;
        do{
            checks++;
            i = start_index_1 + rng->get_int() % (end_index_1 - start_index_1 + 1);
            j = start_index_2 + rng->get_int() % (end_index_2 - start_index_2 + 1);
            coordinates = std::make_pair(i,j);
            is_empty = ((*sector)[coordinates].first == 0)&&((*sector)[coordinates].second == 0);
            // if (is_empty) cout << "coordinates "<<i <<"-"<< j << " are empty"<<endl;
            // else cout << "coordinates "<< i <<"-"<< j << " are NOT empty"<<endl;
        } while (!is_empty);

        // Insert weight and delay
        // (This increases the length of sector map)
        (*sector)[coordinates] = this->get_weight_delay(rng, i, j);
        // cout << "\tAdded a link!!"<<endl;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    rng_dispatch->free();
}

void SparseProjection::build_multithreaded()
{
    // cout << "Building multithreaded" << endl;

    // If the population is really small 
    // start one thread per neuron
    int n_threads = N_THREADS_BUILD;
    if (start_dimension < 50){
        n_threads = 1;
    }

    weights_delays = std::vector<sparse_t>(n_threads);
    std::vector<std::thread> threads;

    // NOTE: seeding is disabled for now, use random source
    // TODO: add a global management of seed
    RNGDispatcher rng_dispatcher(n_threads);

    for (int i=0; i < n_threads; i++){
        // cout << "\tstarting thread " << i << endl;
        // cout << "\tthis thread does "<< "("<<i*start_dimension/n_threads<<","<< (i+1)*start_dimension/n_threads-1<<")";
        // cout << "("<<0 <<","<< end_dimension-1<<")"<<endl;
        weights_delays[i].reserve(n_connections/n_threads);
        threads.emplace_back(&SparseProjection::build_sector, this , 
                                    &(weights_delays[i]), &rng_dispatcher,
                                    connectivity,
                                    i*start_dimension/n_threads, (i+1)*start_dimension/n_threads-1,
                                    0, end_dimension-1);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // cout << "Done building multithreaded:"<< endl;
    // for (auto sector : weights_delays){
    //     for (auto conn : sector){
    //         cout << "[ " << conn.first.first << "->" << conn.first.second << ",  w=" << conn.second.first << " d=" << conn.second.second  << "]" << endl;
    //     }
    // }
}

const std::pair<float, float> SparseLognormProjection::get_weight_delay(RNG* rng, int /*i*/, unsigned int /*j*/)
{
    // DO NOT USE 'delay' and 'weight' as variables, dumbass
    double u;
    float new_weight, new_delay;

    if (weight_sigma > 0.0){
        try{
            u = rng->get_uniform();
            new_weight = std::exp(weight_mu + weight_sigma * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));
        }catch (const boost::wrapexcept<std::overflow_error>& e){
            cerr << "overflow in erf_inv:" << endl;
            // cerr << "u: " << u <<endl;
            cerr << "weight mu: " << weight_mu << endl;
            cerr << "weight sigma:"<< weight_sigma <<endl;
            throw(e);
        }
    }else{
        new_weight = _weight;
    }

    if (delay_sigma > 0.0){
        try{
            u = rng->get_uniform();
            new_delay = std::exp(delay_mu + delay_sigma * sqrt(2)* boost::math::erf_inv( 2.0 * u - 1.0));
        }catch (const boost::wrapexcept<std::overflow_error>& e){
            cerr << "overflow:" << endl;
            // cerr << "u: " << u <<endl;
            cerr << "delay mu: " << delay_mu << endl;
            cerr << "delay sigma:"<< delay_sigma << endl;
            throw(e);
        }
    }else{
        new_delay = _delay;
    }

    // Inhibitory 
    if (type == 1) new_weight *=  -1;

    // // Zero-delay
    // if ((delay_mu == 0.0)&&(delay_sigma ==0.0)){
    //     cerr << "SparseLogNormProjection::get_weight_delay -> Delay mu and delay sigma are zero: d_mu="<<delay_mu <<" d_sigma="<<delay_sigma<<endl;
    //     new_delay = 0.0;
    // }


    if (std::isnan(new_weight) ) throw runtime_error("Nan in delay generation");
    if (std::isnan(new_delay) ) throw runtime_error("Nan in weight generation for delay_mu, delay_sigma = " + std::to_string(delay_mu) + ", " + std::to_string(delay_sigma));

    return std::make_pair(new_weight, new_delay);
}


SparseLognormProjection::SparseLognormProjection(double connectivity, int type,
                                unsigned int start_dimension, unsigned int end_dimension,
                                float weight, float weight_delta,
                                float delay, float delay_delta)
    :   SparseProjection(connectivity, type, start_dimension, end_dimension),
        _weight(weight), 
        _delay(delay)
{
    if (weight == 0.0) throw runtime_error("synaptic weight cannot be zero");
    if (delay == 0.0) throw runtime_error("synaptic delay cannot be zero");

    // cout << "SparseLogNormProjection::SparseLogNormProjection : delay is "<<delay<<endl;
    weight_sigma = std::sqrt(std::log1p( (weight_delta*weight_delta)/(weight*weight)));
    delay_sigma  = std::sqrt(std::log1p( (delay_delta*delay_delta)/(delay*delay)));

    weight_mu   = std::log(weight) - 0.5 * weight_sigma * weight_sigma;
    delay_mu    = std::log(delay)  - 0.5 * delay_sigma * delay_sigma;
    
    build_multithreaded();
}


Population::Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network)
    :   n_neurons(n_neurons),
        n_spikes_last_step(0),
        id(&(spiking_network->id)),
        spiking_network(spiking_network),
        timestats_evo(0), 
        timestats_spike_emission(0)
{

    // // Adds itself to the hierarchical structure
    // id = HierarchicalID(&(spiking_network->id));

    // Adds itself to the spiking network populations
    (spiking_network->populations).push_back(this);

    string neuron_type;
    try{ 
        neuron_type = params->get<string>("neuron_type");
    }catch (const std::out_of_range & e) {
        throw std::out_of_range("Neuron params must have field neuron_type");
    }catch (const std::runtime_error & e){
        throw std::runtime_error("Could not get parameter `neuron_type`");
    }

    NeuroFactory * neurofactory = NeuroFactory::get_neuro_factory();

    // Asks the neurofactory to generate the correct neuroparam
    neuroparam = neurofactory->get_neuroparam(params->get<string>("neuron_type"), *params);
    
    // Generates the neurons
    // TODO: neuron add themselves to the population vector.
    // This is not legit and must be corrected
    for ( int i = 0; i < n_neurons; i++){
        neurofactory->get_neuron( params->get<string>("neuron_type"), this);
    }
}

void Population::project(const Projection * projection, Population * efferent_population)
{
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

void Population::project(const SparseProjection * projection, Population * efferent_population )
{
    int connections = 0;
    for (auto sector : projection->weights_delays){
        for (auto connection : sector){
            connections ++;
            neurons[connection.first.first]->connect(efferent_population->neurons[connection.first.second], connection.second.first, connection.second.second);
        }
    }
    // cout << "Performed " << connections << " connections between pop:"<< this->id.get_id() << " and pop:"<< efferent_population->id.get_id() << endl; 
}

void Population::evolve()
{
    auto start = std::chrono::high_resolution_clock::now();

    // Splits the work in equal parts using Nthreads threads
    unsigned int n_threads;

    if (n_neurons < MAX_N_1_THREADS)        n_threads = 0;
    else if (n_neurons < MAX_N_2_THREADS)   n_threads = 2;
    else if (n_neurons < MAX_N_3_THREADS)   n_threads = 3;
    else if (n_neurons < MAX_N_4_THREADS)   n_threads = 4;
    else                                    n_threads = std::thread::hardware_concurrency();

    auto evolve_bunch = [this](unsigned int from, unsigned int to){
                            for (unsigned int i = from; i< to; i++){
                                this->neurons[i]->evolve();
                            }
                        };

    // In case few neurons are present, do not use multithreading
    // to avoid overhead
    if (n_threads == 0){
        evolve_bunch(0, n_neurons);
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
            evolver_threads[i] = std::thread(evolve_bunch, bunch_starts[i], bunch_ends[i] );
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
        if ((neuron->getV()) >= neuroparam->V_peak){
            neuron->emit_spike();
            // cout << "V over threshold neuron: spiked at t: "<< evo->now << endl;
        }
        
    }

    end = std::chrono::high_resolution_clock::now();
    timestats_spike_emission += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
}

void Population::print_info()
{
    cout << "Population "<< this->id.get_id() << " (over " << this->spiking_network->populations.size() << ")" << " infos:"<< endl;
    cout << "\tN:" << this->n_neurons << endl;
    cout << "\tparams:" << endl;
    cout << neuroparam ->paramap;
    
    // Counts the avg connection with each population
    vector <int> connection_counts(this->spiking_network->populations.size(), 0);

    for (auto neuron : neurons){
        for (auto syn : neuron->efferent_synapses){
            connection_counts[syn.get_efferent_pop_id()]++;
        }
    }

    cout << "Each neuron is connected (on average) with:"<<endl;
    for (unsigned int i = 0; i < connection_counts.size(); i++){
        cout << "\t" << static_cast<float>(connection_counts[i])/this->n_neurons << " neurons of population " << i << endl;
    }

 }

void Population::set_evolution_context(EvolutionContext * evo)
    {
        get_global_logger().log(DEBUG, "set EvolutionContext of Population");
        this->evo = evo;
        for (auto neuron : neurons)
        {
            neuron->set_evolution_context(evo);
        }
    }

Population::~Population()
{
    delete neuroparam;
    for (Neuron * neur : neurons){
        delete neur;
    }
}

SpikingNetwork::SpikingNetwork()
    :   evocontext_initialized(false)
{
    id = HierarchicalID();
}

PopulationSpikeMonitor * SpikingNetwork::add_spike_monitor(Population * population)
{
    PopulationSpikeMonitor * new_monitor = new PopulationSpikeMonitor(population);
    this->population_monitors.push_back(new_monitor);
    return new_monitor;
};

PopulationStateMonitor * SpikingNetwork::add_state_monitor(Population * population)
{
    PopulationStateMonitor * new_monitor = new PopulationStateMonitor(population);
    this->population_monitors.push_back(new_monitor);
    return new_monitor;
};

void SpikingNetwork::set_evolution_context(EvolutionContext * evo)
        {
            get_global_logger().log(DEBUG, "set EvolutionContext of SpikingNetwork");
            this->evo = evo;
            for (auto population : populations)
            {
                population->set_evolution_context(evo);
            }
            for (auto monitor : population_monitors)
            {
                monitor->set_evolution_context(evo);
            }
            evocontext_initialized = true;
        }

void SpikingNetwork::evolve(){

    if (!evocontext_initialized){
        throw("Cannot evolve SpikingNetwok until its EvolutionContext is initialized.");
    }

    cout << "Evolving SPIKING network (t = "<<evo->now <<" -> "<< evo->now + evo->dt << ")" << endl;
    for (const auto& population_monitor : this->population_monitors){
            population_monitor->gather();
        }

        // Injection of currents
        for (auto injector : this->injectors){
            injector->inject(evo);
        }

        // Evolution of each population
        for (auto population : this -> populations)
        {
            population -> evolve();
        }
        evo -> do_step();
}

void SpikingNetwork::run(EvolutionContext * evo, double time, int verbosity)
{  
    stringstream ss;
    ss << "Evolving spiking network from t= "<< evo->now << " to t= " << time;
    get_global_logger().log(INFO, ss.str());

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

    // Synchronize evolution of each nested object
    set_evolution_context(evo);

    // Evolve
    progress bar(n_steps_total, verbosity);

    while (evo -> now < time){

        // Gathering from populations
        auto start_gather = std::chrono::high_resolution_clock::now();
        for (const auto& population_monitor : this->population_monitors){
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
        for (auto population : this -> populations)
        {
            population -> evolve();
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
    
    for (auto monitor : population_monitors){
        delete monitor;
    }
}