#pragma once
#include "base.hpp"
#include "devices.hpp"

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>

#include <unordered_map>
#include <chrono>
#include <mutex>
#include <variant>

using std::vector;

#define N_THREADS_POP_EVOLVE 8

// Forward declarations
class Neuron;
class NeuroParam;
enum class neuron_type : unsigned int;
class Population;
class SpikingNetwork;

struct SparseIntHash {
    size_t operator()(const std::pair<int, int>& k) const 
    {
        return std::hash<int>()(k.first) ^ std::hash<int>()(k.second);
    }
};

struct SparseEqual {
    bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const 
    {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

typedef std::unordered_map<std::pair<int, int>, std::pair<float, float>, SparseIntHash, SparseEqual> sparse_t;


/**
 * @class SparseProjection
 * @brief Implements a sparse random projection between two populations
 * 
*/
class SparseProjection{
    protected:
        std::mutex mutex;
    public: 
        double connectivity;            //!< Density of connection (0 < connectivity < 1) 
        int type;                       //!< Excitatory: 0, Inhibitory: 1
        unsigned int start_dimension;   //!< The number of objects in efferent population
        unsigned int end_dimension;     //!< The number of objects in afferent population
        unsigned int n_connections;

        std::vector<sparse_t> weights_delays;

        SparseProjection(double connectivity,  int type, unsigned int start_dimension, unsigned int end_dimension)
            :   connectivity(connectivity), 
                type(type), 
                start_dimension(start_dimension), 
                end_dimension(end_dimension)
        {
            n_connections = static_cast<unsigned int>(connectivity*start_dimension*end_dimension);
        }  
        virtual ~SparseProjection() = default;
        void build_sector(sparse_t *, RNGDispatcher *, float, unsigned int, unsigned int, unsigned int, unsigned int);
        void build_multithreaded();

        virtual const std::pair<float, float> get_weight_delay(RNG* /*rng*/, int /*i*/, unsigned int /*j*/)
        {
            throw std::runtime_error("Using virtual get_weight_delay of sparse projection");
        }
};

class SparseLognormProjection : public SparseProjection{
    public:         
        float weight_mu;        //!< Average weight
        float weight_sigma;     //!< Weight std
        float delay_mu;         //!< Average delay
        float delay_sigma;      //!< Delay std

        float _weight, _delay;

        SparseLognormProjection(double connectivity, int type,
                                unsigned int start_dimension, unsigned int end_dimension,
                                float weight, float weight_delta,
                                float delay, float delay_delta);

        const std::pair<float, float> get_weight_delay(RNG* rng, int i, unsigned int j) override;
};

/**
 * @class Population
 * @brief A collection of neurons sharing neuron model
 * 
 * @param[in] n_neurons, neur_type, spiking_network
 * 
 * This class server the purpose of storing quantities that are
 * related to a group of neurons rather than a neuron itself:
 * 
 * @param n_spikes_last_step 
 * 
*/
class Population{
    public:
        int n_neurons;
        std::vector<Neuron*> neurons;

        // Biophysical attributes
        int n_spikes_last_step;
        NeuroParam * neuroparam;
        
        Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network);
        ~Population();
        
        void project(const Projection * projection, Population * efferent_population);
        void project(const SparseProjection * projection, Population * efferent_population);

        void evolve();

        // Bureaucracy
        HierarchicalID id;
        SpikingNetwork * spiking_network;

        void print_info();
        void set_evolution_context(EvolutionContext * evo);
        
        PerformanceManager perf_mgr;
    private:
        boost::asio::thread_pool thread_pool; //!< Thread pool for evolution and spike handling
        EvolutionContext * evo;

};

/**
 * @class SpikingNetwork
 * @brief A collection of populations
 * 
 * This class is intended to manage input/output to/from populations
 * using the objects contained in devices.cpp
 * 
 * Also, coordinates the evolution of the populations.
*/

class SpikingNetwork{
    public:
        std::vector<Population*> populations;
        HierarchicalID id;

        SpikingNetwork();
        ~SpikingNetwork();

        // Injectors (inputs)
        std::vector<PopInjector*> injectors;
        void add_injector(PopInjector * injector)
        {
            this->injectors.push_back(injector);
        }

        void set_evolution_context(EvolutionContext * evo);

        EvolutionContext * get_evolution_context(){return evo;}

        // Monitors (outputs)
        std::vector<PopulationMonitor*> population_monitors;

        PopulationSpikeMonitor * add_spike_monitor(Population * population);
        PopulationStateMonitor * add_state_monitor(Population * population);

        // Evolution stuff
        void evolve();
        void run(EvolutionContext * evo, double time, int verbosity);

        PerformanceManager perf_mgr;
    private:
        EvolutionContext * evo;
        bool evocontext_initialized;
};