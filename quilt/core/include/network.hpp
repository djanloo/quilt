#pragma once
#include "base.hpp"

#include <unordered_map>
#include <chrono>
#include <mutex>
#include <variant>


#define WEIGHT_EPS 0.00001

using std::vector;

// The menu
class HierarchicalID;
class EvolutionContext;

class Neuron;
class NeuroParam;
enum class neuron_type : unsigned int;

class Projection;
class Population;
class SpikingNetwork;

class PopulationSpikeMonitor;
class PopulationStateMonitor;
class PopInjector;


/**
 * @class Projection
 * @brief Implements a projection between two populations
 * 
 * @param[in] weights, delays
 * @param[in] start_dimension, end_dimension
 * 
 * This class is not really useful anymore and will be deprecated.
 * 
*/
class Projection{
    public:
        vector<vector<float>>  weights, delays;
        unsigned int start_dimension, end_dimension;

        Projection(vector<vector<float>> weights, vector<vector<float>> delays);
    
    private:
        int n_links = 0;
};

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
        void build_sector(sparse_t *, RNGDispatcher *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
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

        void evolve(EvolutionContext * evo);

        // Bureaucracy
        HierarchicalID id;
        SpikingNetwork * spiking_network;
        double timestats_evo;
        double timestats_spike_emission;
        void print_info();

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

        // Monitors (outputs)
        std::vector<PopulationSpikeMonitor*> population_spike_monitors;
        std::vector<PopulationStateMonitor*> population_state_monitors;

        PopulationSpikeMonitor * add_spike_monitor(Population * population);
        PopulationStateMonitor * add_state_monitor(Population * population);

        // Evolution stuff
        void run(EvolutionContext * evo, double time, int verbosity);
};