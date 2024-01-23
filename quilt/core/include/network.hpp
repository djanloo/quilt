#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <variant>
#include "base_objects.hpp"
#include "../pcg/include/pcg_random.hpp"

#define WEIGHT_EPS 0.00001

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


namespace random_utils{
    extern pcg32 rng;
};

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
        float ** weights, **delays;
        unsigned int start_dimension, end_dimension;

        Projection(float ** weights, float ** delays, unsigned int start_dimension, unsigned int end_dimension);
};

struct SparseIntHash {
    size_t operator()(const std::pair<int, int>& k) const {
        return std::hash<int>()(k.first) ^ std::hash<int>()(k.second);
    }
};

struct SparseEqual {
    bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};


/**
 * @class SparseLognormProjection
 * @brief Implements a sparse random projection between two populations
 * 
 * Weights and delays are lognorm-distributed. TODO: mean and sigma of the distribution are not 
 * what one espects them to be.
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

        std::unordered_map<std::pair<int, int>, std::pair<float, float>, SparseIntHash, SparseEqual> weights_delays;

        SparseProjection(double connectivity,  int type, unsigned int start_dimension, unsigned int end_dimension):
                    connectivity(connectivity), type(type), start_dimension(start_dimension), end_dimension(end_dimension){
                        weights_delays.reserve(static_cast<int>(connectivity*start_dimension*end_dimension));
                        std::cout << "building SP with params "<< connectivity << " " << type << " "<< start_dimension << " "<< end_dimension << std::endl;
                    }      
        void build(unsigned int N);
        void build_multithreaded();

          // Costruttore di copia
        SparseProjection(const SparseProjection& other) :
            connectivity(other.connectivity),
            type(other.type),
            start_dimension(other.start_dimension),
            end_dimension(other.end_dimension),
            weights_delays(other.weights_delays) {
            std::cout << "copy constructor called" << std::endl;
        }

        // Costruttore di movimento
        SparseProjection(SparseProjection&& other) noexcept :
            connectivity(other.connectivity),
            type(other.type),
            start_dimension(other.start_dimension),
            end_dimension(other.end_dimension),
            weights_delays(std::move(other.weights_delays)) {
            std::cout << "move constructor called" << std::endl;
        }

        // Operatore di assegnazione di copia
        SparseProjection& operator=(const SparseProjection& other) {
            if (this != &other) {
                connectivity = other.connectivity;
                type = other.type;
                start_dimension = other.start_dimension;
                end_dimension = other.end_dimension;
                weights_delays = other.weights_delays;
                std::cout << "copy assignment operator called" << std::endl;
            }
            return *this;
        }

        // Operatore di assegnazione di movimento
        SparseProjection& operator=(SparseProjection&& other) noexcept {
            if (this != &other) {
                connectivity = other.connectivity;
                type = other.type;
                start_dimension = other.start_dimension;
                end_dimension = other.end_dimension;
                weights_delays = std::move(other.weights_delays);
                std::cout << "move assignment operator called" << std::endl;
            }
            return *this;
        }

        virtual std::pair<float, float> get_weight_delay(unsigned int /*i*/, unsigned int /*j*/){
            throw std::runtime_error("Using virtual get_weight_delay of sparse projection");
        }
};

class SparseLognormProjection : public SparseProjection{
    public:         
        float weight;           //!< Average weight
        float weight_delta;     //!< Weight standard deviation
        float delay;            //!< Average delay
        float delay_delta;      //!< Delay standard deviation

        SparseLognormProjection(double connectivity, int type,
                                unsigned int start_dimension, unsigned int end_dimension,
                                float weight, float weight_delta,
                                float delay, float delay_delta):
                                SparseProjection(connectivity, type, start_dimension, end_dimension), 
                                weight(weight), weight_delta(weight_delta), 
                                delay(delay), delay_delta(delay_delta){ 
                                    std::cout << "Starting sparselognorm constructor" << std::endl;
                                    auto start = std::chrono::high_resolution_clock::now();
                                    // build_multithreaded();
                                    build(static_cast<int>(connectivity*start_dimension*end_dimension));
                                    auto end = std::chrono::high_resolution_clock::now();

                                    std::cout << "Ended sparselognorm constructor" << std::endl;
                                    std::cout<< "LogNorm: Check connections: " << weights_delays.size() << std::endl;
                                    std::cout << "LogNorm: Check time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() <<" ms"<<std::endl;
                                }

        std::pair<float, float> get_weight_delay(unsigned int i, unsigned int j) override;
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
        HierarchicalID id;
        double timestats_evo;
        double timestats_spike_emission;

        // Biophysical attributes
        int n_spikes_last_step;
        NeuroParam * neuroparam;
        
        Population(int n_neurons, ParaMap * params, SpikingNetwork * spiking_network);
        void project(const Projection * projection, Population * efferent_population);
        void project(const SparseProjection * projection, Population * efferent_population);

        void evolve_bunch(EvolutionContext * evo, unsigned int from, unsigned int to);
        void evolve(EvolutionContext * evo);
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
        HierarchicalID * id;
        unsigned int verbosity;
        SpikingNetwork();

        // Injectors (inputs)
        std::vector<PopInjector*> injectors;
        void add_injector(PopInjector * injector){this->injectors.push_back(injector);}

        // Monitors (outputs)
        std::vector<PopulationSpikeMonitor*> population_spike_monitors;
        std::vector<PopulationStateMonitor*> population_state_monitors;

        PopulationSpikeMonitor * add_spike_monitor(Population * population);
        PopulationStateMonitor * add_state_monitor(Population * population);

        // Evolution stuff
        void run(EvolutionContext * evo, double time);
};