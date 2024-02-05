/**
 * @file neurons_base.h
 * @brief Base-level neural dynamics (spikes, spike porcessing and generation, population parameters)
 */
#pragma once
#include "base_objects.hpp"
#include <queue>
#include <limits>
#include <string>

// #define MAX_GSYN_EXC 15.0 //!< Not in use: max value of excitatory synaptic conductance
// #define MAX_GSYN_INH 15.0 //!< Not in use: max value of inhibitory synaptic conductance

#define MAX_POTENTIAL_SLOPE 50/0.1      //!< Cutoff value of potential slope [mV/ms]
#define OSCILLATORY_AMPLITUDE_MIN  0.01 //!< Minimal value for an oscillatory input to be considered valid [pA]
// #define MAX_SPIKE_QUEUE_LENGTH 10000 //!< Not in use: spike queue lenght before warning

/**
 * @enum neuron_type
 * @brief the currently available neuron models
*/
enum class neuron_type : unsigned int {base_neuron, aqif, aqif2, izhikevich, aeif};

typedef std::vector<double> neuron_state;

// The menu:
class EvolutionContext;
class HierarchicalID;
class ParaMap;
class Spike;
class Synapse;
class Neuron;
class Population;
class Projection;

/**
 * @class Spike
 * @brief A spike object is a weight and arrival time
 * 
 * This object is the one through which neurons communicate.
 * The overriden 'operator <' is used to insert the spike in the priority queue 
 * of the neuron. 
*/
class Spike{
    public:
        float weight;           //!< Increase in postsynaptic conductance
        float arrival_time;     //!< The (absolute) time of arrival 
        bool processed;         //!< Flag to check missed spikes

        Spike(float weight, double arrival_time):
        weight(weight), arrival_time(arrival_time), processed(false){}

        /**
         * 
         * @brief Priority function of the spike queue
         * 
         * The greater the time of arrival, the less the importance. 
         * This operator is called when inserting spikes in spike queues.
         * 
        */
        bool operator<(const Spike& other) const { return this->arrival_time > other.arrival_time; }
};

/**
 * @class Synapse
 * @brief A weight-delay connection object between two neurons
 * 
 * The method fire() of the synapse creates a spike and adds it to the spike queue
 * of the postsynaptic neuron. 
*/
class Synapse{
    public:
        Synapse(Neuron * presynaptic, Neuron * postsynaptic, float weight, float delay):
            presynaptic(presynaptic),postsynaptic(postsynaptic),
            weight(weight), delay(delay){
                if (this->delay < min_delay){
                    min_delay = this->delay;
                }
            }
            
        void fire(EvolutionContext * evo);
        static float min_delay; //!< Smallest synaptic delay of the model. Used to check timestep.
        void set_delay(float new_delay){delay = new_delay;}
        float get_delay(){return delay;}
    private:
        Neuron * presynaptic;   //!< Pointer to the postsynaptic neuron
        Neuron * postsynaptic;  //!< Pointer to the presynaptic neuron
        float weight;           //!< Synapse weight. Each spike produced from this synapse will have this weight
        float delay;            //!< Synapse delay. Each spike produced from this synapse will have this delay

};

/**
 * @class Neuron
 * @brief The base spiking dynamical object
 * 
 * This class is a virtual base class. The explicit dynamic model must override the `evolve_state()` method.
 * 
 * The main method of `Neuron` is `evolve()`, in which are called in sequence:
 *  - `on_spike()` (if `spike_flag` is true)
 *  - `handle_incoming_spikes()`
 *  - `evolve_state()`
 * 
 * To declare a new neuron:
 *  - override the `evolve_state` method
 *  - add the neuron to neuron_type enum class
 * 
 * After profiling, it became evident that spike processing has a significantly greater impact on computational 
 * cost than I had initially anticipated. Therefore, each improvement attempt should start with an analysis 
 * of `handle_incoming_spikes()`.
*/
class Neuron{
    protected:
        neuron_state state;

        // Spike flag is introduced to record the threshold value
        // so that on_spike() is called the timestep after the spike
        bool spike_flag; 
    public:
        // Base properties
        neuron_type nt = neuron_type::base_neuron;
        HierarchicalID id;
        Population * population;
        neuron_state get_state(){return state;}

        // Spike stuff
        std::vector<Synapse> efferent_synapses;
        std::priority_queue<Spike> incoming_spikes;
        double last_spike_time;

        Neuron(Population * population); 
        virtual ~Neuron() = default;

        /**The evolution function*/
        void evolve(EvolutionContext * evo);

        /** Connection function. Adds an efferent synapse to the synapse list.*/        
        void connect(Neuron * neuron, double weight, double delay);

        /** Manages the incoming spikes. */
        void handle_incoming_spikes(EvolutionContext * evo);

        /** Makes the neuron's efferent synapses fire */
        void emit_spike(EvolutionContext * evo);

        double getV(){return state[0];}

        // These must be implemented for each specific neuron

        /** The actions to take when the model has a spike*/
        virtual void on_spike(EvolutionContext * evo);
        /** The differential equations of the neuron*/
        virtual void evolve_state(const neuron_state & /*x*/ , neuron_state & /*dxdt*/ , const double /*t*/ ){
                std::cout << "WARNING: using virtual evolve_state of <Neuron>";
        };
};

/** The container of neuron parameters. Each specific model has its own.*/
class NeuroParam{ 

    protected:
        neuron_type neur_type;

    public:
        ParaMap paramap;

        float E_l;      //!< Rest potential [mV]
        float V_reset;  //!< Reset potential [mV]
        float V_peak;   //!< Spike-emission threshold [mV]
        float E_ex;     //!< Excitatory synaptic potential [mV]
        float E_in;     //!< Inhibitory synaptic potential [mV]

        float C_m;      //!< Membrane potential [pF]

        float tau_ex;   //!< Excitatory synapse decay time [ms]
        float tau_in;   //!< Inhibitory synapse decay time [ms]
        float tau_refrac;   //!< Refractory time
        
        float I_e;      //!< External current (constant) amplitude [nA]
        float I_osc;    //!< External current (oscillatory) amplitude [nA]
        float omega_I;  //!< External current (oscillatory) angular frequency [rad/s]
        
        /**Builds the default neuroparam. External currents are set to zero.*/
        NeuroParam();
        /**Constructor of `NeuroParam` given a `ParaMap`*/
        NeuroParam(const ParaMap & paramap);

        neuron_type get_neuron_type(){return neur_type;}
        void add(const std::string & key, float value);
};
