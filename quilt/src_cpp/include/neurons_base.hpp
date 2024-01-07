#pragma once
#include <iostream>
#include <vector>
#include <queue>

#define MAX_GSYN_EXC 15.0
#define MAX_GSYN_INH 15.0

#define MAX_POTENTIAL_INCREMENT 10 // mV
#define MAX_POTENTIAL_SLOPE 50/0.1 // mV/ms

enum class neuron_type : unsigned int {dummy, aqif, izhikevich, aeif};
typedef std::vector<double> neuron_state;

// The menu:
class EvolutionContext;
class HierarchicalID;
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
        double weight, arrival_time;
        bool processed;

        Spike(double weight, double arrival_time):
        weight(weight), arrival_time(arrival_time), processed(false){}

        // Set the importance: the smaller the arrival time the greater the importance
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
        Synapse(Neuron * presynaptic, Neuron * postsynaptic, double weight, double delay):
            presynaptic(presynaptic),postsynaptic(postsynaptic),
            weight(weight), delay(delay){}
            
        void fire(EvolutionContext * evo);
    private:
        Neuron * presynaptic;
        Neuron * postsynaptic;
        double weight, delay;
};

/**
 * @class Neuron
 * @brief The base dynamical object
 * 
 * 
 * The 9 to 5 job of a neuron is:
 *  - process incoming spikes
 *  - evolve the state
 *  - fire if it's the case
 * 
 * The first and the last stage are (roughly) equal for every model,
 * while the evolution equation is model dependent.
 * 
 * To declare a new neuron:
 *  - override the `evolve_state` method
 *  - add the neuron to neuron_type enum class
 * 
 * After profiling, it became evident that spike processing has a significantly greater impact on computational 
 * cost than I had initially anticipated. Therefore, each improvement attempt should start with an analysis 
 * of handle_incoming_spikes()
*/
class Neuron{
    protected:
        neuron_state state;
    public:
        // Base properties
        neuron_type nt = neuron_type::dummy;
        HierarchicalID * id;
        Population * population;
        neuron_state get_state(){return state;}


        // Physiological properties
        float C_m;
        float tau_refrac, tau_e, tau_i, tau_m;
        float E_exc, E_inh, E_rest, E_thr, E_reset;

        // Adaptive variables
        float ada_a, ada_b, ada_tau_w;

        // External currents
        float I_ext, I_osc, omega_I;


        // Spike stuff
        std::vector<Synapse*> efferent_synapses;
        std::priority_queue<Spike> incoming_spikes;
        double last_spike_time;

        Neuron(Population * population); 
        void connect(Neuron * neuron, double weight, double delay);
        void handle_incoming_spikes(EvolutionContext * evo);
        void evolve(EvolutionContext * evo);
        void emit_spike(EvolutionContext * evo);

        // These must be implemented for each specific neuron
        virtual void on_spike(EvolutionContext * evo);
        virtual void evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ){std::cout << "WARNING: using virtual evolve_state of <Neuron>";};
};
