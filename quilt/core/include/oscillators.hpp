#pragma once
#include "base.hpp"

#include <typeinfo>
#include <memory>

// Forward declarations
class Link;

using std::vector;
using std::cout;
using std::endl;
using std::runtime_error;

class OscillatorNetwork;
class Oscillator;

/**
 * @class Oscillator
 * @brief Base class of oscillators
 * 
 * The `evolve_state()` method defines the oscillator dynamics and must be overridden.
*/
class Oscillator {
public:
    HierarchicalID id;                  /**< Unique identifier for the oscillator. */
    ParaMap* params;                    /**< Pointer to parameter map for the oscillator. */
    OscillatorNetwork* oscnet;          /**< Pointer to the oscillator network to which the oscillator belongs. */
    ContinuousRK memory_integrator;     /**< Continuous Runge-Kutta integrator for storing state history. */
    string oscillator_type = "base";    /**< Type of the oscillator. Default is "base". */
    unsigned int space_dimension = 2;   /**< Dimension of the oscillator's state space. Default is 2. */

    vector<Link*> incoming_osc;         /**< Vector of incoming links to the oscillator. */
    vector<double> input_history;       /**< Vector of the history of inputs*/
    /**
     * @brief Constructor for the Oscillator class.
     * @param params Parameter map for the oscillator.
     * @param oscnet Pointer to the oscillator network.
    */
    Oscillator(ParaMap* params, OscillatorNetwork* oscnet);
    ~Oscillator(){
        get_global_logger().log(DEBUG, "Destroyed Oscillator");
    }
    /**
     * @brief Virtual function representing the evolution of the oscillator's state.
     * @param x Current state of the oscillator.
     * @param dxdt Output for the derivative of the state.
     * @param t Current time.
    */
    std::function<void(const dynamical_state& x, dynamical_state& dxdt, double t)> evolve_state;

    /**
     * @brief Virtual function representing the variable of interest for EEG of the oscillator
     * 
    */
    std::function<double(const dynamical_state &x)> eeg_voi;

    // Getter methods
    string get_type() { return oscillator_type; }

    unsigned int get_space_dimension() { return space_dimension; }

    vector<dynamical_state> get_history()
    {
        return memory_integrator.state_history;
    }

    // This returns the interpolated past using the continuous Runge-Kutta method
    double get_past(unsigned int axis, double t)
    {
        return memory_integrator.get_past(axis, t);
    }

    vector<double> get_eeg(){
            unsigned int T = memory_integrator.state_history.size();
            vector<double> eeg_history(T, 0);
            for (unsigned int i = 0; i < T; i++){
                eeg_history[i] = eeg_voi(memory_integrator.state_history[i]);
            }
            return eeg_history;
        }
    
    virtual vector<double> get_rate_history(){throw runtime_error("Using virtual get_rate_history() of base Oscillator");};

    // Setter methods
    void set_evolution_context(EvolutionContext* evo);

    // Utils
    void print_info();

private:
    EvolutionContext* evo;  /**< Pointer to the evolution context for the oscillator. */
};

/**
 * @brief Builder method for creating an oscillator instance of a specific type.
 * @param OSC Type of the oscillator.
 * @param params Parameter map for the oscillator.
 * @param osc Pointer to the oscillator network.
 * @return Raw pointer to the created oscillator.
*/
template <class OSC>
Oscillator * oscillator_maker(ParaMap* params, OscillatorNetwork* osc){
    return new OSC(params, osc);
}

/**
 * @class OscillatorFactory
 * @brief Factory class for creating oscillators of different types.
*/
class OscillatorFactory {
    typedef std::function<Oscillator *(ParaMap*, OscillatorNetwork*)> constructor;

public:
    /**
     * @brief Adds a constructor for a specific oscillator type.
     * @param oscillator_type Type of the oscillator.
     * @param lker Constructor function for the oscillator type.
     * @return True if the constructor was added successfully.
    */
    bool add_constructor(string const& oscillator_type, constructor const& lker){
        return _constructor_map.insert(std::make_pair(oscillator_type, lker)).second;
    };

    /**
     * @brief Gets an oscillator instance based on the type.
     * @param oscillator_type Type of the oscillator.
     * @param params Parameter map for the oscillator.
     * @param osc_net Pointer to the oscillator network.
     * @return Pointer to the created oscillator.
    */
    Oscillator * get_oscillator(string const& oscillator_type, ParaMap* params, OscillatorNetwork* osc_net);

    /**
     * @brief Constructor for the OscillatorFactory class.
    */
    OscillatorFactory();

private:
    map<string, constructor> _constructor_map;  /**< Map of oscillator types to their corresponding constructors. */
};

/**
 * @brief Singleton method to return a unique instance of OscillatorFactory.
 * @return Reference to the OscillatorFactory instance.
*/
OscillatorFactory& get_oscillator_factory();

class harmonic_oscillator : public Oscillator{
    public:
        float k;
        harmonic_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
};

class test_oscillator : public Oscillator{
    public:
        float k;
        test_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
};


/** 
 * @brief The Jansen-Rit oscillator
 * 
 * 
 * From Jansen, B.H., Rit, V.G. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biol. Cybern. 73, 357–366 (1995). https://doi.org/10.1007/BF00199471
 * 
 */
class jansen_rit_oscillator : public Oscillator{
    public:
        float ke;   //!< Rate constant for postsynaptic population response to excitatory input [ms^(-1)]. Default 0.1.
        float ki;   //!< Rate constant for postsynaptic population response to inhibitory input [ms^(-1)]. Default 0.05.
        float He;   //!< Maximum amplitude of the excitatory postsynaptic population response [mV]. Default 3.25.
        float Hi;   //!< Maximum amplitude of the inhibitory postsynaptic population response [mV]. Default 22.0.
        float C;    //!< Connectivity constant: pyramidal to spiny stellate. Default 135.
        float v0;   //!< Population mean ﬁring threshold potential [mV]. Default 6.0.
        float s;    //!< Firing rate sigmoid function voltage sensitivity parameter [mV]. Default 0.56.
        float rmax; //!< Maximum population mean ﬁring rate [ms^(-1)]. Default 0.005.
        float U;    //!< Bifurcation parameter: background constant input [ms^(-1)]. Default 0.13.
        jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
        double sigm(double v);
        vector<double> get_rate_history() override;
};


class noisy_jansen_rit_oscillator : public Oscillator{
    public:
        float ke;   //!< Rate constant for postsynaptic population response to excitatory input [ms^(-1)]. Default 0.1.
        float ki;   //!< Rate constant for postsynaptic population response to inhibitory input [ms^(-1)]. Default 0.05.
        float He;   //!< Maximum amplitude of the excitatory postsynaptic population response [mV]. Default 3.25.
        float Hi;   //!< Maximum amplitude of the inhibitory postsynaptic population response [mV]. Default 22.0.
        float C;    //!< Connectivity constant: pyramidal to spiny stellate. Default 135.
        float v0;   //!< Population mean ﬁring threshold potential [mV]. Default 6.0.
        float s;    //!< Firing rate sigmoid function voltage sensitivity parameter [mV]. Default 0.56.
        float rmax; //!< Maximum population mean ﬁring rate [ms^(-1)]. Default 0.005.
        float U;    //!< Bifurcation parameter: background constant input [ms^(-1)]. Default 0.13.
        float sigma_noise;

        noisy_jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
        double sigm(double v);
        vector<double> get_rate_history() override;
    
    private:
        RNG rng;
};



class leon_jansen_rit_oscillator : public Oscillator{
    public:
        float He, Hi, ke, ki;
        float gamma_1, gamma_2, gamma_3, gamma_4, gamma_5;
        float gamma_1T, gamma_2T, gamma_3T;
        float e0, rho1, rho2;
        float U, P, Q;
        leon_jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet);
        double sigm(double v);
};

class OscillatorNetwork{
    public:
        HierarchicalID id;
        vector<Oscillator*> oscillators;

        // Homogeneous constructor: 
        // each oscillator has the same parameters
        OscillatorNetwork(int N, ParaMap * params);

        // The inhomogeneous constructor:
        // each oscillator has it own parameters
        OscillatorNetwork(vector<ParaMap*> params);

        ~OscillatorNetwork();
        
        // Homogenous link builder:
        // each link has the same parameters
        void build_connections(Projection * proj, ParaMap * link_params);

        //  Inhomogenous link builder
        void build_connections(Projection * proj, vector<ParaMap*> links_params);


        void initialize(EvolutionContext * evo, double tau, double vmin, double vmax);
        void evolve();
        void run(EvolutionContext * evo, double time, int verbosity);

        void set_evolution_context(EvolutionContext * evo){
            get_global_logger().log(DEBUG, "set EvolutionContext of OscillatorNetwork");

            this->evo = evo;
            for (auto & oscillator : oscillators){
                oscillator->set_evolution_context(evo);
            }
        };
        
        EvolutionContext * get_evolution_context(){return evo;}

        PerformanceManager perf_mgr;
        
        // Control variables for the building steps
        bool has_oscillators = false;
        bool has_links = false;
        bool is_initialized = false;

        // Control variables for the initialization
        float max_delay, min_delay;

    private:

        EvolutionContext * evo;
};