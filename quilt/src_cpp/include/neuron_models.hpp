

// The menu:
class EvolutionContext;
class HierarchicalID;
class Spike;
class Synapse;
class Neuron;
class Population;
class Projection;


/**
 * 
 * Real models
 * -----------
 * 
 * Each model must override these functions:
 * - the constructor
 * - evolve_state
 * 
 * Each model can override these functions:
 * - on_spike
 * 
*/


/**
 * @class aqif_neuron
 * @brief The adaptive quadratic integrate and fire model
 * 
*/
class aqif_neuron : public Neuron {
    public:
        aqif_neuron(Population * population);
        void evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ) override;
        void on_spike(EvolutionContext * evo) override;
    private:
        float k;
};

/**
 * @class izhikevich_neuron
 * @brief The adaptive quadratic neuron model of Izhikevich
*/
class izhikevich_neuron : public Neuron {
    public:
        izhikevich_neuron(Population * population);
        void evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ) override;
        void on_spike(EvolutionContext * evo) override;
    private:
        float a,b,c,d;
};


/**
 * @class aeif_neuron
 * @brief The adaptive exponential integrate-and-fire model
 * 
*/
class aeif_neuron : public Neuron {
    public:
        aeif_neuron(Population * population);
        void evolve_state(const neuron_state &x , neuron_state &dxdt , const double t ) override;
        void on_spike(EvolutionContext * evo) override;
    private:
        float  Delta, R, E_reset, C_m, g_L, exp_threshold;
};
