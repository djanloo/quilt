#pragma once

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
};

class aqif_param : public NeuroParam {
    public:
        float k, ada_a, ada_b, ada_tau_w;
        aqif_param(ParaMap paramap):NeuroParam(paramap){
            k = paramap.get("k");
            ada_a = paramap.get("ada_a");
            ada_b = paramap.get("ada_b");
            ada_tau_w = paramap.get("ada_tau_w");
        }
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
};

class izhikevich_param : public NeuroParam {
    public:
        float a,b,c,d;
        izhikevich_param(const ParaMap & paramap):NeuroParam(paramap){
            a = paramap.get("a");
            b = paramap.get("b");
            c = paramap.get("c");
            d = paramap.get("d");
        };
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
};

class aeif_param : public NeuroParam{
    public:
        float Delta, R, g_L, exp_threshold, ada_a, ada_b, ada_tau_w;
        aeif_param (const ParaMap & paramap) : NeuroParam(paramap){
            Delta = paramap.get("Delta");
            R = paramap.get("R");
            g_L = paramap.get("g_L");
            exp_threshold = paramap.get("exp_threshold");
            ada_a = paramap.get("ada_a");
            ada_b = paramap.get("ada_b");
            ada_tau_w = paramap.get("ada_tau_w");
        }
};