#pragma once

#include "base_objects.hpp"
#include "neurons_base.hpp"

#include <stdexcept>

// The menu:
class EvolutionContext;
class HierarchicalID;
class ParaMap;
class Spike;
class Synapse;
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
            if (static_cast<neuron_type>(paramap.get("neuron_type")) != neuron_type::aqif){
                    throw std::runtime_error("Incompatible type of neuron in ParaMap");
            }
            std::string last = "";
            try {
                last = "k";
                k = paramap.get(last);
                last = "ada_a";
                ada_a = paramap.get(last);
                last = "ada_b";
                ada_b = paramap.get(last);
                last = "ada_tau_w";
                ada_tau_w = paramap.get(last);
            }catch(const std::out_of_range & e){
                throw std::out_of_range("Missing parameter for aqif neuron:" + last);
            }
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
        izhikevich_param(const ParaMap & paramap): NeuroParam(paramap){
            if (static_cast<neuron_type>(paramap.get("neuron_type")) != neuron_type::izhikevich){
                    throw std::runtime_error("Incompatible type of neuron in ParaMap");
            }
            std::string last = "";
            try{
                last = "a";
                a = paramap.get(last);
                last = "b";
                b = paramap.get(last);
                last = "c";
                c = paramap.get(last);
                last = "d";
                d = paramap.get(last);
            } catch (const std::out_of_range & e){
                throw (std::out_of_range("Missing parameter for Izhikevich neuron: "+ last));
            }
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
        float Delta, exp_threshold, ada_a, ada_b, ada_tau_w;

        aeif_param (const ParaMap & paramap):
        NeuroParam(paramap){
            std::cout << "Initializing an aeif_param..";
            if (static_cast<neuron_type>(paramap.get("neuron_type")) != neuron_type::aeif){
                    throw std::runtime_error("Incompatible type of neuron in ParaMap");
            }
            std::string last = "";
            try{
                last = "Delta";
                Delta = paramap.get(last);
                last = "exp_threshold";
                exp_threshold = paramap.get(last);
                last = "ada_a";
                ada_a = paramap.get(last);
                last="ada_b";
                ada_b = paramap.get(last);
                last = "ada_tau_w";
                ada_tau_w = paramap.get(last);
            } catch (const std::out_of_range & e){
                throw std::out_of_range("Missing parameter for aeif neuron:" + last);
            }
        }
};