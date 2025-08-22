#pragma once

#include "base.hpp"
#include "neurons_base.hpp"

/**
 * @brief The adaptive quadratic integrate-and-fire neuron
 *
 * Parameters are hold in an `aqif_param` object
 * 
 * @f[
 *   C_m \frac{dV}{dt} = k(V-V_{rest})(V - V_{thr}) - w
 * @f]
 * 
 * @f[
 *      \tau_w \frac{dw}{dt} = -w + a(V-V_{rest})
 * @f]
 * 
 * On spike:
 * 
 * @f[
 *      V\rightarrow V_{reset}
 * @f]
 * @f[
 *      w \rightarrow w + b
 * @f]
 * 
 * 
 */
class aqif_neuron : public Neuron {
    public:
        aqif_neuron(Population * population);
        void evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ) override;
        void on_spike() override;
};

/**
 * @class aqif_param
 * @brief Container for parameters of `aqif_neuron`
*/
class aqif_param : public NeuroParam {
    public:
        float k;        //!< Quadratic term constant
        float V_th;     //!< Quadratic term shift

        float ada_a;    //!< Adaptive variable drift term
        float ada_b;    //!< Adaptive variable jump term
        float ada_tau_w;//!< Adaptive variable decay time

        aqif_param(ParaMap & paramap)
            :   NeuroParam(paramap)
        {
            if ( (paramap.get<string>("neuron_type") != "aqif") &&\
                 (paramap.get<string>("neuron_type") != "aqif2") ){
                    throw std::runtime_error("Incompatible type of neuron in ParaMap");
            }
            try {
                k = paramap.get("k", 1.0f);
                V_th = paramap.get("V_th", -30.0f);
                ada_a = paramap.get("ada_a", -20.0f);
                ada_b = paramap.get("ada_b", 70.0f);
                ada_tau_w = paramap.get("ada_tau_w", 100.0f);
            }catch (const std::out_of_range & e){
                std::string error_message = "Error in aqif neuron: ";
                error_message += e.what();
                throw std::out_of_range(error_message);
            }
        }
};

/**
 * @class izhikevich_neuron
 * @brief The adaptive quadratic neuron model of Izhikevich
 * 
 * Parameters are hold in `izhikevich_param` object
 * 
 * @f[
 *      C_m\frac{dV}{dt} = 0.04 V^2 + 5V + 140 - w
 * @f]
 * @f[
 *      \frac{dw}{dt} = a(b(V-V_{rest})-w)
 * @f]
*/
class izhikevich_neuron : public Neuron {
    public:
        izhikevich_neuron(Population * population);
        void evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ) override;
        void on_spike() override;
};


/**
 * @class izhikevich_param
 * @brief Container for parameters of `izhikevich_neuron`
*/
class izhikevich_param : public NeuroParam {
    public:
        float a,b,d;
        izhikevich_param(ParaMap & paramap)
            :   NeuroParam(paramap)
        {
            if (paramap.get<string>("neuron_type") != "izhikevich"){
                    throw std::runtime_error("Incompatible type of neuron in ParaMap");
            }
            try{
                a = paramap.get<float>("a");
                b = paramap.get<float>("b");
                d = paramap.get<float>("d");
            } catch (const std::out_of_range & e){
                std::string error_message = "Error in izhikevich neuron: ";
                error_message += e.what();
                throw std::out_of_range(error_message);
            }
        };
};


/**
 * @class aeif_neuron
 * @brief The adaptive exponential integrate-and-fire model
 * 
 *  Parameters are hold in `aeif_param` object
 * 
 * @f[
 *      \frac{dV}{dt} = - (V - V_{rest}) + \Delta \exp{\left(\frac{V - V_{expthresh}}{\Delta}\right)} - w
 * @f]
 * 
 * @f[
 *      \tau_w \frac{dw}{dt} = -w + a(V-V_{rest})
 * @f]
*/
class aeif_neuron : public Neuron {
    public:
        aeif_neuron(Population * population);
        void evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ) override;
        void on_spike() override;
};

/**
 * @class aeif_param
 * @brief Container for parameters of `aeif_neuron`
 * 
*/
class aeif_param : public NeuroParam{
    public:
        float delta_T;          //!< exponential width factor [mV] 
        float V_th;             //!< threshold value of the exponential factor [mV] 

        float ada_a;            //!< drift coefficient of the recovery variable 
        float ada_b;            //!< jump coefficient of the recovery variable
        float ada_tau_w;        //!< timescale parameter of the recovery variable [ms]
        float G_L;              //!< Membrane leak conductance [nS]

        aeif_param (ParaMap & paramap)
            :   NeuroParam(paramap)
        {
            if (paramap.get<string>("neuron_type") != "aeif"){
                    throw std::runtime_error("Incompatible type of neuron in ParaMap");
            }
            try{
                G_L = paramap.get("G_L", 3.0f);
                delta_T = paramap.get("delta_T", 2.0f);
                V_th = paramap.get("V_th", -55.0f);
                ada_a = paramap.get("ada_a", 3.0f);
                ada_b = paramap.get("ada_b", 200.0f);
                ada_tau_w = paramap.get("ada_tau_w", 20.0f);
            } catch (const std::out_of_range & e){
                std::string error_message = "Error in aeif neuron: ";
                error_message += e.what();
                throw std::out_of_range(error_message);
            }
        }
};
// neuron_type: aeif
//   C_m: 80 
//   G_L: 3
//   E_l: -55.8
//   delta_T: 1.8
//   V_th: -55.2
//   E_ex: 0
//   E_in: -72
//   I_e: 15
//   tau_ex: 120 # Estimated from Ammari et al 2012 (was 12)
//   tau_in: 2.1
//   ada_tau_w: 20
//   ada_a: 3
//   ada_b: 200
//   V_peak: 20

class aqif2_neuron : public Neuron{
    public:
        aqif2_neuron(Population * population);
        void evolve_state(const dynamical_state &x , dynamical_state &dxdt , const double t ) override;
        void on_spike() override;
};

class aqif2_param : public aqif_param {
    public:
        float V_b;
        aqif2_param(ParaMap & paramap) 
            :   aqif_param(paramap)
        {
            this->neur_type = "aqif2";
            try{
                V_b = paramap.get<float>("V_b");
            }
            catch (const std::out_of_range & e){
                std::string error_message = "Error in aqif2 neuron: ";
                error_message += e.what();
                throw std::out_of_range(error_message);
            }
        }
}; 


