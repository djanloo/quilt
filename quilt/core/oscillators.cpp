#include "include/oscillators.hpp"
#include "include/base.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"

#include <cmath>
#include <stdexcept>
#include <limits>

template <>
double Link<Oscillator,Oscillator>::get(int axis, double now)
{
    double past_state = source->get_past(axis, now - delay);

    // Here do whatever the funk you want with the state variable
    // It depends on which types of oscillators you are linking

    return weight*past_state;
}

Oscillator::Oscillator(OscillatorNetwork * oscnet)
    :   oscnet(oscnet), 
        memory_integrator()
{
    id = HierarchicalID(oscnet->id);
    oscnet->oscillators.push_back(this); 
    evolve_state = [](const dynamical_state & /*x*/, dynamical_state & /*dxdt*/, double /*t*/){cout << "Warning: using virtual evolve_state of Oscillator" << endl;};
}

void Oscillator::connect(Oscillator * osc, float weight, float delay)
{
    osc->incoming_osc.push_back(Link<Oscillator, Oscillator>(this, osc, weight, delay));
}

void OscillatorNetwork::initialize(EvolutionContext * evo, vector<dynamical_state> init_conds)
{
    cout << "Initializing oscillators" << endl;
    if (init_conds.size() != oscillators.size()) throw runtime_error("Number of initial conditions is not equal to number of oscillators");
    
    // brutal search of maximum delay
    float max_tau = 0.0;
    for (auto osc : oscillators){
        for (auto l : osc->incoming_osc){
            if (l.delay > max_tau) max_tau = l.delay;
        }
    }
    // ~brutal search of maximum delay
    
    int n_init_pts = static_cast<int>(max_tau/evo->dt) + 1;

    for (unsigned int i = 0; i < init_conds.size(); i++ ){
        vector<dynamical_state> new_K(4, vector<double>(oscillators[i]->space_dimension));

        // Computes the value of X and K for the past values
        for (int n = 0; n < n_init_pts; n++){
            
            // Adds initial condition as value of X 
            oscillators[i]->memory_integrator.state_history.push_back(init_conds[i]);

            // Computes the values of K for each intermediate step
            for (int nu = 0; nu < 4; nu++){
                for (unsigned int dim = 0; dim < oscillators[i]->space_dimension; dim++){
                    new_K[nu][dim] = 0.0;
                }
            }
            oscillators[i]->memory_integrator.evaluation_history.push_back(new_K);
        }
    }


    // Set the current time to max_tau
    // This makes the initialization to be in [0, T] instead of [-T, 0]
    // but otherwise we should consider negative times in the `get()` method of links
    for (int n = 0; n < n_init_pts - 1; n++){  // If I make n points I am at t = (n-1)*h because zero is included
        evo->do_step();
    }

    is_initialized = true;
}

void OscillatorNetwork::run(EvolutionContext * evo, double time, int verbosity)
{
    if (!is_initialized) throw runtime_error("The network must be initialized before running");
    
    // Synchronizes every component
    set_evolution_context(evo);

    // Some verbose output
    double t0 = evo->now;
    int n_steps_total = static_cast<int>(time/evo->dt);
    if (verbosity > 0){
        std::cout << "Running network consisting of " << oscillators.size() << " oscillators for " << n_steps_total <<" timesteps"<<std::endl;
    }

    // Evolve
    progress bar(n_steps_total, verbosity);

    auto start = std::chrono::high_resolution_clock::now();
    while (evo->now < t0 + time){
            
        // Gets the new values
        for (auto oscillator : oscillators){
            oscillator->memory_integrator.compute_next();
        }

        // Fixes them
        for (auto oscillator : oscillators){
            oscillator->memory_integrator.fix_next();
        }
        evo->do_step();
        ++bar;
    }
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Simulation took " << std::chrono::duration_cast<std::chrono::seconds>(end-start).count()<< " seconds ";
    cout << "( " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count())/n_steps_total<< " ms/step)" << endl;
}


// **************************************** Models ***************************************** //
harmonic_oscillator::harmonic_oscillator(const ParaMap * paramap, OscillatorNetwork * oscnet)    
    :   Oscillator(oscnet)
{
    k = paramap->get("k");


    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){
        dxdt[0] = x[1];

        for (auto input : incoming_osc){
            dxdt[1] += x[0] - k*input.get(0, t) ;
        }  
    };

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);

}

test_oscillator::test_oscillator(const ParaMap * /*paramap*/, OscillatorNetwork * oscnet)
    :   Oscillator(oscnet)
{
    space_dimension = 6;

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){

        dxdt[0] = x[1];
        dxdt[1] = -x[0];

        dxdt[2] = x[3];
        dxdt[3] = -x[2];

        dxdt[4] = x[5];
        dxdt[5] = -x[4];
    };

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

// Auxiliary for Jansen-Rit
double jansen_rit_oscillator::sigm(double v, float nu_max, float v0, float r)
{
    return nu_max / (1.0 + std::exp(r*(v0-v)));
}

jansen_rit_oscillator::jansen_rit_oscillator( const ParaMap * paramap, OscillatorNetwork * oscnet) 
    :   Oscillator(oscnet)
{
    space_dimension = 6;

    // Parameters default from references for now
    A = paramap->get("A", 3.25);
    B = paramap->get("B", 22.0);
    a = paramap->get("a", 100.0/1000.0);   // ms^(-1)
    b = paramap->get("b", 50.0/1000.0);    // ms^(-1)
    vmax = paramap->get("vmax", 5.0/1000.0); // ms^(-1)
    v0 = paramap->get("v0", 6.0);
    C = paramap->get("C", 135.0);
    r = paramap->get("r", 0.56);

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){

        double external_currents = 0;
        for (auto input : incoming_osc){
            external_currents += input.get(0, t);
        }
        double external_inputs = 130.0/1000.0 + external_currents;

        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];

        dxdt[3] = A*a*sigm( x[1] - x[2], vmax, v0, r) - 2*a*x[3] - a*a*x[0];
        dxdt[4] = A*a*(  external_inputs + 0.8*C*sigm(C*x[0], vmax, v0, r) ) - 2*a*x[4] - a*a*x[1];
        dxdt[5] = B*b*0.25*C*sigm(0.25*C*x[0], vmax, v0, r) - 2*b*x[5] - b*b*x[2];

    };

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}