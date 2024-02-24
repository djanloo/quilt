#include "include/oscillators.hpp"
#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"

#include <cmath>
#include <stdexcept>
#include <limits>
#include <boost/numeric/odeint.hpp>

using namespace std;

template <>
double Link<Oscillator,Oscillator>::get(int axis, double now){
    // cout << "Link is requesting " << now - delay << " since now = "<< now << " , delay = " << delay<< endl;
    double past_state = source->memory_integrator.get_past(axis, now - delay);

    // Here do whatever the funk you want with the state variable
    // It depends on which types of oscillators you are linking
    // cout << "Link has weight " << weight << " while past state is "<< past_state <<endl;
    return weight*past_state;
}

template <class SOURCE, class DESTINATION>
float Link<SOURCE, DESTINATION>::timestep = 0.0;

Oscillator::Oscillator(OscillatorNetwork * oscnet, EvolutionContext * evo)
    :oscnet(oscnet), evo(evo), memory_integrator(evo){
    id = HierarchicalID(oscnet->id);
    oscnet->oscillators.push_back(this); 
    evolve_state = [](const dynamical_state & x, dynamical_state & dxdt, double t){cout << "Warning: using virtual evolve_state of Oscillator" << endl;};
}

void Oscillator::connect(Oscillator * osc, float weight, float delay){
    osc->incoming_osc.push_back(Link<Oscillator, Oscillator>(this, osc, weight, delay, evo));
}

void OscillatorNetwork::init_oscillators(vector<dynamical_state> init_conds){
    cout << "Initializing oscillators" << endl;
    if (init_conds.size() != oscillators.size()) throw runtime_error("Number of initial conditions is not equal to number of oscillators");
    
    // begin brutal search of maximum delay
    float max_tau = 0;
    for (auto osc : oscillators){
        for (auto l : osc->incoming_osc){
            if (l.delay > max_tau) max_tau = l.delay;
        }
    }
    cout << "Max tau is " << max_tau<<endl;
    // end brutal search of maximum delay
    
    int n_init_pts = static_cast<int>(max_tau/evo->dt) + 1;

    cout << "Adding " << n_init_pts << " initial points"<<endl;
    for (int i = 0; i < init_conds.size(); i++ ){
        cout << "oscillator "<<i<<endl; 
        vector<dynamical_state> new_K(4, vector<double>(oscillators[i]->space_dimension));

        // Computes the value of X and K for the past values
        for (int n = 0; n < n_init_pts; n++){
            
            // Adds initial condition as value of X 
            oscillators[i]->memory_integrator.state_history.push_back(init_conds[i]);
            cout << "\tpushed initial state " << init_conds[i][0] << " "<<init_conds[i][1] << endl;

            // Computes the values of K for each intermediate step
            for (int nu = 0; nu < 4; nu++){
                for (int dim = 0; dim < oscillators[i]->space_dimension; dim++){
                    new_K[nu][dim] = 0.0;
                }
            }
            oscillators[i]->memory_integrator.evaluation_history.push_back(new_K);
        }
        cout << "Oscillator "<<i<< " has history of size "<< oscillators[i]->memory_integrator.state_history.size() << endl;
    }


    // Set the current time to max_tau
    // This makes the initialization to be in [0, T] instead of [-T, 0]
    // but otherwise we should consider negative times in the `get()` method of links
    for (int n = 0; n < n_init_pts - 1; n++){  // If I make n points I am at t = (n-1)*h because zero is included
        evo->do_step();
    }

    cout << "After initialization Evolutioncontext is at "<< evo->now<<endl;
}

void OscillatorNetwork::run(EvolutionContext * evo, double time){
    while (evo->now < time){
        cout << "Time: "<< evo->now <<endl;
        for (auto oscillator : oscillators){
            oscillator->memory_integrator.compute_next();
        }
        for (auto oscillator : oscillators){
            oscillator->memory_integrator.fix_next();
        }
        evo->do_step();
    }
}


// *************** Models **************** //
harmonic_oscillator::harmonic_oscillator(const ParaMap * paramap,       // Required to be a pointer for the interface                 
                                        OscillatorNetwork * oscnet, EvolutionContext * evo)     // Required to be a pointer for the interface
                                        :
                                        Oscillator(oscnet, evo)
    {
    try{
        k = paramap->get("k");
    } catch (const std::out_of_range & e){
        std::string error_message = "Error in harmonic oscillator: ";
        error_message += e.what();
        throw std::out_of_range(error_message);
    }

    memory_integrator.set_dimension(space_dimension);

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){
        dxdt[0] = x[1]; // dx/dt = v

        for (auto input : incoming_osc){
            dxdt[1] += x[0] - k*input.get(0, t) ;
        }  
    };

    memory_integrator.set_evolution_equation(evolve_state);

}

test_oscillator::test_oscillator(const ParaMap * paramap,       // Required to be a pointer for the interface                 
                                        OscillatorNetwork * oscnet, EvolutionContext * evo)     // Required to be a pointer for the interface
                                        :
                                        Oscillator(oscnet, evo)
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

    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

// Auxiliary for Jansen-Rit
double jansen_rit_oscillator::sigm(double v, float nu_max, float v0, float r) {
    double result = nu_max / (1.0 + std::exp(r*(v0-v)));
    return nu_max / (1.0 + std::exp(r*(v0-v)));
}

jansen_rit_oscillator::jansen_rit_oscillator(   const ParaMap * paramap,                                // Required to be a pointer for the interface                 
                                                OscillatorNetwork * oscnet, EvolutionContext * evo)     // Required to be a pointer for the interface
                                        : Oscillator(oscnet, evo){
    
    // Parameters directly from references for now
    A = 3.25,
    B = 22.0;
    a = 100.0/1000.0;   // ms^(-1)
    b = 50.0/1000.0;    // ms^(-1)

    vmax = 5.0/1000.0; // ms^(-1)
    v0 = 6.0;
    C = 135.0;
    r = 0.56;

    space_dimension = 6;
    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){

        double external_currents = 0;
        for (auto input : incoming_osc){
            external_currents += input.get(0, t);
        }
        double external_inputs = 130.0/1000.0 + external_currents ;//+ 130*static_cast<double>(rand())/RAND_MAX/1000/10;
        // cout << "External input is " << external_inputs << endl;
        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];

        dxdt[3] = A*a*sigm( x[1] - x[2], vmax, v0, r) - 2*a*x[3] - a*a*x[0];
        dxdt[4] = A*a*(  external_inputs + 0.8*C*sigm(C*x[0], vmax, v0, r) ) - 2*a*x[4] - a*a*x[1];
        dxdt[5] = B*b*0.25*C*sigm(0.25*C*x[0], vmax, v0, r) - 2*b*x[5] - b*b*x[2];

    };
    memory_integrator.set_dimension(space_dimension); // Solve this, it's stupid to say things twice
    memory_integrator.set_evolution_equation(evolve_state);
}

