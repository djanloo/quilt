#include "include/oscillators.hpp"
#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/network.hpp"

#include <stdexcept>
#include <limits>
#include <boost/numeric/odeint.hpp>

using namespace std;

vector<double> ContinuousRK::b_functions(double theta){
    vector<double> result(4);

    result[0] = 2*(1-4*b[0])*std::pow(theta, 3) + 3*(3*b[0] - 1)*theta*theta + theta;
    for (int i=1; i<4; i++){
        result[i] = 4*(3*c[i] - 2)*b[i]*std::pow(theta, 3) + 3*(3-4*c[i])*b[i]*theta*theta;
    }
    return result;
}

double ContinuousRK::get_past(int axis, double abs_time){
    cout << "getting past of CRK (abs time: "<<abs_time <<")" << endl;
    // Split in bin_index + fractionary part
    int bin_id = static_cast<int>(abs_time/evo->dt); // This is dangerous in case of variable step! ACHTUNG!
    double theta = abs_time/evo->dt - bin_id;
    
    // Cutoff small negative values (1e-16) to zero
    if (theta < 0) theta = 0.0; 

    if (bin_id<0) throw runtime_error("Requested past state that lays before initialization");

    cout << "Asked for point " << bin_id << " + " << theta<< endl;
    cout << "\tA" << endl;
    // Get the values and the interpolation weights related to that moment in time
    double y = state_history[bin_id][axis];
    vector<double> b_func_values = b_functions(theta);
    cout << "\tB" << endl;

    // Updates using the interpolant
    for (int nu = 0; nu < 4; nu++){
        y += evo->dt * b_func_values[nu] * evaluation_history[bin_id][nu][axis];
    }
    cout << "\tC" << endl;

    return y;
}

void ContinuousRK::compute_next(){
    cout << "Computing next in CRK" << endl;
    if (space_dimension < 0) throw runtime_error("Space dimension not set in ContinuousRK");

    proposed_evaluation = vector<osc_state>(4, osc_state(space_dimension, 0));
    double t_eval;
    osc_state x_eval;
    
    for (int nu = 0; nu < 4; nu++){ // Compute the K values
        t_eval = evo->now + c[nu] * evo->dt;
        cout << "nu: "<< nu << " t_eval: "<<t_eval<<endl;  
        x_eval = state_history.back();
        if (nu != 0) {
            for (int i = 0; i < space_dimension; i++){
                x_eval[i] += a[nu] * proposed_evaluation[nu-1][i];
            }
        }
        // Assigns the new K evaluation to the value of the evolution function
        evolve_state(x_eval, proposed_evaluation[nu], t_eval);

    }//End compute K values

    // Updates the state
    proposed_state = state_history.back();
    for (int i = 0; i < space_dimension; i++){
        for (int nu = 0; nu < 4; nu++){
            proposed_state[i] += evo->dt * b[nu] * proposed_evaluation[nu][i];
        }
    }
    cout << "Done computing next in CRK" <<endl;
}

void ContinuousRK::fix_next(){
    cout << "Fixing next in CRK" << endl;
    state_history.push_back(proposed_state);
    evaluation_history.push_back(proposed_evaluation);
}

template <>
double Link<Oscillator,Oscillator>::get(int axis, double now){
    cout << "Link is requesting " << now - delay << " since now = "<< now << " , delay = " << delay<< endl;
    double past_state = source->memory_integrator.get_past(axis, now - delay);

    // Here do whatever the funk you want with the state variable
    // It depends on which types of oscillators you are linking

    return weight*past_state;
}

template <class SOURCE, class DESTINATION>
float Link<SOURCE, DESTINATION>::timestep = 0.0;

Oscillator::Oscillator(OscillatorNetwork * oscnet, EvolutionContext * evo)
    :oscnet(oscnet), evo(evo), memory_integrator(evo){
    id = HierarchicalID(oscnet->id);
    oscnet->oscillators.push_back(this); 
    evolve_state = [](const osc_state & x, osc_state & dxdt, double t){cout << "Warning: using virtual evolve_state of Oscillator" << endl;};
}

void Oscillator::connect(Oscillator * osc, float weight, float delay){
    osc->incoming_osc.push_back(Link<Oscillator, Oscillator>(this, osc, weight, delay, evo));
}

void OscillatorNetwork::init_oscillators(vector<osc_state> init_conds){
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
    
    int n_init_pts = static_cast<int>(max_tau/evo->dt);
    if (n_init_pts == 0) n_init_pts = 1;
    cout << "Adding " << n_init_pts << " initial points"<<endl;
    for (int i = 0; i < init_conds.size(); i++ ){
        cout << "oscillator "<<i<<endl; 
        vector<osc_state> new_K(4, vector<double>(oscillators[i]->space_dimension));

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
    for (int n = 0; n < n_init_pts; n++){
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

osc_state Oscillator::none_state = {0.0, 0.0};

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

    evolve_state = [this](const osc_state & x, osc_state & dxdt, double t){
        cout << "calling evolution of harmonic oscillator "<<endl;
        dxdt[0] = x[1]; // dx/dt = v

        for (auto input : incoming_osc){
            dxdt[1] += -k * (x[0] - input.get(0, t) );
        }  
    };

    memory_integrator.set_evolution_equation(evolve_state);

}

osc_state harmonic_oscillator::none_state = {0.0, 0.0};


