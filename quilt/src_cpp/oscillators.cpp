#include "include/base_objects.hpp"
#include "include/neurons_base.hpp"
#include "include/oscillators.hpp"
#include "include/network.hpp"

#include <stdexcept>
#include <limits>
#include <boost/numeric/odeint.hpp>

using namespace std;

template <>
osc_state Link<Oscillator,Oscillator>::get(double now){
    int n = static_cast<int>((now-delay)/timestep);
    float delta = ((now - delay)/timestep) - static_cast<float>(n);

    // cout <<"Getting from link"<<endl;
    // cout << "\tdt = " << Link::timestep<<endl;
    // cout <<"\tdelay = " << delay << " (" << static_cast<int>(delay/timestep) << " steps)"<<endl;
    // cout <<"\thistory size = " << source->history.size() << endl;
    // cout <<"\trequested = " << n << endl;
    // cout << "\tdelta = "<<delta<<endl;

    if ( n < 0 ){
        cout << "\tPast does not exist ( n = " << n <<" ) passing the none state" << endl;
        return source->none_state;
    }

    if ( n + 2 > static_cast<int>(source->history.size()) ){
        cout << "\tNot enough history: past is long " << source->history.size() << " and " << n << " was requested " << endl;   
        return source->none_state;
    }

    if (delta > 1.0){throw runtime_error("Delta was > 1");}
    

    osc_state past_state_left  = source->history[n];
    osc_state past_state_right = source->history[n+1];

    osc_state past_state_weighted(past_state_left);

    for (unsigned int i = 0; i < past_state_weighted.size(); i++){
        past_state_weighted[i] += delta * ( past_state_right[i] - past_state_left[i]);
        past_state_weighted[i] *= weight;
    }

    // cout << "Returning from link: ";
    // for (auto val : past_state_weighted){
    //     cout << val << " "; 
    // }
    // cout << endl;

    return past_state_weighted ;
}

template <class SOURCE, class DESTINATION>
float Link<SOURCE, DESTINATION>::timestep = 0.0;

void Oscillator::connect(Oscillator * osc, float weight, float delay){
    osc->incoming_osc.push_back(Link<Oscillator, Oscillator>(this, osc, weight, delay));
}

void Oscillator::evolve(EvolutionContext * evo){
    cout << "Evolving oscillator" << endl;
    Link<Oscillator, Oscillator>::timestep = evo->dt;
    history.push_back(state);

    boost::numeric::odeint::runge_kutta4<osc_state> stepper;
    auto lambda = [this](const osc_state & state, osc_state & dxdt, double t) {
                                    this->evolve_state(state, dxdt, t);
                                };

    stepper.do_step(lambda, this->state, evo->now, evo->dt);
}


void OscillatorNetwork::add_oscillator(Oscillator * oscillator){
    this->oscillators.push_back(oscillator);
}

void OscillatorNetwork::run(EvolutionContext * evo, double time){
    for (auto oscillator : oscillators){
        oscillator->evolve(evo);
    }
    evo->do_step();
}


osc_state Oscillator::none_state = {0.0, 0.0};

dummy_osc::dummy_osc(float k, double x, double v):k(k){
    cout << "creating oscillator" << endl;
    state = {x, v};
}

void dummy_osc::evolve_state(const osc_state & state, osc_state & dxdt, double t){
    dxdt[0] =   state[1];
    dxdt[1] = - k*state[0];

    for (auto input : incoming_osc){
        dxdt[1] += input.get(t)[0];
    }  
}

osc_state dummy_osc::none_state = {0.0, 0.0};
