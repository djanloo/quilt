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

OscillatorNetwork::OscillatorNetwork(oscillator_type osc_type, vector<ParaMap*> params, const Projection & self_projection){

    if (self_projection.start_dimension != self_projection.end_dimension){
        throw std::runtime_error("Oscillator Network: projection is not between the same objects.");
    }
    if (self_projection.start_dimension != params.size()){
        throw std::runtime_error("Oscillator Network: number of ParaMaps does not match space dimension.");
    }

    int N = self_projection.start_dimension;

    oscillators.reserve(self_projection.start_dimension);
    for (int i = 0; i < self_projection.start_dimension; i++){
        switch (osc_type)
        {
        case oscillator_type::harmonic:
            oscillators.push_back(new harmonic(*(params[i])));
            break;
        
        case oscillator_type::jensen_rit:
            break;

        default:
            throw std::runtime_error("OscillatorNetwork: the given oscillator type is not acceptable");
            break;
        }
    }
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++){
            if (i !=j ){
                if (std::abs(self_projection.weights[i][j]) > WEIGHT_EPS){
                    std::cout << "Connected " << i << " to " << j << std::endl;
                    oscillators[i]->connect(oscillators[j], self_projection.weights[i][j], self_projection.delays[i][j]);
                }
            }
        }
    }
}

void OscillatorNetwork::add_oscillator(Oscillator * oscillator){
    this->oscillators.push_back(oscillator);
}

void OscillatorNetwork::run(EvolutionContext * evo, double time){
    while (evo->now < time){
        cout << "Time: "<< evo->now <<endl;
        for (auto oscillator : oscillators){
            oscillator->evolve(evo);
        }
        evo->do_step();
    }
}

osc_state Oscillator::none_state = {0.0, 0.0};

harmonic::harmonic(const ParaMap & params){
    cout << "creating harmonic oscillator" << endl;
    state = {params.get("x0"), params.get("v0")};
    k = params.get("k");
    cout << "harmonic oscill state set" <<endl;
}

void harmonic::evolve_state(const osc_state & state, osc_state & dxdt, double t){
    dxdt[0] =   state[1];
    dxdt[1] = - k*state[0];

    for (auto input : incoming_osc){
        dxdt[1] += input.get(t)[0];
    }  
}

osc_state harmonic::none_state = {0.0, 0.0};


