/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
*/
#pragma once
#include <vector>
#include <stdexcept>
#include "base_objects.hpp"
#include "network.hpp"

class EvolutionContext;
class Population;

typedef std::vector<double> osc_state;

enum class oscillator_type : unsigned int {harmonic, jensen_rit, red_wong_wang};

template <class SOURCE, class DESTINATION>
class Link{
    public:
        SOURCE * source;
        DESTINATION * destination;
        float weight, delay;
        static float timestep;

        Link(SOURCE * source, DESTINATION * destination, float weight, float delay):
        source(source), destination(destination),weight(weight),delay(delay){}
        osc_state get(double now);
};

class Oscillator{
    public:
        osc_state state;
        static osc_state none_state;

        std::vector<osc_state> history;

        std::vector< Link<Oscillator, Oscillator>> incoming_osc;
        // std::vector< Link<Population, Oscillator>> incoming_pops;

        Oscillator(){state = {0.0, 0.0};}
        void connect(Oscillator * osc, float weight, float delay);

        void evolve(EvolutionContext * evo);
        
        virtual void evolve_state(const osc_state & state, osc_state & dxdt, double t){
            throw std::runtime_error("Using virtual evolve_state of oscillator");
            };
};

class OscillatorNetwork{
    public:
        OscillatorNetwork(oscillator_type osc_type, std::vector<ParaMap*> params, const Projection & self_projection);
        
        std::vector<Oscillator*> oscillators;
        
        void run(EvolutionContext * evo, double time);
        void add_oscillator(Oscillator * oscillator);
};


class harmonic : public Oscillator{
    public:
        float k;
        static osc_state none_state;
        harmonic(const ParaMap & params);
        void evolve_state(const osc_state & state, osc_state & dxdt, double t) override;
};