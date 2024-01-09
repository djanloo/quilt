/**
 * Stuff for oscillators. First I try a sparse version.
 * I'm really aware that a dense "matrix" version of this would 
 * probably be better, but let's see.
 * 
*/
#pragma once
#include <vector>
#include <stdexcept>

class EvolutionContext;
class Population;

typedef std::vector<double> osc_state;

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
        vector<Oscillator*> oscillators;
        void run(EvolutionContext * evo, double time);
        void add_oscillator(Oscillator * oscillator);
};


class dummy_osc : public Oscillator{
    public:
        float k;
        static osc_state none_state;
        dummy_osc(float k, double x, double v);
        void evolve_state(const osc_state & state, osc_state & dxdt, double t) override;
};