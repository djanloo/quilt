#pragma once
#include "base.hpp"
#include "oscillators.hpp"

#include <stdexcept>
#include <memory>


using std::vector;
using std::shared_ptr;

class Oscillator;

/*************************************************** LINK BASE ************************************************/

// NOTE: the linking strategy is still 'not elegant' (pronounced 'notto ereganto' with the voice of Housemaster Henry Henderson from Spy x Family)
// 
// My intent was finding a strategy to template & polymorph stuff
// in order to reduce to the minimum value the number of templates in cython code.

// Clearly, the ideal minimum is 0.
// I will consider myself satisfied only when this minimum is reached, i.e. 
// when the code will deduce the right link function when just:
// 
// osc1.connect(osc2)
// 
// is called.

/**
 * @brief Base class of links
 * 
*/
class Link{
    public:
        shared_ptr<Oscillator> source;
        shared_ptr<Oscillator> target;
        float weight, delay;
        Link(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
            :   source(source),
                target(target),
                weight(weight),
                delay(delay)
        {
            if (weight == 0.0)
            {
                // No link-making procedure must arrive at this point
                // zero-valued links must be treated upstream
                throw runtime_error("Initialized a zero-weighted link between two oscillators");
            }
        }
        ~Link(){}
        
        virtual double get(int axis, double now) // Note: it needs `now` for the innner steps of RK 
        {
            throw runtime_error("Using virtual `get()` of LinkBase");
        };
        
        void set_evolution_context(EvolutionContext * evo)
        {
            this->evo = evo;
        };
    protected:
        EvolutionContext * evo;
};

/*************************************** LINK FACTORY *************************************/
// Builder method for Link-derived objects
template <typename DERIVED>
Link * link_maker(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
{
    return new DERIVED(source, target, weight, delay);
}

class LinkFactory{
    typedef std::function<Link*(shared_ptr<Oscillator>, shared_ptr<Oscillator>, float, float)> linker;
    public:
        LinkFactory();
        bool add_linker(std::pair<string, string> const& key, linker const& lker) {
            return _linker_map.insert(std::make_pair(key, lker)).second;
        }

        Link * get_link(shared_ptr<Oscillator> source,shared_ptr<Oscillator> target, float weight, float delay);

    private:
        map<std::pair<string, string>, linker> _linker_map;
};

// Singleton method to return a unique instance of LinkFactory
LinkFactory& get_link_factory();

/****************************************************** LINK MODELS ****************************************************/
class JRJRLink : public Link{
    public:
        JRJRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
            :   Link(source, target, weight, delay){}
        double get(int axis, double now) override;
};

class LJRLJRLink : public Link{
    public:
        LJRLJRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
            :   Link(source, target, weight, delay){}
        double get(int axis, double now) override;
};
