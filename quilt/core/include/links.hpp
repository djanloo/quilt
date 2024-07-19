#pragma once
#include "base.hpp"

#include <stdexcept>
#include <memory>

using std::vector;
using std::shared_ptr;

// Forward declarations
class Oscillator;

/**
 * @brief Base class for links between oscillators.
 *
 * This class serves as the base class for different link models connecting oscillators.
 * It provides a common interface and structure for various link types.
 */
class Link {
public:
    shared_ptr<Oscillator> source; ///< The source oscillator of the link.
    shared_ptr<Oscillator> target; ///< The target oscillator of the link.
    float weight; ///< Weight of the link.
    float delay; ///< Delay in the link.
    ParaMap* params; ///< Parameters associated with the link.

    /**
     * @brief Constructor for the Link class.
     * @param source The source oscillator.
     * @param target The target oscillator.
     * @param weight Weight of the link.
     * @param delay Delay in the link.
     * @param params Parameters associated with the link.
     */
    Link(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : source(source),
          target(target),
          weight(weight),
          delay(delay),
          params(params) {}
    // Virtual destructor since it's a base class  
    virtual ~Link(){};

    /**
     * @brief Virtual function to get the value of the link.
     * @param axis The axis for which to get the value.
     * @param now Current time for the inner steps of Runge-Kutta.
     * @return The value of the link.
     */
    virtual double get(int /*axis*/, double /*now*/) {
        throw std::runtime_error("Using virtual `get()` of LinkBase");
    };

    /**
     * @brief Set the evolution context for the link.
     * @param evo EvolutionContext for the link.
     */
    void set_evolution_context(EvolutionContext* evo) {
        // get_global_logger().log(DEBUG, "set EvolutionContext of Link");
        this->evo = evo;
    };

protected:
    EvolutionContext* evo; ///< EvolutionContext associated with the link.
};

/**
 * @brief Factory class for creating different link types.
 */
class LinkFactory {
    typedef std::function<Link*(shared_ptr<Oscillator>, shared_ptr<Oscillator>, float, float, ParaMap*)> linker;

public:
    LinkFactory();

    /**
     * @brief Add a linker function for creating a specific type of link.
     * @param key Pair of oscillator types as a key.
     * @param lker Linker function.
     * @return True if the linker was added successfully, false otherwise.
     */
    bool add_linker(std::pair<std::string, std::string> const& key, linker const& lker) {
        stringstream ss;
        ss << "Registering a new Link: "<< key.first << " -> " << key.second;
        get_global_logger().log(DEBUG, ss.str());
        return _linker_map.insert(std::make_pair(key, lker)).second;
    }

    /**
     * @brief Get a specific type of link using the provided parameters.
     * @param source Source oscillator.
     * @param target Target oscillator.
     * @param weight Weight of the link.
     * @param delay Delay in the link.
     * @param params Parameters associated with the link.
     * @return A pointer to the created link.
     */
    Link* get_link(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params);

private:
    std::map<std::pair<std::string, std::string>, linker> _linker_map; ///< Map of oscillator type pairs to linker functions.
};

// Builder method for Link-derived objects
template <typename DERIVED>
Link * link_maker(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap * params)
{
    return new DERIVED(source, target, weight, delay, params);
}

/**
 * @brief Get a reference to the singleton instance of LinkFactory.
 * @return Reference to the LinkFactory instance.
 */
LinkFactory& get_link_factory();

/**
 * @brief Derived class for a specific link model (JRJR).
 */
class JRJRLink : public Link {
public:
    /**
     * @brief Constructor for JRJRLink.
     * @param source Source oscillator.
     * @param target Target oscillator.
     * @param weight Weight of the link.
     * @param delay Delay in the link.
     * @param params Parameters associated with the link.
     */
    JRJRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

    /**
     * @brief Get the value of the link for a specific axis at the given time.
     * @param axis The axis for which to get the value.
     * @param now Current time.
     * @return The value of the link.
     */
    double get(int axis, double now) override;
};

/**
 * @brief Derived class for another specific link model (LJRLJR).
 */
class LJRLJRLink : public Link {
public:
    /**
     * @brief Constructor for LJRLJRLink.
     * @param source Source oscillator.
     * @param target Target oscillator.
     * @param weight Weight of the link.
     * @param delay Delay in the link.
     * @param params Parameters associated with the link.
     */
    LJRLJRLink(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay, ParaMap* params)
        : Link(source, target, weight, delay, params) {}

    /**
     * @brief Get the value of the link for a specific axis at the given time.
     * @param axis The axis for which to get the value.
     * @param now Current time.
     * @return The value of the link.
     */
    double get(int axis, double now) override;
};

