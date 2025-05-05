#include "include/oscillators.hpp"
// #include "include/multiscale.hpp"
#include "include/links.hpp"

/******************************************* LINK BASE ****************************************/
// Singleton method to return a unique instance of LinkFactory
LinkFactory& get_link_factory(){
    static LinkFactory link_factory;
    return link_factory;
}

LinkFactory::LinkFactory()
{
    add_linker(std::make_pair("base", "base"), link_maker<Link>);
    add_linker(std::make_pair("jansen-rit", "jansen-rit"), link_maker<JRJRLink>);
    add_linker(std::make_pair("noisy-jansen-rit", "noisy-jansen-rit"), link_maker<NJRNJRLink>);
    add_linker(std::make_pair("binoisy-jansen-rit", "binoisy-jansen-rit"), link_maker<BNJRBNJRLink>);
    add_linker(std::make_pair("leon-jansen-rit", "leon-jansen-rit"), link_maker<LJRLJRLink>);
}

Link * LinkFactory::get_link(Oscillator * source, Oscillator * target, float weight, float delay,  ParaMap * params)
{
    std::pair<string, string> key = std::make_pair(source->oscillator_type, target->oscillator_type);
    // cout << "Link factory: making link (" + key.first + "-->" + key.second <<")"<< endl;
    auto it = _linker_map.find(key);
    if (it == _linker_map.end()) { 
        get_global_logger().log(ERROR, "No linker was found for the couple (" + key.first + " "+ key.second + ")");
        throw runtime_error("No linker was found for the couple (" + key.first + " "+ key.second + ")"); 
    }
    return (it->second)(source, target, weight, delay, params);
};

/************************************************* LINK MODELS ************************************************8*/
double JRJRLink::get(int axis, double now){
    double result = weight * static_cast<jansen_rit_oscillator*>(source)->sigm(source->get_past(axis, now - delay));
    if (axis != 0){
        get_global_logger().log(ERROR, "Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
        throw runtime_error("Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
    }
    return result;
}

double LJRLJRLink::get(int axis, double now){
    if (axis != 6){
        get_global_logger().log(ERROR, "Leon-Jansen-Rit model can only ask for axis 6 (differential activity)");
        throw runtime_error("Leon-Jansen-Rit model can only ask for axis 6 (differential activity)");
    }
    double result = weight * static_cast<leon_jansen_rit_oscillator*>(source)->sigm(source->get_past(axis, now - delay));
    return result;
}

double NJRNJRLink::get(int axis, double now){
    double result = weight * static_cast<noisy_jansen_rit_oscillator*>(source)->sigm(source->get_past(axis, now - delay));
    if (axis != 0){
        get_global_logger().log(ERROR, "noisy Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
        throw runtime_error("noisy Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
    }
    return result;
}

double BNJRBNJRLink::get(int axis, double now){
    double result = weight * static_cast<noisy_jansen_rit_oscillator*>(source)->sigm(source->get_past(axis, now - delay));
    if (axis != 0){
        get_global_logger().log(ERROR, "binoisy Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
        throw runtime_error("binoisy Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
    }
    return result;
}
