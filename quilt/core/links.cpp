#include "include/oscillators.hpp"
// #include "include/multiscale.hpp"
#include "include/links.hpp"

bool LONG_INTERNEURONS = false;

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
double JRJRLink::get_rate(double now){
    double result = 0;
    if (LONG_INTERNEURONS){
        result = weight * static_cast<jansen_rit_oscillator*>(source)->sigm(source->get_past(0, now - delay));
    }
    else{
        result = weight * static_cast<jansen_rit_oscillator*>(source)->sigm(source->get_past(1, now - delay)-source->get_past(2, now - delay));
    }
    return result;
}

double LJRLJRLink::get_rate(double now){
    double result = weight * static_cast<leon_jansen_rit_oscillator*>(source)->sigm(source->get_past(6, now - delay));
    return result;
}

double NJRNJRLink::get_rate(double now){
    double result = 0;
    if (LONG_INTERNEURONS){
        result = weight * static_cast<noisy_jansen_rit_oscillator*>(source)->sigm(source->get_past(0, now - delay));
    }
    else{
        result = weight * static_cast<noisy_jansen_rit_oscillator*>(source)->sigm(source->get_past(1, now - delay)-source->get_past(2, now - delay));
    }
    return result;
}

double BNJRBNJRLink::get_rate(double now){
    double result = 0;
    if (LONG_INTERNEURONS){
        result = weight * static_cast<binoisy_jansen_rit_oscillator*>(source)->sigm(source->get_past(0, now - delay));
    }
    else{
        result = weight * static_cast<binoisy_jansen_rit_oscillator*>(source)->sigm(source->get_past(1, now - delay)-source->get_past(2, now - delay));
    }
    return result;
}
