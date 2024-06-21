#include "include/oscillators.hpp"
#include "include/multiscale.hpp"
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
    
    add_linker(std::make_pair("jansen-rit", "transducer"), link_maker<JR2TLink>);
    add_linker(std::make_pair("transducer", "jansen-rit"), link_maker<T2JRLink>);
    
    add_linker(std::make_pair("leon-jansen-rit", "leon-jansen-rit"), link_maker<LJRLJRLink>);
}

Link * LinkFactory::get_link(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay,  ParaMap * params)
{
    std::pair<string, string> key = std::make_pair(source->oscillator_type, target->oscillator_type);
    // cout << "Link factory: making link (" + key.first + "-->" + key.second <<")"<< endl;
    auto it = _linker_map.find(key);
    if (it == _linker_map.end()) { throw runtime_error("No linker was found for the couple (" + key.first + " "+ key.second + ")"); }
    return (it->second)(source, target, weight, delay, params);
};

/************************************************* LINK MODELS ************************************************8*/
double JRJRLink::get(int axis, double now){
    double result = weight * std::static_pointer_cast<jansen_rit_oscillator>(source)->sigm(source->get_past(axis, now - delay));
    if (axis != 0) throw runtime_error("Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");
    // cout << "Getting past from JRJR link" << endl;
    // cout << "JRJR got "<<result<< endl;
    return result;
}

double LJRLJRLink::get(int axis, double now){
    // cout << "Getting past from LJRLJR link" << endl;
    if (axis != 6) throw runtime_error("Jansen-Rit model can only ask for axis 6 (differential activity)");
    double result = weight * std::static_pointer_cast<leon_jansen_rit_oscillator>(source)->sigm(source->get_past(axis, now - delay));
    return result;
}

/******************************************** MULTISCALE LINK MODELS *******************************************8*/


double T2JRLink::get(int axis, double now){
    // This function is called by Oscillator objects linked to this transducer
    // during their evolution function
    get_global_logger().log(DEBUG, "T2JRLink: getting t=" + to_string(now-delay) );

    // Returns the activity of the spiking population back in the past
    // Note that the average on the large time scale is done by Transducer::get_past()
    double result = weight * std::static_pointer_cast<Transducer>(source)->get_past(axis, now - delay); //axis is useless
    return result;
}

double JR2TLink::get(int axis, double now){

    if (axis != 0) throw runtime_error("Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");

    // Returns the rate of the oscillator back in the past 
    double v0 =  source->get_past(axis, now - delay);
    double rate = std::static_pointer_cast<jansen_rit_oscillator>(source)->sigm(v0);
    double result = weight * rate;

    // cout << "Getting past from JRJR link" << endl;
    // cout << "JRJR got "<<result<< endl;

    //NOTE: Jansen-Rit Model is in ms^-1. Result must be converted.
    result *= 1e3;

    std::stringstream ss;
    ss << "JR2TLink: getting t = " << now-delay << " : v0 = " << v0 << " mV, rate = " << rate << " ms^-1, weight = " << weight << " (returning " << result << " Hz)";
    get_global_logger().log(DEBUG, ss.str());

    return result;
}