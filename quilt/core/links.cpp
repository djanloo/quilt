#include "include/links.hpp"

/******************************************* LINK BASE ****************************************/
// Singleton method to return a unique instance of LinkFactory
LinkFactory& get_link_factory(){
    static LinkFactory link_factory;
    return link_factory;
}

LinkFactory::LinkFactory()
{
    // Here I place the builders of each type of link
    add_linker(std::make_pair("base", "base"), link_maker<Link>);
    add_linker(std::make_pair("jansen-rit", "jansen-rit"), link_maker<JRJRLink>);
    add_linker(std::make_pair("leon-jansen-rit", "leon-jansen-rit"), link_maker<LJRLJRLink>);
}

Link * LinkFactory::get_link(shared_ptr<Oscillator> source, shared_ptr<Oscillator> target, float weight, float delay)
        {
            
            std::pair<string, string> key = std::make_pair(source->oscillator_type, target->oscillator_type);
            cout << "Link factory: making link (" + key.first + "-->" + key.second <<")"<< endl;
            auto it = _linker_map.find(key);
            if (it == _linker_map.end()) { throw runtime_error("No linker was found for the couple (" + key.first + " "+ key.second + ")"); }
            return (it->second)(source, target, weight, delay);
};

/************************************************* LINK MODELS ************************************************8*/
double JRJRLink::get(int axis, double now){
    // cout << "Getting past from JRJR link" << endl;
    return source->get_past(axis, now - delay);
}

double LJRLJRLink::get(int axis, double now){
    // cout << "Getting past from LJRLJR link" << endl;
    return source->get_past(axis, now - delay);
}