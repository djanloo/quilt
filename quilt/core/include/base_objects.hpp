#pragma once
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>

/**
 * @class HierarchicalID
 * @brief The ID object to identify nested structures
 * 
 * Create without arguments to get a supercontainer.
 * Then create a nested object by passing as argument 
 * the pointer to the parent container.
 * 
*/
class HierarchicalID{
    public:
        HierarchicalID * parent;
        HierarchicalID():parent(NULL),local_id(-1),n_subclasses(0){}
        HierarchicalID(HierarchicalID * parent);
        unsigned int get_id();
    private:
        unsigned int local_id;
        unsigned int n_subclasses;
};

/**
 * @class EvolutuionContext
 * @brief The object that contains time evolution infos
 * 
 * @param[in] dt
 * 
*/
class EvolutionContext{
    public:
        double dt, now; // time in millis

        EvolutionContext(double dt);
        void do_step();
};

/**
 * 
 * @class ParaMap
 * @brief a dictionary class
 * 
 * It's just a bridge between python and C++
 * 
*/
class ParaMap{
    public:
        std::map<std::string, float> value_map;
        ParaMap();
        ParaMap(const std::map<std::string, float> & value_map);

        void update(const ParaMap & new_values);
        void add(const std::string & key, float value);
        float get(const std::string & key) const ;
};