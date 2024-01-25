#pragma once
#include "../pcg/include/pcg_random.hpp"

#include <iostream>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>
#include <random>
#include <thread>
#include <mutex>

class RNG{
    public:
        RNG(const uint64_t seed):
        rng(seed),
        uniform(std::numeric_limits<double>::epsilon(),1.0 - std::numeric_limits<double>::epsilon()){
        }
        double get_uniform(){return uniform(rng);}
        int get_int(){return static_cast<int>(rng());}

    private:        
        pcg32 rng;
        std::uniform_real_distribution<double> uniform;
};

class RNGDispatcher{
    public:
        RNGDispatcher(unsigned int n_rngs, const uint64_t seed0):mutex(){
            for (unsigned int i=0; i < n_rngs; i++){
                rngs.push_back(new RNG(seed0 + static_cast<uint64_t>(i)));
            }
        }
        ~RNGDispatcher(){
            for (RNG * rng : rngs ){
                // std::cout << "Deleting TLRNG at "<<rng<<std::endl;
                delete rng;
            }
            // std::cout << "Deleted all " <<std::endl;
        }

        RNG * get_rng(){
            std::lock_guard<std::mutex> lock(mutex);

            for (auto& rng : rngs){
                if (!is_occupied[rng]){
                    // std::cout << "Giving TLRNG "<< rng << " to PID " << std::this_thread::get_id()<< std::endl; 
                    pids[std::this_thread::get_id()] = rng;
                    is_occupied[rng] = true;
                    return rng;
                }
            }
            throw std::runtime_error("No thread-locked random number generator was found to be free");
        }

        void free(){
            std::lock_guard<std::mutex> lock(mutex);
            RNG * rng = pids[std::this_thread::get_id()];
            // std::cout << "Freeing " <<  rng << " from PID: "<< std::this_thread::get_id()<<std::endl;
            is_occupied[rng] = false;
            pids.erase(std::this_thread::get_id());
        }

    private: 
        std::mutex mutex; 
        std::vector<RNG*> rngs;
        std::map<RNG*, bool> is_occupied;
        std::map<std::thread::id, RNG*> pids;

};

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

/**
 * @class progress
 * @brief A really basic progress bar
 * 
 * Because boost::timer::progress_display can't set a verbosity level.
*/
class progress{
    public:
        int max, _max, count, _count;
        unsigned int verbosity;
        progress(int max, unsigned int verbosity):max(max-1), _max(50), count(0), _count(0), verbosity(verbosity){
            if (verbosity > 0){
                for (int i = 0; i < _max; i++){
                    std::cout << "-" ;
                }
                std::cout << std::endl;
            }
        }
        unsigned long  operator++(){ 
            if ( static_cast<int>(static_cast<float>(count)/max*_max) > _count) print(); 
            count++;
            return count; 
        }
        
        void print(){
            if (verbosity > 0){
                std::cout << "*"<< std::flush; ;
                if (_count == _max-1) std::cout << std::endl;
            }
            _count ++;
        }
        
};
        