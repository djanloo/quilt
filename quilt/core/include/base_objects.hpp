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

class ThreadLockedRNG{
    public:
        ThreadLockedRNG(const uint64_t seed):
        rng(seed),
        mutex(),
        uniform(std::numeric_limits<double>::epsilon(),1.0 - std::numeric_limits<double>::epsilon()){
            unlock();
        }

        void lock()  {uniqueLock.lock();  }
        void unlock(){uniqueLock.unlock();}
        bool is_locked(){ 
            // std::cout << "TLRNG at "<<this<< " was asked if free and returned "<<uniqueLock.owns_lock()<<std::endl;
            return uniqueLock.owns_lock();
            }

        double get_uniform(){return uniform(rng);}
        int get_int(){return static_cast<int>(rng());}

    private:        
        pcg32 rng;

        std::mutex mutex;
        std::unique_lock<std::mutex> uniqueLock{mutex};

        std::uniform_real_distribution<double> uniform;
};

class ThreadLockedRNGDispatcher{
    public:
        ThreadLockedRNGDispatcher(unsigned int n_rngs, const uint64_t seed0){
            for (unsigned int i=0; i < n_rngs; i++){
                tl_rngs.push_back(new ThreadLockedRNG(seed0 + static_cast<uint64_t>(i)));
            }
        }
        ~ThreadLockedRNGDispatcher(){
            for (auto tl_rng : tl_rngs ){
                delete tl_rng;
            }
        }

        ThreadLockedRNG * get_rng(){
            for (auto& rng : tl_rngs){
                if (!rng->is_locked()){
                    rng->lock();
                    // std::cout << "Giving TLRNG "<< rng << " to PID " << std::this_thread::get_id()<< std::endl; 
                    pids[std::this_thread::get_id()] = rng;
                    return rng;
                }
            }
            throw std::runtime_error("No thread-locked random number generator was found to be free");
        }
        void free(){
            // std::cout << "Freeing for PID: "<< std::this_thread::get_id()<<std::endl;
            ThreadLockedRNG * rng = pids[std::this_thread::get_id()];
            pids.erase(std::this_thread::get_id());
            rng->unlock();
        }

    private:
        std::vector<ThreadLockedRNG*> tl_rngs; //!< Thread locked random number generator
        std::map<std::thread::id, ThreadLockedRNG*> pids;

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
        