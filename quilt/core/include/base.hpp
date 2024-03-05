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
#include <functional>

using std::cout;
using std::endl;
using std::runtime_error;
using std::vector;
using std::map;
using std::string;

class RNG{
    public:
        RNG(const uint64_t seed):
        rng(seed),
        uniform(std::numeric_limits<double>::epsilon(), 1.0 - std::numeric_limits<double>::epsilon()){
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
        map<RNG*, bool> is_occupied;
        map<std::thread::id, RNG*> pids;

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
 * @class EvolutionContext
 * @brief The object that contains time evolution infos
 * 
 * @param[in] dt
 * 
*/
class EvolutionContext{
    public:
        double dt, now; // time in millis

        vector<double> times;
        unsigned int n_steps_done;

        EvolutionContext(double dt);
        void do_step();

        // Get index and deviation of a time value
        int     index_of(double time);
        double  deviation_of(double time);
};

typedef vector<double> dynamical_state;

/**
 * @class ContinuousRK
 * @brief A Natural Continuous Extension (Zennaro, 1986) of the famous fourth-order Runge Kutta method.
*/
class ContinuousRK{
    public:
        
        // These are the coefficients of the RK method
        vector<double> a = {0, 0.5, 0.5, 1};
        vector<double> b = {1.0/3.0, 1.0/6.0, 1.0/6.0, 1.0/3.0};
        vector<double> c = {0, 0.5, 0.5, 1};

        // These two make it possible to do a sequential updating of a set of CRK.
        // The system of equation (if no vanishing delays are present)
        // requires to update just one subsystem at a time since all the other variables
        // are locked to past values. To prevent the histories of two subsystems
        // from "shearing" and refernecing wrong past elements, the new point is first proposed
        // calling `compute_next()`
        // then when every CRK has done its proposal they are fixed using `fix_next()`
        dynamical_state proposed_state;
        vector<dynamical_state> proposed_evaluation;

        void set_dimension(unsigned int dimension)
        {
            space_dimension = dimension;
        }
        void set_evolution_equation(std::function<void(const dynamical_state & x, dynamical_state & dxdt, double t)> F)
        {
            evolve_state = F;
        };

        /**
         * The continuous parameters of the NCE. See "Natural Continuous extensions of Runge-Kutta methods", M. Zennaro, 1986.
        */
        vector<double> b_functions(double theta);

        ContinuousRK(){};

        vector<dynamical_state> state_history;

        /**
         * The K coefficients of RK method.
         * 
         * For each step previously computed, there are nu intermediate steps function evalutaions.
         * For each evluation the number of coeffiecients is equal to the dimension of the oscillator.
         * Thus for a N-long history of a nu-stage RK of an M-dimensional oscillator, the K coefficients
         * have shape (N, nu, M).
        */
        vector<vector<dynamical_state>> evaluation_history;
        /**
         * Computes the interpolation using the Natural Continuous Extension at a given time for a given axis (one variable of interest).
        */
        double get_past(int axis, double abs_time);
        void compute_next();
        void fix_next();

        void set_evolution_context(EvolutionContext * evo){
            this->evo = evo;
        }
    private:
        bool initialized = false;
        EvolutionContext * evo;
        unsigned int space_dimension = 0;
        std::function<void(const dynamical_state & x, dynamical_state & dxdt, double t)> evolve_state;
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
        map<string, float> value_map;
        ParaMap();
        ParaMap(const map<string, float> & value_map);

        void update(const ParaMap & new_values);
        void add(const string & key, float value);
        float get(const string & key) const ;
        float get(const string & key, float default_value);
        float has(const string & key){
            return value_map.find(key) != value_map.end();
        }
        
        friend std::ostream& operator<<(std::ostream& os, const ParaMap& obj)
        {
            os << "<ParaMap>" << endl;
            for (const auto & couple : obj.value_map)
            {
                string key = couple.first;
                float value = couple.second;
                os << "\t" << key << ": " << value << endl;
            }
            os << "</ParaMap>" << endl;
            return os;
        }
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
                std::cout << "*"<< std::flush;
                if (_count == _max-1) std::cout << std::endl;
            }
            _count ++;
        }
        
};
        