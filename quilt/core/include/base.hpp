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

#include <variant>

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
 * @brief A class representing a parameter map with keys and associated variant values.
 * 
 * Possible types are int, float and string.
 */
class ParaMap{
    public:
        typedef std::variant<int, float, string> param_t;
        map<string, param_t> value_map;

        /** 
         * @brief Default constructor for ParaMap.
         */
        ParaMap():value_map(){};

        /**
         * @brief Constructor for ParaMap with initial values.
         * @param values A map containing initial key-value pairs.
         */
        ParaMap(map<string, param_t> values){
            for (auto pair : values){
                add(pair.first, pair.second);
            }
        }

        /**
         * @brief Template function to retrieve the value associated with a key.
         * @tparam T The type of the value to retrieve.
         * @param key The key to look up in the map.
         * @return The value associated with the key.
         * @throw std::out_of_range if the key is not found.
         */
        template <typename T>
        T get(const string& key){
            auto it = value_map.find(key);
            if (it == value_map.end()){
                throw std::out_of_range("Attribute " + key +" not found in ParaMap");
            }
            // cout << "Getting " << key << endl;
            return std::get<T>(it->second);
        }
        /**
         * @brief Function to retrieve the value associated with a key with a default value.
         * @param key The key to look up in the map.
         * @param default_value The default value to return if the key is not found.
         * @return The value associated with the key or the default value.
         * 
         * Note: in case the value is not found it is assigned to `default_value`. 
         * Future requests will return this value.
         */
        float get(const string& key, float default_value) {
            auto it = value_map.find(key);
            if (it == value_map.end()){
                value_map[key] = default_value;
                return default_value;
            }
            else{
                return std::get<float>(it->second);
            }
        }

        /**
         * @brief Template function to add a key-value pair to the map.
         * @tparam T The type of the value to add.
         * @param key The key to add.
         * @param value The value to associate with the key.
         */
        template <typename T>
        void add(const string& key, const T& value){
            value_map[key] = value;
        }

        /**
         * @brief Function to update the map with key-value pairs from another ParaMap.
         * @param other Another ParaMap to merge with the current one.
         */
        void update(const ParaMap& other) {
            for (auto entry : other.value_map) {
                value_map[entry.first] = entry.second;
            }
        }

        /**
         * @brief Overloaded output stream operator to print the contents of the ParaMap.
         * @param os The output stream.
         * @param paramap The ParaMap to print.
         * @return The modified output stream.
         */
        friend std::ostream& operator<<(std::ostream& os, const ParaMap& paramap){
            os << "<ParaMap>" << endl;
            for (const auto& entry : paramap.value_map) {
                os << "\t" <<entry.first << ": ";
                if (std::holds_alternative<int>(entry.second)){
                    os <<std::get<int>(entry.second);
                    os << " (int)";
                }
                if (std::holds_alternative<float>(entry.second)){
                    os <<std::get<float>(entry.second);
                    os << " (float)";
                }
                if (std::holds_alternative<string>(entry.second)){
                    os << std::get<string>(entry.second);
                    os << " (string)";
                }
                os << endl;
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
        