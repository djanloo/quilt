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


class ParaMap{
    public:
        typedef std::variant<int, float, string> param_t;
        map<string, param_t> value_map;

        ParaMap():value_map(){};

        ParaMap(map<string, param_t> values){
            for (auto pair : values){
                add(pair.first, pair.second);
            }
        }

        template <typename T>
        T get(string key){
            auto it = value_map.find(key);
            cout << "Getting " << key << endl;
            return std::get<T>(it->second);
        }

        float get(string key, float default_value) {
            auto it = value_map.find(key);
            if (it == value_map.end()){
                return default_value;
            }
            else{
                return std::get<float>(it->second);
            }
        }

        template <typename T>
        void add(string key, T value){
            value_map[key] = value;
            // cout <<"Inserted" << value_map[key] << endl; 
        }

        void update(ParaMap other) {
            for (auto entry : other.value_map) {
                value_map[entry.first] = entry.second;
            }
        }
        friend std::ostream& operator<<(std::ostream& os, const ParaMap& paramap) {
            os << "ALbert printed a paramap"<< endl;
            return os;
        }
};

// /**
//  * @class ParaMap
//  * @brief A class representing a parameter map with key-value pairs.
//  * 
//  * The ParaMap class allows storing and retrieving values of various types associated with string keys.
//  * It provides methods for adding, getting, checking existence, removing, and updating key-value pairs.
//  * Additionally, it supports ostream output for easy debugging and visualization of the parameter map.
//  */
// class ParaMap {
// public:
//     /**
//      * @brief Default constructor for ParaMap.
//      */
//     ParaMap() = default;

//     /**
//      * @brief Constructor for ParaMap with initial values.
//      * @param init_values Initial key-value pairs to populate the ParaMap.
//      */
//     ParaMap(map<string, float> init_values) {
//         for (auto& pair : init_values) {
//             add(pair.first, pair.second);
//         }
//     }

//     /**
//      * @brief Template method to add a key-value pair to the ParaMap.
//      * @tparam T Type of the value.
//      * @param key Key for the parameter.
//      * @param value Value associated with the key.
//      */
//     template <typename T>
//     void add(const std::string& key, const T& value) {
//         value_map[key] = value;
//     }

//     /**
//      * @brief Template method to get the value associated with a key.
//      * @tparam T Type of the expected value.
//      * @param key Key for the parameter.
//      * @return The value associated with the key.
//      * @throws std::out_of_range if the key is not found.
//      * @throws std::runtime_error if the stored value type mismatches the expected type.
//      */
//     template <typename T>
//     T get(const std::string& key) {
//         auto it = value_map.find(key);
//         if (it == value_map.end()) {
//             throw std::out_of_range("Key not found in Paramap");
//         }
//         // const auto& val = it->second;
//         // if (!val.has_value() || val.type() != typeid(T)) {
//         //     throw std::runtime_error("Value type mismatch in Paramap for key " + it->first +
//         //                              ": should be of type <" + typeid(T).name() +
//         //                              "> but is of type <" + val.type().name() + ">");
//         // }
//         // return std::any_cast<const T&>(val);
//         return std::get<T>(it->second);
//     }

//     /**
//      * @brief Template method to get the value associated with a key or a default value if the key is not found.
//      * @tparam T Type of the expected value.
//      * @param key Key for the parameter.
//      * @param default_value Default value to return if the key is not found.
//      * @return The value associated with the key or the default value if the key is not found.
//      * @throws std::runtime_error if the stored value type mismatches the expected type.
//      */
//     float get(string key, float default_value) {
//         auto it = value_map.find(key);
//         if (it == value_map.end()) {
//             this->add(key, default_value);
//             return default_value;
//         }
//         // const auto& val = it->second;
//         // if (!val.has_value() || val.type() != typeid(T)) {
//         //     throw std::runtime_error("Value type mismatch in Paramap for key " + key);
//         // }
//         return std::get<float>(it->second);
//     }

//     /**
//      * @brief Checks if a key exists in the ParaMap.
//      * @param key Key to check.
//      * @return True if the key exists, false otherwise.
//      */
//     bool has(const std::string& key) const {
//         return value_map.find(key) != value_map.end();
//     }

//     /**
//      * @brief Removes a key-value pair from the ParaMap.
//      * @param key Key to remove.
//      */
//     void remove(const std::string& key) {
//         value_map.erase(key);
//     }

//     /**
//      * @brief Updates the ParaMap with key-value pairs from another ParaMap.
//      * @param other Another ParaMap from which to update key-value pairs.
//      */
//     void update(const ParaMap& other) {
//         for (const auto& entry : other.value_map) {
//             value_map[entry.first] = entry.second;
//         }
//     }

//     /**
//      * @brief Friend function to output the ParaMap to an ostream.
//      * @param os Output stream.
//      * @param paramap ParaMap to output.
//      * @return Reference to the output stream.
//      */
//     friend std::ostream& operator<<(std::ostream& os, const ParaMap& paramap) {
//         for (const auto& entry : paramap.value_map) {
//             os << entry.first << ": ";
//             if (std::holds_alternative<int>(entry.second))
//                 os <<std::get<int>(entry.second);
//             if (std::holds_alternative<float>(entry.second))
//                 os <<std::get<float>(entry.second);
//             if (std::holds_alternative<string>(entry.second))
//                 os << std::get<string>(entry.second);
//             os << endl;
//         }

//         return os;
//     }

// private:
//     map<string, std::variant<int, float, double>> value_map;  /**< Map to store key-value pairs. */
// };


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
        