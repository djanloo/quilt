#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>
#include <iostream>
#include <string>

#include <boost/timer/progress_display.hpp>
#include "include/base.hpp"

using std::cout;
using std::endl;
using std::runtime_error;
using std::vector;

HierarchicalID::HierarchicalID(HierarchicalID * parent): parent(parent),n_subclasses(0){
    local_id = parent->n_subclasses;
    parent->n_subclasses ++;
}
unsigned int HierarchicalID::get_id(){return local_id;}

EvolutionContext::EvolutionContext(double dt)
    :   dt(dt),
        now(0.0){}

void EvolutionContext::do_step(){
    now += dt;
    n_steps_done ++;
    times.push_back(now);
}

int EvolutionContext::index_of(double time)
{
    if (time < 0) throw runtime_error("Requested index of a negative time: " + std::to_string(time) );
    return static_cast<int>(time/dt);
}
double EvolutionContext::deviation_of(double time)
{
    double deviation = time/dt - index_of(time);

    // Deviation is by definition positive
    deviation = (deviation < 0) ? 0.0 : deviation;
    
    return deviation;
}

ParaMap::ParaMap(){}

ParaMap::ParaMap(const std::map<std::string, float> & value_map)
    :   value_map(value_map){}

void ParaMap::add(const std::string& key, float value){
        value_map[key] = value;
        }

float ParaMap::get(const std::string& key) const 
{
    float return_value = 0.0;
    try{
        return_value = value_map.at(key);
    }catch (const std::out_of_range & e){
        throw std::out_of_range("Missing parameter " + key);
    }
    return return_value;
    }

float ParaMap::get(const std::string& key, float default_value) const 
{
    float return_value = 0.0;
    try{
        return_value = value_map.at(key);
    }catch (const std::out_of_range & e){
        return default_value;
    }
    return return_value;
}

void ParaMap::update(const ParaMap & new_values){
    for (const auto & couple: new_values.value_map){
        this->value_map[couple.first] = couple.second;
    }
}

vector<double> ContinuousRK::b_functions(double theta){
    vector<double> result(4);

    result[0] = 2*(1-4*b[0])*std::pow(theta, 3) + 3*(3*b[0] - 1)*theta*theta + theta;
    for (int i = 1; i < 4; i++){
        result[i] = 4*(3*c[i] - 2)*b[i]*std::pow(theta, 3) + 3*(3-4*c[i])*b[i]*theta*theta;
    }
    return result;
}

double ContinuousRK::get_past(int axis, double abs_time){

    // Split in bin_index + fractionary part
    int bin_id = evo->index_of(abs_time);
    double theta = evo->deviation_of(abs_time);

    if (bin_id == static_cast<int>(state_history.size())) bin_id -= 1;

    if (bin_id<0) throw runtime_error("Requested past state that lays before initialization");

    // Get the values and the interpolation weights related to that moment in time
    double y = state_history[bin_id][axis];
    vector<double> b_func_values = b_functions(theta);

    // Updates using the interpolant
    for (int nu = 0; nu < 4; nu++){
        y += evo->dt * b_func_values[nu] * evaluation_history[bin_id][nu][axis];
    }

    return y;
}

void ContinuousRK::compute_next(){
    if (space_dimension == 0) throw runtime_error("Space dimension not set in ContinuousRK");

    proposed_evaluation = vector<dynamical_state>(4, dynamical_state(space_dimension, 0));

    double t_eval;
    dynamical_state x_eval;

    // Compute the K values
    for (int nu = 0; nu < 4; nu++){ 

        // When
        t_eval = evo->now + c[nu] * evo->dt;

        // Where
        x_eval = state_history.back();
        if (nu != 0) {
            // This could be extended for other methods
            // by letting a to be a matrix
            for (unsigned int i = 0; i < space_dimension; i++){
                x_eval[i] += evo->dt * a[nu] * proposed_evaluation[nu-1][i];
            }
        }//~Where


        // Assigns the new K evaluation to the value of the evolution function
        evolve_state(x_eval, proposed_evaluation[nu], t_eval);

    }//~Compute K values

    // Updates the state
    proposed_state = state_history.back();
    for (unsigned int i = 0; i < space_dimension; i++){
        for (int nu = 0; nu < 4; nu++){
            proposed_state[i] += evo->dt * b[nu] * proposed_evaluation[nu][i];
        }
    }//~Updates the state
}

void ContinuousRK::fix_next(){
    state_history.push_back(proposed_state);
    evaluation_history.push_back(proposed_evaluation);
}



