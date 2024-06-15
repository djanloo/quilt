#include "include/links.hpp"
#include "include/network.hpp"
#include "include/multiscale.hpp"


Transducer::Transducer(Population * population, ParaMap * params, MultiscaleNetwork * multinet)
    :   Oscillator(params, multinet->oscnet),
        population(population),
        multinet(multinet)
{
    oscillator_type = "transducer";

    // Adds the monitor
    monitor = population->spiking_network->add_spike_monitor(population);

    // Adds the injector
    injector = new PoissonSpikeSource(population, 10, 0.5, 0.2, -1, 0 );
    population->spiking_network->add_injector(injector);

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t)
    {
        throw runtime_error("Calling 'evolve_state' of a transducer object is not allowed");
    };

    eeg_voi = [](const dynamical_state & x){ 
        throw runtime_error("Calling 'eeg_voi' of a transducer object is not allowed");
        return 0.0;
    };
}

/**
 * @brief Evolution method of the transducer. Must be called once every big time step.
*/
void Transducer::evolve()
{
    // Sets the rate of the PoissonSpikeSource as 
    // the weighted sum of the rates of the incoming oscillators
    double rate = 0; 
    double single_input_rate = 0;
    for (auto input : incoming_osc){
        single_input_rate = input->get(0, oscnet->get_evolution_context()->now);
        rate += single_input_rate;
        cout << "\tone input is " << single_input_rate << endl;
    }
    cout << "\t\tOverall input is " << single_input_rate << endl;

    injector->set_rate(rate);
}

double Transducer::get_past(unsigned int /*axis*/, double time)
{

    // I want to get the avg rate of the pop in [t-T/2, t+T/2]
    EvolutionContext * oscnet_evo = multinet->oscnet->get_evolution_context();
    double T = oscnet_evo->dt;

    EvolutionContext * spikenet_evo = multinet->spikenet->get_evolution_context();
    int time_idx_1 = spikenet_evo->index_of(time - T/2);
    int time_idx_2 = spikenet_evo->index_of(time + T/2);

    double theta = evo->deviation_of(time); //TODO: make this not useless

    vector<int> activity_history = monitor->get_history();
    double avg_rate = 0.0;
    // Compute the average rate
    for (int i = time_idx_1; i < time_idx_2; i++){
        avg_rate += activity_history[i];
    }
    avg_rate /= (T*population->n_neurons);

    cout << "Transducer::get_past() : returning rate from t="<<time-T/2<<"(index "<<time_idx_1 << ")";
    cout << "to t=" << time-T/2<<"(index "<<time_idx_1 << ")"; 
    cout << "\n\tavg_rate is "<< avg_rate;
    cout << endl;
    // return monitor->get_history()[time_idx] * (1 - theta) + monitor->get_history()[time_idx + 1]* theta;
    return avg_rate;
}

MultiscaleNetwork::MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet)
    :   spikenet(spikenet),
        oscnet(oscnet)
{
   n_populations = spikenet->populations.size();
   n_oscillators = oscnet->oscillators.size();
}


// void MultiscaleNetwork::pop_to_osc(unsigned int i, unsigned int j){
//     // Makes the population pop an input of oscillator osc
//     Transducer new_transducer(spikenet->populations[i], params);
//     oscnet->oscillators[j]->incoming_osc.push_back();
// }

// void MultiscaleNetwork::build_connections(Projection * projection){
//     // First does a quick check that no intra-scale connections are given
//     // For convention lets say that a multiscale projection matrix is given by
//     //  an (N+M)x(N+M) matrix in which the first N indexes are related to 
//     // the spiking part and the last M indexes are relate to the oscillator part
//     for (unsigned int i = 0; i < n_populations; i++){
//         for (unsigned int j = 0; j < n_populations; j++){
//             if (projection->weights[i][j] != 0.0 ){
//                 throw runtime_error("Connections of a multiscale network must not have intra-scale weights");
//             }
//         }
//     }
//     for (unsigned int i = n_populations; i < n_populations + n_oscillators; i++){
//         for (unsigned int j = n_populations + n_oscillators; j < n_populations; j++){
//             if (projection->weights[i][j] != 0.0 ){
//                 throw runtime_error("Connections of a multiscale network must not have intra-scale weights");
//             }
//         }
//     }

//     // Now creates the transducers
//     for (unsigned int i = n_populations; i < n_populations; i++){
//         for (unsigned int j = n_populations + n_oscillators; j < n_populations; j++){
//             if ( (projection->weights[i][j] != 0.0 ) | (projection->weights[j][i] != 0.0 )){
//                 // If there is a nonzero element between i-j or j-i the two dynamical objects communicate
//                 transducers.emplace_back(spikenet->populations[i]);
//             }
//             if ( projection->weights[i][j] != 0.0){

//             }
//         }
//     }

// }

void MultiscaleNetwork::run(EvolutionContext * evo, double time, int verbosity){

}