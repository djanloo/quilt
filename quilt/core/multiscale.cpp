#include "include/multiscale.hpp"



Transducer::Transducer(Population * population, const ParaMap * params)
    :   population(population)
{
    // Adds the monitor
    monitor = population->spiking_network->add_spike_monitor(population);

    // Adds the injector
    injector = new PoissonSpikeSource(population, 10, 0.5, 0.2, -1, 0 );
    population->spiking_network->add_injector(injector);
}

void Transducer::evolve()
{
    // Adds to history the mean rate of the population during the last big step

}

double Transducer::get_past(unsigned int axis, double time)
{
    int time_idx = evo->index_of(time);
    double theta = evo->deviation_of(time);

    return state_history[time_idx][axis] * (1 - theta) + state_history[time_idx + 1][axis] * theta;
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