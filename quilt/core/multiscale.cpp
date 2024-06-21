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

    // Builds the injector
    std::function<double(double)> bound_rate_function = [this](double now){ return this->incoming_rate(now); };
    injector = new InhomPoissonSpikeSource(population, bound_rate_function, 10, 0.5, 
                                        static_cast<int>(params->get<float>("generation_window")) 
                                        );

    // Adds the injector to the list of injectors of the spiking network
    population->spiking_network->add_injector(injector);

    // Sets the rate for negative times (Adds the injectorinitialization)
    initialization_rate = static_cast<double>(params->get<float>("initialization_rate")); 
    get_global_logger().log(INFO, "Transducer initialization rate is " + to_string(initialization_rate));

    evolve_state = [this](const dynamical_state & /*x*/, dynamical_state & /*dxdt*/, double /*t*/)
    {
        throw runtime_error("Calling 'evolve_state' of a transducer object is not allowed");
    };

    eeg_voi = [](const dynamical_state & /*x*/){ 
        throw runtime_error("Calling 'eeg_voi' of a transducer object is not allowed");
        return 0.0;
    };

    cout << "\ttransducer created"<<endl;
}

Transducer::~Transducer(){
    delete injector;
}

ThreadSafeFile Transducer::outfile("td_incoming_rates.txt");

double Transducer::incoming_rate(double now){

    double rate = 0;    // Remember that this is a weighted sum
    double single_input_rate = 0;

    for (auto input : incoming_osc){
        try {
        single_input_rate = input->get(0, now);
        }
        catch (negative_time_exception & e){
        // If this error is raised the link tried to get a non existing past
            get_global_logger().log(DEBUG, "transducer using burn-in value for incoming oscillator rate: " + to_string(initialization_rate) + " Hz");
            return initialization_rate;
        }
        rate += single_input_rate;
        get_global_logger().log(DEBUG, "single input to transducer is " + to_string(single_input_rate));
    }
    get_global_logger().log(DEBUG, "total input to transducer is " + to_string(rate) + " Hz");
    
    stringstream ss;
    ss << now << " " << rate;
    outfile.write(ss.str());
    
    return rate;
}


double Transducer::get_past(unsigned int /*axis*/, double time)
{
    stringstream ss;
    ss << "Getting past from transducer: time requested = " << time; 
    get_global_logger().log(DEBUG, ss.str());

    // I want to get the avg rate of the pop in [t-T/2, t+T/2]
    EvolutionContext * oscnet_evo = multinet->oscnet->get_evolution_context();
    double T = oscnet_evo->dt;

    EvolutionContext * spikenet_evo = multinet->spikenet->get_evolution_context();
    int time_idx_1 = spikenet_evo->index_of(time - T/2);
    int time_idx_2 = spikenet_evo->index_of(time + T/2);

    cout << "\tdt = " << spikenet_evo->dt << ", dT = " << oscnet_evo->dt <<endl;
    cout << "time indexes: [" << time_idx_1 << "," << time_idx_2 << "]" << endl;
    
    // double theta = evo->deviation_of(time); //TODO: make this not useless

    vector<int> activity_history = monitor->get_history();
    double avg_rate = 0.0;

    // Compute the average rate
    for (int i = time_idx_1; i < time_idx_2; i++){
        avg_rate += activity_history[i];
    }
    avg_rate /= (T*population->n_neurons); // ms^(-1)
    avg_rate *= 1000; // Hz


    ss.str(""); ss.clear();
    ss << "Transducer::get_past() : returning rate from t="<<time-T/2<<"(index "<<time_idx_1 << ")";
    ss << " to t=" << time+T/2<<"(index "<<time_idx_2 << ")"; 
    ss << ": avg_rate is "<< avg_rate << "Hz";

    get_global_logger().log(DEBUG, ss.str());
    // return monitor->get_history()[time_idx] * (1 - theta) + monitor->get_history()[time_idx + 1]* theta;
    return avg_rate;
}

MultiscaleNetwork::MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet)
    :   spikenet(spikenet),
        oscnet(oscnet),
        timescales_initialized(false)
{
   n_populations = spikenet->populations.size();
   n_oscillators = oscnet->oscillators.size();
   cout << "MultiscaleNetwork has " << n_populations << " populations and " << n_oscillators << " oscillators."<<endl;
}

void MultiscaleNetwork::set_evolution_contextes(EvolutionContext * evo_short, EvolutionContext * evo_long){
    // First approx: integer number of steps
    time_ratio = static_cast<int> (floor( evo_long->dt/evo_short->dt + 0.5));
    cout << "Multiscale time ratio is " << time_ratio << endl;
    if (time_ratio == 0){
        throw runtime_error("Invalid timescale ratio: short scale must be shorter than the long timescale -- :o");
    }

    this->evo_short = evo_short;
    this->evo_long = evo_long;

    spikenet->set_evolution_context(evo_short);
    oscnet->set_evolution_context(evo_long);

    timescales_initialized = true;
}

/**
 * Builds projection between oscillators and transducers.
 * 
 * 
*/
void MultiscaleNetwork::build_OT_projections(Projection * projT2O, Projection * projO2T){
    
    // For now: link params is just nothing
    // In future I have to find a way to express variability in this stuff
    // Probably I must step onto matrices of ParaMaps
    ParaMap * link_params = new ParaMap();
    std::stringstream logmsg;
    Logger &log = get_global_logger();

    // Dimension check 1
    if( (projT2O->start_dimension != transducers.size())|((projT2O->end_dimension != n_oscillators))){
        string msg = "Projection shape mismatch while building T->O projections.\n";
        msg += "N_transd = " + std::to_string(transducers.size()) +  " N_oscill = " + std::to_string(n_oscillators) + "\n";
        msg += "Projection shape: (" + std::to_string(projT2O->start_dimension) + "," +std::to_string(projT2O->end_dimension) + ")\n";
        throw std::invalid_argument(msg);
    }

    // Dimension check 2
    if( (projO2T->start_dimension != n_oscillators)|((projO2T->end_dimension != transducers.size()))){
        string msg = "Projection shape mismatch while building O->T projections.\n";
        msg += "N_oscill = " + std::to_string(n_oscillators) + " N_transd = " + std::to_string(transducers.size())  +"\n";
        msg += "Projection shape: (" + std::to_string(projO2T->start_dimension) + "," +std::to_string(projO2T->end_dimension) + ")\n";
        throw std::invalid_argument(msg);
    }

    logmsg << "Performing connections between "<< transducers.size() << " transducers and " << n_oscillators << " oscillators";
    log.log(INFO, logmsg.str());

    logmsg.str(""); logmsg.clear();

    // T->O
    for (unsigned int i = 0; i < transducers.size(); i++){
        for (unsigned int j= 0; j < n_oscillators ; j++){

            if (projT2O->weights[i][j] != 0)
            {   
                oscnet->oscillators[j]->incoming_osc.push_back(get_link_factory().get_link(transducers[i], oscnet->oscillators[j], 
                                                                                            projT2O->weights[i][j], projT2O->delays[i][j], 
                                                                                            link_params)); // Remember: link params is none here
            }else{
                // cout << "\t\tweigth is zero"<<endl;
            }
        }
    }

    // O->T
    for (unsigned int i = 0; i < n_oscillators; i++){
        for (unsigned int j= 0; j < transducers.size() ; j++){

            if (projO2T->weights[i][j] != 0)
            {   
                transducers[j]->incoming_osc.push_back(get_link_factory().get_link( oscnet->oscillators[i], transducers[j],
                                                                                            projO2T->weights[i][j], projO2T->delays[i][j], 
                                                                                            link_params)); // Remember: link params is none here
            }else{
                // cout << "\t\tweigth is zero"<<endl;
            }
        }
    }
    logmsg << "Multiscale connections done";
    log.log(INFO, logmsg.str());
    return;
}

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

void MultiscaleNetwork::run(double time, int verbosity){
    if (!timescales_initialized){
        throw runtime_error("Evolution contextes for the slow and fast timescale must be initialized.");
    }

    std::stringstream ss;

    ss  << "Multiscale network running ( time ctxts: " 
        << "["<< evo_long->now <<"|" <<  evo_long->dt <<"]" 
        << " and " << "["<< evo_long->now <<"|" <<  evo_short->dt <<"]"
        << " )";

    get_global_logger().log(INFO, ss.str());

    int n_steps_total = static_cast<int>(time / evo_long->dt) ;
    
    progress bar(n_steps_total, verbosity); 
    while (evo_long -> now < time){

        // Evolve the short timescale until it catches up with 
        // the long timescale
        // cout << "Doing one big step" << endl;
        while (evo_short->now < evo_long->now){
            // cout << "Doing one small step"<<endl;
            spikenet->evolve();
        }

        oscnet->evolve();
        ++bar;
    }

}