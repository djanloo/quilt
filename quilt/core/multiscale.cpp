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
        timescales_initialized(false),
        perf_mgr({"evolve_spikenet", "evolve_oscnet"})
{
   n_populations = spikenet->populations.size();
   n_oscillators = oscnet->oscillators.size();
   stringstream ss;
   ss << "MultiscaleNetwork has " << n_populations << " populations and " << n_oscillators << " oscillators";

   perf_mgr.set_label("multiscale network");
   get_global_logger().log(INFO, ss.str());
}

void MultiscaleNetwork::set_evolution_contextes(EvolutionContext * evo_short, EvolutionContext * evo_long){
    // First approx: integer number of steps
    time_ratio = static_cast<int> (floor( evo_long->dt/evo_short->dt + 0.5));

    stringstream ss;
    ss << "Multiscale time ratio is " << time_ratio;
    get_global_logger().log(INFO, ss.str());

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
void MultiscaleNetwork::build_multiscale_projections(Projection * projT2O, Projection * projO2T){
    
    // Initialization check
    // The delays between transducers and oscillators may increase the initialization time
    // so the oscillator network must not be initialized befor building the multiscale connections
    if (oscnet->is_initialized){
        string msg = "OscillatorNetwork was initialized before building multiscale connections";
        get_global_logger().log(ERROR, msg);
        throw runtime_error(msg); 
    }

    // For now: link params is just nothing
    // In future I have to find a way to express variability in this stuff
    // Probably I must step onto matrices of ParaMaps
    ParaMap * link_params = new ParaMap();
    std::stringstream logmsg;
    Logger &log = get_global_logger();

    // Dimension check 1
    if( (projT2O->start_dimension != transducers.size())|((projT2O->end_dimension != n_oscillators))){
        string msg = "Projection shape mismatch while building T->O projections.\n";
        msg += "n_transducers = " + std::to_string(transducers.size()) +  " n_oscillators = " + std::to_string(n_oscillators) + "\n";
        msg += "Projection shape: (" + std::to_string(projT2O->start_dimension) + "," +std::to_string(projT2O->end_dimension) + ")\n";
        throw std::invalid_argument(msg);
    }

    // Dimension check 2
    if( (projO2T->start_dimension != n_oscillators)|((projO2T->end_dimension != transducers.size()))){
        string msg = "Projection shape mismatch while building O->T projections.\n";
        msg += "n_oscillators = " + std::to_string(n_oscillators) + " n_transducers = " + std::to_string(transducers.size())  +"\n";
        msg += "Projection shape: (" + std::to_string(projO2T->start_dimension) + "," +std::to_string(projO2T->end_dimension) + ")\n";
        throw std::invalid_argument(msg);
    }

    logmsg << "Performing connections between "<< transducers.size() << " transducers and " << n_oscillators << " oscillators";
    log.log(INFO, logmsg.str());

    logmsg.str(""); logmsg.clear();

    // T->O
    int new_connections = 0;
    for (unsigned int i = 0; i < transducers.size(); i++){
        for (unsigned int j= 0; j < n_oscillators ; j++){

            if (std::abs(projT2O->weights[i][j]) > WEIGHT_EPS)
            {   
                oscnet->oscillators[j]->incoming_osc.push_back(get_link_factory().get_link(transducers[i], oscnet->oscillators[j], 
                                                                                            projT2O->weights[i][j], projT2O->delays[i][j], 
                                                                                            link_params)); // Remember: link params is none here
                new_connections++;

                // Takes trace of minimum delay
                if (projT2O->delays[i][j] < oscnet->min_delay) oscnet->min_delay = projT2O->delays[i][j];
                // Takes trace of maximum delay
                if (projT2O->delays[i][j] > oscnet->max_delay) oscnet->max_delay = projT2O->delays[i][j];

            }else{
                // cout << "\t\tweigth is zero"<<endl;
            }
        }
    }

    // O->T
    for (unsigned int i = 0; i < n_oscillators; i++){
        for (unsigned int j= 0; j < transducers.size() ; j++){

            if ( std::abs(projO2T->weights[i][j]) > WEIGHT_EPS)
            {   
                transducers[j]->incoming_osc.push_back(get_link_factory().get_link( oscnet->oscillators[i], transducers[j],
                                                                                            projO2T->weights[i][j], projO2T->delays[i][j], 
                                                                                            link_params)); // Remember: link params is none here
                new_connections++;

                // Takes trace of minimum delay
                if (projO2T->delays[i][j] < oscnet->min_delay) oscnet->min_delay = projO2T->delays[i][j];
                // Takes trace of maximum delay
                if (projO2T->delays[i][j] > oscnet->max_delay) oscnet->max_delay = projO2T->delays[i][j];

            }else{
                // cout << "\t\tweigth is zero"<<endl;
            }
        }
    }
    logmsg << "Multiscale connections done "<< "(added "<< new_connections<< " links)";
    log.log(INFO, logmsg.str());
    return;
}

void MultiscaleNetwork::add_transducer(Population * population, ParaMap * params){
    transducers.push_back(make_shared<Transducer>(population, params, this));
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
        perf_mgr.start_recording("evolve_spikenet");
        while (evo_short->now < evo_long->now){
            // cout << "Doing one small step"<<endl;
            spikenet->evolve();
        }
        perf_mgr.end_recording("evolve_spikenet");

        perf_mgr.start_recording("evolve_oscnet");
        oscnet->evolve();
        perf_mgr.end_recording("evolve_oscnet");
        ++bar;
    }

}


/******************************************** MULTISCALE LINK MODELS *******************************************8*/

double T2JRLink::get(int axis, double now){
    // This function is called by Oscillator objects linked to this transducer
    // during their evolution function
    get_global_logger().log(DEBUG, "T2JRLink: getting t=" + to_string(now-delay) );

    // Returns the activity of the spiking population back in the past
    // Note that the average on the large time scale is done by Transducer::get_past()
    double result = weight * std::static_pointer_cast<Transducer>(source)->get_past(axis, now - delay); //axis is useless
    return result;
}

double JR2TLink::get(int axis, double now){

    if (axis != 0) throw runtime_error("Jansen-Rit model can only ask for axis 0 (pyramidal neurons)");

    // Returns the rate of the oscillator back in the past 
    double v0 =  source->get_past(axis, now - delay);
    double rate = std::static_pointer_cast<jansen_rit_oscillator>(source)->sigm(v0);
    double result = weight * rate;

    // cout << "Getting past from JRJR link" << endl;
    // cout << "JRJR got "<<result<< endl;

    //NOTE: Jansen-Rit Model is in ms^-1. Result must be converted.
    result *= 1e3;

    std::stringstream ss;
    ss << "JR2TLink:" << "now is t=" << now << " and getting " << now-delay << ":\n v0 = " << v0 << " mV, rate = " << rate << " ms^-1, weight = " << weight << " (returning " << result << " Hz)\n";
    get_global_logger().log(DEBUG, ss.str());

    return result;
}

/**************************** SIOF workaround *******************/
/**
 * Registers the Jansen-Rit to Transducer link in the link factory
 */
struct JR2TTypeRegistrar {
    JR2TTypeRegistrar() {
        get_link_factory().add_linker(std::make_pair("jansen-rit", "transducer"), link_maker<JR2TLink>);
    }
} jr2t_registrar;


/**
 * Registers the Transducer to Jansen-Rit link in the link factory
 */
struct T2JRTypeRegistrar {
    T2JRTypeRegistrar() {
        get_link_factory().add_linker(std::make_pair("transducer", "jansen-rit"), link_maker<T2JRLink>);
    }
} t2jr_registrar;