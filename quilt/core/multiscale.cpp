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
    std::function<double(double)> bound_rate_function = [this](double now){ return this->neural_mass_rate(now); };
    float weight, weight_delta;

    weight = params->get<float>("weight");
    weight_delta = params->get<float>("weight_delta");

    injector = new InhomPoissonSpikeSource(population, bound_rate_function, 
                                        weight, weight_delta, 
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

double Transducer::neural_mass_rate(double now){
    //  THIS METHOD MUST RETURN ms^-1 !!!!
    
    double rate = 0;    // Remember that this is a weighted sum
    double single_input_rate = 0;

    stringstream ss;
    ss << "Transducer: incoming rates are: "<< endl;
    for (auto input : incoming_osc){
        try {
        single_input_rate = input->get_rate(now); // This is a rate since it is "sigmed"
        ss << single_input_rate << ", ";
        }
        catch (negative_time_exception & e){
            // If this error is raised the link tried to get a non existing past
            get_global_logger().log(DEBUG, "transducer using burn-in value for incoming oscillator rate: " + to_string(initialization_rate) + " Hz");
            return initialization_rate;
        }

        rate += single_input_rate;
    }
    ss<<endl;
    ss << "Transducer::incoming_rate returning " << rate;
    get_global_logger().log(DEBUG, ss.str());
    
    history.push_back(rate);

    return rate;
}


double Transducer::spiking_pop_rate(double time)
{
    // I want to get the avg rate of the pop in [t-T/2, t+T/2]
    EvolutionContext * oscnet_evo = multinet->oscnet->get_evolution_context();
    double T = oscnet_evo->dt;

    EvolutionContext * spikenet_evo = multinet->spikenet->get_evolution_context();
    int time_idx_1 = spikenet_evo->index_of(time - 0.5*T);
    int time_idx_2 = spikenet_evo->index_of(time + 0.5*T);

    vector<int> activity_history = monitor->get_history();
    double avg_rate = 0.0;

    // Compute the average rate
    for (int i = time_idx_1; i < time_idx_2; i++){
        avg_rate += activity_history[i];
    }

    // This conversion is awful to see (issue 36)
    avg_rate /= (T*population->n_neurons); // ms^(-1)
    // avg_rate *= 1000; // Hz

    stringstream ss;
    ss << "Transducer::get_past() returning " << avg_rate << " ms^-1" << endl;
    get_global_logger().log(DEBUG, ss.str());


    return avg_rate;
}

MultiscaleNetwork::MultiscaleNetwork(SpikingNetwork * spikenet, OscillatorNetwork * oscnet)
    :   spikenet(spikenet),
        oscnet(oscnet),
        timescales_initialized(false)
{
    n_populations = spikenet->populations.size();
    n_oscillators = oscnet->oscillators.size();
    stringstream ss;
    ss << "MultiscaleNetwork has " << n_populations << " populations and " << n_oscillators << " oscillators";

    perf_mgr = std::make_shared<PerformanceManager>("multiscale network");
    perf_mgr->set_tasks({"evolve_spikenet", "evolve_oscnet"});
    PerformanceRegistrar::get_instance().add_manager(perf_mgr);

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
 * @brief Builds the projection between oscillators and transducers.
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

    // Checks for disconnected transducers
    vector<bool> transducer_has_inputs(transducers.size(), false);
    vector<bool> transducer_has_outputs(transducers.size(), false);

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
                
                transducer_has_outputs[i] = true;
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

                transducer_has_inputs[j] = true;
            }
        }
    }

    // Counts disconnected
    // No inputs
    int count = 0;
    for (unsigned int i = 0; i < transducers.size(); i++){
        if (!transducer_has_inputs[i]) count ++;
    }
    if (count > 0){
        stringstream ss;
        ss << "MultiscaleNetwork: " << count << " transducers were found to have no inputs";
        get_global_logger().log(WARNING, ss.str());
    }
    // No outputs
    count = 0;
    for (unsigned int i = 0; i < transducers.size(); i++){
        if (!transducer_has_outputs[i]) count ++;
    }
    if (count > 0){
        stringstream ss;
        ss << "MultiscaleNetwork: " << count << " transducers were found to have no outputs";
        get_global_logger().log(WARNING, ss.str());
    }

    logmsg << "Multiscale connections done "<< "(added "<< new_connections<< " links)";
    log.log(INFO, logmsg.str());
    return;
}

void MultiscaleNetwork::add_transducer(Population * population, ParaMap * params){
    transducers.push_back(new Transducer(population, params, this));
}

void MultiscaleNetwork::run(double time, int verbosity){
    if (!timescales_initialized){
        throw runtime_error("Evolution contextes for the slow and fast timescale must be initialized.");
    }

    std::stringstream ss;

    ss  << "Multiscale network running ( now time ctxts are: " 
        << "["<< evo_long->now <<"|" <<  evo_long->dt <<"]" 
        << " and " << "["<< evo_long->now <<"|" <<  evo_short->dt <<"]"
        << " )";

    get_global_logger().log(INFO, ss.str());
    
    while (evo_long -> now < time){

        // Evolve the short timescale until it catches up with 
        // the long timescale
        perf_mgr->start_recording("evolve_spikenet");
        while (evo_short->now < evo_long->now){
            
            spikenet->evolve();
        }
        perf_mgr->end_recording("evolve_spikenet");

        perf_mgr->start_recording("evolve_oscnet");
        oscnet->evolve();
        perf_mgr->end_recording("evolve_oscnet");
    }
    PerformanceRegistrar::get_instance().print_records();
}


/******************************************** MULTISCALE LINK MODELS *******************************************8*/

double T2JRLink::get_rate(double now){
    // This function is called by Oscillator objects linked to this transducer
    // during their evolution function

    // Returns the activity of the spiking population back in the past
    // Note that the average on the large time scale is done by Transducer::get_past()
    double result =  static_cast<Transducer*>(source)->spiking_pop_rate(now - delay);
    result *= weight;
    return result;
}

double JR2TLink::get_rate(double now){

    // Returns the rate of the oscillator back in the past 
    jansen_rit_oscillator * casted = static_cast<jansen_rit_oscillator*>(source);
    double v_p =  casted->get_past(1, now - delay)-casted->get_past(2, now - delay);
    double rate = casted->sigm(v_p);
    double result = weight * rate;

    return result;
}

double T2NJRLink::get_rate(double now){
    // This function is called by Oscillator objects linked to this transducer
    // during their evolution function

    // Returns the activity of the spiking population back in the past
    // Note that the average on the large time scale is done by Transducer::get_past()
    // Note 2: the negative rates are interpreted as inhibitory inputs. 
    double result =  static_cast<Transducer*>(source)->spiking_pop_rate(now - delay);
    result *= weight; //returns the same type of Transducer::get_past() : should be ms^-1
    return result;
}

double NJR2TLink::get_rate(double now){


    // Returns the rate of the oscillator back in the past 
    noisy_jansen_rit_oscillator * casted = static_cast<noisy_jansen_rit_oscillator*>(source);
    double v_p =  casted->get_past(1, now - delay)-casted->get_past(2, now - delay);
    double rate = casted->sigm(v_p);
    double result = weight * rate;


    return result;
}

double T2BNJRLink::get_rate(double now){
    // This function is called by Oscillator objects linked to this transducer
    // during their evolution function

    // Returns the activity of the spiking population back in the past
    // Note that the average on the large time scale is done by Transducer::get_past()
    // Note 2: the negative rates are interpreted as inhibitory inputs. 
    double result =  static_cast<Transducer*>(source)->spiking_pop_rate(now - delay);
    stringstream ss;
    ss << "T2BNJRLink::get_rate() is returning " << result << " ms^-1" << endl;
    get_global_logger().log(DEBUG, ss.str());
    return result;
}

double BNJR2TLink::get_rate( double now){

    // Returns the rate of the oscillator back in the past 
    double v_p =  source->get_past(1, now - delay)-source->get_past(2, now - delay);
    double rate = static_cast<binoisy_jansen_rit_oscillator*>(source)->sigm(v_p);
    double result = weight * rate;
    

    return result;
}
/**************************** SIOF workaround *******************/

template <typename LinkType>
void register_link(const std::string& from, const std::string& to) {
    get_link_factory().add_linker(std::make_pair(from, to), link_maker<LinkType>);
}

struct LinkRegistrar {
    LinkRegistrar() {
        register_link<JR2TLink>("jansen-rit", "transducer");
        register_link<T2JRLink>("transducer", "jansen-rit");
        register_link<NJR2TLink>("noisy-jansen-rit", "transducer");
        register_link<T2NJRLink>("transducer", "noisy-jansen-rit");
        register_link<BNJR2TLink>("binoisy-jansen-rit", "transducer");
        register_link<T2BNJRLink>("transducer", "binoisy-jansen-rit");
    }
} link_registrar;