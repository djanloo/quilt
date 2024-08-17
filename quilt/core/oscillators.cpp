#include "include/links.hpp"
#include "include/oscillators.hpp"

#include <limits>
/******************************************* OSCILLATORS BASE **********************************/
Oscillator::Oscillator(ParaMap * params, OscillatorNetwork * oscnet)
    :   params(params),
        oscnet(oscnet),
        memory_integrator()
{
    id = HierarchicalID(&(oscnet->id));
    evolve_state = [](const dynamical_state & /*x*/, dynamical_state & /*dxdt*/, double /*t*/){cout << "Warning: using virtual evolve_state of Oscillator" << endl;};
}

void Oscillator::set_evolution_context(EvolutionContext * evo)
{
    get_global_logger().log(DEBUG, "set EvolutionContext of Oscillator");
    this->evo = evo;
    memory_integrator.set_evolution_context(evo);
    for (auto & incoming_link : incoming_osc)
    {
        incoming_link->set_evolution_context(evo);
    }
};

void Oscillator::print_info(){
    stringstream ss;
    ss << "Printing info for oscillator:" << endl;
    ss << "Oscillator type: "<< oscillator_type << endl;
    cout << "Paramap at index " << params << endl;
    ss << "Parameters: "<< *(params); 
    cout << ss.str()<<endl;
}

// Homogeneous network builder
OscillatorNetwork::OscillatorNetwork(int N, ParaMap * params)
    :   perf_mgr("oscillator network"),
        max_delay(0.0),
        min_delay(std::numeric_limits<float>::max())
{    
    // Bureaucracy
    id = HierarchicalID();

    string oscillator_type = params->get<string>("oscillator_type");

    for (int i = 0; i < N; i++){
        oscillators.push_back(get_oscillator_factory().get_oscillator(oscillator_type, params, this));
    }
    has_oscillators = true;

    perf_mgr.set_tasks({"evolution"});
    perf_mgr.set_scales({{"evolution", N}});

    get_global_logger().log(INFO, "Built HOMOGENEOUS OscillatorNetwork");
}

// Homogeneous network builder
OscillatorNetwork::OscillatorNetwork(vector<ParaMap *> params)
    :   perf_mgr("oscillator network"),
        max_delay(0.0),
        min_delay(std::numeric_limits<float>::max())
{   
    // Bureaucracy
    id = HierarchicalID();
    string oscillator_type;

    for (unsigned int i = 0; i < params.size() ; i++){
        oscillator_type = params[i]->get<string>("oscillator_type");
        oscillators.push_back(get_oscillator_factory().get_oscillator(oscillator_type, params[i], this));
    }
    has_oscillators = true;

    perf_mgr.set_tasks({"evolution"});
    perf_mgr.set_scales({{"evolution", params.size()}});

    get_global_logger().log(INFO, "Built NON-HOMOGENEOUS OscillatorNetwork");

}

void OscillatorNetwork::build_connections(Projection * proj, ParaMap * link_params)
{
    get_global_logger().log(DEBUG, "Starting to build links in OscillatorNetwork");
    if (!has_oscillators){
        get_global_logger().log(ERROR,"Could not link oscillators since the network does not have oscillators yet.");
        throw runtime_error("Could not link oscillators since the network does not have oscillators yet.");
    }

    if (proj->start_dimension != proj->end_dimension)
    {
        get_global_logger().log(ERROR,"Projection matrix of OscillatorNetwork must be a square matrix");
        throw std::invalid_argument("Projection matrix of OscillatorNetwork must be a square matrix");
    }

    if (proj->start_dimension != oscillators.size())
    {
        string msg = "Shape mismatch in projection matrix: size is "\
                    + std::to_string(proj->end_dimension) + "x" + std::to_string(proj->end_dimension) \
                    + " but network has n_oscillators = " + std::to_string(oscillators.size());
        get_global_logger().log(ERROR,msg);
        throw std::invalid_argument(msg);
    }

    // Checks whether a node is uselesss
    vector<bool> has_outputs(proj->start_dimension, true);
    vector<bool> has_inputs(proj->end_dimension, true);

    for (unsigned int i =0; i < proj->start_dimension; i++)
    {
        for (unsigned int j = 0; j < proj->end_dimension; j++)
        {
            if (std::abs(proj->weights[i][j]) > WEIGHT_EPS)
            {                   
                oscillators[j]->incoming_osc.push_back(get_link_factory().get_link(oscillators[i], oscillators[j], proj->weights[i][j], proj->delays[i][j], link_params));
                
                // Takes trace of minimum delay
                if (proj->delays[i][j] < min_delay) min_delay = proj->delays[i][j];
                // Takes trace of maximum delay
                if (proj->delays[i][j] > max_delay) max_delay = proj->delays[i][j];

                has_outputs[i] = false;
                has_inputs[j] = false;
            }
        }
    }
    has_links = true;

    // Counts the disconnected nodes (no outputs)
    int count = 0;
    for ( unsigned int i=0; i< proj->start_dimension; i++){
        if (!has_outputs[i]) count++;
    }
    if (count > 0){
        stringstream ss;
        ss << "OscillatorNetwork: " << count << " nodes were found to have no outputs";
        get_global_logger().log(WARNING, ss.str());
    }

    // Counts the disconnected nodes (no inputs)
    count=0;
    for ( unsigned int i=0; i< proj->end_dimension; i++){
        if (!has_inputs[i]) count++;
    }
    if (count > 0){
        stringstream ss;
        ss << "OscillatorNetwork: " << count << " nodes were found to have no inputs";
        get_global_logger().log(WARNING, ss.str());
    }

    get_global_logger().log(INFO, "Built links in OscillatorNetwork");
}

void OscillatorNetwork::initialize(EvolutionContext * evo, vector<dynamical_state> init_conds)
{
    get_global_logger().log(DEBUG, "Starting initialization of OscillatorNetwork");

    if (init_conds.size() != oscillators.size()){
        get_global_logger().log(ERROR, "Number of initial conditions is not equal to number of oscillators");
        throw runtime_error("Number of initial conditions is not equal to number of oscillators");
    }

    if (!has_links){
        get_global_logger().log(WARNING, "Initializing an OscillatorNetwork without links");
    }
    
    // Adds a timestep to max_delay (this is useful for multiscale)
    // because Transuducers need an half-big-timestep in the past
    // to average the spiking network activity
    max_delay += evo->dt;

    // Guard for bug #24: at least two timesteps of delay are necessary
    if ( min_delay < 2*evo->dt){
        stringstream logmsg;
        logmsg << "During initialization of OscillatorNetwork:"<< endl
                << "minimum delay is too short, at least two timesteps of delay are necessary" << endl
                << "(now min_tau=" << min_delay << " ms, dt=" << evo->dt <<")"<< endl
                << "Delays are considered 2 timestep"<<endl
                << "See https://github.com/djanloo/quilt/issues/24";
        get_global_logger().log(WARNING, logmsg.str());

        int n_rounded = 0;
        int n_links = 0;
        for (auto osc : oscillators){
            for (auto l : osc->incoming_osc){
                if (l->delay < 2*evo->dt){
                    l->delay = 2*evo->dt;
                    n_rounded ++;
                }
                n_links++;
            }
        }
        logmsg.str(""); logmsg.clear();
        logmsg << "Rounded "<< n_rounded << " delays over "<< n_links <<" to " << 2*evo->dt << " ms instead of 0 ms";
        get_global_logger().log(WARNING, logmsg.str());

    }
    
    int n_init_pts = static_cast<int>(std::ceil(max_delay/evo->dt) + 1);

    for (unsigned int i = 0; i < init_conds.size(); i++ ){
        vector<dynamical_state> new_K(4, vector<double>(oscillators[i]->get_space_dimension()));

        // Adds n_init_points values for the state X
        for (int n = 0; n < n_init_pts; n++){
            //Checks that the initialization is right in dimension
            if (init_conds[i].size() != oscillators[i]->space_dimension){
                get_global_logger().log(ERROR, "Initial conditions have not the right space dimension");
                throw runtime_error("Initial conditions have not the right space dimension");
            }
            
            // Adds initial condition as value of X 
            oscillators[i]->memory_integrator.state_history.push_back(init_conds[i]);
        }

        /*

        Adds n_init_points-1 values for the evaluation history K
        This is because given a starting point you compute the K_n
        so when the network actually runs it computes it first real K

        INIT
        ----
        | X: *
        |
        | K: 

        FIRST STEP
        ----------
        | X: *          | X: * *
        |        -->    |
        | K: *          | K: *

        SECOND STEP
        ----------
        | X: * *            | X: * * *
        |          -->      |
        | K: * *            | K: **

        NOTE: this section is the one that originated BUG #18

        */

        for (int n = 0; n < n_init_pts - 1; n++){

            for (int nu = 0; nu < 4; nu++){
                for (unsigned int dim = 0; dim < oscillators[i]->get_space_dimension(); dim++){
                    new_K[nu][dim] = 0.0;
                }
            }
            oscillators[i]->memory_integrator.evaluation_history.push_back(new_K);
        }
    }

    // Set the current time to max_tau + dT
    // This makes the initialization to be in [0, T+dT] instead of [-T, 0]
    for (int n = 0; n < n_init_pts - 1; n++){
        evo->do_step();
    }

    is_initialized = true;
    Logger &log = get_global_logger();
    log.log(INFO, "Initialized past of OscillatorNetwork");
    stringstream ss;
    ss << "Oscillators initialized with " << oscillators[0]->memory_integrator.state_history.size() << " states"
        << " and " << oscillators[0]->memory_integrator.evaluation_history.size() << " evaluations ";
    ss << "since (max_delay + dt) = " << max_delay;
    log.log(INFO, ss.str());
}

void OscillatorNetwork::evolve(){
    if (!is_initialized){
        get_global_logger().log(ERROR,"The network must be initialized before evolving" );
        throw runtime_error("The network must be initialized before evolving");
    }

    Logger &log = get_global_logger();
    std::stringstream ss;

    ss << "Evolving OSCILLATOR network (t = "<<evo->now <<" -> "<< evo->now + evo->dt << ")";
    log.log(DEBUG, ss.str());

    perf_mgr.start_recording("evolution");

    // Gets the new values
    for (auto oscillator : oscillators){
        oscillator->memory_integrator.compute_next();
    }

    // Fixes them
    for (auto oscillator : oscillators){
        oscillator->memory_integrator.fix_next();
    }
    perf_mgr.end_recording("evolution");

    evo->do_step();
}

void OscillatorNetwork::run(EvolutionContext * evo, double time, int verbosity)
{
    if (!is_initialized){
        get_global_logger().log(ERROR, "The network must be initialized before running");
        throw runtime_error("The network must be initialized before running");
    }

    if (!has_links){
        get_global_logger().log(WARNING, "Running an OscillatorNetwork without links");
    }
    
    // Synchronizes every component
    set_evolution_context(evo);

    // Some verbose output
    double t0 = evo->now;
    int n_steps_total = static_cast<int>(time/evo->dt);

    stringstream ss;
    ss <<  "Running network consisting of " << oscillators.size() << " oscillators for " << n_steps_total <<" timesteps";
    get_global_logger().log(INFO, ss.str());

    // Evolve
    progress bar(n_steps_total, verbosity);
    perf_mgr.start_recording("evolution");

    while (evo->now < t0 + time){
            
        // Gets the new values
        for (auto oscillator : oscillators){
            oscillator->memory_integrator.compute_next();
        }

        // Fixes them
        for (auto oscillator : oscillators){
            oscillator->memory_integrator.fix_next();
        }
        evo->do_step();
        ++bar;
    }
    perf_mgr.end_recording("evolution");


    // Prints performance
    if (verbosity > 0){
        PerformanceRegistrar::get_instance().print_records();
    }
}

/****************************************** OSCILLATOR FACTORY ******************************************/
// Singleton method to return a unique instance of OscillatorFactory
OscillatorFactory& get_oscillator_factory(){
    static OscillatorFactory osc_factory;
    return osc_factory;
}
shared_ptr<Oscillator> OscillatorFactory::get_oscillator(string const& oscillator_type, ParaMap * params, OscillatorNetwork * osc_net)
        {
            auto it = _constructor_map.find(oscillator_type);
            if (it == _constructor_map.end()) { 
                get_global_logger().log(ERROR,"No constructor was found for oscillator " + oscillator_type );
                throw runtime_error("No constructor was found for oscillator " + oscillator_type); 
            }
            return (it->second)(params, osc_net);
        };

OscillatorFactory::OscillatorFactory(){
    // Here I place the builders of each type of oscillator
    add_constructor("base", oscillator_maker<Oscillator>);
    add_constructor("harmonic", oscillator_maker<harmonic_oscillator>);
    add_constructor("test", oscillator_maker<test_oscillator>);
    add_constructor("jansen-rit", oscillator_maker<jansen_rit_oscillator>);
    add_constructor("leon-jansen-rit", oscillator_maker<leon_jansen_rit_oscillator>);
}

// **************************************** OSCILLATOR MODELS ***************************************** //
harmonic_oscillator::harmonic_oscillator(ParaMap * params, OscillatorNetwork * oscnet)    
    :   Oscillator(params, oscnet)
{
    k = params->get<float>("k");
    oscillator_type = "harmonic";
    space_dimension = 2;

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){
        dxdt[0] = x[1];

        for (auto input : incoming_osc){
            dxdt[1] += x[0] - k*input->get(0, t) ;
        }  
    };

    // Sets a (dymmy) variable of interest for the EEG
    eeg_voi = [](const dynamical_state & x){return x[0];};

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

test_oscillator::test_oscillator(ParaMap * params, OscillatorNetwork * oscnet)
    :   Oscillator(params, oscnet)
{
    oscillator_type = "test";
    space_dimension = 6;

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double /*t*/){

        dxdt[0] = x[1];
        dxdt[1] = -x[0];

        dxdt[2] = x[3];
        dxdt[3] = -x[2];

        dxdt[4] = x[5];
        dxdt[5] = -x[4];
    };

    // Sets a (dymmy) variable of interest for the EEG
    eeg_voi = [](const dynamical_state & x){return x[0];};

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

// Auxiliary for Jansen-Rit
double jansen_rit_oscillator::sigm(double v)
{
    return rmax / (1.0 + std::exp(s*(v0-v)));
}

jansen_rit_oscillator::jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet) 
    :   Oscillator(params, oscnet)
{
    oscillator_type = "jansen-rit";
    space_dimension = 6;

    // Parameters default from references
    He = params->get("He", 3.25);       // mV
    Hi = params->get("Hi", 22.0);       // mV
    ke = params->get("ke", 0.1);        // ms^(-1)
    ki = params->get("ki", 0.05);       // ms^(-1)
    rmax = params->get("rmax", 0.005);  // ms^(-1)
    v0 = params->get("v0", 6.0);        // mV
    C = params->get("C", 135.0); 
    s = params->get("s", 0.56);         // mV^-1
    U = params->get("U", 0.13);         // ms^(-1)

    // The system of ODEs implementing the evolution equation 
    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t)
    {

        double external_currents = 0;
        for (auto input : incoming_osc)
        {
            external_currents += input->get(0, t);
        }
        double external_inputs = U + external_currents;

        // This is mostly a test
        input_history.push_back(external_inputs);

        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];

        dxdt[3] = He*ke*sigm( x[1] - x[2]) - 2*ke*x[3] - ke*ke*x[0];
        dxdt[4] = He*ke*( external_inputs + 0.8*C*sigm(C*x[0]) ) - 2*ke*x[4] - ke*ke*x[1];
        dxdt[5] = Hi*ki*0.25*C*sigm(0.25*C*x[0]) - 2*ki*x[5] - ki*ki*x[2];

    };

    // Sets the variable of interest for the EEG
    // In this case it corresponds to the pyramidal cells lumped potential
    eeg_voi = [](const dynamical_state & x){return x[1] - x[2];};

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

vector<double> jansen_rit_oscillator::get_rate_history(){
    vector<double> rate(memory_integrator.state_history.size(), 0);
    for (unsigned int i = 0; i < rate.size(); i++){
        rate[i] = sigm(memory_integrator.state_history[i][0]);
    }
    return rate;
}


/************************ LEON JANSEN RIT *************************************/

// Auxiliary for Leon-Jansen-Rit
// I know it's the same function I'm not dumb
// I just want blindly secure compatibility with parameter values
double leon_jansen_rit_oscillator::sigm(double v)
{
    return 2*e0 / (1.0 + std::exp(rho1*(rho2-v)));
}

// float leon_jansen_rit_oscillator::He = 3.25;
// float leon_jansen_rit_oscillator::Hi = 22.0;
// float leon_jansen_rit_oscillator::ke = 0.1;
// float leon_jansen_rit_oscillator::ki = 0.05;

// // Internal coupling parameters
// float leon_jansen_rit_oscillator::gamma_1 = 135;
// float leon_jansen_rit_oscillator::gamma_2 = 108;
// float leon_jansen_rit_oscillator::gamma_3 = 33.75;
// float leon_jansen_rit_oscillator::gamma_4 = 33.75;
// float leon_jansen_rit_oscillator::gamma_5 = 15;

// // External coupling parameters
// float leon_jansen_rit_oscillator::gamma_1T = 1;
// float leon_jansen_rit_oscillator::gamma_2T = 1;
// float leon_jansen_rit_oscillator::gamma_3T = 1;

// // Sigmoid parameters
// float leon_jansen_rit_oscillator::e0 =    0.0025;
// float leon_jansen_rit_oscillator::rho1 =  6;
// float leon_jansen_rit_oscillator::rho2 =  0.56;

// // Bifurcation parameters
// float leon_jansen_rit_oscillator::U = 0.12;
// float leon_jansen_rit_oscillator::P = 0.12;
// float leon_jansen_rit_oscillator::Q = 0.12;

leon_jansen_rit_oscillator::leon_jansen_rit_oscillator(ParaMap * params, OscillatorNetwork * oscnet) 
    :   Oscillator(params, oscnet)
{
    // Referencing (Leon, 2015) for this system of equations
    // The variable of interest for the EEG is v2-v3
    oscillator_type = "leon-jansen-rit";
    space_dimension = 12;

    // Parameters default from references

    // Delay box parameters
    He = params->get("He", 3.25f);
    Hi = params->get("Hi", 22.0f);
    ke = params->get("ke", 0.1f);   // ms^(-1)
    ki = params->get("ki", 0.05f);  // ms^(-1)

    // Internal coupling parameters
    gamma_1 = params->get("gamma_1", 135.0f);
    gamma_2 = params->get("gamma_2", 108.0f);
    gamma_3 = params->get("gamma_3", 33.75f);
    gamma_4 = params->get("gamma_4", 33.75f);
    gamma_5 = params->get("gamma_5", 15.0f);

    // External coupling parameters
    gamma_1T = params->get("gamma_1T", 1.0f);
    gamma_2T = params->get("gamma_2T", 1.0f);
    gamma_3T = params->get("gamma_3T", 1.0f);

    // Sigmoid parameters
    e0 = params->get("e0", 0.0025f); // ms^(-1)
    rho1 = params->get("rho1", 6.0f);
    rho2 = params->get("rho2", 0.56f);

    // Bifurcation parameters
    U = params->get("U", 0.12f); // ms^(-1)
    P = params->get("P", 0.12f); // ms^(-1)
    Q = params->get("Q", 0.12f); // ms^(-1)

    // The system of ODEs implementing the evolution equation 
    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t)
    {
        // x[0] is v1
        // x[1] is v2
        // x[2] is v3
        // x[3] is v4
        // x[4] is v5
        // x[5] is v6
        // x[6] is v7
        // -----------
        // x[7] is y1
        // x[8] is y2
        // x[9] is y3
        // x[10] is y4
        // x[11] is y5

        double external_currents = 0;

        for (auto input : incoming_osc)
        {
            external_currents += input->get(6, t);
        }

        // Vs
        dxdt[0] = x[7];
        dxdt[1] = x[8]; 
        dxdt[2] = x[9];
        dxdt[3] = x[10];
        dxdt[4] = x[11];

        // Auxiliary Vs
        dxdt[5] = x[8] - x[9]; 
        dxdt[6] = x[10] - x[11];

        // Ys
        dxdt[7] = He*ke*(gamma_1 * sigm(x[5]) + gamma_1T * (U + external_currents ));
        dxdt[7] -= 2*ke*x[7];
        dxdt[7] -= ke*ke*x[0];

        dxdt[8] = He*ke*(gamma_2 * sigm(x[0]) + gamma_2T * (P + external_currents ));
        dxdt[8] -= 2*ke*x[8];
        dxdt[8] -= ke*ke*x[1];

        dxdt[9] = Hi*ki*(gamma_4 * sigm(x[6]));
        dxdt[9] -= 2*ki*x[9];
        dxdt[9] -= ki*ki*x[2];  

        dxdt[10] = He*ke*(gamma_3 * sigm(x[5]) + gamma_3T * (Q + external_currents ));
        dxdt[10] -= 2*ke*x[10];
        dxdt[10] -= ke*ke*x[3];

        dxdt[11] = Hi*ki * (gamma_5 * sigm(x[6]));
        dxdt[11] -= 2*ki*x[11];
        dxdt[11] -= ki*ki*x[4];
    };

    // This is still unclear
    eeg_voi = [](const dynamical_state & x){return x[5];};

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}