#include "include/oscillators.hpp"

/******************************************* OSCILLATORS BASE **********************************/
Oscillator::Oscillator(ParaMap * params, OscillatorNetwork * oscnet)
    :   oscnet(oscnet),
        params(params),
        memory_integrator()
{
    id = HierarchicalID(oscnet->id);
    evolve_state = [](const dynamical_state & /*x*/, dynamical_state & /*dxdt*/, double /*t*/){cout << "Warning: using virtual evolve_state of Oscillator" << endl;};
}

void Oscillator::set_evolution_context(EvolutionContext * evo)
{
    this->evo = evo;
    memory_integrator.set_evolution_context(evo);
    for (auto & incoming_link : incoming_osc)
    {
        incoming_link->set_evolution_context(evo);
    }
};

// Homogeneous network builder
OscillatorNetwork::OscillatorNetwork(int N, ParaMap * params)
{
    cout << "Oscillator network homogeneous constructor" << endl;

    // Bureaucracy
    id = HierarchicalID();

    string oscillator_type = params->get<string>("oscillator_type");

    for (int i = 0; i < N; i++){
        oscillators.push_back(get_oscillator_factory().get_oscillator(oscillator_type, params, this));
    }
    has_oscillators = true;
    cout << "Done Oscillator network homogeneous constructor" << endl;
}

// Homogeneous network builder
OscillatorNetwork::OscillatorNetwork(vector<ParaMap *> params)
{   
    cout << "Oscillator network inhomogeneous constructor" << endl;
    // Bureaucracy
    id = HierarchicalID();
    string oscillator_type;

    for (unsigned int i = 0; i < params.size() ; i++){
        oscillator_type = params[i]->get<string>("oscillator_type");
        oscillators.push_back(get_oscillator_factory().get_oscillator(oscillator_type, params[i], this));
    }
    has_oscillators = true;
    cout << "Done Oscillator network inhomogeneous constructor" << endl;
}

void OscillatorNetwork::build_connections(Projection * proj, ParaMap * link_params)
{
    cout << "Oscillator network build_connections" << endl;

    if (!has_oscillators){
        throw runtime_error("Could not link oscillators since the network does not have oscillators yet.");
    }

    if (proj->start_dimension != proj->end_dimension)
    {
        throw std::invalid_argument("Projection matrix of OscillatorNetwork must be a square matrix");
    }

    cout << "Test 0" << endl;
    cout << oscillators.size() << endl;
    cout << "Done Test 0"<< endl;

    if (proj->start_dimension != oscillators.size())
    {
        throw std::invalid_argument("Shape mismatch in projection matrix: size is "\
                    + std::to_string(proj->end_dimension) + "x" + std::to_string(proj->end_dimension) \
                    + " but network has n_oscillators = " + std::to_string(oscillators.size())
        );
    }

    cout << "Oscillator network build_connections (A)" << endl;
    for (unsigned int i =0; i < proj->start_dimension; i++)
    {
        for (unsigned int j = 0; j < proj->end_dimension; j++)
        {
            if (proj->weights[i][j] != 0)
            {                   
                oscillators[j]->incoming_osc.push_back(get_link_factory().get_link(oscillators[i], oscillators[j], proj->weights[i][j], proj->delays[i][j], link_params));
            }
        }
    }
    has_links = true;
    cout << "Done Oscillator network build_connections" << endl;
}

void OscillatorNetwork::initialize(EvolutionContext * evo, vector<dynamical_state> init_conds)
{
    cout << "Initializing oscillators" << endl;
    if (init_conds.size() != oscillators.size()) throw runtime_error("Number of initial conditions is not equal to number of oscillators");
    
    // brutal search of maximum delay
    float max_tau = 0.0;
    for (auto osc : oscillators){
        for (auto l : osc->incoming_osc){
            if (l->delay > max_tau) max_tau = l->delay;
        }
    }
    // cout << "Max delay is " << max_tau << endl;
    // ~brutal search of maximum delay
    
    int n_init_pts = static_cast<int>(std::ceil(max_tau/evo->dt) + 1);

    for (unsigned int i = 0; i < init_conds.size(); i++ ){
        vector<dynamical_state> new_K(4, vector<double>(oscillators[i]->get_space_dimension()));

        // Computes the value of X and K for the past values
        for (int n = 0; n < n_init_pts; n++){
            //Checks that the initialization is right in dimension
            if (init_conds[i].size() != oscillators[i]->space_dimension) 
                throw runtime_error("Initial conditions have not the right space dimension");
            
            // Adds initial condition as value of X 
            oscillators[i]->memory_integrator.state_history.push_back(init_conds[i]);

            // Computes the values of K for each intermediate step
            for (int nu = 0; nu < 4; nu++){
                for (unsigned int dim = 0; dim < oscillators[i]->get_space_dimension(); dim++){
                    new_K[nu][dim] = 0.0;
                }
            }
            oscillators[i]->memory_integrator.evaluation_history.push_back(new_K);
        }
    }


    // Set the current time to max_tau
    // This makes the initialization to be in [0, T] instead of [-T, 0]
    // but otherwise we should consider negative times in the `get()` method of links
    for (int n = 0; n < n_init_pts - 1; n++){  // If I make n points I am at t = (n-1)*h because zero is included
        evo->do_step();
    }

    // cout << "Network initialized (t = "<< evo->now << ")"<< endl;
    is_initialized = true;
}

void OscillatorNetwork::run(EvolutionContext * evo, double time, int verbosity)
{
    if (!is_initialized) throw runtime_error("The network must be initialized before running");
    
    // Synchronizes every component
    set_evolution_context(evo);

    // Some verbose output
    double t0 = evo->now;
    int n_steps_total = static_cast<int>(time/evo->dt);
    if (verbosity > 0){
        std::cout << "Running network consisting of " << oscillators.size() << " oscillators for " << n_steps_total <<" timesteps"<<std::endl;
    }

    // Evolve
    progress bar(n_steps_total, verbosity);

    auto start = std::chrono::high_resolution_clock::now();
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
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Simulation took " << std::chrono::duration_cast<std::chrono::seconds>(end-start).count()<< " seconds ";
    cout << "( " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count())/n_steps_total<< " ms/step)" << endl;
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
            if (it == _constructor_map.end()) { throw runtime_error("No constructor was found for oscillator " + oscillator_type); }
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

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){

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
    He = params->get("He", 3.25);
    Hi = params->get("Hi", 22.0);
    ke = params->get("ke", 0.1);   // ms^(-1)
    ki = params->get("ki", 0.05);    // ms^(-1)
    rmax = params->get("rmax", 0.005); // ms^(-1)
    v0 = params->get("v0", 6.0);
    C = params->get("C", 135.0);
    s = params->get("s", 0.56);

    // The system of ODEs implementing the evolution equation 
    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t)
    {
        double external_currents = 0;
        for (auto input : incoming_osc)
        {
            external_currents += input->get(0, t);
        }
        double external_inputs = 0.13 + external_currents + 0.19*static_cast<double>(rand())/RAND_MAX;

        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];

        dxdt[3] = He*ke*sigm( x[1] - x[2]) - 2*ke*x[3] - ke*ke*x[0];
        dxdt[4] = He*ke*( external_inputs + 0.8*C*sigm(C*x[0]) ) - 2*ke*x[4] -  ke*ke*x[1];
        dxdt[5] = Hi*ki*0.25*C*sigm(0.25*C*x[0]) - 2*ki*x[5] - ki*ki*x[2];
    };

    // Sets the variable of interest for the EEG
    // In this case it corresponds to the pyramidal cells lumped potential
    eeg_voi = [](const dynamical_state & x){return x[1] - x[2];};

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
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