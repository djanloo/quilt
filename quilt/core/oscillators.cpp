#include "include/oscillators.hpp"

map<std::string, int> OSCILLATOR_CODES = {
    {"harmonic", 0}, 
    {"test", 1}, 
    {"jansen-rit", 2}, 
    {"leon-jansen-rit", 3}
};

map<int, std::string> OSCILLATOR_NAMES = {
    {0, "harmonic"}, 
    {1, "test"}, 
    {2, "jansen-rit"}, 
    {3, "leon-jansen-rit"}
};

/******************************************* OSCILLATORS BASE **********************************/
Oscillator::Oscillator(const ParaMap * params, OscillatorNetwork * oscnet)
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
    // Bureaucracy
    id = HierarchicalID();

    string oscillator_type = OSCILLATOR_NAMES.at(static_cast<int>(params->get("oscillator_type")));

    for (int i = 0; i < N; i++){
        oscillators.push_back(get_oscillator_factory().get_oscillator(oscillator_type, params, this));
    }
}

void OscillatorNetwork::build_connections(Projection * proj)
{
    if (proj->start_dimension != proj->end_dimension)
    {
        throw runtime_error("Projection matrix of OscillatorNetwork must be a square matrix");
    }
    cout << "Building connections" << endl;
    for (int i =0; i < proj->start_dimension; i++)
    {
        for (int j = 0; j < proj->end_dimension; j++)
        {
            if (proj->weights[i][j] != 0)
            {                   
                oscillators[j]->incoming_osc.push_back(get_link_factory().get_link(oscillators[i],oscillators[i],proj->weights[i][j], proj->delays[i][j]));
            }
        }
    }
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
    cout << "Max delay is " << max_tau << endl;
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

    cout << "Network initialized (t = "<< evo->now << ")"<< endl;
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
            cout << "Oscillator Factory : building oscillator " << oscillator_type << endl;
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
harmonic_oscillator::harmonic_oscillator(ParaMap * paramap, OscillatorNetwork * oscnet)    
    :   Oscillator(params, oscnet)
{
    k = paramap->get("k");
    oscillator_type = "harmonic";
    space_dimension = 2;

    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t){
        dxdt[0] = x[1];

        for (auto input : incoming_osc){
            dxdt[1] += x[0] - k*input->get(0, t) ;
        }  
    };

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

test_oscillator::test_oscillator(ParaMap * paramap, OscillatorNetwork * oscnet)
    :   Oscillator(paramap, oscnet)
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

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}

// Auxiliary for Jansen-Rit
double jansen_rit_oscillator::sigm(double v, float nu_max, float v0, float r)
{
    return nu_max / (1.0 + std::exp(r*(v0-v)));
}

jansen_rit_oscillator::jansen_rit_oscillator(ParaMap * paramap, OscillatorNetwork * oscnet) 
    :   Oscillator(params, oscnet)
{
    oscillator_type = "jansen-rit";
    space_dimension = 6;

    // Parameters default from references
    A = paramap->get("A", 3.25);
    B = paramap->get("B", 22.0);
    a = paramap->get("a", 100.0/1000.0);   // ms^(-1)
    b = paramap->get("b", 50.0/1000.0);    // ms^(-1)
    vmax = paramap->get("vmax", 5.0/1000.0); // ms^(-1)
    v0 = paramap->get("v0", 6.0);
    C = paramap->get("C", 135.0);
    r = paramap->get("r", 0.56);

    // The system of ODEs implementing the evolution equation 
    evolve_state = [this](const dynamical_state & x, dynamical_state & dxdt, double t)
    {
        double external_currents = 0;
        for (auto input : incoming_osc)
        {
            external_currents += input->get(0, t);
        }
        double external_inputs = 130.0/1000.0 + external_currents;

        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];

        dxdt[3] = A*a*sigm( x[1] - x[2], vmax, v0, r) - 2*a*x[3] - a*a*x[0];
        dxdt[4] = A*a*(  external_inputs + 0.8*C*sigm(C*x[0], vmax, v0, r) ) - 2*a*x[4] - a*a*x[1];
        dxdt[5] = B*b*0.25*C*sigm(0.25*C*x[0], vmax, v0, r) - 2*b*x[5] - b*b*x[2];
    };

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

float leon_jansen_rit_oscillator::He = 3.25;
float leon_jansen_rit_oscillator::Hi = 22.0;
float leon_jansen_rit_oscillator::ke = 0.1;
float leon_jansen_rit_oscillator::ki = 0.05;

// Internal coupling parameters
float leon_jansen_rit_oscillator::gamma_1 = 135;
float leon_jansen_rit_oscillator::gamma_2 = 108;
float leon_jansen_rit_oscillator::gamma_3 = 33.75;
float leon_jansen_rit_oscillator::gamma_4 = 33.75;
float leon_jansen_rit_oscillator::gamma_5 = 15;

// External coupling parameters
float leon_jansen_rit_oscillator::gamma_1T = 1;
float leon_jansen_rit_oscillator::gamma_2T = 1;
float leon_jansen_rit_oscillator::gamma_3T = 1;

// Sigmoid parameters
float leon_jansen_rit_oscillator::e0 =    0.0025;
float leon_jansen_rit_oscillator::rho1 =  6;
float leon_jansen_rit_oscillator::rho2 =  0.56;

// Bifurcation parameters
float leon_jansen_rit_oscillator::U = 0.12;
float leon_jansen_rit_oscillator::P = 0.12;
float leon_jansen_rit_oscillator::Q = 0.12;

leon_jansen_rit_oscillator::leon_jansen_rit_oscillator(ParaMap * paramap, OscillatorNetwork * oscnet) 
    :   Oscillator(params, oscnet)
{
    // Referencing (Leon, 2015) for this system of equations
    // The variable of interest for the EEG is v2-v3
    oscillator_type = "leon-jansen-rit";
    space_dimension = 12;

    // Parameters default from references

    // Delay box parameters
    He = paramap->get("He", 3.25);
    Hi = paramap->get("Hi", 22.0);
    ke = paramap->get("ke", 0.1);   // ms^(-1)
    ki = paramap->get("ki", 0.05);  // ms^(-1)

    // Internal coupling parameters
    gamma_1 = paramap->get("gamma_1", 135);
    gamma_2 = paramap->get("gamma_2", 108);
    gamma_3 = paramap->get("gamma_3", 33.75);
    gamma_4 = paramap->get("gamma_4", 33.75);
    gamma_5 = paramap->get("gamma_5", 15);

    // External coupling parameters
    gamma_1T = paramap->get("gamma_1T", 1);
    gamma_2T = paramap->get("gamma_2T", 1);
    gamma_3T = paramap->get("gamma_3T", 1);

    // Sigmoid parameters
    e0 = paramap->get("e0", 0.0025); // ms^(-1)
    rho1 = paramap->get("rho1", 6);
    rho2 = paramap->get("rho2", 0.56);

    // Bifurcation parameters
    U = paramap->get("U", 0.12); // ms^(-1)
    P = paramap->get("P", 0.12); // ms^(-1)
    Q = paramap->get("Q", 0.12); // ms^(-1)

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
        // The output is thus x[5]

        double external_currents = 0;
        for (auto input : incoming_osc)
        {
            external_currents += input->get(5, t);
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
        dxdt[7] += ke*ke*x[0];

        dxdt[8] = He*ke*(gamma_2 * sigm(x[0]) + gamma_2T * (P + external_currents ));
        dxdt[8] -= ke*x[8];
        dxdt[8] += ke*ke*x[1];

        dxdt[9] = Hi*ki*(gamma_4 * sigm(x[6]));
        dxdt[9] -= 2*ki*x[9];
        dxdt[9] += ki*ki*x[2];

        dxdt[10] = He*ke*(gamma_3 * sigm(x[5]) + gamma_3T * (Q + external_currents ));
        dxdt[10] -= 2*ke*x[10];
        dxdt[10] += ke*ke*x[3];

        dxdt[11] = Hi*ki * (gamma_5 * sigm(x[6]));
        dxdt[11] -= 2*ki*x[11];
        dxdt[11] += ki*ki*x[4];
    };

    // Sets the stuff of the CRK
    memory_integrator.set_dimension(space_dimension);
    memory_integrator.set_evolution_equation(evolve_state);
}