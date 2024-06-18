/**
 * This file is necessary only to profile the C++ part of the code
 * through callgrind & massif. It implements a dummy system and it's not part of the quilt's code.
*/

#include <iostream>
#include <fstream>
#include <string>
// #include <boost/math/special_functions/erf.hpp>

#include "include/base.hpp"
#include "include/devices.hpp"
#include "include/neurons_base.hpp"
#include "include/neuron_models.hpp"
#include "include/network.hpp"
#include "include/oscillators.hpp"
#include "include/multiscale.hpp"


using namespace std;
// ulimit -c unlimited
// sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t

float rand_01(){
    return ((float)rand())/RAND_MAX;
}

vector<vector<float>> get_rand_proj_mat(int N, int M, double min, double max){
    
    vector<vector<float>> matrix(N, vector<float>(M, 0));

    for (int i=0;i<N;i++){
        for (int j=0; j< M; j++){
            matrix[i][j] = min + (max-min)*rand_01();
        }
    }
    return matrix;
}


void test_spiking()
{

    int Na = 1000;
    int Nb = 1000;

    SpikingNetwork sn = SpikingNetwork();


    map<string, ParaMap::param_t> map_of_params{{"neuron_type", "aeif"},
                                        {"C_m", 40.1f},
                                        {"G_L",2.0f},
                                        {"E_l", -70.0f},
                                        {"V_reset", -55.0f},
                                        {"V_peak",0.1f},
                                        {"tau_refrac",0.0f},
                                        {"delta_T",1.7f},
                                        {"V_th", -40.0f},
                                        {"ada_a", 0.0f},
                                        {"ada_b",5.0f},
                                        {"ada_tau_w",100.0f},
                                        {"tau_ex", 10.0f},
                                        {"tau_in", 5.5f},
                                        {"E_ex", 0.0f},
                                        {"E_in",-65.0f}
                                        };

    ParaMap paramap = ParaMap(map_of_params);
    
    Population a = Population(Na, &paramap, &sn);
    Population b = Population(Nb, &paramap, &sn);

    cout << "size of neuron is " << sizeof(*(a.neurons[0])) << " bytes" << endl ;
    cout << "size of population is " << sizeof(a) << " bytes" << endl;

    vector<vector<float>>  weights, delays;

    weights = get_rand_proj_mat(Na,Nb, -0.02,0.1);
    delays = get_rand_proj_mat(Na,Nb, 0.5, 1.0);

    for (int i = 0; i < Na; i ++){
        for (int j=0; j < Nb; j++){
            if (rand_01() > 0.1){
                weights[i][j] = 0.0;
                delays[i][j] = 0.0;
            }
            if (abs(weights[i][j]) < WEIGHT_EPS){
                weights[i][j] = 0.0;
                delays[i][j] = 0.0;
            }
        }
    }

    Projection projection = Projection(weights, delays);

    a.project(&projection, &b);
    b.project(&projection, &a);


    // PopCurrentInjector stimulus_a = PopCurrentInjector(&a, 500.0, 0.0, 10.0);
    // PopCurrentInjector stimulus_b = PopCurrentInjector(&b, 500.0, 0.0, 10.0);
    PoissonSpikeSource stimulus_a = PoissonSpikeSource(&a, 900, 10.0, 2, 0.1, 2);
    PoissonSpikeSource stimulus_b = PoissonSpikeSource(&b, 900, 10.0, 2,0.1, 2);

    sn.add_injector(&stimulus_a);
    sn.add_injector(&stimulus_b);


    sn.add_spike_monitor(&a);

    EvolutionContext evo = EvolutionContext(0.1);
    
    sn.run(&evo, 5, 1);

    // if ( PopulationSpikeMonitor * psm = dynamic_cast<C*>(instance)) {
    //         derivedC->get_history();  // Chiamata a get_history() se l'istanza è di tipo C
    // }
    // for (auto val : sn.population_monitors[0]->get_history()){
    //     cout << val << " "; 
    // }
}

void test_poisson(){
    int Na = 5;
    int Nb = 5;

    SpikingNetwork sn = SpikingNetwork();


    map<string, ParaMap::param_t> map_of_params = {{"neuron_type", "aeif"},
                                        {"C_m", 40.1f},
                                        {"G_L",2.0f},
                                        {"E_l", -70.0f},
                                        {"V_reset", -55.0f},
                                        {"V_peak",0.1f},
                                        {"tau_refrac",0.0f},
                                        {"delta_T",1.7f},
                                        {"V_th", -40.0f},
                                        {"ada_a", 0.0f},
                                        {"ada_b",5.0f},
                                        {"ada_tau_w",100.0f},
                                        {"tau_ex", 10.0f},
                                        {"tau_in", 5.5f},
                                        {"E_ex", 0.0f},
                                        {"E_in",-65.0f}
                                        };

    ParaMap paramap = ParaMap(map_of_params);
    
    Population a = Population(Na, &paramap, &sn);
    Population b = Population(Nb, &paramap, &sn);

    SparseLognormProjection btoa(0.05, 0, 
                                Nb, Na, 
                                1.2, 0.0, 
                                1.5, 0.0 );

    SparseLognormProjection atob(0.05, 0, 
                                Na, Nb, 
                                3, 1, 
                                3, 1 );
    a.project(&atob, &b);
    b.project(&btoa, &a);

    PoissonSpikeSource stimulus_a = PoissonSpikeSource(&a, 6000, 10.0, 2, 0.1, -0.1);
    PoissonSpikeSource stimulus_b = PoissonSpikeSource(&b, 6000, 10.0, 2, 0.1, -0.1);

    sn.add_injector(&stimulus_a);
    sn.add_injector(&stimulus_b);

    EvolutionContext evo = EvolutionContext(0.1);
    sn.run(&evo, 5000, 1);
}

void test_oscill(){

    int N = 50;
    vector<vector<float>> weights, delays;


    weights = get_rand_proj_mat(N,N, 2.0, 5.0);
    delays = get_rand_proj_mat(N,N, 80, 200);

    for (int i = 0; i< N; i++){
        weights[i][i] = 0.0;
    }
    cout << "Making projection" << endl;
    Projection * proj = new Projection(weights, delays);

    EvolutionContext evo = EvolutionContext(1);

    ParaMap * params = new ParaMap();
    ParaMap * link_params = new ParaMap();

    // params->add("Q", 0);
    // params->add("P", 0);
    // params->add("U", 0);
    // params->add("Hi", 0.5);
    // params->add("He", 0.5);
    // params->add("gamma1_T", 0);
    // params->add("gamma2_T", 0);
    // params->add("gamma3_T", 0);
    // params->add("gamma4_T", 0);


    params->add("oscillator_type", "jansen-rit");
    // params->add("C", 1.0f);
    // params->add("P", -0.0f);
    // params->add("Q", -0.0f);
    // params->add("U", -0.0f);

    OscillatorNetwork osc_net = OscillatorNetwork(N, params);

    vector<dynamical_state> init_cond;
    for (int i=0; i< N; i++){
        vector<double> initstate(6, 2);

        // initstate[0] = 0.13 * (1+ static_cast<double>(rand())/RAND_MAX);
        // initstate[1] = 23.9 * (1+ static_cast<double>(rand())/RAND_MAX);
        // initstate[2] = 16.2 * (1+ static_cast<double>(rand())/RAND_MAX);
        // initstate[3] = -0.14/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);
        // initstate[4] = 5.68/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);
        // initstate[5] = 108.2/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);

        init_cond.push_back(initstate);
        cout << *(osc_net.oscillators[i]->params);
    }    

    osc_net.build_connections(proj, link_params);
    osc_net.initialize(&evo, init_cond);

    ofstream file("output.txt");
    osc_net.run(&evo, 2000, 1);

    for (int i=0; i < osc_net.oscillators[0]->memory_integrator.state_history.size(); i++){
        for (auto osc : osc_net.oscillators){
            for (auto val : osc->memory_integrator.state_history[i]){
                file << val << " ";
            }
        }
        file << endl;
    }
}


void test_sparse(){
    int Na = 4000;
    int Nb = 4000;

    SpikingNetwork sn = SpikingNetwork();


    map<string, ParaMap::param_t> map_of_params{{"neuron_type", "aeif"},
                                        {"C_m", 40.1f},
                                        {"G_L",2.0f},
                                        {"E_l", -70.0f},
                                        {"V_reset", -55.0f},
                                        {"V_peak",0.1f},
                                        {"tau_refrac",0.0f},
                                        {"delta_T",1.7f},
                                        {"V_th", -40.0f},
                                        {"ada_a", 0.0f},
                                        {"ada_b",5.0f},
                                        {"ada_tau_w",100.0f},
                                        {"tau_ex", 10.0f},
                                        {"tau_in", 5.5f},
                                        {"E_ex", 0.0f},
                                        {"E_in",-65.0f}
                                        };

    ParaMap paramap = ParaMap(map_of_params);
    
    Population a = Population(Na, &paramap, &sn);
    Population b = Population(Nb, &paramap, &sn);

    cout << "size of neuron is " << sizeof(*(a.neurons[0])) << " bytes" << endl ;
    cout << "size of population is " << sizeof(a) << " bytes" << endl;

    SparseLognormProjection btoa(0.05, 0, 
                                Nb, Na, 
                                1.2, 0.0, 
                                1.5, 0.0 );
    auto start = std::chrono::high_resolution_clock::now();
    SparseLognormProjection atob(0.05, 0, 
                                Na, Nb, 
                                3, 1, 
                                3, 1 );
    a.project(&atob, &b);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() ;
    cout << "MAIN: projection took "<< duration/1000/1000 << "ms ";
    cout << duration/atob.n_connections << " ns/syn" <<endl;
    b.project(&btoa, &a);

    std::ofstream out_file("output.txt");
    for (auto sector : atob.weights_delays){
        for (auto pair : sector){
            out_file << pair.first.first <<" "<< pair.first.second <<" "<< pair.second.first <<" "<<pair.second.second<<endl;
        }
    }
}

void test_multiscale_base(){
    // Create a spiking network of just one population
    SpikingNetwork spike_net = SpikingNetwork();

    map<string, ParaMap::param_t> map_of_params = {{"neuron_type", "aeif"},
                                        {"C_m", 40.1f},
                                        {"G_L",2.0f},
                                        {"E_l", -70.0f},
                                        {"V_reset", -55.0f},
                                        {"V_peak",0.1f},
                                        {"tau_refrac",0.0f},
                                        {"delta_T",1.7f},
                                        {"V_th", -40.0f},
                                        {"ada_a", 0.0f},
                                        {"ada_b",5.0f},
                                        {"ada_tau_w",100.0f},
                                        {"tau_ex", 10.0f},
                                        {"tau_in", 5.5f},
                                        {"E_ex", 0.0f},
                                        {"E_in",-65.0f}
                                        };
    cout << "creating spiking population" << endl;

    ParaMap spiking_paramap = ParaMap(map_of_params);
    Population spikepop = Population(500, &spiking_paramap, &spike_net);
    spike_net.add_injector(new PoissonSpikeSource(&spikepop, 500, 1, 0.0, 0, -1 ));

    cout << "creating oscillator" << endl;
    vector<vector<float>> weights, delays;
    EvolutionContext * evo_long  = new EvolutionContext(1.0);
    EvolutionContext * evo_short = new EvolutionContext(0.1);
    
    weights = get_rand_proj_mat(2,2, 2.0, 5.0);
    delays = get_rand_proj_mat(2,2, 80, 200);

    for (int i = 0; i< 2; i++){
        weights[i][i] = 0.0;
    }

    cout << "Making projection between oscillators" << endl;
    Projection * proj = new Projection(weights, delays);
    ParaMap * oscill_paramap = new ParaMap();
    oscill_paramap->add("oscillator_type", "jansen-rit");
    OscillatorNetwork osc_net = OscillatorNetwork(2, oscill_paramap);

    vector<dynamical_state> init_cond;
    for (int i=0; i< 2; i++){
        vector<double> initstate(6, 2);

        initstate[0] = 0.13 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[1] = 23.9 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[2] = 16.2 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[3] = -0.14/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[4] = 5.68/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);
        initstate[5] = 108.2/1e6 * (1+ static_cast<double>(rand())/RAND_MAX);

        
        init_cond.push_back(initstate);
        cout << *(osc_net.oscillators[i]->params);
    }    

    // osc_net.build_connections(proj, link_params);

    cout << "After init the long timescale is " << evo_long->now <<endl;
    
    cout << "creating multinet" << endl;
    // Create a multiscale network
    MultiscaleNetwork multi_net = MultiscaleNetwork(&spike_net, &osc_net);

    cout << "creating transducer" << endl;

    // Create a transducer
    ParaMap * transd_paramap = new ParaMap();
    shared_ptr <Transducer> transd (new Transducer(&spikepop, transd_paramap, &multi_net));

    cout << "adding transducer" << endl;

    multi_net.transducers.push_back(transd);

    cout << "building connections" <<endl;
    vector<vector<float>> T2Oweights {{1, 1}}; 
    vector<vector<float>> T2Odelays {{5 , 5}}; 
    vector<vector<float>> O2Tweights {{1}, {1}}; 
    vector<vector<float>> O2Tdelays {{5}, {5}}; 

    Projection * T2Oproj = new Projection(T2Oweights, T2Odelays);
    Projection * O2Tproj = new Projection(O2Tweights, O2Tdelays);

    multi_net.build_OT_projections(T2Oproj, O2Tproj);

    cout << "initializing oscillator network"<<endl;
    osc_net.initialize(evo_long, init_cond);

    cout << "initializing spiking network" << endl;
    spike_net.run(evo_short, evo_long->now, 1);
    cout << "running multinet" << endl;

    // Evolve
    multi_net.set_evolution_contextes(evo_short, evo_long);

    multi_net.run(10, 1);

    ofstream SpikeFile("spiking_history.txt");
    for (auto a : static_cast<PopulationSpikeMonitor*>(spike_net.population_monitors[0])->get_history()){
        SpikeFile << a << endl;
    }

    ofstream OscFile("osc_history.txt");

    int TT = osc_net.oscillators[0]->get_history().size();

    for (int i=0; i < TT; i++ ){
        for (auto oscill : osc_net.oscillators){
            OscFile << oscill->get_history()[i][0] << " ";
        }
        OscFile << endl;
    }


}

void test_inhom_poisson(){
    // Preamble: create a spiking population
    SpikingNetwork spike_net = SpikingNetwork();

    map<string, ParaMap::param_t> map_of_params = {{"neuron_type", "aeif"},
                                        {"C_m", 40.1f},
                                        {"G_L",2.0f},
                                        {"E_l", -70.0f},
                                        {"V_reset", -55.0f},
                                        {"V_peak",0.1f},
                                        {"tau_refrac",0.0f},
                                        {"delta_T",1.7f},
                                        {"V_th", -40.0f},
                                        {"ada_a", 0.0f},
                                        {"ada_b",5.0f},
                                        {"ada_tau_w",100.0f},
                                        {"tau_ex", 10.0f},
                                        {"tau_in", 5.5f},
                                        {"E_ex", 0.0f},
                                        {"E_in",-65.0f}
                                        };
    ParaMap spiking_paramap = ParaMap(map_of_params);
    Population spikepop = Population(500, &spiking_paramap, &spike_net);

    // Now create a dummy double(void) function that mimicks the Link::get() method
    std::function<double(float)> ratefunc = [](float now){
        // cout << "Getting rate"<< endl;
        double t_sec = now * 1e-3;
        double u = sin(6.28 * 10 * t_sec);
        return 200*u*u + 1000;
    };

    InhomPoissonSpikeSource ips(&spikepop, ratefunc, 0.5, 0.0 , 1000);
    EvolutionContext evo(0.1);
    ips.inject(&evo);
}

int main(){
    // test_spiking();
    // test_sparse();
    // test_poisson();
    // test_oscill();
    test_inhom_poisson();
    // test_multiscale_base();
}

