/**
 * This is a test for performance in using boost.odeint in evolving
 * options are:
 *  - lambdas
 *  - static class methods
*/
#include <iostream>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

typedef vector<double> state_type;

class EvoContext{
    public:
        double t, dt;
        EvoContext(double dt): dt(dt), t(0.0){}
        void do_step(){this->t+=this->dt;}
};

class EvolvingObject{
    public:
        vector<double> state;
        double par_a, par_b;

        EvolvingObject(double par_a,double par_b):state({static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX}),par_a(par_a),par_b(par_b){}
        void evolve(const state_type &x , state_type &dxdt , const double t ){
            dxdt[0] =   this->par_a * x[1];
            dxdt[1] = - this->par_b * x[0];
        }
};

class EvolvingContainer{
    public:
        vector<EvolvingObject*> objs;
        runge_kutta4<state_type> stepper;

        EvolvingContainer(){};
        void add_obj(EvolvingObject * evolving_obj){
            this-> objs.push_back(evolving_obj);
        }
        void evolve(EvoContext * evo){
            for (auto obj : this->objs){
                auto lambda = [&obj](const state_type &state, state_type &dxdt, double t) {
                                    obj->evolve(state, dxdt, t);
                                };
                stepper.do_step(lambda, obj->state, evo->t, evo->dt);
            }
            evo->do_step();
        }
};

int main(){
    EvolvingContainer evocont = EvolvingContainer();

    for (int i = 0; i < 50; i++){
        evocont.add_obj(new EvolvingObject(1.0, 2.0));
    }
    EvoContext evo = EvoContext(0.1);
    while (evo.t < 100){
        cout << evocont.objs[0]->state[0] << endl;
        evocont.evolve(&evo);
        }

}