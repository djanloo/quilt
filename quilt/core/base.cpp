#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <iomanip> // for std::setprecision

// #include <boost/timer/progress_display.hpp>
#include "include/base.hpp"

#define PERFMGR_OUTPUT_DIGITS 1
#define DEFAULT_LOG_FILE "quilt_log.log"

using std::cout;
using std::endl;
using std::runtime_error;
using std::vector;


namespace settings {
    LogLevel verbosity = INFO;
    string global_logfile = "";

    void set_verbosity(int value) {
        // std::cout << "Set verbosity to " << value << std::endl;
        if ((value < 0)|(value > 4)){
            stringstream ss;
            ss << "Verbosity must be between 0(NO OUTPUT) and 4(DEBUG)";
            get_global_logger().log(ERROR, ss.str());

            ss << endl;
            std::cerr << ss.str();
        }
        verbosity = static_cast<LogLevel>(value);
    }

}
//************************* THREAD SAFE FILE ***************************//

ThreadSafeFile::ThreadSafeFile (const std::string& filename) : filename(filename), file() {
    open();
}

ThreadSafeFile::~ThreadSafeFile() {
    if (file.is_open()) {
        close();
    }
}

void ThreadSafeFile::open() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!file.is_open()) {
        file.open(filename, std::ios::trunc);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }
}

void ThreadSafeFile::write(const std::string& message) {
    std::lock_guard<std::mutex> lock(mtx);
    if (file.is_open()) {
        file << message << std::endl;
        file.flush();
    } else {
        throw std::runtime_error("Writinig on a not-yet-open file");
    }
}

void ThreadSafeFile::close() {
    std::lock_guard<std::mutex> lock(mtx);
    if (file.is_open()) {
        file.close();
    }
}


//******************************* LOGGER ***************************//
Logger::Logger(const string& filename)
    :   logFile(filename){}

Logger::~Logger() { logFile.close(); } 

// void Logger::set_level(LogLevel level){
//     output_level = level;
// }
  
void Logger::log(LogLevel level, const string& message) 
{   
    // Do not print level under the current one
    if (level < ERROR - settings::verbosity){
        return;
    }

    time_t now = time(0); 
    tm* timeinfo = localtime(&now); 
    char timestamp[20]; 
    strftime(timestamp, sizeof(timestamp), 
                "%Y-%m-%d %H:%M:%S", timeinfo); 

    std::ostringstream logEntry; 
    logEntry << "[" << timestamp << "] " 
            //  << "- PID " << std::this_thread::get_id() << " - "
             << levelToString(level) << ": " << message; 

    // Output to console 
    cout << logEntry.str() << endl; 

    // Output to log file
    logFile.write(logEntry.str());
}

Logger& get_global_logger(){
    if (!settings::global_logfile.empty()){
        static Logger logger(settings::global_logfile);
        return logger;
    }else{
        static Logger logger(DEFAULT_LOG_FILE);
        return logger;
    }
}

/************************************ PERFORMANCE MANAGER ***********************************/

PerformanceManager::PerformanceManager(string label)
    :   label(label)
    {

    stringstream ss;
    ss << "Created PerformanceManager at " << this;
    get_global_logger().log(DEBUG, ss.str());   
}

PerformanceManager::~PerformanceManager(){
    stringstream ss;
    ss << "Destroyed PerformanceManager at " << this;
    get_global_logger().log(DEBUG, ss.str());
}

void PerformanceManager::set_tasks(vector<string> task_names){
    for (unsigned int i = 0; i < task_names.size(); i++ ){
        task_duration[task_names[i]] = std::chrono::nanoseconds::zero();
        task_count[task_names[i]] = 0;
        task_scale[task_names[i]] = 1;
    }
}

void PerformanceManager::set_scales(map<string, int> scales){
    for (auto &pair : scales){
        task_scale[pair.first] = pair.second;
    }
}

void PerformanceManager::set_label(string label){
    this->label = label;
}

void PerformanceManager::start_recording(string task){
    task_start_time[task] = std::chrono::high_resolution_clock::now();
}
void PerformanceManager::end_recording(string task){
    task_count[task] ++;
    task_duration[task] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - task_start_time[task]);
}

void PerformanceManager::print_record(){
    stringstream ss;
    ss << "Output for PerformanceManager "  << "<" << label << ">" << endl;
    for (auto &pair : task_duration){
        ss << std::setfill('_') << std::setw(15) << pair.first  << " " << std::setw(10) << format_duration(pair.second);
        std::chrono::nanoseconds time_per_call = pair.second;

        if (task_count[pair.first] > 1){
            time_per_call /= static_cast<double>(task_count[pair.first]);
            ss << " | " << std::setw(8) <<  format_duration(time_per_call)  << "/step for "<< task_count[pair.first] << " steps";
        }

        if (task_scale[pair.first] > 1){
            time_per_call /= task_scale[pair.first];
            ss << " | " << std::setw(8) << format_duration(time_per_call)  << "/step/unit for "<< task_count[pair.first] << " steps and "<< task_scale[pair.first] << " units";
        }
        ss << endl;
    } 
    get_global_logger().log(INFO, ss.str());
}

string PerformanceManager::format_duration (std::chrono::nanoseconds duration) {
    auto nanoseconds = duration.count();
    stringstream ss;
    ss << std::setprecision(PERFMGR_OUTPUT_DIGITS) << std::fixed;
    if (nanoseconds >= 1000000000) {
        double seconds = std::chrono::duration_cast <std::chrono::milliseconds>(duration).count()/1000.0;
        ss << seconds << " s";
    } else if (nanoseconds >= 1000000) { 
        double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count()/1000.0;
        ss << milliseconds << " ms";
    } else if (nanoseconds >= 1000){ 
        double microseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()/1000.0;
        ss << microseconds << " us";
    } else {
        double nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        ss << nanoseconds << " ns";
    }
    return ss.str();
}

PerformanceRegistrar::PerformanceRegistrar(){
    stringstream ss;
    ss <<  "Created PerformanceRegistrar at index: "<<this; 
    get_global_logger().log(DEBUG,ss.str());
}
PerformanceRegistrar::~PerformanceRegistrar(){
    stringstream ss;
    ss <<  "Destroyed PerformanceRegistrar at index: "<<this;
    get_global_logger().log(DEBUG,ss.str());
}

// Static method definition to get the singleton instance
PerformanceRegistrar& PerformanceRegistrar::get_instance() {
    static PerformanceRegistrar instance;
    return instance;
}

// Method to add a PerformanceManager instance to the registrar
void PerformanceRegistrar::add_manager(std::shared_ptr<PerformanceManager> pm) {
    cleanup();
    instances.push_back(pm);
    stringstream ss;
    ss << "PerformanceRegistrar: registered manager " << instances.size() <<":"<< pm->label;
    get_global_logger().log(DEBUG, ss.str());
}

// Method to print records for all registered instances
void PerformanceRegistrar::print_records() {
    stringstream ss;
    ss << "Ouput for PerformanceRegistrar "<< "(" << instances.size() << " managers )";
    get_global_logger().log(INFO, ss.str());
    for (auto pm : instances) {
        //  Locks the weak pointer and prints
        auto locked_ptr = pm.lock();
        if (locked_ptr) locked_ptr->print_record();
    }
}

void PerformanceRegistrar::cleanup(){
    get_global_logger().log(DEBUG, "Cleaning up PerformanceRegistrar");
    auto expiration_criteria = [](const std::weak_ptr<PerformanceManager>& pm) { return pm.expired(); };
    instances.erase(std::remove_if(instances.begin(), instances.end(), expiration_criteria), instances.end());
}

//************************* UTILS FOR DYNAMICAL SYSTEMS **********************//

HierarchicalID::HierarchicalID()
            :   parent(NULL),
                local_id(0),
                n_subclasses(0)
{
    // get_global_logger().log(DEBUG, "Initialized a ROOT HierarchicalID - " + to_string(local_id));
}

HierarchicalID::HierarchicalID(HierarchicalID * parent)
    :   parent(parent),
        n_subclasses(0)
{
    local_id = parent->n_subclasses;
    parent->n_subclasses ++;
    // get_global_logger().log(DEBUG, "Initialized a NON ROOT HierarchicalID - " + to_string(local_id));
}
int HierarchicalID::get_id(){return local_id;}

EvolutionContext::EvolutionContext(double dt)
    :   dt(dt),
        now(0.0){}

void EvolutionContext::do_step(){
    now += dt;
    n_steps_done ++;
    times.push_back(now);
}

int EvolutionContext::index_of(double time)
{
    if (time < 0.0){
        stringstream ss;
        ss <<"Requested index of a negative time: " << time <<" ms.";
        // get_global_logger().log(WARNING, ss.str() + "Throwing an exception, will it be catched?" );
        throw negative_time_exception(ss.str());
        }

    if (time == 0.0 ){
        return 0;
    }
    return static_cast<int>(time/dt);
}

double EvolutionContext::deviation_of(double time)
{
    double deviation = time/dt - index_of(time);

    // Deviation is by definition positive
    deviation = (deviation < 0) ? 0.0 : deviation;
    stringstream ss;
    // ss << "Deviation in EC is " << deviation;
    // get_global_logger().log(WARNING, ss.str());
    return deviation;
}

/*********************************** PROJECTIONS ************************************************/
Projection::Projection(vector<vector<float>> weights, vector<vector<float>> delays):
    weights(weights), delays(delays){
    
    start_dimension = weights.size();
    if (start_dimension == 0){
        get_global_logger().log(ERROR,"Start dimension of projection is zero" );
        throw std::runtime_error("Start dimension of projection is zero");
    }
    
    end_dimension = weights[0].size();
    if (end_dimension == 0){ 
        get_global_logger().log(ERROR,"End dimension of projection is zero" );
        throw std::runtime_error("End dimension of projection is zero");
    }

    for (unsigned int i = 0; i < start_dimension; i++){
        for (unsigned int j =0 ; j< end_dimension; j++){
            if (std::abs(weights[i][j]) >= WEIGHT_EPS){ 
                n_links ++;
            }
        }
    }
}

/*********************************** CONTINUOUS RK **********************************************/
vector<double> ContinuousRK::b_functions(double theta)
{
    if ((theta < 0)){
        get_global_logger().log(ERROR,"NCERK b-functions were called with argument < 0" );
        throw std::invalid_argument("NCERK b-functions were called with argument < 0");
    }
    else if ((theta > 1)){
        get_global_logger().log(ERROR,"NCERK b-functions were called with argument > 1" );
        throw std::invalid_argument("NCERK b-functions were called with argument > 1");
    }

    vector<double> result(4);

    // For the default RK4 this is a NCE of order 3
    result[0] = ( 2.0/3.0 * theta * theta - 3.0/2.0 * theta + 1.0) * theta;
    result[1] = (-2.0/3.0 * theta + 1) * theta * theta;
    result[2] = result[1];
    result[3] = ( 2.0/3.0 * theta - 0.5) * theta * theta;

    return result;
}

double ContinuousRK::get_past(int axis, double abs_time){
    /**
     * Remember that:
     * 
     * X_{n+1} = X_n + h * sum_nu{ b_nu K^{nu}_n}
     * 
     * X_{n+1} is the proposed_state
     * K_^{nu}_{n} (for each nu) is a proposed_evaluation
     * 
     * 
    */

    // Split in bin_index + fractionary part
    int bin_id = evo->index_of(abs_time);
    double theta = evo->deviation_of(abs_time);
    
    if (bin_id < 0){
        get_global_logger().log(ERROR, "Requested past state that lays before initialization");
        throw negative_time_exception("Requested past state that lays before initialization");
    }
    
    // Remember that if state history has N values then the evaluation history has N-1
    else if (bin_id > static_cast<int>(state_history.size()) - 2){
        stringstream ss;
        ss << "The requested past state is not computed yet"<< "(requested " << abs_time << ")";
        get_global_logger().log(ERROR, ss.str());
        throw not_yet_computed_exception(ss.str());
    }

    // Get the values and the interpolation weights related to that moment in time
    double y = state_history[bin_id][axis];
    vector<double> b_func_values = b_functions(theta);

    // Updates using the interpolant
    for (int nu = 0; nu < 4; nu++){
        y += evo->dt * b_func_values[nu] * evaluation_history[bin_id][nu][axis];
    }

    /**      *      *                  DEBUG        *        *                */

    double checkval = state_history[bin_id][axis];
    double delta_value = 0.0;
    b_func_values = b_functions(1.0);

    for (int nu = 0; nu < 4; nu++){
        delta_value += evo->dt * b_func_values[nu] * evaluation_history[bin_id][nu][axis];
    }
    checkval += delta_value;
   
    if ( std::abs(checkval - state_history[bin_id+1][axis])/checkval > 0.005){
        stringstream msg;
        msg << "NCERK interpolated at theta = 1 is different from the next point." << endl
            << "Interpolating at t = " << std::setprecision(std::numeric_limits<double>::max_digits10) << abs_time << "( n = "<< bin_id << " theta = " << theta <<")" << endl
            << "X[n] = "<< state_history[bin_id][axis]<< endl
            << "X[n+1] = "<< state_history[bin_id+1][axis]<< endl
            << "interpolation(theta=1) =" << checkval << endl
            << "delta(theta = 1) = "<< delta_value << endl;

        for (int nu = 0; nu< 4; nu++){
            msg << "b_"<<nu << "(theta = 1 ): "<< b_func_values[nu]<<endl;
            msg << "b_"<<nu << ": "<< b[nu] << endl;

            msg << "\tadding to delta b_"<<nu <<" * K_"<<nu <<endl
                << "\t\t K_"<< nu <<" = " << evaluation_history[bin_id][nu][axis] << endl;
        }
        get_global_logger().log(ERROR, msg.str());
    }

    return y;
}

void ContinuousRK::compute_next(){
    if (space_dimension == 0){
        get_global_logger().log(ERROR,"Space dimension not set in ContinuousRK");
        throw runtime_error("Space dimension not set in ContinuousRK");
    }

    // This is a runtime guard for bug #18
    if (state_history.size() == evaluation_history.size()){
        get_global_logger().log(ERROR,"State history of ContinuousRK has the same length of evaluation history.");
        throw runtime_error("State history of ContinuousRK has the same length of evaluation history.");
    }
    proposed_evaluation = vector<dynamical_state>(4, dynamical_state(space_dimension, 0));

    double t_eval;
    dynamical_state x_eval;

    // Compute the K values
    for (int nu = 0; nu < 4; nu++){ 

        // When
        t_eval = evo->now + c[nu] * evo->dt;

        // Where
        x_eval = state_history.back();

        if (nu != 0) { //This is skipped at first iteration
            for (unsigned int i = 0; i < space_dimension; i++){
                x_eval[i] += evo->dt * a[nu] * proposed_evaluation[nu-1][i];
            }
        }//~Where

        /* NOTE: The WHERE section is equivalent to:
        
        if ( nu == 0){
            x_eval = state_history.back();
        }else if ( nu == 1){
            x_eval = state_history.back();
            for (unsigned int i = 0; i < space_dimension; i++){
                x_eval[i] += evo->dt * 0.5 * proposed_evaluation[0][i];
            }
        }else if(nu==2){
            x_eval = state_history.back();
            for (unsigned int i = 0; i < space_dimension; i++){
                x_eval[i] += evo->dt * 0.5 * proposed_evaluation[1][i];
            }
        }else if (nu == 3){
            x_eval = state_history.back();
            for (unsigned int i = 0; i < space_dimension; i++){
                x_eval[i] += evo->dt * 1 * proposed_evaluation[2][i];
            }
        }
        */

        // Assigns the new K evaluation to the value of the evolution function
        evolve_state(x_eval, proposed_evaluation[nu], t_eval);

    }//~Compute K values

    // Updates the state
    proposed_state = state_history.back();
    for (unsigned int i = 0; i < space_dimension; i++){
        for (int nu = 0; nu < 4; nu++){
            proposed_state[i] += evo->dt * b[nu] * proposed_evaluation[nu][i];
        }
    }//~Updates the state
    
}

void ContinuousRK::fix_next(){
    state_history.push_back(proposed_state);
    evaluation_history.push_back(proposed_evaluation);
}



