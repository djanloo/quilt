#include <iostream>
#include <variant>
#include <map>
#include <string>

using std::string;
using std::cout;
using std::endl;
using std::map;

typedef std::variant<int, float, string> param_t;

class ParaMap{
    public:
        map<string, param_t> value_map;

        ParaMap():value_map(){};
        ParaMap(map<string, param_t> values){
            for (auto pair : values){
                add(pair.first, pair.second);
            }
        }

        template <typename T>
        T get(string key){
            auto it = value_map.find(key);
            return std::get<T>(it->second);
        }

        float get(string key, float default_value) {
            auto it = value_map.find(key);
            if (it == value_map.end()){
                return default_value;
            }
            else{
                return std::get<float>(it->second);
            }
        }

        template <typename T>
        void add(string key, T value){
            value_map[key] = value;
            // cout <<"Inserted" << value_map[key] << endl; 
        }

        void update(ParaMap other) {
            for (auto entry : other.value_map) {
                value_map[entry.first] = entry.second;
            }
        }
        friend std::ostream& operator<<(std::ostream& os, const ParaMap& paramap) {
            os << "ALbert printed a paramap"<< endl;
            return os;
        }
};

int main(){

    ParaMap params(map<string, param_t >{{"B", 1}});

    params.add("A", 0.4f);
    // cout << "inserted single float"<<endl;

    params.add("C", "albert");
    cout <<"Default get is "<< params.get("D", 0.5) << endl;

    cout << "Testing update"<< endl;
    ParaMap newpars(map<string, param_t>{{"a", 0.1f}, {"b", "carlo"}});
    params.update(newpars);

    for (const auto& entry : params.value_map) {
        cout << entry.first << ": ";
        auto val = entry.second;
        if (std::holds_alternative<int>(val)){
            cout <<std::get<int>(entry.second);
            cout << " (int)";
        }
        if (std::holds_alternative<float>(val)){
            cout <<std::get<float>(val);
            cout << " (float)";
        }
        if (std::holds_alternative<string>(val)){
            cout << std::get<string>(val);
            cout << " (string)";
        }
        cout << endl;
    }

}