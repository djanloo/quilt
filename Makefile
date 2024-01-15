.PHONY: generate cleanall clean

CXX := g++
PYTHON := python3
CXXFLAGS := -std=c++11 -Wall -Wextra -g
SOURCES :=  quilt/src_cpp/test_file.cpp quilt/src_cpp/oscillators.cpp quilt/src_cpp/network.cpp quilt/src_cpp/neuron_models.cpp quilt/src_cpp/neurons_base.cpp quilt/src_cpp/devices.cpp quilt/src_cpp/base_objects.cpp

OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE := quilt.exe

generate:
	@ $(PYTHON) setup.py

cleanall: clean
	@echo "Cleaning all.."
	@rm -f quilt/*.so
	@rm -f quilt/*.html
	@rm -R -f quilt/build
	@rm -R -f quilt/bin/
	@rm -R -f quilt/cython_generated/
	@rm -R -f quilt/__pycache__
	@rm -R -f quilt/src_cython/__pycache__
	@rm -R -f quilt/src_cython/*.so
	@echo "Cleaned."


$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
