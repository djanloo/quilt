.PHONY: generate cleanall clean

CXX := g++
PYTHON := python3
CXXFLAGS := -std=c++11 -Wall -Wextra -g
SOURCES :=  quilt/core/test_file.cpp quilt/core/oscillators.cpp quilt/core/network.cpp quilt/core/neuron_models.cpp quilt/core/neurons_base.cpp quilt/core/devices.cpp quilt/core/base_objects.cpp

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
	@rm -R -f quilt/interface/__pycache__
	@rm -R -f quilt/interface/*.so
	@echo "Cleaned."


$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
