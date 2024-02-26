.PHONY: generate cleanall clean test_file.hpp

CXX := g++
PYTHON := python3
CXXFLAGS := -std=c++11 -Wall -Wextra -ggdb
SOURCES := quilt/core/oscillators.cpp quilt/core/network.cpp quilt/core/neuron_models.cpp quilt/core/neurons_base.cpp quilt/core/devices.cpp quilt/core/base.cpp

OBJECTS := $(patsubst %.cpp, %.o, $(SOURCES))
DEPENDS := $(patsubst %.cpp,%.d, $(SOURCES))

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

$(EXECUTABLE): quilt/core/test_file.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

-include $(DEPENDS)

%.o: %.cpp Makefile
	$(CXX) $(WARNING) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
