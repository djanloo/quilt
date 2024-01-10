.PHONY: generate clean

generate:
	@python3 setup.py

cleanall:
	@echo "Cleaning all.."
	@rm -f quilt/*.so
	@rm -f quilt/*.html
	@rm -R -f quilt/build
	@rm -R -f quilt/bin/
	@rm -R -f quilt/cython_generated/
	@rm -R -f quilt/__pycache__
	@echo "Cleaned."

CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra
SOURCES :=  quilt/src_cpp/test_file.cpp quilt/src_cpp/oscillators.cpp quilt/src_cpp/network.cpp quilt/src_cpp/neuron_models.cpp quilt/src_cpp/neurons_base.cpp quilt/src_cpp/devices.cpp quilt/src_cpp/base_objects.cpp

OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE := hello.exe

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
