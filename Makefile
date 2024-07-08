.PHONY: generate lib cleanall clean test_file.hpp

CXX := g++
PYTHON := python3
CXXFLAGS := -std=c++17 -Wall -Wextra -ggdb
BOOST_LIBS := -lboost_filesystem -lboost_system
SOURCES := quilt/core/multiscale.cpp quilt/core/oscillators.cpp quilt/core/links.cpp quilt/core/network.cpp quilt/core/neuron_models.cpp quilt/core/neurons_base.cpp quilt/core/devices.cpp quilt/core/base.cpp

OBJECTS := $(patsubst %.cpp, %.o, $(SOURCES))
DEPENDS := $(patsubst %.cpp,%.d, $(SOURCES))

EXECUTABLE := quilt.exe
LIBFILE := quilt/libquilt.so

generate:
	@ $(PYTHON) setup.py

lib: $(LIBFILE)

$(LIBFILE): $(OBJECTS)
	$(CXX) -shared -fPIC -o $(LIBFILE) $(OBJECTS) $(CXXFLAGS) $(BOOST_LIBS)

$(EXECUTABLE): quilt/core/test_file.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

-include $(DEPENDS)

%.o: %.cpp Makefile
	$(CXX) $(WARNING) $(CXXFLAGS) -fPIC -MMD -MP -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

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
	@rm -R -f quilt/core/*.d
	@rm -R -f quilt/core/*.o
	@echo "Cleaned."
