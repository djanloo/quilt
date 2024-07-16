.PHONY: generate meson cleanmeson debug lib cleanall clean

# CXX := g++
# PYTHON := python3
# CXXFLAGS := -std=c++17 -Wall -Wextra -O3
# CXXDEBUGFLAGS := -std=c++17 -Wall -Wextra -ggdb

# BOOST_LIBS := -lboost_filesystem -lboost_system
# SOURCES := quilt/core/multiscale.cpp quilt/core/oscillators.cpp quilt/core/links.cpp quilt/core/network.cpp quilt/core/neuron_models.cpp quilt/core/neurons_base.cpp quilt/core/devices.cpp quilt/core/base.cpp

# OBJECTS := $(patsubst %.cpp, %.o, $(SOURCES))
# DEPENDS := $(patsubst %.cpp,%.d, $(SOURCES))

# EXECUTABLE := quilt.exe
# LIBFILE := quilt/libquilt.so

meson:
	meson setup build
	meson compile -C build
	meson install -C build --destdir ../venv/

cleanmeson:
	rm -rf build

# generate: lib
# 	@ $(PYTHON) setup.py

# lib: $(LIBFILE)

# $(LIBFILE): $(OBJECTS)
# 	@echo "Building library"
# 	$(CXX) -shared -fPIC -o $(LIBFILE) $(OBJECTS) $(CXXFLAGS) $(BOOST_LIBS)

# $(EXECUTABLE): quilt/core/main.o $(LIBFILE)
# 	$(CXX) $< -o $@ -Lquilt -lquilt -Iquilt/core/include $(CXXFLAGS)

# debug: quilt/core/main.o $(LIBFILE)
# 	$(CXX) $< -o $@ -Lquilt -lquilt -Iquilt/core/include $(CXXDEBUGFLAGS)

# -include $(DEPENDS)

# %.o: %.cpp Makefile
# 	@echo "Compiling .cpp into .o files"
# 	$(CXX) $(WARNING) $(CXXFLAGS) -fPIC -MMD -MP -c $< -o $@

# clean:
# 	rm -f $(OBJECTS) $(EXECUTABLE)

# cleanall: clean
# 	@echo "Cleaning all.."
# 	@rm -f quilt/*.so
# 	@rm -f quilt/*.html
# 	@rm -R -f quilt/build
# 	@rm -R -f quilt/bin/
# 	@rm -R -f quilt/cython_generated/
# 	@rm -R -f quilt/__pycache__
# 	@rm -R -f quilt/interface/__pycache__
# 	@rm -R -f quilt/interface/*.so
# 	@rm -R -f quilt/core/*.d
# 	@rm -R -f quilt/core/*.o
# 	@echo "Cleaned."
