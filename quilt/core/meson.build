cppsources = [
    'base.cpp',
    'devices.cpp',
    'neurons_base.cpp',
    'neuron_models.cpp',
    'network.cpp',
    'links.cpp',
    'oscillators.cpp',
    'multiscale.cpp',
]
boost_dep = dependency('boost')

libcpp = library('quilt', cppsources,
                 dependencies : boost_dep,
                 install: true,
                 )

# This is used only for testing and optimization using valgrind
# so it's placed in the root folder
executable('quilt.exe', cppsources + ['main.cpp'], 
            link_with : libcpp)