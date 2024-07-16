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
# inc_dirs = include_directories('core')
boost_dep = dependency('boost')

libcpp = library('quilt', cppsources,
                 dependencies : boost_dep,
                 install: true,
                #  install_dir: py.get_install_dir() / 'quilt/interface'
                 )