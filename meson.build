project('quilt', 'cpp','cython')

fs=import('fs')

cppsources = [
    'quilt/core/base.cpp',
    'quilt/core/devices.cpp',
    'quilt/core/neurons_base.cpp',
    'quilt/core/neuron_models.cpp',
    'quilt/core/network.cpp',
    'quilt/core/links.cpp',
    'quilt/core/oscillators.cpp',
    'quilt/core/multiscale.cpp',
]

pyxsources = ['quilt/interface/base.pyx',
                        'quilt/interface/spiking.pyx',
                        'quilt/interface/oscill.pyx',
                        'quilt/interface/multiscale.pyx']

inc_dirs = include_directories('quilt/core')
boost_dep = dependency('boost')

libcpp = library('quilt', cppsources,
                 dependencies : boost_dep,
                 include_directories : inc_dirs)

py = import('python').find_installation()
dep_py = py.dependency()

foreach cython_source : pyxsources
    cython_cpp_lib = py.extension_module(
    fs.name(cython_source).split('.')[0],
        cython_source,  # transpiled to C++
        override_options : ['cython_language=cpp'],
        link_with : [libcpp],
        dependencies : dep_py,
        include_directories : inc_dirs  # Aggiungi questa linea per includere la directory core/include
    )
endforeach

# cython_cpp_lib = py.extension_module(
#     'base',
#     'quilt/interface/base.pyx',  # transpiled to C++
#     override_options : ['cython_language=cpp'],
#     link_with : [libcpp],
#     dependencies : dep_py,
#     include_directories : inc_dirs  # Aggiungi questa linea per includere la directory core/include
# )



# project(
#   'quilt',
#   'c', 'cpp', 'cython',
#   version: '0.1.0',
#   license: 'MIT',
#   meson_version: '>= 1.1.0',
#   default_options: [
#     'buildtype=debugoptimized',
#     'b_ndebug=if-release',
#     'c_std=c17',
#     'cpp_std=c++17'
#   ],
# )
# fs = import('fs')

# py3 = import('python').find_installation(pure: false)
# py3_dep = py3.dependency()

# min_numpy_version = '1.23.5'

# cc = meson.get_compiler('c')
# cpp = meson.get_compiler('cpp')
# cy = meson.get_compiler('cython')
# cython = find_program(cy.cmd_array()[0])

# # Check compiler versions
# if cc.get_id() == 'gcc' and not cc.version().version_compare('>=9.1')
#   error('Quilt requires GCC >= 9.1')
# elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl' and not cc.version().version_compare('>=12.0')
#   error('Quilt requires clang >= 12.0')
# elif cc.get_id() == 'msvc' and not cc.version().version_compare('>=19.20')
#   error('Quilt requires at least vc142 (default with Visual Studio 2019)')
# endif
# if not cy.version().version_compare('>=3.0.8')
#   error('Quilt requires Cython >= 3.0.8')
# endif

# _global_c_args = cc.get_supported_arguments(
#   '-Wno-unused-but-set-variable',
#   '-Wno-unused-function',
#   '-Wno-conversion',
#   '-Wno-misleading-indentation'
# )
# add_project_arguments(_global_c_args, language: 'c')

# m_dep = cc.find_library('m', required: false)
# if m_dep.found()
#   add_project_link_arguments('-lm', language: 'c')
# endif

# inc_dirs = include_directories('quilt/core/include', 'quilt/interface')

# cython_sources = files(
#   'quilt/interface/base.pyx',
#   'quilt/interface/oscill.pyx',
#   'quilt/interface/spiking.pyx',
#   'quilt/interface/multiscale.pyx'
# )

# foreach cython_source : cython_sources
#   cython_source_basename = fs.name(cython_source)
#   py3.extension_module(cython_source_basename,
#     sources: [cython_source],
#     include_directories: inc_dirs,
#     dependencies: [py3_dep],
#     install: true
#   )
# endforeach