inc_dirs = include_directories('../core')

pyx_ext_names = ['base',
                            'spiking',
                            'oscill',
                            'multiscale']


dep_py = py.dependency()

incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()
inc_np = include_directories(incdir_numpy)
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

foreach name : pyx_ext_names

    cython_cpp_lib = py.extension_module(
        name,
        name + '.pyx',
        override_options : ['cython_language=cpp'],
        link_with : [libcpp],
        dependencies : [dep_py, np_dep],
        include_directories : inc_dirs,
        subdir: 'quilt/interface',
        install: true
    )
endforeach