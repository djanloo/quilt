from setuptools import Extension, setup
import os

from Cython.Distutils import build_ext
from Cython.Compiler.Options import get_directive_defaults
from Cython.Build import cythonize

import numpy as np
from rich import print
import yaml

BIN_FOLDER = './interface'
CYTHON_GEN_FOLDER = './cython_generated'
DEPENDENCIES = "dependencies.yaml"
DEFAULT_INCLUDES = ["/.", "./quilt", np.get_include()]

old_dir = os.getcwd()
packageDir = "./quilt"

os.chdir(packageDir)

try:
    with open(DEPENDENCIES, "r") as dependencies:
        dependencies = yaml.safe_load(dependencies)
except FileNotFoundError:
    dependencies = dict()
    pass

# Creates directory for generated files if not existing
if not os.path.exists(CYTHON_GEN_FOLDER):
    os.mkdir(CYTHON_GEN_FOLDER)

extension_kwargs = dict( 
        include_dirs = DEFAULT_INCLUDES,
        language="c++",
        libraries=["m"],                      
        extra_compile_args=[#"-fopenmp", 
                            "-O3", 
                            "-std=c++11"],
        # extra_link_args=["-fopenmp"],
        define_macros= [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')] #Silences npy deprecated warn
        )

cython_compiler_directives = get_directive_defaults()
cython_compiler_directives['language_level'] = "3"
cython_compiler_directives['warn'] = True

extensions = []
for extension_name in dependencies.keys():

    current_extension_kwargs = extension_kwargs.copy()
    includes = [os.path.abspath(include_folder) for include_folder in dependencies[extension_name]['include_dirs']]
    current_extension_kwargs['include_dirs'] += includes
    extensions.append(Extension(extension_name, dependencies[extension_name]['sources'], **extension_kwargs))

ext_modules = cythonize(extensions, 
                        nthreads=4,
                        compiler_directives=cython_compiler_directives,
                        include_path=["."],
                        build_dir = CYTHON_GEN_FOLDER,
                        force=False,
                        annotate=False
)

setup(  name='quilt',
        packages=['quilt'],
        cmdclass={"build_ext": build_ext},
        include_dirs=DEFAULT_INCLUDES,
        ext_modules=ext_modules,
        script_args=["build_ext", f"--build-lib=./{BIN_FOLDER}"],
        options={"build_ext": {"inplace": False, "force": True, "parallel":True}},
)

os.chdir(old_dir)