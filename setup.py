from setuptools import Extension, setup
import os

from Cython.Distutils import build_ext
from Cython.Compiler.Options import get_directive_defaults
from Cython.Build import cythonize

import numpy as np
from rich import print
import yaml

BIN_FOLDER = 'bin'
CYTHON_GEN_FOLDER = './cython_generated'
DEPENDENCIES = "dependencies.yaml"

old_dir = os.getcwd()
packageDir = "./quilt"
includedDir = [".", packageDir, np.get_include()]

try:
    with open(DEPENDENCIES, "r") as dependencies:
        dependencies = yaml.safe_load(dependencies)
except FileNotFoundError:
    dependencies = dict()
    pass

os.chdir(packageDir)

# Creates directory for generated files if not existing
if not os.path.exists(CYTHON_GEN_FOLDER):
    os.mkdir(CYTHON_GEN_FOLDER)

extension_kwargs = dict( 
        include_dirs=includedDir,
        language="c++",
        libraries=["m"],                       # Unix-like specific link to C math libraries
        extra_compile_args=["-fopenmp", "-O3"],# Links OpenMP for parallel computing
        extra_link_args=["-fopenmp"],
        define_macros= [('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION')] #Silences npy deprecated warn
        )

cython_compiler_directives = get_directive_defaults()
cython_compiler_directives['language_level'] = "3"
cython_compiler_directives['warn'] = True

extensions = []
for extension_name in dependencies.keys():
    print(dependencies[extension_name])

    current_extension_kwargs = extension_kwargs.copy()
    current_extension_kwargs['include_dirs'] = dependencies[extension_name]['include_dirs']
    extensions.append(Extension(extension_name, dependencies[extension_name]['sources'], **extension_kwargs))

print(extensions)
ext_modules = cythonize(extensions, 
                        nthreads=8,
                        compiler_directives=cython_compiler_directives,
                        include_path=["."],
                        build_dir = CYTHON_GEN_FOLDER,
                        force=False,
                        annotate=False
)
print(ext_modules)

setup(  name='quilt',
        cmdclass={"build_ext": build_ext},
        include_dirs=includedDir,
        ext_modules=ext_modules,
        script_args=["build_ext", f"--build-lib=./{BIN_FOLDER}"],
        options={"build_ext": {"inplace": True, "force": True, "parallel":True}},
)

os.chdir(old_dir)