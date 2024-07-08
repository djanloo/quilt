from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import os

# Path to the compiled library
LIBRARY_PATH = "libquilt.so"

# Package
PACKAGE_DIR = "./quilt"

# Base configs
BIN_FOLDER = './interface'
CYTHON_GEN_FOLDER = './cython_generated'
DEFAULT_INCLUDES = ["./", 
                    "./quilt", 
                    "./core/include", 
                    np.get_include()]

# Cartella corrente
old_dir = os.getcwd()
os.chdir(PACKAGE_DIR)

extension_names = ["base", "spiking", "oscill", "multiscale"]
extension_common_kwargs = dict( include_dirs=DEFAULT_INCLUDES + ["core/include"],
                                language="c++",
                                extra_compile_args=["-O3", "-std=c++17"],
                                extra_link_args=["-std=c++17", LIBRARY_PATH])

extensions = []
for ext_name in extension_names:
    extensions += [Extension(ext_name, sources=[f"{BIN_FOLDER}/{ext_name}.pyx"], **extension_common_kwargs)]
    
# Compiles the extensions
ext_modules = cythonize(
    extensions,
    nthreads=4,
    include_path=["."],
    build_dir=CYTHON_GEN_FOLDER,
    force=False,
    annotate=False
)

# Configurazione di setup.py
setup(
    name='quilt',
    packages=['quilt'],
    cmdclass={"build_ext": build_ext},
    include_dirs=DEFAULT_INCLUDES,
    ext_modules=ext_modules,
    script_args=["build_ext", f"--build-lib=./{BIN_FOLDER}"],
    options={"build_ext": {"inplace": False, "force": True, "parallel": True}},
)

# Goes back to the root folder
os.chdir(old_dir)
