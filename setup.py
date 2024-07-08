from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np
import os

# Cartella contenente la libreria compilata
LIBRARY_PATH = "libquilt.so"

# Configurazione base
BIN_FOLDER = './interface'
CYTHON_GEN_FOLDER = './cython_generated'
DEFAULT_INCLUDES = ["./", "./quilt", np.get_include()]

# Cartella corrente
old_dir = os.getcwd()
packageDir = "./quilt"
os.chdir(packageDir)

# Classe custom build_ext per gestire la compilazione con libreria personalizzata
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Aggiungi qui le opzioni extra se necessario

# Lista delle estensioni Cython
extensions = [
    Extension(
        "base",  # Nome del modulo Cython
        sources=["interface/base.pyx"],
        include_dirs=DEFAULT_INCLUDES + ["core/include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-std=c++17", LIBRARY_PATH],  # Linka la libreria qui
    ),
    Extension(
        "spiking",
        sources=["interface/spiking.pyx"],
        include_dirs=DEFAULT_INCLUDES + ["core/include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-std=c++17", LIBRARY_PATH],  # Linka la libreria qui
    ),
    Extension(
        "oscill",
        sources=["interface/oscill.pyx"],
        include_dirs=DEFAULT_INCLUDES + ["core/include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-std=c++17", LIBRARY_PATH],  # Linka la libreria qui
    ),
    Extension(
        "multiscale",
        sources=["interface/multiscale.pyx"],
        include_dirs=DEFAULT_INCLUDES + ["core/include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-std=c++17", LIBRARY_PATH],  # Linka la libreria qui
    ),
]

# Compila le estensioni Cython
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

# Torna alla cartella originale
os.chdir(old_dir)
