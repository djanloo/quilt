import os
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def run(self):
        os.system('meson setup builddir')
        os.system('ninja -C builddir')
        super().run()

setup(
    name='quilt',
    version="0.3.0",
    description='A neural multiscale simulator',
    author='Gianluca Becuzzi',
    author_email='becuzzigianluca@gmail.com',
    ext_modules=[],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
