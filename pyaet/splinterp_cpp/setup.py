from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'splinterp_cpp',
        ['pybind_wrapper.cpp'],
        include_dirs=[
            # pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
        extra_link_args=[],
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = ['-std=c++11', '-O3']
        if sys.platform == 'darwin':
            opts.append('-mmacosx-version-min=10.9')
        elif ct == 'unix' and sys.platform != 'darwin':
            pass
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='splinterp_cpp',
    version='0.1',
    author='Your Name',
    description='C++ implementation of splinterp with pybind11',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
