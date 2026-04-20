from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

"""
cd /Users/longyang/Documents/Tongji/dev/pyAET/pyaet/splinterp_cpp
python setup.py build_ext --inplace
"""

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# 定义扩展模块
ext_modules = [
    Extension(
        'splinterp_cpp',
        # 移除splinterp3.cpp，只保留pybind_wrapper.cpp
        ['pybind_wrapper.cpp'],
        include_dirs=[
            # pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        # 移除OpenMP标志
        extra_compile_args=['-std=c++11', '-O3'],
        # 移除OpenMP链接标志
        extra_link_args=[],
    ),
]

# 设置编译命令类
class BuildExt(build_ext):
    def build_extensions(self):
        # 添加编译标志以支持C++11
        ct = self.compiler.compiler_type
        opts = ['-std=c++11', '-O3']
        # 根据平台添加不同的编译选项
        if sys.platform == 'darwin':  # macOS
            # 可以添加一些macOS特定的优化选项
            opts.append('-mmacosx-version-min=10.9')
        elif ct == 'unix' and sys.platform != 'darwin':
            # 对于其他Unix系统（如Linux），可以考虑添加OpenMP支持
            # 但由于我们的代码使用的是C++标准线程，实际上不需要OpenMP
            pass
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

# 设置setup参数
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