# -*- coding:utf8 -*-
# File   : setup.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/27/16 19:37
# 
# This file is part of Kaleido

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        "kaleido.opr_kernel.cnn",
        ["kaleido/opr_kernel/cnn.pyx"],
        extra_compile_args=["-Wno-unused-function"],
        include_dirs = [numpy_include]
    ),
]

setup(
    name='Kaleido',
    author='Jiayuan Mao',
    author_email='maojiayuan@gmail.com',
    packages=['kalendo'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
