#!/usr/bin/env python
"""
PyNIT (Python NeuroImaging Toolkit)
"""
from distutils.core import setup
from setuptools import find_packages

__version__ = '0.1.2'
__author__ = 'SungHo Lee'
__email__ = 'shlee@unc.edu'
__url__ = 'https://github.com/dvm-shlee/pynit'

setup(name='PyNIT',
      version=__version__,
      description='Python NeuroImaging Toolkit',
      author=__author__,
      author_email=__email__,
      url= __url__,
      license='GNLv3',
      packages=find_packages(),
      install_requires=['numpy', 'nibabel', 'pandas', 'IPython', 'ipywidgets',
                        'matplotlib', 'seaborn', 'tqdm',
                        'scikit-image', 'xlrd', 'openpyxl']
     )

