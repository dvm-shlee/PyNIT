#!/usr/bin/env python
"""
PyNIT (Python NeuroImaging Toolkit)
"""
from distutils.core import setup
from setuptools import find_packages

__version__ = '0.1.1'
__author__ = 'SungHo Lee'
__email__ = 'shlee@unc.edu'
__url__ = 'https://dvm-shlee.github.io/pynit'

setup(name='PyNIT',
      version=__version__,
      description='Python NeuroImaging Toolkit',
      author=__author__,
      author_email=__email__, 
      url=__url__,
      license='GNLv3',
      packages=find_packages(),
      install_requires=['jupyter', 'tqdm', 'pandas', 'openpyxl', 'xlrd', 'nibabel', #platform
                        'matplotlib', 'seaborn', 'scikit-image',                    #visulaization
                        'numpy', 'scipy', 'scikit-learn', 'bctpy', 'statsmodels'    #computation
                        ],
      classifier=[
            # How mature is this project? Common values are
            #  3 - Alpha
            #  4 - Beta
            #  5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Neuroscience researcher',
            'Topic :: NeuroImaging :: Functional MRI Toolkit',

            # Specify the Python version you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both
            'Programming Language :: Python :: 2.7',
      ],
      keyworks = 'Python NeuroImaging Toolkit'
     )

