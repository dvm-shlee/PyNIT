#!/usr/bin/env python
"""
PyNIT (Python NeuroImaging Toolkit)
"""
from distutils.core import setup
from setuptools import find_packages

__version__ = '0.1.3'
__author__ = 'SungHo Lee'
__email__ = 'shlee@unc.edu'
__url__ = 'https://dvm-shlee.github.io'

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
                        'numpy', 'scipy', 'scikit-learn'                            #computation
                        ],
      scripts=['pynit/bin/brk2nifti',
               'pynit/bin/checkscans',
               'pynit/bin/check_reg',
               'pynit/bin/antsSyN',
               'pynit/bin/onesample_ttest'
               ],
      classifiers=[
            # How mature is this project? Common values are
            #  3 - Alpha
            #  4 - Beta
            #  5 - Production/Stable
            'Development Status :: 4 - Beta',

            # Indicate who your project is intended for
            'Framework :: Jupyter',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Natural Language :: English',

            # Specify the Python version you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both
            'Programming Language :: Python :: 2.7',
      ],
      keywords = 'Python NeuroImaging Toolkit'
     )

