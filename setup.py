#!/usr/bin/env python
"""
PyNIT (Python NeuroImaging Toolkit)
"""
from distutils.core import setup
from setuptools import find_packages
import re, io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('pynit/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

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
      install_requires=['jupyter>=1.0.0',
                        'tqdm',
                        'pandas>=0.22.0',
                        'openpyxl',
                        'xlrd',
                        'nibabel',
                        'matplotlib',
                        'seaborn',
                        'scikit-image',
                        'numpy',
                        'scipy',
                        'scikit-learn',
                        'psutil',
                        'pynsp',
                        ],
      scripts=['pynit/bin/checkbrk',
               'pynit/bin/check_reg',
               'pynit/bin/antsSyN',
               'pynit/bin/onesample_ttest',
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
