import os as _os

__version__='2.2.1'
__description__='Precision Radio Interferometry Simulator'
__author__='Nithyanandan Thyagarajan'
__authoremail__='nithyanandan.t@gmail.com'
__maintainer__='Nithyanandan Thyagarajan'
__maintaineremail__='nithyanandan.t@gmail.com'
__url__='http://github.com/nithyanandan/prisim'

with open(_os.path.dirname(_os.path.abspath(__file__))+'/githash.txt', 'r') as _githash_file:
    __githash__ = _githash_file.readline()
