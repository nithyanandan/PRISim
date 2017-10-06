PRISim
======

A modular 21cm signal simulator, including foregrounds and instrumental effects.


Installation
============
Note that currently this package only supports Python 2.6+, and not Python 3. 

Using Anaconda
--------------
If using the Anaconda python distribution, many of the packages may be installed using ``conda``.

It is best to first create a new env:

`` conda create -n prisim python=2``

Then install conda packages:

``conda install numpy scipy matplotlib astropy  mpi4py progressbar psutil pyyaml h5py``

Now do

``pip install aipy``.

You also need ``astroutils``:

`` pip install git+git://github.com/nithyanandan/general``

Finally, either install PRISim directly:

`` pip install git+git://github.com/nithyanandan/PRISim``

or in the top-level directory:

``pip install .``.