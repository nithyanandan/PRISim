PRISim (Precision Radio Interferometer Simulator)
=================================================

A modular radio interferometer array simulator, including the radio sky and instrumental effects.


Installation
============
Note that currently this package only supports Python 2.6+, and not Python 3. 

Non-Python Dependencies
-----------------------
The only non-python dependencies are ``openmpi`` and ``xterm``.
These are *not required*, but do provide extra functionality.
These can usually be installed via a distro package manager (eg. for Arch Linux,
the package names are exactly ``openmpi`` and ``xterm``).


Using Anaconda
--------------
If using the Anaconda python distribution, many of the packages may be installed using
``conda``. While these dependencies will be installed automatically with the installation
procedure below (i.e. with pip), usually conda users will want to install these with
conda explicitly before installing ``prisim``.

The conda-appropriate packages can be installed with

``conda install mpi4py progressbar psutil pyyaml h5py astropy matplotlib numpy scipy scikit-image``

    NOTE: at this time, you *must* install ``scikit-image`` via conda, or else it will
    try to install packages that are incompatible with python 2. Full python 3
    support is coming soon.

Finally, either install PRISim directly:

``pip install git+https://github.com/nithyanandan/PRISim``

or clone it into a directory and from inside that directory issue the command:

``pip install .``

Getting Package Data
--------------------

First try using the following (from anywhere on your computer, but inside your env)::

    setup_prisim_data.py

If this does not work, try a manual download, as follows:

Find the ``data/`` directory under PRISim installation folder which is usually in

``/path/to/Anaconda/envs/YOURENV/lib/python-2.7/site-packages/prisim/``

Download the contents of PRISim Data from either
`Google Drive <https://drive.google.com/open?id=0Bxl4zmCNSW4tUWxrRFhRQ2l4SDQ>`_

or the zipped version from
`Zenodo (.zip or tar.gz) <https://doi.org/10.5281/zenodo.3892047>`_
or 
`Google Drive (.tar.gz) <https://drive.google.com/open?id=1KNBk6VhlY_rKSfgn8HmAncLkYQ1KGAOi>`_

Extract the contents of the zipped file and place it under 

``/path/to/Anaconda/envs/YOURENV/lib/python-2.7/site-packages/prisim/data/``

Testing MPI for PRISim
----------------------

On terminal, run:

``mpirun -n 2 test_mpi4py_for_prisim.py``

and you must see a message that the test was successful. Otherwise you will have
to ask your system administrator to install ``openmpi`` 

It has also been noted that it is preferable to install ``mpi4py`` using 

``conda install mpi4py`` 

rather than 

``pip install mpi4py``

because the pip installation seems to get the paths to the ``MPI`` libraries
mixed up.

Basic Usage
===========

Run on terminal:

``mpirun -n nproc run_prisim.py -i parameterfile.yaml``

or 

``mpirun -n nproc xterm -e run_prisim.py -i parameterfile.yaml``

where, ``nproc`` is the number of processors (say, 4), and use of option 
``xterm -e`` opens an xterm window where you can view the progress of each of the processes.  

Data Size
=========

Data size is proportional to ``n_bl x nchan x n_acc``

