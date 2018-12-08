PRISim (Precision Radio Interferometer Simulator)
=================================================

A modular 21cm signal simulator, including foregrounds and instrumental effects.


Installation
============
Note that currently this package only supports Python 2.6+, and not Python 3. 

Non-Python Dependencies
-----------------------
The only non-python dependencies required are ``openmpi`` and ``xterm``. These can usually be installed via a distro
package manager (for Arch Linux, the package names are exactly ``openmpi`` and ``xterm``).

Using Anaconda
--------------
If using the Anaconda python distribution, many of the packages may be installed using ``conda``.

It is best to first create a new env:

``conda create -n YOURENV python=2``

Activate this environment:

``source activate YOURENV``

Then install conda packages:

``conda install mpi4py progressbar psutil pyyaml h5py``

You also need ``AstroUtils``:

``pip install git+https://github.com/nithyanandan/AstroUtils``

which will install a list of dependencies.

Now do

``pip install aipy``

if it is not installed already.

Finally, either install PRISim directly:

``pip install git+https://github.com/nithyanandan/PRISim``

or clone it into a directory and from inside that directory issue the command:

``pip install .``

Getting Package Data
--------------------

Find the ``data/`` directory under PRISim installation folder which is usually in

``/path/to/Anaconda/envs/YOURENV/lib/python-2.7/site-packages/prisim/``

Download the contents of  
`PRISim Data <https://drive.google.com/open?id=0Bxl4zmCNSW4tUWxrRFhRQ2l4SDQ>`_

or extract the contents of  
`gzipped PRISim Data <https://drive.google.com/open?id=1u1gDyBhdZPPkf75aeybpuT2r_lDcdurW>`_

and place it under 

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

RUN on terminal: 

``mpirun -n nproc run_prisim.py -i parameterfile.yaml``

or 

``mpirun -n nproc xterm -e run_prisim.py -i parameterfile.yaml``

where, ``nproc`` is the number of processors (say, 4), and use of option 
``xterm -e`` opens an xterm window where you can view the progress of each of the processes.  

Data Size
=========

Data size is proportional to ``n_bl x nchan x n_acc``

