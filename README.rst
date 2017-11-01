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

``conda create -n PRISim python=2``

Then install conda packages:

``conda install mpi4py progressbar psutil pyyaml h5py``

You also need ``astroutils``:

``pip install git+https://github.com/nithyanandan/general``

which will install a list of dependencies.

Now do

``pip install aipy``

if it is not installed already.

Finally, either install PRISim directly:

``pip install git+https://github.com/nithyanandan/PRISim``

or clone it into a directory and from inside that directory issue the command:

``pip install .``.


Basic Usage
===========


data_size \propto n_bl * nchan * n_acc
RUN: ``mpirun -n 2 run_prisim.py -i parameterfile.yaml``
