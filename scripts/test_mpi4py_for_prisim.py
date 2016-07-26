#!python

from mpi4py import MPI

## Set MPI parameters

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
name = MPI.Get_processor_name()

if rank == 0:
    print '\n{0} processes initiated...'.format(nproc)

print '\tProcess #{0} completed'.format(rank)
if rank == 0:
    print 'MPI test successful\n'
