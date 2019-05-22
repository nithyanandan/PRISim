#!python

import yaml
import argparse
import numpy as NP
from prisim import interferometry as RI
import write_PRISim_visibilities as PRISimWriter
import ipdb as PDB

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to duplicate redundant baseline measurements')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-s', '--simfile', dest='simfile', type=str, required=True, help='HDF5 file from PRISim simulation')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', default=None, type=str, required=False, help='File specifying simulation parameters')

    output_group = parser.add_argument_group('Output parameters', 'Output specifications')
    output_group.add_argument('-o', '--outfile', dest='outfile', default=None, type=str, required=True, help='Output File with redundant measurements')
    output_group.add_argument('--outfmt', dest='outfmt', default=['hdf5'], type=str, required=True, nargs='*', choices=['HDF5', 'hdf5', 'UVFITS', 'uvfits', 'UVH5', 'uvh5'], help='Output file format')

    misc_group = parser.add_argument_group('Misc parameters', 'Misc specifications')
    misc_group.add_argument('-w', '--wait', dest='wait', action='store_true', help='Wait after run')
    
    args = vars(parser.parse_args())
    outfile = args['outfile']
    outformats = args['outfmt']
    parmsfile = args['parmsfile']

    simobj = RI.InterferometerArray(None, None, None, init_file=args['simfile'])
    if args['parmsfile'] is not None:
        parmsfile = args['parmsfile']
        with open(parmsfile, 'r') as pfile:
            parms = yaml.safe_load(pfile)

        blinfo = RI.getBaselineInfo(parms)
        bl = blinfo['bl']
        blgroups = blinfo['groups']
        bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
    
        simbl = simobj.baselines
        if simbl.shape[0] == bl.shape[0]:
            simbll = NP.sqrt(NP.sum(simbl**2, axis=1))
            simblo = NP.angle(simbl[:,0] + 1j * simbl[:,1], deg=True)
            simblza = NP.degrees(NP.arccos(simbl[:,2] / simbll))
            
            simblstr = ['{0[0]:.2f}_{0[1]:.3f}_{0[2]:.3f}'.format(lo) for lo in zip(simbll,3.6e3*simblza,3.6e3*simblo)]
        
            inp_blo = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
            inp_blza = NP.degrees(NP.arccos(bl[:,2] / bl_length))
            inp_blstr = ['{0[0]:.2f}_{0[1]:.3f}_{0[2]:.3f}'.format(lo) for lo in zip(bl_length,3.6e3*inp_blza,3.6e3*inp_blo)]
    
            uniq_inp_blstr, inp_ind, inp_invind = NP.unique(inp_blstr, return_index=True, return_inverse=True)  ## if numpy.__version__ < 1.9.0
            uniq_sim_blstr, sim_ind, sim_invind = NP.unique(simblstr, return_index=True, return_inverse=True)  ## if numpy.__version__ < 1.9.0
            # uniq_inp_blstr, inp_ind, inp_invind, inp_frequency = NP.unique(inp_blstr, return_index=True, return_inverse=True, return_counts=True)  ## if numpy.__version__ >= 1.9.0
            # uniq_sim_blstr, sim_ind, sim_invind, sim_frequency = NP.unique(simblstr, return_index=True, return_inverse=True, return_counts=True)  ## if numpy.__version__ >= 1.9.0
    
            if simbl.shape[0] != uniq_sim_blstr.size:
                raise ValueError('Non-redundant baselines already found in the simulations')
            
            if not NP.array_equal(uniq_inp_blstr, uniq_sim_blstr):
                raise ValueError('Layout from input simulation parameters file do not match simulated data.')
            simobj.duplicate_measurements(blgroups=blgroups)
        else:
            raise ValueError('Layout from input simulation parameters file do not match simulated data.')
    else:
        simobj.duplicate_measurements()

    # The following "if" statement is to allow previous buggy saved versions
    # of HDF5 files that did not save the projected_baselines attribute in the
    # right shape when n_acc=1
    
    if simobj.projected_baselines.ndim != 3:
        if simobj.projected_baselines.ndim == 2:
            simobj.projected_baselines = simobj.projected_baselines[...,NP.newaxis] # (nbl,nchan) --> (nbl,nchan,ntimes=1)
        else:
            raise ValueError('Atrribute projected_baselines of PRISim object has incompatible dimensions')

    # if simobj.n_acc == 1:
    #     simobj.projected_baselines = simobj.projected_baselines[...,NP.newaxis] # (nbl,nchan) --> (nbl,nchan,ntimes=1)

    PRISimWriter.save(simobj, outfile, outformats, parmsfile=parmsfile)

    wait_after_run = args['wait']
    if wait_after_run:
        PDB.set_trace()
