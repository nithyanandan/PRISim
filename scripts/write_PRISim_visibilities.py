#!python

import yaml
import argparse
import numpy as NP
from prisim import interferometry as RI
import ipdb as PDB

def save(simobj, outfile, outformats, parmsfile=None):
    parms = None
    for outfmt in outformats:
        if outfmt.lower() == 'hdf5':
            simobj.save(outfile, fmt=outfmt, verbose=True, tabtype='BinTableHDU', npz=False, overwrite=True, uvfits_parms=None)
        else:
            if parmsfile is None:
                parmsfile = simobj.simparms_file
                
            if parms is None:
                with open(parmsfile, 'r') as pfile:
                    parms = yaml.safe_load(pfile)
            
            uvfits_parms = None
            if outfmt.lower() == 'uvfits':
                if parms['save_formats']['phase_center'] is None:
                    phase_center = simobj.pointing_center[0,:].reshape(1,-1)
                    phase_center_coords = simobj.pointing_coords
                    if phase_center_coords == 'dircos':
                        phase_center = GEOM.dircos2altaz(phase_center, units='degrees')
                        phase_center_coords = 'altaz'
                    if phase_center_coords == 'altaz':
                        phase_center = GEOM.altaz2hadec(phase_center, simobj.latitude, units='degrees')
                        phase_center_coords = 'hadec'
                    if phase_center_coords == 'hadec':
                        phase_center = NP.hstack((simobj.lst[0]-phase_center[0,0], phase_center[0,1]))
                        phase_center_coords = 'radec'
                    if phase_center_coords != 'radec':
                        raise ValueError('Invalid phase center coordinate system')
                        
                    uvfits_ref_point = {'location': phase_center.reshape(1,-1), 'coords': 'radec'}
                else:
                    uvfits_ref_point = {'location': NP.asarray(parms['save_formats']['phase_center']).reshape(1,-1), 'coords': 'radec'}

                # Phase the visibilities to a phase reference point
                simobj.rotate_visibilities(uvfits_ref_point)
                uvfits_parms = {'ref_point': None, 'datapool': None, 'method': None}

            simobj.pyuvdata_write(outfile, formats=[outfmt.lower()], uvfits_parms=uvfits_parms, overwrite=True)

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to save PRIS?im visibilities')
    
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
    if parmsfile is None:
        parmsfile = simobj.simparms_file

    with open(parmsfile, 'r') as pfile:
        parms = yaml.safe_load(pfile)

    # The following "if" statement is to allow previous buggy saved versions
    # of HDF5 files that did not save the projected_baselines attribute in the
    # right shape when n_acc=1
    
    update_projected_baselines = False
    if simobj.projected_baselines.ndim != 3:
        update_projected_baselines = True
    else:
        if simobj.projected_baselines.shape[2] != simobj.n_acc:
            update_projected_baselines = True

    if update_projected_baselines:
        uvw_ref_point = None
        if parms['save_formats']['phase_center'] is None:
            phase_center = simobj.pointing_center[0,:].reshape(1,-1)
            phase_center_coords = simobj.pointing_coords
            if phase_center_coords == 'dircos':
                phase_center = GEOM.dircos2altaz(phase_center, units='degrees')
                phase_center_coords = 'altaz'
            if phase_center_coords == 'altaz':
                phase_center = GEOM.altaz2hadec(phase_center, simobj.latitude, units='degrees')
                phase_center_coords = 'hadec'
            if phase_center_coords == 'hadec':
                phase_center = NP.hstack((simobj.lst[0]-phase_center[0,0], phase_center[0,1]))
                phase_center_coords = 'radec'
            if phase_center_coords != 'radec':
                raise ValueError('Invalid phase center coordinate system')
                
            uvw_ref_point = {'location': phase_center.reshape(1,-1), 'coords': 'radec'}
        else:
            uvw_ref_point = {'location': NP.asarray(parms['save_formats']['phase_center']).reshape(1,-1), 'coords': 'radec'}
        
        simobj.project_baselines(uvw_ref_point)

    save(simobj, outfile, outformats, parmsfile=parmsfile)

    wait_after_run = args['wait']
    if wait_after_run:
        PDB.set_trace()
    
