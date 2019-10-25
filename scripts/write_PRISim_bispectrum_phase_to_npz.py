#!python

import yaml, argparse
import numpy as NP
import prisim
from prisim import bispectrum_phase as BSP

prisim_path = prisim.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to extract bispectrum phases and save to output file for further processing')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/ioparms/model_bispectrum_phase_to_npz_parms.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    dirinfo = parms['dirStruct']
    indir = dirinfo['indir']
    infile_prefix = dirinfo['infile_prfx']
    infmt = dirinfo['infmt']
    simdir = dirinfo['prisim_dir']
    simfile_prefix = dirinfo['simfile_prfx']
    if infmt.lower() != 'hdf5':
        if (simdir is None) or (simfile_prefix is None):
            raise TypeError('Inputs prisim_dir and simfile_prfx must both be specified')
        if not isinstance(simdir, str):
            raise TypeError('Input simdir must be a string')
        if not isinstance(simfile_prefix, str):
            raise TypeError('Input simfile_prefix must be a string')
        hdf5file_prefix = simdir + simfile_prefix
    else:
        hdf5file_prefix = None

    outdir = dirinfo['outdir']
    outfile_prefix = dirinfo['outfile_prfx']

    procparms = parms['proc']
    reftriad = NP.asarray(procparms['bltriplet'])
    blltol = procparms['blltol']
    datakey = procparms['datakey']
    triads = procparms['triads']

    BSP.write_PRISim_bispectrum_phase_to_npz(indir+infile_prefix, outdir+outfile_prefix, triads=triads, bltriplet=reftriad, hdf5file_prefix=hdf5file_prefix, infmt=infmt, datakey=datakey, blltol=blltol)
