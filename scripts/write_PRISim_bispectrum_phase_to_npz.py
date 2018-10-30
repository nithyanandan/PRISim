#!python

import yaml, argparse
import numpy as NP
from prisim import bispectrum_phase as BSP

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to duplicate redundant baseline measurements')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default='/data3/t_nithyanandan/codes/mine/python/projects/closure/model_bispectrum_phase_to_npz_parms.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    dirinfo = parms['dirStruct']
    indir = dirinfo['indir']
    infile_prefix = dirinfo['infile_prfx']
    outdir = dirinfo['outdir']
    outfile_prefix = dirinfo['outfile_prfx']

    procparms = parms['proc']
    reftriad = NP.asarray(procparms['bltriplet'])
    blltol = procparms['blltol']

    BSP.write_PRISim_bispectrum_phase_to_npz(indir+infile_prefix, reftriad, outdir+outfile_prefix, blltol=blltol)
