import yaml
import argparse
import numpy as NP
from prisim import interferometry as RI
import ipdb as PDB

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to duplicate redundant baseline measurements')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-s', '--simfile', dest='simfile', type=str, required=True, help='HDF5 file from PRISim simulation')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', default=None, type=str, required=False, help='File specifying simulation parameters')
    
    args = vars(parser.parse_args())

    simvis = RI.InterferometerArray(None, None, None, init_file=args['simfile'])

    if args['parmsfile'] is not None:
        parmsfile = args['parmsfile']
    else:
        parmsfile = simvis.simparms_file
        
    with parmsfile as pfile:
            parms = yaml.safe_load(pfile)

    
    

