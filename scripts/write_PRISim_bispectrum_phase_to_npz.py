#!python

import yaml, argparse
import prisim
from prisim.scriptUtils import write_PRISim_bispectrum_phase_to_npz_util
import ipdb as PDB

prisim_path = prisim.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to extract bispectrum phases and save to output file for further processing')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/ioparms/model_bispectrum_phase_to_npz_parms.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    write_PRISim_bispectrum_phase_to_npz_util.write(parms)

