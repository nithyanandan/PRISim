#!python

import yaml, argparse, copy, warnings
import prisim
from prisim.scriptUtils import replicatesim_util
import ipdb as PDB

prisim_path = prisim.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to replicate simulated interferometer array data')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/simparms/replicatesim.yaml', type=file, required=False, help='File specifying input parameters for replicating PRISim output')
    
    args = vars(parser.parse_args())

    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    if 'wait_before_run' in parms['diagnosis']:
        wait_before_run = parms['diagnosis']['wait_before_run']
    else:
        wait_before_run = False

    if 'wait_after_run' in parms['diagnosis']:
        wait_after_run = parms['diagnosis']['wait_after_run']
    else:
        wait_after_run = False

    if wait_before_run:
        PDB.set_trace()

    # Perform replication
    replicatesim_util.replicate(parms)    

    if wait_after_run:
        PDB.set_trace()
