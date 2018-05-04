#!python

import yaml, argparse
import prisim
from prisim import reflection_constraints as RC
import ipdb as PDB

prisim_path = prisim.__path__[0]+'/'

if __name__ == '__main__':
    
    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to estimate reflection constraints')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'prisim/examples/postprocess/reflection_constraints.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    parms = RC.parsefile(args['infile'])
    RC.estimate(parms, action='save')

    wait_after_run = parms['diagnose']['wait_after_run']
    if wait_after_run:
        PDB.set_trace()
