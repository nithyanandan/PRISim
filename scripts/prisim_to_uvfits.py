#!python

import yaml
import argparse
import numpy as NP
import prisim
from prisim import interferometry as RI

prisim_path = prisim.__path__[0]+'/'

def write(parms, verbose=True):

    if 'infile' not in parms:
        raise KeyError('PRISim input file not specified. See example in {0}examples/ioparms/uvfitsparms.yaml'.format(prisim_path))
    if parms['infile'] is None:
        raise ValueError('PRISim input file not specified. See example in {0}examples/ioparms/uvfitsparms.yaml'.format(prisim_path))
    infile_parsed = parms['infile'].rsplit('.', 1)
    if len(infile_parsed) > 1:
        extn = infile_parsed[-1]
        if extn.lower() in ['hdf5', 'fits']:
            parms['infile'] = '.'.join(infile_parsed[:-1])
    if 'outfile' not in parms:
        parms['outfile'] = parms['infile']
    if parms['outfile'] is None:
        parms['outfile'] = parms['infile']
    if 'phase_center' not in parms:
        raise KeyError('Phase center [ra, dec] (deg) as a numpy array must be specified. See example in {0}examples/ioparms/uvfitsparms.yaml'.format(prisim_path))
    if 'method' not in parms:
        raise KeyError('Key specifying UVFITS method is missing. See example in {0}examples/ioparms/uvfitsparms.yaml'.format(prisim_path))

    if 'overwrite' not in parms:
        parms['overwrite'] = True
    elif not isinstance(parms['overwrite'], bool):
        raise TypeError('Overwrite parameter must be boolean')
    
    ref_point = {'location': NP.asarray(parms['phase_center']).reshape(1,-1), 'coords': 'radec'}
    uvfits_parms = {'ref_point': ref_point, 'method': parms['method']}

    prisimobj = RI.InterferometerArray(None, None, None, init_file=parms['infile'])
    prisimobj.write_uvfits(parms['outfile'], uvfits_parms=uvfits_parms, overwrite=parms['overwrite'], verbose=verbose)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to write PRISim output visibilities in UVFITS format')

    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', type=file, required=True, help='File specifying I/O and UVFITS parameters')
    
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true')

    args = vars(parser.parse_args())
    with args['parmsfile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    write(parms, verbose=args['verbose'])
    
