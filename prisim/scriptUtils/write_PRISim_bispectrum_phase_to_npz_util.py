import pprint, yaml
import numpy as NP
from prisim import bispectrum_phase as BSP
import prisim

prisim_path = prisim.__path__[0]+'/'

def write(parms=None):
    if (parms is None) or (not isinstance(parms, dict)):
        example_yaml_filepath = prisim_path+'examples/ioparms/model_bispectrum_phase_to_npz_parms.yaml'
        print('\nInput parms must be specified as a dictionary in the format below. Be sure to check example with detailed descriptions in {0}\n'.format(example_yaml_filepath))

        with open(example_yaml_filepath, 'r') as parms_file:
            example_parms = yaml.safe_load(parms_file)
        pprint.pprint(example_parms)
        print('-----------------------\n')
        raise ValueError('Current input parameters insufficient or incompatible to proceed with.')
    else:
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
