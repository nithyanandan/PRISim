#!python

import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import yaml, argparse, warnings
from astropy.io import fits
import h5py
import prisim
from prisim import interferometry as RI
from prisim import delay_spectrum as DS
import ipdb as PDB

prisim_path = prisim.__path__[0]+'/'

if __name__ == '__main__':
    
    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to analyze closure phases')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'prisim/examples/postprocess/reflection_constraints.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    projectdir = parms['dirstruct']['projectdir']
    subdir = projectdir + parms['dirstruct']['subdir']
    simdir = subdir + parms['dirstruct']['simdir'] + 'simdata/'
    simfile = simdir + 'simvis'
    analysisdir = subdir + parms['dirstruct']['analysisdir']
    outfile = analysisdir + parms['dirstruct']['outfile']
    figdir = subdir + parms['dirstruct']['figdir']

    processinfo = parms['processing']
    subbandinfo = processinfo['subband']
    freq_window_centers = {key: NP.asarray(subbandinfo['freq_center']) for key in ['cc', 'sim']}
    freq_window_bw = {key: NP.asarray(subbandinfo['bw_eff']) for key in ['cc', 'sim']}
    freq_window_shape = {key: subbandinfo['shape'] for key in ['cc', 'sim']}
    freq_window_fftpow = {key: subbandinfo['fftpow'] for key in ['cc', 'sim']}
    pad = {key: 1.0 for key in ['cc', 'sim']}
    redshifts = NP.asarray(processinfo['redshifts'])
    kprll_cut = NP.asarray(processinfo['kprll_cut'])
    bll = processinfo['bllength']
    max_delay = processinfo['max_delay']
    if max_delay is not None:
        if not isinstance(max_delay, (int,float)):
            raise TypeError('Input max_delay must be a scalar')
        max_delay *= 1e-6 # in seconds from microseconds
        if max_delay < 0.0:
            max_delay = None
        
    lst_range = processinfo['lst_range']
    if lst_range is not None:
        lst_range = NP.asarray(lst_range) # in hours
        lst_range = lst_range.reshape(-1,2)

    lidz_model = parms['eorparm']['lidz_model']
    model_21cmfast = parms['eorparm']['21cmfast_model']
    if lidz_model:
        lidz_modelfile = parms['eorparm']['lidz_modelfile']
    if model_21cmfast:
        modelfile_21cmfast = parms['eorparm']['21cmfast_modelfile']

    if lidz_model:
        hdulist = fits.open(lidz_modelfile)
        eor_model_freqs = hdulist['FREQUENCY'].data
        lidz_eor_model_redshifts = hdulist['REDSHIFT'].data
        eor_modelinfo = [hdulist['{0}'.format(i)].data for i in range(eor_model_freqs.size)]
        lidz_eor_model_k = NP.asarray([modelinfo[:,0] for modelinfo in eor_modelinfo])
        lidz_eor_model_Pk = [modelinfo[:,1] for modelinfo in eor_modelinfo]
        lidz_eor_model_Pk = NP.asarray(lidz_eor_model_Pk)
    
    if model_21cmfast:
        hdulist = fits.open(modelfile_21cmfast)
        eor_model_freqs = hdulist['FREQUENCY'].data
        eor_21cmfast_model_redshifts = hdulist['REDSHIFT'].data
        eor_modelinfo = [hdulist['{0}'.format(i)].data for i in range(eor_model_freqs.size)]
        eor_21cmfast_model_k = NP.asarray([modelinfo[:,0] for modelinfo in eor_modelinfo])
        eor_21cmfast_model_Pk = [modelinfo[:,1] for modelinfo in eor_modelinfo]
        eor_21cmfast_model_Pk = NP.asarray(eor_21cmfast_model_Pk)

    sim = RI.InterferometerArray(None, None, None, init_file=simfile)
    dsobj = DS.DelaySpectrum(interferometer_array=sim)
    dspec = dsobj.delay_transform(action='store')
    subband_DSpec = dsobj.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow , pad=pad, bpcorrect=False, action='return_oversampled')
    dpsobj = DS.DelayPowerSpectrum(dsobj)
    dpsobj.compute_power_spectrum()

    attninfo = {'lidz': {}, '21cmfast': {}}
    for key in attninfo:
        if key == 'lidz':
            eormodel = {'z': lidz_eor_model_redshifts, 'k': lidz_eor_model_k, 'PS': lidz_eor_model_Pk}
        elif key == '21cmfast':
            eormodel = {'z': eor_21cmfast_model_redshifts, 'k': eor_21cmfast_model_k, 'PS': eor_21cmfast_model_Pk}
        else:
            raise ValueError('Specified EoR model not currently supported')
        attninfo[key] = DS.reflection_attenuation(dpsobj, eormodel, redshifts, kprll_cut, bll, max_delay=max_delay, lst_range=lst_range)

    with h5py.File(outfile, 'w') as fobj:
        for key in attninfo:
            for subkey in ['z', 'kprll_cut', 'delays', 'attenuation']:
                dset = fobj.create_dataset('{0}/{1}'.format(key,subkey), data=attninfo[key][subkey])
                if subkey == 'kprll_cut':
                    dset.attrs['units'] = 'h/Mpc'
                elif subkey == 'delays':
                    dset.attrs['units'] = 's'
                elif subkey == 'attenuation':
                    dset.attrs['units'] = 'dB'

    wait_after_run = parms['diagnose']['wait_after_run']
    if wait_after_run:
        PDB.set_trace()
