#!python

import yaml, argparse
import numpy as NP
import prisim
from prisim import interferometry as RI
import write_PRISim_visibilities as PRISimWriter
import ipdb as PDB

prisim_path = prisim.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to update noise in PRISim outputs')

    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-s', '--simfile', dest='simfile', type=str, required=True, help='HDF5 file from PRISim simulation')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', default=None, type=str, required=True, help='File specifying simulation parameters')

    output_group = parser.add_argument_group('Output parameters', 'Output specifications')
    output_group.add_argument('-o', '--outfile', dest='outfile', default=None, type=str, required=True, help='Output File with redundant measurements')
    output_group.add_argument('--outfmt', dest='outfmt', default=['hdf5'], type=str, required=True, nargs='*', choices=['HDF5', 'hdf5', 'UVFITS', 'uvfits', 'UVH5', 'uvh5'], help='Output file format')

    noise_parms_group = parser.add_argument_group('Noise parameters', 'Noise specifications')
    noise_parms_group.add_argument('-n', '--noise_parmsfile', dest='noise_parmsfile', default=prisim_path+'examples/simparms/noiseparms.yaml', type=file, required=True, help='File specifying noise parameters for updating noise in PRISim output')
    
    misc_group = parser.add_argument_group('Misc parameters', 'Misc specifications')
    misc_group.add_argument('-w', '--wait', dest='wait', action='store_true', help='Wait after run')
    
    args = vars(parser.parse_args())

    outfile = args['outfile']
    outformats = args['outfmt']
    parmsfile = args['parmsfile']

    simobj = RI.InterferometerArray(None, None, None, init_file=args['simfile'])

    # The following "if" statement is to allow previous buggy saved versions
    # of HDF5 files that did not save the projected_baselines attribute in the
    # right shape when n_acc=1
    
    if simobj.projected_baselines.ndim != 3:
        if simobj.projected_baselines.ndim == 2:
            simobj.projected_baselines = simobj.projected_baselines[...,NP.newaxis] # (nbl,nchan) --> (nbl,nchan,ntimes=1)
        else:
            raise ValueError('Attribute projected_baselines of PRISim object has incompatible dimensions')

    freqs = simobj.channels
    nchan = freqs.size
    df = simobj.freq_resolution
    t_acc = NP.asarray(simobj.t_acc)
    ntimes = t_acc.shape[-1]
    dt = NP.mean(t_acc)
    nbl = simobj.baseline_lengths.size

    noise_parmsfile = args['noise_parmsfile']
    with args['noise_parmsfile'] as noise_parmsfile:
        noise_parms = yaml.safe_load(noise_parmsfile)

    Tsys = noise_parms['Tsys']
    Trx = noise_parms['Trx']
    Tant_freqref = noise_parms['Tant_freqref']
    Tant_ref = noise_parms['Tant_ref']
    Tant_spindex = noise_parms['Tant_spindex']
    Tsysinfo = {'Trx': Trx, 'Tant':{'f0': Tant_freqref, 'spindex': Tant_spindex, 'T0': Tant_ref}, 'Tnet': Tsys}
    if Tsys is None:
        Tsys_arr = Trx + Tant_ref * (freqs/Tant_freqref)**Tant_spindex

    parmsfile = args['parmsfile']
    with open(parmsfile, 'r') as pfile:
        parms = yaml.safe_load(pfile)

    parms['telescope']['Tsys'] = noise_parms['Tsys']
    parms['telescope']['Trx'] = noise_parms['Trx']
    parms['telescope']['Tant_freqref'] = noise_parms['Tant_freqref']
    parms['telescope']['Tant_ref'] = noise_parms['Tant_ref']
    parms['telescope']['Tant_spindex'] = noise_parms['Tant_spindex']

    Tsys_arr = NP.asarray(Tsys_arr).reshape(1,-1,1)
    A_eff = noise_parms['A_eff']
    eff_aprtr = noise_parms['eff_aprtr']
    A_eff *= eff_aprtr
    eff_Q = noise_parms['eff_Q']

    noiseRMS = RI.thermalNoiseRMS(A_eff, df, dt, Tsys_arr, nbl=nbl, nchan=nchan, ntimes=ntimes, flux_unit='Jy', eff_Q=eff_Q)
    noise = RI.generateNoise(noiseRMS=noiseRMS, A_eff=None, df=None, dt=None, Tsys=None, nbl=nbl, nchan=nchan, ntimes=ntimes, flux_unit=None, eff_Q=None)

    simobj.Tsysinfo = [Tsysinfo] * ntimes
    simobj.Tsys = Tsys_arr + NP.zeros_like(simobj.Tsys)
    simobj.A_eff = A_eff + NP.zeros_like(simobj.A_eff)
    simobj.eff_Q = eff_Q + NP.zeros_like(simobj.eff_Q)
    simobj.vis_rms_freq = noiseRMS + NP.zeros_like(simobj.vis_rms_freq)
    simobj.vis_noise_freq = noise + NP.zeros_like(simobj.vis_noise_freq)
    simobj.vis_freq = simobj.skyvis_freq + noise

    simobj.simparms_file = parmsfile

    PRISimWriter.save(simobj, outfile, outformats, parmsfile=parmsfile)
    
    with open(parmsfile, 'w') as pfile:
        yaml.dump(parms, pfile, default_flow_style=False)

    wait_after_run = args['wait']
    if wait_after_run:
        PDB.set_trace()
    
