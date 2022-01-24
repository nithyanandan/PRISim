from __future__ import print_function
import pprint, yaml
import numpy as NP
from pyuvdata import UVData
from astroutils import geometry as GEOM
import prisim
from prisim import interferometry as RI

prisim_path = prisim.__path__[0]+'/'

def replicate(parms=None):
    if (parms is None) or (not isinstance(parms, dict)):
        example_yaml_filepath = prisim_path+'examples/simparms/replicatesim.yaml'
        print('\nInput parms must be specified as a dictionary in the format below. Be sure to check example with detailed descriptions in {0}\n'.format(example_yaml_filepath))

        with open(example_yaml_filepath, 'r') as parms_file:
            example_parms = yaml.safe_load(parms_file)
        pprint.pprint(example_parms)
        print('-----------------------\n')
        raise ValueError('Current input parameters insufficient or incompatible to proceed with.')
    else:
        indir = parms['dirstruct']['indir']
        infile = parms['dirstruct']['infile']
        infmt = parms['dirstruct']['infmt']
        outdir = parms['dirstruct']['outdir']
        outfile = parms['dirstruct']['outfile']
        outfmt = parms['dirstruct']['outfmt']
    
        if infmt.lower() not in ['hdf5', 'uvfits']:
            raise ValueError('Input simulation format must be "hdf5" or "uvfits"')
        
        if outfmt.lower() not in ['npz', 'uvfits']:
            raise ValueError('Output simulation format must be "npz" or "uvfits"')
    
        if infmt.lower() == 'uvfits':
            if outfmt.lower() != 'uvfits':
                warnings.warn('Forcing output format to "uvfits" since input format is in "uvfits"')
                outfmt = 'uvfits'
    
        if infmt.lower() == 'hdf5':
            simvis = RI.InterferometerArray(None, None, None, init_file=indir+infile)
            freqs = simvis.channels
            nchan = freqs.size
            df = simvis.freq_resolution
            t_acc = NP.asarray(simvis.t_acc)
            ntimes = t_acc.shape[-1]
            dt = NP.mean(t_acc)
            nbl = simvis.baseline_lengths.size
            data_array = simvis.skyvis_freq
        else:
            uvd = UVData()
            uvd.read_uvfits(indir+infile+'.'+infmt)
            freqs = uvd.freq_array.ravel()
            df = uvd.channel_width
            nbl = uvd.Nbls
            t_acc = uvd.integration_time.reshape(-1,nbl)
            dt = NP.mean(t_acc[:,0])
            nchan = freqs.size
            ntimes = t_acc.shape[0]
            data_array = NP.transpose(uvd.data_array[:,0,:,0].reshape(ntimes, nbl, nchan), (1,2,0))
    
        if outfmt.lower() == 'uvfits':
            if infmt.lower() == 'uvfits':
                uvdummy = UVData()
                uvdummy.read_uvfits(indir+infile+'.'+infmt)
    
        Tsys = parms['telescope']['Tsys']
        if Tsys is None:
            Trx = parms['telescope']['Trx']
            Tant_freqref = parms['telescope']['Tant_freqref']
            Tant_ref = parms['telescope']['Tant_ref']
            Tant_spindex = parms['telescope']['Tant_spindex']
            Tsys = Trx + Tant_ref * (freqs/Tant_freqref)**Tant_spindex
    
        Tsys = NP.asarray(Tsys).reshape(1,-1,1)
        A_eff = parms['telescope']['A_eff']
        eff_aprtr = parms['telescope']['eff_aprtr']
        A_eff *= eff_aprtr
        eff_Q = parms['telescope']['eff_Q']
    
        replicate_info = parms['replicate']
        n_avg = replicate_info['n_avg']
        n_realize = replicate_info['n_realize']
        seed = replicate_info['seed']
        if seed is None:
            seed = NP.random.random_integers(100000)
    
        noiseRMS = RI.thermalNoiseRMS(A_eff, df, dt, Tsys, nbl=nbl, nchan=nchan, ntimes=ntimes, flux_unit='Jy', eff_Q=eff_Q)
        noiseRMS = noiseRMS[NP.newaxis,:,:,:] # (1,nbl,nchan,ntimes)
        rstate = NP.random.RandomState(seed)
        noise = noiseRMS / NP.sqrt(2.0*n_avg) * (rstate.randn(n_realize, nbl, nchan, ntimes) + 1j * rstate.randn(n_realize, nbl, nchan, ntimes)) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts
    
        if outfmt.lower() == 'npz':
            outfilename = outdir + outfile + '_{0:03d}-{1:03d}.{2}'.format(1,n_realize,outfmt.lower())
            outarray = data_array[NP.newaxis,...] + noise
            NP.savez(outfilename, noiseless=data_array[NP.newaxis,...], noisy=outarray, noise=noise)
        else:
            for i in range(n_realize):
                outfilename = outdir + outfile + '-{0:03d}'.format(i+1)
                outarray = data_array + noise[i,...]
                if infmt.lower() == 'uvfits':
                    outfilename = outfilename + '-noisy.{0}'.format(outfmt.lower())
                    uvdummy.data_array = NP.transpose(NP.transpose(outarray, (2,0,1)).reshape(nbl*ntimes, nchan, 1, 1), (0,2,1,3)) # (Nbls, Nfreqs, Ntimes) -> (Ntimes, Nbls, Nfreqs) -> (Nblts, Nfreqs, Nspws=1, Npols=1) -> (Nblts, Nspws=1, Nfreqs, Npols=1)
                    uvdummy.write_uvfits(outfilename, force_phase=True, spoof_nonessential=True)
                else:
                    simvis.vis_freq = outarray
        
                    phase_center = simvis.pointing_center[0,:].reshape(1,-1)
                    phase_center_coords = simvis.pointing_coords
                    if phase_center_coords == 'dircos':
                        phase_center = GEOM.dircos2altaz(phase_center, units='degrees')
                        phase_center_coords = 'altaz'
                    if phase_center_coords == 'altaz':
                        phase_center = GEOM.altaz2hadec(phase_center, simvis.latitude, units='degrees')
                        phase_center_coords = 'hadec'
                    if phase_center_coords == 'hadec':
                        phase_center = NP.hstack((simvis.lst[0]-phase_center[0,0], phase_center[0,1]))
                        phase_center_coords = 'radec'
                    if phase_center_coords != 'radec':
                        raise ValueError('Invalid phase center coordinate system')
                        
                    uvfits_ref_point = {'location': phase_center.reshape(1,-1), 'coords': 'radec'}
                    simvis.rotate_visibilities(uvfits_ref_point)
                    simvis.write_uvfits(outfilename, uvfits_parms={'ref_point': None, 'method': None, 'datapool': ['noisy']}, overwrite=True, verbose=True)
