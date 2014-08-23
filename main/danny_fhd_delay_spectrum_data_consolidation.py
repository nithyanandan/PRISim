import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
import progressbar as PGB
import interferometry as RI
import ipdb as PDB

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
bl, bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True) 
neg_bl_orientation_ind = bl_orientation < 0.0
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
bl_id = bl_id[sortind]
n_bins_baseline_orientation = 4
nmax_baselines = 2000
bl = bl[:nmax_baselines,:]
bl_length = bl_length[:nmax_baselines]
bl_id = bl_id[:nmax_baselines]
bl_orientation = bl_orientation[:nmax_baselines]
total_baselines = bl_length.size

fhd_obsid = [1061309344, 1061316544]

freq = 185.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 80e3 # in Hz
bpass_shape = 'bhw'
n_channels = 384
nchan = n_channels

ant1 = []
ant2 = []
for bli in bl_id:
    ants = bli.split('-')
    ant1 += [ants[1]]
    ant2 += [ants[0]]

ant1 = NP.asarray(ant1)
ant2 = NP.asarray(ant2)

for j in range(len(fhd_obsid)):
    fhd_infile = '/home/t_nithyanandan/codes/others/python/Danny/{0:0d}'.format(fhd_obsid[j])+'.fhd.p.npz'
    fhd_outfile = '/data3/t_nithyanandan/project_MWA/fhd_delay_spectrum_{0:0d}_reformatted.npz'.format(fhd_obsid[j])
    fhd_data = NP.load(fhd_infile)
    fhd_ant1 = fhd_data['ant1']
    fhd_ant2 = fhd_data['ant2']
    fhd_C = fhd_data['C']
    # fhd_C[fhd_C == 0.0] = 1.0
    fhd_delays = NP.fft.fftshift(fhd_data['delays']) * 1e-9
    fhd_uvw = fhd_data['uvws'] * 1e-9 * FCNST.c
    fhd_bl_length = NP.sqrt(NP.sum(fhd_data['uvws']**2, axis=1)) * 1e-9 * FCNST.c
    fhd_vis_lag_noisy = None
    for p in range(fhd_data['P'].shape[2]):
        fhd_bl_id = []
        vis_lag_noisy = None
        vis_lag_res = None
        progressbl = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=ant1.size).start()
        for k in range(ant1.size):
            blind = NP.logical_and(fhd_ant1 == int(ant1[k]), fhd_ant2 == int(ant2[k]))
            if NP.sum(blind):
                fhd_bl_id += [ant2[k]+'-'+ant1[k]]
                if vis_lag_noisy is None:
                    vis_lag_noisy = fhd_data['P'][blind,:,p].reshape(1,nchan,1)
                    vis_lag_res = fhd_data['P_res'][blind,:,p].reshape(1,nchan,1)
                else:
                    vis_lag_noisy = NP.vstack((vis_lag_noisy, fhd_data['P'][blind,:,p].reshape(1,nchan,1)))
                    vis_lag_res = NP.vstack((vis_lag_res, fhd_data['P_res'][blind,:,p].reshape(1,nchan,1)))
            progressbl.update(k+1)
        progressbl.finish()
        if fhd_vis_lag_noisy is None:
            fhd_vis_lag_noisy = NP.copy(vis_lag_noisy)
            fhd_vis_lag_res = NP.copy(vis_lag_noisy)
        else:
            fhd_vis_lag_noisy = NP.dstack((fhd_vis_lag_noisy, vis_lag_noisy))
            fhd_vis_lag_res = NP.dstack((fhd_vis_lag_res, vis_lag_res))
    fhd_bl_id = NP.asarray(fhd_bl_id)

    NP.savez_compressed(fhd_outfile, fhd_vis_lag_noisy=fhd_vis_lag_noisy, fhd_vis_lag_res=fhd_vis_lag_res, fhd_C=fhd_C, fhd_bl_id=fhd_bl_id, fhd_delays=fhd_delays, fhd_phased_bl=fhd_uvw, fhd_bl_length=fhd_bl_length, pol=NP.arange(2))




    

