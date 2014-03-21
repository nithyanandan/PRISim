import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.animation as MOV
from scipy.interpolate import griddata
import datetime as DT
import time 
import progressbar as PGB
import healpy as HP
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import ipdb as PDB

## Input/output parameters 

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(1,2,3))
bl = RI.baseline_generator(ant_locs, auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
total_baselines = bl_length.size
# used_baselines = int(NP.sum(bl_length <= 50.0))
baseline_chunk_size = 100
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
# if used_baselines > total_baselines:
#     raise ValueError('Used number of baselines found to exceed total number of baselines')
# bl = bl[:used_baselines,:]
# bl_length = bl_length[:used_baselines]
# bl_orientation = bl_orientation[:used_baselines]
labels = []
labels += ['B'+'{0:0d}'.format(i+1) for i in xrange(bl.shape[0])]

## Observation parameters

telescope = 'mwa'
freq = 150.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 40.0 # in kHz
latitude = -26.701
A_eff = 16.0 * (0.5 * FCNST.c / freq)**2
obs_mode = 'track'
Tsys = 440.0 # in Kelvin
t_snap = 5.0 * 60.0 # in seconds 
t_obs = 2.5 * 3.6e3 # in seconds
# t_obs = 5.0 * 3600.0 # in seconds
pointing_init = [0.0, latitude] # in degrees
lst_init = 0.0 # in hours
n_channels = 256
bpass_shape = 'rect' 
oversampling_factor = 2.0
eff_bw_ratio = 1.0
if bpass_shape == 'bnw':
    eff_bw_ratio = CNST.rect_bnw_ratio 
# n_pad = NP.round(oversampling_factor * n_channels) - NP.round(n_channels * eff_bw_ratio)
# n_pad = NP.round(oversampling_factor * n_channels) - n_channels
n_pad = 0
# elif bpass_shape == 'rect':
#     oversampling_factor = 1.0
# nchan = NP.round(n_channels * oversampling_factor)
nchan = n_channels
base_bpass = 1.0*NP.ones(nchan)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution * 1e3 )/ 1e9 # in GHz
# window = DSP.shaping(nchan, 1/oversampling_factor*eff_bw_ratio, shape=bpass_shape, peak=1.0)
# window = DSP.windowing(NP.round(n_channels * eff_bw_ratio), shape=bpass_shape, pad_width=n_pad, centering=True) 
window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=n_pad, centering=True, area_normalize=True) 
pfb_method = 'theoretical'
if pfb_method == 'empirical':
    bandpass_shape = DSP.PFB_empirical(nchan, 32, 0.25, 0.25)
elif pfb_method == 'theoretical':
    pfbhdulist = fits.open('/data3/t_nithyanandan/project_MWA/foregrounds/PFB/Prabu/test_pfb_512x8.fits')
    pfbdata = pfbhdulist[0].data
    pfbfreq = pfbhdulist[1].data
    pfb_norm = NP.amax(pfbdata, axis=0).reshape(1,-1)
    pfbdata_norm = pfbdata - pfb_norm
    pfbwin = 10 * NP.log10(NP.sum(10**(pfbdata_norm/10), axis=1))
    freq_range = [0.9*chans.min(), 1.1*chans.max()]
    useful_freq_range = NP.logical_and(pfbfreq >= freq_range[0]*1e3, pfbfreq <=freq_range[1]*1e3)
    # pfb_interp_func = interpolate.interp1d(pfbfreq[useful_freq_range]/1e3, pfbwin[useful_freq_range])
    # pfbwin_interp = pfb_interp_func(chans)
    pfbwin_interp = NP.interp(chans, pfbfreq[useful_freq_range]/1e3, pfbwin[useful_freq_range])
    bandpass_shape = 10**(pfbwin_interp/10)

window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=n_pad, centering=True, area_normalize=True) 
bpass = base_bpass * bandpass_shape
n_snaps = int(t_obs/t_snap)
lst = (lst_init + (t_snap/3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
if obs_mode == 'track':
    pointings_radec = NP.repeat(NP.asarray(pointing_init).reshape(-1,2), n_snaps, axis=0)
else:
    pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

## Interferometer parameters

# baseline_orientation = NP.asarray([0.0, 90.0]) # in degrees from East towards North
# baseline_length = 1000.0 + NP.zeros(2) # in m
# baseline_vect = NP.zeros((2,3))
# baseline_vect[:,0] = baseline_length * NP.cos(NP.radians(baseline_orientation))
# baseline_vect[:,1] = baseline_length * NP.sin(NP.radians(baseline_orientation))

# baseline_orientation_str = '{0:.1f}'.format(baseline_orientation)
# baseline_length_str = '{0:.1f}'.format(baseline_length)

# ia = RI.InterferometerArray(labels, bl, chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz')
# # ia = RI.InterferometerArray(['B1','B2'], baseline_vect, chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz')
# # ints = []
# # for i in xrange(baseline_length.size):
# #     ints += [RI.Interferometer('B1', baseline_vect[i,:], chans, telescope='mwa', latitude=latitude, A_eff=A_eff, freq_scale='GHz')]

## Foreground parameters

use_GSM = True  # Global Sky Model
use_DSM = False  # Diffuse Sky Model
use_SUMSS = False  # SUMSS Catalog
use_NVSS = False  # NVSS Catalog
use_MSS = False  # Molonglo Galactic Plane Survey
use_GLEAM = False  # GLEAM Catalog
use_PS = False  # Single Point Source
use_other = False  # Custom Source List

use_FG_model = use_GSM + use_DSM + use_SUMSS + use_NVSS + use_MSS + use_GLEAM + use_PS + use_other
if use_FG_model != 1:
    raise ValueError('One and only one foreground model must be specified.')

fg_str = ''

flux_unit = 'Jy'
freq_catalog = freq/1e9 # in GHz
spindex = 0.0

if use_GSM:
    fg_str = 'asm'

    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata.fits'
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.150 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ra_deg = ra_deg_DSM
    dec_deg = dec_deg_DSM
    majax = NP.degrees(NP.sqrt(pixres*4/NP.pi) * NP.ones(fluxes_DSM.size))
    minax = NP.degrees(NP.sqrt(pixres*4/NP.pi) * NP.ones(fluxes_DSM.size))
    fluxes = fluxes_DSM

    freq_SUMSS = 0.843 # in GHz
    SUMSS_file = '/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt'
    catalog = NP.loadtxt(SUMSS_file, usecols=(0,1,2,3,4,5,10,12,13,14,15,16))
    ra_deg_SUMSS = 15.0 * (catalog[:,0] + catalog[:,1]/60.0 + catalog[:,2]/3.6e3)
    dec_dd = NP.loadtxt(SUMSS_file, usecols=(3,), dtype="|S3")
    sgn_dec_str = NP.asarray([dec_dd[i][0] for i in range(dec_dd.size)])
    sgn_dec = 1.0*NP.ones(dec_dd.size)
    sgn_dec[sgn_dec_str == '-'] = -1.0
    dec_deg_SUMSS = sgn_dec * (NP.abs(catalog[:,3]) + catalog[:,4]/60.0 + catalog[:,5]/3.6e3)
    fmajax = catalog[:,7]
    fminax = catalog[:,8]
    fpa = catalog[:,9]
    dmajax = catalog[:,10]
    dminax = catalog[:,11]
    PS_ind = NP.logical_and(dmajax == 0.0, dminax == 0.0)
    ra_deg_SUMSS = ra_deg_SUMSS[PS_ind]
    dec_deg_SUMSS = dec_deg_SUMSS[PS_ind]
    fint = catalog[PS_ind,6] * 1e-3
    spindex_SUMSS = -0.83 + NP.zeros(fint.size)
    fmajax = fmajax[PS_ind]
    fminax = fminax[PS_ind]
    fpa = fpa[PS_ind]
    dmajax = dmajax[PS_ind]
    dminax = dminax[PS_ind]
    bright_source_ind = fint >= 10.0 * (freq_SUMSS*1e9/freq)**spindex_SUMSS
    ra_deg_SUMSS = ra_deg_SUMSS[bright_source_ind]
    dec_deg_SUMSS = dec_deg_SUMSS[bright_source_ind]
    fint = fint[bright_source_ind]
    fmajax = fmajax[bright_source_ind]
    fminax = fminax[bright_source_ind]
    fpa = fpa[bright_source_ind]
    dmajax = dmajax[bright_source_ind]
    dminax = dminax[bright_source_ind]
    spindex_SUMSS = spindex_SUMSS[bright_source_ind]
    valid_ind = NP.logical_and(fmajax > 0.0, fminax > 0.0)
    ra_deg_SUMSS = ra_deg_SUMSS[valid_ind]
    dec_deg_SUMSS = dec_deg_SUMSS[valid_ind]
    fint = fint[valid_ind]
    fmajax = fmajax[valid_ind]
    fminax = fminax[valid_ind]
    fpa = fpa[valid_ind]
    spindex_SUMSS = spindex_SUMSS[valid_ind]
    freq_catalog = NP.concatenate((freq_catalog, freq_SUMSS*1e9 + NP.zeros(fint.size)))
    catlabel = NP.concatenate((catlabel, NP.repeat('SUMSS', fint.size)))
    ra_deg = NP.concatenate((ra_deg, ra_deg_SUMSS))
    dec_deg = NP.concatenate((dec_deg, dec_deg_SUMSS))
    spindex = NP.concatenate((spindex, spindex_SUMSS))
    majax = NP.concatenate((majax, fmajax/3.6e3))
    minax = NP.concatenate((minax, fminax/3.6e3))
    fluxes = NP.concatenate((fluxes, fint))

    nvss_file = '/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits'
    freq_NVSS = 1.4 # in GHz
    hdulist = fits.open(nvss_file)
    ra_deg_NVSS = hdulist[1].data['RA(2000)']
    dec_deg_NVSS = hdulist[1].data['DEC(2000)']
    nvss_fpeak = hdulist[1].data['PEAK INT']
    nvss_majax = hdulist[1].data['MAJOR AX']
    nvss_minax = hdulist[1].data['MINOR AX']
    spindex_NVSS = -0.83 + NP.zeros(nvss_fpeak.size)
    not_in_SUMSS_ind = NP.logical_and(dec_deg_NVSS > -30.0, dec_deg_NVSS <= min(90.0, latitude+90.0))
    bright_source_ind = nvss_fpeak >= 10.0 * (freq_NVSS*1e9/freq)**(spindex_NVSS)
    PS_ind = NP.sqrt(nvss_majax**2-(0.75/60.0)**2) < 10.0/3.6e3
    count_valid = NP.sum(NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind))
    nvss_fpeak = nvss_fpeak[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]
    freq_catalog = NP.concatenate((freq_catalog, freq_NVSS*1e9 + NP.zeros(count_valid)))
    catlabel = NP.concatenate((catlabel, NP.repeat('NVSS',count_valid)))
    ra_deg = NP.concatenate((ra_deg, ra_deg_NVSS[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    dec_deg = NP.concatenate((dec_deg, dec_deg_NVSS[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    spindex = NP.concatenate((spindex, spindex_NVSS[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    majax = NP.concatenate((majax, nvss_majax[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    minax = NP.concatenate((minax, nvss_minax[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    fluxes = NP.concatenate((fluxes, nvss_fpeak))

    ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    # ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex)

    hdulist.close()
elif use_DSM:
    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata.fits'
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    majax = NP.degrees(NP.sqrt(pixres*4/NP.pi) * NP.ones(fluxes.size))
    minax = NP.degrees(NP.sqrt(pixres*4/NP.pi) * NP.ones(fluxes.size))
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.150 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    fg_str = 'dsm'
    hdulist.close()
elif use_SUMSS:
    SUMSS_file = '/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt'
    catalog = NP.loadtxt(SUMSS_file, usecols=(0,1,2,3,4,5,10,12,13,14,15,16))
    ra_deg = 15.0 * (catalog[:,0] + catalog[:,1]/60.0 + catalog[:,2]/3.6e3)
    dec_dd = NP.loadtxt(SUMSS_file, usecols=(3,), dtype="|S3")
    sgn_dec_str = NP.asarray([dec_dd[i][0] for i in range(dec_dd.size)])
    sgn_dec = 1.0*NP.ones(dec_dd.size)
    sgn_dec[sgn_dec_str == '-'] = -1.0
    dec_deg = sgn_dec * (NP.abs(catalog[:,3]) + catalog[:,4]/60.0 + catalog[:,5]/3.6e3)
    fmajax = catalog[:,7]
    fminax = catalog[:,8]
    fpa = catalog[:,9]
    dmajax = catalog[:,10]
    dminax = catalog[:,11]
    PS_ind = NP.logical_and(dmajax == 0.0, dminax == 0.0)
    ra_deg = ra_deg[PS_ind]
    dec_deg = dec_deg[PS_ind]
    fint = catalog[PS_ind,6] * 1e-3
    fmajax = fmajax[PS_ind]
    fminax = fminax[PS_ind]
    fpa = fpa[PS_ind]
    dmajax = dmajax[PS_ind]
    dminax = dminax[PS_ind]
    bright_source_ind = fint >= 1.0
    ra_deg = ra_deg[bright_source_ind]
    dec_deg = dec_deg[bright_source_ind]
    fint = fint[bright_source_ind]
    fmajax = fmajax[bright_source_ind]
    fminax = fminax[bright_source_ind]
    fpa = fpa[bright_source_ind]
    dmajax = dmajax[bright_source_ind]
    dminax = dminax[bright_source_ind]
    valid_ind = NP.logical_and(fmajax > 0.0, fminax > 0.0)
    ra_deg = ra_deg[valid_ind]
    dec_deg = dec_deg[valid_ind]
    fint = fint[valid_ind]
    fmajax = fmajax[valid_ind]
    fminax = fminax[valid_ind]
    fpa = fpa[valid_ind]
    freq_catalog = 0.843 # in GHz
    spindex = -0.83 + NP.zeros(fint.size)
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcsec','arcsec','degree'])    
    fg_str = 'sumss'
elif use_MSS:
    pass
elif use_GLEAM:
    catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv' # GLEAM catalog
    catdata = ascii.read(catalog_file, data_start=1, delimiter=',')
    dec_deg = catdata['DEJ2000']
    ra_deg = catdata['RAJ2000']
    fpeak = catdata['S150_fit']
    ferr = catdata['e_S150_fit']
    spindex = catdata['Sp+Index']
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex)
    fg_str = 'gleam'
elif use_PS:
    n_src = 1
    fpeak = 1000.0*NP.ones(n_src)
    spindex = NP.ones(n_src) * spindex
    ra_deg = NP.asarray(pointings_radec[0,0])
    dec_deg = NP.asarray(pointings_radec[0,1])
    fmajax = NP.ones(n_src)
    fminax = fmajax
    fpa = NP.zeros(n_src)
    ctlgobj = CTLG.Catalog('PS', freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    fg_str = 'point'
elif use_other:
    n_src = 2
    fpeak = 1000.0 * NP.ones(n_src)
    spindex = NP.ones(n_src) * spindex
    ra_deg = pointings_radec[0,0] + NP.arange(n_src) * 14.5
    dec_deg = NP.ones(n_src) * pointings_radec[0,1]
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex)
    fg_str = 'other'

skymod = CTLG.SkyModel(ctlgobj)

## Animation parameters

backdrop_xsize = 100
bitrate = 128
fps = 1.0
interval = 100

## Start the observation

# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=1).start()
# for i in range(0,min(1,len(baseline_bin_indices))):
#     outfile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
#     ia = RI.InterferometerArray(labels[baseline_bin_indices[i]:min(baseline_bin_indices[i]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[i]:min(baseline_bin_indices[i]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz')    
#     ts = time.time()
#     ia.observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit, memsave=True)
#     print 'The last chunk of {0:0d} baselines required {1:.1f} minutes'.format(baseline_chunk_size, (time.time()-ts)/60.0)
#     ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
#     ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
#     progress.update(i+1)
# progress.finish()

# lags = None
# skyvis_lag = None
# vis_lag = None
# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=20).start()
# for i in range(0, min(20,len(baseline_bin_indices))):
#     infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
#     hdulist = fits.open(infile+'.fits')
#     # extnames = [hdu.header['EXTNAME'] for hdu in hdulist]
#     if i == 0:
#         lags = hdulist['SPECTRAL INFO'].data.field('lag')
#         vis_lag = hdulist['real_lag_visibility'].data + 1j * hdulist['imag_lag_visibility'].data
#         skyvis_lag = hdulist['real_lag_sky_visibility'].data + 1j * hdulist['imag_lag_sky_visibility'].data
#     else:
#         vis_lag = NP.vstack((vis_lag, hdulist['real_lag_visibility'].data + 1j * hdulist['imag_lag_visibility'].data))
#         skyvis_lag = NP.vstack((skyvis_lag, hdulist['real_lag_sky_visibility'].data + 1j * hdulist['imag_lag_sky_visibility'].data))
#     hdulist.close()
#     progress.update(i+1)
# progress.finish()

# small_delays_ind = NP.abs(lags) <= 2.5e-6
# lags = lags[small_delays_ind]
# vis_lag = vis_lag[:,small_delays_ind,:]
# skyvis_lag = skyvis_lag[:,small_delays_ind,:]

# ## Delay limits estimation

delay_matrix = DLY.delay_envelope(bl, pointings_dircos, units='mks')

## Binning baselines by orientation

n_bins_baseline_orientation = 8
blo = bl_orientation[:min(20*baseline_chunk_size, total_baselines)]
blo[blo < -0.5*360.0/n_bins_baseline_orientation] = 360.0 - NP.abs(blo[blo < -0.5*360.0/n_bins_baseline_orientation])
bloh, bloe, blon, blori = OPS.binned_statistic(blo, statistic='count', bins=n_bins_baseline_orientation, range=[(-0.5*360.0/n_bins_baseline_orientation, 360.0-0.5*360.0/n_bins_baseline_orientation)])

# # ia.observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit)
# # print 'Elapsed time = {0:.1f} minutes'.format((time.time()-ts)/60.0)
# # ia.delay_transform()

# for i in xrange(baseline_length.size):
#     ts = time.time()
#     ints[i].observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit)
#     print 'Elapsed time = {0:.1f} minutes'.format((time.time()-ts)/60.0)

# lags = DSP.downsampler(ia.lags, oversampling_factor)
# vis_lag = DSP.downsampler(ia.vis_lag, oversampling_factor, axis=1)
# skyvis_lag = DSP.downsampler(ia.skyvis_lag, oversampling_factor, axis=1)

## Foreground model backdrop

backdrop_coords = 'dircos'
if use_DSM or use_GSM:
    backdrop_coords = 'radec'

if backdrop_coords == 'radec':
    xmin = 0.0
    xmax = 360.0
    ymin = -90.0
    ymax = 90.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
    xvect = xgrid.ravel()
    yvect = ygrid.ravel()
elif backdrop_coords == 'dircos':
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize))
    nanind = (xgrid**2 + ygrid**2) > 1.0
    goodind = (xgrid**2 + ygrid**2) <= 1.0
    zgrid = NP.empty_like(xgrid)
    zgrid[nanind] = NP.nan
    zgrid[goodind] = NP.sqrt(1.0 - (xgrid[goodind]**2 + ygrid[goodind]**2))

    xvect = xgrid.ravel()
    yvect = ygrid.ravel()
    zvect = zgrid.ravel()
    xyzvect = NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1), zvect.reshape(-1,1)))

if use_DSM or use_GSM:
    backdrop = HP.cartview(fluxes_DSM.ravel(), coord=['G','E'], xsize=backdrop_xsize, return_projected_map=True)
elif use_GLEAM or use_SUMSS:
    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
else:
    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)

## Create data for overlay 

overlays = []
roi_obj_inds = []
for i in xrange(n_snaps):
    overlay = {}
    if backdrop_coords == 'radec':
        havect = lst[i] - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
        pb = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', phase_center=pointings_altaz[i,:])
        overlay['pbeam'] = pb
        overlay['roi_obj_inds'] = roi_altaz
        # roi_obj_inds += [roi_altaz]
    elif backdrop_coords == 'dircos':
        havect = lst[i] - ra_deg
        fg_altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),dec_deg.reshape(-1,1))), latitude, units='degrees')
        fg_dircos = GEOM.altaz2dircos(fg_altaz, units='degrees')
        roi_dircos = NP.asarray(NP.where(fg_dircos[:,2] >= 0.0)).ravel()
        overlay['roi_obj_inds'] = roi_dircos
        overlay['fg_dircos'] = fg_dircos
        if obs_mode == 'track':
            pb = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, skyunits='dircos', freq_scale='Hz', phase_center=pointings_dircos[i,:])
            # pb[pb < 0.5] = NP.nan
            overlay['pbeam'] = pb.reshape(backdrop_xsize, backdrop_xsize)
    overlays += [overlay]

## Animation set up

if n_bins_baseline_orientation == 4:
    blo_ax_mapping = [6,2,4,8]
elif n_bins_baseline_orientation == 8:
    blo_ax_mapping = [6,3,2,1,4,7,8,9]

# fig = PLT.figure(figsize=(12,9))

# axs = []
# for i in range(n_bins_baseline_orientation):
#     ax = fig.add_subplot(3,3,blo_ax_mapping[i])
#     ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[i+1]), weight='semibold')
#     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
#     ax.set_xlim(0,bloh[i]-1)
#     # ax.set_ylim(NP.amin(lags*1e6), NP.amax(lags*1e6))
#     ax.set_ylim(0.0, NP.amax(lags*1e6)) 
#     l = ax.plot([], [], 'k-', [], [], 'k-', [], [], 'k:', [], [], 'k:', [], [])
#     axs += [ax]

# ax = fig.add_subplot(3,3,5)
# if backdrop_coords == 'radec':
#     ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=12)
#     ax.set_ylabel(r'$\delta$ [degrees]', fontsize=12)
# elif backdrop_coords == 'dircos':
#     ax.set_xlabel('l')
#     ax.set_ylabel('m')
# ax.set_title('Sky Model', fontsize=18, weight='semibold')
# ax.grid(True)
# ax.tick_params(which='major', length=12, labelsize=12)
# ax.tick_params(which='minor', length=6)

# if use_DSM or use_GSM:
#     linit = ax.imshow(OPS.reverse(backdrop, axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
#     # cbmn = NP.amin(backdrop)
#     # cbmx = NP.amax(backdrop)
#     # cbaxes = fig.add_axes([0.85, 0.1, 0.02, 0.23]) 
#     # cbar = fig.colorbar(linit, cax=cbaxes)
#     # cbmd = 10.0**(0.5*(NP.log10(cbmn)+NP.log10(cbmx)))
#     # cbar.set_ticks([cbmn, cbmd, cbmx])
#     # cbar.set_ticklabels([cbmn, cbmd, cbmx])
# else:
#     ax.set_xlim(NP.amin(xvect), NP.amax(xvect))
#     ax.set_ylim(NP.amin(yvect), NP.amax(yvect))
#     if backdrop_coords == 'radec':
#         linit = ax.scatter(ra_deg, dec_deg, c=fpeak, marker='.', cmap=PLT.cm.get_cmap("rainbow"), norm=PLTC.LogNorm())
#         # cbmn = NP.amin(fpeak)
#         # cbmx = NP.amax(fpeak)
#     else:
#         if (obs_mode == 'drift') or (telescope == 'mwa_dipole'):
#             linit = ax.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
#             # cbaxes = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
#             # cbar = fig.colorbar(linit, cax=cbaxes)

# l = ax.plot([], [], 'w.', [], [])
# # txt = ax.text(0.25, 0.65, '', transform=ax.transAxes, fontsize=18)

# axs += [ax]
# tpc = axs[-1].text(0.5, 1.15, '', transform=ax.transAxes, fontsize=12, weight='semibold', ha='center')

# PLT.tight_layout()
# fig.subplots_adjust(bottom=0.1)

# def update(i, pointing_radec, lst, obsmode, telescope, backdrop_coords, bll, blori, lags, vis_lag, delaymatrix, overlays, xv, yv, xv_uniq, yv_uniq, axs, tpc):

#     delay_ranges = NP.dstack((delaymatrix[:,:vis_lag.shape[0],1] - delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],1] + delaymatrix[:,:vis_lag.shape[0],0]))
#     delay_horizon = NP.dstack((-delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],0]))
#     bl = bll[:vis_lag.shape[0]]

#     label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i])

#     for j in range(len(axs)-1):
#         blindp = blori[blori[j]:blori[j+1]]
#         blindn = blori[blori[(j+len(axs)/2)%len(axs)]:blori[(j+len(axs)/2)%len(axs)+1]]
#         blind = NP.concatenate((blindp, blindn))
#         sortind = NP.argsort(bl[blind], kind='heapsort')
#         dh = 1e6 * NP.concatenate((delay_horizon[i,blindp,1], -delay_horizon[i,blindn,0]))
#         dh = dh[sortind]
#         dr = 1e6 * NP.concatenate((delay_ranges[i,blindp,1], -delay_ranges[i,blindn,0]))
#         dr = dr[sortind]
#         data = NP.empty((blind.size,NP.ceil(0.5*vis_lag.shape[1])))
#         data[:blindp.size,:] = NP.abs(vis_lag[blindp,NP.floor(0.5*vis_lag.shape[1]):,i])
#         data[blindp.size:,:] = OPS.reverse(NP.abs(vis_lag[blindn,NP.floor(0.5*vis_lag.shape[1])-NP.ceil(0.5*vis_lag.shape[1])+1:NP.floor(0.5*vis_lag.shape[1])+1,i]), axis=1)
#         data = NP.take(data, sortind, axis=0)
#         axs[j].lines[0].set_xdata(NP.arange(blind.size))
#         axs[j].lines[0].set_ydata(dr)
#         axs[j].lines[0].set_linewidth(0.5)
#         # axs[j].lines[1].set_xdata(NP.arange(blind.size))
#         # axs[j].lines[1].set_ydata(delay_ranges[i,blind[sortind],1]*1e6)
#         # axs[j].lines[1].set_linewidth(0.5)
#         axs[j].lines[2].set_xdata(NP.arange(blind.size))
#         axs[j].lines[2].set_ydata(dh)
#         axs[j].lines[2].set_linewidth(0.5)
#         # axs[j].lines[3].set_xdata(NP.arange(blind.size))
#         # axs[j].lines[3].set_ydata(delay_horizon[i,blind[sortind],1]*1e6)
#         # axs[j].lines[3].set_linewidth(0.5)
#         axs[j].lines[4] = axs[j].imshow(data.T, origin='lower', extent=(0, blind.size-1, 0.0, NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(data), vmax=NP.amax(data)), interpolation=None)
#         # axs[j].lines[4] = axs[j].imshow(NP.abs(vis_lag[blind[sortind],:,i].T), origin='lower', extent=(0, blind.size-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(vis_lag)), vmax=NP.amax(NP.abs(vis_lag))), interpolation=None)
#         axs[j].set_aspect('auto')

#     cbax = fig.add_axes([0.175, 0.05, 0.7, 0.02])
#     cbar = fig.colorbar(axs[0].lines[4], cax=cbax, orientation='horizontal')
#     cbax.set_xlabel('Jy Hz', labelpad=-1, fontsize=18)

#     if backdrop_coords == 'radec':
#         pbi = griddata(NP.hstack((xv[overlays[i]['roi_obj_inds']].reshape(-1,1),yv[overlays[i]['roi_obj_inds']].reshape(-1,1))), overlays[i]['pbeam'], NP.hstack((xv.reshape(-1,1),yv.reshape(-1,1))), method='nearest')
#         axc = axs[-1]
#         cntr = axc.contour(OPS.reverse(xv_uniq), yv_uniq, OPS.reverse(pbi.reshape(yv_uniq.size, xv_uniq.size), axis=1), 35)
#         axc.set_aspect(1.5)
#         axs[-1] = axc

#         tpc.set_text(r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i]))

#     elif backdrop_coords == 'dircos':
#         if (obsmode != 'drift') and (telescope != 'mwa_dipole'):
#             axs[-1].lines[1] = axs[-1].imshow(overlays[i]['pbeam'], origin='lower', extent=(NP.amin(xv_uniq), NP.amax(xv_uniq), NP.amin(yv_uniq), NP.amax(yv_uniq)), norm=PLTC.LogNorm())
#             # cbaxes3 = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
#             # cbar3 = fig.colorbar(axs[-1].lines[1], cax=cbaxes3)
#         axs[-1].lines[0].set_xdata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],0])
#         axs[-1].lines[0].set_ydata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],1])
#         axs[-1].lines[0].set_marker('.')

#     return axs

# anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lst, obs_mode, telescope, backdrop_coords, bl_length, blori, lags, skyvis_lag, delay_matrix, overlays, xvect, yvect, xgrid[0,:], ygrid[:,0], axs, tpc), frames=len(overlays), interval=interval, blit=False)
# PLT.show()
# animation_file = '/data3/t_nithyanandan/project_MWA/multi_baseline_noiseless_visibilities_'+obs_mode+'_'+'{0:0d}'.format(20*baseline_chunk_size)+'_baselines_{0:0d}_orientations_'.format(n_bins_baseline_orientation)+'gaussian_FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)
# # anim.save(animation_file+'.mp4', fps=fps, codec='x264')



