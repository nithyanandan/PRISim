import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
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
import primary_beams as PB
import baseline_delay_horizon as DLY

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
baseline_chunk_size = 50
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
t_snap = 5 * 60.0 # in seconds 
t_obs = 2.5*3.6e3 # in seconds
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
bandpass_shape = DSP.PFB_empirical(nchan, 32, 0.25, 0.25)
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

use_GSM = True
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False
use_other = False

use_FG_model = use_GSM + use_SUMSS + use_GLEAM + use_PS + use_other
if use_FG_model != 1:
    raise ValueError('One and only one foreground model must be specified.')

fg_str = ''

flux_unit = 'Jy'
freq_catalog = freq/1e9 # in GHz
spindex = 0.0

if use_GSM:
    gsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata.fits'
    hdulist = fits.open(gsm_file)
    pixres = hdulist[0].header['PIXAREA']
    gsm_table = hdulist[1].data
    ra_deg = gsm_table['RA']
    dec_deg = gsm_table['DEC']
    temperatures = gsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = gsm_table['spindex'] + 2.0
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((NP.sqrt(pixres*4/NP.pi)*NP.ones(fluxes.size).reshape(-1,1),NP.sqrt(pixres*4/NP.pi)*NP.ones(fluxes.size).reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['radian','radian','degree'])
    fg_str = 'gsm'
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
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['degree','degree','degree'])
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

progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=40).start()
for i in range(35,min(40,len(baseline_bin_indices))):
    outfile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
    ia = RI.InterferometerArray(labels[baseline_bin_indices[i]:min(baseline_bin_indices[i]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[i]:min(baseline_bin_indices[i]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz')    
    ts = time.time()
    ia.observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit, memsave=True)
    print 'The last chunk of {0:0d} baselines required {1:.1f} minutes'.format(baseline_chunk_size, (time.time()-ts)/60.0)
    ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
    ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
    progress.update(i+1)
progress.finish()

# lags = None
# skyvis_lag = None
# vis_lag = None
# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=40).start()
# for i in range(0, min(40,len(baseline_bin_indices))):
#     infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
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

# small_delays_ind = NP.abs(lags) <= 2.0e-6
# lags = lags[small_delays_ind]
# vis_lag = vis_lag[:,small_delays_ind,:]
# skyvis_lag = skyvis_lag[:,small_delays_ind,:]

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

# ## Delay limits estimation

# delay_matrix = DLY.delay_envelope(bl, pointings_dircos, units='mks')

# ## Foreground model backdrop

# backdrop_coords = 'dircos'
# if use_GSM:
#     backdrop_coords = 'radec'

# if backdrop_coords == 'radec':
#     xmin = 0.0
#     xmax = 360.0
#     ymin = -90.0
#     ymax = 90.0

#     xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
#     xvect = xgrid.ravel()
#     yvect = ygrid.ravel()
# elif backdrop_coords == 'dircos':
#     xmin = -1.0
#     xmax = 1.0
#     ymin = -1.0
#     ymax = 1.0

#     xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize))
#     nanind = (xgrid**2 + ygrid**2) > 1.0
#     goodind = (xgrid**2 + ygrid**2) <= 1.0
#     zgrid = NP.empty_like(xgrid)
#     zgrid[nanind] = NP.nan
#     zgrid[goodind] = NP.sqrt(1.0 - (xgrid[goodind]**2 + ygrid[goodind]**2))

#     xvect = xgrid.ravel()
#     yvect = ygrid.ravel()
#     zvect = zgrid.ravel()
#     xyzvect = NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1), zvect.reshape(-1,1)))

# if use_GSM:
#     backdrop = HP.cartview(fluxes.ravel(), coord=['G','E'], xsize=backdrop_xsize, return_projected_map=True)
# elif use_GLEAM or use_SUMSS:
#     if backdrop_coords == 'radec':
#         backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
#         backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
#     elif backdrop_coords == 'dircos':
#         if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
#             backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
#             backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
# else:
#     if backdrop_coords == 'radec':
#         backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
#         backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
#     elif backdrop_coords == 'dircos':
#         if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
#             backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
#             backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)

# ## Create data for overlay 

# overlays = []
# roi_obj_inds = []
# for i in xrange(n_snaps):
#     overlay = {}
#     if backdrop_coords == 'radec':
#         havect = lst[i] - xvect
#         altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
#         roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
#         pb = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', phase_center=pointings_altaz[i,:])
#         overlay['pbeam'] = pb
#         overlay['roi_obj_inds'] = roi_altaz
#         # roi_obj_inds += [roi_altaz]
#     elif backdrop_coords == 'dircos':
#         havect = lst[i] - ra_deg
#         fg_altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),dec_deg.reshape(-1,1))), latitude, units='degrees')
#         fg_dircos = GEOM.altaz2dircos(fg_altaz, units='degrees')
#         roi_dircos = NP.asarray(NP.where(fg_dircos[:,2] >= 0.0)).ravel()
#         overlay['roi_obj_inds'] = roi_dircos
#         overlay['fg_dircos'] = fg_dircos
#         if obs_mode == 'track':
#             pb = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, skyunits='dircos', freq_scale='Hz', phase_center=pointings_dircos[i,:])
#             overlay['pbeam'] = pb.reshape(backdrop_xsize, backdrop_xsize)
#     overlays += [overlay]

# ## Animation set up

# fig = PLT.figure(figsize=(14,14))

# ax11 = fig.add_subplot(311)
# ax11.set_title('Baseline Vectors', fontsize=18, weight='semibold')
# ax11.set_xlabel('Baseline Index', fontsize=18)
# ax11.set_ylabel('Baseline angle (degrees N of E)', color='r', fontsize=18)
# # ax11.plot(bl_orientation, 'r-', marker='.')
# ax11.plot(bl_orientation[:vis_lag.shape[0]], 'r-', marker='.')
# for tl in ax11.get_yticklabels():
#     tl.set_color('r')
# ax12 = ax11.twinx()
# ax12.plot(bl_length[:vis_lag.shape[0]], 'k.', markersize=2.5)
# ax12.set_ylabel('Baseline Length (m)', color='k', fontsize=18)
# # ax11.set_xlim(0.0, bl.shape[0]-1)
# ax11.set_xlim(0, vis_lag.shape[0]-1)
# ax11.set_ylim(-180, 180)

# ax2 = fig.add_subplot(312)
# ax2.set_xlabel('Baseline Index', fontsize=18)
# ax2.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
# ax2.set_title('Delay Spectrum', fontsize=18, weight='semibold')
# # ax2.set_xlim(0,bl.shape[0]-1)
# ax2.set_xlim(0,vis_lag.shape[0]-1)
# ax2.set_ylim(NP.amin(lags*1e6), NP.amax(lags*1e6))
# l2 = ax2.plot([], [], 'k-', [], [], 'k-', [], [], 'k:', [], [], 'k:', [], [])

# ax3 = fig.add_subplot(313) 
# if backdrop_coords == 'radec':
#     ax3.set_xlabel(r'$\alpha$ [degrees]', fontsize=18)
#     ax3.set_ylabel(r'$\delta$ [degrees]', fontsize=18)
# elif backdrop_coords == 'dircos':
#     ax3.set_xlabel('l', fontsize=18)
#     ax3.set_ylabel('m', fontsize=18)
# ax3.set_title('Sky Model', fontsize=18, weight='semibold')
# ax3.grid(True)
# ax3.tick_params(which='major', length=12, labelsize=18)
# ax3.tick_params(which='minor', length=6)

# if use_GSM:
#     l3init = ax3.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
#     cbmn = NP.amin(backdrop)
#     cbmx = NP.amax(backdrop)
#     cbaxes = fig.add_axes([0.85, 0.1, 0.02, 0.23]) 
#     cbar = fig.colorbar(l3init, cax=cbaxes)
#     cbmd = 10.0**(0.5*(NP.log10(cbmn)+NP.log10(cbmx)))
#     cbar.set_ticks([cbmn, cbmd, cbmx])
#     cbar.set_ticklabels([cbmn, cbmd, cbmx])
# else:
#     ax3.set_xlim(NP.amin(xvect), NP.amax(xvect))
#     ax3.set_ylim(NP.amin(yvect), NP.amax(yvect))
#     if backdrop_coords == 'radec':
#         l3init = ax3.scatter(ra_deg, dec_deg, c=fpeak, marker='.', cmap=PLT.cm.get_cmap("rainbow"), norm=PLTC.LogNorm())
#         # cbmn = NP.amin(fpeak)
#         # cbmx = NP.amax(fpeak)
#     else:
#         if (obs_mode == 'drift') or (telescope == 'mwa_dipole'):
#             l3init = ax3.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
#             cbaxes = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
#             cbar = fig.colorbar(l3init, cax=cbaxes)

# l3 = ax3.plot([], [], 'w.', [], [])

# txt2 = ax2.text(0.05, 0.8, '', transform=ax2.transAxes, fontsize=18)
# txt3 = ax3.text(0.25, 0.65, '', transform=ax3.transAxes, fontsize=18)

# def update(i, pointing_radec, lst, obsmode, telescope, backdrop_coords, lags, vis_lag, delaymatrix, overlays, xv, yv, xv_uniq, yv_uniq, line2, line3, t2, t3):
#     delay_ranges = NP.dstack((delaymatrix[:,:vis_lag.shape[0],1] - delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],1] + delaymatrix[:,:vis_lag.shape[0],0]))
#     delay_horizon = NP.dstack((-delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],0]))

#     label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i])

#     line2[4] = ax2.imshow(NP.abs(vis_lag[:,:,i].T), origin='lower', extent=(0, vis_lag.shape[0]-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm())
#     ax2.set_aspect('auto')
#     line2[0].set_xdata(NP.arange(vis_lag.shape[0]))
#     line2[0].set_ydata(delay_ranges[i,:,0]*1e6)
#     line2[0].set_linewidth(0.5)
#     line2[1].set_xdata(NP.arange(vis_lag.shape[0]))
#     line2[1].set_ydata(delay_ranges[i,:,1]*1e6)
#     line2[1].set_linewidth(0.5)
#     line2[2].set_xdata(NP.arange(vis_lag.shape[0]))
#     line2[2].set_ydata(delay_horizon[i,:,0]*1e6)
#     line2[2].set_linewidth(0.5)
#     line2[3].set_xdata(NP.arange(vis_lag.shape[0]))
#     line2[3].set_ydata(delay_horizon[i,:,1]*1e6)
#     line2[3].set_linewidth(0.5)
#     cbaxes2 = fig.add_axes([0.91, 0.385, 0.02, 0.23])
#     cbar2 = fig.colorbar(line2[4], cax=cbaxes2)

#     if backdrop_coords == 'radec':
#         pbi = griddata(NP.hstack((xv[overlays[i]['roi_obj_inds']].reshape(-1,1),yv[overlays[i]['roi_obj_inds']].reshape(-1,1))), overlays[i]['pbeam'], NP.hstack((xv.reshape(-1,1),yv.reshape(-1,1))), method='nearest')
#         line3[0] = ax3.contour(xv_uniq, yv_uniq, pbi.reshape(yv_uniq.size, xv_uniq.size), 35)
#     elif backdrop_coords == 'dircos':
#         if (obsmode != 'drift') and (telescope != 'mwa_dipole'):
#             line3[1] = ax3.imshow(overlays[i]['pbeam'], origin='lower', extent=(NP.amin(xv_uniq), NP.amax(xv_uniq), NP.amin(yv_uniq), NP.amax(yv_uniq)), norm=PLTC.LogNorm())
#             cbaxes3 = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
#             cbar3 = fig.colorbar(line3[1], cax=cbaxes3)
#         line3[0].set_xdata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],0])
#         line3[0].set_ydata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],1])
#         line3[0].set_marker('.')

#     # if interferometer.pointing_coords == 'hadec':
#     #     label_str = r' $\alpha$ = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.lst[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
#     # elif interferometer.pointing_coords == 'radec':
#     #     label_str = r' HA = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.lst[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
#     t2.set_text(label_str)
#     # # t2.set_text(label_str)
#     t3.set_text('')

#     return line2, line3, t2, t3

# anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lst, obs_mode, telescope, backdrop_coords, lags, skyvis_lag, delay_matrix, overlays, xvect, yvect, xgrid[0,:], ygrid[:,0], l2, l3, txt2, txt3), frames=len(overlays), interval=interval, blit=False)
# PLT.show()
# animation_file = '/data3/t_nithyanandan/project_MWA/multi_baseline_noiseless_visibilities_'+obs_mode+'_'+'{0:0d}'.format(40*baseline_chunk_size)+'_baselines_FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)
# # anim.save(animation_file+'.mp4', fps=fps, codec='x264')


