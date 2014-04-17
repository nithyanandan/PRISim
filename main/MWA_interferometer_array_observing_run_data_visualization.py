import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.animation as MOV
from matplotlib import ticker
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

telescope = 'mwa'
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(1,2,3))
bl = RI.baseline_generator(ant_locs, auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
n_bins_baseline_orientation = 4
n_bl_chunks = 1
baseline_chunk_size = 2000
neg_bl_orientation_ind = bl_orientation < 0.0
# neg_bl_orientation_ind = NP.logical_or(bl_orientation < -0.5*180.0/n_bins_baseline_orientation, bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
total_baselines = bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:baseline_bin_indices[n_bl_chunks],:]
bl_length = bl_length[:baseline_bin_indices[n_bl_chunks]]
bl_orientation = bl_orientation[:baseline_bin_indices[n_bl_chunks]]
neg_bl_orientation_ind = bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation

## Plot distribution of baseline lengths and distributions

bl_length_binsize = 20.0
bl_length_bins = NP.linspace(0.0, NP.ceil(bl_length.max()/bl_length_binsize) * bl_length_binsize, NP.ceil(bl_length.max()/bl_length_binsize)+1)
bl_orientation_binsize=180.0/(2*n_bins_baseline_orientation)
bl_orientation_bins = NP.linspace(bl_orientation.min(), bl_orientation.max(), 2*n_bins_baseline_orientation+1)

fig = PLT.figure(figsize=(6,6))
ax1 = fig.add_subplot(211)
n, bins, patches = ax1.hist(bl_length, bins=bl_length_bins, histtype='step', lw=2, color='black')
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top') 
ax1.set_xlabel('Baseline Length [m]', fontsize=18, weight='medium')
ax1.set_ylabel('Number in bin', fontsize=18, weight='medium')
ax1.tick_params(which='major', length=18, labelsize=12)
ax1.tick_params(which='minor', length=12, labelsize=12)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2)
xticklabels = PLT.getp(ax1, 'xticklabels')
yticklabels = PLT.getp(ax1, 'yticklabels')
PLT.setp(xticklabels, fontsize=15, weight='medium')
PLT.setp(yticklabels, fontsize=15, weight='medium')    

ax2 = fig.add_subplot(212)
n, bins, patches = ax2.hist(bl_orientation, bins=bl_orientation_bins, histtype='step', lw=2, color='black')
ax2.set_xlabel('Baseline Orientation [deg]', fontsize=18, weight='medium')
ax2.set_ylabel('Number in bin', fontsize=18, weight='medium')
ax2.tick_params(which='major', length=18, labelsize=12)
ax2.tick_params(which='minor', length=12, labelsize=12)
for axis in ['top','bottom','left','right']:
    ax2.spines[axis].set_linewidth(2)
xticklabels = PLT.getp(ax2, 'xticklabels')
yticklabels = PLT.getp(ax2, 'yticklabels')
PLT.setp(xticklabels, fontsize=15, weight='medium')
PLT.setp(yticklabels, fontsize=15, weight='medium')    

PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/baseline_properties.eps', bbox_inches=0)
PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/baseline_properties.png', bbox_inches=0)

labels = []
labels += ['B{0:0d}'.format(i+1) for i in xrange(bl.shape[0])]

freq = 150.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 40e3 # in Hz
bpass_shape = 'rect'
f_pad = 1.0
oversampling_factor = 1.0 + f_pad
n_channels = 768
nchan = n_channels
max_abs_delay = 2.5 # in micro seconds

window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=0, centering=True, area_normalize=True) 

obs_mode = 'custom'
avg_drifts = False
avg_drifts_str = ''
if avg_drifts:
    avg_drifts_str = 'drift_averaged_'

nside = 128
use_GSM = False
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = True

if use_GSM:
    fg_str = 'asm'
elif use_DSM:
    fg_str = 'dsm'
elif use_CSM:
    fg_str = 'csm'
elif use_SUMSS:
    fg_str = 'sumss'
elif use_GLEAM:
    fg_str = 'gleam'
elif use_PS:
    fg_str = 'point'
elif use_NVSS:
    fg_str = 'nvss'
else:
    fg_str = 'other'

## Animation set up

backdrop_xsize = 100
fps = 0.5
interval = 100
animation_format = 'MP4'
if animation_format == 'MP4':
    anim_format = '.mp4'
else:
    anim_format = 'gif'
animation_file = None
if animation_file is None:
    animation_file = '/data3/t_nithyanandan/project_MWA/animations/multi_baseline_noiseless_visibilities_'+avg_drifts_str+obs_mode+'_'+'{0:0d}'.format(n_bl_chunks*baseline_chunk_size)+'_baselines_{0:0d}_orientations_'.format(n_bins_baseline_orientation)+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_{0:0d}_sectors'.format(n_bins_baseline_orientation) 

animation2_file = None
if animation2_file is None:
    animation2_file = '/data3/t_nithyanandan/project_MWA/animations/delay_emission_map_'+avg_drifts_str+obs_mode+'_'+'{0:0d}'.format(n_bl_chunks*baseline_chunk_size)+'_baselines_{0:0d}_orientations_'.format(n_bins_baseline_orientation)+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_{0:0d}_sectors'.format(n_bins_baseline_orientation)

lags = None
skyvis_lag = None
vis_lag = None

# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_bl_chunks).start()
# for i in range(0, n_bl_chunks):
#     infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
#     hdulist = fits.open(infile+'.fits')
#     # extnames = [hdu.header['EXTNAME'] for hdu in hdulist]
#     if i == 0:
#         lags = hdulist['SPECTRAL INFO'].data.field('lag')
#         vis_lag = hdulist['real_lag_visibility'].data + 1j * hdulist['imag_lag_visibility'].data
#         skyvis_lag = hdulist['real_lag_sky_visibility'].data + 1j * hdulist['imag_lag_sky_visibility'].data

#         latitude = hdulist[0].header['latitude']
#         pointing_coords = hdulist[0].header['pointing_coords']
#         pointings_table = hdulist['POINTING INFO'].data
#         lst = pointings_table['LST']
#         n_snaps = lst.size
#         if pointing_coords == 'altaz':
#             pointings_altaz = NP.hstack((pointings_table['pointing_latitude'].reshape(-1,1), pointings_table['pointing_longitude'].reshape(-1,1)))
#             pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')
#             pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
#         elif pointing_coords == 'radec':
#             pointings_radec = NP.hstack((pointings_table['pointing_longitude'].reshape(-1,1), pointings_table['pointing_latitude'].reshape(-1,1)))
#             pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
#             pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
#             pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
#         elif pointing_coords == 'hadec':
#             pointings_hadec = NP.hstack((pointings_table['pointing_longitude'].reshape(-1,1), pointings_table['pointing_latitude'].reshape(-1,1)))
#             pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
#             pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
#             pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
#     else:
#         vis_lag = NP.vstack((vis_lag, hdulist['real_lag_visibility'].data + 1j * hdulist['imag_lag_visibility'].data))
#         skyvis_lag = NP.vstack((skyvis_lag, hdulist['real_lag_sky_visibility'].data + 1j * hdulist['imag_lag_sky_visibility'].data))
#     hdulist.close()
#     progress.update(i+1)
# progress.finish()

progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_bl_chunks).start()
for i in range(0, n_bl_chunks):
    infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
    if i == 0:
        ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    

        hdulist = fits.open(infile+'.fits')
        latitude = hdulist[0].header['latitude']
        pointing_coords = hdulist[0].header['pointing_coords']
        pointings_table = hdulist['POINTING AND PHASE CENTER INFO'].data
        lst = pointings_table['LST']
        n_snaps = lst.size
        hdulist.close()
        if pointing_coords == 'altaz':
            pointings_altaz = NP.hstack((pointings_table['pointing_latitude'].reshape(-1,1), pointings_table['pointing_longitude'].reshape(-1,1)))
            pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')
            pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
        elif pointing_coords == 'radec':
            pointings_radec = NP.hstack((pointings_table['pointing_longitude'].reshape(-1,1), pointings_table['pointing_latitude'].reshape(-1,1)))
            pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
            pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
            pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
        elif pointing_coords == 'hadec':
            pointings_hadec = NP.hstack((pointings_table['pointing_longitude'].reshape(-1,1), pointings_table['pointing_latitude'].reshape(-1,1)))
            pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
            pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
            pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    else:
        ia_next = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
        ia.concatenate(ia_next, axis=0)
    progress.update(i+1)
progress.finish()
pc = NP.asarray([90.0, 90.0]).reshape(1,-1)
pc_coords = 'altaz'
# pc = NP.asarray([266.416837, -29.00781]).reshape(1,-1)
# pc_coords = 'radec'
ia.phase_centering(phase_center=pc, phase_center_coords=pc_coords)

#################################################################################        

# Find any negative orientation baselines and conjugate those visibilities

simdata_bl_orientation = NP.angle(ia.baselines[:,0] + 1j * ia.baselines[:,1], deg=True)
simdata_neg_bl_orientation_ind = simdata_bl_orientation < 0.0

ia.baselines[simdata_neg_bl_orientation_ind,:] = -ia.baselines[simdata_neg_bl_orientation_ind,:]
# ia.baseline_orientations[simdata_neg_bl_orientation_ind] = 180.0 + ia.baseline_orientations[simdata_neg_bl_orientation_ind]
ia.vis_freq[simdata_neg_bl_orientation_ind,:,:] = ia.vis_freq[simdata_neg_bl_orientation_ind,:,:].conj()
ia.skyvis_freq[simdata_neg_bl_orientation_ind,:,:] = ia.skyvis_freq[simdata_neg_bl_orientation_ind,:,:].conj()
ia.vis_noise_freq[simdata_neg_bl_orientation_ind,:,:] = ia.vis_noise_freq[simdata_neg_bl_orientation_ind,:,:].conj()

ia.delay_transform(f_pad, freq_wts=window) # delay transform re-estimate
lags = ia.lags
vis_lag = ia.vis_lag
skyvis_lag = ia.skyvis_lag

if max_abs_delay is not None:
    small_delays_ind = NP.abs(lags) <= max_abs_delay * 1e-6
    lags = lags[small_delays_ind]
    vis_lag = vis_lag[:,small_delays_ind,:]
    skyvis_lag = skyvis_lag[:,small_delays_ind,:]

## Delay limits re-estimation

delay_matrix = DLY.delay_envelope(ia.baselines, pointings_dircos, units='mks')

fig = PLT.figure(figsize=(6,8))

ax1 = fig.add_subplot(211)
# ax1.set_xlabel('Baseline Length [m]', fontsize=18)
# ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
# dspec1 = ax1.pcolorfast(bl_length, 1e6*lags, NP.abs(skyvis_lag[:-1,:-1,0].T), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))))
# ax1.set_xlim(bl_length[0], bl_length[-1])
# ax1.set_ylim(1e6*lags[0], 1e6*lags[-1])
ax2.set_xlabel('Baseline Index', fontsize=18)
ax2.set_ylabel(r'lag [$\mu$s]', fontsize=18)
dspec1 = ax1.imshow(NP.abs(skyvis_lag[:,:,0].T), origin='lower', extent=(0, skyvis_lag.shape[0]-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))), interpolation=None)
ax1.set_aspect('auto')

ax2 = fig.add_subplot(212)
# ax2.set_xlabel('Baseline Length [m]', fontsize=18)
# ax2.set_ylabel(r'lag [$\mu$s]', fontsize=18)
# dspec2 = ax2.pcolorfast(bl_length, 1e6*lags, NP.abs(skyvis_lag[:-1,:-1,1].T), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))))
# ax2.set_xlim(bl_length[0], bl_length[-1])
# ax2.set_x=ylim(1e6*lags[0], 1e6*lags[-1])
ax2.set_xlabel('Baseline Index', fontsize=18)
ax2.set_ylabel(r'lag [$\mu$s]', fontsize=18)
dspec2 = ax2.imshow(NP.abs(skyvis_lag[:,:,1].T), origin='lower', extent=(0, skyvis_lag.shape[0]-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))), interpolation=None)
ax2.set_aspect('auto')

cbax = fig.add_axes([0.88, 0.08, 0.03, 0.9])
cb = fig.colorbar(dspec2, cax=cbax, orientation='vertical')
cbax.set_ylabel('Jy', labelpad=-60, fontsize=18)

PLT.tight_layout()
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(left=0.1)

PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_combined_baseline_visibilities_'+avg_drifts_str+obs_mode+'_gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_{0:0d}_snapshots.eps'.format(skyvis_lag.shape[2]), bbox_inches=0)
PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_combined_baseline_visibilities_'+avg_drifts_str+obs_mode+'_gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_{0:0d}_snapshots.png'.format(skyvis_lag.shape[2]), bbox_inches=0)

#################################################################################

backdrop_coords = 'radec'
if use_DSM or use_GSM:
    backdrop_coords = 'radec'

if backdrop_coords == 'radec':
    xmin = -180.0
    xmax = 180.0
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
    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata{0:0d}.fits'.format(nside)
    hdulist = fits.open(dsm_file)
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures
    backdrop = HP.cartview(temperatures.ravel(), coord=['G','E'], rot=[0,0,0], xsize=backdrop_xsize, return_projected_map=True)
elif use_GLEAM or use_SUMSS:
    if use_GLEAM:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv' # GLEAM catalog
        catdata = ascii.read(catalog_file, data_start=1, delimiter=',')
        dec_deg = catdata['DEJ2000']
        ra_deg = catdata['RAJ2000']
        fpeak = catdata['S150_fit']
        ferr = catdata['e_S150_fit']
        freq_catalog = 1.4 # GHz
        spindex = -0.83 + NP.zeros(fpeak.size)
        fluxes = fpeak * (freq_catalog * 1e9 / freq)**spindex
    else:
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
        fluxes = fint * (freq_catalog*1e9/freq)**spindex

    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
else:
    if use_PS:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/PS_catalog.txt'
        catdata = ascii.read(catalog_file, comment='#', header_start=0, data_start=1)
        ra_deg = catdata['RA'].data
        dec_deg = catdata['DEC'].data
        fluxes = catdata['F_INT'].data
        
    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)

## Create data for overlay 

cardinal_blo = 180.0 / n_bins_baseline_orientation * NP.arange(n_bins_baseline_orientation).reshape(-1,1)
cardinal_bll = 100.0
cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

overlays = []
roi_obj_inds = []

for i in xrange(n_snaps):
    overlay = {}
    if backdrop_coords == 'radec':
        havect = lst[i] - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        dircos = GEOM.altaz2dircos(altaz, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()

        pb = NP.empty(xvect.size)
        pb.fill(NP.nan)
        pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', phase_center=pointings_altaz[i,:])
        bd = NP.empty(xvect.size)
        bd.fill(NP.nan)
        bd[roi_altaz] = backdrop.ravel()[roi_altaz]
        overlay['pbeam'] = pb
        overlay['backdrop'] = bd
        overlay['roi_obj_inds'] = roi_altaz
        overlay['delay_map'] = NP.empty((n_bins_baseline_orientation, xvect.size))
        overlay['delay_map'].fill(NP.nan)
        delay_envelope = DLY.delay_envelope(cardinal_bl, dircos[roi_altaz,:])
        overlay['delay_map'][:,roi_altaz] = (DLY.geometric_delay(cardinal_bl, altaz[roi_altaz,:], altaz=True, dircos=False, hadec=False, latitude=latitude)-DLY.geometric_delay(cardinal_bl, pointings_altaz[i,:], altaz=True, dircos=False, hadec=False, latitude=latitude)).T
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
        overlay['delay_map'] = NP.empty((n_bins_baseline_orientation, xyzvect.shape[0])).fill(NP.nan)
    overlays += [overlay]

mnd = [NP.nanmin(olay['delay_map']) for olay in overlays]
mxd = [NP.nanmax(olay['delay_map']) for olay in overlays]
mindelay = min(mnd)
maxdelay = max(mxd)

#################################################################################

# # if n_bins_baseline_orientation == 4:
# #     blo_ax_mapping = [6,9,8,7,4,1,2,3]

# # if n_bins_baseline_orientation == 4:
# #     blo_ax_mapping = [6,2,4,8]
# # elif n_bins_baseline_orientation == 8:
# #     blo_ax_mapping = [6,3,2,1,4,7,8,9]

# # fig = PLT.figure(figsize=(14,14))

# # axs = []
# # for i in range(2*n_bins_baseline_orientation):
# #     ax = fig.add_subplot(3,3,blo_ax_mapping[i])
# #     if i < n_bins_baseline_orientation:
# #         ax.set_xlim(0,bloh[i]-1)
# #         ax.set_ylim(0.0, NP.amax(lags*1e6))
# #     else:
# #         # ax = fig.add_subplot(3,3,blo_ax_mapping[i%n_bins_baseline_orientation])
# #         ax.set_xlim(0,bloh[i%n_bins_baseline_orientation]-1)
# #         ax.set_ylim(NP.amin(lags*1e6), 0.0)

# #     l = ax.plot([], [], 'k-', [], [], 'k:', [], [])
# #     ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i%n_bins_baseline_orientation], bloe[(i%n_bins_baseline_orientation)+1]), weight='semibold')
# #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
# #     # ax.set_aspect('auto')
# #     axs += [ax]

# # ax = fig.add_subplot(3,3,5)
# # if backdrop_coords == 'radec':
# #     ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=12)
# #     ax.set_ylabel(r'$\delta$ [degrees]', fontsize=12)
# # elif backdrop_coords == 'dircos':
# #     ax.set_xlabel('l')
# #     ax.set_ylabel('m')
# # ax.set_title('Sky Model', fontsize=18, weight='semibold')
# # ax.grid(True)
# # ax.tick_params(which='major', length=12, labelsize=12)
# # ax.tick_params(which='minor', length=6)

# # if use_DSM or use_GSM:
# #     # linit = ax.imshow(OPS.reverse(backdrop, axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
# #     linit = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
# #     # cbmn = NP.amin(backdrop)
# #     # cbmx = NP.amax(backdrop)
# #     # cbaxes = fig.add_axes([0.85, 0.1, 0.02, 0.23]) 
# #     # cbar = fig.colorbar(linit, cax=cbaxes)
# #     # cbmd = 10.0**(0.5*(NP.log10(cbmn)+NP.log10(cbmx)))
# #     # cbar.set_ticks([cbmn, cbmd, cbmx])
# #     # cbar.set_ticklabels([cbmn, cbmd, cbmx])
# # else:
# #     ax.set_xlim(NP.amin(xvect), NP.amax(xvect))
# #     ax.set_ylim(NP.amin(yvect), NP.amax(yvect))
# #     if backdrop_coords == 'radec':
# #         linit = ax.scatter(ra_deg, dec_deg, c=fpeak, marker='.', cmap=PLT.cm.get_cmap("rainbow"), norm=PLTC.LogNorm())
# #         # cbmn = NP.amin(fpeak)
# #         # cbmx = NP.amax(fpeak)
# #     else:
# #         if (obs_mode == 'drift') or (telescope == 'mwa_dipole'):
# #             linit = ax.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
# #             # cbaxes = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
# #             # cbar = fig.colorbar(linit, cax=cbaxes)

# # l = ax.plot([], [], 'w.', [], [])
# # # txt = ax.text(0.25, 0.65, '', transform=ax.transAxes, fontsize=18)

# # axs += [ax]
# # tpc = axs[-1].text(0.5, 1.15, '', transform=ax.transAxes, fontsize=12, weight='semibold', ha='center')

# # PLT.tight_layout()
# # fig.subplots_adjust(bottom=0.1)

# # def update(i, pointing_radec, lst, obsmode, telescope, backdrop_coords, bll, blori, lags, vis_lag, delaymatrix, overlays, xv, yv, xv_uniq, yv_uniq, axs, tpc):

# #     delay_ranges = NP.dstack((delaymatrix[:,:vis_lag.shape[0],1] - delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],1] + delaymatrix[:,:vis_lag.shape[0],0]))
# #     delay_horizon = NP.dstack((-delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],0]))
# #     bl = bll[:vis_lag.shape[0]]

# #     label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i])

# #     for j in range((len(axs)-1)/2):
# #         blind = blori[blori[j]:blori[j+1]]
# #         sortind = NP.argsort(bl[blind], kind='heapsort')
# #         axs[j].lines[0].set_xdata(NP.arange(blind.size))
# #         axs[j].lines[0].set_ydata(delay_ranges[i,blind[sortind],1]*1e6)
# #         axs[j].lines[0].set_linewidth(0.5)
# #         axs[j].lines[1].set_xdata(NP.arange(blind.size))
# #         axs[j].lines[1].set_ydata(delay_horizon[i,blind[sortind],1]*1e6)
# #         axs[j].lines[1].set_linewidth(0.5)
# #         axs[j].lines[2] = axs[j].imshow(NP.abs(vis_lag[blind[sortind],NP.floor(0.5*vis_lag.shape[1]):,i].T), origin='lower', extent=(0, blind.size-1, 0.0, NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(vis_lag)), vmax=NP.amax(NP.abs(vis_lag))), interpolation=None)
# #         axs[j].set_aspect('auto')
# #         axs[j+(len(axs)-1)/2].lines[0].set_xdata(NP.arange(blind.size))
# #         axs[j+(len(axs)-1)/2].lines[0].set_ydata(delay_ranges[i,blind[sortind],0]*1e6)
# #         axs[j+(len(axs)-1)/2].lines[0].set_linewidth(0.5)
# #         axs[j+(len(axs)-1)/2].lines[1].set_xdata(NP.arange(blind.size))
# #         axs[j+(len(axs)-1)/2].lines[1].set_ydata(delay_horizon[i,blind[sortind],0]*1e6)
# #         axs[j+(len(axs)-1)/2].lines[1].set_linewidth(0.5)
# #         axs[j+(len(axs)-1)/2].lines[2] = axs[j+(len(axs)-1)/2].imshow(NP.abs(vis_lag[blind[sortind],:NP.floor(0.5*vis_lag.shape[1]),i].T), origin='lower', extent=(0, blind.size-1, NP.amin(lags*1e6), 1e6*lags[NP.floor(0.5*lags.size)-1]), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(vis_lag)), vmax=NP.amax(NP.abs(vis_lag))), interpolation=None)
# #         axs[j+(len(axs)-1)/2].set_aspect('auto')

# #     cbax = fig.add_axes([0.175, 0.05, 0.7, 0.02])
# #     cbar = fig.colorbar(axs[0].lines[2], cax=cbax, orientation='horizontal')
# #     cbax.set_xlabel('Jy Hz', labelpad=-1, fontsize=18)

# #     if backdrop_coords == 'radec':
# #         pbi = griddata(NP.hstack((xv[overlays[i]['roi_obj_inds']].reshape(-1,1),yv[overlays[i]['roi_obj_inds']].reshape(-1,1))), overlays[i]['pbeam'], NP.hstack((xv.reshape(-1,1),yv.reshape(-1,1))), method='nearest')
# #         axc = axs[-1]
# #         cntr = axc.contour(OPS.reverse(xv_uniq), yv_uniq, OPS.reverse(pbi.reshape(yv_uniq.size, xv_uniq.size), axis=1), 35)
# #         axc.set_aspect(1.5)
# #         axs[-1] = axc

# #         tpc.set_text(r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i]))

# #     elif backdrop_coords == 'dircos':
# #         if (obsmode != 'drift') and (telescope != 'mwa_dipole'):
# #             axs[-1].lines[1] = axs[-1].imshow(overlays[i]['pbeam'], origin='lower', extent=(NP.amin(xv_uniq), NP.amax(xv_uniq), NP.amin(yv_uniq), NP.amax(yv_uniq)), norm=PLTC.LogNorm())
# #             # cbaxes3 = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
# #             # cbar3 = fig.colorbar(axs[-1].lines[1], cax=cbaxes3)
# #         axs[-1].lines[0].set_xdata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],0])
# #         axs[-1].lines[0].set_ydata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],1])
# #         axs[-1].lines[0].set_marker('.')

# #     return axs

# # anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lst, obs_mode, telescope, backdrop_coords, bl_length, blori, lags, skyvis_lag, delay_matrix, overlays, xvect, yvect, xgrid[0,:], ygrid[:,0], axs, tpc), frames=len(overlays), interval=interval, blit=False)
# # PLT.show()

# # anim.save(animation_file+anim_format, fps=fps, codec='x264')

# #################################################################################

# # fig2 = PLT.figure(figsize=(15,15))
# # f2axs = []
# # for i in range(n_bins_baseline_orientation):
# #     for j in [1,2]:
# #         ax = fig2.add_subplot(4,2,2*i+j)
# #         if backdrop_coords == 'radec':
# #             ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=12)
# #             ax.set_ylabel(r'$\delta$ [degrees]', fontsize=12)
# #         elif backdrop_coords == 'dircos':
# #             ax.set_xlabel('l')
# #             ax.set_ylabel('m')

# #         if j == 1:
# #             iml = ax.imshow(1e6 * OPS.reverse(overlays[0]['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))            
# #             if i == 0:
# #                 ax.set_title('Delay Map', fontsize=18, weight='semibold')
# #         # elif j == 2:
# #         #     imc = ax.imshow(overlays[j]['pbeam'].reshape(-1,backdrop_xsize) * backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
# #         #     if i == 0:
# #         #         ax.set_title('PB x Foregrounds', fontsize=18, weight='semibold')
# #         else:
# #             imr1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
# #             imr2 = ax.imshow(OPS.reverse((overlays[0]['pbeam']*overlays[0]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85)
# #             # , norm=PLTC.LogNorm(vmin=NP.nanmin(overlays[0]['pbeam']*overlays[0]['backdrop']), vmax=NP.nanmax(overlays[0]['pbeam']*overlays[0]['backdrop']))
# #             if i == 0:
# #                 ax.set_title('Foregrounds', fontsize=18, weight='semibold')
                
# #         ax.grid(True)
# #         ax.tick_params(which='major', length=12, labelsize=12)
# #         ax.tick_params(which='minor', length=6)

# #         f2axs += [ax]

# # cbmnl = NP.nanmin(overlays[0]['delay_map']) * 1e6
# # cbmxl = NP.nanmax(overlays[0]['delay_map']) * 1e6
# # cbaxl = fig2.add_axes([0.08, 0.04, 0.395, 0.02])
# # cbarl = fig2.colorbar(iml, cax=cbaxl, orientation='horizontal')
# # cbaxl.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-1, fontsize=18)

# # # cbmnc = NP.nanmin(backdrop)
# # # cbmxc = NP.nanmax(backdrop)
# # # cbaxc = fig2.add_axes([0.42, 0.04, 0.24, 0.02])
# # # cbarc = fig2.colorbar(imc, cax=cbaxc, orientation='horizontal')
# # # cbaxc.set_xlabel('Temperature [K]', labelpad=-1, fontsize=18)

# # PLT.tight_layout()
# # fig2.subplots_adjust(bottom=0.1)

# # def update2(i, pointing_radec, lst, obsmode, telescope, backdrop, overlays, f2axs):
# #     for j in range(len(f2axs)):
# #         if j%2 == 0:
# #             f2axs[j].images[0].set_array(1e6 * OPS.reverse(overlays[i]['delay_map'][j/2,:].reshape(-1,backdrop_xsize), axis=1))
# #         elif j%2 == 1:
# #             f2axs[j].images[0].set_array(backdrop)
# #             f2axs[j].images[1].set_array(OPS.reverse((overlays[i]['pbeam']*overlays[i]['backdrop']).reshape(-1,backdrop_xsize), axis=1))
# #             # f2axs[j].images[1].set_array(backdrop * overlays[i]['pbeam'].reshape(-1,backdrop_xsize))
# #             # f2axs[j].images[0].set_alpha(1.0)
# #             # f2axs[j].imshow(backdrop * overlays[i]['pbeam'].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(), alpha=0.6)
# #     cbmnr = NP.nanmin(overlays[i]['pbeam']*overlays[i]['backdrop'])
# #     cbmxr = NP.nanmax(overlays[i]['pbeam']*overlays[i]['backdrop'])
# #     cbaxr = fig2.add_axes([0.57, 0.04, 0.395, 0.02])
# #     cbarr = fig2.colorbar(f2axs[j].images[1], cax=cbaxr, orientation='horizontal')
# #     cbaxr.set_xlabel('Temperature [K]', labelpad=-1, fontsize=18)


# #     return f2axs

# # anim2 = MOV.FuncAnimation(fig2, update2, fargs=(pointings_radec, lst, obs_mode, telescope, backdrop, overlays, f2axs), frames=len(overlays), interval=interval, blit=True)

# # anim2.save(animation2_file+anim_format, fps=fps, codec='x264')

# #################################################################################

simdata_neg_bl_orientation_ind = simdata_bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation

ia.baselines[simdata_neg_bl_orientation_ind,:] = -ia.baselines[simdata_neg_bl_orientation_ind,:]
# ia.baseline_orientations[simdata_neg_bl_orientation_ind] = 180.0 + ia.baseline_orientations[simdata_neg_bl_orientation_ind]
ia.vis_freq[simdata_neg_bl_orientation_ind,:,:] = ia.vis_freq[simdata_neg_bl_orientation_ind,:,:].conj()
ia.skyvis_freq[simdata_neg_bl_orientation_ind,:,:] = ia.skyvis_freq[simdata_neg_bl_orientation_ind,:,:].conj()
ia.vis_noise_freq[simdata_neg_bl_orientation_ind,:,:] = ia.vis_noise_freq[simdata_neg_bl_orientation_ind,:,:].conj()

ia.delay_transform(f_pad, freq_wts=window) # delay transform re-estimate
lags = ia.lags
vis_lag = ia.vis_lag
skyvis_lag = ia.skyvis_lag

if max_abs_delay is not None:
    small_delays_ind = NP.abs(lags) <= max_abs_delay * 1e-6
    lags = lags[small_delays_ind]
    vis_lag = vis_lag[:,small_delays_ind,:]
    skyvis_lag = skyvis_lag[:,small_delays_ind,:]

## Delay limits re-estimation

delay_matrix = DLY.delay_envelope(ia.baselines, pointings_dircos, units='mks')

## Binning baselines by orientation

# blo = bl_orientation[:min(n_bl_chunks*baseline_chunk_size, total_baselines)]
blo = bl_orientation
# blo[blo < -0.5*360.0/n_bins_baseline_orientation] = 360.0 - NP.abs(blo[blo < -0.5*360.0/n_bins_baseline_orientation])
bloh, bloe, blon, blori = OPS.binned_statistic(blo, statistic='count', bins=n_bins_baseline_orientation, range=[(-0.5*180.0/n_bins_baseline_orientation, 180.0-0.5*180.0/n_bins_baseline_orientation)])

if n_bins_baseline_orientation == 4:
    blo_ax_mapping = [6,3,2,1,4,7,8,9]

norm_b = PLTC.Normalize(vmin=mindelay, vmax=maxdelay)

# fig = PLT.figure(figsize=(14,14))
# faxs = []
# for i in xrange(n_bins_baseline_orientation):
#     ax = fig.add_subplot(3,3,blo_ax_mapping[i])
#     ax.set_xlim(0,bloh[i]-1)
#     ax.set_ylim(NP.amin(lags*1e6), NP.amax(lags*1e6))
#     ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='semibold')
#     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
#     blind = blori[blori[i]:blori[i+1]]
#     sortind = NP.argsort(bl_length[blind], kind='heapsort')
#     imdspec = ax.imshow(NP.abs(vis_lag[blind[sortind],:,0].T), origin='lower', extent=(0, blind.size-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=1e-2, vmax=NP.amax(NP.abs(vis_lag))), interpolation=None)
#     l = ax.plot([], [], 'k:', [], [], 'k:', [], [], 'k--', [], [], 'k--')
#     ax.set_aspect('auto')
#     faxs += [ax]

#     ax = fig.add_subplot(3,3,blo_ax_mapping[i+n_bins_baseline_orientation])
#     if backdrop_coords == 'radec':
#         ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=12)
#         ax.set_ylabel(r'$\delta$ [degrees]', fontsize=12)
#     elif backdrop_coords == 'dircos':
#         ax.set_xlabel('l')
#         ax.set_ylabel('m')
#     imdmap = ax.imshow(1e6 * OPS.reverse(overlays[0]['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), vmin=mindelay, vmax=maxdelay)
#     # imdmap.set_clim(mindelay, maxdelay)
#     ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='semibold')
#     ax.grid(True)
#     ax.tick_params(which='major', length=12, labelsize=12)
#     ax.tick_params(which='minor', length=6)
#     faxs += [ax]

# cbmnt = NP.amin(NP.abs(vis_lag))
# cbmxt = NP.amax(NP.abs(vis_lag))
# cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
# cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
# cbaxt.set_xlabel('Jy', labelpad=-60, fontsize=18)

# # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
# # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
# # cbmnb = mindelay * 1e6
# # cbmxb = maxdelay * 1e6
# cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
# cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
# cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-60, fontsize=18)

# ax = fig.add_subplot(3,3,5)
# imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
# imsky2 = ax.imshow(OPS.reverse((overlays[0]['pbeam']*overlays[0]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85)
# ax.set_title('Foregrounds', fontsize=18, weight='semibold')
# ax.grid(True)
# ax.tick_params(which='major', length=12, labelsize=12)
# ax.tick_params(which='minor', length=6)
# if backdrop_coords == 'radec':
#     ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=12)
#     ax.set_ylabel(r'$\delta$ [degrees]', fontsize=12)
# elif backdrop_coords == 'dircos':
#     ax.set_xlabel('l')
#     ax.set_ylabel('m')

# cbmnc = NP.nanmin(overlays[0]['pbeam']*overlays[0]['backdrop'])
# cbmxc = NP.nanmax(overlays[0]['pbeam']*overlays[0]['backdrop'])
# cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
# cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
# cbaxc.set_xlabel('Temperature [K]', labelpad=-60, fontsize=18)

# faxs += [ax]
# tpc = faxs[-1].text(0.5, 1.15, '', transform=ax.transAxes, fontsize=12, weight='semibold', ha='center')

# PLT.tight_layout()
# fig.subplots_adjust(bottom=0.1)
# fig.subplots_adjust(top=0.9)

# def update(i, pointing_radec, lst, obsmode, telescope, bll, blori, lags, vis_lag, delaymatrix, backdrop_coords, backdrop, overlays, xv, yv, xv_uniq, yv_uniq, faxs, tpc):
    
#     delay_ranges = NP.dstack((-delaymatrix[:,:vis_lag.shape[0],0] - delaymatrix[:,:vis_lag.shape[0],1], delaymatrix[:,:vis_lag.shape[0],0] - delaymatrix[:,:vis_lag.shape[0],1]))
#     delay_horizon = NP.dstack((-delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],0]))
#     bl = bll[:vis_lag.shape[0]]

#     # label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i])
    
#     for j in xrange((len(faxs)-1)/2):
#         blind = blori[blori[j]:blori[j+1]]
#         sortind = NP.argsort(bl[blind], kind='heapsort')
        
#         faxs[2*j].images[0].set_array(NP.abs(vis_lag[blind[sortind],:,i].T))
#         faxs[2*j].lines[0].set_xdata(NP.arange(blind.size))
#         faxs[2*j].lines[0].set_ydata(delay_ranges[i,blind[sortind],0]*1e6)
#         faxs[2*j].lines[1].set_xdata(NP.arange(blind.size))
#         faxs[2*j].lines[1].set_ydata(delay_ranges[i,blind[sortind],1]*1e6)
#         faxs[2*j].lines[2].set_xdata(NP.arange(blind.size))
#         faxs[2*j].lines[2].set_ydata(delay_horizon[i,blind[sortind],0]*1e6)
#         faxs[2*j].lines[3].set_xdata(NP.arange(blind.size))
#         faxs[2*j].lines[3].set_ydata(delay_horizon[i,blind[sortind],1]*1e6)

#         faxs[2*j+1].images[0].set_array(1e6 * OPS.reverse(overlays[i]['delay_map'][j,:].reshape(-1,backdrop_xsize), axis=1))

#         faxs[-1].images[0].set_array(backdrop)
#         faxs[-1].images[1].set_array(OPS.reverse((overlays[i]['pbeam']*overlays[i]['backdrop']).reshape(-1,backdrop_xsize), axis=1))
#         tpc.set_text(r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i]))

#     return faxs

# anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lst, obs_mode, telescope, bl_length, blori, lags, skyvis_lag, delay_matrix, backdrop_coords, backdrop, overlays, xvect, yvect, xgrid[0,:], ygrid[:,0], faxs, tpc), frames=len(overlays), interval=interval, blit=True)
# PLT.show()
# anim.save(animation_file+anim_format, fps=fps, codec='x264')

#################################################################################

## Plot the snapshots in the data binned by baseline orientations in contiguous baseline orientations

pointings_radec[pointings_radec[:,0] < 0.0,0] += 360.0 
lst[lst < 0.0] += 360.0
lst /= 15.0

for j in xrange(vis_lag.shape[2]):

    fig = PLT.figure(figsize=(10,10))
    faxs = []
    for i in xrange(n_bins_baseline_orientation):
        ax = fig.add_subplot(3,3,blo_ax_mapping[i])
        ax.set_xlim(0,bloh[i]-1)
        ax.set_ylim(NP.amin(lags*1e6), NP.amax(lags*1e6))
        ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='medium')
        ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
        blind = blori[blori[i]:blori[i+1]]
        sortind = NP.argsort(bl_length[blind], kind='heapsort')
        imdspec = ax.imshow(NP.abs(skyvis_lag[blind[sortind],:,j].T), origin='lower', extent=(0, blind.size-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))), interpolation=None)
        l = ax.plot([], [], 'k:', [], [], 'k:', [], [], 'k--', [], [], 'k--')
        ax.set_aspect('auto')
        faxs += [ax]
    
        ax = fig.add_subplot(3,3,blo_ax_mapping[i+n_bins_baseline_orientation])
        if backdrop_coords == 'radec':
            ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
            ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
        elif backdrop_coords == 'dircos':
            ax.set_xlabel('l')
            ax.set_ylabel('m')
        imdmap = ax.imshow(1e6 * OPS.reverse(overlays[j]['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
        # imdmap.set_clim(mindelay, maxdelay)
        ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='medium')
        ax.grid(True)
        ax.tick_params(which='major', length=12, labelsize=12)
        ax.tick_params(which='minor', length=6)
        ax.locator_params(axis='x', nbins=5)
        faxs += [ax]
    
    cbmnt = NP.amin(NP.abs(skyvis_lag))
    cbmxt = NP.amax(NP.abs(skyvis_lag))
    cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
    cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
    cbaxt.set_xlabel('Jy', labelpad=-50, fontsize=18)
    
    # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
    # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
    # cbmnb = mindelay * 1e6
    # cbmxb = maxdelay * 1e6
    cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
    cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
    cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-45, fontsize=18)
    
    ax = fig.add_subplot(3,3,5)
    imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
    imsky2 = ax.imshow(OPS.reverse((overlays[j]['pbeam']*overlays[j]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85)
    ax.set_title('Foregrounds', fontsize=18, weight='medium')
    ax.grid(True)
    ax.tick_params(which='major', length=12, labelsize=12)
    ax.tick_params(which='minor', length=6)
    if backdrop_coords == 'radec':
        ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
        ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
    elif backdrop_coords == 'dircos':
        ax.set_xlabel('l')
        ax.set_ylabel('m')
    ax.locator_params(axis='x', nbins=5)
    
    cbmnc = NP.nanmin(overlays[j]['pbeam']*overlays[j]['backdrop'])
    cbmxc = NP.nanmax(overlays[j]['pbeam']*overlays[j]['backdrop'])
    cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
    cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
    cbaxc.set_xlabel('Temperature [K]', labelpad=-45, fontsize=18, weight='medium')
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbarc.locator = tick_locator
    cbarc.update_ticks()
    
    faxs += [ax]
    tpc = faxs[-1].text(0.5, 1.25, r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointings_radec[j,:]) + '\nLST = {0:.2f} hrs'.format(lst[j]), transform=ax.transAxes, fontsize=14, weight='medium', ha='center')
    
    PLT.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

## Plot all baselines combined (in contiguous baseline orientation bins)

fig = PLT.figure(figsize=(6,8))

ax1 = fig.add_subplot(211)
# ax1.set_xlabel('Baseline Length [m]', fontsize=18)
# ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
# dspec1 = ax1.pcolorfast(bl_length, 1e6*lags, NP.abs(skyvis_lag[:-1,:-1,0].T), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))))
# ax1.set_xlim(bl_length[0], bl_length[-1])
# ax1.set_ylim(1e6*lags[0], 1e6*lags[-1])
ax1.set_xlabel('Baseline Index', fontsize=18)
ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
dspec1 = ax1.imshow(NP.abs(skyvis_lag[:,:,0].T), origin='lower', extent=(0, skyvis_lag.shape[0]-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))), interpolation=None)
ax1.set_aspect('auto')

ax2 = fig.add_subplot(212)
ax2.set_xlabel('Baseline Length [m]', fontsize=18)
ax2.set_ylabel(r'lag [$\mu$s]', fontsize=18)
# dspec2 = ax2.pcolorfast(bl_length, 1e6*lags, NP.abs(skyvis_lag[:-1,:-1,1].T), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))))
# ax2.set_xlim(bl_length[0], bl_length[-1])
# ax2.set_x=ylim(1e6*lags[0], 1e6*lags[-1])
ax2.set_xlabel('Baseline Index', fontsize=18)
ax2.set_ylabel(r'lag [$\mu$s]', fontsize=18)
dspec2 = ax2.imshow(NP.abs(skyvis_lag[:,:,1].T), origin='lower', extent=(0, skyvis_lag.shape[0]-1, NP.amin(lags*1e6), NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(skyvis_lag)), vmax=NP.amax(NP.abs(skyvis_lag))), interpolation=None)
ax2.set_aspect('auto')

cbax = fig.add_axes([0.88, 0.08, 0.03, 0.9])
cb = fig.colorbar(dspec2, cax=cbax, orientation='vertical')
cbax.set_ylabel('Jy', labelpad=-60, fontsize=18)

PLT.tight_layout()
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(left=0.1)

PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_combined_baseline_visibilities_contiguous_orientations_'+avg_drifts_str+obs_mode+'_gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_{0:0d}_snapshots.eps'.format(skyvis_lag.shape[2]), bbox_inches=0)
PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_combined_baseline_visibilities_contiguous_orientations_'+avg_drifts_str+obs_mode+'_gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_{0:0d}_snapshots.png'.format(skyvis_lag.shape[2]), bbox_inches=0)

