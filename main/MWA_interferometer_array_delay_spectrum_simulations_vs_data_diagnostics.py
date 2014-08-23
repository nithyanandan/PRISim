import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CMAP
import matplotlib.animation as MOV
from matplotlib import ticker
from scipy.interpolate import griddata
import datetime as DT
import time 
import progressbar as PGB
import healpy as HP
from mwapy.pb import primary_beam as MWAPB
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import lookup_operations as LKP
import ipdb as PDB

# reload(GEOM)
# reload(RI)
# reload(CTLG)
# reload(CNST)
# reload(DSP) 
# reload(OPS)
# reload(PB)
# reload(DLY)
# reload(LKP)

telescope_id = 'custom'
element_size = 0.74
element_shape = 'delta'
phased_array = True

if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole'):
    element_size = 0.74
    element_shape = 'dipole'
elif telescope_id == 'vla':
    element_size = 25.0
    element_shape = 'dish'
elif telescope_id == 'gmrt':
    element_size = 45.0
    element_shape = 'dish'
elif telescope_id == 'hera':
    element_size = 14.0
    element_shape = 'dish'
elif telescope_id == 'custom':
    if (element_shape is None) or (element_size is None):
        raise ValueError('Both antenna element shape and size must be specified for the custom telescope type.')
    elif element_size <= 0.0:
        raise ValueError('Antenna element size must be positive.')
else:
    raise ValueError('telescope ID must be specified.')

if telescope_id == 'custom':
    if element_shape == 'delta':
        telescope_id = 'delta'
    else:
        telescope_id = '{0:.1f}m_{1:}'.format(element_size, element_shape)

    if phased_array:
        telescope_id = telescope_id + '_array'
telescope_str = telescope_id+'_'

ground_plane = 0.3 # height of antenna element above ground plane
if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

obs_mode = 'custom'
avg_drifts = False
beam_switch = False
snapshot_type_str = ''
if avg_drifts:
    snapshot_type_str = 'drift_averaged_'
if beam_switch:
    snapshot_type_str = 'beam_switches_'

n_sky_sectors = 1
sky_sector = None # if None, use all sky sector. Accepted values are None, 0, 1, 2, or 3
if sky_sector is None:
    sky_sector_str = '_all_sky_'
    n_sky_sectors = 1
    sky_sector = 0
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(sky_sector)

Tsys = 85.6  # System temperature in K
freq = 185.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 80e3 # in Hz
coarse_channel_resolution = 1.28e6 # in Hz
bpass_shape = 'bnw'
f_pad = 1.0
oversampling_factor = 1.0 + f_pad
n_channels = 384
nchan = n_channels
bw = nchan * freq_resolution
max_abs_delay = 2.5 # in micro seconds

window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=0, centering=True, area_normalize=True) 

nside = 64
use_GSM = True
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False

dsm_base_freq = 408e6 # Haslam map frequency
csm_base_freq = 1.420e9 # NVSS frequency
dsm_dalpha = 0.5 # Spread in spectral index in Haslam map
csm_dalpha = 0.5 # Spread in spectral index in NVSS

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

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ref_bl, ref_bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
ref_bl_length = NP.sqrt(NP.sum(ref_bl**2, axis=1))
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True) 
n_bins_baseline_orientation = 4
neg_bl_orientation_ind = ref_bl_orientation < 0.0
ref_bl[neg_bl_orientation_ind,:] = -1.0 * ref_bl[neg_bl_orientation_ind,:]
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True)
sortind = NP.argsort(ref_bl_length, kind='mergesort')
ref_bl = ref_bl[sortind,:]
ref_bl_length = ref_bl_length[sortind]
ref_bl_orientation = ref_bl_orientation[sortind]
ref_bl_id = ref_bl_id[sortind]

n_bl_chunks = 32
baseline_chunk_size = 64
total_baselines = ref_bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]
ref_bl = ref_bl[:baseline_bin_indices[n_bl_chunks],:]
ref_bl_length = ref_bl_length[:baseline_bin_indices[n_bl_chunks]]
ref_bl_orientation = ref_bl_orientation[:baseline_bin_indices[n_bl_chunks]]
ref_bl_id = ref_bl_id[:baseline_bin_indices[n_bl_chunks]]

roifile = '/data3/t_nithyanandan/project_MWA/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
roi = RI.ROI_parameters(init_file=roifile)
telescope = roi.telescope

# telescope = {}
# telescope['id'] = telescope_id
# telescope['shape'] = element_shape
# telescope['size'] = element_size
# telescope['orientation'] = element_orientation
# telescope['ocoords'] = element_ocoords
# telescope['groundplane'] = ground_plane

latitude = -26.701

fhd_obsid = [1061309344, 1061316544]

lags = None
skyvis_lag = None

pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'

# Read in simulated data

infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)

asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
simdata_bl_orientation = NP.angle(ia.baselines[:,0] + 1j * ia.baselines[:,1], deg=True)
simdata_neg_bl_orientation_ind = simdata_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
simdata_bl_orientation[simdata_neg_bl_orientation_ind] -= 180.0
ia.baselines[simdata_neg_bl_orientation_ind,:] = -ia.baselines[simdata_neg_bl_orientation_ind,:]

hdulist = fits.open(infile+'.fits')
latitude = hdulist[0].header['latitude']
pointing_coords = hdulist[0].header['pointing_coords']
pointings_table = hdulist['POINTING AND PHASE CENTER INFO'].data
lst = pointings_table['LST']
n_snaps = lst.size
hdulist.close()

if n_snaps != len(fhd_obsid):
    raise ValueError('Number of snapshots in simulations and data do not match')

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

hdulist = fits.open(asm_CLEAN_infile+'.fits')
clean_lags = hdulist['SPECTRAL INFO'].data['lag']
clean_lags_orig = NP.copy(clean_lags)
asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
hdulist.close()

asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
asm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = asm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis, axis=1),axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis_res, axis=1),axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
asm_cc_skyvis_lag = asm_cc_skyvis_lag + asm_ccres_sky

asm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis, axis=1),axes=1) * asm_cc_vis.shape[1] * freq_resolution
asm_ccres = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis_res, axis=1),axes=1) * asm_cc_vis.shape[1] * freq_resolution
asm_cc_vis_lag = asm_cc_vis_lag + asm_ccres

asm_cc_skyvis_lag = DSP.downsampler(asm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
asm_cc_vis_lag = DSP.downsampler(asm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
clean_lags = clean_lags.ravel()

vis_noise_lag = NP.copy(ia.vis_noise_lag)

delaymat = DLY.delay_envelope(ia.baselines, pc, units='mks')
bw = nchan * freq_resolution
min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
max_delay = delaymat[0,:,0]-delaymat[0,:,1]
clags = clean_lags.reshape(1,-1)
min_delay = min_delay.reshape(-1,1)
max_delay = max_delay.reshape(-1,1)
thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
thermal_noise_window = NP.repeat(thermal_noise_window, ia.baselines.shape[0], axis=0)
EoR_window = NP.logical_or(clags > max_delay+1/bw, clags < min_delay-1/bw)
wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
# vis_rms_lag = OPS.rms(asm_cc_vis_lag.reshape(-1,n_snaps), mask=NP.logical_not(NP.repeat(thermal_noise_window.reshape(-1,1), n_snaps, axis=1)), axis=0)
# vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
# T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,-1)) * CNST.Jy
# vis_rms_lag_theory = OPS.rms(vis_noise_lag.reshape(-1,n_snaps), mask=NP.logical_not(NP.repeat(EoR_window.reshape(-1,1), n_snaps, axis=1)), axis=0)
# vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
# T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,-1)) * CNST.Jy
vis_rms_lag = OPS.rms(asm_cc_vis_lag, mask=NP.logical_not(NP.repeat(thermal_noise_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
vis_rms_lag_theory = OPS.rms(vis_noise_lag, mask=NP.logical_not(NP.repeat(EoR_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy

if max_abs_delay is not None:
    small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
    clean_lags = clean_lags[small_delays_ind]
    asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
    asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]

## Read in FHD data and other required information

pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(1,2,3), delimiter=',')
obs_id = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(0,), delimiter=',', dtype=str)
obsfile_lst = 15.0 * pointing_info_from_file[:,2]
obsfile_pointings_altaz = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
obsfile_pointings_dircos = GEOM.altaz2dircos(obsfile_pointings_altaz, units='degrees')
obsfile_pointings_hadec = GEOM.altaz2hadec(obsfile_pointings_altaz, latitude, units='degrees')

fhd_info = {}
for j in range(len(fhd_obsid)):
    fhd_infile = '/data3/t_nithyanandan/project_MWA/fhd_delay_spectrum_{0:0d}_reformatted.npz'.format(fhd_obsid[j])
    fhd_data = NP.load(fhd_infile)
    fhdfile_bl_id = fhd_data['fhd_bl_id']
    fhdfile_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhdfile_bl_id)))
    fhd_bl_id = ref_bl_id[fhdfile_bl_ind]
    fhd_bl = ref_bl[fhdfile_bl_ind, :]
    fhd_bl_length = ref_bl_length[fhdfile_bl_ind]
    fhd_bl_orientation = ref_bl_orientation[fhdfile_bl_ind]
    fhd_vis_lag_noisy = fhd_data['fhd_vis_lag_noisy']
    fhd_delays = fhd_data['fhd_delays']
    fhd_C = fhd_data['fhd_C']
    valid_ind = NP.logical_and(NP.abs(NP.sum(fhd_vis_lag_noisy[:,:,0],axis=1))!=0.0, NP.abs(NP.sum(fhd_C[:,:,0],axis=1))!=0.0)
    fhd_C = fhd_C[valid_ind,:,:]
    fhd_vis_lag_noisy = fhd_vis_lag_noisy[valid_ind,:,:]
    fhd_bl_id = fhd_bl_id[valid_ind]
    fhd_bl = fhd_bl[valid_ind,:]
    fhd_bl_length = fhd_bl_length[valid_ind]
    fhd_bl_orientation = fhd_bl_orientation[valid_ind]
    fhd_neg_bl_orientation_ind = fhd_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    fhd_bl_orientation[fhd_neg_bl_orientation_ind] -= 180.0
    fhd_bl[fhd_neg_bl_orientation_ind,:] = -fhd_bl[fhd_neg_bl_orientation_ind,:]

    fhd_vis_lag_noisy *= 2.78*nchan*freq_resolution/fhd_C
    fhd_obsid_pointing_dircos = obsfile_pointings_dircos[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
    fhd_obsid_pointing_altaz = obsfile_pointings_altaz[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
    fhd_obsid_pointing_hadec = obsfile_pointings_hadec[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
    fhd_lst = NP.asscalar(obsfile_lst[obs_id==str(fhd_obsid[j])])
    fhd_obsid_pointing_radec = NP.copy(fhd_obsid_pointing_hadec)
    fhd_obsid_pointing_radec[0,0] = fhd_lst - fhd_obsid_pointing_hadec[0,0]

    fhd_delaymat = DLY.delay_envelope(fhd_bl, pc, units='mks')

    fhd_min_delay = -fhd_delaymat[0,:,1]-fhd_delaymat[0,:,0]
    fhd_max_delay = fhd_delaymat[0,:,0]-fhd_delaymat[0,:,1]
    fhd_min_delay = fhd_min_delay.reshape(-1,1)
    fhd_max_delay = fhd_max_delay.reshape(-1,1)

    fhd_thermal_noise_window = NP.abs(fhd_delays) >= max_abs_delay*1e-6
    fhd_thermal_noise_window = fhd_thermal_noise_window.reshape(1,-1)
    fhd_thermal_noise_window = NP.repeat(fhd_thermal_noise_window, fhd_bl.shape[0], axis=0)
    fhd_EoR_window = NP.logical_or(fhd_delays > fhd_max_delay+1/bw, fhd_delays < fhd_min_delay-1/bw)
    fhd_wedge_window = NP.logical_and(fhd_delays <= fhd_max_delay, fhd_delays >= fhd_min_delay)
    fhd_vis_rms_lag = OPS.rms(fhd_vis_lag_noisy[:,:,0], mask=NP.logical_not(fhd_thermal_noise_window), axis=1)
    fhd_vis_rms_freq = NP.abs(fhd_vis_rms_lag) / NP.sqrt(nchan) / freq_resolution

    if max_abs_delay is not None:
        small_delays_ind = NP.abs(fhd_delays) <= max_abs_delay * 1e-6
        fhd_delays = fhd_delays[small_delays_ind]
        fhd_vis_lag_noisy = fhd_vis_lag_noisy[:,small_delays_ind,:]

    fhd_info[fhd_obsid[j]] = {}
    fhd_info[fhd_obsid[j]]['bl_id'] = fhd_bl_id
    fhd_info[fhd_obsid[j]]['bl'] = fhd_bl
    fhd_info[fhd_obsid[j]]['bl_orientation'] = fhd_bl_orientation
    fhd_info[fhd_obsid[j]]['delays'] = fhd_delays
    fhd_info[fhd_obsid[j]]['C'] = fhd_C
    fhd_info[fhd_obsid[j]]['vis_lag_noisy'] = fhd_vis_lag_noisy
    fhd_info[fhd_obsid[j]]['lst'] = fhd_lst
    fhd_info[fhd_obsid[j]]['pointing_radec'] = fhd_obsid_pointing_radec
    fhd_info[fhd_obsid[j]]['pointing_hadec'] = fhd_obsid_pointing_hadec
    fhd_info[fhd_obsid[j]]['pointing_altaz'] = fhd_obsid_pointing_altaz
    fhd_info[fhd_obsid[j]]['pointing_dircos'] = fhd_obsid_pointing_dircos
    fhd_info[fhd_obsid[j]]['min_delays'] = fhd_min_delay
    fhd_info[fhd_obsid[j]]['max_delays'] = fhd_max_delay
    fhd_info[fhd_obsid[j]]['rms_lag'] = fhd_vis_rms_lag
    fhd_info[fhd_obsid[j]]['rms_freq'] = fhd_vis_rms_freq

## Prepare data for overlays

cardinal_blo = 180.0 / n_bins_baseline_orientation * (NP.arange(n_bins_baseline_orientation)-1).reshape(-1,1)
cardinal_bll = 100.0
cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

backdrop_xsize = 100
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
    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq/1e6,nside)
    hdulist = fits.open(dsm_file)
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures
    backdrop = HP.cartview(temperatures.ravel(), coord=['G','E'], rot=[0,0,0], xsize=backdrop_xsize, return_projected_map=True)
elif use_GLEAM or use_SUMSS or use_NVSS or use_CSM:
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
        fluxes = fint * (freq_catalog*1e9/freq)**spindex
    elif use_NVSS:
        pass
    else:
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
        freq_catalog = freq_SUMSS*1e9 + NP.zeros(fint.size)
        catlabel = NP.repeat('SUMSS', fint.size)
        ra_deg = ra_deg_SUMSS + 0.0
        dec_deg = dec_deg_SUMSS
        spindex = spindex_SUMSS
        majax = fmajax/3.6e3
        minax = fminax/3.6e3
        fluxes = fint + 0.0

        nvss_file = '/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits'
        freq_NVSS = 1.4 # in GHz
        hdulist = fits.open(nvss_file)
        ra_deg_NVSS = hdulist[1].data['RA(2000)']
        dec_deg_NVSS = hdulist[1].data['DEC(2000)']
        nvss_fpeak = hdulist[1].data['PEAK INT']
        nvss_majax = hdulist[1].data['MAJOR AX']
        nvss_minax = hdulist[1].data['MINOR AX']
        hdulist.close()
    
        spindex_NVSS = -0.83 + NP.zeros(nvss_fpeak.size)
        not_in_SUMSS_ind = NP.logical_and(dec_deg_NVSS > -30.0, dec_deg_NVSS <= min(90.0, latitude+90.0))
        bright_source_ind = nvss_fpeak >= 10.0 * (freq_NVSS*1e9/freq)**(spindex_NVSS)
        PS_ind = NP.sqrt(nvss_majax**2-(0.75/60.0)**2) < 14.0/3.6e3
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

    if backdrop_coords == 'radec':
        if use_DSM or use_GSM:
            backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
            backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
        else:
            ra_deg_wrapped = ra_deg.ravel() + 0.0
            ra_deg_wrapped[ra_deg > 180.0] -= 360.0

            dxvect = xgrid[0,1]-xgrid[0,0]
            dyvect = ygrid[1,0]-ygrid[0,0]
            ibind, nnval, distNN = LKP.lookup(ra_deg_wrapped.ravel(), dec_deg.ravel(), fluxes.ravel(), xvect, yvect, distance_ULIM=NP.sqrt(dxvect**2 + dyvect**2), remove_oob=False)
            backdrop = nnval.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope_id == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', pointing_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
else:
    if use_PS:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/PS_catalog.txt'
        catdata = ascii.read(catalog_file, comment='#', header_start=0, data_start=1)
        ra_deg = catdata['RA'].data
        dec_deg = catdata['DEC'].data
        fluxes = catdata['F_INT'].data
        
    if backdrop_coords == 'radec':
        ra_deg_wrapped = ra_deg.ravel() + 0.0
        ra_deg_wrapped[ra_deg > 180.0] -= 360.0
        
        dxvect = xgrid[0,1]-xgrid[0,0]
        dyvect = ygrid[1,0]-ygrid[0,0]
        ibind, nnval, distNN = LKP.lookup(ra_deg_wrapped.ravel(), dec_deg.ravel(), fluxes.ravel(), xvect, yvect, distance_ULIM=NP.sqrt(dxvect**2 + dyvect**2), remove_oob=False)
        backdrop = nnval.reshape(backdrop_xsize/2, backdrop_xsize)
        # backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
        # backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope_id == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', pointing_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)

overlays = []
roi_obj_inds = []

for i in xrange(n_snaps):
    overlay = {}
    if backdrop_coords == 'radec':
        havect = lst[i] - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        dircos = GEOM.altaz2dircos(altaz, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
        az = altaz[:,1] + 0.0
        az[az > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
        roi_sector_altaz = NP.asarray(NP.where(NP.logical_or(NP.logical_and(az[roi_altaz] >= -0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < -0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors), NP.logical_and(az[roi_altaz] >= 180.0 - 0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < 180.0 - 0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors)))).ravel()
        pb = NP.empty(xvect.size)
        pb.fill(NP.nan)
        pbx_MWA_vect = NP.empty(xvect.size)
        pbx_MWA_vect.fill(NP.nan)
        pby_MWA_vect = NP.empty(xvect.size)
        pby_MWA_vect.fill(NP.nan)
        bd = NP.empty(xvect.size)
        bd.fill(NP.nan)
        angdist = GEOM.sphdist(pointings_altaz[i,1], pointings_altaz[i,0], altaz[:,1], altaz[:,0])
        # pinfo = {}
        # pinfo['pointing_center'] = pointings_altaz[i,:]
        # pinfo['pointing_coords'] = 'altaz'
        # pinfo['delayerr'] = 0.1e-9 # in seconds
        pinfo = roi.pinfo[i]
        pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_center=pointings_altaz[i,:], pointing_info=pinfo)
        # bd[roi_altaz] = backdrop.ravel()[roi_altaz]
        # pb[roi_altaz[roi_sector_altaz]] = PB.primary_beam_generator(altaz[roi_altaz[roi_sector_altaz],:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', phase_center=pointings_altaz[i,:])
        bd[roi_altaz[roi_sector_altaz]] = backdrop.ravel()[roi_altaz[roi_sector_altaz]]
        overlay['angdist'] = angdist
        overlay['pbeam'] = pb
        overlay['backdrop'] = bd
        overlay['roi_obj_inds'] = roi_altaz
        overlay['roi_sector_inds'] = roi_altaz[roi_sector_altaz]
        overlay['delay_map'] = NP.empty((n_bins_baseline_orientation, xvect.size))
        overlay['delay_map'].fill(NP.nan)
        overlay['delay_map'][:,roi_altaz] = (DLY.geometric_delay(cardinal_bl, altaz[roi_altaz,:], altaz=True, dircos=False, hadec=False, latitude=latitude)-DLY.geometric_delay(cardinal_bl, pc, altaz=False, dircos=True, hadec=False, latitude=latitude)).T
        pbx_MWA, pby_MWA = MWAPB.MWA_Tile_advanced(NP.radians(90.0-altaz[roi_altaz,0]).reshape(-1,1), NP.radians(altaz[roi_altaz,1]).reshape(-1,1), freq=185e6, delays=roi.pinfo[i]['delays']/435e-12)
        pbx_MWA_vect[roi_altaz] = pbx_MWA.ravel()
        pby_MWA_vect[roi_altaz] = pby_MWA.ravel()
        overlay['pbx_MWA'] = pbx_MWA_vect
        overlay['pby_MWA'] = pby_MWA_vect
        if use_CSM or use_SUMSS or use_NVSS or use_PS:
            src_hadec = NP.hstack(((lst[i]-ctlgobj.location[:,0]).reshape(-1,1), ctlgobj.location[:,1].reshape(-1,1)))
            src_altaz = GEOM.hadec2altaz(src_hadec, latitude, units='degrees')
            roi_src_altaz = NP.asarray(NP.where(src_altaz[:,0] >= 0.0)).ravel()
            pinfo = roi.pinfo[i]
            roi_pbeam = PB.primary_beam_generator(src_altaz[roi_src_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_center=pointings_altaz[i,:], pointing_info=pinfo)
            overlay['src_ind'] = roi_src_altaz
            overlay['pbeam_on_src'] = roi_pbeam.ravel()

        # delay_envelope = DLY.delay_envelope(cardinal_bl, dircos[roi_altaz,:])
        # overlay['delay_map'][:,roi_altaz] = (DLY.geometric_delay(cardinal_bl, altaz[roi_altaz,:], altaz=True, dircos=False, hadec=False, latitude=latitude)-DLY.geometric_delay(cardinal_bl, pointings_altaz[i,:], altaz=True, dircos=False, hadec=False, latitude=latitude)).T
        # roi_obj_inds += [roi_altaz]
    elif backdrop_coords == 'dircos':
        havect = lst[i] - ra_deg
        fg_altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),dec_deg.reshape(-1,1))), latitude, units='degrees')
        fg_dircos = GEOM.altaz2dircos(fg_altaz, units='degrees')
        roi_dircos = NP.asarray(NP.where(fg_dircos[:,2] >= 0.0)).ravel()
        overlay['roi_obj_inds'] = roi_dircos
        overlay['fg_dircos'] = fg_dircos
        if obs_mode == 'track':
            pinfo = roi.pinfo[i]
            pb = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, skyunits='dircos', freq_scale='Hz', pointing_center=pointings_dircos[i,:])
            # pb[pb < 0.5] = NP.nan
            overlay['pbeam'] = pb.reshape(backdrop_xsize, backdrop_xsize)
        overlay['delay_map'] = NP.empty((n_bins_baseline_orientation, xyzvect.shape[0])).fill(NP.nan)
    overlays += [overlay]

mnd = [NP.nanmin(olay['delay_map']) for olay in overlays]
mxd = [NP.nanmax(olay['delay_map']) for olay in overlays]
mindelay = min(mnd)
maxdelay = max(mxd)

if n_bins_baseline_orientation == 4:
    blo_ax_mapping = [7,4,1,2,3,6,9,8]

norm_b = PLTC.Normalize(vmin=mindelay, vmax=maxdelay)

# # Plot angular distance contours from the pointing center for the visible hemisphere

# if backdrop_coords == 'radec':
#     for j in xrange(n_snaps):
#         fig = PLT.figure(figsize=(8,6))
#         ax = fig.add_subplot(111)
#         imsky = ax.imshow(OPS.reverse((overlays[j]['pbeam']*overlays[j]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-3, vmax=1e3))
#         impbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='r')
#         imangc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['angdist'].reshape(-1,backdrop_xsize), levels=[0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0], colors='k')
#         ax.clabel(imangc, inline=1, fontsize=12, colors='k', fmt='%0.1f')
#         ax.clabel(impbc, inline=1, fontsize=8, colors='r', fmt='%0.3f')

#         ax.set_xlim(xvect.max(), xvect.min())
#         ax.set_ylim(yvect.min(), yvect.max())
#         ax.grid(True, which='both')
#         ax.set_aspect('equal')
#         ax.tick_params(which='major', length=12, labelsize=12)
#         ax.tick_params(which='minor', length=6)
#         ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#         ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#         ax.locator_params(axis='x', nbins=5)
    
#         cbax = fig.add_axes([0.15, 0.9, 0.8, 0.02])
#         cbar = fig.colorbar(imsky, cax=cbax, orientation='horizontal')
#         if use_GSM or use_DSM:
#             cbax.set_xlabel('Temperature [K]', labelpad=-50, fontsize=18, weight='medium')
#         else:
#             cbax.set_xlabel('Flux Density [Jy]', labelpad=-50, fontsize=18, weight='medium')

#         PLT.tight_layout()
#         # fig.subplots_adjust(bottom=0.1)
#         # fig.subplots_adjust(top=0.85)

#         PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'foregrounds_powerpattern_angles_'+ground_plane_str+snapshot_type_str+obs_mode+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

# # Plot ratios of power patterns of MWA_Tools to analytic models

# if backdrop_coords == 'radec':
#     for j in xrange(n_snaps):
#         fig = PLT.figure(figsize=(8,6))
#         ax = fig.add_subplot(111)
#         pbsky = ax.imshow(OPS.reverse((overlays[j]['pbx_MWA']/overlays[j]['pbeam']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-3, vmax=1e3))

#         ax.set_xlim(xvect.max(), xvect.min())
#         ax.set_ylim(yvect.min(), yvect.max())
#         ax.grid(True, which='both')
#         ax.set_aspect('equal')
#         ax.tick_params(which='major', length=12, labelsize=12)
#         ax.tick_params(which='minor', length=6)
#         ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#         ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#         ax.locator_params(axis='x', nbins=5)
    
#         cbax = fig.add_axes([0.15, 0.9, 0.8, 0.02])
#         cbar = fig.colorbar(imsky, cax=cbax, orientation='horizontal')

#         PLT.tight_layout()
#         # fig.subplots_adjust(bottom=0.1)
#         # fig.subplots_adjust(top=0.85)

#         PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'powerpattern_ratios_'+ground_plane_str+snapshot_type_str+obs_mode+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

# Plot noisy delay spectra ratios by different baseline orientations

for j in xrange(n_snaps):

    # Determine the baselines common to simulations and data

    common_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
    bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

    fig = PLT.figure(figsize=(10,10))
    faxs = []
    for i in xrange(n_bins_baseline_orientation):
        ax = fig.add_subplot(3,3,blo_ax_mapping[i])
        ax.set_xlim(0,bloh[i]-1)
        ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
        ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='medium')
        ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
        blind = blori[blori[i]:blori[i+1]]
        sortind = NP.argsort(ref_bl_length[common_bl_ind[blind]], kind='heapsort')
        imdspec = ax.imshow(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T), origin='lower', extent=(0, blind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-2, vmax=1e2), interpolation=None)
        # norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_vis_lag))), 
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
        imdmappbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
        # imdmap.set_clim(mindelay, maxdelay)
        ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='medium')
        ax.grid(True)
        ax.tick_params(which='major', length=12, labelsize=12)
        ax.tick_params(which='minor', length=6)
        ax.locator_params(axis='x', nbins=5)
        faxs += [ax]
    
    # cbmnt = NP.amin(NP.abs(asm_cc_vis_lag))
    # cbmxt = NP.amax(NP.abs(asm_cc_vis_lag))
    cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
    cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
    # cbaxt.set_xlabel('Jy', labelpad=-50, fontsize=18)
    
    # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
    # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
    # cbmnb = mindelay * 1e6
    # cbmxb = maxdelay * 1e6
    cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
    cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
    cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-45, fontsize=18)
    
    ax = fig.add_subplot(3,3,5)
    # imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
    impbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
    if use_CSM or use_NVSS or use_SUMSS or use_PS:
        imsky2 = ax.scatter(ra_deg_wrapped[overlays[j]['src_ind']].ravel(), dec_deg[overlays[j]['src_ind']].ravel(), c=overlays[j]['pbeam_on_src']*fluxes[overlays[j]['src_ind']], norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CMAP.jet, edgecolor='none', s=10)
    else:
        imsky2 = ax.imshow(OPS.reverse((overlays[j]['pbeam']*overlays[j]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-3, vmax=1e3))        
    ax.set_xlim(xvect.max(), xvect.min())
    ax.set_ylim(yvect.min(), yvect.max())
    ax.set_title('Foregrounds', fontsize=18, weight='medium')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.tick_params(which='major', length=12, labelsize=12)
    ax.tick_params(which='minor', length=6)
    if backdrop_coords == 'radec':
        ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
        ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
    elif backdrop_coords == 'dircos':
        ax.set_xlabel('l')
        ax.set_ylabel('m')
    ax.locator_params(axis='x', nbins=5)
    
    # cbmnc = NP.nanmin(overlays[j]['pbeam']*overlays[j]['backdrop'])
    # cbmxc = NP.nanmax(overlays[j]['pbeam']*overlays[j]['backdrop'])
    cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
    # cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
    cbarc = fig.colorbar(imsky2, cax=cbaxc, orientation='horizontal')
    if use_GSM or use_DSM:
        cbaxc.set_xlabel('Temperature [K]', labelpad=-50, fontsize=18, weight='medium')
    else:
        cbaxc.set_xlabel('Flux Density [Jy]', labelpad=-50, fontsize=18, weight='medium')
    # tick_locator = ticker.MaxNLocator(nbins=21)
    # cbarc.locator = tick_locator
    # cbarc.update_ticks()
    
    faxs += [ax]
    tpc = faxs[-1].text(0.5, 1.25, r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointings_radec[j,:]) + '\nLST = {0:.2f} hrs'.format(lst[j]), transform=ax.transAxes, fontsize=14, weight='medium', ha='center')
    
    PLT.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

# Plot noisy delay spectra ratios all baseline orientation combined

small_delays_EoR_window = EoR_window.T
if max_abs_delay is not None:
    small_delays_EoR_window = small_delays_EoR_window[small_delays_ind,:]

data_sim_difference_fraction = NP.zeros(len(fhd_obsid))

fig = PLT.figure(figsize=(7,10))
for j in xrange(n_snaps):

    # Determine the baselines common to simulations and data
    common_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
    sortind = NP.argsort(ref_bl_length[common_bl_ind], kind='heapsort')
    data_sim_ratio = NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T)
    relevant_EoR_window = small_delays_EoR_window[:,common_bl_ind[sortind]]

    data_sim_difference_fraction[j] = NP.abs(NP.sum(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T)-NP.abs(NP.mean(fhd_info[fhd_obsid[j]]['rms_lag']))) - NP.sum(NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T)-NP.abs(NP.mean(vis_rms_lag)))) / NP.sum(NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T)-NP.abs(NP.mean(vis_rms_lag)))

    mu = NP.mean(NP.log10(data_sim_ratio[relevant_EoR_window]))
    sig= NP.std(NP.log10(data_sim_ratio[relevant_EoR_window]))

    ax = fig.add_subplot(2,2,2*j+1)
    ax.set_xlim(0,common_bl_ind.size-1)
    ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
    ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
    imdspec = ax.imshow(data_sim_ratio, origin='lower', extent=(0, common_bl_ind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-2, vmax=1e2), interpolation=None)
    ax.set_aspect('auto')

    ax = fig.add_subplot(2,2,2*j+2)
    h, bins, patches = ax.hist(NP.log10(data_sim_ratio[relevant_EoR_window]), bins=50, normed=True, histtype='step', color='gray', linewidth=2.5)
    ax.set_xlabel('log ratio', fontsize=16, weight='medium')
    gauss_model = mlab.normpdf(bins, mu, sig)
    l = ax.plot(bins, gauss_model, 'k-')

cbaxt = fig.add_axes([0.13, 0.95, 0.37, 0.02])
cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
cbaxt.set_xlabel('ratio', labelpad=-50, fontsize=18)

PLT.tight_layout()
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(top=0.9)

PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_combined_baseline_CLEAN_noisy_visibilities_fhd_data_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

# # Plot ratio of delay spectra difference to expected error

# dsm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_dsm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
# csm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

# hdulist = fits.open(dsm_CLEAN_infile+'.fits')
# dsm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
# dsm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
# dsm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
# dsm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
# hdulist.close()

# dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
# dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
# dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
# dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

# dsm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
# dsm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis_res, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
# dsm_cc_skyvis_lag = dsm_cc_skyvis_lag + dsm_ccres_sky

# dsm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
# dsm_ccres = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis_res, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
# dsm_cc_vis_lag = dsm_cc_vis_lag + dsm_ccres

# hdulist = fits.open(csm_CLEAN_infile+'.fits')
# csm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
# csm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
# csm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
# csm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
# hdulist.close()

# csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
# csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
# csm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
# csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

# csm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
# csm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis_res, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
# csm_cc_skyvis_lag = csm_cc_skyvis_lag + csm_ccres_sky

# csm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
# csm_ccres = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis_res, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
# csm_cc_vis_lag = csm_cc_vis_lag + csm_ccres

# dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
# dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
# csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
# csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)

# if max_abs_delay is not None:
#     dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
#     csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]

# dsm_cc_skyvis_lag_err = dsm_cc_skyvis_lag * NP.log(dsm_base_freq/freq) * dsm_dalpha
# csm_cc_skyvis_lag_err = csm_cc_skyvis_lag * NP.log(csm_base_freq/freq) * csm_dalpha
# cc_skyvis_lag_err = NP.abs(dsm_cc_skyvis_lag_err) + NP.abs(csm_cc_skyvis_lag_err)

# for j in xrange(n_snaps):

#     # Determine the baselines common to simulations and data

#     common_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
#     bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

#     fig = PLT.figure(figsize=(10,10))
#     faxs = []
#     for i in xrange(n_bins_baseline_orientation):
#         ax = fig.add_subplot(3,3,blo_ax_mapping[i])
#         ax.set_xlim(0,bloh[i]-1)
#         ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
#         ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='medium')
#         ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
#         blind = blori[blori[i]:blori[i+1]]
#         sortind = NP.argsort(ref_bl_length[common_bl_ind[blind]], kind='heapsort')
#         imdspec = ax.imshow((NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)) / NP.sqrt(NP.abs(cc_skyvis_lag_err[common_bl_ind[blind[sortind]],:,j].T)**2 + NP.abs(vis_rms_lag[common_bl_ind[blind[sortind]],:,j].T)**2 + NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'][blind[sortind],:].T)**2), origin='lower', extent=(0, blind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), vmin=-10.0, vmax=10.0, interpolation=None)
#         l = ax.plot([], [], 'k:', [], [], 'k:', [], [], 'k--', [], [], 'k--')
#         ax.set_aspect('auto')
#         faxs += [ax]
    
#         ax = fig.add_subplot(3,3,blo_ax_mapping[i+n_bins_baseline_orientation])
#         if backdrop_coords == 'radec':
#             ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#             ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#         elif backdrop_coords == 'dircos':
#             ax.set_xlabel('l')
#             ax.set_ylabel('m')
#         imdmap = ax.imshow(1e6 * OPS.reverse(overlays[j]['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
#         imdmappbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
#         # imdmap.set_clim(mindelay, maxdelay)
#         ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='medium')
#         ax.grid(True)
#         ax.tick_params(which='major', length=12, labelsize=12)
#         ax.tick_params(which='minor', length=6)
#         ax.locator_params(axis='x', nbins=5)
#         faxs += [ax]
    
#     cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
#     cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
#     # cbaxt.set_xlabel('Jy', labelpad=-50, fontsize=18)
    
#     # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
#     # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
#     # cbmnb = mindelay * 1e6
#     # cbmxb = maxdelay * 1e6
#     cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
#     cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
#     cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-45, fontsize=18)
    
#     ax = fig.add_subplot(3,3,5)
#     # imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
#     impbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
#     if use_CSM or use_NVSS or use_SUMSS or use_PS:
#         imsky2 = ax.scatter(ra_deg_wrapped[overlays[j]['src_ind']].ravel(), dec_deg[overlays[j]['src_ind']].ravel(), c=overlays[j]['pbeam_on_src']*fluxes[overlays[j]['src_ind']], norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CMAP.jet, edgecolor='none', s=10)
#     else:
#         imsky2 = ax.imshow(OPS.reverse((overlays[j]['pbeam']*overlays[j]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-3, vmax=1e3))        
#     ax.set_xlim(xvect.max(), xvect.min())
#     ax.set_ylim(yvect.min(), yvect.max())
#     ax.set_title('Foregrounds', fontsize=18, weight='medium')
#     ax.grid(True)
#     ax.set_aspect('equal')
#     ax.tick_params(which='major', length=12, labelsize=12)
#     ax.tick_params(which='minor', length=6)
#     if backdrop_coords == 'radec':
#         ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#         ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#     elif backdrop_coords == 'dircos':
#         ax.set_xlabel('l')
#         ax.set_ylabel('m')
#     ax.locator_params(axis='x', nbins=5)
    
#     # cbmnc = NP.nanmin(overlays[j]['pbeam']*overlays[j]['backdrop'])
#     # cbmxc = NP.nanmax(overlays[j]['pbeam']*overlays[j]['backdrop'])
#     cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
#     # cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
#     cbarc = fig.colorbar(imsky2, cax=cbaxc, orientation='horizontal')
#     if use_GSM or use_DSM:
#         cbaxc.set_xlabel('Temperature [K]', labelpad=-50, fontsize=18, weight='medium')
#     else:
#         cbaxc.set_xlabel('Flux Density [Jy]', labelpad=-50, fontsize=18, weight='medium')
#     # tick_locator = ticker.MaxNLocator(nbins=21)
#     # cbarc.locator = tick_locator
#     # cbarc.update_ticks()
    
#     faxs += [ax]
#     tpc = faxs[-1].text(0.5, 1.25, r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointings_radec[j,:]) + '\nLST = {0:.2f} hrs'.format(lst[j]), transform=ax.transAxes, fontsize=14, weight='medium', ha='center')
    
#     PLT.tight_layout()
#     fig.subplots_adjust(bottom=0.1)
#     fig.subplots_adjust(top=0.9)

#     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_diff_error_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

# # Plot error in ratio of delay spectra of data to simulations

# for j in xrange(n_snaps):

#     # Determine the baselines common to simulations and data

#     common_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
#     bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

#     fig = PLT.figure(figsize=(10,10))
#     faxs = []
#     for i in xrange(n_bins_baseline_orientation):
#         ax = fig.add_subplot(3,3,blo_ax_mapping[i])
#         ax.set_xlim(0,bloh[i]-1)
#         ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
#         ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='medium')
#         ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
#         blind = blori[blori[i]:blori[i+1]]
#         sortind = NP.argsort(ref_bl_length[common_bl_ind[blind]], kind='heapsort')
#         # imdspec = ax.imshow((NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T)*NP.sqrt(NP.abs(cc_skyvis_lag_err[common_bl_ind[blind[sortind]],:,j].T)**2 + NP.abs(vis_rms_lag[common_bl_ind[blind[sortind]],:,j].T)**2) + NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)*NP.sqrt(NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'][blind[sortind],:].T)**2)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T)**2, origin='lower', extent=(0, blind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-2, vmax=1e2), interpolation=None)
#         imdspec = ax.imshow((NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T)*NP.sqrt(NP.abs(cc_skyvis_lag_err[common_bl_ind[blind[sortind]],:,j].T)**2 + NP.abs(vis_rms_lag[common_bl_ind[blind[sortind]],:,j].T)**2) + NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)*NP.sqrt(NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'][blind[sortind],:].T)**2)) / NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)**2, origin='lower', extent=(0, blind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-2, vmax=1e2), interpolation=None)

#         l = ax.plot([], [], 'k:', [], [], 'k:', [], [], 'k--', [], [], 'k--')
#         ax.set_aspect('auto')
#         faxs += [ax]
    
#         ax = fig.add_subplot(3,3,blo_ax_mapping[i+n_bins_baseline_orientation])
#         if backdrop_coords == 'radec':
#             ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#             ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#         elif backdrop_coords == 'dircos':
#             ax.set_xlabel('l')
#             ax.set_ylabel('m')
#         imdmap = ax.imshow(1e6 * OPS.reverse(overlays[j]['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
#         imdmappbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
#         # imdmap.set_clim(mindelay, maxdelay)
#         ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='medium')
#         ax.grid(True)
#         ax.tick_params(which='major', length=12, labelsize=12)
#         ax.tick_params(which='minor', length=6)
#         ax.locator_params(axis='x', nbins=5)
#         faxs += [ax]
    
#     cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
#     cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
#     # cbaxt.set_xlabel('Jy', labelpad=-50, fontsize=18)
    
#     # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
#     # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
#     # cbmnb = mindelay * 1e6
#     # cbmxb = maxdelay * 1e6
#     cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
#     cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
#     cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-45, fontsize=18)
    
#     ax = fig.add_subplot(3,3,5)
#     # imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
#     impbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
#     if use_CSM or use_NVSS or use_SUMSS or use_PS:
#         imsky2 = ax.scatter(ra_deg_wrapped[overlays[j]['src_ind']].ravel(), dec_deg[overlays[j]['src_ind']].ravel(), c=overlays[j]['pbeam_on_src']*fluxes[overlays[j]['src_ind']], norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CMAP.jet, edgecolor='none', s=10)
#     else:
#         imsky2 = ax.imshow(OPS.reverse((overlays[j]['pbeam']*overlays[j]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-3, vmax=1e3))        
#     ax.set_xlim(xvect.max(), xvect.min())
#     ax.set_ylim(yvect.min(), yvect.max())
#     ax.set_title('Foregrounds', fontsize=18, weight='medium')
#     ax.grid(True)
#     ax.set_aspect('equal')
#     ax.tick_params(which='major', length=12, labelsize=12)
#     ax.tick_params(which='minor', length=6)
#     if backdrop_coords == 'radec':
#         ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#         ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#     elif backdrop_coords == 'dircos':
#         ax.set_xlabel('l')
#         ax.set_ylabel('m')
#     ax.locator_params(axis='x', nbins=5)
    
#     # cbmnc = NP.nanmin(overlays[j]['pbeam']*overlays[j]['backdrop'])
#     # cbmxc = NP.nanmax(overlays[j]['pbeam']*overlays[j]['backdrop'])
#     cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
#     # cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
#     cbarc = fig.colorbar(imsky2, cax=cbaxc, orientation='horizontal')
#     if use_GSM or use_DSM:
#         cbaxc.set_xlabel('Temperature [K]', labelpad=-50, fontsize=18, weight='medium')
#     else:
#         cbaxc.set_xlabel('Flux Density [Jy]', labelpad=-50, fontsize=18, weight='medium')
#     # tick_locator = ticker.MaxNLocator(nbins=21)
#     # cbarc.locator = tick_locator
#     # cbarc.update_ticks()
    
#     faxs += [ax]
#     tpc = faxs[-1].text(0.5, 1.25, r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointings_radec[j,:]) + '\nLST = {0:.2f} hrs'.format(lst[j]), transform=ax.transAxes, fontsize=14, weight='medium', ha='center')
    
#     PLT.tight_layout()
#     fig.subplots_adjust(bottom=0.1)
#     fig.subplots_adjust(top=0.9)

#     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_ratio_error_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

# # Plot deviation of ratio from unity divided by error in ratio of delay spectra of data to simulations

# for j in xrange(n_snaps):

#     # Determine the baselines common to simulations and data

#     common_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
#     bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

#     fig = PLT.figure(figsize=(10,10))
#     faxs = []
#     for i in xrange(n_bins_baseline_orientation):
#         ax = fig.add_subplot(3,3,blo_ax_mapping[i])
#         ax.set_xlim(0,bloh[i]-1)
#         ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
#         ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='medium')
#         ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
#         blind = blori[blori[i]:blori[i+1]]
#         sortind = NP.argsort(ref_bl_length[common_bl_ind[blind]], kind='heapsort')
#         # imdspec = ax.imshow((NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T)*NP.sqrt(NP.abs(cc_skyvis_lag_err[common_bl_ind[blind[sortind]],:,j].T)**2 + NP.abs(vis_rms_lag[common_bl_ind[blind[sortind]],:,j].T)**2) + NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)*NP.sqrt(NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'][blind[sortind],:].T)**2)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T)**2, origin='lower', extent=(0, blind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-2, vmax=1e2), interpolation=None)
#         imdspec = ax.imshow((NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T) - 1.0)/((NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][blind[sortind],:,0].T)*NP.sqrt(NP.abs(cc_skyvis_lag_err[common_bl_ind[blind[sortind]],:,j].T)**2 + NP.abs(vis_rms_lag[common_bl_ind[blind[sortind]],:,j].T)**2) + NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)*NP.sqrt(NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'][blind[sortind],:].T)**2)) / NP.abs(asm_cc_vis_lag[common_bl_ind[blind[sortind]],:,j].T)**2), origin='lower', extent=(0, blind.size-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), vmin=-10, vmax=10, interpolation=None)

#         l = ax.plot([], [], 'k:', [], [], 'k:', [], [], 'k--', [], [], 'k--')
#         ax.set_aspect('auto')
#         faxs += [ax]
    
#         ax = fig.add_subplot(3,3,blo_ax_mapping[i+n_bins_baseline_orientation])
#         if backdrop_coords == 'radec':
#             ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#             ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#         elif backdrop_coords == 'dircos':
#             ax.set_xlabel('l')
#             ax.set_ylabel('m')
#         imdmap = ax.imshow(1e6 * OPS.reverse(overlays[j]['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
#         imdmappbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
#         # imdmap.set_clim(mindelay, maxdelay)
#         ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='medium')
#         ax.grid(True)
#         ax.tick_params(which='major', length=12, labelsize=12)
#         ax.tick_params(which='minor', length=6)
#         ax.locator_params(axis='x', nbins=5)
#         faxs += [ax]
    
#     cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
#     cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
#     # cbaxt.set_xlabel('Jy', labelpad=-50, fontsize=18)
    
#     # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
#     # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
#     # cbmnb = mindelay * 1e6
#     # cbmxb = maxdelay * 1e6
#     cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
#     cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
#     cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-45, fontsize=18)
    
#     ax = fig.add_subplot(3,3,5)
#     # imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
#     impbc = ax.contour(xgrid[0,:], ygrid[:,0], overlays[j]['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
#     if use_CSM or use_NVSS or use_SUMSS or use_PS:
#         imsky2 = ax.scatter(ra_deg_wrapped[overlays[j]['src_ind']].ravel(), dec_deg[overlays[j]['src_ind']].ravel(), c=overlays[j]['pbeam_on_src']*fluxes[overlays[j]['src_ind']], norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CMAP.jet, edgecolor='none', s=10)
#     else:
#         imsky2 = ax.imshow(OPS.reverse((overlays[j]['pbeam']*overlays[j]['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-3, vmax=1e3))        
#     ax.set_xlim(xvect.max(), xvect.min())
#     ax.set_ylim(yvect.min(), yvect.max())
#     ax.set_title('Foregrounds', fontsize=18, weight='medium')
#     ax.grid(True)
#     ax.set_aspect('equal')
#     ax.tick_params(which='major', length=12, labelsize=12)
#     ax.tick_params(which='minor', length=6)
#     if backdrop_coords == 'radec':
#         ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#         ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#     elif backdrop_coords == 'dircos':
#         ax.set_xlabel('l')
#         ax.set_ylabel('m')
#     ax.locator_params(axis='x', nbins=5)
    
#     # cbmnc = NP.nanmin(overlays[j]['pbeam']*overlays[j]['backdrop'])
#     # cbmxc = NP.nanmax(overlays[j]['pbeam']*overlays[j]['backdrop'])
#     cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
#     # cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
#     cbarc = fig.colorbar(imsky2, cax=cbaxc, orientation='horizontal')
#     if use_GSM or use_DSM:
#         cbaxc.set_xlabel('Temperature [K]', labelpad=-50, fontsize=18, weight='medium')
#     else:
#         cbaxc.set_xlabel('Flux Density [Jy]', labelpad=-50, fontsize=18, weight='medium')
#     # tick_locator = ticker.MaxNLocator(nbins=21)
#     # cbarc.locator = tick_locator
#     # cbarc.update_ticks()
    
#     faxs += [ax]
#     tpc = faxs[-1].text(0.5, 1.25, r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointings_radec[j,:]) + '\nLST = {0:.2f} hrs'.format(lst[j]), transform=ax.transAxes, fontsize=14, weight='medium', ha='center')
    
#     PLT.tight_layout()
#     fig.subplots_adjust(bottom=0.1)
#     fig.subplots_adjust(top=0.9)

#     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_ratio_deviation_error_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)
