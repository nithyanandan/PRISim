import numpy as NP
import copy
from astropy.io import fits
from astropy.io import ascii
from astropy import coordinates as coord
from astropy.coordinates import Galactic, FK5
from astropy import units
import astropy.cosmology as CP
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
from matplotlib.ticker import FuncFormatter
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

latitude = -26.701 
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

max_bl_length = None # Maximum baseline length (in m)
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ref_bl, ref_bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
ref_bl_length = NP.sqrt(NP.sum(ref_bl**2, axis=1))
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True) 
neg_ref_bl_orientation_ind = ref_bl_orientation < 0.0
ref_bl[neg_ref_bl_orientation_ind,:] = -1.0 * ref_bl[neg_ref_bl_orientation_ind,:]
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True)
sortind = NP.argsort(ref_bl_length, kind='mergesort')
ref_bl = ref_bl[sortind,:]
ref_bl_length = ref_bl_length[sortind]
ref_bl_orientation = ref_bl_orientation[sortind]
ref_bl_id = ref_bl_id[sortind]
n_bins_baseline_orientation = 4
nmax_baselines = 2048
ref_bl = ref_bl[:nmax_baselines,:]
ref_bl_length = ref_bl_length[:nmax_baselines]
ref_bl_id = ref_bl_id[:nmax_baselines]
ref_bl_orientation = ref_bl_orientation[:nmax_baselines]
total_baselines = ref_bl_length.size
Tsys = 95.0 # System temperature in K
freq = 185.0e6 # center frequency in Hz
redshift = CNST.rest_freq_HI / freq - 1
oversampling_factor = 2.0
n_sky_sectors = 1
sky_sector = None # if None, use all sky sector. Accepted values are None, 0, 1, 2, or 3
if sky_sector is None:
    sky_sector_str = '_all_sky_'
    n_sky_sectors = 1
    sky_sector = 0
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(sky_sector)

n_bl_chunks = 32
baseline_chunk_size = 64
total_baselines = ref_bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]

truncated_ref_bl_length = NP.copy(ref_bl_length)
truncated_ref_bl = NP.copy(ref_bl)
truncated_ref_bl_ind = NP.arange(ref_bl_length.size)
truncated_ref_bl_id = NP.copy(ref_bl_id)
truncated_ref_bl_orientation = NP.copy(ref_bl_orientation)
truncated_total_baselines = truncated_ref_bl_length.size
if max_bl_length is not None:
    truncated_ref_bl_ind = ref_bl_length <= max_bl_length
    truncated_ref_bl = truncated_ref_bl[truncated_ref_bl_ind,:]
    truncated_ref_bl_id = truncated_ref_bl_id[truncated_ref_bl_ind]
    truncated_ref_bl_orientation = truncated_ref_bl_orientation[truncated_ref_bl_ind]
    truncated_ref_bl_length = truncated_ref_bl_length[truncated_ref_bl_ind]
    truncated_total_baselines = truncated_ref_bl_length.size

bl_orientation_str = ['South-East', 'East', 'North-East', 'North']

spindex_rms = 0.0
spindex_seed = 95
spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

use_alt_spindex = False
alt_spindex_rms = 0.3
alt_spindex_seed = 95
alt_spindex_seed_str = ''
if alt_spindex_rms > 0.0:
    alt_spindex_rms_str = '{0:.1f}'.format(alt_spindex_rms)
else:
    alt_spindex_rms = 0.0

if alt_spindex_seed is not None:
    alt_spindex_seed_str = '{0:0d}_'.format(alt_spindex_seed)

nside = 64
use_GSM = True
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False

obs_mode = 'custom'
avg_drifts = False
beam_switch = True
snapshot_type_str = ''
if avg_drifts:
    snapshot_type_str = 'drift_averaged_'
if beam_switch:
    snapshot_type_str = 'beam_switches_'

freq_resolution = 80e3
nchan = 384
bpass_shape = 'bhw'
max_abs_delay = 1.5 # in micro seconds
coarse_channel_resolution = 1.28e6 # in Hz

dsm_base_freq = 408e6 # Haslam map frequency
csm_base_freq = 1.420e9 # NVSS frequency
dsm_dalpha = 0.7/2 # Spread in spectral index in Haslam map
csm_dalpha = 0.7/2 # Spread in spectral index in NVSS
csm_jacobian_spindex = NP.abs(csm_dalpha * NP.log(freq/csm_base_freq))
dsm_jacobian_spindex = NP.abs(dsm_dalpha * NP.log(freq/dsm_base_freq))

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

roifile = '/data3/t_nithyanandan/project_MWA/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
roi = RI.ROI_parameters(init_file=roifile)
telescope = roi.telescope

pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'

cosmodel = CP.FlatLambdaCDM(H0=100.0, Om0=0.27)

dspec_min = None
dspec_max = None

def kprll(eta, z):
    return 2 * NP.pi * eta * cosmodel.H0.value * CNST.rest_freq_HI * cosmodel.efunc(z) / FCNST.c / (1+z)**2 * 1e3

def kperp(u, z):
    return 2 * NP.pi * u / cosmodel.comoving_transverse_distance(z).value

##########################################

infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)

asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

dsm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_dsm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

csm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_csm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

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

hdulist = fits.open(dsm_CLEAN_infile+'.fits')
dsm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
dsm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
dsm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
dsm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
hdulist.close()

hdulist = fits.open(csm_CLEAN_infile+'.fits')
csm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
csm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
csm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
csm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
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

dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

dsm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
dsm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis_res, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
dsm_cc_skyvis_lag = dsm_cc_skyvis_lag + dsm_ccres_sky

dsm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
dsm_ccres = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis_res, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
dsm_cc_vis_lag = dsm_cc_vis_lag + dsm_ccres

csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
csm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

csm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
csm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis_res, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
csm_cc_skyvis_lag = csm_cc_skyvis_lag + csm_ccres_sky

csm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
csm_ccres = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis_res, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
csm_cc_vis_lag = csm_cc_vis_lag + csm_ccres

asm_cc_skyvis_lag = DSP.downsampler(asm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
asm_cc_vis_lag = DSP.downsampler(asm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
clean_lags = clean_lags.ravel()

vis_noise_lag = NP.copy(ia.vis_noise_lag)
vis_noise_lag = vis_noise_lag[truncated_ref_bl_ind,:,:]
asm_cc_skyvis_lag = asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
asm_cc_vis_lag = asm_cc_vis_lag[truncated_ref_bl_ind,:,:]
csm_cc_skyvis_lag = csm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
csm_cc_vis_lag = csm_cc_vis_lag[truncated_ref_bl_ind,:,:]
dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
dsm_cc_vis_lag = dsm_cc_vis_lag[truncated_ref_bl_ind,:,:]

delaymat = DLY.delay_envelope(ia.baselines[truncated_ref_bl_ind,:], pc, units='mks')
bw = nchan * freq_resolution
min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
max_delay = delaymat[0,:,0]-delaymat[0,:,1]
clags = clean_lags.reshape(1,-1)
min_delay = min_delay.reshape(-1,1)
max_delay = max_delay.reshape(-1,1)
thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
thermal_noise_window = NP.repeat(thermal_noise_window, ia.baselines[truncated_ref_bl_ind,:].shape[0], axis=0)
EoR_window = NP.logical_or(clags > max_delay+3/bw, clags < min_delay-3/bw)
strict_EoR_window = NP.logical_and(EoR_window, NP.abs(clags) < 1/coarse_channel_resolution)
wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
non_wedge_window = NP.logical_not(wedge_window)
vis_rms_lag = OPS.rms(asm_cc_vis_lag, mask=NP.logical_not(NP.repeat(thermal_noise_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
vis_rms_lag_theory = OPS.rms(vis_noise_lag, mask=NP.logical_not(NP.repeat(EoR_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy

if (dspec_min is None) or (dspec_max is None):
    dspec_max = max([NP.abs(asm_cc_skyvis_lag).max(), NP.abs(dsm_cc_skyvis_lag).max(), NP.abs(csm_cc_skyvis_lag).max()])
    dspec_min = min([NP.abs(asm_cc_skyvis_lag).min(), NP.abs(dsm_cc_skyvis_lag).min(), NP.abs(csm_cc_skyvis_lag).min()])

small_delays_EoR_window = EoR_window.T
small_delays_strict_EoR_window = strict_EoR_window.T
small_delays_wedge_window = wedge_window.T

if max_abs_delay is not None:
    small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
    clean_lags = clean_lags[small_delays_ind]
    asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
    asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]
    dsm_cc_vis_lag = dsm_cc_vis_lag[:,small_delays_ind,:]
    dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
    csm_cc_vis_lag = csm_cc_vis_lag[:,small_delays_ind,:]
    csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]
    small_delays_EoR_window = small_delays_EoR_window[small_delays_ind,:]
    small_delays_strict_EoR_window = small_delays_strict_EoR_window[small_delays_ind,:]
    small_delays_wedge_window = small_delays_wedge_window[small_delays_ind,:]

# msk = NP.zeros((truncated_ref_bl_length.size, clean_lags.size, n_snaps))
msk = NP.logical_not(NP.logical_and(small_delays_strict_EoR_window.T[:,:,NP.newaxis],NP.ones((1,1,n_snaps))))

bl_fg_contamination = NP.zeros((3,ia.baselines.shape[0],n_snaps), dtype=NP.complex)

bl_fg_contamination[0,:,:] = NP.squeeze(OPS.rms(dsm_cc_skyvis_lag, mask=msk, axis=1))
bl_fg_contamination[1,:,:] = NP.squeeze(OPS.rms(csm_cc_skyvis_lag, mask=msk, axis=1))
bl_fg_contamination[2,:,:] = NP.squeeze(OPS.rms(asm_cc_skyvis_lag, mask=msk, axis=1))

bl_fg_contamination[NP.abs(bl_fg_contamination <= 0.0)] = NP.nan

# max_fg_cont = NP.nanmax(NP.abs(bl_fg_contamination[:,:1000,:]))
max_fg_cont = 1e7
min_fg_cont = NP.nanmin(NP.abs(bl_fg_contamination[:,:1000,:]))
# min_fg_cont = 1e4

ha = pointings_hadec[:,0]/15.0
ha[ha>12.0] = ha[ha>12.0] - 24
for i in xrange(int(NP.ceil(n_snaps/3.0))):
    fig, axs = PLT.subplots(3, 3, sharey=True, sharex=True, figsize=(8,11))
    # fig, axs = PLT.subplots(min(3,n_snaps-3*i), 3, sharey=True, figsize=(8,11))
    axs = axs.reshape(-1,3)
    for j in xrange(i*3, min((i+1)*3, n_snaps)):

        dsm_fgcont = axs[j%3,0].scatter(ia.baselines[:,0], ia.baselines[:,1], c=NP.abs(bl_fg_contamination[0,:,j]), norm=PLTC.LogNorm(vmin=min_fg_cont, vmax=max_fg_cont), cmap=CM.jet, edgecolor='none', s=5)
        axs[j%3,0].set_aspect('equal')
        axs[j%3,0].set_xlim(-10,90)
        axs[j%3,0].set_ylim(-75,75)
        axs[j%3,0].text(0.5, 0.95, 'DSM', transform=axs[j%3,0].transAxes, fontsize=12, weight='medium', ha='left', va='top')

        csm_fgcont = axs[j%3,1].scatter(ia.baselines[:,0], ia.baselines[:,1], c=NP.abs(bl_fg_contamination[1,:,j]), norm=PLTC.LogNorm(vmin=min_fg_cont, vmax=max_fg_cont), cmap=CM.jet, edgecolor='none', s=5)
        axs[j%3,1].set_aspect('equal')
        axs[j%3,1].set_xlim(-10,90)
        axs[j%3,1].set_ylim(-75,75)
        axs[j%3,1].text(0.5, 0.95, 'CSM', transform=axs[j%3,1].transAxes, fontsize=12, weight='medium', ha='left', va='top')

        asm_fgcont = axs[j%3,2].scatter(ia.baselines[:,0], ia.baselines[:,1], c=NP.abs(bl_fg_contamination[2,:,j]), norm=PLTC.LogNorm(vmin=min_fg_cont, vmax=max_fg_cont), cmap=CM.jet, edgecolor='none', s=5)
        axs[j%3,2].set_aspect('equal')
        axs[j%3,2].set_xlim(-10,90)
        axs[j%3,2].set_ylim(-75,75)
        axs[j%3,2].text(0.5, 0.95, 'DSM + CSM', transform=axs[j%3,2].transAxes, fontsize=12, weight='medium', ha='left', va='top')

        axs[j%3,2].text(1.05, 0.5, 'LST = {0:.2f} hrs,\n HA = {1:.2f} hrs'.format((lst[j]/15.0)%24, ha[j]), fontsize=16, weight='medium', ha='left', va='center', transform=axs[j%3,2].transAxes, rotation=90)

    fig.subplots_adjust(wspace=0, hspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_xlabel('x [m]', fontsize=20, weight='semibold', labelpad=20)
    big_ax.set_ylabel('y [m]', fontsize=20, weight='semibold', labelpad=30)

    cbax = fig.add_axes([0.12, 0.95, 0.78, 0.02])
    cbar = fig.colorbar(asm_fgcont, cax=cbax, orientation='horizontal')
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/foreground_contaminated_baselines_snaps_{0:0d}-{1:0d}.png'.format(3*i+1, min(n_snaps,3*(i+1))), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/foreground_contaminated_baselines_snaps_{0:0d}-{1:0d}.eps'.format(3*i+1, min(n_snaps,3*(i+1))), bbox_inches=0)
    PLT.close(fig)






