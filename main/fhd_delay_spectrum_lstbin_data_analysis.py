import glob
import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import astropy.cosmology as CP
import scipy.constants as FCNST
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
import progressbar as PGB
import interferometry as RI
import my_operations as OPS
import geometry as GEOM
import constants as CNST
import my_DSP_modules as DSP 
import baseline_delay_horizon as DLY
import ipdb as PDB

# rootdir = '/data3/t_nithyanandan/'
rootdir = '/data3/MWA/lstbin_RA0/NT/'

filenaming_convention = 'new'
# filenaming_convention = 'old'

project_MWA = False
project_LSTbin = True
project_HERA = False
project_beams = False
project_drift_scan = False
project_global_EoR = False

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
elif telescope_id == 'mwa_tools':
    pass
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

delayerr = 0.05     # delay error rms in ns
if delayerr is None:
    delayerr_str = ''
    delayerr = 0.0
elif delayerr < 0.0:
    raise ValueError('delayerr must be non-negative.')
else:
    delayerr_str = 'derr_{0:.3f}ns'.format(delayerr)
delayerr *= 1e-9

gainerr = 0.0      # Gain error rms in dB
if gainerr is None:
    gainerr_str = ''
    gainerr = 0.0
elif gainerr < 0.0:
    raise ValueError('gainerr must be non-negative.')
else:
    gainerr_str = '_gerr_{0:.2f}dB'.format(gainerr)

nrand = 1       # Number of random realizations
if nrand is None:
    nrandom_str = ''
    nrand = 1
elif nrand < 1:
    raise ValueError('nrandom must be positive')
else:
    nrandom_str = '_nrand_{0:0d}_'.format(nrand)

if (delayerr_str == '') and (gainerr_str == ''):
    nrand = 1
    nrandom_str = ''

delaygain_err_str = delayerr_str + gainerr_str + nrandom_str

project_dir = ''
if project_MWA: project_dir = 'project_MWA/'
if project_LSTbin:
    if rootdir == '/data3/t_nithyanandan/':
        project_dir = 'project_LSTbin/'
if project_HERA: project_dir = 'project_HERA/'
if project_beams: project_dir = 'project_beams/'
if project_drift_scan: project_dir = 'project_drift_scan/'
if project_global_EoR: project_dir = 'project_global_EoR/'

fhd_indir = '/data3/MWA/lstbin_RA0/NT/'
fhd_infile_prefix = 'fhd_delay_spectrum_'
fhd_infile_suffix = '_reformatted.npz'

fhd_infiles = glob.glob(fhd_indir+fhd_infile_prefix+'*'+fhd_infile_suffix)
fhd_infiles_filenames = [fhd_infile.split('/')[-1] for fhd_infile in fhd_infiles]
fhd_infiles_obsid = [filename.split('_')[3] for filename in fhd_infiles_filenames]
fhd_obsid = fhd_infiles_obsid
fhd_obsid_flagged = ['1062781088', '1064590536']

latitude = -26.701 
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

max_bl_length = 200.0 # Maximum baseline length (in m)
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
# if max_bl_length is not None:
#     ref_bl_ind = ref_bl_length <= max_bl_length
#     ref_bl = ref_bl[ref_bl_ind,:]
#     ref_bl_id = ref_bl_id[ref_bl_ind]
#     ref_bl_orientation = ref_bl_orientation[ref_bl_ind]
#     ref_bl_length = ref_bl_length[ref_bl_ind]
#     total_baselines = ref_bl_length.size

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
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]

truncated_ref_bl = NP.copy(ref_bl)
truncated_ref_bl_id = NP.copy(ref_bl_id)
truncated_ref_bl_length = NP.sqrt(NP.sum(truncated_ref_bl[:,:2]**2, axis=1))
# truncated_ref_bl_length = NP.copy(ref_bl_length)
truncated_ref_bl_orientation = NP.copy(ref_bl_orientation)
truncated_total_baselines = truncated_ref_bl_length.size
if max_bl_length is not None:
    truncated_ref_bl_ind = ref_bl_length <= max_bl_length
    truncated_ref_bl = truncated_ref_bl[truncated_ref_bl_ind,:]
    truncated_ref_bl_id = truncated_ref_bl_id[truncated_ref_bl_ind]
    truncated_ref_bl_orientation = truncated_ref_bl_orientation[truncated_ref_bl_ind]
    truncated_ref_bl_length = truncated_ref_bl_length[truncated_ref_bl_ind]
    truncated_total_baselines = truncated_ref_bl_length.size

spindex_rms = 0.0
spindex_seed = None
spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

nside = 64

Tsys = 95.0 # System temperature in K
freq = 185.0e6 # center frequency in Hz
freq_resolution = 80e3  # in kHz
nchan = 384
bpass_shape = 'bhw'
max_abs_delay = 1.5 # in micro seconds
coarse_channel_resolution = 1.28e6 # in Hz
bw = nchan * freq_resolution
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)

redshift = CNST.rest_freq_HI / freq - 1
wavelength = FCNST.c / freq  # in meters
A_eff = 16 * (0.5*wavelength)**2

use_pfb = True

pfb_str1 = ''
pfb_str2 = ''
if not use_pfb: 
    pfb_str1 = '_no_pfb'
    pfb_str2 = 'no_pfb_'

# obs_mode = 'custom'
obs_mode = 'lstbin'
avg_drifts = False
beam_switch = False
snapshots_range = None
snapshot_type_str = ''
if avg_drifts:
    snapshot_type_str = 'drift_averaged_'
if beam_switch:
    snapshot_type_str = 'beam_switches_'
if snapshots_range is not None:
    snapshot_type_str = 'snaps_{0[0]:0d}-{0[1]:0d}_'.format(snapshots_range)

duration_str = ''
if obs_mode in ['track', 'drift']:
    t_snap = 1080.0    # in seconds
    n_snaps = 40
    if (t_snap is not None) and (n_snaps is not None):
        duration_str = '_{0:0d}x{1:.1f}s'.format(n_snaps, t_snap[0])

pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'

h = 0.7   # Hubble constant coefficient
cosmodel100 = CP.FlatLambdaCDM(H0=100.0, Om0=0.27)  # Using H0 = 100 km/s/Mpc
cosmodel = CP.FlatLambdaCDM(H0=h*100.0, Om0=0.27)  # Using H0 = h * 100 km/s/Mpc

dr_z = (FCNST.c/1e3) * bw * (1+redshift)**2 / CNST.rest_freq_HI / cosmodel100.H0.value / cosmodel100.efunc(redshift)   # in Mpc/h
r_z = cosmodel100.comoving_transverse_distance(redshift).value   # in Mpc/h

volfactor1 = A_eff / wavelength**2 / bw
volfactor2 = r_z**2 * dr_z / bw

Jy2K = wavelength**2 * CNST.Jy / (2*FCNST.k)
mJy2mK = NP.copy(Jy2K)
Jy2mK = 1e3 * Jy2K

mK2Jy = 1/Jy2mK
mK2mJy = 1/mJy2mK
K2Jy = 1/Jy2K

def kprll(eta, z):
    return 2 * NP.pi * eta * cosmodel100.H0.value * CNST.rest_freq_HI * cosmodel100.efunc(z) / FCNST.c / (1+z)**2 * 1e3

def kperp(u, z):
    return 2 * NP.pi * u / cosmodel100.comoving_transverse_distance(z).value

# PLT.ion()

## Read in the observation information file

pointing_file = '/data3/MWA/lstbin_RA0/EoR0_high_sem1_1_obsinfo.txt'
pointing_info_from_file = NP.loadtxt(pointing_file, comments='#', usecols=(1,2,3), delimiter=',')
obs_id = NP.loadtxt(pointing_file, comments='#', usecols=(0,), delimiter=',', dtype=str)
obsfile_lst = 15.0 * pointing_info_from_file[:,2]
obsfile_pointings_altaz = pointing_info_from_file[:,:2].reshape(-1,2)
obsfile_pointings_dircos = GEOM.altaz2dircos(obsfile_pointings_altaz, units='degrees')
obsfile_pointings_hadec = GEOM.altaz2hadec(obsfile_pointings_altaz, latitude, units='degrees')

## Read in the simulations

infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_str1
asm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_str2+bpass_shape
dsm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'dsm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_str2+bpass_shape
csm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'csm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_str2+bpass_shape

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

hdulist = fits.open(dsm_CLEAN_infile+'.fits')
dsm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
dsm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
dsm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
dsm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
hdulist.close()

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

hdulist = fits.open(csm_CLEAN_infile+'.fits')
csm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
csm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
csm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
csm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
hdulist.close()

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

dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)

dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
dsm_cc_vis_lag = dsm_cc_vis_lag[truncated_ref_bl_ind,:,:]
csm_cc_skyvis_lag = csm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
csm_cc_vis_lag = csm_cc_vis_lag[truncated_ref_bl_ind,:,:]

clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
clean_lags = clean_lags.ravel()
if max_abs_delay is not None:
    small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
    clean_lags = clean_lags[small_delays_ind]

    asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
    dsm_cc_vis_lag = dsm_cc_vis_lag[:,small_delays_ind,:]
    csm_cc_vis_lag = csm_cc_vis_lag[:,small_delays_ind,:]

    asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]
    dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
    csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]

avg_asm_cc_skyvis_lag = NP.mean(asm_cc_skyvis_lag, axis=2)

## Read in FHD data and other required information

common_ref_bl_id = NP.copy(ref_bl_id)
for j in range(1,len(fhd_obsid)):
    fhd_infile = fhd_infiles[j]
    fhd_data = NP.load(fhd_infile)
    fhd_vis_lag_noisy = fhd_data['fhd_vis_lag_noisy']
    fhd_C = fhd_data['fhd_C']
    valid_ind = NP.logical_and(NP.abs(NP.sum(fhd_vis_lag_noisy[:,:,0],axis=1))!=0.0, NP.abs(NP.sum(fhd_C[:,:,0],axis=1))!=0.0)
    fhdfile_bl_id = fhd_data['fhd_bl_id'][valid_ind]
    common_bl_id = NP.intersect1d(common_ref_bl_id, fhdfile_bl_id, assume_unique=True)
    common_bl_ind_in_ref = NP.in1d(common_ref_bl_id, common_bl_id, assume_unique=True)
    common_ref_bl_id = common_ref_bl_id[common_bl_ind_in_ref]

common_bl_ind_in_ref = NP.where(NP.in1d(ref_bl_id, common_ref_bl_id, assume_unique=True))[0]

fhd_info = {}
progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' [', right='] '), PGB.Counter(), '/{0:0d} snapshots '.format(len(fhd_obsid)), PGB.ETA()], maxval=len(fhd_obsid)).start()
for j in range(len(fhd_obsid)):
    fhd_infile = fhd_infiles[j]
    fhd_data = NP.load(fhd_infile)
    fhd_vis_lag_noisy = fhd_data['fhd_vis_lag_noisy']
    fhd_vis_lag_res = fhd_data['fhd_vis_lag_res']    
    fhd_C = fhd_data['fhd_C']
    valid_ind = NP.logical_and(NP.abs(NP.sum(fhd_vis_lag_noisy[:,:,0],axis=1))!=0.0, NP.abs(NP.sum(fhd_C[:,:,0],axis=1))!=0.0)
    fhd_C = fhd_C[valid_ind,:,:]
    fhd_vis_lag_noisy = fhd_vis_lag_noisy[valid_ind,:,:]
    fhd_vis_lag_res = fhd_vis_lag_res[valid_ind,:,:]    
    fhd_delays = fhd_data['fhd_delays']
    fhdfile_bl_id = fhd_data['fhd_bl_id'][valid_ind]
    fhdfile_bl_length = fhd_data['fhd_bl_length'][valid_ind]
    common_bl_id = NP.intersect1d(common_ref_bl_id, fhdfile_bl_id, assume_unique=True)
    # common_bl_ind_in_ref = NP.in1d(common_ref_bl_id, common_bl_id, assume_unique=True)
    common_bl_ind_in_fhd = NP.in1d(fhdfile_bl_id, common_bl_id, assume_unique=True)
    fhd_bl_id = fhdfile_bl_id[common_bl_ind_in_fhd]
    # fhd_bl_length = fhdfile_bl_length[common_bl_ind_in_fhd]
    fhd_bl_length = ref_bl_length[common_bl_ind_in_ref]
    fhd_k_perp = 2 * NP.pi * fhd_bl_length / (FCNST.c/freq) / cosmodel100.comoving_transverse_distance(z=redshift).value
    fhd_bl = ref_bl[common_bl_ind_in_ref, :]
    fhd_bl_orientation = ref_bl_orientation[common_bl_ind_in_ref]
    # common_bl_ind_in_ref_snapshots += [common_bl_ind_in_ref]

    fhd_neg_bl_orientation_ind = fhd_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    fhd_bl_orientation[fhd_neg_bl_orientation_ind] -= 180.0
    fhd_bl[fhd_neg_bl_orientation_ind,:] = -fhd_bl[fhd_neg_bl_orientation_ind,:]

    fhd_C = fhd_C[common_bl_ind_in_fhd,:,:]
    fhd_vis_lag_noisy = fhd_vis_lag_noisy[common_bl_ind_in_fhd,:,:]*nchan*freq_resolution/fhd_C
    fhd_vis_lag_res = fhd_vis_lag_res[common_bl_ind_in_fhd,:,:]*nchan*freq_resolution/fhd_C
    # fhd_vis_lag_noisy = fhd_vis_lag_noisy[common_bl_ind_in_fhd,:,:]*2.78*nchan*freq_resolution/fhd_C
    # fhd_vis_lag_res = fhd_vis_lag_res[common_bl_ind_in_fhd,:,:]*2.78*nchan*freq_resolution/fhd_C    
    
    fhd_obsid_pointing_dircos = obsfile_pointings_dircos[obs_id==fhd_obsid[j],:].reshape(1,-1)
    fhd_obsid_pointing_altaz = obsfile_pointings_altaz[obs_id==fhd_obsid[j],:].reshape(1,-1)
    fhd_obsid_pointing_hadec = obsfile_pointings_hadec[obs_id==fhd_obsid[j],:].reshape(1,-1)
    fhd_lst = NP.asscalar(obsfile_lst[obs_id==fhd_obsid[j]])
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
    fhd_non_wedge_window = NP.logical_not(fhd_wedge_window)
    fhd_vis_rms_lag = OPS.rms(fhd_vis_lag_noisy[:,:,0], mask=NP.logical_not(fhd_thermal_noise_window), axis=1)
    fhd_vis_rms_freq = NP.abs(fhd_vis_rms_lag) / NP.sqrt(nchan) / freq_resolution

    # if max_abs_delay is not None:
    #     small_delays_ind = NP.abs(fhd_delays) <= max_abs_delay * 1e-6
    #     fhd_delays = fhd_delays[small_delays_ind]
    #     fhd_vis_lag_noisy = fhd_vis_lag_noisy[:,small_delays_ind,:]
    #     fhd_vis_lag_res = fhd_vis_lag_res[:,small_delays_ind,:]        
    fhd_k_prll = 2 * NP.pi * fhd_delays * cosmodel100.H0.value * CNST.rest_freq_HI * cosmodel100.efunc(z=redshift) / FCNST.c / (1+redshift)**2 * 1e3

    fhd_info[fhd_obsid[j]] = {}
    fhd_info[fhd_obsid[j]]['bl_id'] = fhd_bl_id
    fhd_info[fhd_obsid[j]]['bl'] = fhd_bl
    fhd_info[fhd_obsid[j]]['bl_length'] = fhd_bl_length
    fhd_info[fhd_obsid[j]]['k_perp'] = fhd_k_perp
    fhd_info[fhd_obsid[j]]['bl_orientation'] = fhd_bl_orientation
    fhd_info[fhd_obsid[j]]['delays'] = fhd_delays
    fhd_info[fhd_obsid[j]]['k_prll'] = fhd_k_prll
    fhd_info[fhd_obsid[j]]['C'] = fhd_C
    fhd_info[fhd_obsid[j]]['vis_lag_noisy'] = fhd_vis_lag_noisy
    fhd_info[fhd_obsid[j]]['vis_lag_res'] = fhd_vis_lag_res    
    fhd_info[fhd_obsid[j]]['lst'] = fhd_lst
    fhd_info[fhd_obsid[j]]['pointing_radec'] = fhd_obsid_pointing_radec
    fhd_info[fhd_obsid[j]]['pointing_hadec'] = fhd_obsid_pointing_hadec
    fhd_info[fhd_obsid[j]]['pointing_altaz'] = fhd_obsid_pointing_altaz
    fhd_info[fhd_obsid[j]]['pointing_dircos'] = fhd_obsid_pointing_dircos
    fhd_info[fhd_obsid[j]]['min_delays'] = fhd_min_delay
    fhd_info[fhd_obsid[j]]['max_delays'] = fhd_max_delay
    fhd_info[fhd_obsid[j]]['rms_lag'] = fhd_vis_rms_lag
    fhd_info[fhd_obsid[j]]['rms_freq'] = fhd_vis_rms_freq

    progress.update(j+1)
progress.finish()

fhdblid = fhd_info[fhd_obsid[0]]['bl_id']
fhdbll = fhd_info[fhd_obsid[0]]['bl_length']
fhdblo = fhd_info[fhd_obsid[0]]['bl_orientation']
fhdlags = fhd_info[fhd_obsid[0]]['delays']
sortind = NP.argsort(fhdbll)
fhdbll = fhdbll[sortind]
fhdblo = fhdblo[sortind]
min_horizon_delays = fhd_info[fhd_obsid[0]]['min_delays'].ravel()
max_horizon_delays = fhd_info[fhd_obsid[0]]['max_delays'].ravel()

fhd_vis_lag_noisy_max_pol0 = max([NP.abs(fhd_element['vis_lag_noisy'][:,:,0]).max() for fhd_element in fhd_info.itervalues()])
fhd_vis_lag_noisy_min_pol0 = min([NP.abs(fhd_element['vis_lag_noisy'][:,:,0]).min() for fhd_element in fhd_info.itervalues()])
fhd_vis_lag_noisy_max_pol1 = max([NP.abs(fhd_element['vis_lag_noisy'][:,:,1]).max() for fhd_element in fhd_info.itervalues()])
fhd_vis_lag_noisy_min_pol1 = min([NP.abs(fhd_element['vis_lag_noisy'][:,:,1]).min() for fhd_element in fhd_info.itervalues()])

for j in range(len(fhd_obsid)):
    if j == 0:
        if fhd_obsid[j] not in fhd_obsid_flagged:
            avg_fhd_vis_lag_noisy_pol0 = fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0]
            avg_fhd_vis_lag_noisy_pol1 = fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,1]
    else:
        if fhd_obsid[j] not in fhd_obsid_flagged:
            avg_fhd_vis_lag_noisy_pol0 += fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0]
            avg_fhd_vis_lag_noisy_pol1 += fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,1]
avg_fhd_vis_lag_noisy_pol0 /= len(fhd_obsid)-len(fhd_obsid_flagged)
avg_fhd_vis_lag_noisy_pol1 /= len(fhd_obsid)-len(fhd_obsid_flagged)

avg_fhd_vis_rms_lag_pol0 = OPS.rms(avg_fhd_vis_lag_noisy_pol0, mask=NP.logical_not(fhd_thermal_noise_window), axis=1).ravel()
avg_fhd_vis_rms_lag_pol1 = OPS.rms(avg_fhd_vis_lag_noisy_pol1, mask=NP.logical_not(fhd_thermal_noise_window), axis=1).ravel()

if max_abs_delay is not None:
    small_delays_ind = NP.abs(fhdlags) <= max_abs_delay * 1e-6
    fhdlags = fhdlags[small_delays_ind]
    avg_fhd_vis_lag_noisy_pol0 = avg_fhd_vis_lag_noisy_pol0[:,small_delays_ind]
    avg_fhd_vis_lag_noisy_pol1 = avg_fhd_vis_lag_noisy_pol1[:,small_delays_ind]

avg_fhd_vis_lag_noisy_max_pol0 = NP.abs(avg_fhd_vis_lag_noisy_pol0).max()
avg_fhd_vis_lag_noisy_max_pol1 = NP.abs(avg_fhd_vis_lag_noisy_pol1).max()
avg_fhd_vis_lag_noisy_min_pol0 = NP.abs(avg_fhd_vis_lag_noisy_pol0).min()
avg_fhd_vis_lag_noisy_min_pol1 = NP.abs(avg_fhd_vis_lag_noisy_pol1).min()
    
# Prepare for plots

nrow = 3
ncol = 2
npages = int(NP.ceil(1.0*(len(fhd_obsid) - len(fhd_obsid_flagged))/(nrow*ncol)))

# Plot amplitudes

# for pagenum in range(npages):
#     fig, axs = PLT.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(8,10))
#     axs = axs.reshape(nrow,ncol)
#     for j in range(pagenum*nrow*ncol, min(len(fhd_obsid),(pagenum+1)*nrow*ncol)):
#         jpage = j - pagenum*nrow*ncol

#         imdspec = axs[jpage/ncol,jpage%ncol].imshow(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T), origin='lower', extent=(0,fhdbll.size-1,1e6*fhdlags.min(),1e6*fhdlags.max()), norm=PLTC.LogNorm(vmin=1e6, vmax=fhd_vis_lag_noisy_max_pol0))
#         # horizonb = axs[jpage/ncol,jpage%ncol].plot(NP.arange(fhdbll.size), 1e6*min_horizon_delays[sortind]-3/bw, color='black', ls=':', lw=1.5)
#         # horizont = axs[jpage/ncol,jpage%ncol].plot(NP.arange(fhdbll.size), 1e6*max_horizon_delays[sortind]+3/bw, color='black', ls=':', lw=1.5)
#         axs[jpage/ncol,jpage%ncol].set_xlim(0, fhdbll.size-1)
#         axs[jpage/ncol,jpage%ncol].set_ylim(1e6*fhdlags.min(), 1e6*fhdlags.max())
#         axs[jpage/ncol,jpage%ncol].set_aspect('auto')

#     cbax = fig.add_axes([0.1, 0.93, 0.8, 0.02])
#     cbar = fig.colorbar(imdspec, cax=cbax, orientation='horizontal')
#     cbax.set_xlabel('Jy Hz', labelpad=10, fontsize=12)
#     cbax.xaxis.set_label_position('top')
        
#     fig.subplots_adjust(top=0.88)

#     PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_visibilities_amplitudes_{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+'snapshots_{0:0d}-{1:0d}'.format(pagenum*nrow*ncol, min(len(fhd_obsid),(pagenum+1)*nrow*ncol)-1)+'.png')

# Plot phases

# for pagenum in range(npages):
#     fig, axs = PLT.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(8,10))
#     axs = axs.reshape(nrow,ncol)
#     for j in range(pagenum*nrow*ncol, min(len(fhd_obsid),(pagenum+1)*nrow*ncol)):
#         jpage = j - pagenum*nrow*ncol

#         imdspec = axs[jpage/ncol,jpage%ncol].imshow(NP.angle(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T, deg=True), origin='lower', extent=(0,fhdbll.size-1,1e6*fhdlags.min(),1e6*fhdlags.max()), vmin=-180.0, vmax=180.0)
#         # horizonb = axs[jpage/ncol,jpage%ncol].plot(NP.arange(fhdbll.size), 1e6*min_horizon_delays[sortind]-3/bw, color='black', ls=':', lw=1.5)
#         # horizont = axs[jpage/ncol,jpage%ncol].plot(NP.arange(fhdbll.size), 1e6*max_horizon_delays[sortind]+3/bw, color='black', ls=':', lw=1.5)
#         axs[jpage/ncol,jpage%ncol].set_xlim(0, fhdbll.size-1)
#         axs[jpage/ncol,jpage%ncol].set_ylim(1e6*fhdlags.min(), 1e6*fhdlags.max())
#         axs[jpage/ncol,jpage%ncol].set_aspect('auto')

#     cbax = fig.add_axes([0.1, 0.93, 0.8, 0.02])
#     cbar = fig.colorbar(imdspec, cax=cbax, orientation='horizontal')
#     cbax.set_xlabel('Degrees', labelpad=10, fontsize=12)
#     cbax.xaxis.set_label_position('top')
        
#     fig.subplots_adjust(top=0.88)

#     PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_visibilities_phases_{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+'snapshots_{0:0d}-{1:0d}'.format(pagenum*nrow*ncol, min(len(fhd_obsid),(pagenum+1)*nrow*ncol)-1)+'.png')

# nrow = 3
# ncol = 1
# npages = int(NP.ceil(1.0*len(fhd_obsid)/(nrow*ncol)))

# for pagenum in range(npages):
#     fig, axs = PLT.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(8,10))
#     axs = axs.reshape(nrow,ncol)
#     for j in range(pagenum*nrow*ncol, min(len(fhd_obsid),(pagenum+1)*nrow*ncol)):
#         jpage = j - pagenum*nrow*ncol

#         d0phase = axs[jpage/ncol,jpage%ncol].plot(NP.arange(fhdbll.size), NP.angle(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,int(0.5*fhdlags.size),0].ravel(), deg=True), 'k-')
#         axs[jpage/ncol,jpage%ncol].set_xlim(0, min(1000,fhdbll.size)-1)
#         axs[jpage/ncol,jpage%ncol].set_ylim(-180, 180)
            
#     PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_visibilities_zero_delay_phases_{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+'snapshots_{0:0d}-{1:0d}'.format(pagenum*nrow*ncol, min(len(fhd_obsid),(pagenum+1)*nrow*ncol)-1)+'.png')

# Plot amplitude of averaged delay spectra

# fig = PLT.figure(figsize=(6,4))
# ax = fig.add_subplot(111)

# # imdspec = ax.imshow(NP.abs(avg_fhd_vis_lag_noisy_pol0.T), origin='lower', extent=(0,fhdbll.size-1,1e6*fhdlags.min(),1e6*fhdlags.max()), norm=PLTC.LogNorm(vmin=1e6, vmax=fhd_vis_lag_noisy_max_pol0))
# # horizonb = ax.plot(NP.arange(fhdbll.size), 1e6*min_horizon_delays[sortind]-3/bw, color='black', ls=':', lw=1.5)
# # horizont = ax.plot(NP.arange(fhdbll.size), 1e6*max_horizon_delays[sortind]+3/bw, color='black', ls=':', lw=1.5)
# # ax.set_xlim(0, fhdbll.size-1)

# imdspec = ax.pcolorfast(fhdbll, 1e6*fhdlags, NP.abs(avg_fhd_vis_lag_noisy_pol0[:-1,:-1].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(3e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=fhd_vis_lag_noisy_max_pol0**2 * volfactor1 * volfactor2 * Jy2K**2))
# horizonb = ax.plot(fhdbll, 1e6*min_horizon_delays[sortind], color='white', ls=':', lw=1.5)
# horizont = ax.plot(fhdbll, 1e6*max_horizon_delays[sortind], color='white', ls=':', lw=1.5)
# ax.set_ylim(0.9*1e6*fhdlags.min(), 0.9*1e6*fhdlags.max())
# ax.set_aspect('auto')
# ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
# ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')

# ax_kprll = ax.twinx()
# ax_kprll.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
# ax_kprll.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
# yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
# ax_kprll.yaxis.set_major_formatter(yformatter)

# ax_kperp = ax.twiny()
# ax_kperp.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
# ax_kperp.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
# xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
# ax_kperp.xaxis.set_major_formatter(xformatter)

# ax_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
# ax_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')

# cbax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
# # cbax.set_xlabel('Jy Hz', labelpad=10, fontsize=12)
# cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
# cbax.xaxis.set_label_position('top')

# fig.subplots_adjust(right=0.72)
# fig.subplots_adjust(top=0.85)
# fig.subplots_adjust(bottom=0.15)
    
# PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_avg_visibilities_amplitudes_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.png')
# PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_avg_visibilities_amplitudes_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.eps')

# Plot phase of averaged delay spectra

# fig = PLT.figure(figsize=(6,4))
# ax = fig.add_subplot(111)

# imdspec = ax.imshow(NP.angle(avg_fhd_vis_lag_noisy_pol0.T, deg=True), origin='lower', extent=(0,fhdbll.size-1,1e6*fhdlags.min(),1e6*fhdlags.max()), vmin=-180.0, vmax=180.0)
# # horizonb = ax.plot(NP.arange(fhdbll.size), 1e6*min_horizon_delays[sortind]-3/bw, color='black', ls=':', lw=1.5)
# # horizont = ax.plot(NP.arange(fhdbll.size), 1e6*max_horizon_delays[sortind]+3/bw, color='black', ls=':', lw=1.5)
# ax.set_xlim(0, fhdbll.size-1)
# ax.set_ylim(1e6*fhdlags.min(), 1e6*fhdlags.max())
# ax.set_aspect('auto')

# cbax = fig.add_axes([0.1, 0.92, 0.8, 0.02])
# cbar = fig.colorbar(imdspec, cax=cbax, orientation='horizontal')
# cbax.set_xlabel('Degrees', labelpad=10, fontsize=12)
# cbax.xaxis.set_label_position('top')

# fig.subplots_adjust(top=0.83)
    
# PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_avg_visibilities_phases_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.png')

# Plot zero delay phases of averaged visibilities

# fig = PLT.figure(figsize=(8,4))
# ax = fig.add_subplot(111)

# ax.plot(NP.arange(fhdbll.size), NP.angle(avg_fhd_vis_lag_noisy_pol0[:,int(0.5*fhdlags.size)].ravel(), deg=True), 'k-')
# ax.set_xlim(0, min(1000,fhdbll.size)-1)
# ax.set_ylim(-180, 180)
        
# PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_CLEAN_fhd_avg_visibilities_zero_delay_phases_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.png')

## Plot baseline slices of delay spectra

# select_bl_id = ['78-77', '63-58', '95-51']
# fig, axs = PLT.subplots(len(select_bl_id), sharex=True, sharey=True, figsize=(6,8))

# for j in xrange(len(select_bl_id)):
#     blid = select_bl_id[j]
#     axs[j].plot(1e6*fhdlags, NP.abs(avg_fhd_vis_lag_noisy_pol0[fhdblid == blid,:]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'k-', lw=2)
#     axs[j].plot(1e6*clean_lags, NP.abs(avg_asm_cc_skyvis_lag[ref_bl_id == blid,:]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'r-', lw=2)    
#     axs[j].axvline(x=1e6*min_horizon_delays[fhdblid==blid], ls='--', lw=2, color='black')
#     axs[j].axvline(x=1e6*max_horizon_delays[fhdblid==blid], ls='--', lw=2, color='black')
#     axs[j].axvline(x=1e6/coarse_channel_resolution, ymax=0.6, ls='-.', lw=2, color='black')
#     axs[j].axvline(x=-1e6/coarse_channel_resolution, ymax=0.6, ls='-.', lw=2, color='black')    
#     axs[j].axhline(y=NP.abs(avg_fhd_vis_rms_lag_pol0[fhdblid == blid])**2 * volfactor1 * volfactor2 * Jy2K**2, lw=2, ls=':', color='black')
#     axs[j].text(0.02, 0.88, r'$|\mathbf{b}|$'+' = {0:.1f} m'.format(fhdbll[fhdblid==blid][0]), fontsize=12, weight='medium', transform=axs[j].transAxes)
#     axs[j].text(0.02, 0.79, r'$\theta_b$'+' = {0:+.1f}$^\circ$'.format(fhdblo[fhdblid==blid][0]), fontsize=12, weight='medium', transform=axs[j].transAxes)

#     axs[j].set_yscale('log')
#     axs[j].set_ylim(20.0, 9e7)

#     # axs[j].set_ylim(8.0, avg_fhd_vis_lag_noisy_max_pol0**2 * volfactor1 * volfactor2 * Jy2K**2)
#     # axs[j].set_yticks(NP.logspace(0,10,5,endpoint=True).tolist())

#     if j == 0:
#         axs_kprll = axs[j].twiny()
#         axs_kprll.set_xticks(kprll(axs[j].get_xticks()*1e-6, redshift))
#         axs_kprll.set_xlim(kprll(NP.asarray(axs[j].get_xlim())*1e-6, redshift))
#         xformatter = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
#         axs_kprll.xaxis.set_major_formatter(xformatter)

#     fig.subplots_adjust(hspace=0)
#     big_ax = fig.add_subplot(111)
#     big_ax.set_axis_bgcolor('none')
#     big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_ax.set_xticks([])
#     big_ax.set_yticks([])
#     big_ax.set_xlabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=20)
#     # big_ax.set_ylabel(r'$|V_{b\tau}(\mathbf{b},\tau)|$  [Jy Hz]', fontsize=16, weight='medium', labelpad=30)
#     big_ax.set_ylabel(r"$P_d(k_\perp,k_\parallel)$  [K$^2$ (Mpc/$h)^3$]", fontsize=16, weight='medium', labelpad=28)

#     big_axt = big_ax.twiny()
#     big_axt.set_axis_bgcolor('none')
#     big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_axt.set_xticks([])
#     big_axt.set_yticks([])
#     big_axt.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)

#     PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/{0:0d}_baseline_comparison'.format(len(select_bl_id))+'_CLEAN_fhd_avg_visibilities_amplitudes_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.png')
#     PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/{0:0d}_baseline_comparison'.format(len(select_bl_id))+'_CLEAN_fhd_avg_visibilities_amplitudes_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.eps')    

## Plot individual snapshot, averaged and modeled delay spectra amplitudes

select_obsid = '1063642728'

nrow = 3
ncol = 1

fig, axs = PLT.subplots(nrow, sharex=True, sharey=True, figsize=(5,10))

imdspec0 = axs[0].pcolorfast(fhdbll, 1e6*fhdlags, NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:-1,NP.where(small_delays_ind)[0][:-1],0].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(3e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=fhd_vis_lag_noisy_max_pol0**2 * volfactor1 * volfactor2 * Jy2K**2))
imdspec1 = axs[1].pcolorfast(fhdbll, 1e6*fhdlags, NP.abs(avg_fhd_vis_lag_noisy_pol0[:-1,:-1].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(3e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=fhd_vis_lag_noisy_max_pol0**2 * volfactor1 * volfactor2 * Jy2K**2))
imdspec2 = axs[2].pcolorfast(ref_bl_length, 1e6*clean_lags, NP.abs(avg_asm_cc_skyvis_lag[:-1,:-1].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(3e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=fhd_vis_lag_noisy_max_pol0**2 * volfactor1 * volfactor2 * Jy2K**2))
for j in range(len(axs)):
    horizonb = axs[j].plot(fhdbll, 1e6*min_horizon_delays[sortind], color='white', ls=':', lw=1.5)
    horizont = axs[j].plot(fhdbll, 1e6*max_horizon_delays[sortind], color='white', ls=':', lw=1.5)
    axs[j].set_xlim(fhdbll.min(), fhdbll.max())
    axs[j].set_ylim(0.9*1e6*fhdlags.min(), 0.9*1e6*fhdlags.max())
    axs[j].set_aspect('auto')

    ax_kprll = axs[j].twinx()
    ax_kprll.set_yticks(kprll(axs[j].get_yticks()*1e-6, redshift))
    ax_kprll.set_ylim(kprll(NP.asarray(axs[j].get_ylim())*1e-6, redshift))
    yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
    ax_kprll.yaxis.set_major_formatter(yformatter)

    if j == 0:
        ax_kperp = axs[j].twiny()
        ax_kperp.set_xticks(kperp(axs[j].get_xticks()*freq/FCNST.c, redshift))
        ax_kperp.set_xlim(kperp(NP.asarray(axs[j].get_xlim())*freq/FCNST.c, redshift))
        xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
        ax_kperp.xaxis.set_major_formatter(xformatter)
        
fig.subplots_adjust(wspace=0, hspace=0)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_xticks([])
big_ax.set_yticks([])
big_ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=25)
big_ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=20)

big_axr = big_ax.twinx()
big_axr.set_axis_bgcolor('none')
big_axr.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_axr.set_xticks([])
big_axr.set_yticks([])
big_axr.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=35)

big_axt = big_ax.twiny()
big_axt.set_axis_bgcolor('none')
big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_axt.set_xticks([])
big_axt.set_yticks([])
big_axt.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=25)

cbax = fig.add_axes([0.125, 0.92, 0.72, 0.02])
cbar = fig.colorbar(imdspec1, cax=cbax, orientation='horizontal')
cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=-20, fontsize=12, rotation='horizontal')
cbax.xaxis.tick_top()
# cbax.xaxis.set_label_position('top')
cbax.xaxis.set_label_coords(1.075, 2.4)

# PLT.tight_layout()      
fig.subplots_adjust(right=0.84)
fig.subplots_adjust(top=0.85)
fig.subplots_adjust(bottom=0.09)

PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_fhd_sim_visibilities_amplitudes_comparison_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.png')
PLT.savefig('/data3/MWA/lstbin_RA0/NT/figures/multi_baseline_fhd_sim_visibilities_amplitudes_comparison_{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6,nchan*freq_resolution/1e6)+'.eps')
    
