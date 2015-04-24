import numpy as NP
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
import matplotlib.gridspec as GS
from matplotlib.ticker import FuncFormatter
import healpy as HP
from mwapy.pb import primary_beam as MWAPB
import geometry as GEOM
import interferometry as RI
import catalog as SM
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import lookup_operations as LKP
import ipdb as PDB

rootdir = '/data3/t_nithyanandan/'
# rootdir = '/data3/MWA/lstbin_RA0/NT/'

filenaming_convention = 'new'
# filenaming_convention = 'old'

project_MWA = False
project_LSTbin = False
project_HERA = True
project_beams = False
project_drift_scan = False
project_global_EoR = False

project_dir = ''
if project_MWA: project_dir = 'project_MWA/'
if project_LSTbin:
    if rootdir == '/data3/t_nithyanandan/':
        project_dir = 'project_LSTbin/'
if project_HERA: project_dir = 'project_HERA/'
if project_beams: project_dir = 'project_beams/'
if project_drift_scan: project_dir = 'project_drift_scan/'
if project_global_EoR: project_dir = 'project_global_EoR/'

telescope_id = 'custom'
element_size = 14.0
element_shape = 'dish'
element_orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
element_ocoords = 'altaz'
phased_array = False

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

ground_plane = None # height of antenna element above ground plane
if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

delayerr = 0.0     # delay error rms in ns
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
if project_MWA:
    delaygain_err_str = ''

ref_delayerr = 0.0     # delay error rms in ns
if ref_delayerr is None:
    ref_delayerr_str = ''
    ref_delayerr = 0.0
elif ref_delayerr < 0.0:
    raise ValueError('ref_delayerr must be non-negative.')
else:
    ref_delayerr_str = 'derr_{0:.3f}ns'.format(ref_delayerr)
ref_delayerr *= 1e-9

ref_gainerr = 0.0      # Gain error rms in dB
if ref_gainerr is None:
    ref_gainerr_str = ''
    ref_gainerr = 0.0
elif ref_gainerr < 0.0:
    raise ValueError('ref_gainerr must be non-negative.')
else:
    ref_gainerr_str = '_gerr_{0:.2f}dB'.format(ref_gainerr)

ref_nrand = 1       # Number of random realizations
if ref_nrand is None:
    ref_nrandom_str = ''
    ref_nrand = 1
elif nrand < 1:
    raise ValueError('ref_nrandom must be positive')
else:
    ref_nrandom_str = '_nrand_{0:0d}_'.format(ref_nrand)

if (ref_delayerr_str == '') and (ref_gainerr_str == ''):
    ref_nrand = 1
    ref_nrandom_str = ''

ref_delaygain_err_str = ref_delayerr_str + ref_gainerr_str + ref_nrandom_str
if project_MWA:
    ref_delaygain_err_str = ''

# latitude = -26.701
latitude = -30.7224
latitude_str = 'lat_{0:.3f}_'.format(latitude)

# array_layout = 'MWA-128T'
array_layout = 'HERA-331'

max_bl_length = None # Maximum baseline length (in m)

if array_layout == 'MWA-128T':
    ant_info = NP.loadtxt('/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
    ant_id = ant_info[:,0].astype(int).astype(str)
    ant_locs = ant_info[:,1:]
elif array_layout == 'HERA-7':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=7)
elif array_layout == 'HERA-19':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=19)
elif array_layout == 'HERA-37':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=37)
elif array_layout == 'HERA-61':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=61)
elif array_layout == 'HERA-91':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=91)
elif array_layout == 'HERA-127':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=127)
elif array_layout == 'HERA-169':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=169)
elif array_layout == 'HERA-217':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=217)
elif array_layout == 'HERA-271':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=271)
elif array_layout == 'HERA-331':
    ant_locs, ant_id = RI.hexagon_generator(14.6, n_total=331)

ref_bl, ref_bl_id = RI.baseline_generator(ant_locs, ant_id=ant_id, auto=False, conjugate=False)
ref_bl, select_ref_bl_ind, ref_bl_count = RI.uniq_baselines(ref_bl)
ref_bl_id = ref_bl_id[select_ref_bl_ind]
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

Tsys = 300.0 # System temperature in K
freq = 150.0e6 # center frequency in Hz
freq_resolution = 320e3 # in Hz
eor_freq_resolution = 80e3 # in Hz
n_channels = 128
nchan = n_channels
eor_nchan = 128
bw = n_channels * freq_resolution
eor_bw = eor_nchan * eor_freq_resolution
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution) # in Hz
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)
eor_bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(eor_nchan, eor_freq_resolution/1e3)
bpass_shape = 'bhw'
max_abs_delay = 1.5 # in micro seconds
# coarse_channel_resolution = 1.28e6 # in Hz
coarse_channel_resolution = None

anttemp_nside = 128
anttemp_nchan = 8
anttemp_freq_resolution = 1.28e6  # in Hz
anttemp_bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(anttemp_nchan, anttemp_freq_resolution/1e3)

wavelength = FCNST.c / freq  # in meters
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

n_bl_chunks = 16
baseline_chunk_size = 40
total_baselines = ref_bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]

truncated_ref_bl = NP.copy(ref_bl)
truncated_ref_bl_id = NP.copy(ref_bl_id)
truncated_ref_bl_length = NP.sqrt(NP.sum(truncated_ref_bl[:,:2]**2, axis=1))
# truncated_ref_bl_length = NP.copy(ref_bl_length)
truncated_ref_bl_orientation = NP.copy(ref_bl_orientation)
truncated_total_baselines = truncated_ref_bl_length.size
truncated_ref_bl_ind = ref_bl_length <= truncated_ref_bl_length.max()
if max_bl_length is not None:
    truncated_ref_bl_ind = ref_bl_length <= max_bl_length
    truncated_ref_bl = truncated_ref_bl[truncated_ref_bl_ind,:]
    truncated_ref_bl_id = truncated_ref_bl_id[truncated_ref_bl_ind]
    truncated_ref_bl_orientation = truncated_ref_bl_orientation[truncated_ref_bl_ind]
    truncated_ref_bl_length = truncated_ref_bl_length[truncated_ref_bl_ind]
    truncated_total_baselines = truncated_ref_bl_length.size

bl_orientation_str = ['South-East', 'East', 'North-East', 'North']

spindex_rms = 0.0
spindex_seed = None
spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

nside = 256
use_GSM = True
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False

use_pfb = False

pfb_instr = ''
pfb_outstr = ''
if not use_pfb: 
    pfb_instr = '_no_pfb'
    pfb_outstr = 'no_pfb_'

# obs_mode = 'custom'
# obs_mode = 'lstbin'
obs_mode = 'drift'
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

anttemp_snapshot_type_str = obs_mode

duration_str = ''
if obs_mode in ['track', 'drift']:
    t_snap = 1080.0    # in seconds
    n_snaps = 80
    if (t_snap is not None) and (n_snaps is not None):
        duration_str = '_{0:0d}x{1:.1f}s'.format(n_snaps, t_snap)

anttemp_duration_str = ''
if obs_mode in ['track', 'drift']:
    anttemp_t_snap = 1080.0    # in seconds
    anttemp_n_snaps = 80
    if (anttemp_t_snap is not None) and (anttemp_n_snaps is not None):
        anttemp_duration_str = '_{0:0d}x{1:.1f}s'.format(anttemp_n_snaps, anttemp_t_snap)
        
# pc = NP.asarray([90.0, 90.0]).reshape(1,-1)
# pc = NP.asarray([266.416837, -29.00781]).reshape(1,-1)
pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'
if pc_coords == 'dircos':
    pc_dircos = pc

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

if filenaming_convention == 'old':
    roifile = '/data3/t_nithyanandan/'+project_dir+'/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
else:
    roifile = '/data3/t_nithyanandan/'+project_dir+'/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+'.fits'

roi = RI.ROI_parameters(init_file=roifile)
telescope = roi.telescope

# telescope = {}
# if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
#     telescope['id'] = telescope_id
# telescope['shape'] = element_shape
# telescope['size'] = ele<ment_size
# telescope['orientation'] = element_orientation
# telescope['ocoords'] = element_ocoords
# telescope['groundplane'] = ground_plane
# element_locs = None

element_locs = None
if phased_array:
    phased_elements_file = '/data3/t_nithyanandan/project_MWA/MWA_tile_dipole_locations.txt'
    try:
        element_locs = NP.loadtxt(phased_elements_file, skiprows=1, comments='#', usecols=(0,1,2))
    except IOError:
        raise IOError('Could not open the specified file for phased array of antenna elements.')

if telescope_id == 'mwa':
    xlocs, ylocs = NP.meshgrid(1.1*NP.linspace(-1.5,1.5,4), 1.1*NP.linspace(1.5,-1.5,4))
    element_locs = NP.hstack((xlocs.reshape(-1,1), ylocs.reshape(-1,1), NP.zeros(xlocs.size).reshape(-1,1)))

if element_locs is not None:
    telescope['element_locs'] = element_locs

if (telescope['shape'] == 'dipole') or (telescope['shape'] == 'delta'):
    A_eff = (0.5*wavelength)**2
    if (telescope_id == 'mwa') or phased_array:
        A_eff *= 16
if telescope['shape'] == 'dish':
    A_eff = NP.pi * (0.5*element_size)**2

# fhd_obsid = [1061309344, 1061316544]

# pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
# pointing_info_from_file = NP.loadtxt(pointing_file, comments='#', usecols=(1,2,3), delimiter=',')
# obs_id = NP.loadtxt(pointing_file, comments='#', usecols=(0,), delimiter=',', dtype=str)
# if (telescope_id == 'mwa') or (phased_array):
#     delays_str = NP.loadtxt(pointing_file, comments='#', usecols=(4,), delimiter=',', dtype=str)
#     delays_list = [NP.fromstring(delaystr, dtype=float, sep=';', count=-1) for delaystr in delays_str]
#     delay_settings = NP.asarray(delays_list)
#     delay_settings *= 435e-12
#     delays = NP.copy(delay_settings)

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

dspec_min = None
dspec_max = None

def kprll(eta, z):
    return 2 * NP.pi * eta * cosmodel100.H0.value * CNST.rest_freq_HI * cosmodel100.efunc(z) / FCNST.c / (1+z)**2 * 1e3

def kperp(u, z):
    return 2 * NP.pi * u / cosmodel100.comoving_transverse_distance(z).value

def ha(ra, lst):
    ha = lst - ra
    ha[ha > 180.0] = ha[ha > 180.0] - 360.0
    return ha

antenna_temperature_file = rootdir+project_dir+'antenna_power_'+telescope_str+ground_plane_str+latitude_str+anttemp_snapshot_type_str+anttemp_duration_str+'_'+fg_str+'_sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(anttemp_nside)+'{0}_{1:.1f}_MHz'.format(anttemp_bandpass_str, freq/1e6)+'.fits'

hdulist = fits.open(antenna_temperature_file)
antpower_K = hdulist['Antenna Temperature'].data
anttemp_pointing_coords = hdulist['PRIMARY'].header['pointing_coords']
anttemp_pointing_center = hdulist['POINTINGS'].data['pointing_center']
anttemp_lst = hdulist['POINTINGS'].data['LST']
if (anttemp_pointing_coords == 'RADEC') or (anttemp_pointing_coords == 'radec'):
    anttemp_pointings_radec = NP.copy(anttemp_pointing_center)
    anttemp_pointings_hadec = NP.hstack(((anttemp_lst - anttemp_pointings_radec[:,0]).reshape(-1,1), anttemp_pointings_radec[:,1].reshape(-1,1)))
    anttemp_pointings_altaz = GEOM.hadec2altaz(anttemp_pointings_hadec, latitude, units='degrees')
elif (anttemp_pointing_coords == 'HADEC') or (anttemp_pointing_coords == 'hadec'):
    anttemp_pointings_hadec = NP.copy(anttemp_pointing_center)
    anttemp_pointings_radec = NP.hstack(((anttemp_lst - anttemp_pointings_hadec[:,0]).reshape(-1,1), anttemp_pointings_hadec[:,1].reshape(-1,1)))    
    anttemp_pointings_altaz = GEOM.hadec2altaz(anttemp_pointings_hadec, latitude, units='degrees')
elif (anttemp_pointing_coords == 'ALTAZ') or (anttemp_pointing_coords == 'altaz'):
    anttemp_pointings_altaz = NP.copy(anttemp_pointing_center)
    anttemp_pointings_hadec = GEOM.altaz2hadec(anttemp_pointings_altaz, latitude, units='degrees')
    anttemp_pointings_radec = NP.hstack(((anttemp_lst - anttemp_pointings_hadec[:,0]).reshape(-1,1), anttemp_pointings_hadec[:,1].reshape(-1,1)))    

##################

if filenaming_convention == 'old':
    asm_infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_instr
    asm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_outstr+bpass_shape        
else:
    asm_infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr
    asm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_outstr+bpass_shape
    dsm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'dsm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_outstr+bpass_shape
    csm_CLEAN_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'csm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_outstr+bpass_shape
    eor_infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'HI_cube'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, eor_bandpass_str, freq/1e6)+pfb_instr

ia = RI.InterferometerArray(None, None, None, init_file=asm_infile+'.fits') 
simdata_bl_orientation = NP.angle(ia.baselines[:,0] + 1j * ia.baselines[:,1], deg=True)
simdata_neg_bl_orientation_ind = simdata_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
simdata_bl_orientation[simdata_neg_bl_orientation_ind] -= 180.0
ia.baselines[simdata_neg_bl_orientation_ind,:] = -ia.baselines[simdata_neg_bl_orientation_ind,:]

# PDB.set_trace()
# mwdt = ia.multi_window_delay_transform([4e6, 8e6], freq_center=[145e6, 160e6], shape='bhw')

hdulist = fits.open(asm_infile+'.fits')
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

eor_ia = RI.InterferometerArray(None, None, None, init_file=eor_infile+'.fits') 
if NP.sum(simdata_neg_bl_orientation_ind) > 0:
    eor_ia.conjugate(ind=NP.where(simdata_neg_bl_orientation_ind)[0])
eor_ia.delay_transform(oversampling_factor-1.0)

asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
asm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = asm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
csm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()

asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis, axis=1),axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis_res, axis=1),axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
asm_cc_skyvis_lag = asm_cc_skyvis_lag + asm_ccres_sky

asm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis, axis=1),axes=1) * asm_cc_vis.shape[1] * freq_resolution
asm_ccres = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis_res, axis=1),axes=1) * asm_cc_vis.shape[1] * freq_resolution
asm_cc_vis_lag = asm_cc_vis_lag + asm_ccres

dsm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
dsm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis_res, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
dsm_cc_skyvis_lag = dsm_cc_skyvis_lag + dsm_ccres_sky

dsm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
dsm_ccres = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis_res, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
dsm_cc_vis_lag = dsm_cc_vis_lag + dsm_ccres

csm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
csm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis_res, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
csm_cc_skyvis_lag = csm_cc_skyvis_lag + csm_ccres_sky

csm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
csm_ccres = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis_res, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
csm_cc_vis_lag = csm_cc_vis_lag + csm_ccres

eor_skyvis_lag = eor_ia.skyvis_lag

asm_cc_skyvis_lag = DSP.downsampler(asm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
asm_cc_vis_lag = DSP.downsampler(asm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
# eor_skyvis_lag = DSP.downsampler(eor_skyvis_lag, 1.0*clean_lags.size/eor_ia.lags.size, axis=1)

clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
clean_lags = clean_lags.ravel()

vis_noise_lag = NP.copy(ia.vis_noise_lag)
vis_noise_lag = vis_noise_lag[truncated_ref_bl_ind,:,:]
asm_cc_skyvis_lag = asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
asm_cc_vis_lag = asm_cc_vis_lag[truncated_ref_bl_ind,:,:]
dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
dsm_cc_vis_lag = dsm_cc_vis_lag[truncated_ref_bl_ind,:,:]
csm_cc_skyvis_lag = csm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
csm_cc_vis_lag = csm_cc_vis_lag[truncated_ref_bl_ind,:,:]
eor_skyvis_lag = eor_skyvis_lag[truncated_ref_bl_ind,:,:]

delaymat = DLY.delay_envelope(ia.baselines[truncated_ref_bl_ind,:], pc, units='mks')
min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
max_delay = delaymat[0,:,0]-delaymat[0,:,1]
clags = clean_lags.reshape(1,-1)
min_delay = min_delay.reshape(-1,1)
max_delay = max_delay.reshape(-1,1)
thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
thermal_noise_window = NP.repeat(thermal_noise_window, ia.baselines[truncated_ref_bl_ind,:].shape[0], axis=0)
EoR_window = NP.logical_or(clags > max_delay+1/bw, clags < min_delay-1/bw)
strict_EoR_window = NP.copy(EoR_window)
if coarse_channel_resolution is not None:
    strict_EoR_window = NP.logical_and(EoR_window, NP.abs(clags) < 1/coarse_channel_resolution)
wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
non_wedge_window = NP.logical_not(wedge_window)
# vis_rms_lag = OPS.rms(asm_cc_vis_lag.reshape(-1,n_snaps), mask=NP.logical_not(NP.repeat(thermal_noise_window.reshape(-1,1), n_snaps, axis=1)), axis=0)
# vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
# T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,-1)) * CNST.Jy
# vis_rms_lag_theory = OPS.rms(vis_noise_lag.reshape(-1,n_snaps), mask=NP.logical_not(NP.repeat(EoR_window.reshape(-1,1), n_snaps, axis=1)), axis=0)
# vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
# T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,-1)) * CNST.Jy
vis_rms_lag = OPS.rms(asm_cc_vis_lag, mask=NP.logical_not(NP.repeat(thermal_noise_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
vis_rms_lag_theory = OPS.rms(vis_noise_lag, mask=NP.logical_not(NP.repeat(EoR_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy

if max_abs_delay is not None:
    small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
    eor_small_delays_ind = NP.abs(eor_ia.lags) <= max_abs_delay * 1e-6    
    clean_lags = clean_lags[small_delays_ind]
    asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
    asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]
    dsm_cc_vis_lag = dsm_cc_vis_lag[:,small_delays_ind,:]
    dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
    csm_cc_vis_lag = csm_cc_vis_lag[:,small_delays_ind,:]
    csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]
    eor_skyvis_lag = eor_skyvis_lag[:,eor_small_delays_ind,:]

if (dspec_min is None) or (dspec_max is None):
    # dspec_min = NP.abs(asm_cc_vis_lag).min()
    dspec_min = NP.abs(eor_skyvis_lag).min()
    dspec_max = NP.abs(asm_cc_vis_lag).max()
    dspec_min = dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2
    dspec_max = dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2

cardinal_blo = 180.0 / n_bins_baseline_orientation * (NP.arange(n_bins_baseline_orientation)-1).reshape(-1,1)
cardinal_bll = 100.0
cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

small_delays_EoR_window = EoR_window.T
small_delays_strict_EoR_window = strict_EoR_window.T
small_delays_wedge_window = wedge_window.T
if max_abs_delay is not None:
    small_delays_EoR_window = small_delays_EoR_window[small_delays_ind,:]
    small_delays_strict_EoR_window = small_delays_strict_EoR_window[small_delays_ind,:]
    small_delays_wedge_window = small_delays_wedge_window[small_delays_ind,:]

small_delays_non_wedge_window = NP.logical_not(small_delays_wedge_window)
    
backdrop_xsize = 500
xmin = -180.0
xmax = 180.0
ymin = -90.0
ymax = 90.0

xgrid, ygrid = NP.meshgrid(NP.linspace(xmax, xmin, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
xvect = xgrid.ravel()
yvect = ygrid.ravel()

pb_snapshots = []
pbx_MWA_snapshots = []
pby_MWA_snapshots = []

src_ind_csm_snapshots = []
dsm_snapshots = []

m1, m2, d12 = GEOM.spherematch(pointings_radec[:,0], pointings_radec[:,1], anttemp_pointings_radec[:,0], anttemp_pointings_radec[:,1], nnearest=1)

# Construct the bright point source catalog

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

if spindex_seed is None:
    spindex_SUMSS = -0.83 + spindex_rms * NP.random.randn(fint.size)
else:
    NP.random.seed(spindex_seed)
    spindex_SUMSS = -0.83 + spindex_rms * NP.random.randn(fint.size)

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
fluxes = NP.copy(fint)

nvss_file = '/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits'
freq_NVSS = 1.4 # in GHz
hdulist = fits.open(nvss_file)
ra_deg_NVSS = hdulist[1].data['RA(2000)']
dec_deg_NVSS = hdulist[1].data['DEC(2000)']
nvss_fpeak = hdulist[1].data['PEAK INT']
nvss_majax = hdulist[1].data['MAJOR AX']
nvss_minax = hdulist[1].data['MINOR AX']
hdulist.close()

if spindex_seed is None:
    spindex_NVSS = -0.83 + spindex_rms * NP.random.randn(nvss_fpeak.size)
else:
    NP.random.seed(2*spindex_seed)
    spindex_NVSS = -0.83 + spindex_rms * NP.random.randn(nvss_fpeak.size)

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
ra_deg_wrapped = ra_deg.ravel()
ra_deg_wrapped[ra_deg_wrapped > 180.0] -= 360.0

spec_parms = {}
# spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
spec_parms['power-law-index'] = spindex
# spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
spec_parms['flux-scale'] = fluxes
spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
spec_parms['freq-width'] = NP.zeros(ra_deg.size)

csmskymod = SM.SkyModel(catlabel, freq, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
csm_fluxes = csmskymod.spec_parms['flux-scale'] * (freq/csmskymod.spec_parms['freq-ref'])**csmskymod.spec_parms['power-law-index']

# Construct the Diffuse sky model

dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq/1e6,nside)
hdulist = fits.open(dsm_file)
dsm_table = hdulist[1].data
dsm_ra_deg = dsm_table['RA']
dsm_dec_deg = dsm_table['DEC']
dsm_temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
dsm = HP.cartview(dsm_temperatures.ravel(), coord=['G','E'], rot=[0,0,0], xsize=backdrop_xsize, return_projected_map=True)
dsm = dsm.ravel()
PLT.close()

for i in xrange(n_snaps):
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
    
    pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_info=roi.pinfo[i])
    if (telescope_id == 'mwa') or (phased_array):
        pbx_MWA, pby_MWA = MWAPB.MWA_Tile_advanced(NP.radians(90.0-altaz[roi_altaz,0]).reshape(-1,1), NP.radians(altaz[roi_altaz,1]).reshape(-1,1), freq=185e6, delays=roi.pinfo[i]['delays']/435e-12)
        pbx_MWA_vect[roi_altaz] = pbx_MWA.ravel()
        pby_MWA_vect[roi_altaz] = pby_MWA.ravel()
    
    pb_snapshots += [pb]
    pbx_MWA_snapshots += [pbx_MWA_vect]
    pby_MWA_snapshots += [pby_MWA_vect]

    csm_hadec = NP.hstack(((lst[i]-csmskymod.location[:,0]).reshape(-1,1), csmskymod.location[:,1].reshape(-1,1)))
    csm_altaz = GEOM.hadec2altaz(csm_hadec, latitude, units='degrees')
    roi_csm_altaz = NP.asarray(NP.where(csm_altaz[:,0] >= 0.0)).ravel()
    src_ind_csm_snapshots += [roi_csm_altaz]

    dsm_snapshot = NP.empty(xvect.size)
    dsm_snapshot.fill(NP.nan)
    dsm_snapshot[roi_altaz] = dsm[roi_altaz]
    dsm_snapshots += [dsm_snapshot]

n_fg_ticks = 5
dsm_fg_ticks = NP.round(NP.logspace(NP.log10(dsm.min()), NP.log10(dsm.max()), n_fg_ticks)).astype(NP.int)
csm_fg_ticks = NP.round(NP.logspace(NP.log10(csm_fluxes.min()), NP.log10(csm_fluxes.max()), n_fg_ticks)).astype(NP.int)

# for j in range(n_snaps):
#     fig, axs = PLT.subplots(3, figsize=(6,8))

#     dsmsky = axs[0].imshow(dsm_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=dsm.min(), vmax=dsm.max()), cmap=CM.jet)
#     pbskyc = axs[0].contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='k', linewidths=1.5)
#     # pbskyc = axs[0].contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[0.001953125, 0.0078125, 0.03125, 0.125, 0.5], colors='k', linewidths=1.5)
    
#     axs[0].set_xlim(xvect.max(), xvect.min())
#     axs[0].set_ylim(yvect.min(), yvect.max())
#     axs[0].grid(True, which='both')
#     axs[0].set_aspect('auto')
#     axs[0].tick_params(which='major', length=12, labelsize=12)
#     axs[0].tick_params(which='minor', length=6)
#     axs[0].locator_params(axis='x', nbins=5)
#     axs[0].set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#     axs[0].set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
    
#     axs_ha = axs[0].twiny()
#     axs_ha.set_xticks(lst[j]-axs[0].get_xticks())
#     axs_ha.set_xlim(lst[j]-NP.asarray(axs[0].get_xlim()))
#     xformatter = FuncFormatter(lambda x, pos: '{0:.1f}'.format(x))
#     axs_ha.xaxis.set_major_formatter(xformatter)
#     axs_ha.set_xlabel('HA [degrees]', fontsize=16, weight='medium')

#     cbax0 = fig.add_axes([0.85, 0.75, 0.02, 0.16])
#     cbar0 = fig.colorbar(dsmsky, cax=cbax0, orientation='vertical')
#     cbar0.set_ticks(fg_ticks.tolist())
#     cbar0.set_ticklabels(fg_ticks.tolist())
#     cbax0.set_ylabel('Temperature [K]', labelpad=-60, fontsize=14)
#     # cbax0.xaxis.set_label_position('top')
                
#     axs[1].plot(anttemp_lst, antpower_K[:,anttemp_nchan/2], '-', lw=2, color='gray')
#     axs[1].plot(anttemp_lst[m2[j]], antpower_K[m2[j],anttemp_nchan/2], 'ko', ms=8, mew=2, mfc='none')
#     axs[1].set_xlim(0, 360)
#     axs[1].set_xlabel('RA [degrees]', fontsize=18, weight='medium')
#     axs[1].set_ylabel(r'$T_\mathrm{ant}$'+' [ K ]', fontsize=16, weight='medium')

#     imdspec = axs[2].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_vis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
#     horizonb = axs[2].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
#     horizont = axs[2].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
#     axs[2].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
#     axs[2].set_aspect('auto')
#     axs[2].set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
#     axs[2].set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')

#     axs_kprll = axs[2].twinx()
#     axs_kprll.set_yticks(kprll(axs[2].get_yticks()*1e-6, redshift))
#     axs_kprll.set_ylim(kprll(NP.asarray(axs[2].get_ylim())*1e-6, redshift))
#     yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
#     axs_kprll.yaxis.set_major_formatter(yformatter)

#     axs_kperp = axs[2].twiny()
#     axs_kperp.set_xticks(kperp(axs[2].get_xticks()*freq/FCNST.c, redshift))
#     axs_kperp.set_xlim(kperp(NP.asarray(axs[2].get_xlim())*freq/FCNST.c, redshift))
#     xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
#     axs_kperp.xaxis.set_major_formatter(xformatter)

#     axs_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
#     axs_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')

#     cbax2 = fig.add_axes([0.9, 0.1, 0.02, 0.16])
#     cbar2 = fig.colorbar(imdspec, cax=cbax2, orientation='vertical')
#     cbax2.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
#     cbax2.xaxis.set_label_position('top')
    
#     PLT.tight_layout()
#     fig.subplots_adjust(right=0.72)
#     fig.subplots_adjust(top=0.92)

#     PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.png', bbox_inches=0)
#     PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.eps', bbox_inches=0)
#     PLT.close()

anttemp_lst_orig = NP.copy(anttemp_lst)
anttemp_lst[anttemp_lst > 180.0] -= 360.0

# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots '.format(n_snaps), PGB.ETA()], maxval=n_snaps).start()
# for j in range(n_snaps):
#     fig = PLT.figure(figsize=(6,9))
#     gs1 = GS.GridSpec(2,1)
#     gs1.update(left=0.15, right=0.8, top=0.93, bottom=0.5, hspace=0)
#     ax0 = PLT.subplot(gs1[0,0])
#     ax1 = PLT.subplot(gs1[1,0], sharex=ax0)
#     gs2 = GS.GridSpec(1,1)
#     gs2.update(left=0.15, right=0.72, top=0.36, bottom=0.07)
#     ax2 = PLT.subplot(gs2[0,0])

#     dsmsky = ax0.imshow(dsm_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=dsm.min(), vmax=dsm.max()), cmap=CM.jet)
#     pbskyc = ax0.contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='k', linewidths=1.5)
#     # pbskyc = ax0.contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[0.001953125, 0.0078125, 0.03125, 0.125, 0.5], colors='k', linewidths=1.5)
    
#     ax0.set_xlim(xvect.max(), xvect.min())
#     ax0.set_ylim(yvect.min(), yvect.max())
#     ax0.grid(True, which='both')
#     ax0.set_aspect('auto')
#     ax0.tick_params(which='major', length=12, labelsize=12)
#     ax0.tick_params(which='minor', length=6)
#     ax0.locator_params(axis='x', nbins=5)
#     ax0.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
#     # ax0.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
#     PLT.setp(ax0.get_xticklabels(), visible=False)

#     ax_ha = ax0.twiny()
#     ax_ha.set_xticks(lst[j]-ax0.get_xticks())
#     ax_ha.set_xlim(lst[j]-NP.asarray(ax0.get_xlim()))
#     # ax_ha.set_xticks(ha(ax0.get_xticks(), lst[j]))
#     # ax_ha.set_xlim(ha(NP.asarray(ax0.get_xlim()), lst[j]))
#     # ha_ticks = lst[j] - ax0.get_xticks()
#     # ha_ticks[ha_ticks > 180.0] = ha_ticks[ha_ticks > 180.0] - 360.0
#     # ha_ticks[ha_ticks > 180.0] = ha_ticks[ha_ticks > 180.0] - 360.0    
#     # ha_ticks[ha_ticks <= -180.0] = ha_ticks[ha_ticks <= -180.0] + 360.0
#     # ha_ticks[ha_ticks <= -180.0] = ha_ticks[ha_ticks <= -180.0] + 360.0
#     # ax_ha.set_xticks(ha_ticks)
#     # if lst[j] <= 180.0:
#     #     ha_xlim = lst[j] - NP.asarray(ax0.get_xlim())
#     # else:
#     #     ha_xlim = lst[j]-360 - NP.asarray(ax0.get_xlim())
#     # ax_ha.set_xlim(ha_xlim)
#     xformatter = FuncFormatter(lambda x, pos: '{0:.1f}'.format(x))
#     ax_ha.xaxis.set_major_formatter(xformatter)
#     ax_ha.set_xlabel('HA [degrees]', fontsize=16, weight='medium')

#     cbax0 = fig.add_axes([0.88, 0.72, 0.02, 0.2])
#     cbar0 = fig.colorbar(dsmsky, cax=cbax0, orientation='vertical')
#     cbar0.set_ticks(fg_ticks.tolist())
#     cbar0.set_ticklabels(fg_ticks.tolist())
#     cbax0.set_ylabel('Temperature [K]', labelpad=-60, fontsize=14)
#     # cbax0.xaxis.set_label_position('top')
                
#     ax1.plot(anttemp_lst, antpower_K[:,anttemp_nchan/2], '.', lw=2, color='gray')
#     ax1.plot(anttemp_lst[m2[j]], antpower_K[m2[j],anttemp_nchan/2], 'ko', ms=8, mew=2, mfc='none')
#     ax1.text(0.3, 0.9, 'LST = {0:.1f} deg'.format(anttemp_lst[m2[j]]), transform=ax1.transAxes, fontsize=14, ha='center', color='black')
#     ax1.text(0.3, 0.8, r'$T_\mathrm{ant} = $'+'{0:0d} K'.format(int(NP.round(antpower_K[m2[j],anttemp_nchan/2]))), transform=ax1.transAxes, fontsize=14, ha='center', color='black')
#     ax1.set_xlim(xvect.max(), xvect.min())
#     ax1.set_ylim(0, 1.05*antpower_K[:,anttemp_nchan/2].max())
#     ax1.set_xlabel('RA [degrees]', fontsize=18, weight='medium')
#     ax1.set_ylabel(r'$T_\mathrm{ant}$'+' [ K ]', fontsize=16, weight='medium')

#     imdspec = ax2.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_vis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
#     horizonb = ax2.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
#     horizont = ax2.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
#     ax2.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
#     ax2.set_aspect('auto')
#     ax2.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
#     ax2.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')

#     axs_kprll = ax2.twinx()
#     axs_kprll.set_yticks(kprll(ax2.get_yticks()*1e-6, redshift))
#     axs_kprll.set_ylim(kprll(NP.asarray(ax2.get_ylim())*1e-6, redshift))
#     yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
#     axs_kprll.yaxis.set_major_formatter(yformatter)

#     axs_kperp = ax2.twiny()
#     axs_kperp.set_xticks(kperp(ax2.get_xticks()*freq/FCNST.c, redshift))
#     axs_kperp.set_xlim(kperp(NP.asarray(ax2.get_xlim())*freq/FCNST.c, redshift))
#     xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
#     axs_kperp.xaxis.set_major_formatter(xformatter)

#     axs_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
#     axs_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')

#     cbax2 = fig.add_axes([0.9, 0.07, 0.02, 0.29])
#     cbar2 = fig.colorbar(imdspec, cax=cbax2, orientation='vertical')
#     cbax2.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
#     cbax2.xaxis.set_label_position('top')
    
#     # PLT.tight_layout()
#     # fig.subplots_adjust(right=0.72)
#     # fig.subplots_adjust(top=0.92)

#     PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.png', bbox_inches=0)
#     PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.eps', bbox_inches=0)
#     PLT.close()
    
#     progress.update(j+1)
# progress.finish()

# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots '.format(n_snaps), PGB.ETA()], maxval=n_snaps).start()
# for j in range(n_snaps):
#     fig = PLT.figure(figsize=(11,8))
#     gs1 = GS.GridSpec(3,1)
#     gs2 = GS.GridSpec(3,1)
#     gs1.update(left=0.08, right=0.4, top=0.93, bottom=0.07, hspace=0)
#     gs2.update(left=0.55, right=0.84, top=0.93, bottom=0.07, hspace=0)
#     ax10 = PLT.subplot(gs1[0,0])
#     ax11 = PLT.subplot(gs1[1,0], sharex=ax10, sharey=ax10)
#     ax12 = PLT.subplot(gs1[2,0], sharex=ax10)
#     # big_ax1 = PLT.subplot(gs1[:2,0])
#     # big_ax1 = fig.add_axes([0.55, 0.07, 0.27, 0.86])
#     ax20 = PLT.subplot(gs2[0,0])
#     ax21 = PLT.subplot(gs2[1,0], sharex=ax20, sharey=ax20)
#     ax22 = PLT.subplot(gs2[2,0], sharex=ax20, sharey=ax20)    

#     dsmsky = ax10.imshow(dsm_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=dsm.min(), vmax=dsm.max()), cmap=CM.jet)
#     pbskyc0 = ax10.contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='k', linewidths=1.5)
#     ax10.text(0.5, 0.9, 'Diffuse', transform=ax10.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

#     csmsky = ax11.scatter(ra_deg_wrapped[src_ind_csm_snapshots[j]], dec_deg[src_ind_csm_snapshots[j]], c=csm_fluxes[src_ind_csm_snapshots[j]], norm=PLTC.LogNorm(vmin=csm_fluxes.min(), vmax=csm_fluxes.max()), cmap=CM.jet, edgecolor='none', s=20)
#     pbskyc1 = ax11.contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='k', linewidths=1.5)
#     ax11.text(0.5, 0.9, 'Point Sources', transform=ax11.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    
#     ax12.plot(anttemp_lst, antpower_K[:,anttemp_nchan/2], '.', lw=2, color='gray')
#     ax12.plot(anttemp_lst[m2[j]], antpower_K[m2[j],anttemp_nchan/2], 'ko', ms=8, mew=2, mfc='none')
#     ax12.text(0.3, 0.9, 'LST = {0:.1f} deg'.format(anttemp_lst[m2[j]]), transform=ax12.transAxes, fontsize=14, ha='center', color='black')
#     ax12.text(0.3, 0.8, r'$T_\mathrm{ant} = $'+'{0:0d} K'.format(int(NP.round(antpower_K[m2[j],anttemp_nchan/2]))), transform=ax12.transAxes, fontsize=14, ha='center', color='black')
#     ax12.set_xlim(xvect.max(), xvect.min())
#     ax12.set_ylim(0, 1.05*antpower_K[:,anttemp_nchan/2].max())
#     ax12.set_xlabel('RA [degrees]', fontsize=18, weight='medium')
#     ax12.set_ylabel(r'$T_\mathrm{ant}$'+' [ K ]', fontsize=16, weight='medium')
#     # ax12.text(0.5, 0.9, 'Diffuse + Point Sources', transform=ax12.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

#     ax10.set_xlim(xvect.max(), xvect.min())
#     ax10.set_ylim(yvect.min(), yvect.max())
#     ax10.grid(True, which='both')
#     ax10.set_aspect('auto')
#     ax10.tick_params(which='major', length=12, labelsize=12)
#     ax10.tick_params(which='minor', length=6)
#     ax10.locator_params(axis='x', nbins=5)
#     PLT.setp(ax10.get_xticklabels(), visible=False)

#     ax11.set_xlim(xvect.max(), xvect.min())
#     ax11.set_ylim(yvect.min(), yvect.max())
#     ax11.grid(True, which='both')
#     ax11.set_aspect('auto')
#     ax11.tick_params(which='major', length=12, labelsize=12)
#     ax11.tick_params(which='minor', length=6)
#     ax11.locator_params(axis='x', nbins=5)
#     PLT.setp(ax11.get_xticklabels(), visible=False)

#     ax_ha = ax10.twiny()
#     ax_ha.set_xticks(lst[j]-ax10.get_xticks())
#     ax_ha.set_xlim(lst[j]-NP.asarray(ax10.get_xlim()))
#     # ax_ha.set_xticks(ha(ax10.get_xticks(), lst[j]))
#     # ax_ha.set_xlim(ha(NP.asarray(ax10.get_xlim()), lst[j]))
#     # ha_ticks = lst[j] - ax10.get_xticks()
#     # ha_ticks[ha_ticks > 180.0] = ha_ticks[ha_ticks > 180.0] - 360.0
#     # ha_ticks[ha_ticks > 180.0] = ha_ticks[ha_ticks > 180.0] - 360.0    
#     # ha_ticks[ha_ticks <= -180.0] = ha_ticks[ha_ticks <= -180.0] + 360.0
#     # ha_ticks[ha_ticks <= -180.0] = ha_ticks[ha_ticks <= -180.0] + 360.0
#     # ax_ha.set_xticks(ha_ticks)
#     # if lst[j] <= 180.0:
#     #     ha_xlim = lst[j] - NP.asarray(ax10.get_xlim())
#     # else:
#     #     ha_xlim = lst[j]-360 - NP.asarray(ax10.get_xlim())
#     # ax_ha.set_xlim(ha_xlim)
#     xformatter = FuncFormatter(lambda x, pos: '{0:.1f}'.format(x))
#     ax_ha.xaxis.set_major_formatter(xformatter)
#     ax_ha.set_xlabel('HA [degrees]', fontsize=16, weight='medium')

#     pos10 = ax10.get_position()
#     pos11 = ax11.get_position()
#     x0 = pos10.x0
#     x1 = pos10.x1
#     y0 = pos11.y0
#     y1 = pos10.y1
#     big_ax1 = fig.add_axes([x0, y0, x1-x0, y1-y0])
#     big_ax1.set_axis_bgcolor('none')
#     big_ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_ax1.set_xticks([])
#     big_ax1.set_yticks([])
#     big_ax1.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)

#     cbax10 = fig.add_axes([0.43, 1.02*pos10.y0, 0.02, 0.98*pos10.y1-1.02*pos10.y0])
#     cbar10 = fig.colorbar(dsmsky, cax=cbax10, orientation='vertical')
#     cbar10.set_ticks(dsm_fg_ticks.tolist())
#     cbar10.set_ticklabels(dsm_fg_ticks.tolist())
#     cbax10.set_ylabel('Temperature [K]', labelpad=-65, fontsize=14)

#     cbax11 = fig.add_axes([0.43, 1.02*pos11.y0, 0.02, 0.98*pos11.y1-1.02*pos11.y0])
#     cbar11 = fig.colorbar(csmsky, cax=cbax11, orientation='vertical')
#     cbar11.set_ticks(csm_fg_ticks.tolist())
#     cbar11.set_ticklabels(csm_fg_ticks.tolist())
#     cbax11.set_ylabel('Flux Density [Jy]', labelpad=-60, fontsize=14)

#     dsmdspec = ax20.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
#     horizonb = ax20.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
#     horizont = ax20.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
#     ax20.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
#     ax20.set_aspect('auto')
#     ax20.text(0.5, 0.9, 'Diffuse', transform=ax20.transAxes, fontsize=14, weight='semibold', ha='center', color='white')

#     csmspec = ax21.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
#     horizonb = ax21.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
#     horizont = ax21.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
#     ax21.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
#     ax21.set_aspect('auto')
#     ax21.text(0.5, 0.9, 'Point Sources', transform=ax21.transAxes, fontsize=14, weight='semibold', ha='center', color='white')

#     asmdspec = ax22.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_vis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
#     horizonb = ax22.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
#     horizont = ax22.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
#     ax22.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
#     ax22.set_aspect('auto')
#     ax22.text(0.5, 0.9, 'Diffuse + Point Sources', transform=ax22.transAxes, fontsize=14, weight='semibold', ha='center', color='white')

#     axs = [ax20, ax21, ax22]
#     for k in xrange(len(axs)):
#         axs_kprll = axs[k].twinx()
#         axs_kprll.set_yticks(kprll(axs[k].get_yticks()*1e-6, redshift))
#         axs_kprll.set_ylim(kprll(NP.asarray(axs[k].get_ylim())*1e-6, redshift))
#         yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
#         axs_kprll.yaxis.set_major_formatter(yformatter)
#         if k == 0:
#             axs_kperp = axs[k].twiny()
#             axs_kperp.set_xticks(kperp(axs[k].get_xticks()*freq/FCNST.c, redshift))
#             axs_kperp.set_xlim(kperp(NP.asarray(axs[k].get_xlim())*freq/FCNST.c, redshift))
#             xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
#             axs_kperp.xaxis.set_major_formatter(xformatter)
                
#     # fig.subplots_adjust(wspace=0, hspace=0)
#     # big_ax2 = fig.add_subplot(111)
#     big_ax2 = fig.add_axes([gs2.left, gs2.bottom, gs2.right-gs2.left, gs2.top-gs2.bottom])
#     big_ax2.set_axis_bgcolor('none')
#     big_ax2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_ax2.set_xticks([])
#     big_ax2.set_yticks([])
#     big_ax2.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=30)
#     big_ax2.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=15)

#     big_axr2 = big_ax2.twinx()
#     big_axr2.set_axis_bgcolor('none')
#     big_axr2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_axr2.set_xticks([])
#     big_axr2.set_yticks([])
#     big_axr2.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=35)

#     big_axt2 = big_ax2.twiny()
#     big_axt2.set_axis_bgcolor('none')
#     big_axt2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_axt2.set_xticks([])
#     big_axt2.set_yticks([])
#     big_axt2.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=25)

#     cbax2 = fig.add_axes([0.93, 1.02*gs2.bottom, 0.02, 0.98*gs2.top-1.02*gs2.bottom])
#     cbar2 = fig.colorbar(asmdspec, cax=cbax2, orientation='vertical')
#     cbax2.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
#     cbax2.xaxis.set_label_position('top')
    
#     # PLT.tight_layout()
#     # fig.subplots_adjust(right=0.72)
#     # fig.subplots_adjust(top=0.92)

#     PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.png', bbox_inches=0)
#     PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.eps', bbox_inches=0)
#     PLT.close()
    
#     progress.update(j+1)
# progress.finish()

progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots '.format(n_snaps), PGB.ETA()], maxval=n_snaps).start()
for j in range(n_snaps):
    fig = PLT.figure(figsize=(13,8))
    gs1 = GS.GridSpec(3,1)
    gs2 = GS.GridSpec(2,2)
    gs1.update(left=0.08, right=0.35, top=0.93, bottom=0.07, hspace=0)
    gs2.update(left=0.49, right=0.86, top=0.93, bottom=0.07, hspace=0, wspace=0)
    ax10 = PLT.subplot(gs1[0,0])
    ax11 = PLT.subplot(gs1[1,0], sharex=ax10, sharey=ax10)
    ax12 = PLT.subplot(gs1[2,0], sharex=ax10)
    ax20 = PLT.subplot(gs2[0,0])
    ax21 = PLT.subplot(gs2[0,1], sharex=ax20, sharey=ax20)
    ax22 = PLT.subplot(gs2[1,0], sharex=ax20, sharey=ax20)
    ax23 = PLT.subplot(gs2[1,1], sharex=ax21, sharey=ax22)

    dsmsky = ax10.imshow(dsm_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=dsm.min(), vmax=dsm.max()), cmap=CM.jet)
    pbskyc0 = ax10.contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='k', linewidths=1.5)
    ax10.text(0.5, 0.9, 'Diffuse', transform=ax10.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

    csmsky = ax11.scatter(ra_deg_wrapped[src_ind_csm_snapshots[j]], dec_deg[src_ind_csm_snapshots[j]], c=csm_fluxes[src_ind_csm_snapshots[j]], norm=PLTC.LogNorm(vmin=csm_fluxes.min(), vmax=csm_fluxes.max()), cmap=CM.jet, edgecolor='none', s=20)
    pbskyc1 = ax11.contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='k', linewidths=1.5)
    ax11.text(0.5, 0.9, 'Point Sources', transform=ax11.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    
    ax12.plot(anttemp_lst, antpower_K[:,anttemp_nchan/2], '.', lw=2, color='gray')
    ax12.plot(anttemp_lst[m2[j]], antpower_K[m2[j],anttemp_nchan/2], 'ko', ms=8, mew=2, mfc='none')
    ax12.text(0.3, 0.9, 'LST = {0:.1f} deg'.format(anttemp_lst[m2[j]]), transform=ax12.transAxes, fontsize=14, ha='center', color='black')
    ax12.text(0.3, 0.8, r'$T_\mathrm{ant} = $'+'{0:0d} K'.format(int(NP.round(antpower_K[m2[j],anttemp_nchan/2]))), transform=ax12.transAxes, fontsize=14, ha='center', color='black')
    ax12.set_xlim(xvect.max(), xvect.min())
    ax12.set_ylim(0, 1.05*antpower_K[:,anttemp_nchan/2].max())
    ax12.set_xlabel('RA [degrees]', fontsize=18, weight='medium')
    ax12.set_ylabel(r'$T_\mathrm{ant}$'+' [ K ]', fontsize=16, weight='medium')
    # ax12.text(0.5, 0.9, 'Diffuse + Point Sources', transform=ax12.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

    ax10.set_xlim(xvect.max(), xvect.min())
    ax10.set_ylim(yvect.min(), yvect.max())
    ax10.grid(True, which='both')
    ax10.set_aspect('auto')
    ax10.tick_params(which='major', length=12, labelsize=12)
    ax10.tick_params(which='minor', length=6)
    ax10.locator_params(axis='x', nbins=5)
    PLT.setp(ax10.get_xticklabels(), visible=False)

    ax11.set_xlim(xvect.max(), xvect.min())
    ax11.set_ylim(yvect.min(), yvect.max())
    ax11.grid(True, which='both')
    ax11.set_aspect('auto')
    ax11.tick_params(which='major', length=12, labelsize=12)
    ax11.tick_params(which='minor', length=6)
    ax11.locator_params(axis='x', nbins=5)
    PLT.setp(ax11.get_xticklabels(), visible=False)

    ax_ha = ax10.twiny()
    ax_ha.set_xticks(lst[j]-ax10.get_xticks())
    ax_ha.set_xlim(lst[j]-NP.asarray(ax10.get_xlim()))
    # ax_ha.set_xticks(ha(ax10.get_xticks(), lst[j]))
    # ax_ha.set_xlim(ha(NP.asarray(ax10.get_xlim()), lst[j]))
    # ha_ticks = lst[j] - ax10.get_xticks()
    # ha_ticks[ha_ticks > 180.0] = ha_ticks[ha_ticks > 180.0] - 360.0
    # ha_ticks[ha_ticks > 180.0] = ha_ticks[ha_ticks > 180.0] - 360.0    
    # ha_ticks[ha_ticks <= -180.0] = ha_ticks[ha_ticks <= -180.0] + 360.0
    # ha_ticks[ha_ticks <= -180.0] = ha_ticks[ha_ticks <= -180.0] + 360.0
    # ax_ha.set_xticks(ha_ticks)
    # if lst[j] <= 180.0:
    #     ha_xlim = lst[j] - NP.asarray(ax10.get_xlim())
    # else:
    #     ha_xlim = lst[j]-360 - NP.asarray(ax10.get_xlim())
    # ax_ha.set_xlim(ha_xlim)
    xformatter = FuncFormatter(lambda x, pos: '{0:.1f}'.format(x))
    ax_ha.xaxis.set_major_formatter(xformatter)
    ax_ha.set_xlabel('HA [degrees]', fontsize=16, weight='medium')

    pos10 = ax10.get_position()
    pos11 = ax11.get_position()
    x0 = pos10.x0
    x1 = pos10.x1
    y0 = pos11.y0
    y1 = pos10.y1
    big_ax1 = fig.add_axes([x0, y0, x1-x0, y1-y0])
    big_ax1.set_axis_bgcolor('none')
    big_ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax1.set_xticks([])
    big_ax1.set_yticks([])
    big_ax1.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)

    cbax10 = fig.add_axes([0.38, 1.02*pos10.y0, 0.02, 0.98*pos10.y1-1.02*pos10.y0])
    cbar10 = fig.colorbar(dsmsky, cax=cbax10, orientation='vertical')
    cbar10.set_ticks(dsm_fg_ticks.tolist())
    cbar10.set_ticklabels(dsm_fg_ticks.tolist())
    cbax10.set_ylabel('Temperature [K]', labelpad=-65, fontsize=14)

    cbax11 = fig.add_axes([0.38, 1.02*pos11.y0, 0.02, 0.98*pos11.y1-1.02*pos11.y0])
    cbar11 = fig.colorbar(csmsky, cax=cbax11, orientation='vertical')
    cbar11.set_ticks(csm_fg_ticks.tolist())
    cbar11.set_ticklabels(csm_fg_ticks.tolist())
    cbax11.set_ylabel('Flux Density [Jy]', labelpad=-60, fontsize=14)

    dsmdspec = ax20.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e-2)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
    horizonb = ax20.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
    horizont = ax20.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
    ax20.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
    ax20.set_aspect('auto')
    ax20.text(0.5, 0.9, 'Diffuse', transform=ax20.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

    csmspec = ax21.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e-2)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
    horizonb = ax21.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
    horizont = ax21.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
    ax21.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
    ax21.set_aspect('auto')
    ax21.text(0.5, 0.9, 'Point Sources', transform=ax21.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

    asmdspec = ax22.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e-2)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
    horizonb = ax22.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
    horizont = ax22.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
    ax22.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
    ax22.set_aspect('auto')
    ax22.text(0.5, 0.9, 'Diffuse + Point', transform=ax22.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

    eordspec = ax23.pcolorfast(truncated_ref_bl_length, 1e6*eor_ia.lags[eor_small_delays_ind], NP.abs(eor_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e-2)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
    horizonb = ax23.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
    horizont = ax23.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
    ax23.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
    ax23.set_aspect('auto')
    ax23.text(0.5, 0.9, 'HI signal', transform=ax23.transAxes, fontsize=14, weight='semibold', ha='center', color='black')

    
    axs = [ax20, ax21, ax22, ax23]
    for k in xrange(len(axs)):
        if k%2 == 1:
            axs_kprll = axs[k].twinx()
            axs_kprll.set_yticks(kprll(axs[k].get_yticks()*1e-6, redshift))
            axs_kprll.set_ylim(kprll(NP.asarray(axs[k].get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            axs_kprll.yaxis.set_major_formatter(yformatter)
            axs[k].yaxis.set_ticklabels([])
        if k/2 == 0:
            axs_kperp = axs[k].twiny()
            axs_kperp.set_xticks(kperp(axs[k].get_xticks()*freq/FCNST.c, redshift))
            axs_kperp.set_xlim(kperp(NP.asarray(axs[k].get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
            axs_kperp.xaxis.set_major_formatter(xformatter)
                
    big_ax2 = fig.add_axes([gs2.left, gs2.bottom, gs2.right-gs2.left, gs2.top-gs2.bottom])
    big_ax2.set_axis_bgcolor('none')
    big_ax2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax2.set_xticks([])
    big_ax2.set_yticks([])
    big_ax2.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=30)
    big_ax2.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=15)

    big_axr2 = big_ax2.twinx()
    big_axr2.set_axis_bgcolor('none')
    big_axr2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_axr2.set_xticks([])
    big_axr2.set_yticks([])
    big_axr2.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=35)

    big_axt2 = big_ax2.twiny()
    big_axt2.set_axis_bgcolor('none')
    big_axt2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_axt2.set_xticks([])
    big_axt2.set_yticks([])
    big_axt2.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=25)

    cbax2 = fig.add_axes([0.93, 1.02*gs2.bottom, 0.02, 0.98*gs2.top-1.02*gs2.bottom])
    cbar2 = fig.colorbar(asmdspec, cax=cbax2, orientation='vertical')
    cbax2.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
    cbax2.xaxis.set_label_position('top')
    
    PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.png', bbox_inches=0)
    PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_PS_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'_snapshot_{0:03d}'.format(j)+'.eps', bbox_inches=0)
    PLT.close()

    progress.update(j+1)
progress.finish()

