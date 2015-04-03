from mpi4py import MPI 
import argparse
import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import Galactic, FK5
from astropy import units
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
import my_MPI_modules as my_MPI
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

## Set MPI parameters

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
name = MPI.Get_processor_name()

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')

project_group = parser.add_mutually_exclusive_group(required=True)
project_group.add_argument('--project-MWA', dest='project_MWA', action='store_true')
project_group.add_argument('--project-LSTbin', dest='project_LSTbin', action='store_true')
project_group.add_argument('--project-HERA', dest='project_HERA', action='store_true')
project_group.add_argument('--project-beams', dest='project_beams', action='store_true')
project_group.add_argument('--project-drift-scan', dest='project_drift_scan', action='store_true')
project_group.add_argument('--project-global-EoR', dest='project_global_EoR', action='store_true') 

array_config_group = parser.add_mutually_exclusive_group(required=True)
array_config_group.add_argument('--antenna-file', help='File containing antenna locations', type=file, dest='antenna_file')
array_config_group.add_argument('--array-layout', help='Identifier specifying antenna array layout', choices=['MWA-128T', 'HERA-7', 'HERA-19', 'HERA-37', 'HERA-61', 'HERA-91', 'HERA-127', 'HERA-169', 'HERA-217', 'HERA-271', 'HERA-331'], type=str, dest='array_layout')

# parser.add_argument('--antenna-file', help='File containing antenna locations', default='/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', type=file, dest='antenna_file')

telescope_group = parser.add_argument_group('Telescope parameters', 'Telescope/interferometer specifications')
telescope_group.add_argument('--label-prefix', help='Prefix for baseline labels [str, Default = ""]', default='', type=str, dest='label_prefix')
telescope_group.add_argument('--telescope', help='Telescope name [str, default="custom"]', default='custom', type=str, dest='telescope_id', choices=['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'paper_dipole', 'custom', 'mwa_tools'])
telescope_group.add_argument('--latitude', help='Latitude of interferometer array in degrees [float, Default=-26.701]', default=-26.701, type=float, dest='latitude')
telescope_group.add_argument('--A-eff', help='Effective area in m^2', type=float, dest='A_eff', nargs='?')
telescope_group.add_argument('--Tsys', help='System temperature in K [float, Default=440.0]', default=440.0, type=float, dest='Tsys')
telescope_group.add_argument('--pfb-method', help='PFB coarse channel shape computation method [str, Default="theoretical"]', dest='pfb_method', default=None, choices=['theoretical', 'empirical', None])
telescope_group.add_argument('--pfb-file', help='File containing PFB coefficients', type=file, dest='pfb_file', default=None)

antenna_element_group = parser.add_argument_group('Antenna element parameters', 'Antenna element specifications')
antenna_element_group.add_argument('--shape', help='Shape of antenna element [no default]', type=str, dest='antenna_element_shape', default=None, choices=['dish', 'dipole', 'delta'])
antenna_element_group.add_argument('--size', help='Size of dish or length of dipole (in meters) [float, no default]', default=None, type=float, dest='antenna_element_size')
antenna_element_group.add_argument('--orientation', help='Orientation of dipole or pointing direction of dish [float, (altitude azimuth) or (l m [n])]', default=None, type=float, nargs='*', dest='antenna_element_orientation')
antenna_element_group.add_argument('--ocoords', help='Coordinates of dipole orientation or dish pointing direction [str]', default=None, type=str, dest='antenna_element_orientation_coords', choices=['dircos', 'altaz'])
antenna_element_group.add_argument('--phased-array', dest='phased_array', action='store_true')
antenna_element_group.add_argument('--phased-array-file', help='Locations of antenna elements to be phased', default='/data3/t_nithyanandan/project_MWA/MWA_tile_dipole_locations.txt', type=file, dest='phased_elements_file')
antenna_element_group.add_argument('--groundplane', help='Height of antenna element above ground plane (in meters) [float]', default=None, type=float, dest='ground_plane')

obsparm_group = parser.add_argument_group('Observation setup', 'Parameters specifying the observation')
obsparm_group.add_argument('-f', '--freq', help='Foreground center frequency in Hz [float, Default=185e6]', default=185e6, type=float, dest='freq')
obsparm_group.add_argument('--dfreq', help='Frequency resolution in Hz [float, Default=40e3]', default=40e3, type=float, dest='freq_resolution')
obsparm_group.add_argument('--obs-mode', help='Observing mode [str, track/drift/drift-shift/custom]', default=None, type=str, dest='obs_mode', choices=['track', 'drift', 'dns', 'lstbin', 'custom'])
# obsparm_group.add_argument('--t-snap', help='Integration time (seconds) [float, Default=300.0]', default=5.0*60.0, type=float, dest='t_snap')
obsparm_group.add_argument('--nchan', help='Number of frequency channels [int, Default=256]', default=256, type=int, dest='n_channels')
obsparm_group.add_argument('--delayerr', dest='delayerr', type=float, default=0.0, help='RMS error in beamformer delays [ns], default=0')
obsparm_group.add_argument('--gainerr', dest='gainerr', type=float, default=0.0, help='RMS error in beamformer gains [dB], default=0')
obsparm_group.add_argument('--nrandom', dest='nrand', type=int, default=1, help='numner of random realizations of gains and/or delays, default=1')
# obsparm_group.add_argument('--lst-init', help='LST at beginning of observing run (hours) [float]', type=float, dest='lst_init', required=True, metavar='LST')
# obsparm_group.add_argument('--pointing-init', help='Pointing (RA, Dec) at beginning of observing run (degrees) [float]', type=float, dest='pointing_init', metavar=('RA', 'Dec'), required=True, nargs=2)

duration_group = parser.add_argument_group('Observing duration parameters', 'Parameters specifying observing duration')
duration_group.add_argument('--t-obs', help='Duration of observation [seconds]', dest='t_obs', default=None, type=float, metavar='t_obs')
duration_group.add_argument('--n-snap', help='Number of snapshots or records that make up the observation', dest='n_snaps', default=None, type=int, metavar='n_snapshots')
duration_group.add_argument('--t-snap', help='integration time of each snapshot [seconds]', dest='t_snap', default=None, type=float, metavar='t_snap')

snapshot_selection_group = parser.add_mutually_exclusive_group(required=True)
snapshot_selection_group.add_argument('--avg-drifts', dest='avg_drifts', action='store_true')
snapshot_selection_group.add_argument('--beam-switch', dest='beam_switch', action='store_true')
snapshot_selection_group.add_argument('--snap-sampling', dest='snapshot_sampling', default=None, type=int, nargs=1)
snapshot_selection_group.add_argument('--snap-pick', dest='pick_snapshots', default=None, type=int, nargs='*')
snapshot_selection_group.add_argument('--snap-range', dest='snapshots_range', default=None, nargs=2, type=int)
snapshot_selection_group.add_argument('--all-snaps', dest='all_snapshots', action='store_true')

pointing_group = parser.add_mutually_exclusive_group(required=True)
pointing_group.add_argument('--pointing-file', dest='pointing_file', type=str, nargs=1, default=None)
pointing_group.add_argument('--pointing-info', dest='pointing_info', type=float, nargs=3, metavar=('lst_init', 'ra_init', 'dec_init'))

processing_group = parser.add_argument_group('Processing arguments', 'Processing parameters')
processing_group.add_argument('--n-bins-blo', help='Number of bins for baseline orientations [int, Default=4]', default=4, type=int, dest='n_bins_baseline_orientation')
processing_group.add_argument('--bl-chunk-size', help='Baseline chunk size [int, Default=100]', default=100, type=int, dest='baseline_chunk_size')
processing_group.add_argument('--bl-chunk', help='Baseline chunk indices to process [int(s), Default=None: all chunks]', default=None, type=int, dest='bl_chunk', nargs='*')
processing_group.add_argument('--n-bl-chunks', help='Upper limit on baseline chunks to be processed [int, Default=None]', default=None, type=int, dest='n_bl_chunks')
processing_group.add_argument('--n-sky-sectors', help='Divide sky into sectors relative to zenith [int, Default=1]', default=1, type=int, dest='n_sky_sectors')
processing_group.add_argument('--bpw', help='Bandpass window shape [str, "rect"]', default='rect', type=str, dest='bpass_shape', choices=['rect', 'bnw', 'bhw'])
processing_group.add_argument('--f-pad', help='Frequency padding fraction for delay transform [float, Default=1.0]', type=float, dest='f_pad', default=1.0)
processing_group.add_argument('--coarse-channel-width', help='Width of coarse channel [int: number of fine channels]', dest='coarse_channel_width', default=32, type=int)
processing_group.add_argument('--bp-correct', help='Bandpass correction', dest='bp_correct', action='store_true')
processing_group.add_argument('--noise-bp-correct', help='Bandpass correction for Tsys', dest='noise_bp_correct', action='store_true')
processing_group.add_argument('--bpw-pad', help='Bandpass window padding length [int, Default=0]', dest='n_pad', default=0, type=int)

mpi_group = parser.add_mutually_exclusive_group(required=True)
mpi_group.add_argument('--mpi-on-src', action='store_true')
mpi_group.add_argument('--mpi-on-bl', action='store_true')

more_mpi_group = parser.add_mutually_exclusive_group(required=True)
more_mpi_group.add_argument('--mpi-async', action='store_true')
more_mpi_group.add_argument('--mpi-sync', action='store_true')

freq_flags_group = parser.add_argument_group('Frequency flagging', 'Parameters to describe flagging of bandpass')
freq_flags_group.add_argument('--flag-channels', help='Bandpass channels to be flagged. If bp_flag_repeat is set, bp_flag(s) will be forced in the range 0 <= flagged channel(s) < coarse_channel_width and applied to all coarse channels periodically [int,  default=-1: no flag]', dest='flag_chan', nargs='*', default=-1, type=int)
freq_flags_group.add_argument('--bp-flag-repeat', help='If set, will repeat any flag_chan(s) for all coarse channels after converting flag_chan(s) to lie in the range 0 <= flagged channel(s) < coarse_channel_width using flag_chan modulo coarse_channel_width', action='store_true', dest='bp_flag_repeat')
freq_flags_group.add_argument('--flag-edge-channels', help='Flag edge channels in the band. If flag_repeat_edge_channels is set, specified number of channels leading and trailing the coarse channel edges are flagged. First number includes the coarse channel minimum while the second number does not. Otherwise, specified number of channels are flagged at the beginning and end of the band. [int,int Default=0,0]', dest='n_edge_flag', nargs=2, default=[0,0], metavar=('NEDGE1','NEDGE2'), type=int)
freq_flags_group.add_argument('--flag-repeat-edge-channels', help='If set, will flag the leading and trailing channels whose number is specified in n_edge_flag. Otherwise, will flag the beginning and end of the band.', action='store_true', dest='flag_repeat_edge_channels')

skymodel_group = parser.add_mutually_exclusive_group(required=True)
skymodel_group.add_argument('--ASM', action='store_true') # Diffuse (GSM) + Compact (NVSS+SUMSS) All-sky model 
skymodel_group.add_argument('--DSM', action='store_true') # Diffuse all-sky model
skymodel_group.add_argument('--CSM', action='store_true') # Point source model (NVSS+SUMSS)
skymodel_group.add_argument('--SUMSS', action='store_true') # SUMSS catalog
skymodel_group.add_argument('--NVSS', action='store_true') # NVSS catalog
skymodel_group.add_argument('--MSS', action='store_true') # Molonglo Sky Survey
skymodel_group.add_argument('--GLEAM', action='store_true') # GLEAM catalog
skymodel_group.add_argument('--PS', action='store_true') # Point sources 
skymodel_group.add_argument('--USM', action='store_true') # Uniform all-sky model
skymodel_group.add_argument('--HI-monopole', action='store_true') # Global EoR model
skymodel_group.add_argument('--HI-fluctuations', action='store_true') # HI EoR fluctuations
skymodel_group.add_argument('--HI-cube', action='store_true') # HI EoR simulation cube

skyparm_group = parser.add_argument_group('Sky Model Setup', 'Parameters describing sky model')
skyparm_group.add_argument('--flux-unit', help='Units of flux density [str, Default="Jy"]', type=str, dest='flux_unit', default='Jy', choices=['Jy','K'])
skyparm_group.add_argument('--lidz', help='Simulations of Adam Lidz', action='store_true')
skyparm_group.add_argument('--21cmfast', help='21CMFAST Simulations of Andrei Mesinger', action='store_true')
skyparm_group.add_argument('--HI-monopole-parms', help='Parameters defining global HI signal', dest='global_HI_parms', default=None, type=float, nargs=3, metavar=('T_xi0', 'freq_half', 'dfreq_half'))

skycat_group = parser.add_argument_group('Catalog files', 'Catalog file locations')
skycat_group.add_argument('--dsm-file-prefix', help='Diffuse sky model filename prefix [str]', type=str, dest='DSM_file_prefix', default='/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata')
skycat_group.add_argument('--sumss-file', help='SUMSS catalog file [str]', type=str, dest='SUMSS_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt')
skycat_group.add_argument('--nvss-file', help='NVSS catalog file [str]', type=file, dest='NVSS_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits')
skycat_group.add_argument('--GLEAM-file', help='GLEAM catalog file [str]', type=str, dest='GLEAM_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv')
skycat_group.add_argument('--PS-file', help='Point source catalog file [str]', type=str, dest='PS_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/PS_catalog.txt')
# skycat_group.add_argument('--HI-file-prefix', help='EoR simulation filename [str]', type=str, dest='HI_file_prefix', default='/data3/t_nithyanandan/EoR_simulations/Adam_Lidz/Boom_tiles/hpxextn_138.915-195.235_MHz_80.0_kHz_nside_64.fits')

fgparm_group = parser.add_argument_group('Foreground Setup', 'Parameters describing foreground sky')
fgparm_group.add_argument('--spindex', help='Spectral index, ~ f^spindex [float, Default=0.0]', type=float, dest='spindex', default=0.0)
fgparm_group.add_argument('--spindex-rms', help='Spectral index rms [float, Default=0.0]', type=float, dest='spindex_rms', default=0.0)
fgparm_group.add_argument('--spindex-seed', help='Spectral index seed [float, Default=None]', type=int, dest='spindex_seed', default=None)
fgparm_group.add_argument('--nside', help='nside parameter for healpix map [int, Default=64]', type=int, dest='nside', default=64, choices=[64, 128])

parser.add_argument('--plots', help='Create plots', action='store_true', dest='plots')

args = vars(parser.parse_args())

project_MWA = args['project_MWA']
project_LSTbin = args['project_LSTbin']
project_HERA = args['project_HERA']
project_beams = args['project_beams']
project_drift_scan = args['project_drift_scan']
project_global_EoR = args['project_global_EoR']

if project_MWA: project_dir = 'project_MWA'
if project_LSTbin: project_dir = 'project_LSTbin'
if project_HERA: project_dir = 'project_HERA'
if project_beams: project_dir = 'project_beams'
if project_drift_scan: project_dir = 'project_drift_scan'
if project_global_EoR: project_dir = 'project_global_EoR'

antenna_file = args['antenna_file']
array_layout = args['array_layout']

if antenna_file is not None: 
    try:
        ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
        ant_id = ant_info[:,0].astype(int).astype(str)
        ant_locs = ant_info[:,1:]
    except IOError:
        raise IOError('Could not open file containing antenna locations.')
else:
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

n_bins_baseline_orientation = args['n_bins_baseline_orientation']
baseline_chunk_size = args['baseline_chunk_size']
bl_chunk = args['bl_chunk']
n_bl_chunks = args['n_bl_chunks']
telescope_id = args['telescope_id']
element_shape = args['antenna_element_shape']
element_size = args['antenna_element_size']
element_orientation = args['antenna_element_orientation']
element_ocoords = args['antenna_element_orientation_coords']
phased_array = args['phased_array']
phased_elements_file = args['phased_elements_file']

if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole'):
    element_size = 0.74
    element_shape = 'dipole'
    if telescope_id == 'mwa': phased_array = True
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
    if element_shape != 'delta':
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

if element_orientation is None:
    if element_ocoords is not None:
        if element_ocoords == 'altaz':
            if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole') or (element_shape == 'dipole'):
                element_orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
            else:
                element_orientation = NP.asarray([90.0, 270.0]).reshape(1,-1)
        elif element_ocoords == 'dircos':
            if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole') or (element_shape == 'dipole'):
                element_orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1)
            else:
                element_orientation = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
        else:
            raise ValueError('Invalid value specified antenna element orientation coordinate system.')
    else:
        if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole') or (element_shape == 'dipole'):
            element_orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
        else:
            element_orientation = NP.asarray([90.0, 270.0]).reshape(1,-1)
        element_ocoords = 'altaz'
else:
    if element_ocoords is None:
        raise ValueError('Antenna element orientation coordinate system must be specified to describe the specified antenna orientation.')

element_orientation = NP.asarray(element_orientation).reshape(1,-1)
if (element_orientation.size < 2) or (element_orientation.size > 3):
    raise ValueError('Antenna element orientation must be a two- or three-element vector.')
elif (element_ocoords == 'altaz') and (element_orientation.size != 2):
    raise ValueError('Antenna element orientation must be a two-element vector if using Alt-Az coordinates.')

ground_plane = args['ground_plane']
if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane

freq = args['freq']
freq_resolution = args['freq_resolution']
latitude = args['latitude']

if args['A_eff'] is None:
    if (telescope['shape'] == 'dipole') or (telescope['shape'] == 'delta'):
        A_eff = (0.5*FCNST.c/freq)**2
        if (telescope_id == 'mwa') or phased_array:
            A_eff *= 16
    if telescope['shape'] == 'dish':
        A_eff = NP.pi * (0.5*element_size)**2
else:
    A_eff = args['A_eff']

obs_mode = args['obs_mode']
Tsys = args['Tsys']
t_snap = args['t_snap']
t_obs = args['t_obs']
n_snaps = args['n_snaps']
avg_drifts = args['avg_drifts']
beam_switch = args['beam_switch']
snapshot_sampling = args['snapshot_sampling']
pick_snapshots = args['pick_snapshots']
all_snapshots = args['all_snapshots']
snapshots_range = args['snapshots_range']
snapshot_type_str = ''

if avg_drifts and (obs_mode == 'dns'):
    snapshot_type_str = 'drift_averaged_'

if beam_switch and (obs_mode == 'dns'):
    snapshot_type_str = 'beam_switches_'

if (snapshots_range is not None) and ((obs_mode == 'dns') or (obs_mode == 'lstbin')):
    snapshot_type_str = 'snaps_{0[0]:0d}-{0[1]:0d}_'.format(snapshots_range)

pointing_file = args['pointing_file']
if pointing_file is not None:
    pointing_file = pointing_file[0]
pointing_info = args['pointing_info']

delayerr = args['delayerr']
if delayerr is None:
    delayerr_str = ''
    delayerr = 0.0
elif delayerr < 0.0:
    raise ValueError('delayerr must be non-negative.')
else:
    delayerr_str = 'derr_{0:.3f}ns'.format(delayerr)
delayerr *= 1e-9

gainerr = args['gainerr']
if gainerr is None:
    gainerr_str = ''
    gainerr = 0.0
elif gainerr < 0.0:
    raise ValueError('gainerr must be non-negative.')
else:
    gainerr_str = '_gerr_{0:.2f}dB'.format(gainerr)

nrand = args['nrand']
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
    
element_locs = None
if phased_array:
    try:
        element_locs = NP.loadtxt(phased_elements_file, skiprows=1, comments='#', usecols=(0,1,2))
    except IOError:
        raise IOError('Could not open the specified file for phased array of antenna elements.')

if telescope_id == 'mwa':
    xlocs, ylocs = NP.meshgrid(1.1*NP.linspace(-1.5,1.5,4), 1.1*NP.linspace(1.5,-1.5,4))
    element_locs = NP.hstack((xlocs.reshape(-1,1), ylocs.reshape(-1,1), NP.zeros(xlocs.size).reshape(-1,1)))

if element_locs is not None:
    telescope['element_locs'] = element_locs

duration_str = ''
if pointing_file is not None:
    pointing_init = None
    pointing_info_from_file = NP.loadtxt(pointing_file, comments='#', usecols=(1,2,3), delimiter=',')
    obs_id = NP.loadtxt(pointing_file, comments='#', usecols=(0,), delimiter=',', dtype=str)
    if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
        delays_str = NP.loadtxt(pointing_file, comments='#', usecols=(4,), delimiter=',', dtype=str)
        delays_list = [NP.fromstring(delaystr, dtype=float, sep=';', count=-1) for delaystr in delays_str]
        delay_settings = NP.asarray(delays_list)
        delay_settings *= 435e-12
        delays = NP.copy(delay_settings)
    if n_snaps is None:
        n_snaps = pointing_info_from_file.shape[0]
    pointing_info_from_file = pointing_info_from_file[:min(n_snaps, pointing_info_from_file.shape[0]),:]
    obs_id = obs_id[:min(n_snaps, pointing_info_from_file.shape[0])]
    if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
        delays = delay_settings[:min(n_snaps, pointing_info_from_file.shape[0]),:]
    n_snaps = min(n_snaps, pointing_info_from_file.shape[0])
    pointings_altaz = pointing_info_from_file[:,:2].reshape(-1,2)
    pointings_altaz_orig = pointing_info_from_file[:,:2].reshape(-1,2)
    lst = 15.0 * pointing_info_from_file[:,2]
    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

    if obs_mode is None:
        obs_mode = 'custom'
    if (obs_mode == 'dns') and (avg_drifts or beam_switch):
        angle_diff = GEOM.sphdist(pointings_altaz[1:,1], pointings_altaz[1:,0], pointings_altaz[:-1,1], pointings_altaz[:-1,0])
        angle_diff = NP.concatenate(([0.0], angle_diff))
        shift_threshold = 1.0 # in degrees
        # lst_edges = NP.concatenate(([lst_edges[0]], lst_edges[angle_diff > shift_threshold], [lst_edges[-1]]))
        lst_wrapped = NP.concatenate(([lst_wrapped[0]], lst_wrapped[angle_diff > shift_threshold], [lst_wrapped[-1]]))
        n_snaps = lst_wrapped.size - 1
        pointings_altaz = NP.vstack((pointings_altaz[0,:].reshape(-1,2), pointings_altaz[angle_diff>shift_threshold,:].reshape(-1,2)))
        obs_id = NP.concatenate(([obs_id[0]], obs_id[angle_diff>shift_threshold]))
        if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
            delays = NP.vstack((delay_settings[0,:], delay_settings[angle_diff>shift_threshold,:]))
        obs_mode = 'custom'
        if avg_drifts:
            lst_edges = NP.concatenate(([lst_edges[0]], lst_edges[angle_diff > shift_threshold], [lst_edges[-1]]))
        else:
            lst_edges_left = lst_wrapped[:-1] + 0.0
            lst_edges_right = NP.concatenate(([lst_edges[1]], lst_edges[NP.asarray(NP.where(angle_diff > shift_threshold)).ravel()+1]))
    elif snapshots_range is not None:
        snapshots_range[1] = snapshots_range[1] % n_snaps
        if snapshots_range[0] > snapshots_range[1]:
            raise IndexError('min snaphost # must be <= max snapshot #')
        lst_wrapped = lst_wrapped[snapshots_range[0]:snapshots_range[1]+2]
        lst_edges = NP.copy(lst_wrapped)
        pointings_altaz = pointings_altaz[snapshots_range[0]:snapshots_range[1]+1,:]
        obs_id = obs_id[snapshots_range[0]:snapshots_range[1]+1]
        if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
            delays = delay_settings[snapshots_range[0]:snapshots_range[1]+1,:]
        n_snaps = snapshots_range[1]-snapshots_range[0]+1
    elif pick_snapshots is not None:
        pick_snapshots = NP.asarray(pick_snapshots)
        n_snaps = pick_snapshots.size
        lst_begin = NP.asarray(lst_wrapped[pick_snapshots])
        pointings_altaz = pointings_altaz[pick_snapshots,:]
        obs_id = obs_id[pick_snapshots]
        if (telescope_id == 'mwa') or (phased_array) or (telescope_id == 'mwa_tools'):
            delays = delay_settings[pick_snapshots,:]

        if obs_mode != 'lstbin':
            lst_end = NP.asarray(lst_wrapped[pick_snapshots+1])
            t_snap = (lst_end - lst_begin) / 15.0 * 3.6e3
            # n_snaps = t_snap.size
            lst = 0.5 * (lst_begin + lst_end)
            obs_mode = 'custom'
        else:
            t_snap = 112.0 + NP.zeros(n_snaps)   # in seconds (needs to be generalized)
            lst = lst_wrapped + 0.5 * t_snap/3.6e3 * 15.0
    if pick_snapshots is None:
        if obs_mode != 'lstbin':        
            if not beam_switch:
                lst = 0.5*(lst_edges[1:]+lst_edges[:-1])
                t_snap = (lst_edges[1:]-lst_edges[:-1]) / 15.0 * 3.6e3
            else:
                lst = 0.5*(lst_edges_left + lst_edges_right)
                t_snap = (lst_edges_right - lst_edges_left) / 15.0 * 3.6e3
        else:
            t_snap = 112.0 + NP.zeros(n_snaps)   # in seconds (needs to be generalized)
            lst = lst_wrapped + 0.5 * t_snap/3.6e3 * 15.0

    # pointings_dircos_orig = GEOM.altaz2dircos(pointings_altaz_orig, units='degrees')
    # pointings_hadec_orig = GEOM.altaz2hadec(pointings_altaz_orig, latitude, units='degrees')
    # pointings_radec_orig = NP.hstack(((lst-pointings_hadec_orig[:,0]).reshape(-1,1), pointings_hadec_orig[:,1].reshape(-1,1)))
    # pointings_radec_orig[:,0] = pointings_radec_orig[:,0] % 360.0

    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')
    pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
    pointings_radec[:,0] = pointings_radec[:,0] % 360.0
    t_obs = NP.sum(t_snap)
elif pointing_info is not None:
    pointing_init = NP.asarray(pointing_info[1:])
    lst_init = pointing_info[0]
    pointing_file = None
    if t_snap is None:
        raise NameError('t_snap must be provided for an automated observing run')

    if (n_snaps is None) and (t_obs is None):
        raise NameError('n_snaps or t_obs must be provided for an automated observing run')
    elif (n_snaps is not None) and (t_obs is not None):
        raise ValueError('Only one of n_snaps or t_obs must be provided for an automated observing run')
    elif n_snaps is None:
        n_snaps = int(t_obs/t_snap)
    else:
        t_obs = n_snaps * t_snap
    t_snap = t_snap + NP.zeros(n_snaps)
    lst = (lst_init + (t_snap/3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
    if obs_mode is None:
        obs_mode = 'track'

    if obs_mode == 'track':
        pointings_radec = NP.repeat(NP.asarray(pointing_init).reshape(-1,2), n_snaps, axis=0)
    else:
        ha_init = lst_init * 15.0 - pointing_init[0]
        pointings_radec = NP.hstack((NP.asarray(lst-ha_init).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

    pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
    pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

    pointings_radec_orig = NP.copy(pointings_radec)
    pointings_hadec_orig = NP.copy(pointings_hadec)
    pointings_altaz_orig = NP.copy(pointings_altaz)
    pointings_dircos_orig = NP.copy(pointings_dircos)

    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    if lst_wrapped.size > 1:
        lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))
    else:
        lst_edges = NP.concatenate((lst_wrapped, lst_wrapped+t_snap/3.6e3*15))

    duration_str = '_{0:0d}x{1:.1f}s'.format(n_snaps, t_snap[0])

n_channels = args['n_channels']
bpass_shape = args['bpass_shape']
oversampling_factor = 1.0 + args['f_pad']
n_pad = args['n_pad']
pfb_method = args['pfb_method']
bandpass_correct = args['bp_correct']
noise_bandpass_correct = args['noise_bp_correct']
flag_chan  = NP.asarray(args['flag_chan']).reshape(-1)
bp_flag_repeat = args['bp_flag_repeat']
coarse_channel_width = args['coarse_channel_width']
n_edge_flag = NP.asarray(args['n_edge_flag']).reshape(-1)
flag_repeat_edge_channels = args['flag_repeat_edge_channels']

nside = args['nside']
use_GSM = args['ASM']
use_DSM = args['DSM']
use_CSM = args['CSM']
use_NVSS = args['NVSS']
use_SUMSS = args['SUMSS']
use_MSS = args['MSS']
use_GLEAM = args['GLEAM']
use_PS = args['PS']
use_USM = args['USM']
use_HI_monopole = args['HI_monopole']
use_HI_fluctuations = args['HI_fluctuations']
use_HI_cube = args['HI_cube']
use_lidz = args['lidz']
use_21cmfast = args['21cmfast']
global_HI_parms = args['global_HI_parms']
if global_HI_parms is not None:
    T_xi0 = global_HI_parms[0]
    freq_half = global_HI_parms[1]
    dfreq_half = global_HI_parms[2]

bl, bl_id = RI.baseline_generator(ant_locs, ant_id=ant_id, auto=False, conjugate=False)
bl, select_bl_ind, bl_count = RI.uniq_baselines(bl)
bl_id = bl_id[select_bl_ind]
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_id = bl_id[sortind]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
bl_count = bl_count[sortind]
neg_bl_orientation_ind = bl_orientation < 0.0
# neg_bl_orientation_ind = NP.logical_or(bl_orientation < -0.5*180.0/n_bins_baseline_orientation, bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)

if use_HI_monopole:
    bllstr = map(str, bl_length)
    uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
    count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
    count_uniq_bll = NP.asarray(count_uniq_bll)

    bl = bl[ind_uniq_bll,:]
    bl_id = bl_id[ind_uniq_bll]
    bl_orientation = bl_orientation[ind_uniq_bll]
    bl_length = bl_length[ind_uniq_bll]

    sortind = NP.argsort(bl_length, kind='mergesort')
    bl = bl[sortind,:]
    bl_id = bl_id[sortind]
    bl_length = bl_length[sortind]
    bl_orientation = bl_orientation[sortind]
    count_uniq_bll = count_uniq_bll[sortind]

total_baselines = bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
try:
    labels = bl_id.tolist()
except NameError:
    labels = []
    labels += [args['label_prefix']+'{0:0d}'.format(i+1) for i in xrange(bl.shape[0])]
if bl_chunk is None:
    bl_chunk = range(len(baseline_bin_indices))
if n_bl_chunks is None:
    n_bl_chunks = len(bl_chunk)
bl_chunk = bl_chunk[:n_bl_chunks]

mpi_on_src = args['mpi_on_src']
mpi_on_bl = args['mpi_on_bl']
mpi_async = args['mpi_async']
mpi_sync = args['mpi_sync']

plots = args['plots']

nchan = n_channels
base_bpass = 1.0*NP.ones(nchan)
bandpass_shape = 1.0*NP.ones(nchan)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution)/ 1e9 # in GHz
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)

flagged_edge_channels = []
pfb_str = ''
if pfb_method is not None:
    if pfb_method == 'empirical':
        bandpass_shape = DSP.PFB_empirical(nchan, 32, 0.25, 0.25)
    elif pfb_method == 'theoretical':
        pfbhdulist = fits.open(args['pfb_file'])
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
    if flag_repeat_edge_channels:
        if NP.any(n_edge_flag > 0): 
            pfb_edge_channels = (bandpass_shape.argmin() + NP.arange(n_channels/coarse_channel_width)*coarse_channel_width) % n_channels
            # pfb_edge_channels = bandpass_shape.argsort()[:int(1.0*n_channels/coarse_channel_width)]
            # wts = NP.exp(-0.5*((NP.arange(bandpass_shape.size)-0.5*bandpass_shape.size)/4.0)**2)/(4.0*NP.sqrt(2*NP.pi))
            # wts_shift = NP.fft.fftshift(wts)
            # freq_wts = NP.fft.fft(wts_shift)
            # pfb_filtered = DSP.fft_filter(bandpass_shape.ravel(), wts=freq_wts.ravel(), passband='high')
            # pfb_edge_channels = pfb_filtered.argsort()[:int(1.0*n_channels/coarse_channel_width)]

            pfb_edge_channels = NP.hstack((pfb_edge_channels.ravel(), NP.asarray([pfb_edge_channels.min()-coarse_channel_width, pfb_edge_channels.max()+coarse_channel_width])))
            flagged_edge_channels += [range(max(0,pfb_edge-n_edge_flag[0]),min(n_channels,pfb_edge+n_edge_flag[1])) for pfb_edge in pfb_edge_channels]
else:
    pfb_str = 'no_pfb_'

window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=n_pad, centering=True, area_normalize=True) 
if bandpass_correct:
    bpcorr = 1/bandpass_shape
    bandpass_shape = NP.ones(base_bpass.size)
else:
    bpcorr = 1.0*NP.ones(nchan)

noise_bpcorr = 1.0*NP.ones(nchan)
if noise_bandpass_correct:
    noise_bpcorr = NP.copy(bpcorr)

if not flag_repeat_edge_channels:
    flagged_edge_channels += [range(0,n_edge_flag[0])]
    flagged_edge_channels += [range(n_channels-n_edge_flag[1],n_channels)]

flagged_channels = flagged_edge_channels
if flag_chan[0] >= 0:
    flag_chan = flag_chan[flag_chan < n_channels]
    if bp_flag_repeat:
        flag_chan = NP.mod(flag_chan, coarse_channel_width)
        flagged_channels += [[i*coarse_channel_width+flagchan for i in range(n_channels/coarse_channel_width) for flagchan in flag_chan]]
    else:
        flagged_channels += [flag_chan.tolist()]
flagged_channels = [x for y in flagged_channels for x in y]
flagged_channels = list(set(flagged_channels))

bandpass_shape[flagged_channels] = 0.0
bpass = base_bpass * bandpass_shape

n_sky_sectors = args['n_sky_sectors']
if (n_sky_sectors < 1):
    n_sky_sectors = 1

if use_HI_monopole or use_HI_fluctuations or use_HI_cube:
    if use_lidz and use_21cmfast:
        raise ValueError('Only one of Adam Lidz or 21CMFAST simulations can be chosen')
    if not use_lidz and not use_21cmfast:
        use_lidz = True
        use_21cmfast = False
        eor_simfile = '/data3/t_nithyanandan/EoR_simulations/Adam_Lidz/Boom_tiles/hpxcube_138.915-195.235_MHz_80.0_kHz_nside_{0:0d}.fits'.format(nside)
    elif use_lidz:
        eor_simfile = '/data3/t_nithyanandan/EoR_simulations/Adam_Lidz/Boom_tiles/hpxcube_138.915-195.235_MHz_80.0_kHz_nside_{0:0d}.fits'.format(nside)
    elif use_21cmfast:
        pass

# if plots:
#     if rank == 0:

#         ## Plot the pointings

#         pointings_ha_orig = pointings_hadec_orig[:,0]
#         pointings_ha_orig[pointings_ha_orig > 180.0] = pointings_ha_orig[pointings_ha_orig > 180.0] - 360.0
    
#         pointings_ra_orig = pointings_radec_orig[:,0]
#         pointings_ra_orig[pointings_ra_orig > 180.0] = pointings_ra_orig[pointings_ra_orig > 180.0] - 360.0
    
#         pointings_dec_orig = pointings_radec_orig[:,1]
    
#         fig = PLT.figure(figsize=(6,6))
#         ax1a = fig.add_subplot(111)
#         ax1a.set_xlabel('Local Sidereal Time [hours]', fontsize=18, weight='medium')
#         ax1a.set_ylabel('Longitude [degrees]', fontsize=18, weight='medium')
#         ax1a.set_xlim((lst_wrapped.min()-1)/15.0, (lst_wrapped.max()+1)/15.0)
#         ax1a.set_ylim(pointings_ha_orig.min()-15.0, pointings_ha_orig.max()+15.0)
#         ax1a.plot(lst_wrapped/15.0, pointings_ha_orig, 'k--', lw=2, label='HA')
#         ax1a.plot(lst_wrapped/15.0, pointings_ra_orig, 'k-', lw=2, label='RA')
#         ax1a.tick_params(which='major', length=18, labelsize=12)
#         ax1a.tick_params(which='minor', length=12, labelsize=12)
#         legend1a = ax1a.legend(loc='upper left')
#         legend1a.draw_frame(False)
#         for axis in ['top','bottom','left','right']:
#             ax1a.spines[axis].set_linewidth(2)
#         xticklabels = PLT.getp(ax1a, 'xticklabels')
#         yticklabels = PLT.getp(ax1a, 'yticklabels')
#         PLT.setp(xticklabels, fontsize=15, weight='medium')
#         PLT.setp(yticklabels, fontsize=15, weight='medium')    
    
#         ax1b = ax1a.twinx()
#         ax1b.set_ylabel('Declination [degrees]', fontsize=18, weight='medium')
#         ax1b.set_ylim(pointings_dec_orig.min()-5.0, pointings_dec_orig.max()+5.0)
#         ax1b.plot(lst_wrapped/15.0, pointings_dec_orig, 'k:', lw=2, label='Dec')
#         ax1b.tick_params(which='major', length=12, labelsize=12)
#         legend1b = ax1b.legend(loc='upper center')
#         legend1b.draw_frame(False)
#         yticklabels = PLT.getp(ax1b, 'yticklabels')
#         PLT.setp(yticklabels, fontsize=15, weight='medium')    
    
#         fig.subplots_adjust(right=0.85)
    
#         PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+obs_mode+'_pointings.eps', bbox_inches=0)
#         PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+obs_mode+'_pointings.png', bbox_inches=0)

#         ## Plot bandpass properties

#         fig = PLT.figure(figsize=(7,6))
#         ax = fig.add_subplot(111)
#         ax.set_xlabel('frequency [MHz]', fontsize=18, weight='medium')
#         ax.set_ylabel('gain', fontsize=18, weight='medium')
#         ax.set_xlim(freq*1e-6 - 2.0, freq*1e-6 + 2.0)
#         ax.set_ylim(0.05, 2.0*bpcorr.max())
#         ax.set_yscale('log')
#         try:
#             ax.plot(1e3*chans, 10**(pfbwin_interp/10), 'k.--', lw=2, ms=10, label='Instrumental PFB Bandpass')
#         except NameError:
#             pass
#         ax.plot(1e3*chans, bpcorr, 'k+:', lw=2, ms=10, label='Bandpass Correction')
#         ax.plot(1e3*chans, bandpass_shape, 'k-', lw=2, label='Corrected Bandpass (Flagged)')
#         # ax.plot(1e3*chans, 3.0+NP.zeros(n_channels), 'k-.', label='Flagging threshold')
#         legend = ax.legend(loc='lower center')
#         legend.draw_frame(False)
#         ax.tick_params(which='major', length=18, labelsize=12)
#         ax.tick_params(which='minor', length=12, labelsize=12)
#         for axis in ['top','bottom','left','right']:
#             ax.spines[axis].set_linewidth(2)
#         xticklabels = PLT.getp(ax, 'xticklabels')
#         yticklabels = PLT.getp(ax, 'yticklabels')
#         PLT.setp(xticklabels, fontsize=15, weight='medium')
#         PLT.setp(yticklabels, fontsize=15, weight='medium')    

#         PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/bandpass_properties.eps', bbox_inches=0)
#         PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/bandpass_properties.png', bbox_inches=0)

fg_str = ''
flux_unit = args['flux_unit']
spindex_seed = args['spindex_seed']
spindex_rms = args['spindex_rms']
spindex_rms_str = ''
spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

if use_HI_fluctuations or use_HI_cube:
    # if freq_resolution != 80e3:
    #     raise ValueError('Currently frequency resolution can only be set to 80 kHz')

    fg_str = 'HI_cube'
    hdulist = fits.open(eor_simfile)
    nexten = hdulist['PRIMARY'].header['NEXTEN']
    fitstype = hdulist['PRIMARY'].header['FITSTYPE']
    temperatures = None
    extnames = [hdulist[i].header['EXTNAME'] for i in xrange(1,nexten+1)]
    if fitstype == 'IMAGE':
        eor_simfreq = hdulist['FREQUENCY'].data['Frequency [MHz]']
    else:
        eor_simfreq = [float(extname.split(' ')[0]) for extname in extnames]
        eor_simfreq = NP.asarray(eor_simfreq)

    eor_freq_resolution = eor_simfreq[1] - eor_simfreq[0]
    ind_chans, ind_eor_simfreq, dfrequency = LKP.find_1NN(eor_simfreq.reshape(-1,1), 1e3*chans.reshape(-1,1), distance_ULIM=0.5*eor_freq_resolution, remove_oob=True)

    eor_simfreq = eor_simfreq[ind_eor_simfreq]
    if fitstype == 'IMAGE':
        temperatures = hdulist['TEMPERATURE'].data[:,ind_eor_simfreq]
    else:
        for i in xrange(eor_simfreq.size):
            if i == 0:
                temperatures = hdulist[ind_eor_simfreq[i]+1].data['Temperature'].reshape(-1,1)
            else:
                temperatures = NP.hstack((temperatures, hdulist[ind_eor_simfreq[i]+1].data['Temperature'].reshape(-1,1)))

    if use_HI_fluctuations:
        temperatures = temperatures - NP.mean(temperatures, axis=0, keepdims=True)
        fg_str = 'HI_fluctuations'

    # if use_HI_monopole:
    #     shp_temp = temperatures.shape
    #     temperatures = NP.mean(temperatures, axis=0, keepdims=True) + NP.zeros(shp_temp)
    #     fg_str = 'HI_monopole'
    # elif use_HI_fluctuations:
    #     temperatures = temperatures - NP.mean(temperatures, axis=0, keepdims=True)
    #     fg_str = 'HI_fluctuations'

    pixres = hdulist['PRIMARY'].header['PIXAREA']
    coords_table = hdulist['COORDINATE'].data
    ra_deg_EoR = coords_table['RA']
    dec_deg_EoR = coords_table['DEC']
    fluxes_EoR = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    freq_EoR = freq/1e9
    hdulist.close()

    catlabel = 'HI-cube'
    spec_type = 'spectrum'
    spec_parms = {}
    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg_EoR.reshape(-1,1), dec_deg_EoR.reshape(-1,1))), spec_type, spectrum=fluxes_EoR, spec_parms=None)

elif use_HI_monopole:
    fg_str = 'HI_monopole'

    theta, phi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
    gc = Galactic(l=NP.degrees(phi), b=90.0-NP.degrees(theta), unit=(units.degree, units.degree))
    radec = gc.fk5
    ra_deg_EoR = radec.ra.degree
    dec_deg_EoR = radec.dec.degree
    pixres = HP.nside2pixarea(nside)   # pixel solid angle (steradians)

    catlabel = 'HI-monopole'
    spec_type = 'func'
    spec_parms = {}
    spec_parms['name'] = NP.repeat('tanh', ra_deg_EoR.size)
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_half + NP.zeros(ra_deg_EoR.size)
    spec_parms['flux-scale'] = T_xi0 * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spec_parms['flux-offset'] = 0.5*spec_parms['flux-scale'] + NP.zeros(ra_deg_EoR.size)
    spec_parms['freq-width'] = dfreq_half + NP.zeros(ra_deg_EoR.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg_EoR.reshape(-1,1), dec_deg_EoR.reshape(-1,1))), spec_type, spec_parms=spec_parms)
    spectrum = skymod.generate_spectrum()

elif use_GSM:
    fg_str = 'asm'

    dsm_file = args['DSM_file_prefix']+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = freq/1e9 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ra_deg = ra_deg_DSM + 0.0
    dec_deg = dec_deg_DSM + 0.0
    majax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_DSM.size)
    minax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_DSM.size)
    # majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    # minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    fluxes = fluxes_DSM + 0.0

    freq_SUMSS = 0.843 # in GHz
    SUMSS_file = args['SUMSS_file']
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
    freq_catalog = NP.concatenate((freq_catalog, freq_SUMSS*1e9 + NP.zeros(fint.size)))
    catlabel = NP.concatenate((catlabel, NP.repeat('SUMSS', fint.size)))
    ra_deg = NP.concatenate((ra_deg, ra_deg_SUMSS))
    dec_deg = NP.concatenate((dec_deg, dec_deg_SUMSS))
    spindex = NP.concatenate((spindex, spindex_SUMSS))
    majax = NP.concatenate((majax, fmajax/3.6e3))
    minax = NP.concatenate((minax, fminax/3.6e3))
    fluxes = NP.concatenate((fluxes, fint))

    nvss_file = args['NVSS_file']
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

    # ctlgobj = SM.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    # ctlgobj = SM.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex)

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_DSM:
    fg_str = 'dsm'

    dsm_file = args['DSM_file_prefix']+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0 * FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = freq/1e9 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ra_deg = ra_deg_DSM
    dec_deg = dec_deg_DSM
    majax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_DSM.size)
    minax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_DSM.size)
    # majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    # minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    fluxes = fluxes_DSM
    # ctlgobj = SM.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    hdulist.close()

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_USM:
    fg_str = 'usm'

    dsm_file = args['DSM_file_prefix']+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    avg_temperature = NP.mean(temperatures)
    fluxes_USM = avg_temperature * (2.0 * FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy * NP.ones(temperatures.size)
    spindex = NP.zeros(fluxes_USM.size)
    freq_USM = 0.185 # in GHz
    freq_catalog = freq_USM * 1e9 + NP.zeros(fluxes_USM.size)
    catlabel = NP.repeat('USM', fluxes_USM.size)
    majax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_USM.size)
    minax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_USM.size)
    # ctlgobj = SM.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes_USM, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes_USM.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    hdulist.close()  

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
  
elif use_CSM:
    fg_str = 'csm'
    freq_SUMSS = 0.843 # in GHz
    SUMSS_file = args['SUMSS_file']
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
    fluxes = fint + 0.0
    nvss_file = args['NVSS_file']
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

    # ctlgobj = SM.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_SUMSS:
    SUMSS_file = args['SUMSS_file']
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
    if spindex_seed is None:
        spindex = -0.83 + spindex_rms * NP.random.randn(fint.size)
    else:
        NP.random.seed(spindex_seed)
        spindex = -0.83 + spindex_rms * NP.random.randn(fint.size)

    # ctlgobj = SM.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcsec','arcsec','degree'])    
    fg_str = 'sumss'

    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = 1.0e-3 + NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_MSS:
    pass
elif use_GLEAM:
    catalog_file = args['GLEAM_file']
    catdata = ascii.read(catalog_file, data_start=1, delimiter=',')
    dec_deg = catdata['DEJ2000']
    ra_deg = catdata['RAJ2000']
    fpeak = catdata['S150_fit']
    ferr = catdata['e_S150_fit']
    spindex = catdata['Sp+Index']
    # ctlgobj = SM.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex)
    fg_str = 'gleam'

    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_PS:
    fg_str = 'point'
    catalog_file = args['PS_file']
    catdata = ascii.read(catalog_file, comment='#', header_start=0, data_start=1)
    ra_deg = catdata['RA'].data
    dec_deg = catdata['DEC'].data
    fint = catdata['F_INT'].data
    spindex = catdata['SPINDEX'].data
    majax = catdata['MAJAX'].data
    minax = catdata['MINAX'].data
    pa = catdata['PA'].data
    freq_PS = 0.185 # in GHz
    freq_catalog = freq_PS * 1e9 + NP.zeros(fint.size)
    catlabel = NP.repeat('PS', fint.size)
    # ctlgobj = SM.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), src_shape_units=['arcmin','arcmin','degree'])

    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

# elif use_PS:
#     n_src = 1
#     fpeak = 1000.0*NP.ones(n_src)
#     spindex = NP.ones(n_src) * spindex
#     ra_deg = NP.asarray(pointings_radec[0,0])
#     dec_deg = NP.asarray(pointings_radec[0,1])
#     fmajax = NP.ones(n_src)
#     fminax = fmajax
#     fpa = NP.zeros(n_src)
#     ctlgobj = SM.Catalog('PS', freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcmin','arcmin','degree'])
#     fg_str = 'point'

# skymod = SM.SkyModel(ctlgobj)

## Set up the observing run

if mpi_on_src: # MPI based on source multiplexing

    for i in range(len(bl_chunk)):
        print 'Working on baseline chunk # {0:0d} ...'.format(bl_chunk[i])

        ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz', pointing_coords='hadec')    

        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
        for j in range(n_snaps):
            src_altaz_current = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst[j]-skymod.location[:,0]).reshape(-1,1), skymod.location[:,1].reshape(-1,1))), latitude, units='degrees')
            roi_ind = NP.where(src_altaz_current[:,0] >= 0.0)[0]
            n_src_per_rank = NP.zeros(nproc, dtype=int) + roi_ind.size/nproc
            if roi_ind.size % nproc > 0:
                n_src_per_rank[:roi_ind.size % nproc] += 1
            cumm_src_count = NP.concatenate(([0], NP.cumsum(n_src_per_rank)))
            # timestamp = str(DT.datetime.now())
            timestamp = lst[j]
            pbinfo = None
            if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
                pbinfo = {}
                pbinfo['delays'] = delays[j,:]
                if (telescope_id == 'mwa') or (phased_array):
                    # pbinfo['element_locs'] = element_locs
                    pbinfo['delayerr'] = delayerr
                    pbinfo['gainerr'] = gainerr
                    pbinfo['nrand'] = nrand

            ts = time.time()
            if j == 0:
                ts0 = ts
            ia.observe(timestamp, Tsys*noise_bpcorr, bpass, pointings_hadec[j,:], skymod.subset(roi_ind[cumm_src_count[rank]:cumm_src_count[rank+1]].tolist()), t_snap[j], pb_info=pbinfo, brightness_units=flux_unit, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
            te = time.time()
            # print '{0:.1f} seconds for snapshot # {1:0d}'.format(te-ts, j)
            progress.update(j+1)
        progress.finish()
    
        # svf = NP.zeros_like(ia.skyvis_freq.astype(NP.complex128), dtype='complex128')
        if rank == 0:
            for k in range(1,nproc):
                print 'receiving from process {0}'.format(k)
                ia.skyvis_freq = ia.skyvis_freq + comm.recv(source=k)
                # comm.Recv([svf, svf.size, MPI.DOUBLE_COMPLEX], source=i)
                # ia.skyvis_freq = ia.skyvis_freq + svf
            te0 = time.time()
            print 'Time on process 0 was {0:.1f} seconds'.format(te0-ts0)
            ia.t_obs = t_obs
            ia.generate_noise()
            ia.add_noise()
            ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
            outfile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+fg_str+'_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_str+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
            ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
        else:
            comm.send(ia.skyvis_freq, dest=0)
            # comm.Send([ia.skyvis_freq, ia.skyvis_freq.size, MPI.DOUBLE_COMPLEX])

else: # MPI based on baseline multiplexing

    if mpi_async: # does not impose equal volume per process
        print 'Processing next baseline chunk asynchronously...'
        processed_chunks = []
        process_sequence = []
        counter = my_MPI.Counter(comm)
        count = -1
        ptb = time.time()
        ptb_str = str(DT.datetime.now())
        while (count+1 < len(bl_chunk)):
            count = counter.next()
            if count < len(bl_chunk):
                processed_chunks.append(count)
                process_sequence.append(rank)
                print 'Process {0:0d} working on baseline chunk # {1:0d} ...'.format(rank, count)

                outfile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[count]],bl_length[min(baseline_bin_indices[count]+baseline_chunk_size-1,total_baselines-1)])+fg_str+'_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_str+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(count)
                ia = RI.InterferometerArray(labels[baseline_bin_indices[count]:min(baseline_bin_indices[count]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[count]:min(baseline_bin_indices[count]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz', pointing_coords='hadec')        

                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
                for j in range(n_snaps):
                    if obs_mode in ['custom', 'dns', 'lstbin']:
                        timestamp = obs_id[j]
                    else:
                        timestamp = lst[j]

                    pbinfo = None
                    if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
                        pbinfo = {}
                        pbinfo['delays'] = delays[j,:]
                        if (telescope_id == 'mwa') or (phased_array):
                            # pbinfo['element_locs'] = element_locs
                            pbinfo['delayerr'] = delayerr
                            pbinfo['gainerr'] = gainerr
                            pbinfo['nrand'] = nrand

                    ts = time.time()
                    if j == 0:
                        ts0 = ts
                    ia.observe(timestamp, Tsys*noise_bpcorr, bpass, pointings_hadec[j,:], skymod, t_snap[j], pb_info=pbinfo, brightness_units=flux_unit, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
                    te = time.time()
                    # print '{0:.1f} seconds for snapshot # {1:0d}'.format(te-ts, j)
                    progress.update(j+1)
                progress.finish()

                te0 = time.time()
                print 'Process {0:0d} took {1:.1f} minutes to complete baseline chunk # {2:0d}'.format(rank, (te0-ts0)/60, count)
                ia.t_obs = t_obs
                ia.generate_noise()
                ia.add_noise()
                ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
                ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
        counter.free()
        pte = time.time()
        pte_str = str(DT.datetime.now())
        pt = pte - ptb
        processed_chunks = comm.allreduce(processed_chunks)
        process_sequence = comm.allreduce(process_sequence)

    else: # impose equal volume per process
        n_bl_chunk_per_rank = NP.zeros(nproc, dtype=int) + len(bl_chunk)/nproc
        if len(bl_chunk) % nproc > 0:
            n_bl_chunk_per_rank[:len(bl_chunk)%nproc] += 1
        cumm_bl_chunks = NP.concatenate(([0], NP.cumsum(n_bl_chunk_per_rank)))        

        ptb_str = str(DT.datetime.now())

        for k in range(n_sky_sectors):
            if n_sky_sectors == 1:
                sky_sector_str = '_all_sky_'
            else:
                sky_sector_str = '_sky_sector_{0:0d}_'.format(k)

            if rank == 0: # Compute ROI parameters for only one process and broadcast to all
                roi = RI.ROI_parameters()
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
                for j in range(n_snaps):
                    src_altaz_current = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst[j]-skymod.location[:,0]).reshape(-1,1), skymod.location[:,1].reshape(-1,1))), latitude, units='degrees')
                    hemisphere_current = src_altaz_current[:,0] >= 0.0
                    # hemisphere_src_altaz_current = src_altaz_current[hemisphere_current,:]
                    src_az_current = NP.copy(src_altaz_current[:,1])
                    src_az_current[src_az_current > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
                    roi_ind = NP.logical_or(NP.logical_and(src_az_current >= -0.5*180.0/n_sky_sectors + k*180.0/n_sky_sectors, src_az_current < -0.5*180.0/n_sky_sectors + (k+1)*180.0/n_sky_sectors), NP.logical_and(src_az_current >= 180.0 - 0.5*180.0/n_sky_sectors + k*180.0/n_sky_sectors, src_az_current < 180.0 - 0.5*180.0/n_sky_sectors + (k+1)*180.0/n_sky_sectors))
                    roi_subset = NP.where(NP.logical_and(hemisphere_current, roi_ind))[0].tolist()
                    src_dircos_current_subset = GEOM.altaz2dircos(src_altaz_current[roi_subset,:], units='degrees')
                    fgmod = skymod.subset(roi_subset)
   
                    pbinfo = {}
                    if (telescope_id == 'mwa') or (phased_array) or (telescope_id == 'mwa_tools'):
                        if pointing_file is not None:
                            pbinfo['delays'] = delays[j,:]
                        else:
                            pbinfo['pointing_center'] = pointings_altaz[j,:]
                            pbinfo['pointing_coords'] = 'altaz'
                            
                        if (telescope_id == 'mwa') or (phased_array):
                            # pbinfo['element_locs'] = element_locs
                            pbinfo['delayerr'] = delayerr
                            pbinfo['gainerr'] = gainerr
                            pbinfo['nrand'] = nrand
                    else:
                        pbinfo['pointing_center'] = pointings_altaz[j,:]
                        pbinfo['pointing_coords'] = 'altaz'

                    roiinfo = {}
                    roiinfo['ind'] = NP.asarray(roi_subset)
                    roiinfo['pbeam'] = None
                    roiinfo['radius'] = 90.0
                    roiinfo_center_hadec = GEOM.altaz2hadec(NP.asarray([90.0, 270.0]).reshape(1,-1), latitude, units='degrees').ravel()
                    roiinfo_center_radec = [lst[j]-roiinfo_center_hadec[0], roiinfo_center_hadec[1]]
                    roiinfo['center'] = NP.asarray(roiinfo_center_radec).reshape(1,-1)
                    roiinfo['center_coords'] = 'radec'

                    roi.append_settings(skymod, chans, pinfo=pbinfo, latitude=latitude, lst=lst[j], roi_info=roiinfo, telescope=telescope, freq_scale='GHz')
                    
                    progress.update(j+1)
                progress.finish()
            else:
                roi = None
                pbinfo = None

            roi = comm.bcast(roi, root=0) # Broadcast information in ROI instance to all processes
            pbinfo = comm.bcast(pbinfo, root=0) # Broadcast PB synthesis info
            if (rank == 0):
                roifile = '/data3/t_nithyanandan/'+project_dir+'/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)
                roi.save(roifile, tabtype='BinTableHDU', overwrite=True, verbose=True)

                if plots:
                    for j in xrange(n_snaps):
                        src_ra = roi.skymodel.location[roi.info['ind'][j],0]
                        src_dec = roi.skymodel.location[roi.info['ind'][j],1]
                        src_ra[src_ra > 180.0] = src_ra[src_ra > 180.0] - 360.0
                        fig, axs = PLT.subplots(2, sharex=True, sharey=True, figsize=(6,6))
                        modelsky = axs[0].scatter(src_ra, src_dec, c=roi.skymodel.flux_density[roi.info['ind'][j]], norm=PLTC.LogNorm(vmin=roi.skymodel.flux_density.min(), vmax=roi.skymodel.flux_density.max()), edgecolor='none', s=20)
                        axs[0].set_xlim(180.0, -180.0)
                        axs[0].set_ylim(-90.0, 90.0)
                        pbsky = axs[1].scatter(src_ra, src_dec, c=roi.info['pbeam'][j][:,NP.argmax(NP.abs(chans-freq))], norm=PLTC.LogNorm(vmin=roi.info['pbeam'][j].min(), vmax=1.0), edgecolor='none', s=20)
                        axs[1].set_xlim(180.0, -180.0)
                        axs[1].set_ylim(-90.0, 90.0)

                        cbax0 = fig.add_axes([0.88, 0.5, 0.02, 0.35])
                        cbar0 = fig.colorbar(modelsky, cax=cbax0, orientation='vertical')
                        cbax0.set_ylabel('Flux Density [Jy]', labelpad=0, fontsize=14)

                        cbax1 = fig.add_axes([0.88, 0.1, 0.02, 0.35])
                        cbar1 = fig.colorbar(pbsky, cax=cbax1, orientation='vertical')
                        
                        fig.subplots_adjust(hspace=0)
                        big_ax = fig.add_subplot(111)
                        big_ax.set_axis_bgcolor('none')
                        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                        big_ax.set_xticks([])
                        big_ax.set_yticks([])
                        big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
                        big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)

                        fig.subplots_adjust(right=0.88)

            for i in range(cumm_bl_chunks[rank], cumm_bl_chunks[rank+1]):
                print 'Process {0:0d} working on baseline chunk # {1:0d} ...'.format(rank, bl_chunk[i])
        
                outfile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_str+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
                ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz', pointing_coords='hadec')        
        
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
                for j in range(n_snaps):
                    if obs_mode in ['custom', 'dns', 'lstbin']:
                        timestamp = obs_id[j]
                    else:
                        timestamp = lst[j]
                 
                    ts = time.time()
                    if j == 0:
                        ts0 = ts
                  
                    # ia.observe(timestamp, Tsys*noise_bpcorr, bpass, pointings_hadec[j,:], fgmod, t_snap[j], pb_info=pbinfo, brightness_units=flux_unit, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
                    ia.observe(timestamp, Tsys*noise_bpcorr, bpass, pointings_hadec[j,:], skymod, t_snap[j], pb_info=pbinfo, brightness_units=flux_unit, roi_info={'ind': roi.info['ind'][j], 'pbeam': roi.info['pbeam'][j]}, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
                    te = time.time()
                    # print '{0:.1f} seconds for snapshot # {1:0d}'.format(te-ts, j)
                    progress.update(j+1)
                progress.finish()
    
                te0 = time.time()
                print 'Process {0:0d} took {1:.1f} minutes to complete baseline chunk # {2:0d}'.format(rank, (te0-ts0)/60, bl_chunk[i])
                ia.t_obs = t_obs
                ia.generate_noise()
                ia.add_noise()
                ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
                ia.project_baselines()
                ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
        pte_str = str(DT.datetime.now())                

print 'Process {0} has completed.'.format(rank)
PDB.set_trace()
