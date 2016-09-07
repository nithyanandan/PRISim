#!python

import os, pwd, errno
from mpi4py import MPI
import yaml
import argparse
import copy
import numpy as NP
import ephem as EP
from astropy.io import fits, ascii
from astropy.coordinates import Galactic, FK5, SkyCoord
from astropy import units
from astropy.time import Time
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
from astroutils import MPI_modules as my_MPI
from astroutils import geometry as GEOM
from astroutils import catalog as SM
from astroutils import constants as CNST
from astroutils import DSP_modules as DSP 
from astroutils import lookup_operations as LKP
from astroutils import mathops as OPS
import prisim
from prisim import interferometry as RI
from prisim import primary_beams as PB
from prisim import baseline_delay_horizon as DLY
import ipdb as PDB

## Set MPI parameters

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
name = MPI.Get_processor_name()

## global parameters

sday = CNST.sday
sday_correction = 1 / sday
prisim_path = prisim.__path__[0]+'/'

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')

input_group = parser.add_argument_group('Input parameters', 'Input specifications')
input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/simparms/defaultparms.yaml', type=file, required=False, help='File specifying input parameters')

args = vars(parser.parse_args())

default_parms = {}
with args['infile'] as custom_parms_file:
    custom_parms = yaml.safe_load(custom_parms_file)
if custom_parms['preload']['template'] is not None:
    with open(custom_parms['preload']['template']) as default_parms_file:
        default_parms = yaml.safe_load(default_parms_file)

if not default_parms:
    parms = custom_parms
else:
    parms = default_parms
    if custom_parms['preload']['template'] is not None:
        for key in custom_parms:
            if key != 'preload':
                if key in default_parms:
                    if not isinstance(custom_parms[key], dict):
                        parms[key] = custom_parms[key]
                    else:
                        for subkey in custom_parms[key]:
                            if subkey in default_parms[key]:
                                if not isinstance(custom_parms[key][subkey], dict):
                                    parms[key][subkey] = custom_parms[key][subkey]
                                else:
                                    for subsubkey in custom_parms[key][subkey]:
                                        if subsubkey in default_parms[key][subkey]:
                                            if not isinstance(custom_parms[key][subkey][subsubkey], dict):
                                                parms[key][subkey][subsubkey] = custom_parms[key][subkey][subsubkey]
                                            else:
                                                raise TypeError('Parsing YAML simulation parameter files with this level of nesting is not supported')
                                        else:
                                            raise KeyError('Invalid parameter found in custom simulation parameters file')
                            else:
                                raise KeyError('Invalid parameter found in custom simulation parameters file')
                else:
                    raise KeyError('Invalid parameter found in custom simulation parameters file')                            

rootdir = parms['dirstruct']['rootdir']
project = parms['dirstruct']['project']
simid = parms['dirstruct']['simid']
telescope_id = parms['telescope']['id']
label_prefix = parms['telescope']['label_prefix']
Trx = parms['telescope']['Trx']
Tant_freqref = parms['telescope']['Tant_freqref']
Tant_ref = parms['telescope']['Tant_ref']
Tant_spindex = parms['telescope']['Tant_spindex']
Tsysinfo = {'Trx': Trx, 'Tant':{'f0': Tant_freqref, 'spindex': Tant_spindex, 'T0': Tant_ref}, 'Tnet': None}
# Tsys = parms['telescope']['Tsys']
A_eff = parms['telescope']['A_eff']
latitude = parms['telescope']['latitude']
longitude = parms['telescope']['longitude']
if longitude is None:
    longitude = 0.0
pfb_method = parms['bandpass']['pfb_method']
pfb_filepath = parms['bandpass']['pfb_filepath']
pfb_file = parms['bandpass']['pfb_file']
if pfb_method is not None:
    if pfb_method not in ['theoretical', 'empirical']:
        raise ValueError('Value specified for pfb_method is not one of accepted values')
    if not isinstance(pfb_file, str):
        raise TypeError('Filename containing PFB information must be a string')
    if pfb_filepath == 'default':
        pfb_file = prisim_path + 'data/bandpass/'+pfb_file
element_shape = parms['antenna']['shape']
element_size = parms['antenna']['size']
element_ocoords = parms['antenna']['ocoords']
element_orientation = parms['antenna']['orientation']
ground_plane = parms['antenna']['ground_plane']
phased_array = parms['antenna']['phased_array']
phased_elements_file = parms['phasedarray']['file']
if phased_array:
    if not isinstance(phased_elements_file, str):
        raise TypeError('Filename containing phased array elements must be a string')
    if parms['phasedarray']['filepathtype'] == 'default':
        phased_elements_file = prisim_path+'data/phasedarray_layouts/'+phased_elements_file
delayerr = parms['phasedarray']['delayerr']
gainerr = parms['phasedarray']['gainerr']
nrand = parms['phasedarray']['nrand']
array_is_redundant = parms['array']['redundant']
if not isinstance(array_is_redundant, bool):
    raise TypeError('Parameter specifying array redundancy must be a boolean value')
antenna_file = parms['array']['file']
array_layout = parms['array']['layout']
minR = parms['array']['minR']
maxR = parms['array']['maxR']
antpos_rms_tgtplane = parms['array']['rms_tgtplane']
antpos_rms_elevation = parms['array']['rms_elevation']
antpos_rms_seed = parms['array']['seed']
if antpos_rms_seed is None:
    antpos_rms_seed = NP.random.randint(1, high=100000)
elif isinstance(antpos_rms_seed, (int,float)):
    antpos_rms_seed = int(NP.abs(antpos_rms_seed))
else:
    raise ValueError('Random number seed must be a positive integer')
minbl = parms['baseline']['min']
maxbl = parms['baseline']['max']
bldirection = parms['baseline']['direction']
obs_date = parms['obsparm']['obs_date']
obs_mode = parms['obsparm']['obs_mode']
n_acc = parms['obsparm']['n_acc']
t_acc = parms['obsparm']['t_acc']
t_obs = parms['obsparm']['t_obs']
freq = parms['bandpass']['freq']
freq_resolution = parms['bandpass']['freq_resolution']
nchan = parms['bandpass']['nchan']
timeformat = parms['obsparm']['timeformat']
beam_info = parms['beam']
use_external_beam = beam_info['use_external']
if use_external_beam:
    if not isinstance(beam_info['file'], str):
        raise TypeError('Filename containing external beam information must be a string')
    external_beam_file = beam_info['file']
    if beam_info['filepathtype'] == 'default':
        external_beam_file = prisim_path+'data/beams/'+external_beam_file
    if beam_info['filefmt'] in ['HDF5', 'hdf5', 'fits', 'FITS']:
        beam_filefmt = beam_info['filefmt']
    else:
        raise ValueError('Invalid beam file format specified')
    beam_pol = beam_info['pol']
    beam_id = beam_info['identifier']
    select_beam_freq = beam_info['select_freq']
    if select_beam_freq is None:
        select_beam_freq = freq
    pbeam_spec_interp_method = beam_info['spec_interp']
beam_chromaticity = beam_info['chromatic']
avg_drifts = parms['snapshot']['avg_drifts']
beam_switch = parms['snapshot']['beam_switch']
pick_snapshots = parms['snapshot']['pick']
all_snapshots = parms['snapshot']['all']
snapshots_range = parms['snapshot']['range']
pointing_file = parms['pointing']['file']
# pointing_info = parms['pointing']['initial']
pointing_drift_init = parms['pointing']['drift_init']
pointing_track_init = parms['pointing']['track_init']
n_bins_baseline_orientation = parms['processing']['n_bins_blo']
baseline_chunk_size = parms['processing']['bl_chunk_size']
bl_chunk = parms['processing']['bl_chunk']
n_bl_chunks = parms['processing']['n_bl_chunks']
frequency_chunk_size = parms['processing']['freq_chunk_size']
freq_chunk = parms['processing']['freq_chunk']
n_freq_chunks = parms['processing']['n_freq_chunks']
n_sky_sectors = parms['processing']['n_sky_sectors']
bpass_shape = parms['processing']['bpass_shape']
ant_bpass_file = parms['processing']['ant_bpass_file']
max_abs_delay = parms['processing']['max_abs_delay']
f_pad = parms['processing']['f_pad']
n_pad = parms['processing']['n_pad']
coarse_channel_width = parms['processing']['coarse_channel_width']
bandpass_correct = parms['processing']['bp_correct']
noise_bandpass_correct = parms['processing']['noise_bp_correct']
do_delay_transform = parms['processing']['delay_transform']
memsave = parms['processing']['memsave']
cleanup = parms['processing']['cleanup']
flag_chan = NP.asarray(parms['flags']['flag_chan']).reshape(-1)
bp_flag_repeat = parms['flags']['bp_flag_repeat']
n_edge_flag = NP.asarray(parms['flags']['n_edge_flag']).reshape(-1)
flag_repeat_edge_channels = parms['flags']['flag_repeat_edge_channels']
fg_str = parms['fgparm']['model']
fgcat_epoch = parms['fgparm']['epoch']
nside = parms['fgparm']['nside']
flux_unit = parms['fgparm']['flux_unit']
spindex = parms['fgparm']['spindex']
spindex_rms = parms['fgparm']['spindex_rms']
spindex_seed = parms['fgparm']['spindex_seed']
use_lidz = parms['fgparm']['lidz']
use_21cmfast = parms['fgparm']['21cmfast']
global_HI_parms = parms['fgparm']['global_EoR_parms']
catalog_filepathtype = parms['catalog']['filepathtype']
DSM_file_prefix = parms['catalog']['DSM_file_prefix']
SUMSS_file = parms['catalog']['SUMSS_file']
NVSS_file = parms['catalog']['NVSS_file']
MWACS_file = parms['catalog']['MWACS_file']
GLEAM_file = parms['catalog']['GLEAM_file']
custom_catalog_file = parms['catalog']['custom_file']
if catalog_filepathtype == 'default':
    DSM_file_prefix = prisim_path + 'data/catalogs/' + DSM_file_prefix
    SUMSS_file = prisim_path + 'data/catalogs/' + SUMSS_file
    NVSS_file = prisim_path + 'data/catalogs/' + NVSS_file
    MWACS_file = prisim_path + 'data/catalogs/' + MWACS_file
    GLEAM_file = prisim_path + 'data/catalogs/' + GLEAM_file
    custom_catalog_file = prisim_path + 'data/catalogs/' + custom_catalog_file
pc = parms['phasing']['center']
pc_coords = parms['phasing']['coords']
mpi_key = parms['pp']['key']
mpi_eqvol = parms['pp']['eqvol']
save_formats = parms['save_formats']
save_to_npz = save_formats['npz']
save_to_uvfits = save_formats['uvfits']
savefmt = save_formats['fmt']
if savefmt not in ['HDF5', 'hdf5', 'FITS', 'fits']:
    raise ValueError('Output format invalid')
if save_to_uvfits:
    if save_formats['uvfits_method'] not in [None, 'uvdata', 'uvfits']:
        raise ValueError('Invalid method specified for saving to UVFITS format')
plots = parms['plots']
diagnosis_parms = parms['diagnosis']

project_dir = project + '/'
try:
    os.makedirs(rootdir+project_dir, 0755)
except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(rootdir+project_dir):
        pass
    else:
        raise

if rank == 0:
    if simid is None:
        simid = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
    elif not isinstance(simid, str):
        raise TypeError('simid must be a string')
else:
    simid = None

simid = comm.bcast(simid, root=0) # Broadcast simulation ID

simid = simid + '/'
try:
    os.makedirs(rootdir+project_dir+simid, 0755)
except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(rootdir+project_dir+simid):
        pass
    else:
        raise

if telescope_id not in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'custom', 'paper_dipole', 'mwa_tools']:
    raise ValueError('Invalid telescope specified')

if element_shape is None:
    element_shape = 'delta'
elif element_shape not in ['dish', 'delta', 'dipole']:
    raise ValueError('Invalid antenna element shape specified')

if element_shape != 'delta':
    if element_size is None:
        raise ValueError('No antenna element size specified')
    elif element_size <= 0.0:
        raise ValueError('Antenna element size must be positive')

if not isinstance(phased_array, bool):
    raise TypeError('phased_array specification must be boolean')

if delayerr is None:
    delayerr_str = ''
    delayerr = 0.0
elif delayerr < 0.0:
    raise ValueError('delayerr must be non-negative.')
else:
    delayerr_str = 'derr_{0:.3f}ns'.format(delayerr)
delayerr *= 1e-9

if gainerr is None:
    gainerr_str = ''
    gainerr = 0.0
elif gainerr < 0.0:
    raise ValueError('gainerr must be non-negative.')
else:
    gainerr_str = '_gerr_{0:.2f}dB'.format(gainerr)

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

if element_ocoords not in ['altaz', 'dircos']:
    if element_ocoords is not None:
        raise ValueError('Antenna element orientation must be "altaz" or "dircos"')

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

if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

if use_external_beam:
    external_beam = fits.getdata(external_beam_file, extname='BEAM_{0}'.format(beam_pol))
    external_beam_freqs = fits.getdata(external_beam_file, extname='FREQS_{0}'.format(beam_pol))
    external_beam = external_beam.reshape(-1,external_beam_freqs.size)
    beam_usage_str = 'extpb_'+beam_id
    if beam_chromaticity:
        if pbeam_spec_interp_method == 'fft':
            external_beam = external_beam[:,:-1]
            external_beam_freqs = external_beam_freqs[:-1]
        beam_usage_str = beam_usage_str + '_chromatic'
    else:
        beam_usage_str = beam_usage_str + '_{0:.1f}_MHz'.format(select_beam_freq/1e6)+'_achromatic'
else:
    beam_usage_str = 'funcpb'
    beam_usage_str = beam_usage_str + '_chromatic'

if (antenna_file is None) and (array_layout is None):
    raise ValueError('One of antenna array file or layout must be specified')
if (antenna_file is not None) and (array_layout is not None):
    raise ValueError('Only one of antenna array file or layout must be specified')

if antenna_file is not None:
    if not isinstance(antenna_file, str):
        raise TypeError('Filename containing antenna array elements must be a string')
    if parms['array']['filepathtype'] == 'default':
        antenna_file = prisim_path+'data/array_layouts/'+antenna_file
    
    antfile_parser = parms['array']['parser']
    if 'comment' in antfile_parser:
        comment = antfile_parser['comment']
        if comment is None:
            comment = '#'
        elif not isinstance(comment, str):
            raise TypeError('Comment expression must be a string')
    else:
        comment = '#'
    if 'delimiter' in antfile_parser:
        delimiter = antfile_parser['delimiter']
        if delimiter is not None:
            if not isinstance(delimiter, str):
                raise TypeError('Delimiter expression must be a string')
        else:
            delimiter = ' '
    else:
        delimiter = ' '

    if 'data_start' in antfile_parser:
        data_start = antfile_parser['data_start']
        if not isinstance(data_start, int):
            raise TypeError('data_start parameter must be an integer')
    else:
        raise KeyError('data_start parameter not provided')
    if 'data_end' in antfile_parser:
        data_end = antfile_parser['data_end']
        if data_end is not None:
            if not isinstance(data_end, int):
                raise TypeError('data_end parameter must be an integer')
    else:
        data_end = None
    if 'header_start' in antfile_parser:
        header_start = antfile_parser['header_start']
        if not isinstance(header_start, int):
            raise TypeError('header_start parameter must be an integer')
    else:
        raise KeyError('header_start parameter not provided')

    if 'label' not in antfile_parser:
        antfile_parser['label'] = None
    elif antfile_parser['label'] is not None:
        antfile_parser['label'] = str(antfile_parser['label'])

    if 'east' not in antfile_parser:
        raise KeyError('Keyword for "east" coordinates not provided')
    else:
        if not isinstance(antfile_parser['east'], str):
            raise TypeError('Keyword for "east" coordinates must be a string')
    if 'north' not in antfile_parser:
        raise KeyError('Keyword for "north" coordinates not provided')
    else:
        if not isinstance(antfile_parser['north'], str):
            raise TypeError('Keyword for "north" coordinates must be a string')
    if 'up' not in antfile_parser:
        raise KeyError('Keyword for "up" coordinates not provided')
    else:
        if not isinstance(antfile_parser['up'], str):
            raise TypeError('Keyword for "up" coordinates must be a string')

    try:
        ant_info = ascii.read(antenna_file, comment=comment, delimiter=delimiter, header_start=header_start, data_start=data_start, data_end=data_end, guess=False)
    except IOError:
        raise IOError('Could not open file containing antenna locations.')

    if (antfile_parser['east'] not in ant_info.colnames) or (antfile_parser['north'] not in ant_info.colnames) or (antfile_parser['up'] not in ant_info.colnames):
        raise KeyError('One of east, north, up coordinates incompatible with the table in antenna_file')

    if antfile_parser['label'] is not None:
        ant_label = ant_info[antfile_parser['label']].data.astype('str')
    else:
        ant_label = NP.arange(len(ant_info)).astype('str')

    east = ant_info[antfile_parser['east']].data
    north = ant_info[antfile_parser['north']].data
    elev = ant_info[antfile_parser['up']].data

    if (east.dtype != NP.float) or (north.dtype != NP.float) or (elev.dtype != NP.float):
        raise TypeError('Antenna locations must be of floating point type')

    ant_locs = NP.hstack((east.reshape(-1,1), north.reshape(-1,1), elev.reshape(-1,1)))
else:
    if array_layout not in ['MWA-128T', 'HERA-7', 'HERA-19', 'HERA-37', 'HERA-61', 'HERA-91', 'HERA-127', 'HERA-169', 'HERA-217', 'HERA-271', 'HERA-331', 'CIRC']:
        raise ValueError('Invalid array layout specified')

    if array_layout == 'MWA-128T':
        ant_info = NP.loadtxt(prisim_path+'data/array_layouts/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
        ant_label = ant_info[:,0].astype(int).astype(str)
        ant_locs = ant_info[:,1:]
    elif array_layout == 'HERA-7':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=7)
    elif array_layout == 'HERA-19':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=19)
    elif array_layout == 'HERA-37':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=37)
    elif array_layout == 'HERA-61':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=61)
    elif array_layout == 'HERA-91':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=91)
    elif array_layout == 'HERA-127':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=127)
    elif array_layout == 'HERA-169':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=169)
    elif array_layout == 'HERA-217':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=217)
    elif array_layout == 'HERA-271':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=271)
    elif array_layout == 'HERA-331':
        ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=331)
    elif array_layout == 'CIRC':
        ant_locs, ant_label = RI.circular_antenna_array(element_size, minR, maxR=maxR)
    ant_label = NP.asarray(ant_label)
if ant_locs.shape[1] == 2:
    ant_locs = NP.hstack((ant_locs, NP.zeros(ant_label.size).reshape(-1,1)))
if rank == 0:
    antpos_rstate = NP.random.RandomState(antpos_rms_seed)
    deast = antpos_rms_tgtplane/NP.sqrt(2.0) * antpos_rstate.randn(ant_label.size)
    dnorth = antpos_rms_tgtplane/NP.sqrt(2.0) * antpos_rstate.randn(ant_label.size)
    dup = antpos_rms_elevation * antpos_rstate.randn(ant_label.size)
    denu = NP.hstack((deast.reshape(-1,1), dnorth.reshape(-1,1), dup.reshape(-1,1)))
else:
    denu = None
denu = comm.bcast(denu, root=0) # Broadcast antenna position perturbations
ant_locs = ant_locs + denu
ant_locs_orig = NP.copy(ant_locs)
ant_label_orig = NP.copy(ant_label)
ant_id = NP.arange(ant_label.size, dtype=int)
ant_id_orig = NP.copy(ant_id)
layout_info = {'positions': ant_locs_orig, 'labels': ant_label_orig, 'ids': ant_id_orig, 'coords': 'ENU'}

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane
telescope['latitude'] = latitude
telescope['longitude'] = longitude

if A_eff is None:
    if (telescope['shape'] == 'dipole') or (telescope['shape'] == 'delta'):
        A_eff = (0.5*FCNST.c/freq)**2
        if (telescope_id == 'mwa') or phased_array:
            A_eff *= 16
    if telescope['shape'] == 'dish':
        A_eff = NP.pi * (0.5*element_size)**2

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

if avg_drifts + beam_switch + (pick_snapshots is not None) + (snapshots_range is not None) + all_snapshots != 1:
    raise ValueError('One and only one of avg_drifts, beam_switch, pick_snapshots, snapshots_range, all_snapshots must be set')

snapshot_type_str = ''
if avg_drifts and (obs_mode == 'dns'):
    snapshot_type_str = 'drift_averaged_'

if beam_switch and (obs_mode == 'dns'):
    snapshot_type_str = 'beam_switches_'

if (snapshots_range is not None) and ((obs_mode == 'dns') or (obs_mode == 'lstbin')):
    snapshot_type_str = 'snaps_{0[0]:0d}-{0[1]:0d}_'.format(snapshots_range)

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
    if n_acc is None:
        n_acc = pointing_info_from_file.shape[0]
    pointing_info_from_file = pointing_info_from_file[:min(n_acc, pointing_info_from_file.shape[0]),:]
    obs_id = obs_id[:min(n_acc, pointing_info_from_file.shape[0])]
    if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
        delays = delay_settings[:min(n_acc, pointing_info_from_file.shape[0]),:]
    n_acc = min(n_acc, pointing_info_from_file.shape[0])
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
        n_acc = lst_wrapped.size - 1
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
        snapshots_range[1] = snapshots_range[1] % n_acc
        if snapshots_range[0] > snapshots_range[1]:
            raise IndexError('min snaphost # must be <= max snapshot #')
        lst_wrapped = lst_wrapped[snapshots_range[0]:snapshots_range[1]+2]
        lst_edges = NP.copy(lst_wrapped)
        pointings_altaz = pointings_altaz[snapshots_range[0]:snapshots_range[1]+1,:]
        obs_id = obs_id[snapshots_range[0]:snapshots_range[1]+1]
        if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
            delays = delay_settings[snapshots_range[0]:snapshots_range[1]+1,:]
        n_acc = snapshots_range[1]-snapshots_range[0]+1
    elif pick_snapshots is not None:
        pick_snapshots = NP.asarray(pick_snapshots)
        n_acc = pick_snapshots.size
        lst_begin = NP.asarray(lst_wrapped[pick_snapshots])
        pointings_altaz = pointings_altaz[pick_snapshots,:]
        obs_id = obs_id[pick_snapshots]
        if (telescope_id == 'mwa') or (phased_array) or (telescope_id == 'mwa_tools'):
            delays = delay_settings[pick_snapshots,:]

        if obs_mode != 'lstbin':
            lst_end = NP.asarray(lst_wrapped[pick_snapshots+1])
            # t_acc = (lst_end - lst_begin) / 15.0 * 3.6e3
            t_acc = (lst_end - lst_begin) / 15.0 * 3.6e3 * sday
            lst = 0.5 * (lst_begin + lst_end)
            obs_mode = 'custom'
        else:
            t_acc = 112.0 + NP.zeros(n_acc)   # in seconds (needs to be generalized)
            # lst = lst_wrapped[pick_snapshots] + 0.5 * t_acc/3.6e3 * 15.0
            lst = lst_wrapped[pick_snapshots] + 0.5 * t_acc/3.6e3 * 15.0 / sday
            
    if pick_snapshots is None:
        if obs_mode != 'lstbin':        
            if not beam_switch:
                lst = 0.5*(lst_edges[1:]+lst_edges[:-1])
                # t_acc = (lst_edges[1:]-lst_edges[:-1]) / 15.0 * 3.6e3
                t_acc = (lst_edges[1:]-lst_edges[:-1]) / 15.0 * 3.6e3 * sday
            else:
                lst = 0.5*(lst_edges_left + lst_edges_right)
                # t_acc = (lst_edges_right - lst_edges_left) / 15.0 * 3.6e3
                t_acc = (lst_edges_right - lst_edges_left) / 15.0 * 3.6e3 * sday
        else:
            t_acc = 112.0 + NP.zeros(n_acc)   # in seconds (needs to be generalized)
            # lst = lst_wrapped + 0.5 * t_acc/3.6e3 * 15.0
            lst = lst_wrapped + 0.5 * t_acc/3.6e3 * 15.0 / sday

    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')
    pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
    pointings_radec[:,0] = pointings_radec[:,0] % 360.0
    t_obs = NP.sum(t_acc)
# elif pointing_info is not None:
elif (pointing_drift_init is not None) or (pointing_track_init is not None):
    pointing_file = None
    timestamps = []
    timestamps_JD = []
    if t_acc is None:
        raise NameError('t_acc must be provided for an automated observing run')

    if (n_acc is None) and (t_obs is None):
        raise NameError('n_acc or t_obs must be provided for an automated observing run')
    elif (n_acc is not None) and (t_obs is not None):
        raise ValueError('Only one of n_acc or t_obs must be provided for an automated observing run')
    elif n_acc is None:
        n_acc = int(t_obs/t_acc)
    else:
        t_obs = n_acc * t_acc

    if obs_mode is None:
        obs_mode = 'track'
    elif obs_mode not in ['track', 'drift']:
        raise ValueError('Invalid specification for obs_mode')

    lstobj = EP.FixedBody()
    if obs_mode == 'drift':
        alt = pointing_drift_init['alt']
        az = pointing_drift_init['az']
        ha = pointing_drift_init['ha']
        dec = pointing_drift_init['dec']
        lst_init = pointing_drift_init['lst']
        lst_init *= 15.0 # initial LST is now in degrees
        lstobj._epoch = obs_date
       
        if (alt is None) or (az is None):
            if (ha is None) or (dec is None):
                raise ValueError('One of alt-az or ha-dec pairs must be specified')
            hadec_init = NP.asarray([ha, dec])
        else:
            altaz_init = NP.asarray([alt, az])
            hadec_init = GEOM.altaz2hadec(altaz_init.reshape(1,-1), latitude, units='degrees')
        pointings_hadec = NP.repeat(hadec_init.reshape(1,-1), n_acc, axis=0)
            
    if obs_mode == 'track':
        ra = pointing_track_init['ra']
        dec = pointing_track_init['dec']
        ha = pointing_track_init['ha']
        epoch = pointing_track_init['epoch']
        lst_init = ra + ha
        lstobj._epoch = epoch
        pointings_hadec = NP.hstack((ha + (t_acc/3.6e3)*15.0*NP.arange(n_acc).reshape(-1,1), dec+NP.zeros(n_acc).reshape(-1,1)))
        
    lstobj._ra = NP.radians(lst_init)

    obsrvr = EP.Observer()
    obsrvr.lat = NP.radians(latitude)
    obsrvr.lon = NP.radians(longitude)
    obsrvr.date = fgcat_epoch

    lstobj.compute(obsrvr)
    lst_init_fgcat_epoch = NP.degrees(lstobj.ra) / 15.0 # LST (hours) in epoch of foreground catalog
    # lst = (lst_init_fgcat_epoch + (t_acc/3.6e3) * NP.arange(n_acc)) * 15.0 # in degrees at the epoch of the foreground catalog
    lst = (lst_init_fgcat_epoch + (t_acc/3.6e3) * NP.arange(n_acc)) * 15.0 / sday # in degrees at the epoch of the foreground catalog    
    t_acc = t_acc + NP.zeros(n_acc)
    pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
    
    pntgobj = EP.FixedBody()
    pntgobj._epoch = lstobj._epoch
    obsrvr.date = obs_date # Now compute transit times for date of observation
    last_localtime = -1.0
    for j in xrange(n_acc):
        pntgobj._ra = NP.radians(lst[j])
        pntgobj.compute(obsrvr)
        localtime = obsrvr.next_transit(pntgobj)
        # localtime = pntgobj.transit_time
        if len(timestamps) > 0:
            if localtime < last_localtime:
                obsrvr.date = obsrvr.date + EP.hour * 24
                pntgobj.compute(obsrvr)
                localtime = obsrvr.next_transit(pntgobj)
                # localtime = pntgobj.transit_time
        last_localtime = copy.deepcopy(localtime)
        obsrvr.date = last_localtime
        timestamps_JD += [EP.julian_date(localtime)]
        if timeformat == 'JD':
            # timestamps += ['{0:.9f}'.format(EP.julian_date(localtime))]
            timestamps += [EP.julian_date(localtime)]
        else:
            timestamps += ['{0}'.format(localtime.datetime())]

    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    if lst_wrapped.size > 1:
        lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))
    else:
        # lst_edges = NP.concatenate((lst_wrapped, lst_wrapped+t_acc/3.6e3*15))
        lst_edges = NP.concatenate((lst_wrapped, lst_wrapped+t_acc/3.6e3*15/sday))

    duration_str = '_{0:0d}x{1:.1f}s'.format(n_acc, t_acc[0])

pointings_radec = NP.fmod(pointings_radec, 360.0)
pointings_hadec = NP.fmod(pointings_hadec, 360.0)
pointings_altaz = NP.fmod(pointings_altaz, 360.0)
lst = NP.fmod(lst, 360.0)

use_GSM = False
use_DSM = False
use_CSM = False
use_SUMSS = False
use_GLEAM = False
use_USM = False
use_MSS = False
use_custom = False
use_NVSS = False
use_HI_monopole = False
use_HI_cube = False
use_HI_fluctuations = False
use_MSS=False

if fg_str not in ['asm', 'dsm', 'csm', 'nvss', 'sumss', 'gleam', 'mwacs', 'custom', 'usm', 'mss', 'HI_cube', 'HI_monopole', 'HI_fluctuations']:
    raise ValueError('Invalid foreground model string specified.')

if fg_str == 'asm':
    use_GSM = True
elif fg_str == 'dsm':
    use_DSM = True
elif fg_str == 'csm':
    use_CSM = True
elif fg_str == 'sumss':
    use_SUMSS = True
elif fg_str == 'gleam':
    use_GLEAM = True
elif fg_str == 'custom':
    use_custom = True
elif fg_str == 'nvss':
    use_NVSS = True
elif fg_str == 'usm':
    use_USM = True
elif fg_str == 'HI_monopole':
    use_HI_monopole = True
elif fg_str == 'HI_fluctuations':
    use_HI_fluctuations = True
elif fg_str == 'HI_cube':
    use_HI_cube = True

if global_HI_parms is not None:
    try:
        global_HI_parms = NP.asarray(map(float, global_HI_parms))
    except ValueError:
        raise ValueError('Values in global_EoR_parms must be convertible to float')
    T_xi0 = NP.float(global_HI_parms[0])
    freq_half = global_HI_parms[1]
    dz_half = global_HI_parms[2]

bl_orig, bl_label_orig, bl_id_orig = RI.baseline_generator(ant_locs_orig, ant_label=ant_label_orig, ant_id=ant_id_orig, auto=False, conjugate=False)
bl = NP.copy(bl_orig)
bl_label = NP.copy(bl_label_orig)
bl_id = NP.copy(bl_id_orig)
if array_is_redundant:
    bl, select_bl_ind, bl_count = RI.uniq_baselines(bl)
    bl_label = bl_label[select_bl_ind]
    bl_id = bl_id[select_bl_ind]
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_label = bl_label[sortind]
bl_id = bl_id[sortind]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
if array_is_redundant:
    bl_count = bl_count[sortind]
neg_bl_orientation_ind = (bl_orientation < -67.5) | (bl_orientation > 112.5)
# neg_bl_orientation_ind = NP.logical_or(bl_orientation < -0.5*180.0/n_bins_baseline_orientation, bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
maxlen = max(max(len(albl[0]), len(albl[1])) for albl in bl_label)
bl_label = [tuple(reversed(bl_label[i])) if neg_bl_orientation_ind[i] else bl_label[i] for i in xrange(bl_label.size)]
bl_label = NP.asarray(bl_label, dtype=[('A2', '|S{0:0d}'.format(maxlen)), ('A1', '|S{0:0d}'.format(maxlen))])
bl_id = [tuple(reversed(bl_id[i])) if neg_bl_orientation_ind[i] else bl_id[i] for i in xrange(bl_id.size)]
bl_id = NP.asarray(bl_id, dtype=[('A2', int), ('A1', int)])

if minbl is None:
    minbl = 0.0
elif not isinstance(minbl, (int,float)):
    raise TypeError('Minimum baseline length must be a scalar')
elif minbl < 0.0:
    minbl = 0.0

if maxbl is None:
    maxbl = bl_length.max()
elif not isinstance(maxbl, (int,float)):
    raise TypeError('Maximum baseline length must be a scalar')
elif maxbl < minbl:
    maxbl = bl_length.max()

min_blo = -67.5
max_blo = 112.5
select_bl_ind = NP.zeros(bl_length.size, dtype=NP.bool)

if bldirection is not None:
    if isinstance(bldirection, str):
        if bldirection not in ['SE', 'E', 'NE', 'N']:
            raise ValueError('Invalid baseline direction criterion specified')
        else:
            bldirection = [bldirection]
    if isinstance(bldirection, list):
        for direction in bldirection:
            if direction in ['SE', 'E', 'NE', 'N']:
                if direction == 'SE':
                    oind = (bl_orientation >= -67.5) & (bl_orientation < -22.5)
                    select_bl_ind[oind] = True
                elif direction == 'E':
                    oind = (bl_orientation >= -22.5) & (bl_orientation < 22.5)
                    select_bl_ind[oind] = True
                elif direction == 'NE':
                    oind = (bl_orientation >= 22.5) & (bl_orientation < 67.5)
                    select_bl_ind[oind] = True
                else:
                    oind = (bl_orientation >= 67.5) & (bl_orientation < 112.5)
                    select_bl_ind[oind] = True
    else:
        raise TypeError('Baseline direction criterion must specified as string or list of strings')
else:
    select_bl_ind = NP.ones(bl_length.size, dtype=NP.bool)

select_bl_ind = select_bl_ind & (bl_length >= minbl) & (bl_length <= maxbl)
bl_label = bl_label[select_bl_ind]
bl_id = bl_id[select_bl_ind]
bl = bl[select_bl_ind,:]
bl_length = bl_length[select_bl_ind]
bl_orientation = bl_orientation[select_bl_ind]

if use_HI_monopole:
    bllstr = map(str, bl_length)
    uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
    count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
    count_uniq_bll = NP.asarray(count_uniq_bll)

    bl = bl[ind_uniq_bll,:]
    bl_label = bl_label[ind_uniq_bll]
    bl_id = bl_id[ind_uniq_bll]
    bl_orientation = bl_orientation[ind_uniq_bll]
    bl_length = bl_length[ind_uniq_bll]

    sortind = NP.argsort(bl_length, kind='mergesort')
    bl = bl[sortind,:]
    bl_label = bl_label[sortind]
    bl_id = bl_id[sortind]
    bl_length = bl_length[sortind]
    bl_orientation = bl_orientation[sortind]
    count_uniq_bll = count_uniq_bll[sortind]

total_baselines = bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

try:
    labels = bl_label.tolist()
except NameError:
    labels = []
    labels += [label_prefix+'{0:0d}'.format(i+1) for i in xrange(bl.shape[0])]

try:
    ids = bl_id.tolist()
except NameError:
    ids = range(bl.shape[0])
    
if bl_chunk is None:
    bl_chunk = range(len(baseline_bin_indices))
if n_bl_chunks is None:
    n_bl_chunks = len(bl_chunk)
bl_chunk = bl_chunk[:n_bl_chunks]

if not isinstance(mpi_key, str):
    raise TypeError('MPI key must be a string')
if mpi_key not in ['src', 'bl', 'freq']:
    raise ValueError('MPI key must be set on "bl" or "src"')
if mpi_key == 'src':
    mpi_on_src = True
    mpi_ob_bl = False
    mpi_on_freq = False
elif mpi_key == 'bl':
    mpi_on_src = False
    mpi_on_bl = True
    mpi_on_freq = False
else:
    mpi_on_freq = True
    mpi_on_src = False
    mpi_on_bl = False

if not isinstance(mpi_eqvol, bool):
    raise TypeError('MPI equal volume parameter must be boolean')
if mpi_eqvol:
    mpi_sync = True
    mpi_async = False
else:
    mpi_sync = False
    mpi_async = True
    
freq = NP.float(freq)
freq_resolution = NP.float(freq_resolution)
base_bpass = 1.0*NP.ones(nchan)
bandpass_shape = 1.0*NP.ones(nchan)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution)/ 1e9 # in GHz
oversampling_factor = 1.0 + f_pad
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)

frequency_bin_indices = range(0,nchan,frequency_chunk_size)
if freq_chunk is None:
    freq_chunk = range(len(frequency_bin_indices))
if n_freq_chunks is None:
    n_freq_chunks = len(freq_chunk)
freq_chunk = freq_chunk[:n_freq_chunks]

flagged_edge_channels = []
pfb_str = ''
pfb_str2 = ''
if pfb_method is not None:
    if pfb_method == 'empirical':
        bandpass_shape = DSP.PFB_empirical(nchan, 32, 0.25, 0.25)
    elif pfb_method == 'theoretical':
        pfbhdulist = fits.open(pfb_file)
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
            pfb_edge_channels = (bandpass_shape.argmin() + NP.arange(nchan/coarse_channel_width)*coarse_channel_width) % nchan
            # pfb_edge_channels = bandpass_shape.argsort()[:int(1.0*nchan/coarse_channel_width)]
            # wts = NP.exp(-0.5*((NP.arange(bandpass_shape.size)-0.5*bandpass_shape.size)/4.0)**2)/(4.0*NP.sqrt(2*NP.pi))
            # wts_shift = NP.fft.fftshift(wts)
            # freq_wts = NP.fft.fft(wts_shift)
            # pfb_filtered = DSP.fft_filter(bandpass_shape.ravel(), wts=freq_wts.ravel(), passband='high')
            # pfb_edge_channels = pfb_filtered.argsort()[:int(1.0*nchan/coarse_channel_width)]

            pfb_edge_channels = NP.hstack((pfb_edge_channels.ravel(), NP.asarray([pfb_edge_channels.min()-coarse_channel_width, pfb_edge_channels.max()+coarse_channel_width])))
            flagged_edge_channels += [range(max(0,pfb_edge-n_edge_flag[0]),min(nchan,pfb_edge+n_edge_flag[1])) for pfb_edge in pfb_edge_channels]
else:
    pfb_str = 'no_pfb_'
    pfb_str2 = '_no_pfb'

if ant_bpass_file is not None:
    with NP.load(ant_bpass_file) as ant_bpass_fileobj:
        ant_bpass_freq = ant_bpass_fileobj['faxis']
        ant_bpass_ref = ant_bpass_fileobj['band']
        ant_bpass_ref /= NP.abs(ant_bpass_ref).max()
        ant_bpass_freq = ant_bpass_freq[ant_bpass_freq.size/2:]
        ant_bpass_ref = ant_bpass_ref[ant_bpass_ref.size/2:]        
        # ant_bpass_window = DSP.windowing(ant_bpass_freq.size, shape='bhw', pad_width=0, pad_value=0.0, peak=1.0, centering=True)
        # ant_bpass_freq = ant_bpass_freq[1:-1]
        # ant_bpass_ref = ant_bpass_ref[1:-1] / ant_bpass_window[1:-1]
        chanind, ant_bpass, fdist = LKP.lookup_1NN_new(ant_bpass_freq.reshape(-1,1)/1e9, ant_bpass_ref.reshape(-1,1), chans.reshape(-1,1), distance_ULIM=freq_resolution/1e9, remove_oob=True)
else:
    ant_bpass = NP.ones(nchan)

window = nchan * DSP.windowing(nchan, shape=bpass_shape, pad_width=n_pad, centering=True, area_normalize=True) 
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
    flagged_edge_channels += [range(nchan-n_edge_flag[1],nchan)]

flagged_channels = flagged_edge_channels
if flag_chan[0] >= 0:
    flag_chan = flag_chan[flag_chan < nchan]
    if bp_flag_repeat:
        flag_chan = NP.mod(flag_chan, coarse_channel_width)
        flagged_channels += [[i*coarse_channel_width+flagchan for i in range(nchan/coarse_channel_width) for flagchan in flag_chan]]
    else:
        flagged_channels += [flag_chan.tolist()]
flagged_channels = [x for y in flagged_channels for x in y]
flagged_channels = list(set(flagged_channels))

bandpass_shape[flagged_channels] = 0.0
bpass = base_bpass * bandpass_shape

if not isinstance(n_sky_sectors, int):
    raise TypeError('n_sky_sectors must be an integer')
elif (n_sky_sectors < 1):
    n_sky_sectors = 1

if use_HI_cube:
    if not isinstance(use_lidz, bool):
        raise TypeError('Parameter specifying use of Lidz simulations must be Boolean')
    if not isinstance(use_21cmfast, bool):
        raise TypeError('Parameter specifying use of 21cmfast simulations must be Boolean')
    
if use_HI_monopole or use_HI_fluctuations or use_HI_cube:
    if use_lidz and use_21cmfast:
        raise ValueError('Only one of Adam Lidz or 21CMFAST simulations can be chosen')
    if not use_lidz and not use_21cmfast:
        use_lidz = True
        use_21cmfast = False
        eor_simfile = rootdir+'EoR_simulations/Adam_Lidz/Boom_tiles/hpxcube_138.915-195.235_MHz_80.0_kHz_nside_{0:0d}.fits'.format(nside)
    elif use_lidz:
        eor_simfile = rootdir+'EoR_simulations/Adam_Lidz/Boom_tiles/hpxcube_138.915-195.235_MHz_80.0_kHz_nside_{0:0d}.fits'.format(nside)
    elif use_21cmfast:
        pass

spindex_rms_str = ''
spindex_seed_str = ''
if not isinstance(spindex_rms, (int,float)):
    raise TypeError('Spectral Index rms must be a scalar')
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    if not isinstance(spindex_seed, (int, float)):
        raise TypeError('Spectral index random seed must be a scalar')
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

if use_HI_fluctuations or use_HI_cube:
    # if freq_resolution != 80e3:
    #     raise ValueError('Currently frequency resolution can only be set to 80 kHz')

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

    flux_unit = 'Jy'
    catlabel = 'HI-cube'
    spec_type = 'spectrum'
    spec_parms = {}
    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg_EoR.reshape(-1,1), dec_deg_EoR.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'spectrum': fluxes_EoR}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg_EoR.reshape(-1,1), dec_deg_EoR.reshape(-1,1))), spec_type, spectrum=fluxes_EoR, spec_parms=None)
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)
elif use_HI_monopole:

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
    spec_parms['z-width'] = dz_half + NP.zeros(ra_deg_EoR.size)
    flux_unit = 'Jy'

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg_EoR.reshape(-1,1), dec_deg_EoR.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg_EoR.reshape(-1,1), dec_deg_EoR.reshape(-1,1))), spec_type, spec_parms=spec_parms)
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)
    spectrum = skymod.generate_spectrum()

elif use_GSM:
    dsm_file = DSM_file_prefix+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
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

    freq_NVSS = 1.4 # in GHz
    hdulist = fits.open(NVSS_file)
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
    flux_unit = 'Jy'

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

elif use_DSM:
    dsm_file = DSM_file_prefix+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0 * FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    flux_unit = 'Jy'
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

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

elif use_USM:
    dsm_file = DSM_file_prefix+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
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
    hdulist.close()  

    flux_unit = 'Jy'
    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes_USM
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes_USM.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)
  
elif use_CSM:
    freq_SUMSS = 0.843 # in GHz
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
    bright_source_ind = fint >= 10.0 * (freq_SUMSS*1e9/freq)**spindex_SUMSS#XXX flux cut
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
    freq_NVSS = 1.4 # in GHz
    hdulist = fits.open(NVSS_file)
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

    not_in_SUMSS_ind = dec_deg_NVSS > -30.0
    # not_in_SUMSS_ind = NP.logical_and(dec_deg_NVSS > -30.0, dec_deg_NVSS <= min(90.0, latitude+90.0))
    
    bright_source_ind = nvss_fpeak >= 10.0 * (freq_NVSS*1e9/freq)**(spindex_NVSS) #XXX flux cut
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

    spec_type = 'func'
    spec_parms = {}
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)
    flux_unit = 'Jy'

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

elif use_SUMSS:
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

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = 1.0e-3 + NP.zeros(ra_deg.size)
    flux_unit = 'Jy'

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

elif use_MSS:
    pass
elif use_GLEAM:
    freq_GLEAM = 0.200  # in GHz
    hdulist = fits.open(GLEAM_file)
    ra_deg_GLEAM = hdulist[1].data['RAJ2000']
    dec_deg_GLEAM = hdulist[1].data['DECJ2000']
    gleam_fint = hdulist[1].data['int_flux_deep']
    gleam_majax = hdulist[1].data['a_deep']
    gleam_minax = hdulist[1].data['b_deep']
    gleam_pa = hdulist[1].data['pa_deep']
    gleam_psf_majax = hdulist[1].data['psf_a_deep']
    gleam_psf_minax = hdulist[1].data['psf_b_deep']
    hdulist.close()

    if spindex_seed is None:
        spindex_GLEAM = spindex + spindex_rms * NP.random.randn(gleam_fint.size)
    else:
        NP.random.seed(2*spindex_seed)
        spindex_GLEAM = spindex + spindex_rms * NP.random.randn(gleam_fint.size)

    bright_source_ind = gleam_fint >= 10.0 * (freq_GLEAM*1e9/freq)**spindex_GLEAM
    PS_ind = gleam_majax * gleam_minax <= 1.1 * gleam_psf_majax * gleam_psf_minax
    valid_ind = NP.logical_and(bright_source_ind, PS_ind)
    ra_deg_GLEAM = ra_deg_GLEAM[valid_ind]
    dec_deg_GLEAM = dec_deg_GLEAM[valid_ind]
    gleam_fint = gleam_fint[valid_ind]
    spindex_GLEAM = spindex_GLEAM[valid_ind]
    gleam_majax = gleam_majax[valid_ind]
    gleam_minax = gleam_minax[valid_ind]
    gleam_pa = gleam_pa[valid_ind]
    fluxes = gleam_fint + 0.0
    catlabel = NP.repeat('GLEAM', gleam_fint.size)
    ra_deg = ra_deg_GLEAM + 0.0
    dec_deg = dec_deg_GLEAM + 0.0
    freq_catalog = freq_GLEAM*1e9 + NP.zeros(gleam_fint.size)
    majax = gleam_majax / 3.6e3
    minax = gleam_minax / 3.6e3
    spindex = spindex_GLEAM + 0.0

    spec_type = 'func'
    spec_parms = {}
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)
    flux_unit = 'Jy'

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

elif use_custom:
    catdata = ascii.read(custom_catalog_file, comment='#', header_start=0, data_start=1)
    ra_deg = catdata['RA'].data
    dec_deg = catdata['DEC'].data
    fint = catdata['F_INT'].data
    spindex = catdata['SPINDEX'].data
    majax = catdata['MAJAX'].data
    minax = catdata['MINAX'].data
    pa = catdata['PA'].data
    freq_custom = 0.185 # in GHz
    freq_catalog = freq_custom * 1e9 + NP.zeros(fint.size)
    catlabel = NP.repeat('custom', fint.size)

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fint
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)
    flux_unit = 'Jy'

    skymod_init_parms = {'name': catlabel, 'frequency': chans*1e9, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': spec_type, 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}

    # skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

# Create organized directory structure

timestamps_JD = NP.asarray(timestamps_JD)
init_timestamps_JD = timestamps_JD.min()
init_time = Time(init_timestamps_JD, format='jd', scale='utc')
obsdatetime_dir = '{0}{1}{2}_{3}{4}{5}/'.format(init_time.datetime.year, init_time.datetime.month, init_time.datetime.day, init_time.datetime.hour, init_time.datetime.minute, init_time.datetime.second)

sim_dir = 'simdata/'
meta_dir = 'metainfo/'
roi_dir = 'roi/'
skymod_dir = 'skymodel/'

try:
    os.makedirs(rootdir+project_dir+simid+sim_dir, 0755)
except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(rootdir+project_dir+simid+sim_dir):
        pass
    else:
        raise

try:
    os.makedirs(rootdir+project_dir+simid+meta_dir, 0755)
except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(rootdir+project_dir+simid+meta_dir):
        pass
    else:
        raise

try:
    os.makedirs(rootdir+project_dir+simid+roi_dir, 0755)
except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(rootdir+project_dir+simid+roi_dir):
        pass
    else:
        raise
    
try:
    os.makedirs(rootdir+project_dir+simid+skymod_dir, 0755)
except OSError as exception:
    if exception.errno == errno.EEXIST and os.path.isdir(rootdir+project_dir+simid+skymod_dir):
        pass
    else:
        raise
    
## Set up the observing run

process_complete = False
if mpi_on_src: # MPI based on source multiplexing

    for i in range(len(bl_chunk)):
        print 'Working on baseline chunk # {0:0d} ...'.format(bl_chunk[i])

        ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, longitude=longitude, A_eff=A_eff, layout=layout_info, freq_scale='GHz', pointing_coords='hadec')    

        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_acc).start()
        for j in range(n_acc):
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
            ia.observe(timestamp, Tsysinfo, bpass, pointings_hadec[j,:], skymod.subset(roi_ind[cumm_src_count[rank]:cumm_src_count[rank+1]].tolist()), t_acc[j], pb_info=pbinfo, brightness_units=flux_unit, bpcorrect=noise_bpcorr, roi_radius=None, roi_center=None, lst=lst[j], memsave=memsave)
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
            outfile = rootdir+project_dir+simid+sim_dir+'_part_{0:0d}'.format(i)
            ia.save(outfile, fmt=savefmt, verbose=True, tabtype='BinTableHDU', npz=False, overwrite=True, uvfits_parms=None)
        else:
            comm.send(ia.skyvis_freq, dest=0)
            # comm.Send([ia.skyvis_freq, ia.skyvis_freq.size, MPI.DOUBLE_COMPLEX])

elif mpi_on_freq: # MPI based on frequency multiplexing

    n_freq_chunk_per_rank = NP.zeros(nproc, dtype=int) + len(freq_chunk)/nproc
    if len(freq_chunk) % nproc > 0:
        n_freq_chunk_per_rank[:len(freq_chunk)%nproc] += 1
    cumm_freq_chunks = NP.concatenate(([0], NP.cumsum(n_freq_chunk_per_rank)))

    for k in range(n_sky_sectors):
        if n_sky_sectors == 1:
            sky_sector_str = '_all_sky_'
        else:
            sky_sector_str = '_sky_sector_{0:0d}_'.format(k)

        if rank == 0: # Compute ROI parameters for only one process and broadcast to all
            roi = RI.ROI_parameters()
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots'.format(n_acc), PGB.ETA()], maxval=n_acc).start()
            for j in range(n_acc):
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
                if use_external_beam:
                    theta_phi = NP.hstack((NP.pi/2-NP.radians(src_altaz_current[roi_subset,0]).reshape(-1,1), NP.radians(src_altaz_current[roi_subset,1]).reshape(-1,1)))
                    if beam_chromaticity:
                        interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(external_beam), theta_phi=theta_phi, inloc_axis=external_beam_freqs, outloc_axis=chans*1e9, axis=1, kind=pbeam_spec_interp_method, assume_sorted=True)
                        # interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(external_beam), theta_phi=theta_phi, inloc_axis=external_beam_freqs, outloc_axis=chans*1e9, axis=1, kind=pbeam_spec_interp_method, assume_sorted=True)
                        
                    else:
                        nearest_freq_ind = NP.argmin(NP.abs(external_beam_freqs*1e6 - select_beam_freq))
                        interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(NP.repeat(external_beam[:,nearest_freq_ind].reshape(-1,1), chans.size, axis=1)), theta_phi=theta_phi, inloc_axis=chans*1e9, outloc_axis=chans*1e9, axis=1, assume_sorted=True)
                    interp_logbeam_max = NP.nanmax(interp_logbeam, axis=0)
                    interp_logbeam_max[interp_logbeam_max <= 0.0] = 0.0
                    interp_logbeam_max = interp_logbeam_max.reshape(1,-1)
                    interp_logbeam = interp_logbeam - interp_logbeam_max
                    roiinfo['pbeam'] = 10**interp_logbeam
                    # roiinfo['pbeam'] = NP.ones((roiinfo['ind'].size,chans.size), dtype=NP.float32)
                else:
                    roiinfo['pbeam'] = None
                roiinfo['radius'] = 90.0
                roiinfo_center_hadec = GEOM.altaz2hadec(NP.asarray([90.0, 270.0]).reshape(1,-1), latitude, units='degrees').ravel()
                roiinfo_center_radec = [lst[j]-roiinfo_center_hadec[0], roiinfo_center_hadec[1]]
                roiinfo['center'] = NP.asarray(roiinfo_center_radec).reshape(1,-1)
                roiinfo['center_coords'] = 'radec'

                roi.append_settings(skymod, chans, pinfo=pbinfo, lst=lst[j], roi_info=roiinfo, telescope=telescope, freq_scale='GHz')
                
                progress.update(j+1)
            progress.finish()

            roifile = rootdir+project_dir+simid+roi_dir+'roiinfo'
            roi.save(roifile, tabtype='BinTableHDU', overwrite=True, verbose=True)
            del roi   # to save memory if primary beam arrays or n_acc are large
        else:
            roi = None
            pbinfo = None
            roifile = None

        roifile = comm.bcast(roifile, root=0) # Broadcast saved RoI filename
        pbinfo = comm.bcast(pbinfo, root=0) # Broadcast PB synthesis info

        for i in range(cumm_freq_chunks[rank], cumm_freq_chunks[rank+1]):
            print 'Process {0:0d} working on frequency chunk # {1:0d} ...'.format(rank, freq_chunk[i])
    
            chans_chunk_indices = NP.arange(frequency_bin_indices[freq_chunk[i]], min(frequency_bin_indices[freq_chunk[i]]+frequency_chunk_size,nchan))
            chans_chunk = NP.asarray(chans[chans_chunk_indices]).reshape(-1)
            nchan_chunk = chans_chunk.size
            # nchan_chunk = min(frequency_bin_indices[freq_chunk[i]]+frequency_chunk_size,nchan) - frequency_bin_indices[freq_chunk[i]]
            f0_chunk = chans[freq_chunk[i]*frequency_chunk_size] + NP.floor(0.5*nchan_chunk) * freq_resolution / 1e9
            bw_chunk_str = '{0:0d}x{1:.1f}_kHz'.format(nchan_chunk, freq_resolution/1e3)
            outfile = rootdir+project_dir+simid+sim_dir+'_part_{0:0d}'.format(i)
            ia = RI.InterferometerArray(labels, bl, chans_chunk, telescope=telescope, latitude=latitude, longitude=longitude, A_eff=A_eff, layout=layout_info, freq_scale='GHz', pointing_coords='hadec')
            
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots '.format(n_acc), PGB.ETA()], maxval=n_acc).start()
            for j in range(n_acc):
                roi_ind_snap = fits.getdata(roifile+'.fits', extname='IND_{0:0d}'.format(j))
                roi_pbeam_snap = fits.getdata(roifile+'.fits', extname='PB_{0:0d}'.format(j))
                roi_pbeam_snap = roi_pbeam_snap[:,chans_chunk_indices]
                if obs_mode in ['custom', 'dns', 'lstbin']:
                    timestamp = obs_id[j]
                else:
                    # timestamp = lst[j]
                    timestamp = timestamps[j]
             
                ts = time.time()
                if j == 0:
                    ts0 = ts
              
                ia.observe(timestamp, Tsysinfo, bpass[chans_chunk_indices], pointings_hadec[j,:], skymod.subset(chans_chunk_indices, axis='spectrum'), t_acc[j], pb_info=pbinfo, brightness_units=flux_unit, bpcorrect=noise_bpcorr[chans_chunk_indices], roi_info={'ind': roi_ind_snap, 'pbeam': roi_pbeam_snap}, roi_radius=None, roi_center=None, lst=lst[j], memsave=memsave)
                te = time.time()
                # print '{0:.1f} seconds for snapshot # {1:0d}'.format(te-ts, j)
                progress.update(j+1)
            progress.finish()

            te0 = time.time()
            print 'Process {0:0d} took {1:.1f} minutes to complete frequency chunk # {2:0d}'.format(rank, (te0-ts0)/60, freq_chunk[i])
            # ia.t_obs = t_obs
            ia.generate_noise()
            ia.add_noise()
            # ia.delay_transform(oversampling_factor-1.0, freq_wts=window*NP.abs(ant_bpass)**2)
            ia.project_baselines(ref_point={'location': ia.pointing_center, 'coords': ia.pointing_coords})
            ia.save(outfile, fmt=savefmt, verbose=True, tabtype='BinTableHDU', npz=False, overwrite=True, uvfits_parms=None)
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

                outfile = rootdir+project_dir+simid+sim_dir+'_part_{0:0d}'.format(count)
                ia = RI.InterferometerArray(labels[baseline_bin_indices[count]:min(baseline_bin_indices[count]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[count]:min(baseline_bin_indices[count]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, longitude=longitude, A_eff=A_eff, layout=layout_info, freq_scale='GHz', pointing_coords='hadec')        

                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_acc).start()
                for j in range(n_acc):
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
                    ia.observe(timestamp, Tsysinfo, bpass, pointings_hadec[j,:], skymod, t_acc[j], pb_info=pbinfo, brightness_units=flux_unit, bpcorrect=noise_bpcorr, roi_radius=None, roi_center=None, lst=lst[j], memsave=memsave)
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
                ia.save(outfile, fmt=savefmt, verbose=True, tabtype='BinTableHDU', npz=False, overwrite=True, uvfits_parms=None)
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
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots'.format(n_acc), PGB.ETA()], maxval=n_acc).start()
                for j in range(n_acc):
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
                    if use_external_beam:
                        theta_phi = NP.hstack((NP.pi/2-NP.radians(src_altaz_current[roi_subset,0]).reshape(-1,1), NP.radians(src_altaz_current[roi_subset,1]).reshape(-1,1)))
                        if beam_chromaticity:
                            interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(external_beam), theta_phi=theta_phi, inloc_axis=external_beam_freqs, outloc_axis=chans*1e9, axis=1, kind=pbeam_spec_interp_method, assume_sorted=True)
                        else:
                            nearest_freq_ind = NP.argmin(NP.abs(external_beam_freqs*1e6 - freq))
                            interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(NP.repeat(external_beam[:,nearest_freq_ind].reshape(-1,1), chans.size, axis=1)), theta_phi=theta_phi, inloc_axis=chans*1e9, outloc_axis=chans*1e9, axis=1, kind=pbeam_spec_interp_method, assume_sorted=True)
                        
                        interp_logbeam_max = NP.nanmax(interp_logbeam, axis=0)
                        interp_logbeam_max[interp_logbeam_max <= 0.0] = 0.0
                        interp_logbeam_max = interp_logbeam_max.reshape(1,-1)
                        interp_logbeam = interp_logbeam - interp_logbeam_max
                        roiinfo['pbeam'] = 10**interp_logbeam
                        # roiinfo['pbeam'] = NP.ones((roiinfo['ind'].size,chans.size), dtype=NP.float32)
                    else:
                        roiinfo['pbeam'] = None
                    roiinfo['radius'] = 90.0
                    roiinfo_center_hadec = GEOM.altaz2hadec(NP.asarray([90.0, 270.0]).reshape(1,-1), latitude, units='degrees').ravel()
                    roiinfo_center_radec = [lst[j]-roiinfo_center_hadec[0], roiinfo_center_hadec[1]]
                    roiinfo['center'] = NP.asarray(roiinfo_center_radec).reshape(1,-1)
                    roiinfo['center_coords'] = 'radec'

                    roi.append_settings(skymod, chans, pinfo=pbinfo, lst=lst[j], roi_info=roiinfo, telescope=telescope, freq_scale='GHz')
                    
                    progress.update(j+1)
                progress.finish()

                roifile = rootdir+project_dir+simid+roi_dir+'roiinfo'
                roi.save(roifile, tabtype='BinTableHDU', overwrite=True, verbose=True)
                del roi   # to save memory if primary beam arrays or n_acc are large
            else:
                roi = None
                pbinfo = None
                roifile = None

            roifile = comm.bcast(roifile, root=0) # Broadcast saved RoI filename
            pbinfo = comm.bcast(pbinfo, root=0) # Broadcast PB synthesis info

            # if (rank != 0):
            #     roi = RI.ROI_parameters(init_file=roifile+'.fits') # Other processes read in the RoI information
            if rank == 0:
                if plots:
                    for j in xrange(n_acc):
                        src_ra = roi.skymodel.location[roi.info['ind'][j],0]
                        src_dec = roi.skymodel.location[roi.info['ind'][j],1]
                        src_ra[src_ra > 180.0] = src_ra[src_ra > 180.0] - 360.0
                        fig, axs = PLT.subplots(2, sharex=True, sharey=True, figsize=(6,6))
                        modelsky = axs[0].scatter(src_ra, src_dec, c=roi.skymod.spec_parms['flux-scale'][roi.info['ind'][j]], norm=PLTC.LogNorm(vmin=roi.skymod.spec_parms['flux-scale'].min(), vmax=roi.skymod.spec_parms['flux-scale'].max()), edgecolor='none', s=20)
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
        
                outfile = rootdir+project_dir+simid+sim_dir+'_part_{0:0d}'.format(i)
                ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, longitude=longitude, A_eff=A_eff, layout=layout_info, freq_scale='GHz', pointing_coords='hadec')
                
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Snapshots '.format(n_acc), PGB.ETA()], maxval=n_acc).start()
                for j in range(n_acc):
                    roi_ind_snap = fits.getdata(roifile+'.fits', extname='IND_{0:0d}'.format(j))
                    roi_pbeam_snap = fits.getdata(roifile+'.fits', extname='PB_{0:0d}'.format(j))
                    if obs_mode in ['custom', 'dns', 'lstbin']:
                        timestamp = obs_id[j]
                    else:
                        # timestamp = lst[j]
                        timestamp = timestamps[j]
                 
                    ts = time.time()
                    if j == 0:
                        ts0 = ts
                  
                    ia.observe(timestamp, Tsysinfo, bpass, pointings_hadec[j,:], skymod, t_acc[j], pb_info=pbinfo, brightness_units=flux_unit, bpcorrect=noise_bpcorr, roi_info={'ind': roi_ind_snap, 'pbeam': roi_pbeam_snap}, roi_radius=None, roi_center=None, lst=lst[j], memsave=memsave)
                    te = time.time()
                    # print '{0:.1f} seconds for snapshot # {1:0d}'.format(te-ts, j)
                    progress.update(j+1)
                progress.finish()

                te0 = time.time()
                print 'Process {0:0d} took {1:.1f} minutes to complete baseline chunk # {2:0d}'.format(rank, (te0-ts0)/60, bl_chunk[i])
                ia.t_obs = t_obs
                ia.generate_noise()
                ia.add_noise()
                ia.delay_transform(oversampling_factor-1.0, freq_wts=window*NP.abs(ant_bpass)**2)
                ia.project_baselines(ref_point={'location': ia.pointing_center, 'coords': ia.pointing_coords})
                ia.save(outfile, fmt=savefmt, verbose=True, tabtype='BinTableHDU', npz=False, overwrite=True, uvfits_parms=None)
        pte_str = str(DT.datetime.now())                
 
if rank == 0:
    parmsfile = rootdir+project_dir+simid+meta_dir+'simparms.yaml'
    with open(parmsfile, 'w') as pfile:
        yaml.dump(parms, pfile, default_flow_style=False)

    minfo = {'user': pwd.getpwuid(os.getuid())[0], 'git#': prisim.__githash__, 'PRISim': prisim.__version__}
    metafile = rootdir+project_dir+simid+meta_dir+'meta.yaml'
    with open(metafile, 'w') as mfile:
        yaml.dump(minfo, mfile, default_flow_style=False)

process_complete = True
all_process_complete = comm.gather(process_complete, root=0)
if rank == 0:
    for k in range(n_sky_sectors):
        if n_sky_sectors == 1:
            sky_sector_str = '_all_sky_'
        else:
            sky_sector_str = '_sky_sector_{0:0d}_'.format(k)
    
        if mpi_on_bl:
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baseline chunks '.format(n_bl_chunks), PGB.ETA()], maxval=n_bl_chunks).start()
            for i in range(0, n_bl_chunks):
                blchunk_infile = rootdir+project_dir+simid+sim_dir+'_part_{0:0d}'.format(i)
                if i == 0:
                    simvis = RI.InterferometerArray(None, None, None, init_file=blchunk_infile)
                else:
                    simvis_next = RI.InterferometerArray(None, None, None, init_file=blchunk_infile)
                    simvis.concatenate(simvis_next, axis=0)
    
                if cleanup:
                    if os.path.isfile(blchunk_infile+'.'+savefmt.lower()):
                        os.remove(blchunk_infile+'.'+savefmt.lower())
                    
                progress.update(i+1)
            progress.finish()

        elif mpi_on_freq:
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Frequency chunks '.format(n_freq_chunks), PGB.ETA()], maxval=n_freq_chunks).start()
            for i in range(0, n_freq_chunks):
                chans_chunk_indices = NP.arange(frequency_bin_indices[freq_chunk[i]], min(frequency_bin_indices[freq_chunk[i]]+frequency_chunk_size,nchan))
                chans_chunk = NP.asarray(chans[chans_chunk_indices]).reshape(-1)
                nchan_chunk = chans_chunk.size
                f0_chunk = chans[freq_chunk[i]*frequency_chunk_size] + NP.floor(0.5*nchan_chunk) * freq_resolution / 1e9
                bw_chunk_str = '{0:0d}x{1:.1f}_kHz'.format(nchan_chunk, freq_resolution/1e3)
                freqchunk_infile = rootdir+project_dir+simid+sim_dir+'_part_{0:0d}'.format(i)
                if i == 0:
                    simvis = RI.InterferometerArray(None, None, None, init_file=freqchunk_infile)    
                else:
                    simvis_next = RI.InterferometerArray(None, None, None, init_file=freqchunk_infile)    
                    simvis.concatenate(simvis_next, axis=1)
    
                if cleanup:
                    if os.path.isfile(freqchunk_infile+'.'+savefmt.lower()):
                        os.remove(freqchunk_infile+'.'+savefmt.lower())
                    
                progress.update(i+1)
            progress.finish()

        simvis.simparms_file = metafile
        ref_point = {'coords': pc_coords, 'location': NP.asarray(pc).reshape(1,-1)}
        simvis.rotate_visibilities(ref_point, do_delay_transform=do_delay_transform, verbose=True)
        if do_delay_transform:
            simvis.delay_transform(oversampling_factor-1.0, freq_wts=window*NP.abs(ant_bpass)**2)

        consolidated_outfile = rootdir+project_dir+simid+sim_dir+'simvis'
        simvis.save(consolidated_outfile, fmt=savefmt, verbose=True, tabtype='BinTableHDU', npz=save_to_npz, overwrite=True, uvfits_parms=None)

        uvfits_parms = None
        if save_to_uvfits:
            uvfits_ref_point = {'location': NP.asarray(save_formats['phase_center']).reshape(1,-1), 'coords': 'radec'}
            uvfits_parms = {'ref_point': uvfits_ref_point, 'method': save_formats['uvfits_method']}
            simvis.write_uvfits(consolidated_outfile, uvfits_parms=uvfits_parms, overwrite=True)

    skymod_file = rootdir+project_dir+simid+skymod_dir+'skymodel'
    if fg_str not in ['HI_cube', 'HI_fluctuations', 'HI_monopole', 'usm']:
        skymod.save(skymod_file, fileformat='hdf5')
        # skymod.save(skymod_file, fileformat='ascii')
            
print 'Process {0} has completed.'.format(rank)
if diagnosis_parms['wait_after_run']:
    PDB.set_trace()
