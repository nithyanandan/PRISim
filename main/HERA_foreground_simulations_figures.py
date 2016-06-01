import yaml
import argparse
import numpy as NP
import ephem as EP
import copy
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
import matplotlib.gridspec as GS
from matplotlib.collections import PolyCollection
from astropy.io import fits
import healpy as HP
import geometry as GEOM
import interferometry as RI
import delay_spectrum as DS
import my_operations as OPS
import my_DSP_modules as DSP
import constants as CNST
import primary_beams as PB
import catalog as SM
import lookup_operations as LKP
import ipdb as PDB

## global parameters

sday = CNST.sday
sday_correction = 1 / sday

colrmap = copy.copy(CM.jet)
colrmap.set_bad(color='black', alpha=1.0)

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')

input_group = parser.add_argument_group('Input parameters', 'Input specifications')
input_group.add_argument('-i', '--infile', dest='infile', default='/home/t_nithyanandan/codes/mine/python/interferometry/main/figparameters_HERA.yaml', type=file, required=False, help='File specifying input parameters')

args = vars(parser.parse_args())

with args['infile'] as parms_file:
    parms = yaml.safe_load(parms_file)

rootdir = parms['directory']['rootdir']
figdir = parms['directory']['figdir']
project = parms['project']
telescope_id = parms['telescope']['id']
pfb_method = parms['telescope']['pfb_method']
element_shape = parms['antenna']['shape']
element_size = parms['antenna']['size']
element_ocoords = parms['antenna']['ocoords']
element_orientation = parms['antenna']['orientation']
ground_plane = parms['antenna']['ground_plane']
phased_array = parms['antenna']['phased_array']
delayerr = parms['phasedarray']['delayerr']
gainerr = parms['phasedarray']['gainerr']
nrand = parms['phasedarray']['nrand']
antenna_file = parms['array']['file']
array_layout = parms['array']['layout']
minR = parms['array']['minR']
maxR = parms['array']['maxR']
minbl = parms['baseline']['min']
maxbl = parms['baseline']['max']
pointing_file = parms['pointing']['file']
pointing_drift_init = parms['pointing']['drift_init']
pointing_track_init = parms['pointing']['track_init']
baseline_chunk_size = parms['processing']['bl_chunk_size']
n_bl_chunks = parms['processing']['n_bl_chunks']
fg_str = parms['fgparm']['model']
fgcat_epoch = parms['fgparm']['epoch']
nside = parms['fgparm']['nside']
DSM_file_prefix = parms['catalog']['DSM_file_prefix']
SUMSS_file = parms['catalog']['SUMSS_file']
NVSS_file = parms['catalog']['NVSS_file']
MWACS_file = parms['catalog']['MWACS_file']
GLEAM_file = parms['catalog']['GLEAM_file']
PS_file = parms['catalog']['PS_file']
eor_str = parms['eorparm']['model']
eor_nside = parms['eorparm']['nside']
eor_nchan = parms['eorparm']['nchan']
eor_freq_resolution = parms['eorparm']['freq_resolution']
eor_cube_freq = parms['eorparm']['cube_freq']
eor_model_freq = parms['eorparm']['model_freq']
lidz_model = parms['eorparm']['lidz_model']
model_21cmfast = parms['eorparm']['21cmfast_model']
if lidz_model:
    lidz_modelfile = parms['eorparm']['lidz_modelfile']
if model_21cmfast:
    modelfile_21cmfast = parms['eorparm']['21cmfast_modelfile']
Tsys = parms['telescope']['Tsys']
latitude = parms['telescope']['latitude']
longitude = parms['telescope']['longitude']
if longitude is None:
    longitude = 0.0
freq = parms['obsparm']['freq']
freq_resolution = parms['obsparm']['freq_resolution']
obs_date = parms['obsparm']['obs_date']
obs_mode = parms['obsparm']['obs_mode']
nchan = parms['obsparm']['nchan']
n_acc = parms['obsparm']['n_acc']
t_acc = parms['obsparm']['t_acc']
t_obs = parms['obsparm']['t_obs']
timeformat = parms['obsparm']['timeformat']
achrmbeam_info = parms['achrmbeam']
achrmbeam_id = achrmbeam_info['identifier']
achrmbeam_srcfile = achrmbeam_info['srcfile']
select_achrmbeam_freq = achrmbeam_info['select_freq']
if select_achrmbeam_freq is None:
    select_achrmbeam_freq = freq
chrmbeam_info = parms['chrmbeam']
chrmbeam_id = chrmbeam_info['identifier']
chrmbeam_srcfile = chrmbeam_info['srcfile']
chrmbeam_spec_interp_method = chrmbeam_info['spec_interp']
freq_window_centers = {key: parms['subband']['freq_center'] for key in ['cc', 'sim']}
freq_window_bw = {key: parms['subband']['bw_eff'] for key in ['cc', 'sim']}
freq_window_shape={key: parms['subband']['shape'] for key in ['cc', 'sim']}
freq_window_fftpow={key: parms['subband']['fftpow'] for key in ['cc', 'sim']}
rect_freq_window_shape={key: 'rect' for key in ['cc', 'sim']}
n_sky_sectors = parms['processing']['n_sky_sectors']
bpass_shape = parms['clean']['bpass_shape']
spindex_rms = parms['fgparm']['spindex_rms']
spindex_seed = parms['fgparm']['spindex_seed']
k_reflectometry = parms['reflectometry']['k']
bll_reflectometry = parms['reflectometry']['bll']
plot_info = parms['plot']
plots = [key for key in plot_info if plot_info[key]['action']]

if project not in ['project_MWA', 'project_global_EoR', 'project_HERA', 'project_drift_scan', 'project_beams', 'project_LSTbin']:
    raise ValueError('Invalid project specified')
else:
    project_dir = project + '/'
figuresdir = rootdir + project_dir + figdir

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

use_GSM = False
use_DSM = False
use_CSM = False
use_SUMSS = False
use_GLEAM = False
use_USM = False
use_MSS = False
use_PS = False
use_NVSS = False
use_HI_monopole = False
use_HI_cube = False
use_HI_fluctuations = False

if fg_str not in ['asm', 'dsm', 'csm', 'nvss', 'sumss', 'gleam', 'mwacs', 'ps', 'point', 'usm', 'mss', 'HI_cube', 'HI_monopole', 'HI_fluctuations']:
    raise ValueError('Invalid foreground model string specified.')

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
telescope['latitude'] = latitude
telescope['longitude'] = longitude

freq = NP.float(freq)
freq_resolution = NP.float(freq_resolution)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution)/ 1e9 # in GHz
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)
eor_bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(eor_nchan, eor_freq_resolution/1e3)

if (antenna_file is None) and (array_layout is None):
    raise ValueError('One of antenna array file or layout must be specified')
if (antenna_file is not None) and (array_layout is not None):
    raise ValueError('Only one of antenna array file or layout must be specified')

if antenna_file is not None: 
    try:
        ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
        ant_id = ant_info[:,0].astype(int).astype(str)
        ant_locs = ant_info[:,1:]
    except IOError:
        raise IOError('Could not open file containing antenna locations.')
else:
    if array_layout not in ['MWA-128T', 'HERA-7', 'HERA-19', 'HERA-37', 'HERA-61', 'HERA-91', 'HERA-127', 'HERA-169', 'HERA-217', 'HERA-271', 'HERA-331', 'CIRC']:
        raise ValueError('Invalid array layout specified')

    if array_layout == 'MWA-128T':
        ant_info = NP.loadtxt(rootdir+'project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
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
    elif array_layout == 'CIRC':
        ant_locs, ant_id = RI.circular_antenna_array(element_size, minR, maxR=maxR)

bl, bl_id = RI.baseline_generator(ant_locs, ant_id=ant_id, auto=False, conjugate=False)
bl, select_bl_ind, bl_count = RI.uniq_baselines(bl)
bl_id = bl_id[select_bl_ind]
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_id = bl_id[sortind]
bl_length = bl_length[sortind]
blo = NP.angle(bl[:,0]+1j*bl[:,1], deg=True)
total_baselines = bl_length.size

baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

if use_HI_monopole:
    bllstr = map(str, bl_length)
    uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
    count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
    count_uniq_bll = NP.asarray(count_uniq_bll)

    bl = bl[ind_uniq_bll,:]
    bl_id = bl_id[ind_uniq_bll]
    bl_length = bl_length[ind_uniq_bll]

    sortind = NP.argsort(bl_length, kind='mergesort')
    bl = bl[sortind,:]
    bl_id = bl_id[sortind]
    bl_length = bl_length[sortind]
    count_uniq_bll = count_uniq_bll[sortind]

total_baselines = bl_length.size
baseline_bin_indices = range(0, int(NP.ceil(1.0*total_baselines/baseline_chunk_size)+1)*baseline_chunk_size, baseline_chunk_size)
if n_bl_chunks is None:
    n_bl_chunks = int(NP.ceil(1.0*total_baselines/baseline_chunk_size))

bl_chunk = range(len(baseline_bin_indices)-1)
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:min(baseline_bin_indices[n_bl_chunks], total_baselines),:]
bl_length = bl_length[:min(baseline_bin_indices[n_bl_chunks], total_baselines)]
bl_id = bl_id[:min(baseline_bin_indices[n_bl_chunks], total_baselines)]

pfb_instr = ''
pfb_outstr = ''
if pfb_method is None:
    pfb_instr = '_no_pfb'
    pfb_outstr = 'no_pfb_'

external_achrmbeam = fits.getdata(achrmbeam_srcfile, extname='BEAM_X')
external_achrmbeam_freqs = fits.getdata(achrmbeam_srcfile, extname='FREQS_X')
achrmbeam_nside = fits.getval(achrmbeam_srcfile, 'NSIDE', extname='BEAM_X')

external_chrmbeam = fits.getdata(chrmbeam_srcfile, extname='BEAM_X')
external_chrmbeam_freqs = fits.getdata(chrmbeam_srcfile, extname='FREQS_X')
chrmbeam_nside = fits.getval(chrmbeam_srcfile, 'NSIDE', extname='BEAM_X')

achromatic_extbeam_str = 'extpb_'+achrmbeam_id+'_{0:.1f}_MHz_achromatic'.format(select_achrmbeam_freq/1e6)
chromatic_extbeam_str = 'extpb_'+chrmbeam_id+'_chromatic'
funcbeam_str = 'funcpb_chromatic'

spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

snapshot_type_str = ''

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
elif fg_str in ['point', 'PS']:
    use_PS = True
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
else:
    fg_str = 'other'

if use_GSM:
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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes_USM.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
  
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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_MSS:
    pass
elif use_GLEAM:
    catdata = ascii.read(GLEAM_file, data_start=1, delimiter=',')
    dec_deg = catdata['DEJ2000']
    ra_deg = catdata['RAJ2000']
    fpeak = catdata['S150_fit']
    ferr = catdata['e_S150_fit']
    spindex = catdata['Sp+Index']

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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

elif use_PS:
    catdata = ascii.read(PS_file, comment='#', header_start=0, data_start=1)
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

    skymod = SM.SkyModel(catlabel, chans*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

if n_sky_sectors == 1:
    sky_sector_str = '_all_sky_'
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(n_sky_sectors)

fgvisfile_achrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+achromatic_extbeam_str+pfb_instr
fgvisfile_chrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+chromatic_extbeam_str+pfb_instr
fgvisfile_funcbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+funcbeam_str+pfb_instr

fgdsfile_achrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+achromatic_extbeam_str+'_'+pfb_outstr+bpass_shape+'{0:.1f}'.format(freq_window_fftpow['sim'])
fgdsfile_chrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+chromatic_extbeam_str+'_'+pfb_outstr+bpass_shape+'{0:.1f}'.format(freq_window_fftpow['sim'])
fgdsfile_funcbeam = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+funcbeam_str+'_'+pfb_outstr+bpass_shape+'{0:.1f}'.format(freq_window_fftpow['sim'])

fgvis_achrmbeam = RI.InterferometerArray(None, None, None, init_file=fgvisfile_achrmbeam+'.fits')
fgvis_chrmbeam = RI.InterferometerArray(None, None, None, init_file=fgvisfile_chrmbeam+'.fits')
fgvis_funcbeam = RI.InterferometerArray(None, None, None, init_file=fgvisfile_funcbeam+'.fits')
# fgds_achrmbeam = DS.DelaySpectrum(interferometer_array=fgvis_achrmbeam)
# fgds_chrmbeam = DS.DelaySpectrum(interferometer_array=fgvis_chrmbeam)
# fgds_funcbeam = DS.DelaySpectrum(interferometer_array=fgvis_funcbeam)
fgds_achrmbeam = DS.DelaySpectrum(init_file=fgdsfile_achrmbeam+'.ds.fits')
fgds_chrmbeam = DS.DelaySpectrum(init_file=fgdsfile_chrmbeam+'.ds.fits')
fgds_funcbeam = DS.DelaySpectrum(init_file=fgdsfile_funcbeam+'.ds.fits')

# ############

# fgds_achrmbeam_sbds = fgds_achrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=rect_freq_window_shape, pad=None, bpcorrect=False, action='return_resampled')
# fgds_chrmbeam_sbds = fgds_chrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=rect_freq_window_shape, pad=None, bpcorrect=False, action='return_resampled')
# fgds_funcbeam_sbds = fgds_funcbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=rect_freq_window_shape, pad=None, bpcorrect=False, action='return_resampled')

# fgdps_achrmbeam = DS.DelayPowerSpectrum(fgds_achrmbeam)
# fgdps_achrmbeam.compute_power_spectrum()
# fgdps_chrmbeam = DS.DelayPowerSpectrum(fgds_chrmbeam)
# fgdps_chrmbeam.compute_power_spectrum()
# fgdps_funcbeam = DS.DelayPowerSpectrum(fgds_funcbeam)
# fgdps_funcbeam.compute_power_spectrum()

# PDB.set_trace()

# ############

fgds_achrmbeam_sbds = fgds_achrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow, pad=None, bpcorrect=False, action='return_resampled')
fgds_chrmbeam_sbds = fgds_chrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow, pad=None, bpcorrect=False, action='return_resampled')
fgds_funcbeam_sbds = fgds_funcbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow, pad=None, bpcorrect=False, action='return_resampled')

# frac_width = DSP.window_N2width(n_window=None, shape=bpass_shape, area_normalize=False, power_normalize=True)
# n_window = NP.round(chans.size / frac_width).astype(NP.int)
# window = NP.sqrt(n_window * frac_width) * DSP.windowing(nchan, shape=bpass_shape, pad_width=0, centering=True, area_normalize=False, power_normalize=True)
# fgds_achrmbeam_DT = fgds_achrmbeam.delay_transform(pad=1.0, freq_wts=window, downsample=False, action='store')
# fgds_chrmbeam_DT = fgds_chrmbeam.delay_transform(pad=1.0, freq_wts=window, downsample=False, action='store')
# fgds_funcbeam_DT = fgds_funcbeam.delay_transform(pad=1.0, freq_wts=window, downsample=False, action='store')

fgdps_achrmbeam = DS.DelayPowerSpectrum(fgds_achrmbeam)
fgdps_achrmbeam.compute_power_spectrum()
fgdps_chrmbeam = DS.DelayPowerSpectrum(fgds_chrmbeam)
fgdps_chrmbeam.compute_power_spectrum()
fgdps_funcbeam = DS.DelayPowerSpectrum(fgds_funcbeam)
fgdps_funcbeam.compute_power_spectrum()

fgds_achrmbeam1 = DS.DelaySpectrum(interferometer_array=fgvis_achrmbeam)
# fgds_achrmbeam1 = DS.DelaySpectrum(init_file=fgdsfile_achrmbeam+'.ds.fits')
fgds_achrmbeam_sbds1 = fgds_achrmbeam1.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow={'cc': 1.0, 'sim': 1.0}, pad=None, bpcorrect=False, action='return_resampled')
fgdps_achrmbeam1 = DS.DelayPowerSpectrum(fgds_achrmbeam1)
fgdps_achrmbeam1.compute_power_spectrum()

fgds_chrmbeam1 = DS.DelaySpectrum(interferometer_array=fgvis_chrmbeam)
# fgds_chrmbeam1 = DS.DelaySpectrum(init_file=fgdsfile_chrmbeam+'.ds.fits')
fgds_chrmbeam_sbds1 = fgds_chrmbeam1.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow={'cc': 1.0, 'sim': 1.0}, pad=None, bpcorrect=False, action='return_resampled')
fgdps_chrmbeam1 = DS.DelayPowerSpectrum(fgds_chrmbeam1)
fgdps_chrmbeam1.compute_power_spectrum()

fgds_funcbeam1 = DS.DelaySpectrum(interferometer_array=fgvis_funcbeam)
# fgds_funcbeam1 = DS.DelaySpectrum(init_file=fgdsfile_funcbeam+'.ds.fits')
fgds_funcbeam_sbds1 = fgds_funcbeam1.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow={'cc': 1.0, 'sim': 1.0}, pad=None, bpcorrect=False, action='return_resampled')
fgdps_funcbeam1 = DS.DelayPowerSpectrum(fgds_funcbeam1)
fgdps_funcbeam1.compute_power_spectrum()

# fgds_achrmbeam1_DT = fgds_achrmbeam1.delay_transform(pad=1.0, freq_wts=window, downsample=False, action='store')
# fgds_chrmbeam1_DT = fgds_chrmbeam1.delay_transform(pad=1.0, freq_wts=window, downsample=False, action='store')
# fgds_funcbeam1_DT = fgds_funcbeam1.delay_transform(pad=1.0, freq_wts=window, downsample=False, action='store')

fgdps_sbIC = {beamstr: NP.load(rootdir+project_dir+'pspecs_{0}_sbinfo.npz_50.npz'.format(beamstr)) for beamstr in ['achrmbeam', 'chrmbeam', 'funcbeam']}

# eorvisfile_achrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+eor_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(eor_nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, eor_bandpass_str, eor_cube_freq/1e6)+achromatic_extbeam_str+pfb_instr
# eorvisfile_chrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+eor_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(eor_nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, eor_bandpass_str, eor_cube_freq/1e6)+chromatic_extbeam_str+pfb_instr
# eorvisfile_funcbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+eor_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(eor_nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, eor_bandpass_str, eor_cube_freq/1e6)+funcbeam_str+pfb_instr

# eorvis_achrmbeam = RI.InterferometerArray(None, None, None, init_file=eorvisfile_achrmbeam+'.fits')
# eorvis_chrmbeam = RI.InterferometerArray(None, None, None, init_file=eorvisfile_chrmbeam+'.fits')
# eorvis_funcbeam = RI.InterferometerArray(None, None, None, init_file=eorvisfile_funcbeam+'.fits')
# eords_achrmbeam = DS.DelaySpectrum(interferometer_array=eorvis_achrmbeam)
# eords_chrmbeam = DS.DelaySpectrum(interferometer_array=eorvis_chrmbeam)
# eords_funcbeam = DS.DelaySpectrum(interferometer_array=eorvis_funcbeam)

# eords_achrmbeam_sbds = eords_achrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, pad=None, bpcorrect=False, action='return')
# eords_chrmbeam_sbds = eords_chrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, pad=None, bpcorrect=False, action='return')
# eords_funcbeam_sbds = eords_funcbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, pad=None, bpcorrect=False, action='return')
# eordps_achrmbeam = DS.DelayPowerSpectrum(eords_achrmbeam)
# eordps_achrmbeam.compute_power_spectrum()
# eordps_chrmbeam = DS.DelayPowerSpectrum(eords_chrmbeam)
# eordps_chrmbeam.compute_power_spectrum()
# eordps_funcbeam = DS.DelayPowerSpectrum(eords_funcbeam)
# eordps_funcbeam.compute_power_spectrum()

if lidz_model:
    hdulist = fits.open(lidz_modelfile)
    eor_freqinds = [NP.argmin(NP.abs(hdulist['FREQUENCY'].data - subband_freq)) for subband_freq in freq_window_centers['sim']]
    lidz_eor_model_redshifts = [hdulist['REDSHIFT'].data[eor_freqind] for eor_freqind in eor_freqinds]
    eor_model_freqs = [hdulist['FREQUENCY'].data[eor_freqind] for eor_freqind in eor_freqinds]
    eor_modelinfo = [hdulist['{0}'.format(eor_freqind)].data for eor_freqind in eor_freqinds]
    lidz_eor_model_k = eor_modelinfo[0][:,0]
    lidz_eor_model_Pk = [modelinfo[:,1] for modelinfo in eor_modelinfo]
    lidz_eor_model_Pk = NP.asarray(lidz_eor_model_Pk)
    lidz_Pk_interp_func = interpolate.interp1d(NP.log10(lidz_eor_model_k), NP.log10(lidz_eor_model_Pk), axis=1, kind='cubic', bounds_error=False)

if model_21cmfast:
    hdulist = fits.open(modelfile_21cmfast)
    eor_freqinds = [NP.argmin(NP.abs(hdulist['FREQUENCY'].data - subband_freq)) for subband_freq in freq_window_centers['sim']]
    eor_21cmfast_model_redshifts = [hdulist['REDSHIFT'].data[eor_freqind] for eor_freqind in eor_freqinds]
    eor_model_freqs = [hdulist['FREQUENCY'].data[eor_freqind] for eor_freqind in eor_freqinds]
    eor_modelinfo = [hdulist['{0}'.format(eor_freqind)].data for eor_freqind in eor_freqinds]
    eor_21cmfast_model_k = eor_modelinfo[0][:,0]
    eor_21cmfast_model_Pk = [modelinfo[:,1] for modelinfo in eor_modelinfo]
    eor_21cmfast_model_Pk = NP.asarray(eor_21cmfast_model_Pk)
    Pk_21cmfast_interp_func = interpolate.interp1d(NP.log10(eor_21cmfast_model_k), NP.log10(eor_21cmfast_model_Pk), axis=1, kind='cubic', bounds_error=False)

kprll = fgdps_achrmbeam.k_parallel(fgds_achrmbeam.cc_lags, fgdps_achrmbeam.z, action='return')
kperp = fgdps_achrmbeam.k_perp(fgdps_achrmbeam.bl_length, fgdps_achrmbeam.z, action='return')
k = NP.sqrt(kprll.reshape(-1,1)**2 + kperp.reshape(1,-1)**2)

if lidz_model:
    lidz_eor_Pk_interp = lidz_Pk_interp_func(NP.log10(k.flatten()))
    lidz_eor_Pk_interp = 10**lidz_eor_Pk_interp.reshape(lidz_eor_model_Pk.shape[0], -1,kperp.size)

if model_21cmfast:
    eor_21cmfast_Pk_interp = Pk_21cmfast_interp_func(NP.log10(k.flatten()))
    eor_21cmfast_Pk_interp = 10**eor_21cmfast_Pk_interp.reshape(eor_21cmfast_model_Pk.shape[0], -1, kperp.size)

##########################################

if '1a' in plots:
        
    # 01-a) Plot beam chromaticity with single point source at different locations

    alt = NP.asarray([90.0, 45.0, 1.0])
    az = 270.0 + NP.zeros(alt.size)

    altaz = NP.hstack((alt.reshape(-1,1), az.reshape(-1,1)))
    thetaphi = NP.radians(NP.hstack((90.0-alt.reshape(-1,1), az.reshape(-1,1))))

    chrm_extbeam = 10 ** OPS.healpix_interp_along_axis(NP.log10(external_chrmbeam), theta_phi=thetaphi, inloc_axis=external_chrmbeam_freqs, outloc_axis=chans*1e3, axis=1, kind=chrmbeam_spec_interp_method, assume_sorted=True)
    nearest_freq_ind = NP.argmin(NP.abs(external_achrmbeam_freqs*1e6 - select_achrmbeam_freq))
    achrm_extbeam = 10 ** OPS.healpix_interp_along_axis(NP.log10(NP.repeat(external_achrmbeam[:,nearest_freq_ind].reshape(-1,1), chans.size, axis=1)), theta_phi=thetaphi, inloc_axis=chans*1e3, outloc_axis=chans*1e3, axis=1, assume_sorted=True)
    funcbeam = PB.primary_beam_generator(altaz, chans, telescope, freq_scale='GHz', skyunits='altaz', east2ax1=0.0, pointing_info=None, pointing_center=None)

    pad = 0.0
    npad = int(pad * chans.size)
    lags = DSP.spectral_axis(npad+chans.size, delx=chans[1]-chans[0], shift=True)
    wndw = chans.size * DSP.windowing(chans.size, shape=bpass_shape, pad_width=0, peak=None, area_normalize=True, power_normalize=False)
    wndw = wndw.reshape(1,-1)
    chrm_extbeam_FFT = NP.fft.fft(NP.pad(chrm_extbeam * wndw, ((0,0),(0,npad)), mode='constant'), axis=1) / (npad+chans.size)
    chrm_extbeam_FFT = NP.fft.fftshift(chrm_extbeam_FFT, axes=1)
    chrm_extbeam_FFT_max = NP.max(NP.abs(chrm_extbeam_FFT), axis=1, keepdims=True)
    achrm_extbeam_FFT = NP.fft.fft(NP.pad(achrm_extbeam * wndw, ((0,0),(0,npad)), mode='constant'), axis=1) / (npad+chans.size)
    achrm_extbeam_FFT = NP.fft.fftshift(achrm_extbeam_FFT, axes=1)
    achrm_extbeam_FFT_max = NP.max(NP.abs(achrm_extbeam_FFT), axis=1, keepdims=True)
    funcbeam_FFT = NP.fft.fft(NP.pad(funcbeam * wndw, ((0,0),(0,npad)), mode='constant'), axis=1) / (npad+chans.size)
    funcbeam_FFT = NP.fft.fftshift(funcbeam_FFT, axes=1)
    funcbeam_FFT_max = NP.max(NP.abs(funcbeam_FFT), axis=1, keepdims=True)

    fig, axs = PLT.subplots(ncols=3, sharex=True, sharey=True, figsize=(7,3.5))
    for alti, elev in enumerate(alt):
        # axs[alti].plot(lags, NP.abs(achrm_extbeam_FFT[alti,:])**2/achrm_extbeam_FFT_max[alti]**2, lw=2, ls='--', color='k')
        axs[alti].plot(lags, NP.abs(chrm_extbeam_FFT[alti,:])**2, lw=2, ls='--', color='k', label='Sim.')
        axs[alti].plot(lags, NP.abs(funcbeam_FFT[alti,:])**2, lw=2, ls='-', color='k', label='Dish')
        # axs[alti].plot(lags, NP.abs(chrm_extbeam_FFT[alti,:])**2/chrm_extbeam_FFT_max[alti]**2, lw=2, ls=':', color='k', label='Sim.')
        # axs[alti].plot(lags, NP.abs(funcbeam_FFT[alti,:])**2/funcbeam_FFT_max[alti]**2, lw=2, ls='-.', color='k', label='Dish')
        axs[alti].set_yscale('log')
        axs[alti].axvline(x=-1e9*bl_length[0]/FCNST.c, ymax=0.67, color='gray', ls='-', lw=3)
        axs[alti].axvline(x=1e9*bl_length[0]/FCNST.c, ymax=0.67, color='gray', ls='-', lw=3) 
        axs[alti].text(0.22, 0.95, '{0:.1f}'.format(90.0-elev)+r'$^\circ$'+'\noff-axis', transform=axs[alti].transAxes, fontsize=12, weight='medium', ha='center', va='top', color='black')
        axs[alti].legend(frameon=True, fontsize=10)
        axs[alti].set_xlim(-250, 250)
        axs[alti].set_ylim(1e-12, 1e-3)
        axs[alti].set_aspect('auto')
    fig.subplots_adjust(hspace=0, wspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel(r'$|\widetilde{V}(\eta)|^2$ [arbitrary units]', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel(r'$\tau$ [ns]', fontsize=16, weight='medium', labelpad=20)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.17)
    fig.subplots_adjust(left=0.12)
    fig.subplots_adjust(right=0.98)    

    PLT.savefig(figuresdir+'off-axis_point_source_beam_chromaticity.png', bbox_inches=0)
    PLT.savefig(figuresdir+'off-axis_point_source_beam_chromaticity.eps', bbox_inches=0)    

    print '\n\tPlotted and saved off-axis point source beam chromaticity'

if '1b' in plots:
        
    # 01-b) Plot healpix map of beam chromaticity

    theta, phi = HP.pix2ang(chrmbeam_nside, NP.arange(HP.nside2npix(chrmbeam_nside)))
    alt = 90.0 - NP.degrees(theta)
    az = NP.degrees(phi)
    altaz = NP.hstack((alt.reshape(-1,1), az.reshape(-1,1)))
    ind_hemisphere = alt >= 0.0

    chrmbeam = 10 ** OPS.healpix_interp_along_axis(NP.log10(external_chrmbeam), theta_phi=None, inloc_axis=external_chrmbeam_freqs, outloc_axis=chans*1e3, axis=1, kind=chrmbeam_spec_interp_method, assume_sorted=True)
    funcbeam = PB.primary_beam_generator(altaz, chans, telescope, freq_scale='GHz', skyunits='altaz', east2ax1=0.0, pointing_info=None, pointing_center=None)

    pad = 0.0
    npad = int(pad * chans.size)
    window = nchan * DSP.windowing(nchan, shape=bpass_shape, pad_width=0, centering=True, area_normalize=True)
    lags = DSP.spectral_axis(npad+chans.size, delx=freq_resolution, shift=True)
    window_FFT = NP.fft.fft(NP.pad(window, [(0,npad)], mode='constant')) / (npad+chans.size)
    window_FFT = freq_resolution * NP.fft.fftshift(window_FFT) / (1.0+pad) 
    chrmbeam_FFT = freq_resolution * NP.fft.fft(NP.pad(chrmbeam * window.reshape(1,-1), ((0,0),(0,npad)), mode='constant'), axis=1) / (1.0+pad)
    chrmbeam_FFT = NP.fft.fftshift(chrmbeam_FFT, axes=1)
    funcbeam_FFT = freq_resolution * NP.fft.fft(NP.pad(funcbeam * window.reshape(1,-1), ((0,0),(0,npad)), mode='constant'), axis=1) / (1.0+pad)
    funcbeam_FFT = NP.fft.fftshift(funcbeam_FFT, axes=1)

    selected_lag = 60.0
    ind_selected_lag = NP.abs(1e9*lags) > selected_lag
    window_chromaticity = NP.sum(NP.abs(window_FFT[ind_selected_lag])**2 * lags[ind_selected_lag]**2) / ind_selected_lag.size
    chrmbeam_chromaticity = NP.sum(NP.abs(chrmbeam_FFT[:,ind_selected_lag])**2 * lags[ind_selected_lag].reshape(1,-1)**2, axis=1) / ind_selected_lag.size
    funcbeam_chromaticity = NP.sum(NP.abs(funcbeam_FFT[:,ind_selected_lag])**2 * lags[ind_selected_lag].reshape(1,-1)**2, axis=1) / ind_selected_lag.size

    chrmbeam_avgval = NP.sqrt(NP.sum(NP.abs(chrmbeam_FFT[:,ind_selected_lag])**2, axis=1) / ind_selected_lag.size)
    funcbeam_avgval = NP.sqrt(NP.sum(NP.abs(funcbeam_FFT[:,ind_selected_lag])**2, axis=1) / ind_selected_lag.size)
    ratio_avgval = chrmbeam_avgval / funcbeam_avgval
    ratio_avgval[NP.isinf(ratio_avgval)] = NP.nan

    bndry = HP.boundaries(chrmbeam_nside, NP.arange(HP.nside2npix(chrmbeam_nside)))
    verts = NP.swapaxes(bndry[ind_hemisphere,:2,:], 1, 2)
    coll_funcbeam = PolyCollection(verts, array=funcbeam_chromaticity[ind_hemisphere], edgecolors='none')
    coll_chrmbeam = PolyCollection(verts, array=chrmbeam_chromaticity[ind_hemisphere], edgecolors='none')
    # coll_funcbeam.set_clim(vmin=1e-11, vmax=max(funcbeam_chromaticity.max(), chrmbeam_chromaticity.max()))
    # coll_chrmbeam.set_clim(vmin=1e-11, vmax=max(funcbeam_chromaticity.max(), chrmbeam_chromaticity.max()))        
    coll_funcbeam.set_norm(PLTC.LogNorm(vmin=1e-11, vmax=max(funcbeam_chromaticity.max(), chrmbeam_chromaticity.max())))
    coll_chrmbeam.set_norm(PLTC.LogNorm(vmin=1e-11, vmax=max(funcbeam_chromaticity.max(), chrmbeam_chromaticity.max())))

    if '1b0' in plots:
        
        skymod_spectrum = skymod.generate_spectrum(frequency=NP.asarray(freq_window_centers['sim']))
        dsm_ind, = NP.where(skymod.name == 'DSM')
        csm_ind, = NP.where(skymod.name != 'DSM')

        dsm_sortind, = NP.where(skymod_spectrum[dsm_ind,0] >= 50.0)
        csm_sortind, = NP.where(skymod_spectrum[csm_ind,0] >= 50.0)

        bright_dsm_skymod = skymod.subset(indices=dsm_ind[dsm_sortind])
        bright_csm_skymod = skymod.subset(indices=csm_ind[csm_sortind])

        # dsm_sortind = NP.argsort(skymod_spectrum[dsm_ind,0])[::-1]
        # csm_sortind = NP.argsort(skymod_spectrum[csm_ind,0])[::-1]
        
        # bright_dsm_skymod = skymod.subset(indices=dsm_ind[dsm_sortind[:10000]])
        # bright_csm_skymod = skymod.subset(indices=csm_ind[csm_sortind[:1000]])

        dsm_radec = bright_dsm_skymod.location
        csm_radec = bright_csm_skymod.location

        bright_dsm_spectrum = bright_dsm_skymod.generate_spectrum(frequency=NP.asarray(freq_window_centers['sim']))
        bright_csm_spectrum = bright_csm_skymod.generate_spectrum(frequency=NP.asarray(freq_window_centers['sim']))

        dsm_ha = lst.reshape(-1,1,1) - dsm_radec[:,0].reshape(1,-1,1)
        dsm_dec = dsm_radec[:,1].reshape(1,-1,1) + NP.zeros(lst.size).reshape(-1,1,1)
        dsm_hadec = NP.dstack((dsm_ha, dsm_dec))
        dsm_altaz = GEOM.hadec2altaz(dsm_hadec.reshape(-1,2), latitude, units='degrees')
        dsm_dircos = GEOM.altaz2dircos(dsm_altaz, units='degrees')
        dsm_dircos = dsm_dircos.reshape(-1,dsm_radec.shape[0],3)

        csm_ha = lst.reshape(-1,1,1) - csm_radec[:,0].reshape(1,-1,1)
        csm_dec = csm_radec[:,1].reshape(1,-1,1) + NP.zeros(lst.size).reshape(-1,1,1)
        csm_hadec = NP.dstack((csm_ha, csm_dec))
        csm_altaz = GEOM.hadec2altaz(csm_hadec.reshape(-1,2), latitude, units='degrees')
        csm_dircos = GEOM.altaz2dircos(csm_altaz, units='degrees')
        csm_dircos = csm_dircos.reshape(-1,csm_radec.shape[0],3)
        
        bright_dsm_src_info = {}
        bright_csm_src_info = {}        
        for li in range(lst.size):
            ind_hemisphere_csm, = NP.where(csm_dircos[li,:,2] > 0.0)
            # sortind = NP.argsort(bright_csm_spectrum[ind_hemisphere_csm,0])[::-1]
            # bright_src_ind = sortind[:10]
            # for srci in ind_hemisphere_csm[bright_src_ind]:
            for srci in ind_hemisphere_csm:
                if srci not in bright_csm_src_info:
                    bright_csm_src_info[srci] = {}
                    bright_csm_src_info[srci]['dircos'] = NP.asarray(csm_dircos[li,srci,:]).reshape(1,-1)
                    bright_csm_src_info[srci]['flux'] = [bright_csm_spectrum[srci,0]]
                    bright_csm_src_info[srci]['lst'] = [lst[li]]
                else:
                    bright_csm_src_info[srci]['dircos'] = NP.vstack((bright_csm_src_info[srci]['dircos'], NP.asarray(csm_dircos[li,srci,:]).reshape(1,-1)))
                    bright_csm_src_info[srci]['flux'] += [bright_csm_spectrum[srci,0]]
                    bright_csm_src_info[srci]['lst'] += [lst[li]]

            ind_hemisphere_dsm, = NP.where(dsm_dircos[li,:,2] > 0.0)
            sortind = NP.argsort(bright_dsm_spectrum[ind_hemisphere_dsm,0])[::-1]
            bright_src_ind = sortind[:10]
            for srci in ind_hemisphere_dsm[bright_src_ind]:
            # for srci in ind_hemisphere_dsm:
                if srci not in bright_dsm_src_info:
                    bright_dsm_src_info[srci] = {}
                    bright_dsm_src_info[srci]['dircos'] = NP.asarray(dsm_dircos[li,srci,:]).reshape(1,-1)
                    bright_dsm_src_info[srci]['flux'] = [bright_dsm_spectrum[srci,0]]
                    bright_dsm_src_info[srci]['lst'] = [lst[li]]
                else:
                    bright_dsm_src_info[srci]['dircos'] = NP.vstack((bright_dsm_src_info[srci]['dircos'], NP.asarray(dsm_dircos[li,srci,:]).reshape(1,-1)))
                    bright_dsm_src_info[srci]['flux'] += [bright_dsm_spectrum[srci,0]]
                    bright_dsm_src_info[srci]['lst'] += [lst[li]]
        spole_hadec = NP.asarray([0.0, -90.0]).reshape(1,-1)
        spole_altaz = GEOM.hadec2altaz(spole_hadec, latitude, units='degrees')
        spole_dircos = GEOM.altaz2dircos(spole_altaz, units='degrees')

    fig, axs = PLT.subplots(nrows=2, sharex=True, sharey=True, figsize=(3.5,7))
    axs[0].add_collection(coll_funcbeam)
    axs[1].add_collection(coll_chrmbeam)
    axs[0].set_xlim(-1.0, 1.0)
    axs[0].set_ylim(-1.0, 1.0)
    axs[0].set_aspect('auto')
    axs[1].set_xlim(-1.0, 1.0)
    axs[1].set_ylim(-1.0, 1.0)
    axs[1].set_aspect('auto')
    cbax = fig.add_axes([0.12, 0.95, 0.82, 0.02])
    cbar = fig.colorbar(coll_funcbeam, cax=cbax, orientation='horizontal')
    cbax.xaxis.set_label_position('top')    

    fig.subplots_adjust(wspace=0, hspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=20)

    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.18)    
    fig.subplots_adjust(top=0.88)

    coll_funcbeam = PolyCollection(verts, array=funcbeam_avgval[ind_hemisphere], edgecolors='none')
    coll_chrmbeam = PolyCollection(verts, array=chrmbeam_avgval[ind_hemisphere], edgecolors='none')
    coll_ratio = PolyCollection(verts, array=ratio_avgval[ind_hemisphere], edgecolors='none')
    coll_funcbeam.set_norm(PLTC.LogNorm(vmin=1e-1, vmax=max(funcbeam_avgval.max(), chrmbeam_avgval.max())))
    coll_chrmbeam.set_norm(PLTC.LogNorm(vmin=1e-1, vmax=max(funcbeam_avgval.max(), chrmbeam_avgval.max())))
    coll_ratio.set_norm(PLTC.LogNorm(vmin=1e-1, vmax=5e3))    

    descriptor_str = ['Airy', 'Sim.', 'Sim. / Airy']
    fig, axs = PLT.subplots(ncols=3, sharex=True, sharey=True, figsize=(8,3.5))
    axs[0].add_collection(coll_funcbeam)
    axs[1].add_collection(coll_chrmbeam)
    axs[2].add_collection(coll_ratio)
    if '1b0' in plots:
        for src in bright_csm_src_info:
            lst_ind = (NP.asarray(bright_csm_src_info[src]['lst']) >= 0.0) & (NP.asarray(bright_csm_src_info[src]['lst']) < 180.0)
            axs[2].plot(bright_csm_src_info[src]['dircos'][lst_ind,0], bright_csm_src_info[src]['dircos'][lst_ind,1], color='black', marker='.', ms=2, linestyle='None')
        # for src in bright_dsm_src_info:
        #     lst_ind = (NP.asarray(bright_dsm_src_info[src]['lst']) >= 0.0) & (NP.asarray(bright_dsm_src_info[src]['lst']) < 180.0)
        #     axs[2].plot(bright_dsm_src_info[src]['dircos'][lst_ind,0], bright_dsm_src_info[src]['dircos'][lst_ind,1], color='black', marker='+', ms=4, linestyle='None')
        axs[2].plot(spole_dircos[0,0], spole_dircos[0,1], marker='o', linestyle='None', ms=8, mfc='none', mec='black', mew=2)

    for j in range(len(axs)):
        axs[j].text(0.25, 0.92, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='medium', ha='center', color='black')
        axs[j].set_xlim(-1.0, 1.0)
        axs[j].set_ylim(-1.0, 1.0)
        axs[j].set_aspect('equal')
    cbaxt = fig.add_axes([0.1, 0.95, 0.5, 0.02])
    cbart = fig.colorbar(coll_funcbeam, cax=cbaxt, orientation='horizontal')
    cbaxt.xaxis.set_label_position('top')    
    cbaxr = fig.add_axes([0.92, 0.12, 0.02, 0.73])
    cbarr = fig.colorbar(coll_ratio, cax=cbaxr, orientation='vertical')
    cbaxr.yaxis.set_label_position('right')    

    fig.subplots_adjust(wspace=0, hspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=15)
    big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=20)

    fig.subplots_adjust(right=0.9)
    fig.subplots_adjust(left=0.09)    
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(bottom=0.13)

    PLT.savefig(figuresdir + 'directional_high_delay_average_in_beam.png', bbox_inches=0)
    PLT.savefig(figuresdir + 'directional_high_delay_average_in_beam.eps', bbox_inches=0)    
    
    selected_lags = NP.asarray([70.0, 100.0, 130.0])
    ind_selected_lags = [NP.abs(1e9*lags) < lag for lag in selected_lags]
    ind_maxdelay_beam = NP.empty(HP.nside2npix(chrmbeam_nside), dtype=NP.int)
    maxdelay_chrmbeam = NP.empty((HP.nside2npix(chrmbeam_nside), selected_lags.size))
    maxdelay_funcbeam = NP.empty((HP.nside2npix(chrmbeam_nside), selected_lags.size))
    for lagi, lag in enumerate(selected_lags):
        maskbeam_FFT = NP.copy(chrmbeam_FFT)
        maskbeam_FFT[:,ind_selected_lags[lagi]] = NP.nan
        ind_maxdelay_beam = NP.nanargmax(NP.abs(maskbeam_FFT), axis=1)
        maxdelay_chrmbeam[:,lagi] = 1e9 * lags[ind_maxdelay_beam]
        maskbeam_FFT = NP.copy(funcbeam_FFT)
        maskbeam_FFT[:,ind_selected_lags[lagi]] = NP.nan
        ind_maxdelay_beam = NP.nanargmax(NP.abs(maskbeam_FFT), axis=1)
        maxdelay_funcbeam[:,lagi] = 1e9 * lags[ind_maxdelay_beam]

    maxdelay_funcbeam[NP.logical_not(ind_hemisphere),:] = NP.nan
    maxdelay_chrmbeam[NP.logical_not(ind_hemisphere),:] = NP.nan
    
    fig, axs = PLT.subplots(nrows=selected_lags.size, ncols=2, sharex=True, sharey=True, figsize=(6,8))
    for lagi, lag in enumerate(selected_lags):
        coll_funcbeam = PolyCollection(verts, array=NP.abs(maxdelay_funcbeam[ind_hemisphere,lagi]), edgecolors='none')
        coll_chrmbeam = PolyCollection(verts, array=NP.abs(maxdelay_chrmbeam[ind_hemisphere,lagi]), edgecolors='none')
        coll_funcbeam.set_clim(vmin=selected_lags.min(), vmax=200)
        coll_chrmbeam.set_clim(vmin=selected_lags.min(), vmax=200)        
        # coll_funcbeam.set_norm(PLTC.LogNorm(vmin=selected_lags.min(), vmax=200))
        # coll_chrmbeam.set_norm(PLTC.LogNorm(vmin=selected_lags.min(), vmax=200))
        axs[lagi,0].add_collection(coll_funcbeam)
        axs[lagi,1].add_collection(coll_chrmbeam)
        axs[lagi,0].set_xlim(-1.0, 1.0)
        axs[lagi,0].set_ylim(-1.0, 1.0)
        axs[lagi,0].set_aspect('auto')
        axs[lagi,1].set_xlim(-1.0, 1.0)
        axs[lagi,1].set_ylim(-1.0, 1.0)
        axs[lagi,1].set_aspect('auto')
    cbax = fig.add_axes([0.12, 0.93, 0.82, 0.02])
    cbar = fig.colorbar(coll_funcbeam, cax=cbax, orientation='horizontal')
    cbax.set_xlabel('Delay [ns]', labelpad=10, fontsize=14)
    cbax.xaxis.set_label_position('top')    

    fig.subplots_adjust(wspace=0, hspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=20)

    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(top=0.88)
    
if '1c' in plots:
        
    # 01-c) Plot all-sky foreground delay power spectra with different beam chromaticities with EoR models overplotted for a 14.6m baseline for one LST
   
    if 'bli' in plot_info:
        bli = plot_info['bli']
    else:
        bli = 0
    if 'lsti' in plot_info:
        lsti = plot_info['lsti']
    else:
        lsti = 0
    if 'sbi' in plot_info:
        sbi = plot_info['sbi']
    else:
        sbi = 0

    kprll = fgdps_achrmbeam.k_parallel(fgds_achrmbeam.cc_lags, fgdps_achrmbeam.z, action='return')
    for fwi, fw in enumerate(freq_window_centers['cc']):

        fig = PLT.figure(figsize=(8,6))
        gs1 = GS.GridSpec(1,3)
        gs2 = GS.GridSpec(1,2)
        gs1.update(left=0.13, right=0.98, top=0.98, bottom=0.6, hspace=0, wspace=0)
        gs2.update(left=0.13, right=0.98, top=0.5, bottom=0.1, hspace=0, wspace=0)
        ax10 = PLT.subplot(gs1[0,0])
        ax11 = PLT.subplot(gs1[0,1], sharex=ax10)
        ax12 = PLT.subplot(gs1[0,2], sharex=ax10)
        ax20 = PLT.subplot(gs2[0,0])
        ax21 = PLT.subplot(gs2[0,1], sharex=ax20)

        ax10.plot(kprll, fgdps_achrmbeam.dps['skyvis'][bli,:,lsti], color='black', lw=2, ls='-')
        ax10.plot(kprll, fgdps_chrmbeam.dps['skyvis'][bli,:,lsti], color='blue', lw=2, ls='-')
        ax10.plot(kprll, fgdps_funcbeam.dps['skyvis'][bli,:,lsti], color='red', lw=2, ls='-')
        ax10.axvline(x=fgdps_achrmbeam.horizon_kprll_limits[lsti,bli,0], ymax=0.9, ls='-', lw=2, color='gray')
        ax10.axvline(x=fgdps_achrmbeam.horizon_kprll_limits[lsti,bli,1], ymax=0.9, ls='-', lw=2, color='gray')
        ax10.set_yscale('log')
        ax10.set_xlim(-0.49, 0.49)
        ax10.set_ylim(5e-5, 5e12)
       
        ax11.plot(kprll, fgdps_achrmbeam.dps['cc_skyvis'][bli,:,lsti], color='black', lw=2, ls='-')
        ax11.plot(kprll, fgdps_chrmbeam.dps['cc_skyvis'][bli,:,lsti], color='blue', lw=2, ls='-')
        ax11.plot(kprll, fgdps_funcbeam.dps['cc_skyvis'][bli,:,lsti], color='red', lw=2, ls='-')
        ax11.axvline(x=fgdps_achrmbeam.horizon_kprll_limits[lsti,bli,0], ymax=0.9, ls='-', lw=2, color='gray')
        ax11.axvline(x=fgdps_achrmbeam.horizon_kprll_limits[lsti,bli,1], ymax=0.9, ls='-', lw=2, color='gray')
        ax11.set_yscale('log')
        ax11.set_xlim(-0.49, 0.49)
        ax11.set_ylim(5e-5, 5e12)
        PLT.setp(ax11.get_yticklabels(), visible=False)

        ax12.plot(kprll, fgdps_achrmbeam.dps['cc_skyvis_res'][bli,:,lsti], color='black', lw=2, ls='-')
        ax12.plot(kprll, fgdps_chrmbeam.dps['cc_skyvis_res'][bli,:,lsti], color='blue', lw=2, ls='-')
        ax12.plot(kprll, fgdps_funcbeam.dps['cc_skyvis_res'][bli,:,lsti], color='red', lw=2, ls='-')
        ax12.axvline(x=fgdps_achrmbeam.horizon_kprll_limits[lsti,bli,0], ymax=0.9, ls='-', lw=2, color='gray')
        ax12.axvline(x=fgdps_achrmbeam.horizon_kprll_limits[lsti,bli,1], ymax=0.9, ls='-', lw=2, color='gray')        
        ax12.set_yscale('log')
        ax12.set_xlim(-0.49, 0.49)
        ax12.set_ylim(5e-5, 5e12)
        PLT.setp(ax12.get_yticklabels(), visible=False)

        ax20.plot(fgdps_achrmbeam.subband_delay_power_spectra['sim']['kprll'][fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'][bli,fwi,:,lsti], color='black', lw=2, ls='-')
        ax20.plot(fgdps_chrmbeam.subband_delay_power_spectra['sim']['kprll'][fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'][bli,fwi,:,lsti], color='blue', lw=2, ls='-')
        ax20.plot(fgdps_funcbeam.subband_delay_power_spectra['sim']['kprll'][fwi,:], fgdps_funcbeam.subband_delay_power_spectra['sim']['skyvis_lag'][bli,fwi,:,lsti], color='red', lw=2, ls='-')
        if model_21cmfast:
            ax20.plot(kprll, eor_21cmfast_Pk_interp[fwi,:,bli], 'k--', lw=2)
        
        ax20.axvline(x=fgdps_achrmbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,bli,0], ymax=0.9, ls='-', lw=2, color='gray')
        ax20.axvline(x=fgdps_achrmbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,bli,1], ymax=0.9, ls='-', lw=2, color='gray')        
        ax20.set_yscale('log')
        ax20.set_xlim(-0.49, 0.49)
        ax20.set_ylim(5e-7, 5e12)

        ax21.plot(fgdps_achrmbeam.subband_delay_power_spectra['cc']['kprll'][fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][bli,fwi,:,lsti], color='black', lw=2, ls='-')
        ax21.plot(fgdps_chrmbeam.subband_delay_power_spectra['cc']['kprll'][fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][bli,fwi,:,lsti], color='blue', lw=2, ls='-')
        ax21.plot(fgdps_funcbeam.subband_delay_power_spectra['cc']['kprll'][fwi,:], fgdps_funcbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][bli,fwi,:,lsti], color='red', lw=2, ls='-')
        if model_21cmfast:
            ax21.plot(kprll, eor_21cmfast_Pk_interp[fwi,:,bli], 'k--', lw=2)
        ax21.axvline(x=fgdps_achrmbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,bli,0], ymax=0.9, ls='-', lw=2, color='gray')
        ax21.axvline(x=fgdps_achrmbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,bli,1], ymax=0.9, ls='-', lw=2, color='gray')        
        ax21.set_yscale('log')
        ax21.set_xlim(-0.49, 0.49)
        ax21.set_ylim(5e-7, 5e12)
        PLT.setp(ax21.get_yticklabels(), visible=False)

        pos10 = ax10.get_position()
        pos12 = ax12.get_position()
        x0 = pos10.x0
        x1 = pos12.x1
        y0 = pos10.y0
        y1 = pos10.y1
        big_ax1 = fig.add_axes([x0, y0, x1-x0, y1-y0])
        big_ax1.set_axis_bgcolor('none')
        big_ax1.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax1.set_xticks([])
        big_ax1.set_yticks([])
        big_ax1.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=20)

        pos20 = ax20.get_position()
        pos21 = ax21.get_position()
        x0 = pos20.x0
        x1 = pos21.x1
        y0 = pos20.y0
        y1 = pos20.y1
        big_ax2 = fig.add_axes([x0, y0, x1-x0, y1-y0])
        big_ax2.set_axis_bgcolor('none')
        big_ax2.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax2.set_xticks([])
        big_ax2.set_yticks([])
        big_ax2.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=20)

        x0 = pos20.x0
        x1 = pos21.x1
        y0 = pos20.y0
        y1 = pos10.y1
        big_ax3 = fig.add_axes([x0, y0, x1-x0, y1-y0])
        big_ax3.set_axis_bgcolor('none')
        big_ax3.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax3.set_xticks([])
        big_ax3.set_yticks([])
        big_ax3.spines['right'].set_visible(False)
        big_ax3.spines['top'].set_visible(False)
        big_ax3.spines['left'].set_visible(False)
        big_ax3.spines['bottom'].set_visible(False)
        big_ax3.set_ylabel(r'$P(k_\parallel)$ [K$^2$(h$^{-1}$ Mpc)$^3$]', fontsize=16, weight='medium', labelpad=30)
        
        PLT.savefig(figuresdir+'asm_beam_chromaticity_bl{0:0d}_LST_{1:.2f}_hrs_subband_{2:.1f}_MHz.png'.format(bli,lst[lsti]/15.0,fw/1e6), bbox_inches=0)
        PLT.savefig(figuresdir+'asm_beam_chromaticity_bl{0:0d}_LST_{1:.2f}_hrs_subband_{2:.1f}_MHz.eps'.format(bli,lst[lsti]/15.0,fw/1e6), bbox_inches=0)    

    print '\n\tPlotted and saved ASM beam chromaticity'

if '1d' in plots:
        
    # 01-d) # Plot all-sky foreground delay power spectra with different beam chromaticities with EoR models overplotted for all baselines (wedge) for all LST
   
    kprll = fgdps_achrmbeam.k_parallel(fgds_achrmbeam.cc_lags, fgdps_achrmbeam.z, action='return')
    dps_skyvis_max = max([fgdps_achrmbeam.dps['skyvis'].max(), fgdps_funcbeam.dps['skyvis'].max(), fgdps_chrmbeam.dps['skyvis'].max()])

    for lsti in range(len(lst)):
        fig, axs = PLT.subplots(ncols=3, sharex=True, sharey=True, figsize=(8,3))
        fgdps_achrmbeam_plot = axs[0].pcolorfast(fgdps_achrmbeam.kperp, kprll, fgdps_achrmbeam.dps['skyvis'][:-1,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
        horizonb = axs[0].plot(fgdps_achrmbeam.kperp, fgdps_achrmbeam.horizon_kprll_limits[lsti,:,0], color='black', ls=':', lw=1.5)
        horizont = axs[0].plot(fgdps_achrmbeam.kperp, fgdps_achrmbeam.horizon_kprll_limits[lsti,:,1], color='black', ls=':', lw=1.5)
        axs[0].set_xlim(fgdps_achrmbeam.kperp.min(), fgdps_achrmbeam.kperp.max())
        axs[0].set_ylim(kprll.min(), kprll.max())
        axs[0].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[0].transAxes, fontsize=14, weight='medium', ha='center', color='white')

        fgdps_funcbeam_plot = axs[1].pcolorfast(fgdps_funcbeam.kperp, kprll, fgdps_funcbeam.dps['skyvis'][:-1,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
        horizonb = axs[1].plot(fgdps_funcbeam.kperp, fgdps_funcbeam.horizon_kprll_limits[lsti,:,0], color='black', ls=':', lw=1.5)
        horizont = axs[1].plot(fgdps_funcbeam.kperp, fgdps_funcbeam.horizon_kprll_limits[lsti,:,1], color='black', ls=':', lw=1.5)
        axs[1].set_xlim(fgdps_funcbeam.kperp.min(), fgdps_funcbeam.kperp.max())
        axs[1].set_ylim(kprll.min(), kprll.max())
        axs[1].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[1].transAxes, fontsize=14, weight='medium', ha='center', color='white')

        fgdps_chrmbeam_plot = axs[2].pcolorfast(fgdps_chrmbeam.kperp, kprll, fgdps_chrmbeam.dps['skyvis'][:-1,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
        horizonb = axs[2].plot(fgdps_chrmbeam.kperp, fgdps_chrmbeam.horizon_kprll_limits[lsti,:,0], color='black', ls=':', lw=1.5)
        horizont = axs[2].plot(fgdps_chrmbeam.kperp, fgdps_chrmbeam.horizon_kprll_limits[lsti,:,1], color='black', ls=':', lw=1.5)
        axs[2].set_xlim(fgdps_chrmbeam.kperp.min(), fgdps_chrmbeam.kperp.max())
        axs[2].set_ylim(kprll.min(), kprll.max())
        axs[2].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[2].transAxes, fontsize=14, weight='medium', ha='center', color='white')

        fig.subplots_adjust(wspace=0, hspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)
        big_ax.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=16)

        cbax = fig.add_axes([0.92, 0.18, 0.02, 0.7])
        cbar = fig.colorbar(fgdps_achrmbeam_plot, cax=cbax, orientation='vertical')
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')

        # PLT.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.subplots_adjust(top=0.9)
        fig.subplots_adjust(left=0.1)
        fig.subplots_adjust(bottom=0.18)

        PLT.savefig(figuresdir+'asm_wedge_beam_chromaticity_fullband_lst_{0:03d}.png'.format(lsti), bbox_inches=0)
        # PLT.savefig(figuresdir+'asm_wedge_beam_chromaticity_fullband_lst_{0:03d}.eps'.format(lsti), bbox_inches=0)
        PLT.close()
        
    kprll_sb = fgdps_achrmbeam.subband_delay_power_spectra['sim']['kprll']
    kperp_sb = fgdps_achrmbeam.subband_delay_power_spectra['sim']['kperp']

    # # dps_skyvis_max = max([fgdps_achrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'].max(), fgdps_funcbeam.subband_delay_power_spectra['sim']['skyvis_lag'].max(), fgdps_chrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'].max()])
    for fwi, fw in enumerate(freq_window_centers['sim'][:1]):
        for lsti in range(len(lst)):
            fig, axs = PLT.subplots(ncols=3, sharex=True, sharey=True, figsize=(8,3))
            fgdps_achrmbeam_plot = axs[0].pcolorfast(kperp_sb[fwi,:], kprll_sb[fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'][:-1,fwi,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
            horizonb = axs[0].plot(kperp_sb[fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,:,0], color='black', ls=':', lw=1.5)
            horizont = axs[0].plot(kperp_sb[fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,:,1], color='black', ls=':', lw=1.5)
            axs[0].set_xlim(kperp_sb[fwi,:].min(), kperp_sb[fwi,:].max())
            axs[0].set_ylim(kprll_sb[fwi,:].min(), kprll_sb[fwi,:].max())
            axs[0].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[0].transAxes, fontsize=14, weight='medium', ha='center', color='white')
            
            fgdps_funcbeam_plot = axs[1].pcolorfast(kperp_sb[fwi,:], kprll_sb[fwi,:], fgdps_funcbeam.subband_delay_power_spectra['sim']['skyvis_lag'][:-1,fwi,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
            horizonb = axs[1].plot(kperp_sb[fwi,:], fgdps_funcbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,:,0], color='black', ls=':', lw=1.5)
            horizont = axs[1].plot(kperp_sb[fwi,:], fgdps_funcbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,:,1], color='black', ls=':', lw=1.5)
            axs[1].set_xlim(kperp_sb[fwi,:].min(), kperp_sb[fwi,:].max())
            axs[1].set_ylim(kprll_sb[fwi,:].min(), kprll_sb[fwi,:].max())
            axs[1].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[1].transAxes, fontsize=14, weight='medium', ha='center', color='white')
            
            fgdps_chrmbeam_plot = axs[2].pcolorfast(kperp_sb[fwi,:], kprll_sb[fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'][:-1,fwi,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
            horizonb = axs[2].plot(kperp_sb[fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,:,0], color='black', ls=':', lw=1.5)
            horizont = axs[2].plot(kperp_sb[fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['sim']['horizon_kprll_limits'][lsti,fwi,:,1], color='black', ls=':', lw=1.5)
            axs[2].set_xlim(kperp_sb[fwi,:].min(), kperp_sb[fwi,:].max())
            axs[2].set_ylim(kprll_sb[fwi,:].min(), kprll_sb[fwi,:].max())
            axs[2].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[2].transAxes, fontsize=14, weight='medium', ha='center', color='white')
            
            fig.subplots_adjust(wspace=0, hspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)
            big_ax.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=16)
    
            cbax = fig.add_axes([0.92, 0.18, 0.02, 0.7])
            cbar = fig.colorbar(fgdps_achrmbeam_plot, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
    
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.9)
            fig.subplots_adjust(top=0.9)
            fig.subplots_adjust(left=0.1)
            fig.subplots_adjust(bottom=0.18)
    
            PLT.savefig(figuresdir+'sim_asm_wedge_beam_chromaticity_subband_{0:.1f}_MHz_lst_{1:03d}.png'.format(fw/1e6,lsti), bbox_inches=0)
            # PLT.savefig(figuresdir+'sim_asm_wedge_beam_chromaticity_subband_{0:.1f}_MHz_lst_{1:03d}.eps'.format(fw/1e6,lsti), bbox_inches=0)
            PLT.close()

    # # dps_skyvis_max = max([fgdps_achrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'].max(), fgdps_funcbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'].max(), fgdps_chrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'].max()])
    # for fwi, fw in enumerate(freq_window_centers['cc'][:1]):
    #     for lsti in range(len(lst)):
    #         fig, axs = PLT.subplots(ncols=3, sharex=True, sharey=True, figsize=(8,3))
    #         fgdps_achrmbeam_plot = axs[0].pcolorfast(kperp_sb[fwi,:], kprll_sb[fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][:-1,fwi,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
    #         horizonb = axs[0].plot(kperp_sb[fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,:,0], color='black', ls=':', lw=1.5)
    #         horizont = axs[0].plot(kperp_sb[fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,:,1], color='black', ls=':', lw=1.5)
    #         axs[0].set_xlim(kperp_sb[fwi,:].min(), kperp_sb[fwi,:].max())
    #         axs[0].set_ylim(kprll_sb[fwi,:].min(), kprll_sb[fwi,:].max())
    #         axs[0].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[0].transAxes, fontsize=14, weight='medium', ha='center', color='white')
            
    #         fgdps_funcbeam_plot = axs[1].pcolorfast(kperp_sb[fwi,:], kprll_sb[fwi,:], fgdps_funcbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][:-1,fwi,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
    #         horizonb = axs[1].plot(kperp_sb[fwi,:], fgdps_funcbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,:,0], color='black', ls=':', lw=1.5)
    #         horizont = axs[1].plot(kperp_sb[fwi,:], fgdps_funcbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,:,1], color='black', ls=':', lw=1.5)
    #         axs[1].set_xlim(kperp_sb[fwi,:].min(), kperp_sb[fwi,:].max())
    #         axs[1].set_ylim(kprll_sb[fwi,:].min(), kprll_sb[fwi,:].max())
    #         axs[1].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[1].transAxes, fontsize=14, weight='medium', ha='center', color='white')
            
    #         fgdps_chrmbeam_plot = axs[2].pcolorfast(kperp_sb[fwi,:], kprll_sb[fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][:-1,fwi,:-1,lsti].T, norm=PLTC.LogNorm(vmin=1e-4, vmax=dps_skyvis_max))
    #         horizonb = axs[2].plot(kperp_sb[fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,:,0], color='black', ls=':', lw=1.5)
    #         horizont = axs[2].plot(kperp_sb[fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['cc']['horizon_kprll_limits'][lsti,fwi,:,1], color='black', ls=':', lw=1.5)
    #         axs[2].set_xlim(kperp_sb[fwi,:].min(), kperp_sb[fwi,:].max())
    #         axs[2].set_ylim(kprll_sb[fwi,:].min(), kprll_sb[fwi,:].max())
    #         axs[2].text(0.5, 0.9, '{0:.2f} hrs'.format(lst[lsti]/15.0), transform=axs[2].transAxes, fontsize=14, weight='medium', ha='center', color='white')
            
    #         fig.subplots_adjust(wspace=0, hspace=0)
    #         big_ax = fig.add_subplot(111)
    #         big_ax.set_axis_bgcolor('none')
    #         big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    #         big_ax.set_xticks([])
    #         big_ax.set_yticks([])
    #         big_ax.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)
    #         big_ax.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=16)
    
    #         cbax = fig.add_axes([0.92, 0.18, 0.02, 0.7])
    #         cbar = fig.colorbar(fgdps_achrmbeam_plot, cax=cbax, orientation='vertical')
    #         cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
    #         cbax.xaxis.set_label_position('top')
    
    #         # PLT.tight_layout()
    #         fig.subplots_adjust(right=0.9)
    #         fig.subplots_adjust(top=0.9)
    #         fig.subplots_adjust(left=0.1)
    #         fig.subplots_adjust(bottom=0.18)
    
    #         PLT.savefig(figuresdir+'ccres_asm_wedge_beam_chromaticity_subband_{0:.1f}_MHz_lst_{1:03d}.png'.format(fw/1e6,lsti), bbox_inches=0)
    #         # PLT.savefig(figuresdir+'ccres_asm_wedge_beam_chromaticity_subband_{0:.1f}_MHz_lst_{1:03d}.eps'.format(fw/1e6,lsti), bbox_inches=0)
    #         PLT.close()
            
if '1e' in plots:

    # 01-e) Plot all-sky foreground delay power spectra on selected baselines, selected LST and selected subband

    mdl = parms['plot']['1e']['eor_mdl']
    dpstype = parms['plot']['1e']['dpstype']
    strategies = parms['plot']['1e']['strategies']
    subband_freq = parms['plot']['1e']['subband']
    subband_index = NP.argmin(NP.abs(freq_window_centers['sim'] - subband_freq))
    beamtype = ['achrmbeam', 'funcbeam', 'chrmbeam']
    blx = parms['plot']['1e']['blx_list']
    bly = parms['plot']['1e']['bly_list']
    blx = NP.asarray(blx)
    bly = NP.asarray(bly)
    blxy = NP.hstack((blx.reshape(-1,1), bly.reshape(-1,1)))
    blref = fgvis_achrmbeam.baselines[:,:2]
    lst_inp = parms['plot']['1e']['lst_list']
    lst_inp = NP.asarray(lst_inp).reshape(-1,1)
    inpblind, refblind, distblNN = LKP.find_1NN(blref, blxy, distance_ULIM=1.0, remove_oob=False)
    inplstind, reflstind, distlstNN = LKP.find_1NN(lst.reshape(-1,1)/15.0, lst_inp, distance_ULIM=0.1, remove_oob=False)
    sortind = NP.argsort(bl_length[refblind])
    refblind_sorted = refblind[sortind]
    inpblind_sorted = inpblind[sortind]
    reflstind_sorted = reflstind[sortind]
    inplstind_sorted = inplstind[sortind]
    bl_angle = NP.degrees(NP.angle(blref[refblind_sorted,0]+1j*blref[refblind_sorted,1]))

    dfreq = fgds_achrmbeam.f[1] - fgds_achrmbeam.f[0]
    sbIC_lags = DSP.spectral_axis(fgdps_sbIC['achrmbeam']['pCnorm'].shape[2], delx=dfreq, shift=True)
    eta2kprll = DS.dkprll_deta(fgdps_achrmbeam.z, fgdps_achrmbeam.cosmo)
    sbIC_kprll = sbIC_lags * eta2kprll

    ymin = 5e-7
    ymax = 5e9
    for strategy in strategies:
        if strategy == 'sim': dpskey = 'skyvis_lag'
        if strategy == 'cc': dpskey = 'skyvis_res_lag'
        fig, axs = PLT.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(8,4))
        for row in range(axs.shape[0]):
            for col in range(axs.shape[1]):
                i = col + row * axs.shape[1]
                if dpstype == 'ic':
                    clrs = ['black', 'red', 'blue']
                    beamstr = ['achrmbeam', 'funcbeam', 'chrmbeam']
                    reflstind_sorted = NP.where(reflstind_sorted < fgdps_sbIC[beamstr[0]]['pCnorm'].shape[3], reflstind_sorted, fgdps_sbIC[beamstr[0]]['pCnorm'].shape[3]-1)
                    for bmi,bmstr in enumerate(beamstr):
                        axs[row,col].plot(sbIC_kprll, NP.abs(fgdps_sbIC[bmstr]['pCnorm'][subband_index,refblind_sorted[i],:,reflstind_sorted[i]]), color=clrs[bmi], lw=2, ls='-')
                else:
                    axs[row,col].plot(fgdps_achrmbeam.subband_delay_power_spectra_resampled[strategy]['kprll'][subband_index,:], fgdps_achrmbeam.subband_delay_power_spectra_resampled[strategy][dpskey][refblind_sorted[i],subband_index,:,reflstind_sorted[i]], color='black', lw=2, ls='-')
                    axs[row,col].plot(fgdps_funcbeam.subband_delay_power_spectra_resampled[strategy]['kprll'][subband_index,:], fgdps_funcbeam.subband_delay_power_spectra_resampled[strategy][dpskey][refblind_sorted[i],subband_index,:,reflstind_sorted[i]], color='red', lw=2, ls='-')
                    axs[row,col].plot(fgdps_chrmbeam.subband_delay_power_spectra_resampled[strategy]['kprll'][subband_index,:], fgdps_chrmbeam.subband_delay_power_spectra_resampled[strategy][dpskey][refblind_sorted[i],subband_index,:,reflstind_sorted[i]], color='blue', lw=2, ls='-')
                    # axs[row,col].plot(fgdps_achrmbeam.subband_delay_power_spectra[strategy]['kprll'][subband_index,:], fgdps_achrmbeam.subband_delay_power_spectra[strategy][dpskey][refblind_sorted[i],subband_index,:,reflstind_sorted[i]], color='black', lw=2, ls='-')
                    # axs[row,col].plot(fgdps_funcbeam.subband_delay_power_spectra[strategy]['kprll'][subband_index,:], fgdps_funcbeam.subband_delay_power_spectra[strategy][dpskey][refblind_sorted[i],subband_index,:,reflstind_sorted[i]], color='red', lw=2, ls='-')
                    # axs[row,col].plot(fgdps_chrmbeam.subband_delay_power_spectra[strategy]['kprll'][subband_index,:], fgdps_chrmbeam.subband_delay_power_spectra[strategy][dpskey][refblind_sorted[i],subband_index,:,reflstind_sorted[i]], color='blue', lw=2, ls='-')

                    if freq_window_fftpow[strategy] > 1.0:
                        ymin = 5e-9
                axs[row,col].axvline(x=fgdps_achrmbeam.subband_delay_power_spectra[strategy]['horizon_kprll_limits'][reflstind_sorted[i],subband_index,refblind_sorted[i],0], ymax=0.9, ls=':', lw=2, color='black')
                axs[row,col].axvline(x=fgdps_achrmbeam.subband_delay_power_spectra[strategy]['horizon_kprll_limits'][reflstind_sorted[i],subband_index,refblind_sorted[i],1], ymax=0.9, ls=':', lw=2, color='black')        
    
                # if mdl == '21cmfast':
                axs[row,col].plot(kprll, eor_21cmfast_Pk_interp[subband_index,:,refblind_sorted[i]], ls='-', lw=3, color='cyan')
                # elif mdl == 'lidz':
                axs[row,col].plot(kprll, lidz_eor_Pk_interp[subband_index,:,refblind_sorted[i]], ls='-', lw=3, color='gray')
                # else:
                #     raise ValueError('This EoR model is not currently implemented')
        
                axs[row,col].text(0.95, 0.95, '{0:.1f} m\n{1:.1f}'.format(bl_length[refblind_sorted[i]], bl_angle[i])+r'$^\circ$', transform=axs[row,col].transAxes, fontsize=10, weight='medium', ha='right', va='top', color='black')
                axs[row,col].set_yscale('log')
                axs[row,col].set_xlim(-0.59, 0.59)
                axs[row,col].set_ylim(ymin, ymax)
                axs[row,col].tick_params(axis='x', labelsize=9)
                # if dpstype == 'win':
                #     if freq_window_fftpow[strategy] > 1.0:
                #         axs[row,col].set_yticks(yticks)

        fig.subplots_adjust(hspace=0, wspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=12, weight='medium', labelpad=16)
        big_ax.set_ylabel(r'$P(k_\parallel)$ [K$^2$(h$^{-1}$ Mpc)$^3$]', fontsize=12, weight='medium', labelpad=30)
        fig.subplots_adjust(top=0.96)
        fig.subplots_adjust(bottom=0.12)
        fig.subplots_adjust(left=0.09)
        fig.subplots_adjust(right=0.98)    
    
        if dpstype == 'ic':
            plotfilename = '{0}_asm_foreground_eor_beam_chromaticity_{1:.1f}_MHz_subband_{2}_method'.format(strategy, subband_freq/1e6, dpstype)
        else:
            plotfilename = '{0}_asm_foreground_eor_beam_chromaticity_{1:.1f}_MHz_subband_{2}_method_{3}{4:.1f}'.format(strategy, subband_freq/1e6, dpstype, freq_window_shape[strategy], freq_window_fftpow[strategy])
        PLT.savefig(figuresdir+plotfilename+'.png'.format(strategy, subband_freq/1e6, dpstype), bbox_inches=0)
        PLT.savefig(figuresdir+plotfilename+'.eps'.format(strategy, subband_freq/1e6, dpstype), bbox_inches=0)

    print '\n\tPlotted and saved EoR and ASM foreground with varying beam chromaticity in {0:.1f} MHz subband'.format(subband_freq/1e6)

    fig, axs = PLT.subplots(ncols=4, nrows=2, sharex=True, sharey=True, figsize=(8,4))
    for row in range(axs.shape[0]):
        for col in range(axs.shape[1]):
            i = col + row * axs.shape[1]
            axs[row,col].plot(kprll, fgdps_achrmbeam.dps['skyvis'][refblind_sorted[i],:,reflstind_sorted[i]], color='black', lw=2, ls='-')
            axs[row,col].plot(kprll, fgdps_funcbeam.dps['skyvis'][refblind_sorted[i],:,reflstind_sorted[i]], color='red', lw=2, ls='-', alpha=0.7)
            axs[row,col].plot(kprll, fgdps_chrmbeam.dps['skyvis'][refblind_sorted[i],:,reflstind_sorted[i]], color='blue', lw=2, ls='-', alpha=0.7)

            axs[row,col].axvline(x=fgdps_achrmbeam.horizon_kprll_limits[reflstind_sorted[i],refblind_sorted[i],0], ymax=0.9, ls=':', lw=2, color='black')
            axs[row,col].axvline(x=fgdps_achrmbeam.horizon_kprll_limits[reflstind_sorted[i],refblind_sorted[i],1], ymax=0.9, ls=':', lw=2, color='black')

            axs[row,col].text(0.95, 0.95, '{0:.1f} m\n{1:.1f}'.format(bl_length[refblind_sorted[i]], bl_angle[i])+r'$^\circ$', transform=axs[row,col].transAxes, fontsize=10, weight='medium', ha='right', va='top', color='black')
            axs[row,col].set_yscale('log')
            axs[row,col].set_xlim(-0.59, 0.59)
            axs[row,col].set_ylim(ymin, ymax)
            axs[row,col].tick_params(axis='x', labelsize=9)

    fig.subplots_adjust(hspace=0, wspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=12, weight='medium', labelpad=16)
    big_ax.set_ylabel(r'$P(k_\parallel)$ [K$^2$(h$^{-1}$ Mpc)$^3$]', fontsize=12, weight='medium', labelpad=30)
    fig.subplots_adjust(top=0.96)
    fig.subplots_adjust(bottom=0.12)
    fig.subplots_adjust(left=0.09)
    fig.subplots_adjust(right=0.98)

    PLT.savefig(figuresdir+'asm_foreground_eor_beam_chromaticity_fullband_{0}{1:.1f}.png'.format(freq_window_shape['sim'], freq_window_fftpow['sim']), bbox_inches=0)
    PLT.savefig(figuresdir+'asm_foreground_eor_beam_chromaticity_fullband_{0}{1:.1f}.eps'.format(freq_window_shape['sim'], freq_window_fftpow['sim']), bbox_inches=0)

    print '\n\tPlotted and saved EoR and ASM foreground with varying beam chromaticity in {0:.1f} MHz subband'.format(subband_freq/1e6)
    
if '2a' in plots:
        
    # 02-a) Plot ratio of signal to foreground ratio as a function of staggered delays to set a spec on reflectometry
        
    k_reflectometry = NP.asarray(k_reflectometry)
    kprll_reflectometry = k_reflectometry
    subband_freq = parms['plot']['2a']['sbfreq']
    subband_index = NP.argmin(NP.abs(freq_window_centers['sim'] - subband_freq))
    bll = parms['plot']['2a']['bll']
    lst_begin = parms['plot']['2a']['lst_beg']
    lst_end = parms['plot']['2a']['lst_end']
    if lst_begin > lst_end:
        lst_ind = (lst/15.0 >= lst_begin) | (lst/15.0 <= lst_end)
    elif lst_begin < lst_end:
        lst_ind = (lst/15.0 >= lst_begin) & (lst/15.0 <= lst_end)
    else:
        raise ValueError('LST begin and end must not be identical')

    bleps = 0.2
    blind = NP.abs(bl_length - bll) <= bleps
    # eormdl = 10 ** Pk_21cmfast_interp_func(NP.log10(k_reflectometry))
    
    eta2kprll = DS.dkprll_deta(fgdps_achrmbeam.z, fgdps_achrmbeam.cosmo)
    tau_reflectometry = kprll_reflectometry / eta2kprll
    
    fg_lags = fgds_achrmbeam.cc_lags
    fg_kprll = eta2kprll * fg_lags
    dlags = fg_lags[1] - fg_lags[0]
    dkprll = fg_kprll[1] - fg_kprll[0]
    fgmdl_achrmbeam = NP.mean(fgdps_achrmbeam.dps['cc_skyvis_net'][blind,...], axis=0)
    fgmdl_chrmbeam = NP.mean(fgdps_chrmbeam.dps['cc_skyvis_net'][blind,...], axis=0)
    fgmdl_funcbeam = NP.mean(fgdps_funcbeam.dps['cc_skyvis_net'][blind,...], axis=0)

    fg_lags_fold = fg_lags[fg_lags.size/2:]
    fg_kprll_fold = fg_kprll[fg_kprll.size/2:]

    fgmdl_achrmbeam_fold = 0.5 * (fgmdl_achrmbeam[fg_lags.size/2:,lst_ind] + fgmdl_achrmbeam[-fg_lags.size/2:0:-1,lst_ind])
    fgmdl_chrmbeam_fold = 0.5 * (fgmdl_chrmbeam[fg_lags.size/2:,lst_ind] + fgmdl_chrmbeam[-fg_lags.size/2:0:-1,lst_ind])
    fgmdl_funcbeam_fold = 0.5 * (fgmdl_funcbeam[fg_lags.size/2:,lst_ind] + fgmdl_funcbeam[-fg_lags.size/2:0:-1,lst_ind])

    # # My version

    # eormdl = 10 ** Pk_21cmfast_interp_func(NP.log10(fg_kprll_fold))
    # ind_cutoff = [NP.argmin(NP.abs(fg_lags_fold <= tau_r)) for tau_r in tau_reflectometry]    
    # ind_cutoff = NP.asarray(ind_cutoff)
    # dspec = {'achrmbeam': {}, 'chrmbeam': {}, 'funcbeam': {}}
    # for i,indcut in enumerate(ind_cutoff):
    #     dspec['achrmbeam']['k{0:0d}'.format(i)] = NP.zeros((fgmdl_achrmbeam_fold.shape[1], indcut+1))
    #     dspec['chrmbeam']['k{0:0d}'.format(i)] = NP.zeros((fgmdl_chrmbeam_fold.shape[1], indcut+1))
    #     dspec['funcbeam']['k{0:0d}'.format(i)] = NP.zeros((fgmdl_funcbeam_fold.shape[1], indcut+1))
    #     for ti, dtau in enumerate(fg_lags_fold[:indcut+1]):
    #         nearest_tau_ind = NP.argmin(NP.abs(fg_lags_fold+dtau - tau_reflectometry[i]))
    #         dspec['achrmbeam']['k{0:0d}'.format(i)][:,ti] = NP.sqrt(NP.min(eormdl[subband_index,indcut:].reshape(-1,1) / fgmdl_achrmbeam_fold[nearest_tau_ind:fg_kprll_fold.size-ti,:], axis=0))
    #         dspec['chrmbeam']['k{0:0d}'.format(i)][:,ti] = NP.sqrt(NP.min(eormdl[subband_index,indcut:].reshape(-1,1) / fgmdl_chrmbeam_fold[nearest_tau_ind:fg_kprll_fold.size-ti,:], axis=0))
    #         dspec['funcbeam']['k{0:0d}'.format(i)][:,ti] = NP.sqrt(NP.min(eormdl[subband_index,indcut:].reshape(-1,1) / fgmdl_funcbeam_fold[nearest_tau_ind:fg_kprll_fold.size-ti,:], axis=0))

    # # tau_bll = bl_length / FCNST.c
    # tau_bll = NP.arange(0.0, fg_lags_fold.max(), bll/FCNST.c)
    # kprll_bll = eta2kprll * tau_bll
    beamkeys = ['chrmbeam', 'funcbeam', 'achrmbeam']
    beamlabels = ['Sim.', 'Airy', 'Achrm.']
    ls = ['-', '--', ':']
    bcolors = ['0.75', '0.5', '0.25']
    kcolors = ['0.75', '0.5', '0.25']
    # zorder = [3,2,1]

    # # fig, axs = PLT.subplots(ncols=kprll_reflectometry.size, sharex=True, sharey=True, figsize=(7,3.5))
    # # for kpi in range(kprll_reflectometry.size):
    # #     for bi,bkey in enumerate(beamkeys):
    # #         axs[kpi].plot(1e9 * fg_lags_fold[:dspec[bkey]['k{0:0d}'.format(kpi)].shape[1]], 10*NP.log10(NP.min(dspec[bkey]['k{0:0d}'.format(kpi)], axis=0)), ls=ls[bi], lw=2, color=bcolors[bi], label=beamlabels[bi])
    # #         axs[kpi].fill_between(1e9 * fg_lags_fold[:dspec[bkey]['k{0:0d}'.format(kpi)].shape[1]], NP.zeros(dspec[bkey]['k{0:0d}'.format(kpi)].shape[1]), 10*NP.log10(NP.min(dspec[bkey]['k{0:0d}'.format(kpi)], axis=0)), facecolor='black', alpha=0.5)
    # #     for lag_refl in tau_bll:
    # #         axs[kpi].axvline(x=1e9*lag_refl, ymin=0.0, ymax=0.85, lw=1, ls='-', color='0.33')
    # #     axs[kpi].set_xlim(0, 240)
    # #     lgnd = axs[kpi].legend(frameon=True, fontsize=10, loc='upper right', bbox_to_anchor=(0.99,0.93))
    # #     for txt in lgnd.get_texts():
    # #         txt.set_color('black')
    # #     axs[kpi].text(0.02, 0.98, r'$k_\parallel \geq$'+' {0:.2f} h Mpc'.format(kprll_reflectometry[kpi])+r'$^{-1}$', transform=axs[kpi].transAxes, fontsize=10, weight='medium', ha='left', va='top', color='1.0')
    # # fig.subplots_adjust(hspace=0, wspace=0)
    # # big_ax = fig.add_subplot(111)
    # # big_ax.set_axis_bgcolor('none')
    # # big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # # big_ax.set_xticks([])
    # # big_ax.set_yticks([])
    # # big_ax.set_ylabel('Attenuation [dB]', fontsize=16, weight='medium', labelpad=30)
    # # big_ax.set_xlabel(r'$\tau$ [ns]', fontsize=16, weight='medium', labelpad=18)
    # # fig.subplots_adjust(top=0.95)
    # # fig.subplots_adjust(bottom=0.15)
    # # fig.subplots_adjust(left=0.1)
    # # fig.subplots_adjust(right=0.95)    
    # 
    # # PLT.savefig(figuresdir+'spec_on_foreground_reflected_power_21cmfast_{0:.1f}m_{1:.1f}_MHz_subband_v1.png'.format(bll, subband_freq/1e6), bbox_inches=0)
    # # PLT.savefig(figuresdir+'spec_on_foreground_reflected_power_21cmfast_{0:.1f}m_{1:.1f}_MHz_subband_v1.eps'.format(bll, subband_freq/1e6), bbox_inches=0)    

    # Aaron Parson's version

    tau = NP.linspace(0, 500, 1000)
    kprll = (1e-9 * tau) * eta2kprll
    dspec = {'achrmbeam': {}, 'chrmbeam': {}, 'funcbeam': {}}
    for beamtype in ['achrmbeam', 'chrmbeam', 'funcbeam']:
        dspec[beamtype] = NP.ones((k_reflectometry.size, tau.size), dtype=NP.float)
        if beamtype == 'achrmbeam': fg_pow = fgmdl_achrmbeam[:,lst_ind]
        if beamtype == 'chrmbeam': fg_pow = fgmdl_chrmbeam[:,lst_ind]
        if beamtype == 'funcbeam': fg_pow = fgmdl_funcbeam[:,lst_ind]
        fg_pow = NP.mean(fg_pow, axis=1)
        # fg_mdl = NP.copy(fg_pow)
        fg_mdl = NP.where(fg_pow > 1e1, fg_pow, 0.0) # Need to understand why this threshold is used
        for cuti,kprll_cut in enumerate(kprll_reflectometry):
            for j,fgkpl in enumerate(fg_kprll):
                if fg_mdl[j] <= 0.0: continue
                # eorvals = 10 ** lidz_Pk_interp_func(NP.log10(kprll+fgkpl))
                eorvals = 10 ** Pk_21cmfast_interp_func(NP.log10(kprll+fgkpl))
                resp = NP.sqrt(eorvals[subband_index,:] / fg_mdl[j])
                resp = NP.where(kprll+fgkpl < kprll_cut, 1.0, resp)
                dspec[beamtype][cuti,:] = NP.where(resp < dspec[beamtype][cuti,:], resp, dspec[beamtype][cuti,:])
        dspec[beamtype] = -10.0 * NP.log10(dspec[beamtype])

    bkey = 'achrmbeam'
    fig = PLT.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(111)
    for kpi in range(kprll_reflectometry.size):
        ax.plot(tau, dspec[bkey][kpi,:], ls=ls[kpi], lw=2, color='black', zorder=2, label=r'$k_\parallel \geq$'+' {0:.2f} h Mpc'.format(kprll_reflectometry[kpi])+r'$^{-1}$')
        ax.fill_between(tau, NP.zeros_like(tau), dspec[bkey][kpi,:], facecolor=kcolors[kpi], edgecolor='none', zorder=1)
        ax.set_xlim(0, 490)
        ax.set_ylim(80, 0)
        ax.set_ylabel('Attenuation [dB]', fontsize=14, weight='medium')
        ax.set_xlabel(r'$\tau$ [ns]', fontsize=14, weight='medium')
    lgnd = ax.legend(frameon=True, fontsize=8, loc='upper right', bbox_to_anchor=(0.999,0.999))

    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(right=0.98)

    PLT.savefig(figuresdir+'spec_on_{0}_foreground_reflected_power_21cmfast_{1:.1f}m_{2:.1f}_MHz_subband_v2.png'.format(bkey, bll, subband_freq/1e6), bbox_inches=0)
    PLT.savefig(figuresdir+'spec_on_{0}_foreground_reflected_power_21cmfast_{1:.1f}m_{2:.1f}_MHz_subband_v2.eps'.format(bkey, bll, subband_freq/1e6), bbox_inches=0)

    fig, axs = PLT.subplots(ncols=kprll_reflectometry.size, sharex=True, sharey=True, figsize=(7,3.5))
    for kpi in range(kprll_reflectometry.size):
        for bi,bkey in enumerate(beamkeys[:2]):
            axs[kpi].plot(tau, dspec[bkey][kpi,:], ls=ls[bi], lw=2, color='black', label=beamlabels[bi], zorder=2)
            axs[kpi].fill_between(tau, NP.zeros_like(tau), dspec[bkey][kpi,:], facecolor=bcolors[bi], edgecolor='none', zorder=1)
        # for lag_refl in tau_bll:
        #     axs[kpi].axvline(x=1e9*lag_refl, ymin=0.0, ymax=0.85, lw=1, ls='-', color='0.33')
        axs[kpi].set_xlim(0, 490)
        axs[kpi].set_ylim(80, 0)
        lgnd = axs[kpi].legend(frameon=True, fontsize=10, loc='upper right', bbox_to_anchor=(0.99,0.93))
        for txt in lgnd.get_texts():
            txt.set_color('black')
        axs[kpi].text(0.98, 0.98, r'$k_\parallel \geq$'+' {0:.2f} h Mpc'.format(kprll_reflectometry[kpi])+r'$^{-1}$', transform=axs[kpi].transAxes, fontsize=10, weight='medium', ha='right', va='top', color='0.0')
    fig.subplots_adjust(hspace=0, wspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel('Attenuation [dB]', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel(r'$\tau$ [ns]', fontsize=16, weight='medium', labelpad=18)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(right=0.95)    
    
    PLT.savefig(figuresdir+'spec_on_foreground_reflected_power_21cmfast_{0:.1f}m_{1:.1f}_MHz_subband_v2.png'.format(bll, subband_freq/1e6), bbox_inches=0)
    PLT.savefig(figuresdir+'spec_on_foreground_reflected_power_21cmfast_{0:.1f}m_{1:.1f}_MHz_subband_v2.eps'.format(bll, subband_freq/1e6), bbox_inches=0)
    # PLT.savefig(figuresdir+'spec_on_foreground_reflected_power_21cmfast_{0:.1f}m_{1:.1f}_MHz_subband_v2.pdf'.format(bll, subband_freq/1e6), bbox_inches=0)
    
    print '\n\tPlotted and saved spec on foreground reflected power'
        
if ('3a' in plots) or ('3b' in plots) or ('3c' in plots):
        
    # 03-a) Plot signal-to-foreground ratio as a function of LST and k_parallel for a given baseline

    kprll_sb = fgdps_achrmbeam.subband_delay_power_spectra_resampled['sim']['kprll']
    kperp_sb = fgdps_achrmbeam.subband_delay_power_spectra_resampled['sim']['kperp']
    k_sb = NP.sqrt(kprll_sb[:,NP.newaxis,:]**2 + kperp_sb[:,:,NP.newaxis]**2)

    eormdl_sb = {}
    eor_fg_ratio_sb = {}
    for mdl in ['lidz', '21cmfast']:
        eor_fg_ratio_sb[mdl] = {}
        eormdl_sb[mdl] = NP.empty((freq_window_centers['sim'].size, kperp_sb.shape[1], kprll_sb.shape[1]))
        for beamtype in ['achrmbeam', 'chrmbeam', 'funcbeam']:
            eor_fg_ratio_sb[mdl][beamtype] = {}
            for strategy in ['cc', 'sim']:
                eor_fg_ratio_sb[mdl][beamtype][strategy] = {}

    for si, subband_freq in enumerate(freq_window_centers['sim']):
        for mdl in ['lidz', '21cmfast']:
            if mdl == 'lidz':
                eorPk = lidz_Pk_interp_func(NP.log10(k_sb[si,:,:].flatten()))
            if mdl == '21cmfast':
                eorPk = Pk_21cmfast_interp_func(NP.log10(k_sb[si,:,:].flatten()))
            eorPk = 10 ** eorPk.reshape(freq_window_centers['sim'].size, kperp_sb.shape[1], -1)
            eormdl_sb[mdl][si,:,:] = NP.copy(eorPk[si,:,:])

    fgpower = None
    for mdl in ['lidz', '21cmfast']:
        eormdl_sb[mdl] = NP.swapaxes(eormdl_sb[mdl], 0, 1)
        eormdl_sb[mdl] = eormdl_sb[mdl][:,:,:,NP.newaxis]
        for beamtype in ['achrmbeam', 'chrmbeam', 'funcbeam']:
            for strategy in ['cc', 'sim']:
                if strategy == 'cc':
                    if beamtype == 'achrmbeam':
                        fgpower = fgdps_achrmbeam.subband_delay_power_spectra_resampled[strategy]['skyvis_res_lag']
                    if beamtype == 'chrmbeam':
                        fgpower = fgdps_chrmbeam.subband_delay_power_spectra_resampled[strategy]['skyvis_res_lag']
                    if beamtype == 'funcbeam':
                        fgpower = fgdps_funcbeam.subband_delay_power_spectra_resampled[strategy]['skyvis_res_lag']
                else:
                    if beamtype == 'achrmbeam':
                        fgpower = fgdps_achrmbeam.subband_delay_power_spectra_resampled[strategy]['skyvis_lag']
                    if beamtype == 'chrmbeam':
                        fgpower = fgdps_chrmbeam.subband_delay_power_spectra_resampled[strategy]['skyvis_lag']
                    if beamtype == 'funcbeam':
                        fgpower = fgdps_funcbeam.subband_delay_power_spectra_resampled[strategy]['skyvis_lag']
                # fgpower = NP.mean(fgpower[:3,...], axis=0, keepdims=True)
                eor_fg_ratio_sb[mdl][beamtype][strategy] = eormdl_sb[mdl] / fgpower

    mdl_str = '21cmfast'
    subband_index = 0

    if '3a' in plots:

        # fig, axs = PLT.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8.5,3.5))
        # for strati, strategy in enumerate(['sim', 'cc']):
        #     for beami, beamtype in enumerate(['achrmbeam', 'funcbeam', 'chrmbeam']):
        #         ratio2D = axs[strati,beami].imshow(eor_fg_ratio_sb[mdl_str][beamtype][strategy][0,subband_index,:,:].T, origin='lower', extent=[kprll_sb[subband_index,:].min(), kprll_sb[subband_index,:].max(), 0, 24], norm=PLTC.LogNorm(vmin=1e-2, vmax=2))
        #         # axs[strati,beami].set_xlim(kprll_sb[subband_index,:].min(), kprll_sb[subband_index,:].max())
        #         axs[strati,beami].set_xlim(-0.39, 0.39)
        #         axs[strati,beami].set_ylim(0, 24)
        #         axs[strati,beami].set_aspect('auto')
    
        # cbax = fig.add_axes([0.92, 0.12, 0.02, 0.83])
        # cbar = fig.colorbar(ratio2D, cax=cbax, orientation='vertical')
        # cbax.yaxis.set_label_position('right')    
    
        # fig.subplots_adjust(hspace=0, wspace=0)
        # big_ax = fig.add_subplot(111)
        # big_ax.set_axis_bgcolor('none')
        # big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        # big_ax.set_xticks([])
        # big_ax.set_yticks([])
        # big_ax.set_ylabel('RA [hours]', fontsize=16, weight='medium', labelpad=20)
        # big_ax.set_xlabel(r'$k_\parallel$'+'h Mpc'+r'$^{-1}$', fontsize=16, weight='medium', labelpad=20)
    
        # # PLT.tight_layout()
        # fig.subplots_adjust(right=0.9)
        # fig.subplots_adjust(top=0.95) 
        # fig.subplots_adjust(bottom=0.15)   
        # fig.subplots_adjust(left=0.08)
    
        mdl_str = '21cmfast'
        beamtype = 'chrmbeam'
        subband_index = 0
        bl_index = 2
        ratiomax = max(NP.nanmax(eor_fg_ratio_sb[mdl_str][beamtype][strat][bl_index,subband_index,:,:]) for strat in ['sim','cc'])
        fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(7,3.5))
        for strati, strategy in enumerate(['sim', 'cc']):
            ratio2D = axs[strati].imshow(eor_fg_ratio_sb[mdl_str][beamtype][strategy][bl_index,subband_index,:,:].T, origin='lower', extent=[kprll_sb[subband_index,:].min(), kprll_sb[subband_index,:].max(), 0, 24], norm=PLTC.LogNorm(vmin=1e-4, vmax=10), cmap=colrmap)
            # axs[strati,beami].set_xlim(kprll_sb[subband_index,:].min(), kprll_sb[subband_index,:].max())
            axs[strati].axvline(x=fgdps_achrmbeam.horizon_kprll_limits[0,0,0], ls='-', lw=2, color='gray')
            axs[strati].axvline(x=fgdps_achrmbeam.horizon_kprll_limits[0,0,1], ls='-', lw=2, color='gray')        
            axs[strati].set_xlim(-0.35, 0.35)
            axs[strati].set_ylim(0, 24)
            axs[strati].set_aspect('auto')
    
        cbax = fig.add_axes([0.91, 0.14, 0.02, 0.81])
        cbar = fig.colorbar(ratio2D, cax=cbax, orientation='vertical')
        cbax.yaxis.set_label_position('right')    
    
        fig.subplots_adjust(hspace=0, wspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel('RA [hours]', fontsize=16, weight='medium', labelpad=20)
        big_ax.set_xlabel(r'$k_\parallel$'+' [h Mpc'+r'$^{-1}$]', fontsize=16, weight='medium', labelpad=15)
    
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.subplots_adjust(top=0.95) 
        fig.subplots_adjust(bottom=0.15)   
        fig.subplots_adjust(left=0.08)
    
        PLT.savefig(figuresdir+'eor_fg_ratio_{0}_{1}_bl{2}_{3:.1f}_MHz_subband.png'.format(mdl_str,beamtype,bl_index,freq_window_centers[strategy][subband_index]/1e6), bbox_inches=0)
        PLT.savefig(figuresdir+'eor_fg_ratio_{0}_{1}_bl{2}_{3:.1f}_MHz_subband.eps'.format(mdl_str,beamtype,bl_index,freq_window_centers[strategy][subband_index]/1e6), bbox_inches=0)

    if '3b' in plots:
       
        # 03-b) Plot signal-to-foreground ratio summed in quadrature over k_parallel as a function of LST and baseline
            
        mdl_str = ['21cmfast', 'lidz']
        beamtype = ['chrmbeam', 'achrmbeam', 'funcbeam']
        strategy = ['sim', 'cc']
        subband_index = range(len(freq_window_centers['sim']))
        eor_fg_ratio_sb_netkprll = {}

        for mdl in mdl_str:
            eor_fg_ratio_sb_netkprll[mdl] = {}
            for bmtype in beamtype:
                eor_fg_ratio_sb_netkprll[mdl][bmtype] = {}
                for strtgy in strategy:
                    eor_fg_ratio_sb_netkprll[mdl][bmtype][strtgy] = NP.sqrt(NP.nansum(eor_fg_ratio_sb[mdl][bmtype][strtgy]**2, axis=2))

        for mdl in mdl_str:
            for bmtype in beamtype:
                for strtgy in strategy:
                    for sbi in subband_index:
                        fig = PLT.figure(figsize=(5,4))
                        ax = fig.add_subplot(111)
                        ratio2D = ax.imshow(eor_fg_ratio_sb_netkprll[mdl][bmtype][strtgy][:,sbi,:].T, origin='lower', extent=[0,bl_length.size,0,24], norm=PLTC.LogNorm(1e-2,eor_fg_ratio_sb_netkprll[mdl][bmtype][strtgy].max()), interpolation='none')
                        ax.set_xlim(0, bl_length.size)
                        for bli in range(bl_length.size):
                            ax.text(bli+1, -0.5, '{0:.1f}'.format(bl_length[bli]), transform=ax.transData, fontsize=8, weight='medium', ha='right', va='top', rotation=90, color='black')
                            ax.text(bli+1, 24.5, '{0:.1f}'.format(blo[bli]), transform=ax.transData, fontsize=8, weight='medium', ha='right', va='bottom', rotation=90, color='black')
                        ax.set_xlabel('Baseline length [m]', weight='medium', fontsize=14, labelpad=25)
                        ax.set_aspect('auto')
                        axt = ax.twiny()
                        axt.set_xlim(0, bl_length.size)
                        PLT.setp(ax.get_xticklabels(), visible=False)
                        PLT.setp(axt.get_xticklabels(), visible=False)
                        ax.set_ylabel('RA [hours]', weight='medium', fontsize=14)
                        axt.set_xlabel('Baseline Orientation [deg]', weight='medium', fontsize=14, labelpad=33)
                        cbaxr = fig.add_axes([0.88, 0.18, 0.02, 0.6])
                        cbarr = fig.colorbar(ratio2D, cax=cbaxr, orientation='vertical')
                        
                        fig.subplots_adjust(right=0.87)
                        fig.subplots_adjust(top=0.81) 
                        fig.subplots_adjust(bottom=0.16)   
                        fig.subplots_adjust(left=0.12)

                        PLT.savefig(figuresdir+'eor_fg_ratio_quadrature_sum_{0}_{1}_{2}_{3:.1f}_MHz_subband.png'.format(mdl,bmtype,strtgy,freq_window_centers[strtgy][sbi]/1e6), bbox_inches=0)
                        PLT.savefig(figuresdir+'eor_fg_ratio_quadrature_sum_{0}_{1}_{2}_{3:.1f}_MHz_subband.png'.format(mdl,bmtype,strtgy,freq_window_centers[strtgy][sbi]/1e6), bbox_inches=0)
                        PLT.close()

    if '3c' in plots:

        # 03-c) Plot signal-to-foreground ratio lower limit over k_parallel as a function of LST and baseline

        mdl_str = ['21cmfast', 'lidz']
        beamtype = ['chrmbeam', 'achrmbeam', 'funcbeam']
        strategy = ['sim']
        subband_index = range(len(freq_window_centers['sim']))
        kprll_buffer = parms['plot']['3c']['kprll_buffer']
        eor_fg_ratio_sb_kprll_masked = {}
        eor_fg_ratio_sb_kprll_llim = {}

        eta2kprll_sb = DS.dkprll_deta(fgdps_achrmbeam.subband_delay_power_spectra['sim']['z'], fgdps_achrmbeam.cosmo)
        kprll_bw_sb = eta2kprll_sb * (1 / fgds_achrmbeam.subband_delay_spectra['sim']['bw_eff'])
        kprll_sb_horizon = fgdps_achrmbeam.subband_delay_power_spectra_resampled['sim']['horizon_kprll_limits'][:,:,:,1]
        kprll_sb_llim = kprll_sb_horizon + kprll_buffer * kprll_bw_sb[NP.newaxis,:,NP.newaxis]
        kprll_sb_llim_reax = NP.swapaxes(kprll_sb_llim, 0, 2)
        kprll_sb_llim_reax = kprll_sb_llim_reax[:,:,NP.newaxis,:]

        kprll_sb_reax = kprll_sb[NP.newaxis,:,:,NP.newaxis]
        kprll_mask = NP.abs(kprll_sb_reax) <= kprll_sb_llim_reax

        for mdl in mdl_str:
            eor_fg_ratio_sb_kprll_masked[mdl] = {}
            eor_fg_ratio_sb_kprll_llim[mdl] = {}
            for bmtype in beamtype:
                eor_fg_ratio_sb_kprll_masked[mdl][bmtype] = {}
                eor_fg_ratio_sb_kprll_llim[mdl][bmtype] = {}
                for strtgy in strategy:
                    eor_fg_ratio_sb_kprll_masked[mdl][bmtype][strtgy] = NP.ma.array(eor_fg_ratio_sb[mdl][bmtype][strtgy], mask=kprll_mask)
                    eor_fg_ratio_sb_kprll_llim[mdl][bmtype][strtgy] = NP.nanmin(eor_fg_ratio_sb_kprll_masked[mdl][bmtype][strtgy], axis=2)

        for mdl in mdl_str:
            for bmtype in beamtype:
                for strtgy in strategy:
                    for sbi in subband_index:
                        fig = PLT.figure(figsize=(5,4))
                        ax = fig.add_subplot(111)
                        ratio2D = ax.imshow(eor_fg_ratio_sb_kprll_llim[mdl][bmtype][strtgy][:,sbi,:].T, origin='lower', extent=[0,bl_length.size,0,24], norm=PLTC.LogNorm(1e-2,1e3), interpolation='none')
                        # ratio2D = ax.imshow(eor_fg_ratio_sb_kprll_llim[mdl][bmtype][strtgy][:,sbi,:].T, origin='lower', extent=[0,bl_length.size,0,24], norm=PLTC.LogNorm(eor_fg_ratio_sb_kprll_llim[mdl][bmtype][strtgy].min(),eor_fg_ratio_sb_kprll_llim[mdl][bmtype][strtgy].max()), interpolation='none')
                        ax.set_xlim(0, bl_length.size)
                        for bli in range(bl_length.size):
                            ax.text(bli+1, -0.5, '{0:.1f}'.format(bl_length[bli]), transform=ax.transData, fontsize=8, weight='medium', ha='right', va='top', rotation=90, color='black')
                            ax.text(bli+1, 24.5, '{0:.1f}'.format(blo[bli]), transform=ax.transData, fontsize=8, weight='medium', ha='right', va='bottom', rotation=90, color='black')
                        ax.set_xlabel('Baseline length [m]', weight='medium', fontsize=14, labelpad=25)
                        ax.set_aspect('auto')
                        axt = ax.twiny()
                        axt.set_xlim(0, bl_length.size)
                        PLT.setp(ax.get_xticklabels(), visible=False)
                        PLT.setp(axt.get_xticklabels(), visible=False)
                        ax.set_ylabel('RA [hours]', weight='medium', fontsize=14)
                        axt.set_xlabel('Baseline Orientation [deg]', weight='medium', fontsize=14, labelpad=33)
                        cbaxr = fig.add_axes([0.88, 0.18, 0.02, 0.6])
                        cbarr = fig.colorbar(ratio2D, cax=cbaxr, orientation='vertical')

                        fig.subplots_adjust(right=0.87)
                        fig.subplots_adjust(top=0.81)
                        fig.subplots_adjust(bottom=0.16)
                        fig.subplots_adjust(left=0.12)

                        PLT.savefig(figuresdir+'eor_fg_ratio_llim_{0}_{1}_{2}_{3:.1f}_MHz_subband.png'.format(mdl,bmtype,strtgy,freq_window_centers[strtgy][sbi]/1e6), dpi=600, bbox_inches=0)
                        PLT.savefig(figuresdir+'eor_fg_ratio_llim_{0}_{1}_{2}_{3:.1f}_MHz_subband.eps'.format(mdl,bmtype,strtgy,freq_window_centers[strtgy][sbi]/1e6), dpi=600, bbox_inches=0)

                        PLT.close()
if '4a' in plots:
    # 04-a) Plot Blackman-Harris window and its modified version based on fftpow and their responses in delay space

    subband_freq = parms['plot']['4a']['sbfreq']
    subband_index = NP.argmin(NP.abs(freq_window_centers['sim'] - subband_freq))

    fullband_freq_wts = fgds_achrmbeam.bp_wts[0,:,0]
    fullband_freq_wts /= fullband_freq_wts.max()
    # subband_freq_wts = fgds_achrmbeam.subband_delay_spectra['sim']['freq_wts'][subband_index,:]
    # subband_freq_wts /= subband_freq_wts.max()

    frac_width = DSP.window_N2width(n_window=None, shape=bpass_shape, fftpow=1.0, area_normalize=False, power_normalize=True)
    n_window = NP.round(nchan / frac_width).astype(NP.int)
    fullband_freq_wts_1 = DSP.window_fftpow(nchan, shape=bpass_shape, fftpow=1.0, pad_width=0, centering=True, area_normalize=False, power_normalize=True)
    fullband_freq_wts_1 /= fullband_freq_wts_1.max()
    # subband_freq_wts_1 = fgds_achrmbeam1.subband_delay_spectra['sim']['freq_wts'][subband_index,:]
    # subband_freq_wts_1 /= subband_freq_wts_1.max()

    fullband_delay_response = DSP.FT1D(fullband_freq_wts, ax=-1, inverse=True, use_real=False, shift=True) * fgds_achrmbeam.f.size * fgds_achrmbeam.df
    fullband_delay_response_1 = DSP.FT1D(fullband_freq_wts_1, ax=-1, inverse=True, use_real=False, shift=True) * fgds_achrmbeam1.f.size * fgds_achrmbeam1.df
    # subband_delay_response = DSP.FT1D(subband_freq_wts, ax=-1, inverse=True, use_real=False, shift=True) * fgds_achrmbeam.f.size * fgds_achrmbeam.df
    # subband_delay_response_1 = DSP.FT1D(subband_freq_wts_1, ax=-1, inverse=True, use_real=False, shift=True) * fgds_achrmbeam1.f.size * fgds_achrmbeam1.df

    lags = DSP.spectral_axis(fgds_achrmbeam.f.size, delx=fgds_achrmbeam.df, shift=True)
    fig, axs = PLT.subplots(nrows=2, figsize=(3.5,7))
    axs[0].plot(fgds_achrmbeam1.f/1e6, fullband_freq_wts_1, ls='-', lw=2, color='gray', label='BH')
    axs[0].plot(fgds_achrmbeam.f/1e6, fullband_freq_wts, ls='-', lw=2, color='black', label=r'BH '+r'$\ast$'+' BH')
    axs[0].set_xlim(100,200)
    axs[0].set_ylim(5e-9, 1.1)
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'$f$'+' [MHz]', fontsize=12, weight='medium')
    axs[0].set_ylabel(r'$W(f)$', fontsize=12, weight='medium')
    lgnd0 = axs[0].legend(frameon=True, fontsize=8, loc='center', bbox_to_anchor=(0.5,0.5))

    axs[1].plot(1e9 * lags, NP.abs(fullband_delay_response_1)**2, ls='-', lw=2, color='gray', label='BH')
    axs[1].plot(1e9 * lags, NP.abs(fullband_delay_response)**2, ls='-', lw=2, color='black', label='BH '+r'$\ast$'+' BH')
    axs[1].set_xlim(-490, 490)
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'$\tau$'+' [ns]', fontsize=12, weight='medium')
    axs[1].set_ylabel('|'+r'$\widetilde{W}(\tau)$'+'|'+r'$^2$'+' [Hz'+r'$^2$'+']', fontsize=12, weight='medium', labelpad=-5)
    lgnd1 = axs[1].legend(frameon=True, fontsize=8, loc='upper right', bbox_to_anchor=(0.999,0.999))

    fig.subplots_adjust(right=0.93)
    fig.subplots_adjust(left=0.22)
    fig.subplots_adjust(top=0.97)
    fig.subplots_adjust(bottom=0.08)

    PLT.savefig(figuresdir+'window_function_modifications.png', dpi=600, bbox_inches=0)
    PLT.savefig(figuresdir+'window_function_modifications.eps', dpi=600, bbox_inches=0)

PDB.set_trace()
