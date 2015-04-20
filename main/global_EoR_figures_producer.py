import numpy as NP
import astropy.cosmology as CP
import scipy.constants as FCNST
import argparse
import yaml
import astropy
from astropy.io import fits, ascii
import progressbar as PGB
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
from matplotlib.ticker import FuncFormatter
import geometry as GEOM
import interferometry as RI
import catalog as SM
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import ipdb as PDB

parser = argparse.ArgumentParser(description='Program to analyze and plot global EoR data')

input_group = parser.add_argument_group('Input parameters', 'Input specifications')
input_group.add_argument('-i', '--infile', dest='infile', default='/home/t_nithyanandan/codes/mine/python/interferometry/main/simparameters.yaml', type=file, required=False, help='File specifying input parameters')

args = vars(parser.parse_args())

rootdir = '/data3/t_nithyanandan/'

with args['infile'] as parms_file:
    parms = yaml.safe_load(parms_file)

project = parms['project']
telescope_id = parms['telescope']['id']
Tsys = parms['telescope']['Tsys']
latitude = parms['telescope']['latitude']
pfb_method = parms['telescope']['pfb_method']
element_shape = parms['antenna']['shape']
element_size = parms['antenna']['size']
element_ocoords = parms['antenna']['ocoords']
element_orientation = parms['antenna']['orientation']
ground_plane = parms['antenna']['ground_plane']
phased_array = parms['antenna']['phased_array']
phased_elements_file = parms['phasedarray']['file']
delayerr = parms['phasedarray']['delayerr']
gainerr = parms['phasedarray']['gainerr']
nrand = parms['phasedarray']['nrand']
antenna_file = parms['array']['file']
array_layout = parms['array']['layout']
minR = parms['array']['minR']
maxR = parms['array']['maxR']
minbl = parms['baseline']['min']
maxbl = parms['baseline']['max']
bldirection = parms['baseline']['direction']
obs_mode = parms['obsparm']['obs_mode']
n_snaps = parms['obsparm']['n_snaps']
t_snap = parms['obsparm']['t_snap']
t_obs = parms['obsparm']['t_obs']
freq = parms['obsparm']['freq']
freq_resolution = parms['obsparm']['freq_resolution']
nchan = parms['obsparm']['nchan']
avg_drifts = parms['snapshot']['avg_drifts']
beam_switch = parms['snapshot']['beam_switch']
pick_snapshots = parms['snapshot']['pick']
all_snapshots = parms['snapshot']['all']
snapshots_range = parms['snapshot']['range']
pointing_file = parms['pointing']['file']
pointing_info = parms['pointing']['initial']
n_bins_baseline_orientation = parms['processing']['n_bins_blo']
baseline_chunk_size = parms['processing']['bl_chunk_size']
bl_chunk = parms['processing']['bl_chunk']
n_bl_chunks = parms['processing']['n_bl_chunks']
n_sky_sectors = parms['processing']['n_sky_sectors']
bpass_shape = parms['processing']['bpass_shape']
max_abs_delay = parms['processing']['max_abs_delay']
fg_str = parms['fgparm']['model']
nside = parms['fgparm']['nside']
spindex_rms = parms['fgparm']['spindex_rms']
spindex_seed = parms['fgparm']['spindex_seed']
pc = parms['phasing']['center']
pc_coords = parms['phasing']['coords']

if project not in ['project_MWA', 'project_global_EoR', 'project_HERA', 'project_drift_scan', 'project_beams', 'project_LSTbin']:
    raise ValueError('Invalid project specified')
else:
    project_dir = project + '/'

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

if element_ocoords not in ['altaz', 'dircos']:
    if element_ocoords is not None:
        raise ValueError('Antenna element orientation must be "altaz" or "dircos"')

if element_orientation is None:
    if element_ocoords == 'altaz':
        element_orientation = NP.asarray([0.0, 90.0])
    elif element_ocoords == 'dircos':
        element_orientation = NP.asarray([1.0, 0.0, 0.0])
else:
    element_orientation = NP.asarray(element_orientation)

if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

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
    elif array_layout == 'CIRC':
        ant_locs, ant_id = RI.circular_antenna_array(element_size, minR, maxR=maxR)

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane

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

if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

if obs_mode is None:
    obs_mode = 'custom'
elif obs_mode not in ['drift', 'track', 'dns']:
    raise ValueError('Invalid observing mode specified')

if avg_drifts + beam_switch + (pick_snapshots is not None) + (snapshots_range is not None) + all_snapshots != 1:
    raise ValueError('One and only one of avg_drifts, beam_switch, pick_snapshots, snapshots_range, all_snapshots must be set')

snapshot_type_str = ''
if avg_drifts and (obs_mode == 'dns'):
    snapshot_type_str = 'drift_averaged_'

if beam_switch and (obs_mode == 'dns'):
    snapshot_type_str = 'beam_switches_'

if (snapshots_range is not None) and ((obs_mode == 'dns') or (obs_mode == 'lstbin')):
    snapshot_type_str = 'snaps_{0[0]:0d}-{0[1]:0d}_'.format(snapshots_range)

if (pointing_file is None) and (pointing_info is None):
    raise ValueError('One and only one of pointing file and initial pointing must be specified')
elif (pointing_file is not None) and (pointing_info is not None):
    raise ValueError('One and only one of pointing file and initial pointing must be specified')

duration_str = ''
if obs_mode in ['track', 'drift']:
    if (t_snap is not None) and (n_snaps is not None):
        duration_str = '_{0:0d}x{1:.1f}s'.format(n_snaps, t_snap)
        geor_duration_str = '_{0:0d}x{1:.1f}s'.format(1, t_snap)    

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
    geor_duration_str = '_{0:0d}x{1:.1f}s'.format(1, t_snap[0])    

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
neg_bl_orientation_ind = (bl_orientation < -67.5) | (bl_orientation > 112.5)
# neg_bl_orientation_ind = NP.logical_or(bl_orientation < -0.5*180.0/n_bins_baseline_orientation, bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)

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
bl_id = bl_id[select_bl_ind]
bl = bl[select_bl_ind,:]
bl_length = bl_length[select_bl_ind]
bl_orientation = bl_orientation[select_bl_ind]

total_baselines = bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

bllstr = map(str, bl_length)
uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
count_uniq_bll = NP.asarray(count_uniq_bll)

geor_bl = bl[ind_uniq_bll,:]
geor_bl_id = bl_id[ind_uniq_bll]
geor_bl_orientation = bl_orientation[ind_uniq_bll]
geor_bl_length = bl_length[ind_uniq_bll]

sortind = NP.argsort(geor_bl_length, kind='mergesort')
geor_bl = geor_bl[sortind,:]
geor_bl_id = geor_bl_id[sortind]
geor_bl_length = geor_bl_length[sortind]
geor_bl_orientation = geor_bl_orientation[sortind]
count_uniq_bll = count_uniq_bll[sortind]

use_GSM = False
use_DSM = False
use_CSM = False
use_SUMSS = False
use_GLEAM = False
use_USM = False
use_NVSS = False
use_HI_monopole = False
use_HI_cube = False
use_HI_fluctuations = False

if fg_str not in ['asm', 'dsm', 'csm', 'nvss', 'sumss', 'gleam', 'mwacs', 'ps', 'usm', 'mss', 'HI_cube', 'HI_monopole', 'HI_fluctuations']:
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
elif fg_str == 'point':
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

if n_sky_sectors == 1:
    sky_sector_str = '_all_sky_'

freq = NP.float(freq)
freq_resolution = NP.float(freq_resolution)
wavelength = FCNST.c / freq  # in meters
redshift = CNST.rest_freq_HI / freq - 1
bw = nchan * freq_resolution
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)
if bpass_shape not in ['bnw', 'bhw', 'rect']:
    raise ValueError('Invalid bandpass shape specified')

if pc_coords not in ['altaz', 'radec', 'hadec', 'dircos']:
    raise ValueError('Invalid coordinate system specified for phase center')
else:
    pc = NP.asarray(pc).ravel()
    if pc_coords == 'radec':
        if pc.size != 2:
            raise ValueError('Phase center must be a 2-element vector')
        pc_hadec = NP.hstack((lst.reshape(-1,1)-pc[0], pc[1]+NP.zeros((lst.size,1))))
        pc_altaz = GEOM.hadec2altaz(pc_hadec, latitude, units='degrees')
        pc_dircos = GEOM.altaz2dircos(pc_altaz, units='degrees')
    elif pc_coords == 'hadec':
        if pc.size != 2:
            raise ValueError('Phase center must be a 2-element vector')
        pc_altaz = GEOM.hadec2altaz(pc.reshape(1,-1), latitude, units='degrees')
        pc_dircos = GEOM.altaz2dircos(pc_altaz, units='degrees')
    elif pc_coords == 'altaz':
        if pc.size != 2:
            raise ValueError('Phase center must be a 2-element vector')
        pc_dircos = GEOM.altaz2dircos(pc.reshape(1,-1), units='degrees')
    else:
        if pc.size != 3:
            raise ValueError('Phase center must be a 3-element vector in dircos coordinates')
        pc_coords = NP.asarray(pc).reshape(1,-1)

if pfb_method is not None:
    use_pfb = True
else:
    use_pfb = False

h = 0.7   # Hubble constant coefficient
cosmodel100 = CP.FlatLambdaCDM(H0=100.0, Om0=0.27)  # Using H0 = 100 km/s/Mpc
cosmodel = CP.FlatLambdaCDM(H0=h*100.0, Om0=0.27)  # Using H0 = h * 100 km/s/Mpc

def kprll(eta, z):
    return 2 * NP.pi * eta * cosmodel100.H0.value * CNST.rest_freq_HI * cosmodel100.efunc(z) / FCNST.c / (1+z)**2 * 1e3

def kperp(u, z):
    return 2 * NP.pi * u / cosmodel100.comoving_transverse_distance(z).value

geor_infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+geor_duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'HI_monopole'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+'no_pfb.fits'

fg_infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+'no_pfb.fits'

geor_clean_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+geor_duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'HI_monopole'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+'no_pfb_'+bpass_shape+'.fits'

fg_clean_infile = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+'no_pfb_'+bpass_shape+'.fits'                

PDB.set_trace()
ia = RI.InterferometerArray(None, None, None, init_file=fg_infile)

hdulist = fits.open(geor_clean_infile)
clean_lags = hdulist['SPECTRAL INFO'].data['lag']
geor_cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data
geor_cc_skyvis_lag_res = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data
hdulist.close()

hdulist = fits.open(fg_clean_infile)
fg_cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data
fg_cc_skyvis_lag_res = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data
hdulist.close()

geor_cc_skyvis_lag += geor_cc_skyvis_lag_res
fg_cc_skyvis_lag += fg_cc_skyvis_lag_res

geor_cc_skyvis_lag = DSP.downsampler(geor_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
fg_cc_skyvis_lag = DSP.downsampler(fg_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
fg_cc_skyvis_lag_res = DSP.downsampler(fg_cc_skyvis_lag_res, 1.0*clean_lags.size/ia.lags.size, axis=1)

clean_lags_orig = NP.copy(clean_lags)
clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
clean_lags = clean_lags.ravel()

delaymat = DLY.delay_envelope(bl, pc_dircos, units='mks')
min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
max_delay = delaymat[0,:,0]-delaymat[0,:,1]
clags = clean_lags.reshape(1,-1)
min_delay = min_delay.reshape(-1,1)
max_delay = max_delay.reshape(-1,1)
thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
thermal_noise_window = NP.repeat(thermal_noise_window, bl.shape[0], axis=0)
EoR_window = NP.logical_or(clags > max_delay+1/bw, clags < min_delay-1/bw)
wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
non_wedge_window = NP.logical_not(wedge_window)

bll_bin_count, bll_edges, bll_binnum, bll_ri = OPS.binned_statistic(bl_length, values=None, statistic='count', bins=NP.hstack((geor_bl_length-1e-10, geor_bl_length.max()+1e-10)))

snap_min = 0
snap_max = 39
fg_cc_skyvis_lag_tavg = NP.mean(fg_cc_skyvis_lag[:,:,snap_min:snap_max+1], axis=2)
fg_cc_skyvis_lag_res_tavg = NP.mean(fg_cc_skyvis_lag_res[:,:,snap_min:snap_max+1], axis=2)

fg_cc_skyvis_lag_blavg = NP.zeros((geor_bl_length.size, clags.size, snap_max-snap_min+1), dtype=NP.complex64)
fg_cc_skyvis_lag_res_blavg = NP.zeros((geor_bl_length.size, clags.size, snap_max-snap_min+1), dtype=NP.complex64)
for i in xrange(geor_bl_length.size):
    blind = bll_ri[bll_ri[i]:bll_ri[i+1]]
    if blind.size != bll_bin_count[i]: PDB.set_trace()
    fg_cc_skyvis_lag_blavg[i,:,:] = NP.mean(fg_cc_skyvis_lag[blind,:,snap_min:snap_max+1], axis=0)
    fg_cc_skyvis_lag_res_blavg[i,:,:] = NP.mean(fg_cc_skyvis_lag_res[blind,:,snap_min:snap_max+1], axis=0)
fg_cc_skyvis_lag_avg = NP.mean(fg_cc_skyvis_lag_blavg, axis=2)
fg_cc_skyvis_lag_res_avg = NP.mean(fg_cc_skyvis_lag_res_blavg, axis=2)

for i in xrange(int(NP.ceil(geor_bl_length.size/4.0))):
    fig, axs = PLT.subplots(min(4,geor_bl_length.size-4*i), sharex=True, figsize=(6,9))
    for j in range(4*i, min(4*(i+1),geor_bl_length.size)):
        blind = bll_ri[bll_ri[j]:bll_ri[j+1]]
        axs[j%len(axs)].plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag[blind[0],:,0]), ls='--', lw=2, color='black')
        axs[j%len(axs)].plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_blavg[j,:,0]), ls='-.', lw=2, color='black')
        axs[j%len(axs)].plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_tavg[blind[0],:]), ls=':', lw=2, color='black')
        axs[j%len(axs)].plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_avg[j,:]), ls='-', lw=2, color='black')
        axs[j%len(axs)].plot(1e6*clags.ravel(), NP.abs(geor_cc_skyvis_lag[j,:,0]), ls='-', lw=2, color='gray')
        axs[j%len(axs)].plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_res_avg[j,:]), ls=':', lw=2, color='red')
        axs[j%len(axs)].axvline(x=1e6*min_delay[blind[0],0], ls=':', lw=2, color='gray')
        axs[j%len(axs)].axvline(x=1e6*max_delay[blind[0],0], ls=':', lw=2, color='gray')
        axs[j%len(axs)].text(0.05, 0.8, r'$|\mathbf{b}|$'+' = {0:.1f} m'.format(geor_bl_length[j]), fontsize=12, weight='medium', transform=axs[j%len(axs)].transAxes)        
        axs[j%len(axs)].set_ylim(NP.abs(geor_cc_skyvis_lag).min(), NP.abs(fg_cc_skyvis_lag[:,:,snap_min:snap_max+1]).max())
        axs[j%len(axs)].set_xlim(1e6*clags.min(), 1e6*clags.max())
        axs[j%len(axs)].set_yscale('log')
        axs[j%len(axs)].set_yticks(NP.logspace(4,12,5,endpoint=True).tolist())

        if j%len(axs) == len(axs)-1:
            axs[j%len(axs)].set_xlabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
        if j%len(axs) == 0:
            axs_kprll = axs[j%len(axs)].twiny()
            axs_kprll.set_xticks(kprll(axs[j%len(axs)].get_xticks()*1e-6, redshift))
            axs_kprll.set_xlim(kprll(NP.asarray(axs[j%len(axs)].get_xlim())*1e-6, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
            axs_kprll.xaxis.set_major_formatter(xformatter)
            axs_kprll.xaxis.tick_top()
            axs_kprll.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
            axs_kprll.xaxis.set_label_position('top') 

    fig.subplots_adjust(hspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel(r"$|V_b|$ [Jy Hz]", fontsize=16, weight='medium', labelpad=30)

    PLT.savefig(rootdir+project_dir+'figures/'+telescope_str+'delay_spectra_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(geor_bl_length[4*i],geor_bl_length[j])+fg_str+'_nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+'no_pfb_'+bpass_shape+'.png', bbox_inches=0)





# fig = PLT.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# ax.plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag[-1,:,0]), ls='--', lw=2, color='black')
# ax.plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_blavg[-1,:,0]), ls='-.', lw=2, color='black')
# ax.plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_tavg[-1,:]), ls=':', lw=2, color='black')
# ax.plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_avg[-1,:]), ls='-', lw=2, color='black')
# ax.plot(1e6*clags.ravel(), NP.abs(geor_cc_skyvis_lag[-1,:,0]), ls='-', lw=2, color='gray')
# ax.plot(1e6*clags.ravel(), NP.abs(fg_cc_skyvis_lag_res_avg[-1,:]), ls='-', lw=2, color='red')
# ax.set_ylim(NP.abs(geor_cc_skyvis_lag).min(), NP.abs(fg_cc_skyvis_lag).max())
# ax.set_xlim(1e6*clags.min(), 1e6*clags.max())
# ax.set_yscale('log')
# ax.set_xlabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
# ax.set_ylabel(r'$|V_b|$'+' [Jy Hz]', fontsize=16, weight='medium')
# PLT.show()

PDB.set_trace()
