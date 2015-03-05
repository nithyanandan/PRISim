import argparse
import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import progressbar as PGB
import healpy as HP
import geometry as GEOM
import interferometry as RI
import catalog as SM
import constants as CNST
import my_operations as OPS
import primary_beams as PB
import ipdb as PDB

def Jy2K(fluxJy, freq, pixres):
    return fluxJy * CNST.Jy / pixres / (2.0* FCNST.k * (freq)**2 / FCNST.c**2)

def K2Jy(tempK, freq, pixres):
    return tempK * (2.0* FCNST.k * (freq)**2 / FCNST.c**2) * pixres / CNST.Jy

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')

project_group = parser.add_mutually_exclusive_group(required=True)
project_group.add_argument('--project-MWA', dest='project_MWA', action='store_true')
project_group.add_argument('--project-HERA', dest='project_HERA', action='store_true')
project_group.add_argument('--project-beams', dest='project_beams', action='store_true')
project_group.add_argument('--project-drift-scan', dest='project_drift_scan', action='store_true')
project_group.add_argument('--project-global-EoR', dest='project_global_EoR', action='store_true')

telescope_group = parser.add_argument_group('Telescope parameters', 'Telescope/interferometer specifications')
telescope_group.add_argument('--label-prefix', help='Prefix for baseline labels [str, Default = ""]', default='', type=str, dest='label_prefix')
telescope_group.add_argument('--telescope', help='Telescope name [str, default="custom"]', default='custom', type=str, dest='telescope_id', choices=['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'paper_dipole', 'custom', 'mwa_tools'])
telescope_group.add_argument('--latitude', help='Latitude of interferometer array in degrees [float, Default=-26.701]', default=-26.701, type=float, dest='latitude')
telescope_group.add_argument('--A-eff', help='Effective area in m^2', type=float, dest='A_eff', nargs='?')

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
obsparm_group.add_argument('--obs-mode', help='Observing mode [str, track/drift/drift-shift/custom]', default=None, type=str, dest='obs_mode', choices=['track', 'drift', 'dns', 'custom'])
# obsparm_group.add_argument('--t-snap', help='Integration time (seconds) [float, Default=300.0]', default=5.0*60.0, type=float, dest='t_snap')
obsparm_group.add_argument('--nchan', help='Number of frequency channels [int, Default=256]', default=256, type=int, dest='n_channels')

duration_group = parser.add_argument_group('Observing duration parameters', 'Parameters specifying observing duration')
duration_group.add_argument('--t-obs', help='Duration of observation [seconds]', dest='t_obs', default=None, type=float, metavar='t_obs')
duration_group.add_argument('--n-snap', help='Number of snapshots or records that make up the observation', dest='n_snaps', default=None, type=int, metavar='n_snapshots')
duration_group.add_argument('--t-snap', help='integration time of each snapshot [seconds]', dest='t_snap', default=None, type=int, metavar='t_snap')

pointing_group = parser.add_mutually_exclusive_group(required=True)
pointing_group.add_argument('--pointing-file', dest='pointing_file', type=str, nargs=1, default=None)
pointing_group.add_argument('--pointing-info', dest='pointing_info', type=float, nargs=3, metavar=('lst_init', 'ra_init', 'dec_init'))

snapshot_selection_group = parser.add_mutually_exclusive_group(required=False)
snapshot_selection_group.add_argument('--beam-switch', dest='beam_switch', action='store_true')
snapshot_selection_group.add_argument('--snap-pick', dest='pick_snapshots', default=None, type=int, nargs='*')
snapshot_selection_group.add_argument('--snap-range', dest='snapshots_range', default=None, nargs=2, type=int)
snapshot_selection_group.add_argument('--all-snaps', dest='all_snapshots', action='store_true')

fgmodel_group = parser.add_mutually_exclusive_group(required=True)
fgmodel_group.add_argument('--ASM', action='store_true') # Diffuse (GSM) + Compact (NVSS+SUMSS) All-sky model 
fgmodel_group.add_argument('--DSM', action='store_true') # Diffuse all-sky model
fgmodel_group.add_argument('--CSM', action='store_true') # Point source model (NVSS+SUMSS)
fgmodel_group.add_argument('--SUMSS', action='store_true') # SUMSS catalog
fgmodel_group.add_argument('--NVSS', action='store_true') # NVSS catalog
fgmodel_group.add_argument('--MSS', action='store_true') # Molonglo Sky Survey
fgmodel_group.add_argument('--GLEAM', action='store_true') # GLEAM catalog
fgmodel_group.add_argument('--PS', action='store_true') # Point sources 
fgmodel_group.add_argument('--USM', action='store_true') # Uniform all-sky model

fgparm_group = parser.add_argument_group('Foreground Setup', 'Parameters describing foreground sky')
fgparm_group.add_argument('--flux-unit', help='Units of flux density [str, Default="Jy"]', type=str, dest='flux_unit', default='Jy', choices=['Jy','K'])
fgparm_group.add_argument('--spindex', help='Spectral index, ~ f^spindex [float, Default=0.0]', type=float, dest='spindex', default=0.0)
fgparm_group.add_argument('--spindex-rms', help='Spectral index rms [float, Default=0.0]', type=float, dest='spindex_rms', default=0.0)
fgparm_group.add_argument('--spindex-seed', help='Spectral index seed [float, Default=None]', type=int, dest='spindex_seed', default=None)
fgparm_group.add_argument('--nside', help='nside parameter for healpix map [int, Default=64]', type=int, dest='nside', default=64, choices=[64, 128])

fgcat_group = parser.add_argument_group('Catalog files', 'Catalog file locations')
fgcat_group.add_argument('--dsm-file-prefix', help='Diffuse sky model filename prefix [str]', type=str, dest='DSM_file_prefix', default='/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata')
fgcat_group.add_argument('--sumss-file', help='SUMSS catalog file [str]', type=str, dest='SUMSS_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt')
fgcat_group.add_argument('--nvss-file', help='NVSS catalog file [str]', type=file, dest='NVSS_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits')
fgcat_group.add_argument('--GLEAM-file', help='GLEAM catalog file [str]', type=str, dest='GLEAM_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv')
fgcat_group.add_argument('--PS-file', help='Point source catalog file [str]', type=str, dest='PS_file', default='/data3/t_nithyanandan/project_MWA/foregrounds/PS_catalog.txt')
# parser.add_argument('--', help='', type=, dest='', required=True)

parser.add_argument('--plots', help='Create plots', action='store_true', dest='plots')

args = vars(parser.parse_args())

project_MWA = args['project_MWA']
project_HERA = args['project_HERA']
project_beams = args['project_beams']
project_drift_scan = args['project_drift_scan']
project_global_EoR = args['project_global_EoR']

if project_MWA: project_dir = 'project_MWA'
if project_HERA: project_dir = 'project_HERA'
if project_beams: project_dir = 'project_beams'
if project_drift_scan: project_dir = 'project_drift_scan'
if project_global_EoR: project_dir = 'project_global_EoR'

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

latitude = args['latitude']
latitude_str = 'lat_{0:.3f}_'.format(latitude)

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane
telescope['latitude'] = latitude

freq = args['freq']
freq_resolution = args['freq_resolution']
n_channels = args['n_channels']
nchan = n_channels
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution)/ 1e9 # in GHz

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
t_snap = args['t_snap']
t_obs = args['t_obs']
n_snaps = args['n_snaps']

snapshot_type_str = obs_mode+'_'

pointing_file = args['pointing_file']
if pointing_file is not None:
    pointing_file = pointing_file[0]
pointing_info = args['pointing_info']

element_locs = None
if phased_array:
    try:
        element_locs = NP.loadtxt(phased_elements_file, skiprows=1, comments='#', usecols=(0,1,2))
    except IOError:
        raise IOError('Could not open the specified file for phased array of antenna elements.')

if telescope_id == 'mwa':
    xlocs, ylocs = NP.meshgrid(1.1*NP.linspace(-1.5,1.5,4), 1.1*NP.linspace(1.5,-1.5,4))
    element_locs = NP.hstack((xlocs.reshape(-1,1), ylocs.reshape(-1,1), NP.zeros(xlocs.size).reshape(-1,1)))

if pointing_file is not None:
    pointing_init = None
    pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(1,2,3), delimiter=',')
    obs_id = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(0,), delimiter=',', dtype=str)
    if (telescope_id == 'mwa') or (telescope_id == 'mwa_tools') or (phased_array):
        delays_str = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(4,), delimiter=',', dtype=str)
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
    pointings_altaz = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
    pointings_altaz_orig = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
    lst = 15.0 * pointing_info_from_file[:,2]
    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

    if obs_mode is None:
        obs_mode = 'custom'
    if (obs_mode == 'dns') and beam_switch:
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
        lst_begin = NP.asarray(lst_wrapped[pick_snapshots])
        lst_end = NP.asarray(lst_wrapped[pick_snapshots+1])
        t_snap = (lst_end - lst_begin) / 15.0 * 3.6e3
        n_snaps = t_snap.size
        lst = 0.5 * (lst_begin + lst_end)
        pointings_altaz = pointings_altaz[pick_snapshots,:]
        obs_id = obs_id[pick_snapshots]
        if (telescope_id == 'mwa') or (phased_array) or (telescope_id == 'mwa_tools'):
            delays = delay_settings[pick_snapshots,:]
        obs_mode = 'custom'
    if pick_snapshots is None:
        if not beam_switch:
            lst = 0.5*(lst_edges[1:]+lst_edges[:-1])
            t_snap = (lst_edges[1:]-lst_edges[:-1]) / 15.0 * 3.6e3
        else:
            lst = 0.5*(lst_edges_left + lst_edges_right)
            t_snap = (lst_edges_right - lst_edges_left) / 15.0 * 3.6e3

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
        pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

    pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
    pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

    pointings_radec_orig = NP.copy(pointings_radec)
    pointings_hadec_orig = NP.copy(pointings_hadec)
    pointings_altaz_orig = NP.copy(pointings_altaz)
    pointings_dircos_orig = NP.copy(pointings_dircos)

    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

pointing_info = {}
pointing_info['pointing_center'] = pointings_altaz
pointing_info['pointing_coords'] = 'altaz'
pointing_info['lst'] = lst
if element_locs is not None:
    telescope['element_locs'] = element_locs

plots = args['plots']

use_GSM = args['ASM']
use_DSM = args['DSM']
use_CSM = args['CSM']
use_NVSS = args['NVSS']
use_SUMSS = args['SUMSS']
use_MSS = args['MSS']
use_GLEAM = args['GLEAM']
use_PS = args['PS']
use_USM = args['USM']

fg_str = ''
nside = args['nside']
pixres = HP.nside2pixarea(nside)
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


if use_GSM:
    fg_str = 'asm'

    dsm_file = args['DSM_file_prefix']+'_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0 * FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.185 # in GHz
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
    freq_DSM = 0.185 # in GHz
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
    hdulist.close()  

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

antpower_Jy = RI.antenna_power(skymod, telescope, pointing_info, freq_scale='Hz')
antpower_K = antpower_Jy * CNST.Jy / pixres / (2.0* FCNST.k * (1e9*chans.reshape(1,-1))**2 / FCNST.c**2)

if plots:
    fig = PLT.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    if flux_unit == 'Jy':
        ax.plot(lst/15, antpower_Jy[:,nchan/2], 'k-', lw=2)
    elif flux_unit == 'K':
        ax.plot(lst/15, antpower_K[:,nchan/2], 'k-', lw=2)
    ax.set_xlim(0, 24)
    ax.set_xlabel('RA [hours]', fontsize=18, weight='medium')
    ax.set_ylabel(r'$T_\mathrm{sky}$'+' [ '+flux_unit+' ]', fontsize=16, weight='medium')
    ax_y2 = ax.twinx()
    if flux_unit == 'Jy':
        ax_y2.set_yticks(Jy2K(ax.get_yticks(), chans[nchan/2]*1e9, pixres))
        ax_y2.set_ylim(Jy2K(NP.asarray(ax.get_ylim())), chans[nchan/2]*1e9, pixres)
        ax_y2.set_ylabel(r'$T_\mathrm{sky}$'+' [ K ]', fontsize=16, weight='medium')
    elif flux_unit == 'K':
        ax_y2.set_yticks(K2Jy(ax.get_yticks(), chans[nchan/2]*1e9, pixres))
        ax_y2.set_ylim(K2Jy(NP.asarray(ax.get_ylim()), chans[nchan/2]*1e9, pixres))
        ax_y2.set_ylabel(r'$T_\mathrm{sky}$'+' [ Jy ]', fontsize=16, weight='medium')

    ax.text(0.5, 0.9, '{0:.1f} MHz'.format(chans[nchan/2]*1e3), transform=ax.transAxes, fontsize=12, weight='medium', ha='center', color='black')

    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(left=0.15)

    PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/antenna_power_'+telescope_str+ground_plane_str+latitude_str+snapshot_type_str+'FG_model_'+fg_str+'_nside_{0:0d}'.format(nside)+'_sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+'{0:.1f}_hrs'.format(NP.sum(t_snap)/3.6e3)+'.png', bbox_inches=0)

PDB.set_trace()



