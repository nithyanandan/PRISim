from mpi4py import MPI 
import argparse
import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
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
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import ipdb as PDB

## Set MPI parameters

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
name = MPI.Get_processor_name()

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to simulate MWA interferometer array data')

parser.add_argument('--antenna-file', help='File containing antenna locations', default='/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', type=file, dest='antenna_file')

telescope_group = parser.add_argument_group('Telescope parameters', 'Telescope/interferometer specifications')
telescope_group.add_argument('--label-prefix', help='Prefix for baseline labels [str, Default = "B"]', default='B', type=str, dest='label_prefix')
telescope_group.add_argument('--telescope', help='Telescope name [str, Default = "mwa"]', default='mwa', type=str, dest='telescope', choices=['mwa', 'vla', 'gmrt'])
telescope_group.add_argument('--latitude', help='Latitude of interferometer array in degrees [float, Default=-26.701]', default=-26.701, type=float, dest='latitude')
telescope_group.add_argument('--A-eff', help='Effective area in m^2', type=float, dest='A_eff', nargs='?')
telescope_group.add_argument('--Tsys', help='System temperature in K [float, Default=440.0]', default=440.0, type=float, dest='Tsys')
telescope_group.add_argument('--pfb-method', help='PFB coarse channel shape computation method [str, Default="theoretical"]', dest='pfb_method', default='theoretical', choices=['theoretical', 'empirical', None])
telescope_group.add_argument('--pfb-file', help='File containing PFB coefficients', type=file, dest='pfb_file', default=None)

obsparm_group = parser.add_argument_group('Observation setup', 'Parameters specifying the observation')
obsparm_group.add_argument('-f', '--freq', help='Foreground center frequency in Hz [float, Default=150e6]', default=150e6, type=float, dest='freq')
obsparm_group.add_argument('--dfreq', help='Frequency resolution in Hz [float, Default=40e3]', default=40e3, type=float, dest='freq_resolution')
obsparm_group.add_argument('--obs-mode', help='Observing mode [str, track/drift/drift-shift/custom]', default=None, type=str, dest='obs_mode', choices=['track', 'drift', 'dns', 'custom'])
# obsparm_group.add_argument('--t-snap', help='Integration time (seconds) [float, Default=300.0]', default=5.0*60.0, type=float, dest='t_snap')
obsparm_group.add_argument('--nchan', help='Number of frequency channels [int, Default=256]', default=256, type=int, dest='n_channels')
# obsparm_group.add_argument('--lst-init', help='LST at beginning of observing run (hours) [float]', type=float, dest='lst_init', required=True, metavar='LST')
# obsparm_group.add_argument('--pointing-init', help='Pointing (RA, Dec) at beginning of observing run (degrees) [float]', type=float, dest='pointing_init', metavar=('RA', 'Dec'), required=True, nargs=2)

duration_group = parser.add_argument_group()
duration_group.add_argument('--t-obs', help='Duration of observation [seconds]', dest='t_obs', default=None, type=float, metavar='t_obs')
duration_group.add_argument('--n-snap', help='Number of snapshots or records that make up the observation', dest='n_snaps', default=None, type=int, metavar='n_snapshots')
duration_group.add_argument('--t-snap', help='integration time of each snapshot [seconds]', dest='t_snap', default=None, type=int, metavar='t_snap')

snapshot_selection_group = parser.add_mutually_exclusive_group(required=True)
snapshot_selection_group.add_argument('--avg-drifts', dest='avg_drifts', action='store_true')
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
processing_group.add_argument('--bpw', help='Bandpass window shape [str, "rect"]', default='rect', type=str, dest='bpass_shape', choices=['rect', 'bnw'])
processing_group.add_argument('--f-pad', help='Frequency padding fraction for delay transform [float, Default=1.0]', type=float, dest='f_pad', default=1.0)
processing_group.add_argument('--coarse-channel-width', help='Width of coarse channel [int: number of fine channels]', dest='coarse_channel_width', default=32, type=int)
processing_group.add_argument('--bp-correct', help='Bandpass correction', dest='bp_correct', action='store_true')
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

fgmodel_group = parser.add_mutually_exclusive_group(required=True)
fgmodel_group.add_argument('--ASM', action='store_true')
fgmodel_group.add_argument('--DSM', action='store_true')
fgmodel_group.add_argument('--CSM', action='store_true')
fgmodel_group.add_argument('--SUMSS', action='store_true')
fgmodel_group.add_argument('--NVSS', action='store_true')
fgmodel_group.add_argument('--MSS', action='store_true')
fgmodel_group.add_argument('--GLEAM', action='store_true')
fgmodel_group.add_argument('--PS', action='store_true')

fgparm_group = parser.add_argument_group('Foreground Setup', 'Parameters describing foreground sky')
fgparm_group.add_argument('--flux-unit', help='Units of flux density [str, Default="Jy"]', type=str, dest='flux_unit', default='Jy', choices=['Jy','K'])
fgparm_group.add_argument('--spindex', help='Spectral index, ~ f^spindex [float, Default=0.0]', type=float, dest='spindex', default=0.0)
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

try:
    ant_locs = NP.loadtxt(args['antenna_file'], skiprows=6, comments='#', usecols=(1,2,3))
except IOError:
    raise IOError('Could not open file containing antenna locations.')

n_bins_baseline_orientation = args['n_bins_baseline_orientation']
baseline_chunk_size = args['baseline_chunk_size']
bl_chunk = args['bl_chunk']
n_bl_chunks = args['n_bl_chunks']
telescope = args['telescope']
freq = args['freq']
freq_resolution = args['freq_resolution']
latitude = args['latitude']
if args['A_eff'] is None:
    A_eff = 16.0 * (0.5 * FCNST.c / freq)**2
else:
    A_eff = args['A_eff']
obs_mode = args['obs_mode']
Tsys = args['Tsys']
t_snap = args['t_snap']
t_obs = args['t_obs']
n_snaps = args['n_snaps']
avg_drifts = args['avg_drifts']
snapshot_sampling = args['snapshot_sampling']
pick_snapshots = args['pick_snapshots']
snapshots_range = args['snapshots_range']
avg_drifts_str = ''
if avg_drifts and (obs_mode == 'dns'):
    avg_drifts_str = 'drift_averaged_'
pointing_file = args['pointing_file']
if pointing_file is not None:
    pointing_file = pointing_file[0]
pointing_info = args['pointing_info']
if pointing_file is not None:
    pointing_init = None
    pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(1,2,3), delimiter=',')
    obs_id = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(0,), delimiter=',', dtype=str)
    if n_snaps is None:
        n_snaps = pointing_info_from_file.shape[0]
    pointing_info_from_file = pointing_info_from_file[:min(n_snaps, pointing_info_from_file.shape[0]),:]
    obs_id = obs_id[:min(n_snaps, pointing_info_from_file.shape[0])]
    n_snaps = min(n_snaps, pointing_info_from_file.shape[0])
    pointings_altaz = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
    lst = 15.0 * pointing_info_from_file[:,2]
    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

    if obs_mode is None:
        obs_mode = 'custom'
    if (obs_mode == 'dns') and avg_drifts:
        angle_diff = GEOM.sphdist(pointings_altaz[1:,1], pointings_altaz[1:,0], pointings_altaz[:-1,1], pointings_altaz[:-1,0])
        angle_diff = NP.concatenate(([0.0], angle_diff))
        shift_threshold = 1.0 # in degrees
        # lst_edges = NP.concatenate(([lst_edges[0]], lst_edges[angle_diff > shift_threshold], [lst_edges[-1]]))
        lst_wrapped = NP.concatenate(([lst_wrapped[0]], lst_wrapped[angle_diff > shift_threshold], [lst_wrapped[-1]]))
        n_snaps = lst_edges.size - 1
        pointings_altaz = NP.vstack((pointings_altaz[0,:].reshape(-1,2), pointings_altaz[angle_diff>shift_threshold,:].reshape(-1,2)))
        obs_id = NP.concatenate(([obs_id[0]], obs_id[angle_diff>shift_threshold]))
        obs_mode = 'custom'
    elif snapshots_range is not None:
        snapshots_range[1] = snapshots_range[1] % n_snaps
        if snapshots_range[0] > snapshots_range[1]:
            raise IndexError('min snaphost # must be <= max snapshot #')
        lst_wrapped = lst_wrapped[snapshots_range[0]:snapshots_range[1]+2]
        pointings_altaz = pointings_altaz[snapshots_range[0]:snapshots_range[1]+1,:]
        obs_id = obs_id[snapshots_range[0]:snapshots_range[1]+1]
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
        obs_mode = 'custom'
    if pick_snapshots is None:
        lst = 0.5*(lst_edges[1:]+lst_edges[:-1])
        t_snap = (lst_edges[1:]-lst_edges[:-1]) / 15.0 * 3.6e3

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
        pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

    pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
    pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

n_channels = args['n_channels']
bpass_shape = args['bpass_shape']
oversampling_factor = 1.0 + args['f_pad']
n_pad = args['n_pad']
pfb_method = args['pfb_method']
bandpass_correct = args['bp_correct']
flag_chan  = NP.asarray(args['flag_chan']).reshape(-1)
bp_flag_repeat = args['bp_flag_repeat']
coarse_channel_width = args['coarse_channel_width']
n_edge_flag = NP.asarray(args['n_edge_flag']).reshape(-1)
flag_repeat_edge_channels = args['flag_repeat_edge_channels']

bl = RI.baseline_generator(ant_locs, auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
neg_bl_orientation_ind = bl_orientation < 0.0
# neg_bl_orientation_ind = NP.logical_or(bl_orientation < -0.5*180.0/n_bins_baseline_orientation, bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
total_baselines = bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
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
flagged_edge_channels = []
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
            pfb_edge_channels = bandpass_shape.argsort()[:int(1.0*n_channels/coarse_channel_width)]
            flagged_edge_channels += [range(max(0,pfb_edge-n_edge_flag[0]),min(n_channels-1,pfb_edge+n_edge_flag[1])) for pfb_edge in pfb_edge_channels]

window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=n_pad, centering=True, area_normalize=True) 
if bandpass_correct:
    bpcorr = 1/bandpass_shape
    bandpass_shape = NP.ones(base_bpass.size)
else:
    bpcorr = 1.0*NP.ones(nchan)

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

use_GSM = args['ASM']
use_DSM = args['DSM']
use_CSM = args['CSM']
use_NVSS = args['NVSS']
use_SUMSS = args['SUMSS']
use_MSS = args['MSS']
use_GLEAM = args['GLEAM']
use_PS = args['PS']

if plots:
    if rank == 0:

        ## Plot the pointings

        pointings_ha = pointings_hadec[:,0]
        pointings_ha[pointings_ha > 180.0] = pointings_ha[pointings_ha > 180.0] - 360.0
    
        pointings_ra = pointings_radec[:,0]
        pointings_ra[pointings_ra > 180.0] = pointings_ra[pointings_ra > 180.0] - 360.0
    
        pointings_dec = pointings_radec[:,1]
    
        fig = PLT.figure(figsize=(6,6))
        ax1a = fig.add_subplot(111)
        ax1a.set_xlabel('Local Sidereal Time [hours]', fontsize=18, weight='medium')
        ax1a.set_ylabel('Longitude [degrees]', fontsize=18, weight='medium')
        ax1a.set_xlim((lst_wrapped.min()-1)/15.0, (lst_wrapped.max()-1)/15.0)
        ax1a.set_ylim(pointings_ha.min()-15.0, pointings_ha.max()+15.0)
        ax1a.plot(lst_wrapped, pointings_ha, 'k--', lw=2, label='HA')
        ax1a.plot(lst_wrapped, pointings_ra, 'k-', lw=2, label='RA')
        ax1a.tick_params(which='major', length=18, labelsize=12)
        ax1a.tick_params(which='minor', length=12, labelsize=12)
        legend1a = ax1a.legend(loc='upper left')
        legend1a.draw_frame(False)
        for axis in ['top','bottom','left','right']:
            ax1a.spines[axis].set_linewidth(2)
        xticklabels = PLT.getp(ax1a, 'xticklabels')
        yticklabels = PLT.getp(ax1a, 'yticklabels')
        PLT.setp(xticklabels, fontsize=15, weight='medium')
        PLT.setp(yticklabels, fontsize=15, weight='medium')    
    
        ax1b = ax1a.twinx()
        ax1b.set_ylabel('Declination [degrees]', fontsize=18, weight='medium')
        ax1b.set_ylim(pointings_dec.min()-5.0, pointings_dec.max()+5.0)
        ax1b.plot(lst_wrapped, pointings_dec, 'k:', lw=2, label='Dec')
        ax1b.tick_params(which='major', length=12, labelsize=12)
        legend1b = ax1b.legend(loc='upper center')
        legend1b.draw_frame(False)
        yticklabels = PLT.getp(ax1b, 'yticklabels')
        PLT.setp(yticklabels, fontsize=15, weight='medium')    
    
        fig.subplots_adjust(right=0.85)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+obs_mode+'_pointings.eps', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+obs_mode+'_pointings.png', bbox_inches=0)

        ## Plot bandpass properties

        fig = PLT.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('frequency [MHz]', fontsize=18, weight='medium')
        ax.set_ylabel('gain', fontsize=18, weight='medium')
        ax.set_xlim(149.0, 152.0)
        ax.set_ylim(0.05, 2.0*bpcorr.max())
        ax.set_yscale('log')
        try:
            ax.plot(1e3*chans, 10**(pfbwin_interp/10), 'k.:', lw=2, ms=10, label='Instrumental PFB Bandpass')
        except NameError:
            pass
        ax.plot(1e3*chans, bpcorr, 'k+--', lw=2, ms=10, label='Bandpass Correction')
        ax.plot(1e3*chans, bandpass_shape, 'k-', lw=2, label='Corrected Bandpass (Flagged)')
        ax.plot(1e3*chans, 3.0+NP.zeros(n_channels), 'k-.', label='Flagging threshold')
        legend = ax.legend(loc='lower center')
        legend.draw_frame(False)
        ax.tick_params(which='major', length=18, labelsize=12)
        ax.tick_params(which='minor', length=12, labelsize=12)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        xticklabels = PLT.getp(ax, 'xticklabels')
        yticklabels = PLT.getp(ax, 'yticklabels')
        PLT.setp(xticklabels, fontsize=15, weight='medium')
        PLT.setp(yticklabels, fontsize=15, weight='medium')    

        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/bandpass_properties.eps', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/bandpass_properties.png', bbox_inches=0)

fg_str = ''
nside = args['nside']
flux_unit = args['flux_unit']

if use_GSM:
    fg_str = 'asm'

    dsm_file = args['DSM_file_prefix']+'{0:0d}.fits'.format(nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.150 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ra_deg = ra_deg_DSM + 0.0
    dec_deg = dec_deg_DSM + 0.0
    majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
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
    # ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex)
elif use_DSM:
    fg_str = 'dsm'

    dsm_file = args['DSM_file_prefix']+'{0:0d}.fits'.format(nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.150 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ra_deg = ra_deg_DSM
    dec_deg = dec_deg_DSM
    majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    fluxes = fluxes_DSM
    ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    hdulist.close()
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

    nvss_file = args['NVSS_file']
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
    spindex = -0.83 + NP.zeros(fint.size)
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcsec','arcsec','degree'])    
    fg_str = 'sumss'
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
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex)
    fg_str = 'gleam'
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
    freq_PS = 0.150 # in GHz
    freq_catalog = freq_PS * 1e9 + NP.zeros(fint.size)
    catlabel = NP.repeat('PS', fint.size)
    ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), src_shape_units=['arcmin','arcmin','degree'])

# elif use_PS:
#     n_src = 1
#     fpeak = 1000.0*NP.ones(n_src)
#     spindex = NP.ones(n_src) * spindex
#     ra_deg = NP.asarray(pointings_radec[0,0])
#     dec_deg = NP.asarray(pointings_radec[0,1])
#     fmajax = NP.ones(n_src)
#     fminax = fmajax
#     fpa = NP.zeros(n_src)
#     ctlgobj = CTLG.Catalog('PS', freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcmin','arcmin','degree'])
#     fg_str = 'point'

skymod = CTLG.SkyModel(ctlgobj)

## Set up the observing run

if mpi_on_src: # MPI based on source multiplexing

    for i in range(len(bl_chunk)):
        print 'Working on baseline chunk # {0:0d} ...'.format(bl_chunk[i])

        ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz', pointing_coords='hadec')    

        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
        for j in range(n_snaps):
            src_altaz_current = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst[j]-skymod.catalog.location[:,0]).reshape(-1,1), skymod.catalog.location[:,1].reshape(-1,1))), latitude, units='degrees')
            roi_ind = NP.where(src_altaz_current[:,0] >= 0.0)[0]
            n_src_per_rank = NP.zeros(nproc, dtype=int) + roi_ind.size/nproc
            if roi_ind.size % nproc > 0:
                n_src_per_rank[:roi_ind.size % nproc] += 1
            cumm_src_count = NP.concatenate(([0], NP.cumsum(n_src_per_rank)))
            # timestamp = str(DT.datetime.now())
            timestamp = lst[j]
            ts = time.time()
            if j == 0:
                ts0 = ts
            ia.observe(timestamp, Tsys/bpcorr, bpass, pointings_hadec[j,:], CTLG.SkyModel(skymod.catalog.subset(roi_ind[cumm_src_count[rank]:cumm_src_count[rank+1]].tolist())), t_snap[j], brightness_units=flux_unit, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
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
            outfile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
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

                outfile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[count]],bl_length[min(baseline_bin_indices[count]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(count)
                ia = RI.InterferometerArray(labels[baseline_bin_indices[count]:min(baseline_bin_indices[count]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[count]:min(baseline_bin_indices[count]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz', pointing_coords='hadec')        

                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
                for j in range(n_snaps):
                    if (obs_mode == 'custom') or (obs_mode == 'dns'):
                        timestamp = obs_id[j]
                    else:
                        timestamp = lst[j]
                    ts = time.time()
                    if j == 0:
                        ts0 = ts
                    ia.observe(timestamp, Tsys/bpcorr, bpass, pointings_hadec[j,:], skymod, t_snap[j], brightness_units=flux_unit, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
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
        for i in range(cumm_bl_chunks[rank], cumm_bl_chunks[rank+1]):
            print 'Process {0:0d} working on baseline chunk # {1:0d} ...'.format(rank, bl_chunk[i])
    
            outfile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
            ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz', pointing_coords='hadec')        
    
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_snaps).start()
            for j in range(n_snaps):
                if (obs_mode == 'custom') or (obs_mode == 'dns'):
                    timestamp = obs_id[j]
                else:
                    timestamp = lst[j]
                ts = time.time()
                if j == 0:
                    ts0 = ts
                ia.observe(timestamp, Tsys/bpcorr, bpass, pointings_hadec[j,:], skymod, t_snap[j], brightness_units=flux_unit, roi_radius=None, roi_center=None, lst=lst[j], memsave=True)
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
            ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
            pte_str = str(DT.datetime.now())                

print 'Process {0} has completed.'.format(rank)
PDB.set_trace()
