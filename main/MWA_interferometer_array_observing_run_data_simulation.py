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
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
# import ipdb as PDB

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
obsparm_group.add_argument('--obs-mode', help='Observing mode [str, Default="track"]', default='track', type=str, dest='obs_mode', choices=['track', 'drift'])
obsparm_group.add_argument('--t-snap', help='Integration time (seconds) [float, Default=300.0]', default=5.0*60.0, type=float, dest='t_snap')
obsparm_group.add_argument('--t-obs', help='Duration of observation (seconds) [float, Default=9000.0]', default=2.5*3600, type=float, dest='t_obs')
obsparm_group.add_argument('--nchan', help='Number of frequency channels [int, Default=256]', default=256, type=int, dest='n_channels')
obsparm_group.add_argument('--lst-init', help='LST at beginning of observing run (hours) [float]', type=float, dest='lst_init', required=True, metavar='LST')
obsparm_group.add_argument('--pointing-init', help='Pointing (RA, Dec) at beginning of observing run (degrees) [float]', type=float, dest='pointing_init', metavar=('RA', 'Dec'), required=True, nargs=2)
# obsparm_group.add_argument('--ra-init', help='RA at the beginning of observing run (in degrees) [float]', metavar='RA', type=float, dest='ra_init', required=True)
# obsparm_group.add_argument('--dec-init', help='Dec at the beginning of observing run (in degrees) [float]', metavar='Dec', type=float, dest='dec_init', required=True)

processing_group = parser.add_argument_group('Processing arguments', 'Processing parameters')
processing_group.add_argument('--n-bins-blo', help='Number of bins for baseline orientations [int, Default=4]', default=4, type=int, dest='n_bins_baseline_orientation')
processing_group.add_argument('--bl-chunk-size', help='Baseline chunk size [int, Default=100]', default=100, type=int, dest='baseline_chunk_size')
processing_group.add_argument('--bl-chunk', help='Baseline chunk indices to process [int(s), Default=None: all chunks]', default=None, type=int, dest='bl_chunk', nargs='*')
processing_group.add_argument('--bpw', help='Bandpass window shape [str, "rect"]', default='rect', type=str, dest='bpass_shape', choices=['rect', 'bnw'])
processing_group.add_argument('--f-pad', help='Frequency padding fraction for delay transform [float, Default=1.0]', type=float, dest='f_pad', default=1.0)
processing_group.add_argument('--coarse-channel-width', help='Width of coarse channel [int: number of fine channels]', dest='coarse_channel_width', default=32, type=int)
processing_group.add_argument('--bp-correct', help='Bandpass correction', dest='bp_correct', action='store_true')
processing_group.add_argument('--bpw-pad', help='Bandpass window padding length [int, Default=0]', dest='n_pad', default=0, type=int)

freq_flags_group = parser.add_argument_group('Frequency flagging', 'Parameters to describe flagging of bandpass')
freq_flags_group.add_argument('--flag-channels', help='Bandpass channels to be flagged. If bp_flag_repeat is set, bp_flag(s) will be forced in the range 0 <= flagged channel(s) < coarse_channel_width and applied to all coarse channels periodically [int,  default=-1: no flag]', dest='flag_chan', nargs='*', default=-1, type=int)
freq_flags_group.add_argument('--bp-flag-repeat', help='If set, will repeat any flag_chan(s) for all coarse channels after converting flag_chan(s) to lie in the range 0 <= flagged channel(s) < coarse_channel_width using flag_chan modulo coarse_channel_width', action='store_true', dest='bp_flag_repeat')
freq_flags_group.add_argument('--flag-edge-channels', help='Flag edge channels in the band. If flag_repeat_edge_channels is set, specified number of channels leading and trailing the coarse channel edges are flagged. First number includes the coarse channel minimum while the second number does not. Otherwise, specified number of channels are flagged at the beginning and end of the band. [int,int Default=0,0]', dest='n_edge_flag', nargs=2, default=[0,0], metavar=('NEDGE1','NEDGE2'), type=int)
freq_flags_group.add_argument('--flag-repeat-edge-channels', help='If set, will flag the leading and trailing channels whose number is specified in n_edge_flag. Otherwise, will flag the beginning and end of the band.', action='store_true', dest='flag_repeat_edge_channels')

fgmodel_group = parser.add_mutually_exclusive_group(required=True)
fgmodel_group.add_argument('--ASM', action='store_true')
fgmodel_group.add_argument('--DSM', action='store_true')
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
# parser.add_argument('--', help='', type=, dest='', required=True)

args = vars(parser.parse_args())

try:
    ant_locs = NP.loadtxt(args['antenna_file'], skiprows=6, comments='#', usecols=(1,2,3))
except IOError:
    raise IOError('Could not open file containing antenna locations.')

n_bins_baseline_orientation = args['n_bins_baseline_orientation']
baseline_chunk_size = args['baseline_chunk_size']
bl_chunk = args['bl_chunk']
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
pointing_init = NP.asarray(args['pointing_init'])
lst_init = args['lst_init']
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
neg_bl_orientation_ind = NP.logical_or(bl_orientation < -0.5*180.0/n_bins_baseline_orientation, bl_orientation > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
total_baselines = bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
labels = []
labels += [args['label_prefix']+'{0:0d}'.format(i+1) for i in xrange(bl.shape[0])]
if bl_chunk is None:
    bl_chunk = range(len(baseline_bin_indices))

nchan = n_channels
base_bpass = 1.0*NP.ones(nchan)
bandpass_shape = base_bpass
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
    bandpass_shape = base_bpass
else:
    bpcorr = 1.0*NP.ones(nchan)

bpass = base_bpass * bandpass_shape
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

n_snaps = int(1.0*t_obs/t_snap)
lst = (lst_init + (t_snap/3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
if obs_mode == 'track':
    pointings_radec = NP.repeat(NP.asarray(pointing_init).reshape(-1,2), n_snaps, axis=0)
else:
    pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

use_GSM = args['ASM']
use_DSM = args['DSM']
use_NVSS = args['NVSS']
use_SUMSS = args['SUMSS']
use_MSS = args['MSS']
use_GLEAM = args['GLEAM']
use_PS = args['PS']

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
    ra_deg = ra_deg_DSM
    dec_deg = dec_deg_DSM
    majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    fluxes = fluxes_DSM

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
    dsm_file = args['DSM_file_prefix']+'{0:0d}.fits'.format(nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes.size))
    minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes.size))
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.150 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    fg_str = 'dsm'
    hdulist.close()
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
    n_src = 1
    fpeak = 1000.0*NP.ones(n_src)
    spindex = NP.ones(n_src) * spindex
    ra_deg = NP.asarray(pointings_radec[0,0])
    dec_deg = NP.asarray(pointings_radec[0,1])
    fmajax = NP.ones(n_src)
    fminax = fmajax
    fpa = NP.zeros(n_src)
    ctlgobj = CTLG.Catalog('PS', freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcmin','arcmin','degree'])
    fg_str = 'point'

skymod = CTLG.SkyModel(ctlgobj)

## Start the observation

# PDB.set_trace()
# progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=len(bl_chunk)).start()
# for i in range(len(bl_chunk)):
#     outfile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
#     ia = RI.InterferometerArray(labels[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines)], bl[baseline_bin_indices[bl_chunk[i]]:min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size,total_baselines),:], chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz')    
#     ts = time.time()
#     ia.observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit, memsave=True)
#     print 'The last chunk of {0:0d} baselines required {1:.1f} minutes'.format(baseline_chunk_size, (time.time()-ts)/60.0)
#     ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
#     ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)
#     progress.update(i+1)
# progress.finish()

