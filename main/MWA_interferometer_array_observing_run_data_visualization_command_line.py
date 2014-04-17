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
import ipdb as PDB

parser = argparse.ArgumentParser(description='Program to visualize MWA interferometer array simulated data')
parser.add_argument('--antenna-file', help='File containing antenna locations', default='/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', type=file, dest='antenna_file')

telescope_group = parser.add_argument_group('Telescope parameters', 'Telescope/interferometer specifications')
telescope_group.add_argument('--telescope', help='Telescope name [str, Default = "mwa"]', default='mwa', type=str, dest='telescope', choices=['mwa', 'vla', 'gmrt'])

obsparm_group = parser.add_argument_group('Observation setup', 'Parameters specifying the observation')
obsparm_group.add_argument('-f', '--freq', help='Foreground center frequency in Hz [float, Default=150e6]', default=150e6, type=float, dest='freq')
obsparm_group.add_argument('--dfreq', help='Frequency resolution in Hz [float, Default=40e3]', default=40e3, type=float, dest='freq_resolution')
obsparm_group.add_argument('--obs-mode', help='Observing mode [str, Default="track"]', default='track', type=str, dest='obs_mode', choices=['track', 'drift', 'custom'])
obsparm_group.add_argument('--nchan', help='Number of frequency channels [int, Default=256]', default=256, type=int, dest='n_channels')

fgmodel_group = parser.add_mutually_exclusive_group(required=True)
fgmodel_group.add_argument('--ASM', action='store_true')
fgmodel_group.add_argument('--DSM', action='store_true')
fgmodel_group.add_argument('--SUMSS', action='store_true')
fgmodel_group.add_argument('--NVSS', action='store_true')
fgmodel_group.add_argument('--MSS', action='store_true')
fgmodel_group.add_argument('--GLEAM', action='store_true')
fgmodel_group.add_argument('--PS', action='store_true')

processing_group = parser.add_argument_group('Processing arguments', 'Processing parameters')
processing_group.add_argument('--n-bins-blo', help='Number of bins for baseline orientations [int, Default=4]', default=4, type=int, dest='n_bins_baseline_orientation')
processing_group.add_argument('--bl-chunk-size', help='Baseline chunk size [int, Default=100]', default=100, type=int, dest='baseline_chunk_size')
processing_group.add_argument('--bl-chunk', help='Baseline chunk indices to process [int(s), Default=None: all chunks]', default=None, type=int, dest='bl_chunk', nargs='*')
processing_group.add_argument('--n-bl-chunks', help='Upper limit on baseline chunks to be processed [int, Default=None]', default=None, type=int, dest='n_bl_chunks')
processing_group.add_argument('--bpw', help='Bandpass window shape [str, "rect"]', default='rect', type=str, dest='bpass_shape', choices=['rect', 'bnw'])
processing_group.add_argument('--f-pad', help='Frequency padding fraction for delay transform [float, Default=1.0]', type=float, dest='f_pad', default=1.0)

parser.add_argument('--max-abs-delay', help='Maximum absolute delay (micro seconds) [float, Default=None]', default=None, type=float, dest='max_abs_delay')

backdrop_group = parser.add_argument_group('Backdrop arguments', 'Backdrop parameters')
backdrop_group.add_argument('--backdrop-coords', help='Backdrop coordinates [str, Default="dircos"]', default='dircos', type=str, dest='backdrop_coords', choices=['radec', 'dircos'])
backdrop_group.add_argument('--backdrop-size', help='Backdrop size (x, y) [int, Default=(100,50)]', type=int, dest='backdrop_size', metavar=('xsize', 'ysize'), nargs=2, default=[100,50])
backdrop_group.add_argument('--nside', help='nside parameter for healpix map [int, Default=64]', type=int, dest='nside', default=64, choices=[64, 128])

visual_group = parser.add_argument_group('Visualization arguments', 'Visualization setup parameters')
visual_group.add_argument('--fig-size', help='Figure size in inches [float, Default=(14,14)]', default=[14,14], type=float, dest='figsize', metavar=('xsize', 'ysize'), nargs=2)
visual_group.add_argument('--fps', help='Frame rate in fps [float, Default=1.0]', default=1.0, type=float, dest='fps', metavar='framerate')
visual_group.add_argument('--interval', help='Frame interval in ms [float, Default=100.0]', default=100.0, dest='interval', metavar='interval')
visual_group.add_argument('--animation-file', help='Animal filename prefix [str, Default=None]', dest='animation_file', default=None, type=str)
visual_group.add_argument('--animation-format', help='Animation file format [str, Default=MP4]', default='MP4', choices=['MP4', 'GIF'], dest='animation_format', type=str)

args = vars(parser.parse_args())

try:
    ant_locs = NP.loadtxt(args['antenna_file'], skiprows=6, comments='#', usecols=(1,2,3))
except IOError:
    raise IOError('Could not open file containing antenna locations.')

freq = args['freq']
freq_resolution = args['freq_resolution']
bpass_shape = args['bpass_shape']

n_bins_baseline_orientation = args['n_bins_baseline_orientation']
baseline_chunk_size = args['baseline_chunk_size']
bl_chunk = args['bl_chunk']
n_bl_chunks = args['n_bl_chunks']
telescope = args['telescope']
obs_mode = args['obs_mode']
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
if bl_chunk is None:
    bl_chunk = range(len(baseline_bin_indices))
if n_bl_chunks is None:
    n_bl_chunks = len(bl_chunk)
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:baseline_bin_indices[n_bl_chunks],:]
bl_length = bl_length[:baseline_bin_indices[n_bl_chunks]]
bl_orientation = bl_orientation[:baseline_bin_indices[n_bl_chunks]]

oversampling_factor = 1.0 + args['f_pad']
n_channels = args['n_channels']
nchan = n_channels
max_abs_delay = args['max_abs_delay']

nside = args['nside']
use_GSM = args['ASM']
use_DSM = args['DSM']
use_NVSS = args['NVSS']
use_SUMSS = args['SUMSS']
use_MSS = args['MSS']
use_GLEAM = args['GLEAM']
use_PS = args['PS']

if use_GSM:
    fg_str = 'asm'
elif use_DSM:
    fg_str = 'dsm'
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

PDB.set_trace()

lags = None
skyvis_lag = None
vis_lag = None
progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_bl_chunks).start()
for i in range(0, n_bl_chunks):
    infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
    hdulist = fits.open(infile+'.fits')
    # extnames = [hdu.header['EXTNAME'] for hdu in hdulist]
    if i == 0:
        lags = hdulist['SPECTRAL INFO'].data.field('lag')
        vis_lag = hdulist['real_lag_visibility'].data + 1j * hdulist['imag_lag_visibility'].data
        skyvis_lag = hdulist['real_lag_sky_visibility'].data + 1j * hdulist['imag_lag_sky_visibility'].data

        latitude = hdulist[0].header['latitude']
        pointing_coords = hdulist[0].header['pointing_coords']
        pointings_table = hdulist['POINTING INFO'].data
        lst = pointings_table['LST']
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
    else:
        vis_lag = NP.vstack((vis_lag, hdulist['real_lag_visibility'].data + 1j * hdulist['imag_lag_visibility'].data))
        skyvis_lag = NP.vstack((skyvis_lag, hdulist['real_lag_sky_visibility'].data + 1j * hdulist['imag_lag_sky_visibility'].data))
    hdulist.close()
    progress.update(i+1)
progress.finish()

if max_abs_delay is not None:
    small_delays_ind = NP.abs(lags) <= max_abs_delay * 1e-6
    lags = lags[small_delays_ind]
    vis_lag = vis_lag[:,small_delays_ind,:]
    skyvis_lag = skyvis_lag[:,small_delays_ind,:]

## Delay limits estimation

delay_matrix = DLY.delay_envelope(bl, pointings_dircos, units='mks')

## Binning baselines by orientation

# blo = bl_orientation[:min(n_bl_chunks*baseline_chunk_size, total_baselines)]
blo = bl_orientation
# blo[blo < -0.5*360.0/n_bins_baseline_orientation] = 360.0 - NP.abs(blo[blo < -0.5*360.0/n_bins_baseline_orientation])
PDB.set_trace()
bloh, bloe, blon, blori = OPS.binned_statistic(blo, statistic='count', bins=n_bins_baseline_orientation, range=[(-0.5*180.0/n_bins_baseline_orientation, 180.0-0.5*180.0/n_bins_baseline_orientation)])

if use_DSM or use_GSM:
    backdrop_coords = 'radec'

if backdrop_coords == 'radec':
    xmin = 0.0
    xmax = 360.0
    ymin = -90.0
    ymax = 90.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
    xvect = xgrid.ravel()
    yvect = ygrid.ravel()
elif backdrop_coords == 'dircos':
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize))
    nanind = (xgrid**2 + ygrid**2) > 1.0
    goodind = (xgrid**2 + ygrid**2) <= 1.0
    zgrid = NP.empty_like(xgrid)
    zgrid[nanind] = NP.nan
    zgrid[goodind] = NP.sqrt(1.0 - (xgrid[goodind]**2 + ygrid[goodind]**2))

    xvect = xgrid.ravel()
    yvect = ygrid.ravel()
    zvect = zgrid.ravel()
    xyzvect = NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1), zvect.reshape(-1,1)))

if use_DSM or use_GSM:
    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata{0:0d}.fits'.format(nside)
    hdulist = fits.open(dsm_file)
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures
    backdrop = HP.cartview(temperatures.ravel(), coord=['G','E'], rot=[180,0,0], xsize=backdrop_xsize, return_projected_map=True)
elif use_GLEAM or use_SUMSS:
    if use_GLEAM:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv' # GLEAM catalog
        catdata = ascii.read(catalog_file, data_start=1, delimiter=',')
        dec_deg = catdata['DEJ2000']
        ra_deg = catdata['RAJ2000']
        fpeak = catdata['S150_fit']
        ferr = catdata['e_S150_fit']
        freq_catalog = 1.4 # GHz
        spindex = -0.83 + NP.zeros(fpeak.size)
        fluxes = fpeak * (freq_catalog * 1e9 / freq)**spindex
    else:
        SUMSS_file = '/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt'
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
        fluxes = fint * (freq_catalog*1e9/freq)**spindex

    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
else:
    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)

## Create data for overlay 

overlays = []
roi_obj_inds = []
for i in xrange(n_snaps):
    overlay = {}
    if backdrop_coords == 'radec':
        havect = lst[i] - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
        pb = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', phase_center=pointings_altaz[i,:])
        overlay['pbeam'] = pb
        overlay['roi_obj_inds'] = roi_altaz
        # roi_obj_inds += [roi_altaz]
    elif backdrop_coords == 'dircos':
        havect = lst[i] - ra_deg
        fg_altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),dec_deg.reshape(-1,1))), latitude, units='degrees')
        fg_dircos = GEOM.altaz2dircos(fg_altaz, units='degrees')
        roi_dircos = NP.asarray(NP.where(fg_dircos[:,2] >= 0.0)).ravel()
        overlay['roi_obj_inds'] = roi_dircos
        overlay['fg_dircos'] = fg_dircos
        if obs_mode == 'track':
            pb = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, skyunits='dircos', freq_scale='Hz', phase_center=pointings_dircos[i,:])
            # pb[pb < 0.5] = NP.nan
            overlay['pbeam'] = pb.reshape(backdrop_xsize, backdrop_xsize)
    overlays += [overlay]

## Animation set up

fps = args['fps']
interval = args['interval']
animation_format = args['animation_format']
if animation_format == 'MP4':
    anim_format = '.mp4'
else:
    anim_format = 'gif'
animation_file = args['animation_file']
if animation_file is None:
    animation_file = '/data3/t_nithyanandan/project_MWA/multi_baseline_noiseless_visibilities_'+obs_mode+'_'+'{0:0d}'.format(80*baseline_chunk_size)+'_baselines_{0:0d}_orientations_'.format(n_bins_baseline_orientation)+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_8_sectors' 

if n_bins_baseline_orientation == 4:
    blo_ax_mapping = [6,3,2,1,4,7,8,9]

# if n_bins_baseline_orientation == 4:
#     blo_ax_mapping = [6,2,4,8]
# elif n_bins_baseline_orientation == 8:
#     blo_ax_mapping = [6,3,2,1,4,7,8,9]

fig = PLT.figure(figsize=(14,14))

axs = []
for i in range(2*n_bins_baseline_orientation):
    ax = fig.add_subplot(3,3,blo_ax_mapping[i])
    if i < n_bins_baseline_orientation:
        ax.set_xlim(0,bloh[i]-1)
        ax.set_ylim(0.0, NP.amax(lags*1e6))
    else:
        # ax = fig.add_subplot(3,3,blo_ax_mapping[i%n_bins_baseline_orientation])
        ax.set_xlim(0,bloh[i%n_bins_baseline_orientation]-1)
        ax.set_ylim(NP.amin(lags*1e6), 0.0)

    l = ax.plot([], [], 'k-', [], [], 'k:', [], [])
    ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i%n_bins_baseline_orientation], bloe[(i%n_bins_baseline_orientation)+1]), weight='semibold')
    ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
    # ax.set_aspect('auto')
    axs += [ax]

ax = fig.add_subplot(3,3,5)
if backdrop_coords == 'radec':
    ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=12)
    ax.set_ylabel(r'$\delta$ [degrees]', fontsize=12)
elif backdrop_coords == 'dircos':
    ax.set_xlabel('l')
    ax.set_ylabel('m')
ax.set_title('Sky Model', fontsize=18, weight='semibold')
ax.grid(True)
ax.tick_params(which='major', length=12, labelsize=12)
ax.tick_params(which='minor', length=6)

if use_DSM or use_GSM:
    # linit = ax.imshow(OPS.reverse(backdrop, axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
    linit = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
    # cbmn = NP.amin(backdrop)
    # cbmx = NP.amax(backdrop)
    # cbaxes = fig.add_axes([0.85, 0.1, 0.02, 0.23]) 
    # cbar = fig.colorbar(linit, cax=cbaxes)
    # cbmd = 10.0**(0.5*(NP.log10(cbmn)+NP.log10(cbmx)))
    # cbar.set_ticks([cbmn, cbmd, cbmx])
    # cbar.set_ticklabels([cbmn, cbmd, cbmx])
else:
    ax.set_xlim(NP.amin(xvect), NP.amax(xvect))
    ax.set_ylim(NP.amin(yvect), NP.amax(yvect))
    if backdrop_coords == 'radec':
        linit = ax.scatter(ra_deg, dec_deg, c=fpeak, marker='.', cmap=PLT.cm.get_cmap("rainbow"), norm=PLTC.LogNorm())
        # cbmn = NP.amin(fpeak)
        # cbmx = NP.amax(fpeak)
    else:
        if (obs_mode == 'drift') or (telescope == 'mwa_dipole'):
            linit = ax.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
            # cbaxes = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
            # cbar = fig.colorbar(linit, cax=cbaxes)

l = ax.plot([], [], 'w.', [], [])
# txt = ax.text(0.25, 0.65, '', transform=ax.transAxes, fontsize=18)

axs += [ax]
tpc = axs[-1].text(0.5, 1.15, '', transform=ax.transAxes, fontsize=12, weight='semibold', ha='center')

PLT.tight_layout()
fig.subplots_adjust(bottom=0.1)

def update(i, pointing_radec, lst, obsmode, telescope, backdrop_coords, bll, blori, lags, vis_lag, delaymatrix, overlays, xv, yv, xv_uniq, yv_uniq, axs, tpc):

    delay_ranges = NP.dstack((delaymatrix[:,:vis_lag.shape[0],1] - delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],1] + delaymatrix[:,:vis_lag.shape[0],0]))
    delay_horizon = NP.dstack((-delaymatrix[:,:vis_lag.shape[0],0], delaymatrix[:,:vis_lag.shape[0],0]))
    bl = bll[:vis_lag.shape[0]]

    label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i])

    for j in range((len(axs)-1)/2):
        blind = blori[blori[j]:blori[j+1]]
        sortind = NP.argsort(bl[blind], kind='heapsort')
        axs[j].lines[0].set_xdata(NP.arange(blind.size))
        axs[j].lines[0].set_ydata(delay_ranges[i,blind[sortind],1]*1e6)
        axs[j].lines[0].set_linewidth(0.5)
        axs[j].lines[1].set_xdata(NP.arange(blind.size))
        axs[j].lines[1].set_ydata(delay_horizon[i,blind[sortind],1]*1e6)
        axs[j].lines[1].set_linewidth(0.5)
        axs[j].lines[2] = axs[j].imshow(NP.abs(vis_lag[blind[sortind],NP.floor(0.5*vis_lag.shape[1]):,i].T), origin='lower', extent=(0, blind.size-1, 0.0, NP.amax(lags*1e6)), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(vis_lag)), vmax=NP.amax(NP.abs(vis_lag))), interpolation=None)
        axs[j].set_aspect('auto')
        axs[j+(len(axs)-1)/2].lines[0].set_xdata(NP.arange(blind.size))
        axs[j+(len(axs)-1)/2].lines[0].set_ydata(delay_ranges[i,blind[sortind],0]*1e6)
        axs[j+(len(axs)-1)/2].lines[0].set_linewidth(0.5)
        axs[j+(len(axs)-1)/2].lines[1].set_xdata(NP.arange(blind.size))
        axs[j+(len(axs)-1)/2].lines[1].set_ydata(delay_horizon[i,blind[sortind],0]*1e6)
        axs[j+(len(axs)-1)/2].lines[1].set_linewidth(0.5)
        axs[j+(len(axs)-1)/2].lines[2] = axs[j+(len(axs)-1)/2].imshow(NP.abs(vis_lag[blind[sortind],:NP.floor(0.5*vis_lag.shape[1]),i].T), origin='lower', extent=(0, blind.size-1, NP.amin(lags*1e6), 1e6*lags[NP.floor(0.5*lags.size)-1]), norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(vis_lag)), vmax=NP.amax(NP.abs(vis_lag))), interpolation=None)
        axs[j+(len(axs)-1)/2].set_aspect('auto')

    cbax = fig.add_axes([0.175, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(axs[0].lines[2], cax=cbax, orientation='horizontal')
    cbax.set_xlabel('Jy Hz', labelpad=-1, fontsize=18)

    if backdrop_coords == 'radec':
        pbi = griddata(NP.hstack((xv[overlays[i]['roi_obj_inds']].reshape(-1,1),yv[overlays[i]['roi_obj_inds']].reshape(-1,1))), overlays[i]['pbeam'], NP.hstack((xv.reshape(-1,1),yv.reshape(-1,1))), method='nearest')
        axc = axs[-1]
        cntr = axc.contour(OPS.reverse(xv_uniq), yv_uniq, OPS.reverse(pbi.reshape(yv_uniq.size, xv_uniq.size), axis=1), 35)
        axc.set_aspect(1.5)
        axs[-1] = axc

        tpc.set_text(r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.2f} deg'.format(lst[i]))

    elif backdrop_coords == 'dircos':
        if (obsmode != 'drift') and (telescope != 'mwa_dipole'):
            axs[-1].lines[1] = axs[-1].imshow(overlays[i]['pbeam'], origin='lower', extent=(NP.amin(xv_uniq), NP.amax(xv_uniq), NP.amin(yv_uniq), NP.amax(yv_uniq)), norm=PLTC.LogNorm())
            # cbaxes3 = fig.add_axes([0.65, 0.1, 0.02, 0.23]) 
            # cbar3 = fig.colorbar(axs[-1].lines[1], cax=cbaxes3)
        axs[-1].lines[0].set_xdata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],0])
        axs[-1].lines[0].set_ydata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],1])
        axs[-1].lines[0].set_marker('.')

    return axs

anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lst, obs_mode, telescope, backdrop_coords, bl_length, blori, lags, skyvis_lag, delay_matrix, overlays, xvect, yvect, xgrid[0,:], ygrid[:,0], axs, tpc), frames=len(overlays), interval=interval, blit=False)
PLT.show()

anim.save(animation_file+anim_format, fps=fps, codec='x264')



