import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.animation as MOV
from scipy.interpolate import griddata
import healpy as HP
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import primary_beams as PB
import baseline_delay_horizon as DLY

## Observation parameters

telescope = 'mwa_dipole'
freq = 150.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 40.0 # in kHz
latitude = -26.701
A_eff = 16.0 * (0.5 * FCNST.c / freq)**2
obs_mode = 'drift'
Tsys = 440.0 # in Kelvin
t_snap = 2*60.0 # in seconds 
t_obs = 3600.0 # in seconds
pointing_init = [0.0, latitude] # in degrees
lst_init = 0.0 # in hours
n_channels = 256
bpass_shape = 'rect' 
oversampling_factor = 4.0
eff_bw_ratio = 1.0
if bpass_shape == 'bnw':
    eff_bw_ratio = CNST.rect_bnw_ratio
n_pad = NP.round(oversampling_factor * n_channels) - NP.round(n_channels * eff_bw_ratio)
# elif bpass_shape == 'rect':
#     oversampling_factor = 1.0
nchan = NP.round(n_channels * oversampling_factor)
base_bpass = 1.0*NP.ones(nchan)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution * 1e3 )/ 1e9 # in GHz
# window = DSP.shaping(nchan, 1/oversampling_factor*eff_bw_ratio, shape=bpass_shape, peak=1.0)
window = DSP.windowing(NP.round(n_channels * eff_bw_ratio), shape=bpass_shape, pad_width=n_pad, centering=True)
bpass = base_bpass * window
n_snaps = int(t_obs/t_snap)
lst = (lst_init + (t_snap/3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
if obs_mode == 'track':
    pointings_radec = NP.repeat(NP.asarray(pointing_init).reshape(-1,2), n_snaps, axis=0)
else:
    pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

## Interferometer parameters

baseline_orientation = 0.0 # in degrees from East towards North
baseline_length = 1000.0 # in m
baseline_vect = baseline_length * NP.asarray([NP.cos(NP.radians(baseline_orientation)), NP.sin(NP.radians(baseline_orientation)), 0.0]) # in m

baseline_orientation_str = '{0:.1f}'.format(baseline_orientation)
baseline_length_str = '{0:.1f}'.format(baseline_length)

intrfrmtr = RI.Interferometer('B1', baseline_vect, chans, telescope=telescope, latitude=latitude, A_eff=A_eff, freq_scale='GHz')

## Foreground parameters

use_GSM = False
use_GLEAM = True
use_PS = False
use_other = False

use_FG_model = use_GSM + use_GLEAM + use_PS + use_other
if use_FG_model != 1:
    raise ValueError('One and only one foreground model must be specified.')

fg_str = ''

flux_unit = 'Jy'
freq_catalog = freq/1e9 # in GHz
spindex = 0.0

if use_GSM:
    gsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata.fits'
    hdulist = fits.open(gsm_file)
    flux_unit = 'K'
    gsm_table = hdulist[1].data
    ra_deg = gsm_table['RA']
    dec_deg = gsm_table['DEC']
    fluxes = gsm_table['f_{0:.0f}'.format(freq/1e6)]
    spindex = gsm_table['spindex']
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex)
    fg_str = 'gsm'
elif use_GLEAM:
    catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv' # GLEAM catalog
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
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex)
    fg_str = 'point'
elif use_other:
    n_src = 3
    fpeak = 1000.0 * NP.ones(n_src)
    spindex = NP.ones(n_src) * spindex
    ra_deg = pointings_radec[0,0] + NP.arange(n_src) * 14.5
    dec_deg = NP.ones(n_src) * pointings_radec[0,1]
    ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, spectral_index=spindex)
    fg_str = 'other'

skymod = CTLG.SkyModel(ctlgobj)

## Animation parameters

backdrop_xsize = 100
bitrate = 128
fps = 2.5
interval = 100

## Start the observing run

intrfrmtr.observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit)

intrfrmtr.delay_transform()
lags = intrfrmtr.lags
vis_lag = intrfrmtr.skyvis_lag
if oversampling_factor > 1.0:
    lags = DSP.downsampler(intrfrmtr.lags, oversampling_factor)
    vis_lag = DSP.downsampler(intrfrmtr.skyvis_lag, oversampling_factor)

band_avg_noise_info = intrfrmtr.band_averaged_noise_estimate(filter_method='hpf')

outfile = '/data3/t_nithyanandan/project_MWA/obs_data_'+telescope+'_'+obs_mode+'_baseline_'+baseline_length_str+'m_'+baseline_orientation_str+'_deg_FG_model_'+fg_str+'_'+bpass_shape+'{0:.1f}'.format(oversampling_factor)

intrfrmtr.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)

## Delay limits estimation

delay_matrix = DLY.delay_envelope(intrfrmtr.baseline, pointings_dircos, units='mks')

## Foreground model backdrop

backdrop_coords = 'dircos'
if use_GSM:
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

if use_GSM:
    backdrop = HP.cartview(fluxes.ravel(), coord=['G','E'], xsize=backdrop_xsize, return_projected_map=True)
elif use_GLEAM:
    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', phase_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
else:
    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
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
            overlay['pbeam'] = pb.reshape(backdrop_xsize, backdrop_xsize)
    overlays += [overlay]

## Animation set up

fig = PLT.figure(figsize=(14,14))
ax1 = fig.add_subplot(211)
ax1.set_xlabel(r'$\eta$ [$\mu$s]', fontsize=18)
if flux_unit == 'Jy':
    ax1.set_ylabel('Amplitude [Jy Hz]', fontsize=18)    
elif flux_unit == 'K':
    ax1.set_ylabel('Amplitude [K Hz]', fontsize=18)
ax1.set_title('Delay Spectrum', fontsize=18, weight='semibold')
ax1.set_yscale('log')
ax1.set_xlim(1e6*NP.amin(lags)-1.0, 1e6*NP.amax(lags)+1.0)
ax1.set_ylim(0.5*NP.amin(NP.abs(vis_lag)),2.0*NP.amax(NP.abs(vis_lag)))
l1 = ax1.plot([], [], 'k+', [], [], 'k-', [], [], 'k-', [], [], 'k:', [], [], 'k:', markersize=10)
ax1.tick_params(which='major', length=12, labelsize=18)
ax1.tick_params(which='minor', length=6)

ax2 = fig.add_subplot(212)
if backdrop_coords == 'radec':
    ax2.set_xlabel(r'$\alpha$ [degrees]', fontsize=18)
    ax2.set_ylabel(r'$\delta$ [degrees]', fontsize=18)
elif backdrop_coords == 'dircos':
    ax2.set_xlabel('l', fontsize=18)
    ax2.set_ylabel('m', fontsize=18)
ax2.set_title('Sky Model', fontsize=18, weight='semibold')
ax2.grid(True)
ax2.tick_params(which='major', length=12, labelsize=18)
ax2.tick_params(which='minor', length=6)

if use_GSM:
    l2init = ax2.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
    cbmn = NP.amin(backdrop)
    cbmx = NP.amax(backdrop)
    cbaxes = fig.add_axes([0.9, 0.1, 0.02, 0.365]) 
    cbar = fig.colorbar(l2init, cax=cbaxes)
    cbmd = 10.0**(0.5*(NP.log10(cbmn)+NP.log10(cbmx)))
    cbar.set_ticks([cbmn, cbmd, cbmx])
    cbar.set_ticklabels([cbmn, cbmd, cbmx])
else:
    ax2.set_xlim(NP.amin(xvect), NP.amax(xvect))
    ax2.set_ylim(NP.amin(yvect), NP.amax(yvect))
    if backdrop_coords == 'radec':
        l2init = ax2.scatter(ra_deg, dec_deg, c=fpeak, marker='.', cmap=PLT.cm.get_cmap("rainbow"), norm=PLTC.LogNorm())
        # cbmn = NP.amin(fpeak)
        # cbmx = NP.amax(fpeak)
    else:
        if (obs_mode == 'drift') or (telescope == 'mwa_dipole'):
            l2init = ax2.imshow(backdrop, origin='lower', extent=(NP.amin(xvect), NP.amax(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm())
            cbaxes = fig.add_axes([0.9, 0.1, 0.02, 0.365]) 
            cbar = fig.colorbar(l2init, cax=cbaxes)

l2 = ax2.plot([], [], 'w.', [], [])

txt1 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, fontsize=18)
txt2 = ax2.text(0.25, 0.8, '', transform=ax2.transAxes, fontsize=18)

def update(i, pointing_radec, lst ,obsmode, telescope, backdrop_coords, lags, vis_lag, delaymatrix, overlays, xv, yv, xv_uniq, yv_uniq, line1, line2, t1, t2):
    line1[0].set_xdata(1e6*lags)
    line1[0].set_ydata(NP.abs(vis_lag[i,:]))

    delay_ranges = NP.hstack((delaymatrix[:,:,1] - delaymatrix[:,:,0], delaymatrix[:,:,1] + delaymatrix[:,:,0]))
    delay_horizon = NP.hstack((-delaymatrix[:,:,0], delaymatrix[:,:,0]))

    line1[1].set_xdata(1e6*delay_ranges[i,0]+NP.zeros(2))
    line1[1].set_ydata(NP.asarray([NP.amin(NP.abs(vis_lag[i,:])), NP.max(NP.abs(vis_lag[i,:]))]))

    line1[2].set_xdata(1e6*delay_ranges[i,1]+NP.zeros(2))
    line1[2].set_ydata(NP.asarray([NP.amin(NP.abs(vis_lag[i,:])), NP.amax(NP.abs(vis_lag[i,:]))]))

    line1[3].set_xdata(1e6*delay_horizon[i,0]+NP.zeros(2))
    line1[3].set_ydata(NP.asarray([NP.amin(NP.abs(vis_lag[i,:])), NP.max(NP.abs(vis_lag[i,:]))]))

    line1[4].set_xdata(1e6*delay_horizon[i,1]+NP.zeros(2))
    line1[4].set_ydata(NP.asarray([NP.amin(NP.abs(vis_lag[i,:])), NP.amax(NP.abs(vis_lag[i,:]))]))

    label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:]) + '\nLST = {0:.1f} deg'.format(lst[i])

    if backdrop_coords == 'radec':
        pbi = griddata(NP.hstack((xv[overlays[i]['roi_obj_inds']].reshape(-1,1),yv[overlays[i]['roi_obj_inds']].reshape(-1,1))), overlays[i]['pbeam'], NP.hstack((xv.reshape(-1,1),yv.reshape(-1,1))), method='nearest')
        line2[0] = ax2.contour(xv_uniq, yv_uniq, pbi.reshape(yv_uniq.size, xv_uniq.size), 35)
    elif backdrop_coords == 'dircos':
        if (obsmode != 'drift') and (telescope != 'mwa_dipole'):
            line2[1] = ax2.imshow(overlays[i]['pbeam'], origin='lower', extent=(NP.amin(xv_uniq), NP.amax(xv_uniq), NP.amin(yv_uniq), NP.amax(yv_uniq)), norm=PLTC.LogNorm())
            cbaxes = fig.add_axes([0.9, 0.1, 0.02, 0.365]) 
            cbar = fig.colorbar(line2[1], cax=cbaxes)
        line2[0].set_xdata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],0])
        line2[0].set_ydata(overlays[i]['fg_dircos'][overlays[i]['roi_obj_inds'],1])
        line2[0].set_marker('.')

    # if interferometer.pointing_coords == 'hadec':
    #     label_str = r' $\alpha$ = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.lst[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
    # elif interferometer.pointing_coords == 'radec':
    #     label_str = r' HA = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.lst[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
    t1.set_text(label_str)
    # # t2.set_text(label_str)
    t2.set_text('')

    return line1, line2, t1, t2

anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lst, obs_mode, telescope, backdrop_coords, lags, vis_lag, delay_matrix, overlays, xvect, yvect, xgrid[0,:], ygrid[:,0], l1, l2, txt1, txt2), frames=len(overlays), interval=interval, blit=False)
PLT.show()
anim.save(outfile+'.mp4', fps=fps, codec='x264')
