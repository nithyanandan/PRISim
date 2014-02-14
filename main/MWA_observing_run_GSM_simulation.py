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

freq = 150.0 * 1e6 # foreground center frequency in Hz

gsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata.fits'
obs_file = '/data3/t_nithyanandan/project_MWA/MWA_interferometer_gsm_drift.B1.fits'

hdulist = fits.open(gsm_file)
flux_unit = 'K'
gsm_table = hdulist[1].data
ra_deg = gsm_table['RA']
dec_deg = gsm_table['DEC']
fluxes = gsm_table['f_{0:.0f}'.format(freq/1e6)]
spindex = gsm_table['spindex']
freq_catalog = freq/1e9 # in GHz
freq_resolution = 40.0 # in kHz
nchan = 256
chans = freq_catalog + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution * 1e3 / 1e9 # in GHz
MWA_latitude = -26.701

bpass = 1.0*NP.ones(nchan)

# Do the following next few lines only for MWA
notch_interval = NP.round(1.28e6 / (freq_resolution * 1e3))
# bpass[::notch_interval] = 0.0
# bpass[1::notch_interval] = 0.0
# bpass[2::notch_interval] = 0.0

oversampling_factor = 1.0
# window = DSP.shaping(nchan, 1/oversampling_factor*CNST.rect_bnw_ratio, shape='bnw', peak=1.0)
window = DSP.shaping(nchan, 1/oversampling_factor, shape='rect', peak=1.0)
bpass *= window

ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex)
# ctlgobj = CTLG.Catalog(freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak)

skymod = CTLG.SkyModel(ctlgobj)
   
A_eff = 16.0 * (0.5 * FCNST.c / (freq_catalog * 1e9))**2

obs_mode = 'track'
Tsys = 440.0 # in Kelvin
t_snap = 2*60.0 # in seconds 
t_obs = 3600.0 # in seconds
pointing_init = [180.0, MWA_latitude] # in degrees
lst_init = 12.0  # in hours

# intrfrmtr = RI.Interferometer('B1', [40.0, 0.0, 0.0], chans, telescope='mwa',
#                               latitude=MWA_latitude, A_eff=A_eff, freq_scale='GHz')

# intrfrmtr.observing_run(pointing_init, skymod, t_snap, t_obs, chans, bpass, Tsys, lst_init, mode=obs_mode, freq_scale='GHz', brightness_units=flux_unit)

# intrfrmtr.delay_transform()
# lags = intrfrmtr.lags
# vis_lag = intrfrmtr.vis_lag
# if oversampling_factor > 1.0:
#     lags = DSP.downsampler(intrfrmtr.lags, oversampling_factor)
#     vis_lag = DSP.downsampler(intrfrmtr.vis_lag, oversampling_factor)

band_avg_noise_info = intrfrmtr.band_averaged_noise_estimate(filter_method='hpf')
# freq_diff_noise = intrfrmtr.freq_differenced_noise_estimate()

# intrfrmtr.save('/data3/t_nithyanandan/project_MWA/MWA_interferometer_gsm_'+obs_mode, verbose=True, tabtype='BinTableHDU', overwrite=True)

## Delay spectrum data

hdulist = fits.open(obs_file)
extnames = [hdu.header['EXTNAME'] for hdu in hdulist]
lags = hdulist['SPECTRAL INFO'].data['lag']
vis_lag = hdulist['REAL_LAG_VISIBILITY'].data + 1j * hdulist['IMAG_LAG_VISIBILITY'].data

## GSM and Power pattern animation data

gsm_cartesian = HP.cartview(fluxes.ravel(), coord=['G','E'], xsize=200, return_projected_map=True)
ragrid, decgrid = NP.meshgrid(NP.linspace(NP.amin(ra_deg), NP.amax(ra_deg), gsm_cartesian.shape[1]), NP.linspace(NP.amin(dec_deg), NP.amax(dec_deg), gsm_cartesian.shape[0]))
ravect = ragrid.ravel()
decvect = decgrid.ravel()
n_snaps = int(t_obs/t_snap)
lst = (lst_init + (t_snap/3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
# Get pointings in RA-Dec coordinates for animation
if obs_mode == 'track':
    pointings_radec = NP.repeat(NP.asarray(pointing_init).reshape(-1,2), n_snaps, axis=0)
else:
    pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))

pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
pointings_altaz = GEOM.hadec2altaz(pointings_hadec, MWA_latitude, units='degrees')
pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
delay_matrix = DLY.delay_envelope(intrfrmtr.baseline, pointings_dircos, units='mks')

pbeams = []
m2s = []
for i in xrange(n_snaps):
    havect = lst[i] - ravect
    altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),decvect.reshape(-1,1))), MWA_latitude, units='degrees')
    # m1, m2, d12 = GEOM.spherematch(pointings[i,0], pointings[i,1], ravect, decvect, 90.0, maxmatches=0)
    roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
    pb = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope='mwa', skyunits='altaz', freq_scale='Hz', phase_center=pointings_altaz[i,:])
    pbeams += [pb]
    m2s += [roi_altaz]

## Plotting animation

fig = PLT.figure(figsize=(14,14))
ax1 = fig.add_subplot(211)
# fig, (ax1, ax2) = PLT.subplots(2,1,figsize=(14,12))
ax1.set_xlabel(r'$\eta$ [$\mu$s]', fontsize=18)
ax1.set_ylabel('Amplitude [K Hz]', fontsize=18)
ax1.set_title('Delay Spectrum', fontsize=18, weight='semibold')
ax1.set_yscale('log')
ax1.set_xlim(1e6*NP.amin(lags)-1.0, 1e6*NP.amax(lags)+1.0)
ax1.set_ylim(0.5*NP.amin(NP.abs(vis_lag)),2.0*NP.amax(NP.abs(vis_lag)))
l1 = ax1.plot([], [], 'k+', [], [], 'k-', [], [], 'k-', [], [], 'k:', [], [], 'k:', markersize=10)
ax1.tick_params(which='major', length=12, labelsize=18)
ax1.tick_params(which='minor', length=6)
# ax2 = fig.add_subplot(212)
# ax2.set_xlim(NP.min(skymod.catalog.location[:,0]), NP.max(skymod.catalog.location[:,0]))
# ax2.set_ylim(NP.min(skymod.catalog.location[:,1])-5.0, NP.max(skymod.catalog.location[:,1])+5.0)

ax2 = fig.add_subplot(212)
ax2.set_xlabel(r'$\alpha$ [degrees]', fontsize=18)
ax2.set_ylabel(r'$\delta$ [degrees]', fontsize=18)
ax2.set_title('Sky Model', fontsize=18, weight='semibold')
# ax2.text(-2.0, -2.0, 'Sky Model', fontsize=18, va='bottom')
ax2.grid(True)
ax2.tick_params(which='major', length=12, labelsize=18)
ax2.tick_params(which='minor', length=6)

# l2init, = ax2.plot(skymod.catalog.location[:,0], skymod.catalog.location[:,1], 'k.', markersize=1)
l2init = ax2.imshow(gsm_cartesian, origin='lower', extent=(NP.amin(ravect), NP.amax(ravect), NP.amin(decvect), NP.amax(decvect)), norm=PLTC.LogNorm())
cbaxes = fig.add_axes([0.9, 0.1, 0.02, 0.365]) 
# cb = fig.colorbar(ax2, cax = cbaxes)  
cbar = fig.colorbar(l2init, cax=cbaxes)
cbmn = NP.amin(gsm_cartesian)
cbmx = NP.amax(gsm_cartesian)
cbmd = 10.0**(0.5*(NP.log10(cbmn)+NP.log10(cbmx)))
cbar.set_ticks([cbmn, cbmd, cbmx])
cbar.set_ticklabels([cbmn, cbmd, cbmx])

l2, = ax2.plot([], [], 'g+', markersize=3)

txt1 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, fontsize=18)
txt2 = ax2.text(0.25, 0.8, '', transform=ax2.transAxes, fontsize=18)

# def init():
#     l1.set_xdata([])
#     l1.set_ydata([])
#     l2.set_xdata(skymod.catalog.location[:,0])
#     l2.set_ydata(skymod.catalog.location[:,1])
#     l2.set_marker('.')
#     txt1.set_text('')
#     txt2.set_text('')
#     return l1, l2, txt1, txt2

def update(i, pointing_radec, lags, vis_lag, delaymatrix, pbs, m2list, ra, dec, ra_uniq, dec_uniq, line1, line2, t1, t2):
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

    pbi = griddata(NP.hstack((ra[m2s[i]].reshape(-1,1),dec[m2s[i]].reshape(-1,1))), pbs[i], NP.hstack((ra.reshape(-1,1),dec.reshape(-1,1))), method='nearest')
    line2 = ax2.contour(ra_uniq, dec_uniq, pbi.reshape(dec_uniq.size, ra_uniq.size), 35)

    label_str = r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(pointing_radec[i,:])
    # if interferometer.pointing_coords == 'hadec':
    #     label_str = r' $\alpha$ = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.lst[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
    # elif interferometer.pointing_coords == 'radec':
    #     label_str = r' HA = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.lst[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
    t1.set_text(label_str)
    # # t2.set_text(label_str)
    t2.set_text('')

    return line1, line2, t1, t2

anim = MOV.FuncAnimation(fig, update, fargs=(pointings_radec, lags, vis_lag, delay_matrix, pbeams, m2s, ravect, decvect, ragrid[0,:], decgrid[:,0], l1, l2, txt1, txt2), frames=len(pbeams), interval=100, blit=False)
PLT.show()
# # anim.save('/data3/t_nithyanandan/project_MWA/delay_spectrum_animation.gif', fps=2.5, writer='imagemagick')
# anim.save('/Users/t_nithyanandan/Downloads/delay_spectrum_animation_10MHz_1_notch_RECT.gif', fps=2.5, writer='imagemagick')
# # anim.save('/Users/t_nithyanandan/Downloads/delay_spectrum_animation.mp4', fps=2.5, writer='ffmpeg')
# anim.save('/Users/t_nithyanandan/Downloads/delay_spectrum_animation_10MHz_1_notch_RECT.mp4', fps=2.5, writer='ffmpeg')


