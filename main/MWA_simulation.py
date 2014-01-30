import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
import matplotlib.pyplot as PLT
import matplotlib.animation as MOV
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 

catalog_file = '/data3/t_nithyanandan/project_MWA/mwacs_b1_131016.csv'
# catalog_file = '/Users/t_nithyanandan/Downloads/mwacs_b1_131016.csv'

catdata = ascii.read(catalog_file, data_start=1, delimiter=',')

dec_deg = catdata['DEJ2000']
ra_deg = catdata['RAJ2000']
fpeak = catdata['S150_fit']
ferr = catdata['e_S150_fit']
spindex = catdata['Sp+Index']
freq_catalog = 0.150 # in GHz
freq_resolution = 40.0 # in kHz
nchan = 512
chans = freq_catalog + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution * 1e3 / 1e9

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

ctlgobj = CTLG.Catalog(freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak)

skymod = CTLG.SkyModel(ctlgobj)
   
A_eff = 16.0 * (0.5 * FCNST.c / (freq_catalog * 1e9))**2

intrfrmtr = RI.Interferometer('B1', [1000.0, 0.0, 0.0], chans, telescope='mwa',
                              latitude=-26.701, A_eff=A_eff, freq_scale='GHz')

Tsys = 440.0 # in Kelvin
t_snap = 40 * 60.0 # in seconds
# ha_range = 15.0*NP.arange(-1.0, t_snap/3.6e3, 1.0)
n_snaps = 16
lst_obs = (0.0 + (t_snap / 3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
ha_obs = NP.zeros(n_snaps)
dec_obs = intrfrmtr.latitude + NP.zeros(n_snaps)

for i in xrange(n_snaps):
    intrfrmtr.observe(str(lst_obs[i]), Tsys, bpass, [ha_obs[i], dec_obs[i]], skymod, t_snap, fov_radius=30.0, lst=lst_obs[i])

intrfrmtr.delay_transform()
lags = intrfrmtr.lags
vis_lag = intrfrmtr.vis_lag
if oversampling_factor > 1.0:
    lags = DSP.downsampler(intrfrmtr.lags, oversampling_factor)
    vis_lag = DSP.downsampler(intrfrmtr.vis_lag, oversampling_factor)

noise_info = intrfrmtr.band_averaged_noise_estimate(filter_method='hpf')

fig = PLT.figure(figsize=(14,14))
ax1 = fig.add_subplot(211)
# fig, (ax1, ax2) = PLT.subplots(2,1,figsize=(14,12))
ax1.set_xlabel(r'$\eta$ [$\mu$s]', fontsize=18)
ax1.set_ylabel('Amplitude [Jy Hz]', fontsize=18)
ax1.set_title('Delay Spectrum', fontsize=18, weight='semibold')
ax1.set_yscale('log')
ax1.set_xlim(1e6*NP.amin(lags)-1.0, 1e6*NP.amax(lags)+1.0)
ax1.set_ylim(0.5*NP.amin(NP.abs(intrfrmtr.vis_lag)),2.0*NP.amax(NP.abs(intrfrmtr.vis_lag)))
l1, = ax1.plot([], [], 'g+', markersize=10)
ax1.tick_params(which='major', length=12, labelsize=18)
ax1.tick_params(which='minor', length=6)
# ax2 = fig.add_subplot(212)
# ax2.set_xlim(NP.min(skymod.catalog.location[:,0]), NP.max(skymod.catalog.location[:,0]))
# ax2.set_ylim(NP.min(skymod.catalog.location[:,1])-5.0, NP.max(skymod.catalog.location[:,1])+5.0)

ax2 = fig.add_subplot(212, projection='hammer')
ra_deg = skymod.catalog.location[:,0]
neg_ra = skymod.catalog.location[:,0] > 180.0
ra_deg[neg_ra] = ra_deg[neg_ra] - 360.0

ax2.set_xlabel(r'$\alpha$ [degrees]', fontsize=18)
ax2.set_ylabel(r'$\delta$ [degrees]', fontsize=18)
ax2.set_title('Sky Model', fontsize=18, weight='semibold')
# ax2.text(-2.0, -2.0, 'Sky Model', fontsize=18, va='bottom')
ax2.grid(True)
ax2.tick_params(which='major', length=12, labelsize=18)
ax2.tick_params(which='minor', length=6)

# l2init, = ax2.plot(skymod.catalog.location[:,0], skymod.catalog.location[:,1], 'k.', markersize=1)
l2init, = ax2.plot(NP.radians(ra_deg), NP.radians(skymod.catalog.location[:,1]), 'k.', markersize=1)

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

def update(i, interferometer, eta, delay_spectra, line1, line2, t1, t2):
    line1.set_xdata(1e6 * eta)
    line1.set_ydata(NP.abs(delay_spectra[i,:]))
    # line1.set_xdata(1e6 * interferometer.lags)
    # line1.set_ydata(NP.abs(interferometer.vis_lag[i,:]))

    # line2.set_xdata(skymod.catalog.location[NP.asarray(interferometer.obs_catalog_indices[i]),0])
    # line2.set_ydata(skymod.catalog.location[NP.asarray(interferometer.obs_catalog_indices[i]),1])
    line2.set_xdata(NP.radians(ra_deg[NP.asarray(interferometer.obs_catalog_indices[i])]))
    line2.set_ydata(NP.radians(skymod.catalog.location[NP.asarray(interferometer.obs_catalog_indices[i]),1]))

    label_str = r' $\alpha$ = {0:+.3f} deg, $\delta$ = {1:+.2f} deg'.format(float(interferometer.timestamp[i])-interferometer.pointing_center[i,0], interferometer.pointing_center[i,1])
    t1.set_text(label_str)
    # t2.set_text(label_str)
    t2.set_text('')

    return line1, line2, t1, t2

anim = MOV.FuncAnimation(fig, update, fargs=(intrfrmtr, lags, vis_lag, l1, l2, txt1, txt2), frames=vis_lag.shape[0], interval=400, blit=False)
PLT.show()
# # anim.save('/data3/t_nithyanandan/project_MWA/delay_spectrum_animation.gif', fps=2.5, writer='imagemagick')
# anim.save('/Users/t_nithyanandan/Downloads/delay_spectrum_animation_10MHz_1_notch_RECT.gif', fps=2.5, writer='imagemagick')
# # anim.save('/Users/t_nithyanandan/Downloads/delay_spectrum_animation.mp4', fps=2.5, writer='ffmpeg')
# anim.save('/Users/t_nithyanandan/Downloads/delay_spectrum_animation_10MHz_1_notch_RECT.mp4', fps=2.5, writer='ffmpeg')


