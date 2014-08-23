import numpy as NP
from astropy.io import fits
from astropy.io import ascii
from astropy import coordinates as coord
from astropy.coordinates import Galactic, FK5
from astropy import units
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import healpy as HP
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import lookup_operations as LKP
import ipdb as PDB

# 01) Pick representative baselines and show individual contributions from point
#     sources and diffuse emission 

# 02) Show cleaned delay spectra as a function of baseline length and delay

# 03) Plot EoR window power and wedge power as a function of LST for quality
#     assurance purposes from different beamformer settings

# 04) Plot sky power as a function of LST 

# 05) Plot fraction of pixels relatively free of contamination as a function
#     of baseline length

plot_01 = False
plot_02 = False
plot_03 = True
plot_04 = False
plot_05 = False

# PLT.ioff()
PLT.ion()

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

telescope = 'mwa'
telescope_str = telescope + '_'
if telescope == 'mwa':
    telescope_str = ''
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
bl, bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True) 
neg_bl_orientation_ind = bl_orientation < 0.0
bl[neg_bl_orientation_ind,:] = -1.0 * bl[neg_bl_orientation_ind,:]
bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_length = bl_length[sortind]
bl_orientation = bl_orientation[sortind]
bl_id = bl_id[sortind]
n_bins_baseline_orientation = 4
nmax_baselines = 2000
bl = bl[:nmax_baselines,:]
bl_length = bl_length[:nmax_baselines]
bl_id = bl_id[:nmax_baselines]
bl_orientation = bl_orientation[:nmax_baselines]
total_baselines = bl_length.size
nside = 128
Tsys = 300.0 # System temperature in K
freq = 185.0e6 # center frequency in Hz
max_abs_delay = None # in micro seconds
oversampling_factor = 2.0
n_sky_sectors = 1
sky_sector = None # if None, use all sky sector. Accepted values are None, 0, 1, 2, or 3
if sky_sector is None:
    sky_sector_str = '_all_sky_'
    n_sky_sectors = 1
    sky_sector = 0
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(sky_sector)

if plot_01 or plot_02:

    #############################################################################
    # 01) Pick representative baselines and show individual contributions from 
    #     point sources and diffuse emission 

    # 02) Show cleaned delay spectra as a function of baseline length and delay
    
    obs_mode = 'custom'
    freq_resolution = 80e3
    nchan = 384
    bpass_shape = 'bnw'
    snapshot_type_str = ''
    dalpha = 0.35
    csm_ref_freq = NP.sqrt(1420.0 * 843.0) * 1e6
    jacobian_spindex = NP.abs(dalpha * NP.log(freq/csm_ref_freq))
    use_fhd_data = True
    use_unbiased = True
    if use_unbiased:
        bias_str = 'unbiased'
    else:
        bias_str = 'biased'

    pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    pc_coords = 'dircos'
    
    csm_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[0],bl_length[-1])+'gaussian_FG_model_csm_all_sky_nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+'.fits'

    csm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[0],bl_length[-1])+'gaussian_FG_model_csm_all_sky_nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'.fits'
    dsm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[0],bl_length[-1])+'gaussian_FG_model_dsm_all_sky_nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'.fits'
    asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[0],bl_length[-1])+'gaussian_FG_model_asm_all_sky_nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'.fits'

    fhd_obsid = [1061309344, 1061316544]
    
    hdulist = fits.open(csm_infile)
    lst = hdulist['POINTING AND PHASE CENTER INFO'].data['LST']
    csm_freq_resolution = hdulist[0].header['freq_resolution']
    vis_rms_freq = hdulist['freq_channel_noise_rms_visibility'].data
    vis_rms_lag = NP.sqrt(vis_rms_freq.shape[1]*1.0) * csm_freq_resolution * NP.mean(vis_rms_freq)
    bp = hdulist['bandpass'].data
    bp_wts = hdulist['bandpass_weights'].data

    # bl = hdulist['baselines'].data
    # bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1])
    neg_bl_orientation_ind = bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    bl_orientation[neg_bl_orientation_ind] -= 180.0
    bl[neg_bl_orientation_ind,:] *= -1.0

    hdulist = fits.open(csm_CLEAN_infile)
    clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    csm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    csm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    csm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    csm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    csm_cc_skyvis = csm_cc_skyvis + csm_cc_skyvis_res
    csm_cc_vis = csm_cc_vis + csm_cc_vis_res

    # csm_cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data
    # csm_ccres_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data
    # csm_cc_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA IMAG'].data
    # csm_ccres_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG'].data
    # csm_cc_vis_lag = csm_cc_vis_lag + csm_ccres_vis_lag
    hdulist.close()

    hdulist = fits.open(dsm_CLEAN_infile)
    dsm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    dsm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    dsm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    dsm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    dsm_cc_skyvis = dsm_cc_skyvis + dsm_cc_skyvis_res
    dsm_cc_vis = dsm_cc_vis + dsm_cc_vis_res

    # dsm_cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data
    # dsm_ccres_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data
    # dsm_cc_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA IMAG'].data
    # dsm_ccres_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG'].data
    # dsm_cc_vis_lag = dsm_cc_vis_lag + dsm_ccres_vis_lag
    hdulist.close()

    hdulist = fits.open(asm_CLEAN_infile)
    asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    asm_cc_skyvis = asm_cc_skyvis + asm_cc_skyvis_res
    asm_cc_vis = asm_cc_vis + asm_cc_vis_res

    # asm_cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data
    # asm_ccres_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data
    # asm_cc_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA IMAG'].data
    # asm_ccres_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG'].data
    # asm_cc_vis_lag = asm_cc_vis_lag + asm_ccres_vis_lag
    hdulist.close()

    csm_cc_vis[neg_bl_orientation_ind,:,:] = csm_cc_vis[neg_bl_orientation_ind,:,:].conj()
    dsm_cc_vis[neg_bl_orientation_ind,:,:] = dsm_cc_vis[neg_bl_orientation_ind,:,:].conj()
    asm_cc_vis[neg_bl_orientation_ind,:,:] = asm_cc_vis[neg_bl_orientation_ind,:,:].conj()
    csm_cc_skyvis[neg_bl_orientation_ind,:,:] = csm_cc_skyvis[neg_bl_orientation_ind,:,:].conj()
    dsm_cc_skyvis[neg_bl_orientation_ind,:,:] = dsm_cc_skyvis[neg_bl_orientation_ind,:,:].conj()
    asm_cc_skyvis[neg_bl_orientation_ind,:,:] = asm_cc_skyvis[neg_bl_orientation_ind,:,:].conj()

    csm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis, axis=1), axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
    dsm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis, axis=1), axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
    asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis, axis=1), axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
    csm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis, axis=1), axes=1) * csm_cc_vis.shape[1] * freq_resolution
    dsm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis, axis=1), axes=1) * dsm_cc_vis.shape[1] * freq_resolution
    asm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis, axis=1), axes=1) * asm_cc_vis.shape[1] * freq_resolution

    csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags.size/nchan, axis=1)
    dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags.size/nchan, axis=1)
    asm_cc_skyvis_lag = DSP.downsampler(asm_cc_skyvis_lag, 1.0*clean_lags.size/nchan, axis=1)
    csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags.size/nchan, axis=1)
    dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags.size/nchan, axis=1)
    asm_cc_vis_lag = DSP.downsampler(asm_cc_vis_lag, 1.0*clean_lags.size/nchan, axis=1)
    clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/nchan, axis=-1)
    clean_lags = clean_lags.ravel()
    lags = NP.copy(clean_lags)

    if max_abs_delay is not None:
        small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
        clean_lags = clean_lags[small_delays_ind]
        asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]
        csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]
        dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
        asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
        csm_cc_vis_lag = csm_cc_vis_lag[:,small_delays_ind,:]
        dsm_cc_vis_lag = dsm_cc_vis_lag[:,small_delays_ind,:]
    else:
        small_delays_ind = NP.arange(clean_lags.size)
    
    ## Below to be incorporated ##
    
    asm_cc_skyvis_lag_noisy4 = NP.empty_like(asm_cc_skyvis_lag)
    dsm_cc_skyvis_lag_noisy4 = NP.empty_like(dsm_cc_skyvis_lag)
    csm_cc_skyvis_lag_noisy4 = NP.empty_like(csm_cc_skyvis_lag)

    asm_cc_skyvis_power_lag_modified = NP.empty_like(asm_cc_skyvis_lag_noisy4)
    dsm_cc_skyvis_power_lag_modified = NP.empty_like(dsm_cc_skyvis_lag_noisy4)
    csm_cc_skyvis_power_lag_modified = NP.empty_like(csm_cc_skyvis_lag_noisy4)

    for i in xrange(lst.size):
        vis_noise_freq = NP.sqrt(4) * vis_rms_freq[:,:,i].reshape(bl.shape[0],nchan,-1) / NP.sqrt(2.0) * (NP.random.randn(bl.shape[0], nchan, 4) + 1j * NP.random.randn(bl.shape[0], nchan, 4)) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts
        vis_noise_lag = DSP.FT1D(vis_noise_freq * bp[:,:,i].reshape(bl.shape[0],nchan,-1) * bp_wts, ax=1, inverse=True, use_real=False, shift=True) * nchan * freq_resolution
        vis_noise_lag = vis_noise_lag[:,small_delays_ind,:]

        asm_cc_skyvis_lag_noisy4 = asm_cc_skyvis_lag[:,:,i].reshape(bl.shape[0],clean_lags.size,-1) + vis_noise_lag
        temp = 0.5 * (NP.abs(NP.sum(asm_cc_skyvis_lag_noisy4, axis=2))**2 - NP.sum(NP.abs(asm_cc_skyvis_lag_noisy4)**2, axis=2))
        asm_cc_skyvis_power_lag_modified[:,:,i] = temp

        dsm_cc_skyvis_lag_noisy4 = dsm_cc_skyvis_lag[:,:,i].reshape(bl.shape[0],clean_lags.size,-1) + vis_noise_lag
        temp = 0.5 * (NP.abs(NP.sum(dsm_cc_skyvis_lag_noisy4, axis=2))**2 - NP.sum(NP.abs(dsm_cc_skyvis_lag_noisy4)**2, axis=2))
        dsm_cc_skyvis_power_lag_modified[:,:,i] = temp

        csm_cc_skyvis_lag_noisy4 = csm_cc_skyvis_lag[:,:,i].reshape(bl.shape[0],clean_lags.size,-1) + vis_noise_lag
        temp = 0.5 * (NP.abs(NP.sum(csm_cc_skyvis_lag_noisy4, axis=2))**2 - NP.sum(NP.abs(csm_cc_skyvis_lag_noisy4)**2, axis=2))
        csm_cc_skyvis_power_lag_modified[:,:,i] = temp

    ## Above to be incorporated ##

    if plot_01:

        # Pick representative baselines and show individual contributions from point
        # sources and diffuse emission 

        bl_id_ref = ['93-86', '58-31', '68-51', '31-12', '155-154', '72-34', '76-75', '51-28', '48-28', '48-18']
        
        # for i in xrange(len(bl_id_ref)):
            # for j in range(asm_cc_skyvis_lag.shape[2]):
            #     bl_ind = NP.asarray(NP.where(bl_id == bl_id_ref[i])).ravel()
            #     delay_matrix = DLY.delay_envelope(bl[bl_ind,:], pc, units='mks')
            #     min_delay = -delay_matrix[0,0,1]-delay_matrix[0,0,0]
            #     max_delay = delay_matrix[0,0,0]-delay_matrix[0,0,1]
            
            #     fig = PLT.figure(figsize=(6,6))
            #     ax = fig.add_subplot(111)
            #     ax.plot(clean_lags*1e6, NP.abs(asm_cc_skyvis_lag[bl_ind,:,j]).ravel(), 'k-', lw=2, label='ASM')
            #     ax.plot(clean_lags*1e6, NP.abs(csm_cc_skyvis_lag[bl_ind,:,j]).ravel(), 'k:', lw=2, label='CSM')
            #     ax.plot(clean_lags*1e6, NP.abs(dsm_cc_skyvis_lag[bl_ind,:,j]).ravel(), 'k--', lw=2, label='DSM')
            #     ax.plot(min_delay*1e6*NP.ones(2), NP.asarray([1e-4, 1e1]), lw=2, color='0.5')
            #     ax.plot(max_delay*1e6*NP.ones(2), NP.asarray([1e-4, 1e1]), lw=2, color='0.5')
            #     ax.set_xlim(1e6*clean_lags.min(), 1e6*clean_lags.max())
            #     ax.set_ylim(0.5*(NP.abs(asm_cc_skyvis_lag[bl_ind,:,j]).ravel()-NP.sqrt(NP.abs(jacobian_spindex * asm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel()).min(), 2*(NP.abs(asm_cc_skyvis_lag[bl_ind,:,j]).ravel()+NP.sqrt(NP.abs(jacobian_spindex * asm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel()).max())
            #     ax.set_xlabel(r'lag [$\mu$s]', fontsize=18)
            #     ax.set_ylabel('Delay Visibility Amplitude [Jy Hz]', fontsize=18)
            #     ax.set_yscale('log')
            #     ax.text(0.6, 0.85, 'East: {0[0]:+.1f} m \nNorth: {0[1]:+.1f} m \nUp: {0[2]:+.1f} m'.format(bl[bl_ind,:].ravel()), transform=ax.transAxes, fontsize=15)
            #     ax.text(0.33, 0.92, bl_id_ref[i], transform=ax.transAxes, fontsize=15)
            #     legend = ax.legend(loc='upper left')
            #     legend.draw_frame(False)
            #     ax.tick_params(which='major', length=18, labelsize=12)
            #     ax.tick_params(which='minor', length=12, labelsize=12)
            #     for axis in ['top','bottom','left','right']:
            #         ax.spines[axis].set_linewidth(2)
            #     xticklabels = PLT.getp(ax, 'xticklabels')
            #     yticklabels = PLT.getp(ax, 'yticklabels')
            #     PLT.setp(xticklabels, fontsize=15, weight='medium')
            #     PLT.setp(yticklabels, fontsize=15, weight='medium') 
            
            #     PLT.tight_layout()
            #     fig.subplots_adjust(right=0.95)
            #     fig.subplots_adjust(left=0.15)
            
            #     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_'+bl_id_ref[i]+'_composite_noiseless_delay_spectrum_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)
            #     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_'+bl_id_ref[i]+'_composite_noiseless_delay_spectrum_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

        for i in xrange(len(bl_id_ref)):
            for j in range(asm_cc_skyvis_lag.shape[2]):
                if use_fhd_data:
                    ants = bl_id_ref[i].split('-')
                    ant1 = ants[1]
                    ant2 = ants[0]
                    fhd_infile = '/home/t_nithyanandan/codes/others/python/Danny/{0:0d}'.format(fhd_obsid[j])+'.fhd.p.npz'
                    fhd_data = NP.load(fhd_infile)
                    fhd_ant1 = fhd_data['ant1']
                    fhd_ant2 = fhd_data['ant2']
                    fhd_C = fhd_data['C']
                    fhd_bl_length = NP.sqrt(NP.sum(fhd_data['uvws']**2, axis=1)) * 1e-9 * FCNST.c
                    blind = NP.logical_and(fhd_ant1 == int(ant1), fhd_ant2 == int(ant2))
                    fhd_delays = NP.fft.fftshift(fhd_data['delays']) * 1e-9
                    if use_unbiased:
                        fhd_vis_lag = NP.sqrt(NP.abs(fhd_data['P'][blind,:,0].ravel())**2 - fhd_data['P2'][blind,:,0].ravel()) * 30.72e6 
                    else:
                        fhd_vis_lag = fhd_data['P'][blind,:,0].ravel() * 30.72e6 
                    if not fhd_C[blind,0,0] <= 0:
                        fhd_vis_lag /= NP.sqrt(fhd_C[blind,0,0])

                bl_ind = NP.asarray(NP.where(bl_id == bl_id_ref[i])).ravel()
                delay_matrix = DLY.delay_envelope(bl[bl_ind,:], pc, units='mks')
                min_delay = -delay_matrix[0,0,1]-delay_matrix[0,0,0]
                max_delay = delay_matrix[0,0,0]-delay_matrix[0,0,1]
            
                PDB.set_trace()

                fig = PLT.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                if not use_unbiased:
                    ax.plot(clean_lags*1e6, NP.abs(asm_cc_vis_lag[bl_ind,:,j]).ravel(), 'k-', lw=2, label='ASM')
                    ax.plot(clean_lags*1e6, NP.abs(csm_cc_skyvis_lag[bl_ind,:,j]).ravel(), 'k:', lw=2, label='CSM')
                    ax.plot(clean_lags*1e6, NP.abs(dsm_cc_skyvis_lag[bl_ind,:,j]).ravel(), 'k--', lw=2, label='DSM')
                    ax.fill_between(clean_lags*1e6, NP.abs(asm_cc_vis_lag[bl_ind,:,j]).ravel()+NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel(), NP.abs(asm_cc_vis_lag[bl_ind,:,j]).ravel()-NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel(), alpha=0.75, edgecolor='none', facecolor='gray')
                else:
                    ax.plot(clean_lags*1e6, NP.sqrt(NP.abs(asm_cc_skyvis_power_lag_modified[bl_ind,:,j])).ravel(), 'k-', lw=2, label='ASM')
                    ax.plot(clean_lags*1e6, NP.sqrt(NP.abs(dsm_cc_skyvis_power_lag_modified[bl_ind,:,j])).ravel(), 'k-', lw=2, label='DSM')
                    ax.plot(clean_lags*1e6, NP.sqrt(NP.abs(csm_cc_skyvis_power_lag_modified[bl_ind,:,j])).ravel(), 'k-', lw=2, label='CSM')
                    ax.fill_between(clean_lags*1e6, NP.sqrt(NP.abs(asm_cc_skyvis_power_lag_modified[bl_ind,:,j])).ravel()+NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel(), NP.sqrt(NP.abs(asm_cc_skyvis_power_lag_modified[bl_ind,:,j])).ravel()-NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel(), alpha=0.75, edgecolor='none', facecolor='gray')                    

                if use_fhd_data:
                    ax.plot(fhd_delays*1e6, NP.abs(fhd_vis_lag), 'r.-')

                # ax.plot(clean_lags*1e6, NP.abs(asm_cc_skyvis_lag[bl_ind,:,j]).ravel()+NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel(), '-', lw=2, color='gray')
                # ax.plot(clean_lags*1e6, NP.abs(asm_cc_skyvis_lag[bl_ind,:,j]).ravel()-NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel(), '-', lw=2, color='gray')

                ax.axvline(x=min_delay*1e6, lw=2, color='gray')
                ax.axvline(x=max_delay*1e6, lw=2, color='gray')
                # ax.plot(min_delay*1e6*NP.ones(2), NP.asarray([1e-2, 1e1]), lw=2, color='0.5')
                # ax.plot(max_delay*1e6*NP.ones(2), NP.asarray([1e-2, 1e1]), lw=2, color='0.5')
                ax.set_xlim(1e6*clean_lags.min(), 1e6*clean_lags.max())
                ax.set_ylim(0.0, 1.1*(NP.abs(asm_cc_vis_lag[bl_ind,:,j]).ravel()+NP.sqrt(NP.abs(jacobian_spindex * csm_cc_skyvis_lag[bl_ind,:,j])**2 + vis_rms_lag**2).ravel()).max())
                ax.set_xlabel(r'lag [$\mu$s]', fontsize=18)
                ax.set_ylabel('Delay Visibility Amplitude [Jy Hz]', fontsize=18)
                # ax.set_yscale('log')
                ax.text(0.6, 0.85, 'East: {0[0]:+.1f} m \nNorth: {0[1]:+.1f} m \nUp: {0[2]:+.1f} m'.format(bl[bl_ind,:].ravel()), transform=ax.transAxes, fontsize=15)
                ax.text(0.33, 0.92, bl_id_ref[i], transform=ax.transAxes, fontsize=15)
                legend = ax.legend(loc='upper left')
                legend.draw_frame(False)
                ax.tick_params(which='major', length=18, labelsize=12)
                ax.tick_params(which='minor', length=12, labelsize=12)
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(2)
                xticklabels = PLT.getp(ax, 'xticklabels')
                yticklabels = PLT.getp(ax, 'yticklabels')
                PLT.setp(xticklabels, fontsize=15, weight='medium')
                PLT.setp(yticklabels, fontsize=15, weight='medium') 
            
                PLT.tight_layout()
                fig.subplots_adjust(right=0.95)
                fig.subplots_adjust(left=0.15)
            
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_'+bl_id_ref[i]+'_composite_'+bias_str+'_noisy_delay_spectrum_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_'+bl_id_ref[i]+'_composite_'+bias_str+'_noisy_delay_spectrum_snapshot_{0:0d}.png'.format(j), bbox_inches=0)

    if plot_02:
        
        # Show cleaned delay spectra as a function of baseline length and delay

        # xconv = lambda x: bl_length[int(x)]
        xconv = lambda x: '{0:.1f}'.format(bl_length[int(min(x, bl_length.size-1))])

        for fg_str in ['csm', 'dsm', 'asm']:
            if fg_str == 'csm':
                noiseless_cc_vis_lag = csm_cc_skyvis_lag
            if fg_str == 'dsm':
                noiseless_cc_vis_lag = dsm_cc_skyvis_lag
            if fg_str == 'asm':
                noiseless_cc_vis_lag = asm_cc_skyvis_lag

            for i in xrange(noiseless_cc_vis_lag.shape[2]):
    
                if fg_str == 'csm':
                    if i == 0:
                        texts = ['1-C-E-E-P', '1-C-E-N-P', '1-C-W-E-S3', '1-C-NE-NE-S2']
                        xy = [(1500, 0.33), (1250, 0.0), (1750, -1), (1750, 1)]
                        xy_text = [(1000, 1), (750, -1), (1000, -1.5), (1000, 1.5)]
                    if i == 1:
                        texts = ['2-C-Z-A-P', '2-C-N-N-S2', '2-C-S-N-S2', '2-C-N-N-S1', '2-C-S-N-S1']
                        xy = [(1000, 0), (1250, 0.5), (1250, -0.5), (1750, 0.5), (1750, -0.5)]
                        xy_text = [(500, 0.5), (750, 1), (750, -1), (1250, 1.5), (1250, -1.5)]

                if fg_str == 'dsm':
                    if i == 0:
                        texts = ['1-GC-W-E-S3', '1-GP-NE-NE-S2']
                        xy = [(500, -0.25), (1250, 0.5)]
                        xy_text = [(500, -1.5), (1250, 1.5)]
                    if i == 1:
                        texts = ['2-D-Z-A-P', '2-GP-S-N-S2', '2-GP-N-N-S2']
                        xy = [(750, 0), (1750, -1), (1750, 1)]
                        xy_text = [(1500, 0), (1500, -1.5), (1500, 1.5)]

                if fg_str == 'asm':
                    if i == 0:
                        texts = ['1-GC-W-E-S3', '1-GP-NE-NE-S2', '1-C-E-E-P', '1-C-E-N-P']
                        xy = [(500, -0.25), (1250, 0.5), (1500, 0.33), (1250, 0.0)]
                        xy_text = [(500, -1.5), (1250, 1.5), (1500, 1), (1250, -1)]
                    if i == 1:
                        texts = ['2-GP-S-N-S2', '2-GP-N-N-S2', '2-C-Z-A-P', '2-C-N-N-S1', '2-C-S-N-S1']
                        xy = [(1750, -1), (1750, 1), (1000, 0), (1500, 0.3), (1500, -0.3)]          
                        xy_text = [(1500, -2), (1500, 2), (1000, 0.75), (1250, 1.5), (1250, -1.5)]

                fig = PLT.figure(figsize=(5,5))
                
                ax1 = fig.add_subplot(111)
                # ax1.set_xlabel('Baseline Length [m]', fontsize=18)
                # ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
                # dspec1 = ax1.pcolorfast(bl_length, 1e6*clean_lags, NP.abs(noiseless_cc_vis_lag[:-1,:-1,i].T), norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_vis_lag))))
                # ax1.set_xlim(bl_length[0], bl_length[-1])
                # ax1.set_ylim(1e6*clean_lags[0], 1e6*clean_lags[-1])
                ax1.set_xlabel('Baseline Index', fontsize=18)
                ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
                ax1.tick_params(which='major', length=12, labelsize=12, color='white')
                ax1.tick_params(which='minor', length=6, labelsize=12, color='white')
                dspec1 = ax1.imshow(NP.abs(noiseless_cc_vis_lag[:,:,i].T), origin='lower', extent=(0, noiseless_cc_vis_lag.shape[0]-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e7, vmax=NP.amax(NP.abs(asm_cc_skyvis_lag))), interpolation=None)
                ax1.set_aspect('auto')
                
                for k in xrange(len(texts)):
                    if fg_str == 'csm':
                        ax1.annotate(texts[k], xy=xy[k], xytext=xy_text[k], color='w', horizontalalignment='left', arrowprops=dict(facecolor='white', edgecolor='none', shrink=0.05, frac=0.2, width=2, headwidth=6))
                    if fg_str == 'dsm':
                        ax1.annotate(texts[k], xy=xy[k], xytext=xy_text[k], color='w', horizontalalignment='center', verticalalignment='center', arrowprops=dict(facecolor='white', edgecolor='none', shrink=0.05, frac=0.2, width=2, headwidth=6))        
                    if fg_str == 'asm':
                        ax1.annotate(texts[k], xy=xy[k], xytext=xy_text[k], color='w', horizontalalignment='center', verticalalignment='center', arrowprops=dict(facecolor='white', edgecolor='none', shrink=0.05, frac=0.2, width=2, headwidth=6))        
                    
                ax2 = ax1.twiny()
                ax2.set_xlabel('Baseline Length [m]', fontsize=18)
                # ax2.set_xlim(*map(xconv, ax1.get_xlim()))
                ax2.set_xticks(NP.asarray(ax1.get_xticks()))
                ax2.set_xticklabels(map(xconv, ax1.get_xticks()))
                ax2.tick_params(which='major', length=12, labelsize=12, color='white')
                ax2.tick_params(which='minor', length=6, labelsize=12, color='white')
    
                cbax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
                cb = fig.colorbar(dspec1, cax=cbax, orientation='vertical')
                cbax.set_ylabel('Jy Hz', labelpad=-60, fontsize=18)
                
                PLT.tight_layout()
                fig.subplots_adjust(right=0.8)
                fig.subplots_adjust(left=0.13)
                
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'annotated_combined_baseline_noiseless_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.eps'.format(i), bbox_inches=0)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'annotated_combined_baseline_noiseless_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.png'.format(i), bbox_inches=0)

        for fg_str in ['csm', 'dsm', 'asm']:
            if fg_str == 'csm':
                noisy_cc_vis_lag = csm_cc_vis_lag
            if fg_str == 'dsm':
                noisy_cc_vis_lag = dsm_cc_vis_lag
            if fg_str == 'asm':
                noisy_cc_vis_lag = asm_cc_vis_lag

            for i in xrange(noisy_cc_vis_lag.shape[2]):
    
                if fg_str == 'csm':
                    if i == 0:
                        texts = ['1-C-E-E-P', '1-C-E-N-P', '1-C-W-E-S3', '1-C-NE-NE-S2']
                        xy = [(1500, 0.33), (1250, 0.0), (1750, -1), (1750, 1)]
                        xy_text = [(1000, 1), (750, -1), (1000, -1.5), (1000, 1.5)]
                    if i == 1:
                        texts = ['2-C-Z-A-P', '2-C-N-N-S2', '2-C-S-N-S2', '2-C-N-N-S1', '2-C-S-N-S1']
                        xy = [(1000, 0), (1250, 0.5), (1250, -0.5), (1750, 0.5), (1750, -0.5)]
                        xy_text = [(500, 0.5), (750, 1), (750, -1), (1250, 1.5), (1250, -1.5)]

                if fg_str == 'dsm':
                    if i == 0:
                        texts = ['1-GC-W-E-S3', '1-GP-NE-NE-S2']
                        xy = [(500, -0.25), (1250, 0.5)]
                        xy_text = [(500, -1.5), (1250, 1.5)]
                    if i == 1:
                        texts = ['2-D-Z-A-P', '2-GP-S-N-S2', '2-GP-N-N-S2']
                        xy = [(750, 0), (1750, -1), (1750, 1)]
                        xy_text = [(1500, 0), (1500, -1.5), (1500, 1.5)]

                if fg_str == 'asm':
                    if i == 0:
                        texts = ['1-GC-W-E-S3', '1-GP-NE-NE-S2', '1-C-E-E-P', '1-C-E-N-P']
                        xy = [(500, -0.25), (1250, 0.5), (1500, 0.33), (1250, 0.0)]
                        xy_text = [(500, -1.5), (1250, 1.5), (1500, 1), (1250, -1)]
                    if i == 1:
                        texts = ['2-GP-S-N-S2', '2-GP-N-N-S2', '2-C-Z-A-P', '2-C-N-N-S1', '2-C-S-N-S1']
                        xy = [(1750, -1), (1750, 1), (1000, 0), (1500, 0.3), (1500, -0.3)]          
                        xy_text = [(1500, -2), (1500, 2), (1000, 0.75), (1250, 1.5), (1250, -1.5)]

                fig = PLT.figure(figsize=(5,5))
                
                ax1 = fig.add_subplot(111)
                # ax1.set_xlabel('Baseline Length [m]', fontsize=18)
                # ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
                # dspec1 = ax1.pcolorfast(bl_length, 1e6*clean_lags, NP.abs(noisy_cc_vis_lag[:-1,:-1,i].T), norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_vis_lag))))
                # ax1.set_xlim(bl_length[0], bl_length[-1])
                # ax1.set_ylim(1e6*clean_lags[0], 1e6*clean_lags[-1])
                ax1.set_xlabel('Baseline Index', fontsize=18)
                ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
                ax1.tick_params(which='major', length=12, labelsize=12, color='white')
                ax1.tick_params(which='minor', length=6, labelsize=12, color='white')
                dspec1 = ax1.imshow(NP.abs(noisy_cc_vis_lag[:,:,i].T), origin='lower', extent=(0, noisy_cc_vis_lag.shape[0]-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e7, vmax=NP.amax(NP.abs(asm_cc_skyvis_lag))), interpolation=None)
                ax1.set_aspect('auto')
                
                for k in xrange(len(texts)):
                    if fg_str == 'csm':
                        ax1.annotate(texts[k], xy=xy[k], xytext=xy_text[k], color='w', horizontalalignment='left', arrowprops=dict(facecolor='white', edgecolor='none', shrink=0.05, frac=0.2, width=2, headwidth=6))
                    if fg_str == 'dsm':
                        ax1.annotate(texts[k], xy=xy[k], xytext=xy_text[k], color='w', horizontalalignment='center', verticalalignment='center', arrowprops=dict(facecolor='white', edgecolor='none', shrink=0.05, frac=0.2, width=2, headwidth=6))
                    if fg_str == 'asm':
                        ax1.annotate(texts[k], xy=xy[k], xytext=xy_text[k], color='w', horizontalalignment='center', verticalalignment='center', arrowprops=dict(facecolor='white', edgecolor='none', shrink=0.05, frac=0.2, width=2, headwidth=6))

                ax2 = ax1.twiny()
                ax2.set_xlabel('Baseline Length [m]', fontsize=18)
                # ax2.set_xlim(*map(xconv, ax1.get_xlim()))
                ax2.set_xticks(NP.asarray(ax1.get_xticks()))
                ax2.set_xticklabels(map(xconv, ax1.get_xticks()))
                ax2.tick_params(which='major', length=12, labelsize=12, color='white')
                ax2.tick_params(which='minor', length=6, labelsize=12, color='white')
    
                cbax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
                cb = fig.colorbar(dspec1, cax=cbax, orientation='vertical')
                cbax.set_ylabel('Jy Hz', labelpad=-60, fontsize=18)
                
                PLT.tight_layout()
                fig.subplots_adjust(right=0.8)
                fig.subplots_adjust(left=0.13)
                
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'annotated_combined_baseline_noisy_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.eps'.format(i), bbox_inches=0)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'annotated_combined_baseline_noisy_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.png'.format(i), bbox_inches=0)

            # fg_str = 'dsm'
            # fig = PLT.figure(figsize=(5,5))
            
            # ax1 = fig.add_subplot(111)
            # # ax1.set_xlabel('Baseline Length [m]', fontsize=18)
            # # ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
            # # dspec1 = ax1.pcolorfast(bl_length, 1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[:-1,:-1,i].T), norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_vis_lag))))
            # # ax1.set_xlim(bl_length[0], bl_length[-1])
            # # ax1.set_ylim(1e6*clean_lags[0], 1e6*clean_lags[-1])
            # ax1.set_xlabel('Baseline Index', fontsize=18)
            # ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
            # dspec1 = ax1.imshow(NP.abs(dsm_cc_skyvis_lag[:,:,i].T), origin='lower', extent=(0, dsm_cc_skyvis_lag.shape[0]-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_skyvis_lag))), interpolation=None)
            # ax1.set_aspect('auto')
            
            # ax2 = ax1.twiny()
            # ax2.set_xlabel('Baseline Length [m]', fontsize=18)
            # # ax2.set_xlim(*map(xconv, ax1.get_xlim()))
            # ax2.set_xticks(NP.asarray(ax1.get_xticks()))
            # ax2.set_xticklabels(map(xconv, ax1.get_xticks()))

            # cbax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            # cb = fig.colorbar(dspec1, cax=cbax, orientation='vertical')
            # cbax.set_ylabel('Jy', labelpad=-60, fontsize=18)
            
            # PLT.tight_layout()
            # fig.subplots_adjust(right=0.8)
            # fig.subplots_adjust(left=0.13)
            
            # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/annotated_combined_baseline_noiseless_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.eps'.format(i), bbox_inches=0)
            # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/annotated_combined_baseline_noiseless_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.png'.format(i), bbox_inches=0)
        
            # fg_str = 'asm'
            # fig = PLT.figure(figsize=(5,5))
            
            # ax1 = fig.add_subplot(111)
            # # ax1.set_xlabel('Baseline Length [m]', fontsize=18)
            # # ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
            # # dspec1 = ax1.pcolorfast(bl_length, 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[:-1,:-1,i].T), norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_vis_lag))))
            # # ax1.set_xlim(bl_length[0], bl_length[-1])
            # # ax1.set_ylim(1e6*clean_lags[0], 1e6*clean_lags[-1])
            # ax1.set_xlabel('Baseline Index', fontsize=18)
            # ax1.set_ylabel(r'lag [$\mu$s]', fontsize=18)
            # dspec1 = ax1.imshow(NP.abs(asm_cc_skyvis_lag[:,:,i].T), origin='lower', extent=(0, asm_cc_skyvis_lag.shape[0]-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(asm_cc_skyvis_lag))), interpolation=None)
            # ax1.set_aspect('auto')
            
            # ax2 = ax1.twiny()
            # ax2.set_xlabel('Baseline Length [m]', fontsize=18)
            # # ax2.set_xlim(*map(xconv, ax1.get_xlim()))
            # ax2.set_xticks(NP.asarray(ax1.get_xticks()))
            # ax2.set_xticklabels(map(xconv, ax1.get_xticks()))

            # cbax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            # cb = fig.colorbar(dspec1, cax=cbax, orientation='vertical')
            # cbax.set_ylabel('Jy', labelpad=-60, fontsize=18)
            
            # PLT.tight_layout()
            # fig.subplots_adjust(right=0.8)
            # fig.subplots_adjust(left=0.13)
            
            # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/annotated_combined_baseline_noiseless_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.eps'.format(i), bbox_inches=0)
            # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/annotated_combined_baseline_noiseless_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snap_{0:0d}.png'.format(i), bbox_inches=0)        

if plot_03:

    #############################################################################
    # Plot EoR window power as a function of LST for quality assurance purposes
    # from different beamformer settings

    freq_resolution = 80e3 # in Hz
    nchan = 384
    bpass_shape = 'bnw'
    fg_model = 'asm'
    coarse_channel_resolution = 1.28e6 # in Hz
    obs_mode = 'dns'
    avg_drifts = False
    beam_switch = True
    snapshots_range = None
    bw = nchan * freq_resolution

    snapshot_type_str = ''
    if avg_drifts and (obs_mode == 'dns'):
        snapshot_type_str = 'drift_averaged_'
        obs_mode = 'custom'
    if beam_switch and (obs_mode == 'dns'):
        snapshot_type_str = 'beam_switches_'
        obs_mode = 'custom'
    if snapshots_range is not None:
        snapshot_type_str = 'snaps_{0[0]:0d}-{0[1]:0d}_'.format(snapshots_range)

    pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    pc_coords = 'dircos'

    delay_matrix = DLY.delay_envelope(bl, pc, units='mks')
    min_delay = -delay_matrix[0,:,1]-delay_matrix[0,:,0]
    max_delay = delay_matrix[0,:,0]-delay_matrix[0,:,1]

    infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[0],bl_length[-1])+'gaussian_FG_model_asm_all_sky_nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+'.fits'
    CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[0],bl_length[-1])+'gaussian_FG_model_asm_all_sky_nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'.fits'

    hdulist = fits.open(infile)
    lst = hdulist['POINTING AND PHASE CENTER INFO'].data['LST']
    vis_rms_freq = hdulist['freq_channel_noise_rms_visibility'].data
    bp = hdulist['bandpass'].data
    bp_wts = nchan * DSP.windowing(nchan, shape=bpass_shape, pad_width=0, centering=True, area_normalize=True) 
    bp_wts = bp_wts[NP.newaxis,:,NP.newaxis]
    t_acc = hdulist['t_acc'].data
    hdulist.close()

    hdulist = fits.open(CLEAN_infile)
    clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data
    ccres_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data

    cc_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA IMAG'].data
    ccres_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG'].data

    hdulist.close()
    cc_skyvis_lag = cc_skyvis_lag + ccres_skyvis_lag
    cc_vis_lag = cc_vis_lag + ccres_vis_lag

    cc_skyvis_lag = DSP.downsampler(cc_skyvis_lag, 1.0*clean_lags.size/nchan, axis=1)
    cc_vis_lag = DSP.downsampler(cc_vis_lag, 1.0*clean_lags.size/nchan, axis=1)
    clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/nchan, axis=-1)

    cc_skyvis_power_lag_modified = NP.empty((bl.shape[0], nchan, lst.size))
    cc_skyvis_lag_noisy4 = NP.empty((bl.shape[0], nchan, 4))
    for i in xrange(lst.size):
        vis_noise_freq = NP.sqrt(4) * vis_rms_freq[:,:,i].reshape(bl.shape[0],nchan,-1) / NP.sqrt(2.0) * (NP.random.randn(bl.shape[0], nchan, 4) + 1j * NP.random.randn(bl.shape[0], nchan, 4)) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts and sqrt(4) is to divide up the snapshot time into 4 chunks
        vis_noise_lag = DSP.FT1D(vis_noise_freq * bp[:,:,i].reshape(bl.shape[0],nchan,1) * bp_wts, ax=1, inverse=True, use_real=False, shift=True) * nchan * freq_resolution
        cc_skyvis_lag_noisy4 = cc_skyvis_lag[:,:,i].reshape(bl.shape[0],nchan,1) + vis_noise_lag
        temp = 0.5 * (NP.abs(NP.sum(cc_skyvis_lag_noisy4, axis=2))**2 - NP.sum(NP.abs(cc_skyvis_lag_noisy4)**2, axis=2)) / 6
        cc_skyvis_power_lag_modified[:,:,i] = temp

    clean_lags = clean_lags.reshape(-1,1)
    min_delay = min_delay.reshape(1,-1)
    max_delay = max_delay.reshape(1,-1)
    EoR_window = NP.logical_and(NP.logical_or(clean_lags > max_delay+1/bw, clean_lags < min_delay-1/bw), NP.abs(clean_lags) < 1./coarse_channel_resolution)
    wedge_window = NP.logical_and(clean_lags <= max_delay, clean_lags >= min_delay)
    EoR_window = EoR_window.T
    wedge_window = wedge_window.T

    EoR_window_rms_unbiased = OPS.rms(cc_skyvis_power_lag_modified.reshape(-1,lst.size), mask=NP.logical_not(NP.repeat(EoR_window.reshape(-1,1), lst.size, axis=1)), axis=0)

    # EoR_window_power_unbiased = NP.average(NP.abs(cc_skyvis_power_lag_modified[EoR_window]), axis=0)
    # wedge_power = NP.average(NP.abs(cc_skyvis_lag[wedge_window])**2, axis=0)

    # EoR_window_correlated_power_noiseless = 0.5 * (NP.abs(NP.sum(cc_skyvis_lag[EoR_window], axis=0))**2 - NP.sum(NP.abs(cc_skyvis_lag[EoR_window])**2, axis=0))
    # EoR_window_correlated_power_noisy = 0.5 * (NP.abs(NP.sum(cc_vis_lag[EoR_window], axis=0))**2 - NP.sum(NP.abs(cc_vis_lag[EoR_window])**2, axis=0))

    EoR_window_power_noiseless = NP.average(NP.abs(cc_skyvis_lag[EoR_window])**2, axis=0)
    EoR_window_power_noisy = NP.average(NP.abs(cc_vis_lag[EoR_window])**2, axis=0)
    lst_wrapped = lst*15.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0

    fig = PLT.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(lst/15.0, NP.abs(EoR_window_rms_unbiased.ravel()), 'k.', ms=10, label='Unbiased')
    ax.plot(lst/15.0, NP.abs(EoR_window_power_noiseless), 'kx', ms=10, label='Noiseless')
    ax.plot(lst/15.0, NP.abs(EoR_window_power_noisy), 'k+', ms=10, label='Noisy')
    # ax.plot(lst/15.0, EoR_window_correlated_power_noiseless, 'k.', ms=10, label='Noiseless')
    # ax.plot(lst/15.0, EoR_window_correlated_power_noisy, 'k+', ms=10, label='Noisy')
    # ax.plot(lst/15.0, NP.abs(EoR_window_correlated_power_noiseless)/NP.sum(EoR_window), 'k.', ms=10, label='Noiseless')
    # ax.plot(lst/15.0, NP.abs(EoR_window_correlated_power_noisy)/NP.sum(EoR_window), 'k+', ms=10, label='Noisy')
    ax.set_xlabel('LST [hours]', fontsize=18) 
    ax.set_ylabel(r'EoR window foreground power [ Jy$^2$ Hz$^2$]', fontsize=18)
    ax.set_yscale('log')
    legend = ax.legend(loc='upper right')
    legend.draw_frame(False)
    ax.tick_params(which='major', length=18, labelsize=12)
    ax.tick_params(which='minor', length=12, labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    xticklabels = PLT.getp(ax, 'xticklabels')
    yticklabels = PLT.getp(ax, 'yticklabels')
    PLT.setp(xticklabels, fontsize=15, weight='medium')
    PLT.setp(yticklabels, fontsize=15, weight='medium') 

    PLT.tight_layout()
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.25)

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/EoR_window_power_'+snapshot_type_str+obs_mode+'Tsys_{0:.1f}_K_{1:.1f}_MHz'.format(Tsys, freq/1e6)+'.eps', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/EoR_window_power_'+snapshot_type_str+obs_mode+'Tsys_{0:.1f}_K_{1:.1f}_MHz'.format(Tsys, freq/1e6)+'.png', bbox_inches=0)

    # fig = PLT.figure(figsize=(6,6))
    # ax = fig.add_subplot(111)
    # ax.plot(lst/15.0, NP.abs(wedge_power), 'k.', ms=10, label='Noiseless')
    # # ax.plot(lst/15.0, NP.abs(EoR_window_power_noiseless), 'k.', ms=10, label='Noiseless')
    # # ax.plot(lst/15.0, NP.abs(EoR_window_power_noisy), 'k+', ms=10, label='Noisy')
    # # ax.plot(lst/15.0, EoR_window_correlated_power_noiseless, 'k.', ms=10, label='Noiseless')
    # # ax.plot(lst/15.0, EoR_window_correlated_power_noisy, 'k+', ms=10, label='Noisy')
    # # ax.plot(lst/15.0, NP.abs(EoR_window_correlated_power_noiseless)/NP.sum(EoR_window), 'k.', ms=10, label='Noiseless')
    # # ax.plot(lst/15.0, NP.abs(EoR_window_correlated_power_noisy)/NP.sum(EoR_window), 'k+', ms=10, label='Noisy')
    # ax.set_xlabel('LST [hours]', fontsize=18) 
    # ax.set_ylabel(r'Foreground wedge power [ Jy$^2$]', fontsize=18)
    # ax.set_yscale('linear')
    # legend = ax.legend(loc='upper right')
    # legend.draw_frame(False)
    # ax.tick_params(which='major', length=18, labelsize=12)
    # ax.tick_params(which='minor', length=12, labelsize=12)
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)
    # xticklabels = PLT.getp(ax, 'xticklabels')
    # yticklabels = PLT.getp(ax, 'yticklabels')
    # PLT.setp(xticklabels, fontsize=15, weight='medium')
    # PLT.setp(yticklabels, fontsize=15, weight='medium') 

    # PLT.tight_layout()
    # fig.subplots_adjust(right=0.95)
    # fig.subplots_adjust(left=0.25)

    # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/wedge_power_'+snapshot_type_str+obs_mode+'Tsys_{0:.1f}_K_{1:.1f}_MHz'.format(Tsys, freq/1e6)+'.eps', bbox_inches=0)
    # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/wedge_power_'+snapshot_type_str+obs_mode+'Tsys_{0:.1f}_K_{1:.1f}_MHz'.format(Tsys, freq/1e6)+'.png', bbox_inches=0)

    # # for i in xrange(lst.size):
    # #     fig = PLT.figure()
    # #     ax = fig.add_subplot(111)
    # #     ax.set_xlabel('Baseline Index', fontsize=18)
    # #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
    # #     dspec = ax.imshow(NP.abs(cc_skyvis_lag[:,:,i].T), origin='lower', extent=(0, cc_skyvis_lag.shape[0]-1, NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6)), norm=PLTC.LogNorm(vmin=1e5, vmax=NP.amax(NP.abs(cc_skyvis_lag))), interpolation=None)
    # #     # norm=PLTC.LogNorm(vmin=NP.amin(NP.abs(cc_skyvis_lag)), vmax=NP.amax(NP.abs(cc_skyvis_lag))), 
    # #     ax.set_aspect('auto')

    # #     cbax = fig.add_axes([0.88, 0.08, 0.03, 0.9])
    # #     cb = fig.colorbar(dspec, cax=cbax, orientation='vertical')
    # #     cbax.set_ylabel('Jy Hz', labelpad=-60, fontsize=18)

    # #     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_combined_baseline_CLEAN_visibilities_contiguous_orientations_'+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.eps'.format(i), bbox_inches=0)
    # #     PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_combined_baseline_CLEAN_visibilities_contiguous_orientations_'+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(i), bbox_inches=0)

if plot_04:

    #############################################################################
    # 04) Plot sky power as a function of LST 

    latitude = -26.701 
    use_GSM = True
    use_DSM = False
    use_CSM = False
    use_NVSS = False
    use_SUMSS = False
    use_MSS = False
    use_GLEAM = False
    use_PS = False
    obs_mode = 'dns'
    n_sky_sectors = 4
    if (n_sky_sectors < 1):
        n_sky_sectors = 1

    pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
    pointing_info = None
    n_snaps = None
    avg_drifts = False
    beam_switch = True
    snapshot_sampling = None
    pick_snapshots = None
    snapshots_range = None
    snapshot_type_str = ''
    if avg_drifts and (obs_mode == 'dns'):
        snapshot_type_str = 'drift_averaged_'
    if beam_switch and (obs_mode == 'dns'):
        snapshot_type_str = 'beam_switches_'

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
        pointings_altaz_orig = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
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
            pointings_radec = NP.hstack((NP.asarray(lst-pointing_init[0]).reshape(-1,1), pointing_init[1]+NP.zeros(n_snaps).reshape(-1,1)))
    
        pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
        pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
        pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    
        lst_wrapped = lst + 0.0
        lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
        lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

    snapshot_type_str = ''
    if avg_drifts and (obs_mode == 'dns'):
        snapshot_type_str = 'drift_averaged_'
        obs_mode = 'custom'
    if beam_switch and (obs_mode == 'dns'):
        snapshot_type_str = 'beam_switches_'
        obs_mode = 'custom'

    if use_GSM:
        fg_str = 'asm'
    
        dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
        hdulist = fits.open(dsm_file)
        pixres = hdulist[0].header['PIXAREA']
        dsm_table = hdulist[1].data
        ra_deg_DSM = dsm_table['RA']
        dec_deg_DSM = dsm_table['DEC']
        temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
        fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
        spindex = dsm_table['spindex'] + 2.0
        freq_DSM = 0.185 # in GHz
        freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
        catlabel = NP.repeat('DSM', fluxes_DSM.size)
        ra_deg = ra_deg_DSM + 0.0
        dec_deg = dec_deg_DSM + 0.0
        majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
        minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
        fluxes = fluxes_DSM + 0.0
    
        freq_SUMSS = 0.843 # in GHz
        SUMSS_file = '/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt'
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
    
        nvss_file = '/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits'
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
    
        dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
        hdulist = fits.open(dsm_file)
        pixres = hdulist[0].header['PIXAREA']
        dsm_table = hdulist[1].data
        ra_deg_DSM = dsm_table['RA']
        dec_deg_DSM = dsm_table['DEC']
        temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
        fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
        spindex = dsm_table['spindex'] + 2.0
        freq_DSM = 0.185 # in GHz
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
        SUMSS_file = '/data3/t_nithyanandan/project_MWA/foregrounds/sumsscat.Mar-11-2008.txt'
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
    
        nvss_file = '/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits'
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
        ctlgobj = CTLG.Catalog(freq_catalog*1e9, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((fmajax.reshape(-1,1),fminax.reshape(-1,1),fpa.reshape(-1,1))), src_shape_units=['arcsec','arcsec','degree'])    
        fg_str = 'sumss'
    elif use_MSS:
        pass
    elif use_GLEAM:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv'
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
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/PS_catalog.txt'
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
        ctlgobj = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fint, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), src_shape_units=['arcmin','arcmin','degree'])
    
    skymod = CTLG.SkyModel(ctlgobj)
   
    sky_sector_emission = NP.zeros((n_snaps,n_sky_sectors))
    for j in range(n_snaps):
        src_altaz_current = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst[j]-skymod.catalog.location[:,0]).reshape(-1,1), skymod.catalog.location[:,1].reshape(-1,1))), latitude, units='degrees')
        hemisphere_current = src_altaz_current[:,0] >= 0.0
        # hemisphere_src_altaz_current = src_altaz_current[hemisphere_current,:]
        src_az_current = src_altaz_current[:,1] + 0.0
        src_az_current[src_az_current > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
        src_altaz_current_hemisphere = src_altaz_current[hemisphere_current,:]
        src_az_current_hemisphere = src_az_current[hemisphere_current]

        pb_hemisphere_curent = PB.primary_beam_generator(src_altaz_current_hemisphere, freq, telescope=telescope, freq_scale='Hz', skyunits='altaz', pointing_center=[90.0,270.0])
        for k in xrange(n_sky_sectors):
            roi_ind = NP.logical_or(NP.logical_and(src_az_current_hemisphere >= -0.5*180.0/n_sky_sectors + k*180.0/n_sky_sectors, src_az_current_hemisphere < -0.5*180.0/n_sky_sectors + (k+1)*180.0/n_sky_sectors), NP.logical_and(src_az_current_hemisphere >= 180.0 - 0.5*180.0/n_sky_sectors + k*180.0/n_sky_sectors, src_az_current_hemisphere < 180.0 - 0.5*180.0/n_sky_sectors + (k+1)*180.0/n_sky_sectors))
            roi_subset = NP.where(hemisphere_current)[0][roi_ind].tolist()
            # roi_subset = NP.where(NP.logical_and(hemisphere_current, roi_ind))[0].tolist()
            fgmod = CTLG.SkyModel(skymod.catalog.subset(roi_subset))
            flux_densities_roi = fgmod.catalog.flux_density * (freq/fgmod.catalog.frequency)**fgmod.catalog.spectral_index
            sky_sector_emission[j,k] = NP.sum(flux_densities_roi * pb_hemisphere_curent[roi_ind])

    # Plot just the galactic plane power seen through the primary beam

    n_sky_sectors = 1
    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes_DSM = temperatures * (2.0* FCNST.k * freq**2 / FCNST.c**2) * pixres / CNST.Jy
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = 0.185 # in GHz

    dsmradec = coord.FK5(ra=ra_deg_DSM, dec=dec_deg_DSM, unit=(units.degree, units.degree))
    dsmlatlon = dsmradec.galactic
    gp_ind = NP.abs(dsmlatlon.latangle.degree) <= 10.0
    gpradec = coord.FK5(ra=ra_deg_DSM[gp_ind], dec=dec_deg_DSM[gp_ind], unit=(units.degree, units.degree))
    gplatlon = gpradec.galactic
    dsmfluxes = fluxes_DSM[gp_ind]

    gp_sky_sector_emission = NP.zeros((n_snaps,n_sky_sectors))
    for j in range(n_snaps):
        gp_altaz_current = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst[j]-gpradec.ra.degree).reshape(-1,1), gpradec.dec.degree.reshape(-1,1))), latitude, units='degrees')
        hemisphere_current = gp_altaz_current[:,0] >= 0.0
        # hemisphere_gp_altaz_current = gp_altaz_current[hemisphere_current,:]
        gp_az_current = gp_altaz_current[:,1] + 0.0
        gp_az_current[gp_az_current > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
        gp_altaz_current_hemisphere = gp_altaz_current[hemisphere_current,:]
        gp_az_current_hemisphere = gp_az_current[hemisphere_current]

        pb_hemisphere_curent = PB.primary_beam_generator(gp_altaz_current_hemisphere, freq, telescope=telescope, freq_scale='Hz', skyunits='altaz', pointing_center=[90.0,270.0])
        for k in xrange(n_sky_sectors):
            roi_ind = NP.logical_or(NP.logical_and(gp_az_current_hemisphere >= -0.5*180.0/n_sky_sectors + k*180.0/n_sky_sectors, gp_az_current_hemisphere < -0.5*180.0/n_sky_sectors + (k+1)*180.0/n_sky_sectors), NP.logical_and(gp_az_current_hemisphere >= 180.0 - 0.5*180.0/n_sky_sectors + k*180.0/n_sky_sectors, gp_az_current_hemisphere < 180.0 - 0.5*180.0/n_sky_sectors + (k+1)*180.0/n_sky_sectors))
            roi_subset = NP.where(hemisphere_current)[0][roi_ind].tolist()
            # roi_subset = NP.where(NP.logical_and(hemisphere_current, roi_ind))[0].tolist()
            dsmfluxes_roi = dsmfluxes[roi_subset]
            gp_sky_sector_emission[j,k] = NP.sum(dsmfluxes_roi * pb_hemisphere_curent[roi_ind])

    PDB.set_trace()

if plot_05:

    #############################################################################
    # 05) Plot fraction of pixels relatively free of contamination as a function
    #     of baseline length

    freq = 185.0e6 # center frequency in Hz
    freq_resolution = 80e3 # in Hz
    nchan = 384
    bpass_shape = 'bnw'
    fg_model = 'asm'
    coarse_channel_resolution = 1.28e6 # in Hz
    obs_mode = 'custom'
    avg_drifts = False
    beam_switch = False
    bw = nchan * freq_resolution

    n_bl_chunks = 200
    baseline_chunk_size = 10
    baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

    neg_bl_orientation_ind = bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    bl_orientation[neg_bl_orientation_ind] -= 180.0
    bl[neg_bl_orientation_ind,:] *= -1.0

    snapshot_type_str = ''
    if avg_drifts and (obs_mode == 'dns'):
        snapshot_type_str = 'drift_averaged_'
        obs_mode = 'custom'
    if beam_switch and (obs_mode == 'dns'):
        snapshot_type_str = 'beam_switches_'
        obs_mode = 'custom'

    MWA_infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
    MWA_dipole_infile = '/data3/t_nithyanandan/project_MWA/mwa_dipole_multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
    HERA_infile = '/data3/t_nithyanandan/project_MWA/hera_multi_baseline_CLEAN_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

    neg_bl_orientation_ind = bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    bl_orientation[neg_bl_orientation_ind] -= 180.0
    bl[neg_bl_orientation_ind,:] *= -1.0

    hdulist = fits.open(MWA_infile+'.fits')
    mwa_clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    mwa_asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    mwa_asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    mwa_asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    mwa_asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    mwa_asm_cc_skyvis = mwa_asm_cc_skyvis + mwa_asm_cc_skyvis_res
    mwa_asm_cc_vis = mwa_asm_cc_vis + mwa_asm_cc_vis_res
    hdulist.close()

    hdulist = fits.open(MWA_dipole_infile+'.fits')
    mwa_dipole_clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    mwa_dipole_asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    mwa_dipole_asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    mwa_dipole_asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    mwa_dipole_asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    mwa_dipole_asm_cc_skyvis = mwa_dipole_asm_cc_skyvis + mwa_dipole_asm_cc_skyvis_res
    mwa_dipole_asm_cc_vis = mwa_dipole_asm_cc_vis + mwa_dipole_asm_cc_vis_res
    hdulist.close()

    hdulist = fits.open(HERA_infile+'.fits')
    hera_clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    hera_asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    hera_asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    hera_asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    hera_asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hera_asm_cc_skyvis = hera_asm_cc_skyvis + hera_asm_cc_skyvis_res
    hera_asm_cc_vis = hera_asm_cc_vis + hera_asm_cc_vis_res
    hdulist.close()
            
    mwa_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(mwa_asm_cc_skyvis, axis=1), axes=1) * mwa_asm_cc_skyvis.shape[1] * freq_resolution
    mwa_dipole_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(mwa_dipole_asm_cc_skyvis, axis=1), axes=1) * mwa_dipole_asm_cc_skyvis.shape[1] * freq_resolution
    hera_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(hera_asm_cc_skyvis, axis=1), axes=1) * hera_asm_cc_skyvis.shape[1] * freq_resolution

    mwa_asm_cc_skyvis_res_lag = NP.fft.fftshift(NP.fft.ifft(mwa_asm_cc_skyvis_res, axis=1), axes=1) * mwa_asm_cc_skyvis.shape[1] * freq_resolution
    mwa_dipole_asm_cc_skyvis_res_lag = NP.fft.fftshift(NP.fft.ifft(mwa_dipole_asm_cc_skyvis_res, axis=1), axes=1) * mwa_dipole_asm_cc_skyvis.shape[1] * freq_resolution
    hera_asm_cc_skyvis_res_lag = NP.fft.fftshift(NP.fft.ifft(hera_asm_cc_skyvis_res, axis=1), axes=1) * hera_asm_cc_skyvis.shape[1] * freq_resolution

    hera_asm_cc_skyvis_lag = DSP.downsampler(hera_asm_cc_skyvis_lag, 1.0*hera_clean_lags.size/nchan, axis=1)
    mwa_asm_cc_skyvis_lag = DSP.downsampler(mwa_asm_cc_skyvis_lag, 1.0*mwa_clean_lags.size/nchan, axis=1)
    mwa_dipole_asm_cc_skyvis_lag = DSP.downsampler(mwa_dipole_asm_cc_skyvis_lag, 1.0*mwa_dipole_clean_lags.size/nchan, axis=1)
    hera_asm_cc_skyvis_res_lag = DSP.downsampler(hera_asm_cc_skyvis_res_lag, 1.0*hera_clean_lags.size/nchan, axis=1)
    mwa_asm_cc_skyvis_res_lag = DSP.downsampler(mwa_asm_cc_skyvis_res_lag, 1.0*mwa_clean_lags.size/nchan, axis=1)
    mwa_dipole_asm_cc_skyvis_res_lag = DSP.downsampler(mwa_dipole_asm_cc_skyvis_res_lag, 1.0*mwa_dipole_clean_lags.size/nchan, axis=1)
    mwa_clean_lags = DSP.downsampler(mwa_clean_lags, 1.0*mwa_clean_lags.size/nchan, axis=-1)
    mwa_dipole_clean_lags = DSP.downsampler(mwa_dipole_clean_lags, 1.0*mwa_dipole_clean_lags.size/nchan, axis=-1)
    hera_clean_lags = DSP.downsampler(hera_clean_lags, 1.0*hera_clean_lags.size/nchan, axis=-1)

    pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
    pc_coords = 'dircos'

    delay_matrix = DLY.delay_envelope(bl, pc, units='mks')
    min_delay = -delay_matrix[0,:,1]-delay_matrix[0,:,0]
    max_delay = delay_matrix[0,:,0]-delay_matrix[0,:,1]
    
    min_delay = min_delay.reshape(1,-1)
    max_delay = max_delay.reshape(1,-1)
    mwa_clean_lags = mwa_clean_lags.reshape(-1,1)
    mwa_dipole_clean_lags = mwa_dipole_clean_lags.reshape(-1,1)
    hera_clean_lags = hera_clean_lags.reshape(-1,1)

    mwa_wedge_ind = NP.logical_and(mwa_clean_lags >= min_delay, mwa_clean_lags <= max_delay)
    mwa_wedge_power = NP.abs(mwa_asm_cc_skyvis_lag[:,:,1].T * mwa_wedge_ind)**2
    mwa_wedge_power[mwa_wedge_power == 0] = NP.nan
    mwa_wedge_mean = NP.nanmean(mwa_wedge_power, axis=0)
    mwa_wedge_mean = mwa_wedge_mean.reshape(1,-1)
    mwa_wedge_rms = NP.nanstd(mwa_wedge_power, axis=0)
    mwa_var_outside_wedge = NP.nanstd(NP.abs(mwa_asm_cc_skyvis_lag[:,:,1].T * NP.logical_not(mwa_wedge_ind))**2, axis=0).reshape(1,-1)
    # mwa_faint_wedge_ind = mwa_wedge_power < mwa_wedge_mean
    mwa_faint_wedge_ind = mwa_wedge_power < 5 * mwa_var_outside_wedge
    mwa_faint_fraction = NP.sum(mwa_faint_wedge_ind, axis=0).astype(float) / NP.sum(mwa_wedge_ind, axis=0)

    mwa_dipole_wedge_ind = NP.logical_and(mwa_dipole_clean_lags >= min_delay, mwa_dipole_clean_lags <= max_delay)
    mwa_dipole_wedge_power = NP.abs(mwa_dipole_asm_cc_skyvis_lag[:,:,1].T * mwa_dipole_wedge_ind)**2
    mwa_dipole_wedge_power[mwa_dipole_wedge_power == 0] = NP.nan
    mwa_dipole_wedge_mean = NP.nanmean(mwa_dipole_wedge_power, axis=0)
    mwa_dipole_wedge_mean = mwa_dipole_wedge_mean.reshape(1,-1)
    mwa_dipole_wedge_rms = NP.nanstd(mwa_dipole_wedge_power, axis=0)
    mwa_dipole_var_outside_wedge = NP.nanstd(NP.abs(mwa_dipole_asm_cc_skyvis_lag[:,:,1].T * NP.logical_not(mwa_dipole_wedge_ind))**2, axis=0).reshape(1,-1)
    # mwa_dipole_faint_wedge_ind = mwa_dipole_wedge_power < mwa_dipole_wedge_mean
    mwa_dipole_faint_wedge_ind = mwa_dipole_wedge_power < 5 * mwa_dipole_var_outside_wedge
    mwa_dipole_faint_fraction = NP.sum(mwa_dipole_faint_wedge_ind, axis=0).astype(float) / NP.sum(mwa_dipole_wedge_ind, axis=0)

    hera_wedge_ind = NP.logical_and(hera_clean_lags >= min_delay, hera_clean_lags <= max_delay)
    hera_wedge_power = NP.abs(hera_asm_cc_skyvis_lag[:,:,1].T * hera_wedge_ind)**2
    hera_wedge_power[hera_wedge_power == 0] = NP.nan
    hera_wedge_mean = NP.nanmean(hera_wedge_power, axis=0)
    hera_wedge_mean = hera_wedge_mean.reshape(1,-1)
    hera_wedge_rms = NP.nanstd(hera_wedge_power, axis=0)
    hera_var_outside_wedge = NP.nanstd(NP.abs(hera_asm_cc_skyvis_lag[:,:,1].T * NP.logical_not(hera_wedge_ind))**2, axis=0).reshape(1,-1)
    # hera_faint_wedge_ind = hera_wedge_power < hera_wedge_mean
    hera_faint_wedge_ind = hera_wedge_power < 5 * hera_var_outside_wedge
    hera_faint_fraction = NP.sum(hera_faint_wedge_ind, axis=0).astype(float) / NP.sum(hera_wedge_ind, axis=0)
    
    fig = PLT.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(bl_length, mwa_faint_fraction, 'k.', label='MWA tile')
    ax.plot(bl_length, mwa_dipole_faint_fraction, 'b.', lw=2, label='MWA dipole')
    ax.plot(bl_length, hera_faint_fraction, 'r.', lw=2, label='HERA')
    ax.set_xlabel('Baseline length [m]', fontsize=18)
    ax.set_ylabel('Faint Fraction Delay Spectrum', fontsize=18)
    legend = ax.legend(loc='lower right')
    legend.draw_frame(False)
    ax.tick_params(which='major', length=18, labelsize=12)
    ax.tick_params(which='minor', length=12, labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    xticklabels = PLT.getp(ax, 'xticklabels')
    yticklabels = PLT.getp(ax, 'yticklabels')
    PLT.setp(xticklabels, fontsize=15, weight='medium')
    PLT.setp(yticklabels, fontsize=15, weight='medium') 

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/faint_fraction_delay_spectrum_zenith.eps', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/faint_fraction_delay_spectrum_zenith.png', bbox_inches=0)

    
