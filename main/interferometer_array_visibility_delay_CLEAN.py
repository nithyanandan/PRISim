import numpy as NP 
import astropy
from astropy.io import fits
from astropy.io import ascii
import progressbar as PGB
import matplotlib.pyplot as PLT
import aipy as AP
import interferometry as RI
import my_DSP_modules as DSP 
import baseline_delay_horizon as DLY
import CLEAN_wrapper as CLN

project_MWA = False
project_HERA = False
project_beams = True
project_drift_scan = False
project_global_EoR = False

if project_MWA: project_dir = 'project_MWA'
if project_HERA: project_dir = 'project_HERA'
if project_beams: project_dir = 'project_beams'
if project_drift_scan: project_dir = 'project_drift_scan'
if project_global_EoR: project_dir = 'project_global_EoR'

telescope_id = 'custom'
element_size = 0.74
element_shape = 'delta'
phased_array = True

if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole'):
    element_size = 0.74
    element_shape = 'dipole'
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

ground_plane = 0.3 # height of antenna element above ground plane
if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

delayerr = 0.05     # delay error rms in ns
if delayerr is None:
    delayerr_str = ''
    delayerr = 0.0
elif delayerr < 0.0:
    raise ValueError('delayerr must be non-negative.')
else:
    delayerr_str = 'derr_{0:.3f}ns'.format(delayerr)
delayerr *= 1e-9

gainerr = 0.0      # Gain error rms in dB
if gainerr is None:
    gainerr_str = ''
    gainerr = 0.0
elif gainerr < 0.0:
    raise ValueError('gainerr must be non-negative.')
else:
    gainerr_str = '_gerr_{0:.2f}dB'.format(gainerr)

nrand = 1       # Number of random realizations
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

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
bl, bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_length = bl_length[sortind]
bl_id = bl_id[sortind]
total_baselines = bl_length.size

n_bl_chunks = 32
baseline_chunk_size = 64
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

Tsys = 95.0 # System temperature in K
freq = 185.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 80e3 # in Hz
bpass_shape = 'bhw'
f_pad = 1.0
oversampling_factor = 1.0 + f_pad
n_channels = 384
nchan = n_channels
window = n_channels * DSP.windowing(n_channels, shape=bpass_shape, pad_width=0, centering=True, area_normalize=True) 
bw = n_channels * freq_resolution

use_pfb = True

pfb_instr = ''
pfb_outstr = ''
if not use_pfb: 
    pfb_instr = '_no_pfb'
    pfb_outstr = 'no_pfb_'

obs_mode = 'custom'
avg_drifts = False
beam_switch = False
snapshots_range = None
snapshot_type_str = ''
if avg_drifts:
    snapshot_type_str = 'drift_averaged_'
if beam_switch:
    snapshot_type_str = 'beam_switches_'
if snapshots_range is not None:
    snapshot_type_str = 'snaps_{0[0]:0d}-{0[1]:0d}_'.format(snapshots_range)

# pc = NP.asarray([90.0, 90.0]).reshape(1,-1)
# pc = NP.asarray([266.416837, -29.00781]).reshape(1,-1)
pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'
if pc_coords == 'dircos':
    pc_dircos = pc

n_sky_sectors = 1

spindex_rms = 0.0
spindex_seed = None
spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

nside = 64
use_GSM = True
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False
use_USM = False

if use_GSM:
    fg_str = 'asm'
elif use_DSM:
    fg_str = 'dsm'
elif use_CSM:
    fg_str = 'csm'
elif use_SUMSS:
    fg_str = 'sumss'
elif use_GLEAM:
    fg_str = 'gleam'
elif use_PS:
    fg_str = 'point'
elif use_NVSS:
    fg_str = 'nvss'
elif use_USM:
    fg_str = 'usm'
else:
    fg_str = 'other'

for k in range(n_sky_sectors):
    if n_sky_sectors == 1:
        sky_sector_str = '_all_sky_'
    else:
        sky_sector_str = '_sky_sector_{0:0d}_'.format(k)

    infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_instr

    # infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz'.format(nchan*freq_resolution/1e6)
    
    ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits') 
    ia.phase_centering(phase_center=pc, phase_center_coords=pc_coords)   
    ia.delay_transform(oversampling_factor-1.0, freq_wts=window)
    
    delay_matrix = DLY.delay_envelope(ia.baselines, pc_dircos, units='mks')
    
    # lags = DSP.spectral_axis(ia.channels.size, delx=ia.freq_resolution, use_real=False, shift=True)
    # clean_area = NP.zeros(ia.channels.size, dtype=int)
    npad = ia.channels.size
    lags = DSP.spectral_axis(ia.channels.size + npad, delx=ia.freq_resolution, use_real=False, shift=False)
    clean_area = NP.zeros(ia.channels.size + npad, dtype=int)
    skyvis_lag = (npad + ia.channels.size) * ia.freq_resolution * DSP.FT1D(NP.pad(ia.skyvis_freq*ia.bp*ia.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
    vis_lag = (npad + ia.channels.size) * ia.freq_resolution * DSP.FT1D(NP.pad(ia.vis_freq*ia.bp*ia.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
    lag_kernel = (npad + ia.channels.size) * ia.freq_resolution * DSP.FT1D(NP.pad(ia.bp, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)

    ccomponents_noiseless = NP.zeros_like(skyvis_lag)
    ccres_noiseless = NP.zeros_like(skyvis_lag)

    ccomponents_noisy = NP.zeros_like(vis_lag)
    ccres_noisy = NP.zeros_like(vis_lag)
    
    for snap_iter in xrange(ia.skyvis_freq.shape[2]):
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=ia.baselines.shape[0]).start()
        for bl_iter in xrange(ia.baselines.shape[0]):
            # clean_area[NP.logical_and(ia.lags <= delay_matrix[0,bl_iter,0]+delay_matrix[0,bl_iter,1]+40e-9, ia.lags >= -delay_matrix[0,bl_iter,0]+delay_matrix[0,bl_iter,1]-40e-9)] = 1
            # cc, info = CLN.gentle(ia.skyvis_lag[bl_iter,:,snap_iter], ia.lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=True)
    
            clean_area[NP.logical_and(lags <= delay_matrix[0,bl_iter,0]+delay_matrix[0,bl_iter,1]+3/bw, lags >= -delay_matrix[0,bl_iter,0]+delay_matrix[0,bl_iter,1]-3/bw)] = 1

            cc_noiseless, info_noiseless = CLN.gentle(skyvis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=False, autoscale=True)
            ccomponents_noiseless[bl_iter,:,snap_iter] = cc_noiseless
            ccres_noiseless[bl_iter,:,snap_iter] = info_noiseless['res']

            cc_noisy, info_noisy = CLN.gentle(vis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=False, autoscale=True)
            ccomponents_noisy[bl_iter,:,snap_iter] = cc_noisy
            ccres_noisy[bl_iter,:,snap_iter] = info_noisy['res']

            progress.update(bl_iter+1)
        progress.finish()
    
    deta = lags[1] - lags[0]
    cc_skyvis = NP.fft.fft(ccomponents_noiseless, axis=1) * deta
    cc_skyvis_res = NP.fft.fft(ccres_noiseless, axis=1) * deta

    cc_vis = NP.fft.fft(ccomponents_noisy, axis=1) * deta
    cc_vis_res = NP.fft.fft(ccres_noisy, axis=1) * deta

    skyvis_lag = NP.fft.fftshift(skyvis_lag, axes=1)
    vis_lag = NP.fft.fftshift(vis_lag, axes=1)
    lag_kernel = NP.fft.fftshift(lag_kernel, axes=1)
    ccomponents_noiseless = NP.fft.fftshift(ccomponents_noiseless, axes=1)
    ccres_noiseless = NP.fft.fftshift(ccres_noiseless, axes=1)
    ccomponents_noisy = NP.fft.fftshift(ccomponents_noisy, axes=1)
    ccres_noisy = NP.fft.fftshift(ccres_noisy, axes=1)
    lags = NP.fft.fftshift(lags)

    # ccomponents = (1+npad*1.0/ia.channels.size) * DSP.downsampler(ccomponents, 1+npad*1.0/ia.channels.size, axis=1)
    # ccres = (1+npad*1.0/ia.channels.size) * DSP.downsampler(ccres, 1+npad*1.0/ia.channels.size, axis=1)
    # lags = DSP.downsampler(lags, 1+npad*1.0/ia.channels.size, axis=-1)
    
    outfile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_outstr+bpass_shape
    hdulist = []
    hdulist += [fits.PrimaryHDU()]
    
    cols = []
    cols += [fits.Column(name='frequency', format='D', array=ia.channels)]
    cols += [fits.Column(name='lag', format='D', array=lags)]

    if astropy.__version__ == '0.4':
        columns = fits.ColDefs(cols, tbtype='BinTableHDU')
    elif (astropy.__version__ == '0.4.2') or (astropy.__version__ == u'1.0'):
        columns = fits.ColDefs(cols, ascii=False)

    tbhdu = fits.new_table(columns)
    tbhdu.header.set('EXTNAME', 'SPECTRAL INFO')
    hdulist += [tbhdu]
    
    hdulist += [fits.ImageHDU(skyvis_lag.real, name='ORIGINAL NOISELESS DELAY SPECTRA REAL')]
    hdulist += [fits.ImageHDU(skyvis_lag.imag, name='ORIGINAL NOISELESS DELAY SPECTRA IMAG')]
    hdulist += [fits.ImageHDU(vis_lag.real, name='ORIGINAL DELAY SPECTRA REAL')]
    hdulist += [fits.ImageHDU(vis_lag.imag, name='ORIGINAL DELAY SPECTRA IMAG')]
    hdulist += [fits.ImageHDU(lag_kernel.real, name='LAG KERNEL REAL')]
    hdulist += [fits.ImageHDU(lag_kernel.imag, name='LAG KERNEL IMAG')]
    hdulist += [fits.ImageHDU(ccomponents_noiseless.real, name='CLEAN NOISELESS DELAY SPECTRA REAL')]
    hdulist += [fits.ImageHDU(ccomponents_noiseless.imag, name='CLEAN NOISELESS DELAY SPECTRA IMAG')]
    hdulist += [fits.ImageHDU(ccres_noiseless.real, name='CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL')]
    hdulist += [fits.ImageHDU(ccres_noiseless.imag, name='CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG')]
    hdulist += [fits.ImageHDU(cc_skyvis.real, name='CLEAN NOISELESS VISIBILITIES REAL')]
    hdulist += [fits.ImageHDU(cc_skyvis.imag, name='CLEAN NOISELESS VISIBILITIES IMAG')]
    hdulist += [fits.ImageHDU(cc_skyvis_res.real, name='CLEAN NOISELESS VISIBILITIES RESIDUALS REAL')]
    hdulist += [fits.ImageHDU(cc_skyvis_res.imag, name='CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG')]
    hdulist += [fits.ImageHDU(ccomponents_noisy.real, name='CLEAN NOISY DELAY SPECTRA REAL')]
    hdulist += [fits.ImageHDU(ccomponents_noisy.imag, name='CLEAN NOISY DELAY SPECTRA IMAG')]
    hdulist += [fits.ImageHDU(ccres_noisy.real, name='CLEAN NOISY DELAY SPECTRA RESIDUALS REAL')]
    hdulist += [fits.ImageHDU(ccres_noisy.imag, name='CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG')]
    hdulist += [fits.ImageHDU(cc_vis.real, name='CLEAN NOISY VISIBILITIES REAL')]
    hdulist += [fits.ImageHDU(cc_vis.imag, name='CLEAN NOISY VISIBILITIES IMAG')]
    hdulist += [fits.ImageHDU(cc_vis_res.real, name='CLEAN NOISY VISIBILITIES RESIDUALS REAL')]
    hdulist += [fits.ImageHDU(cc_vis_res.imag, name='CLEAN NOISY VISIBILITIES RESIDUALS IMAG')]
    
    hdu = fits.HDUList(hdulist)
    hdu.writeto(outfile+'.fits', clobber=True)




