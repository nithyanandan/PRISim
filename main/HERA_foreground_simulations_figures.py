import yaml
import argparse
import numpy as NP
import scipy.constants as FCNST
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
from astropy.io import fits
import interferometry as RI
import delay_spectrum as DS
import my_operations as OPS
import my_DSP_modules as DSP
import primary_beams as PB
import ipdb as PDB

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')

input_group = parser.add_argument_group('Input parameters', 'Input specifications')
input_group.add_argument('-i', '--infile', dest='infile', default='/home/t_nithyanandan/codes/mine/python/interferometry/main/figparameters_HERA.yaml', type=file, required=False, help='File specifying input parameters')

args = vars(parser.parse_args())

with args['infile'] as parms_file:
    parms = yaml.safe_load(parms_file)

rootdir = parms['directory']['rootdir']
figdir = parms['directory']['figdir']
project = parms['project']
telescope_id = parms['telescope']['id']
pfb_method = parms['telescope']['pfb_method']
element_shape = parms['antenna']['shape']
element_size = parms['antenna']['size']
element_ocoords = parms['antenna']['ocoords']
element_orientation = parms['antenna']['orientation']
ground_plane = parms['antenna']['ground_plane']
phased_array = parms['antenna']['phased_array']
delayerr = parms['phasedarray']['delayerr']
gainerr = parms['phasedarray']['gainerr']
nrand = parms['phasedarray']['nrand']
antenna_file = parms['array']['file']
array_layout = parms['array']['layout']
minR = parms['array']['minR']
maxR = parms['array']['maxR']
minbl = parms['baseline']['min']
maxbl = parms['baseline']['max']
baseline_chunk_size = parms['processing']['bl_chunk_size']
n_bl_chunks = parms['processing']['n_bl_chunks']
fg_str = parms['fgparm']['model']
nside = parms['fgparm']['nside']
Tsys = parms['telescope']['Tsys']
latitude = parms['telescope']['latitude']
longitude = parms['telescope']['longitude']
if longitude is None:
    longitude = 0.0
freq = parms['obsparm']['freq']
freq_resolution = parms['obsparm']['freq_resolution']
obs_mode = parms['obsparm']['obs_mode']
nchan = parms['obsparm']['nchan']
n_acc = parms['obsparm']['n_acc']
t_acc = parms['obsparm']['t_acc']
beam_info = parms['beam']
beam_id = beam_info['identifier']
beam_file = beam_info['file']
select_beam_freq = beam_info['select_freq']
if select_beam_freq is None:
    select_beam_freq = freq
pbeam_spec_interp_method = beam_info['spec_interp']
n_sky_sectors = parms['processing']['n_sky_sectors']
bpass_shape = parms['clean']['bpass_shape']
spindex_rms = parms['fgparm']['spindex_rms']
spindex_seed = parms['fgparm']['spindex_seed']
plot_info = parms['plot']
plots = [key for key in plot_info if plot_info[key]]

if project not in ['project_MWA', 'project_global_EoR', 'project_HERA', 'project_drift_scan', 'project_beams', 'project_LSTbin']:
    raise ValueError('Invalid project specified')
else:
    project_dir = project + '/'
figuresdir = rootdir + project_dir + figdir + '/'

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

use_GSM = False
use_DSM = False
use_CSM = False
use_SUMSS = False
use_GLEAM = False
use_USM = False
use_MSS = False
use_PS = False
use_NVSS = False
use_HI_monopole = False
use_HI_cube = False
use_HI_fluctuations = False

if fg_str not in ['asm', 'dsm', 'csm', 'nvss', 'sumss', 'gleam', 'mwacs', 'ps', 'point', 'usm', 'mss', 'HI_cube', 'HI_monopole', 'HI_fluctuations']:
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
elif fg_str in ['point', 'PS']:
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
else:
    fg_str = 'other'

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

if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane
telescope['latitude'] = latitude
telescope['longitude'] = longitude

freq = NP.float(freq)
freq_resolution = NP.float(freq_resolution)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution)/ 1e9 # in GHz
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)

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
        ant_info = NP.loadtxt(rootdir+'project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
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

bl, bl_id = RI.baseline_generator(ant_locs, ant_id=ant_id, auto=False, conjugate=False)
bl, select_bl_ind, bl_count = RI.uniq_baselines(bl)
bl_id = bl_id[select_bl_ind]
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_id = bl_id[sortind]
bl_length = bl_length[sortind]
total_baselines = bl_length.size

baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

if use_HI_monopole:
    bllstr = map(str, bl_length)
    uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
    count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
    count_uniq_bll = NP.asarray(count_uniq_bll)

    bl = bl[ind_uniq_bll,:]
    bl_id = bl_id[ind_uniq_bll]
    bl_length = bl_length[ind_uniq_bll]

    sortind = NP.argsort(bl_length, kind='mergesort')
    bl = bl[sortind,:]
    bl_id = bl_id[sortind]
    bl_length = bl_length[sortind]
    count_uniq_bll = count_uniq_bll[sortind]

total_baselines = bl_length.size
baseline_bin_indices = range(0, int(NP.ceil(1.0*total_baselines/baseline_chunk_size)+1)*baseline_chunk_size, baseline_chunk_size)
if n_bl_chunks is None:
    n_bl_chunks = int(NP.ceil(1.0*total_baselines/baseline_chunk_size))

bl_chunk = range(len(baseline_bin_indices)-1)
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:min(baseline_bin_indices[n_bl_chunks], total_baselines),:]
bl_length = bl_length[:min(baseline_bin_indices[n_bl_chunks], total_baselines)]
bl_id = bl_id[:min(baseline_bin_indices[n_bl_chunks], total_baselines)]

pfb_instr = ''
pfb_outstr = ''
if pfb_method is None:
    pfb_instr = '_no_pfb'
    pfb_outstr = 'no_pfb_'

external_beam = fits.getdata(beam_file, extname='BEAM_X')
external_beam_freqs = fits.getdata(beam_file, extname='FREQS_X')

achromatic_extbeam_str = 'extpb_'+beam_id+'_{0:.1f}_MHz_achromatic'.format(select_beam_freq/1e6)
chromatic_extbeam_str = 'extpb_'+beam_id+'_chromatic'
funcbeam_str = 'funcpb_chromatic'

spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

snapshot_type_str = ''

duration_str = ''
if obs_mode in ['track', 'drift']:
    if (t_acc is not None) and (n_acc is not None):
        duration_str = '_{0:0d}x{1:.1f}s'.format(n_acc, t_acc)

if n_sky_sectors == 1:
    sky_sector_str = '_all_sky_'
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(n_sky_sectors)

fgvisfile_achrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+achromatic_extbeam_str+pfb_instr
fgvisfile_chrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+chromatic_extbeam_str+pfb_instr
fgvisfile_funcbeam = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+funcbeam_str+pfb_instr

fgdsfile_achrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+achromatic_extbeam_str+'_'+pfb_outstr+bpass_shape
fgdsfile_chrmbeam = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+chromatic_extbeam_str+'_'+pfb_outstr+bpass_shape
fgdsfile_funcbeam = rootdir+project_dir+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+funcbeam_str+'_'+pfb_outstr+bpass_shape

fgvis_achrmbeam = RI.InterferometerArray(None, None, None, init_file=fgvisfile_achrmbeam+'.fits')
fgvis_chrmbeam = RI.InterferometerArray(None, None, None, init_file=fgvisfile_chrmbeam+'.fits')
fgvis_funcbeam = RI.InterferometerArray(None, None, None, init_file=fgvisfile_funcbeam+'.fits')
fgds_achrmbeam = DS.DelaySpectrum(init_file=fgdsfile_achrmbeam+'.ds.fits')
fgds_chrmbeam = DS.DelaySpectrum(init_file=fgdsfile_chrmbeam+'.ds.fits')
fgds_funcbeam = DS.DelaySpectrum(init_file=fgdsfile_funcbeam+'.ds.fits')

##########################################

if '1a' in plots:
        
    # 01-a) Plot beam chromaticity with single point source at different locations

    alt = NP.asarray([90.0, 45.0, 1.0])
    az = 270.0 + NP.zeros(alt.size)

    altaz = NP.hstack((alt.reshape(-1,1), az.reshape(-1,1)))
    thetaphi = NP.radians(NP.hstack((90.0-alt.reshape(-1,1), az.reshape(-1,1))))

    chrm_extbeam = 10 ** OPS.healpix_interp_along_axis(NP.log10(external_beam), theta_phi=thetaphi, inloc_axis=external_beam_freqs, outloc_axis=chans*1e3, axis=1, kind=pbeam_spec_interp_method, assume_sorted=True)
    nearest_freq_ind = NP.argmin(NP.abs(external_beam_freqs*1e6 - select_beam_freq))
    achrm_extbeam = 10 ** OPS.healpix_interp_along_axis(NP.log10(NP.repeat(external_beam[:,nearest_freq_ind].reshape(-1,1), chans.size, axis=1)), theta_phi=thetaphi, inloc_axis=chans*1e3, outloc_axis=chans*1e3, axis=1, assume_sorted=True)
    funcbeam = PB.primary_beam_generator(altaz, chans, telescope, freq_scale='GHz', skyunits='altaz', east2ax1=0.0, pointing_info=None, pointing_center=None)

    pad = 0.0
    npad = int(pad * chans.size)
    lags = DSP.spectral_axis(npad+chans.size, delx=chans[1]-chans[0], shift=True)
    chrm_extbeam_FFT = NP.fft.fft(NP.pad(chrm_extbeam, ((0,0),(0,npad)), mode='constant'), axis=1) / (npad+chans.size)
    chrm_extbeam_FFT = NP.fft.fftshift(chrm_extbeam_FFT, axes=1)
    chrm_extbeam_FFT_max = NP.max(NP.abs(chrm_extbeam_FFT), axis=1, keepdims=True)
    achrm_extbeam_FFT = NP.fft.fft(NP.pad(achrm_extbeam, ((0,0),(0,npad)), mode='constant'), axis=1) / (npad+chans.size)
    achrm_extbeam_FFT = NP.fft.fftshift(achrm_extbeam_FFT, axes=1)
    achrm_extbeam_FFT_max = NP.max(NP.abs(achrm_extbeam_FFT), axis=1, keepdims=True)
    funcbeam_FFT = NP.fft.fft(NP.pad(funcbeam, ((0,0),(0,npad)), mode='constant'), axis=1) / (npad+chans.size)
    funcbeam_FFT = NP.fft.fftshift(funcbeam_FFT, axes=1)
    funcbeam_FFT_max = NP.max(NP.abs(funcbeam_FFT), axis=1, keepdims=True)

    fig, axs = PLT.subplots(ncols=3, sharex=True, sharey=True, figsize=(7,3.5))
    for alti, elev in enumerate(alt):
        # axs[alti].plot(lags, NP.abs(achrm_extbeam_FFT[alti,:])**2/achrm_extbeam_FFT_max[alti]**2, lw=2, ls='--', color='k')
        axs[alti].plot(lags, NP.abs(chrm_extbeam_FFT[alti,:])**2, lw=2, ls='--', color='k', label='Sim.')
        axs[alti].plot(lags, NP.abs(funcbeam_FFT[alti,:])**2, lw=2, ls='-', color='k', label='Dish')
        # axs[alti].plot(lags, NP.abs(chrm_extbeam_FFT[alti,:])**2/chrm_extbeam_FFT_max[alti]**2, lw=2, ls=':', color='k', label='Sim.')
        # axs[alti].plot(lags, NP.abs(funcbeam_FFT[alti,:])**2/funcbeam_FFT_max[alti]**2, lw=2, ls='-.', color='k', label='Dish')
        axs[alti].set_yscale('log')
        axs[alti].axvline(x=-1e9*bl_length[0]/FCNST.c, ymax=0.67, color='gray', ls='-', lw=3)
        axs[alti].axvline(x=1e9*bl_length[0]/FCNST.c, ymax=0.67, color='gray', ls='-', lw=3) 
        axs[alti].text(0.22, 0.95, '{0:.1f}'.format(90.0-elev)+r'$^\circ$'+'\noff-axis', transform=axs[alti].transAxes, fontsize=12, weight='medium', ha='center', va='top', color='black')
        axs[alti].legend(frameon=True, fontsize=10)
        axs[alti].set_xlim(-250, 250)
        axs[alti].set_ylim(1e-12, 1e-3)
        axs[alti].set_aspect('auto')
    fig.subplots_adjust(hspace=0, wspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel(r'$|\widetilde{V}(\eta)|^2$ [arbitrary units]', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel(r'$\tau$ [ns]', fontsize=16, weight='medium', labelpad=20)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.17)
    fig.subplots_adjust(left=0.12)
    fig.subplots_adjust(right=0.98)    

    PLT.savefig(figuresdir+'off-axis_point_source_beam_chromaticity.png', bbox_inches=0)
    PLT.savefig(figuresdir+'off-axis_point_source_beam_chromaticity.eps', bbox_inches=0)    

    print '\n\tPlotted and saved off-axis point source beam chromaticity'

if '1b' in plots:
        
    # 01-b) Plot all-sky foreground delay power spectra with different beam chromaticities

    freq_window_centers = {key: [150e6, 170e6] for key in ['cc', 'sim']}
    freq_window_bw = {key: [10e6, 10e6] for key in ['cc', 'sim']}
    freq_window_shape={key: 'bhw' for key in ['cc', 'sim']}

    fgds_achrmbeam_sbds = fgds_achrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, pad=None, bpcorrect=False, action='return')
    fgds_chrmbeam_sbds = fgds_chrmbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, pad=None, bpcorrect=False, action='return')
    fgds_funcbeam_sbds = fgds_funcbeam.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, pad=None, bpcorrect=False, action='return')
    fgdps_achrmbeam = DS.DelayPowerSpectrum(fgds_achrmbeam)
    fgdps_achrmbeam.compute_power_spectrum()
    fgdps_chrmbeam = DS.DelayPowerSpectrum(fgds_chrmbeam)
    fgdps_chrmbeam.compute_power_spectrum()
    fgdps_funcbeam = DS.DelayPowerSpectrum(fgds_funcbeam)
    fgdps_funcbeam.compute_power_spectrum()
    
    kprll = fgdps_achrmbeam.k_parallel(fgds_achrmbeam.cc_lags, fgdps_achrmbeam.z, action='return')
    for fwi, fw in enumerate(freq_window_centers['cc']):
        fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(4,3))
        axs[0].plot(kprll, fgdps_achrmbeam.dps['skyvis'][0,:,0], color='blue', lw=2, ls='-')
        axs[0].plot(kprll, fgdps_chrmbeam.dps['skyvis'][0,:,0], color='blue', lw=2, ls=':')
        axs[0].plot(kprll, fgdps_funcbeam.dps['skyvis'][0,:,0], color='blue', lw=2, ls='--')
        axs[0].plot(fgdps_achrmbeam.subband_delay_power_spectra['sim']['kprll'][fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'][0,fwi,:,0], color='black', lw=2, ls='-')
        axs[0].plot(fgdps_chrmbeam.subband_delay_power_spectra['sim']['kprll'][fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['sim']['skyvis_lag'][0,fwi,:,0], color='black', lw=2, ls=':')
        axs[0].plot(fgdps_funcbeam.subband_delay_power_spectra['sim']['kprll'][fwi,:], fgdps_funcbeam.subband_delay_power_spectra['sim']['skyvis_lag'][0,fwi,:,0], color='black', lw=2, ls='--')
        axs[1].plot(fgdps_achrmbeam.subband_delay_power_spectra['cc']['kprll'][fwi,:], fgdps_achrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][0,fwi,:,0], color='black', lw=2, ls='-')
        axs[1].plot(fgdps_chrmbeam.subband_delay_power_spectra['cc']['kprll'][fwi,:], fgdps_chrmbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][0,fwi,:,0], color='black', lw=2, ls=':')
        axs[1].plot(fgdps_funcbeam.subband_delay_power_spectra['cc']['kprll'][fwi,:], fgdps_funcbeam.subband_delay_power_spectra['cc']['skyvis_res_lag'][0,fwi,:,0], color='black', lw=2, ls='--')    
        axs[1].plot(kprll, fgdps_achrmbeam.dps['skyvis'][0,:,0], color='blue', lw=2, ls='-')
        axs[1].plot(kprll, fgdps_achrmbeam.dps['cc_skyvis'][0,:,0], color='green', lw=2, ls='-')
        axs[1].plot(kprll, fgdps_achrmbeam.dps['cc_skyvis_res'][0,:,0], color='red', lw=2, ls='-')
        axs[1].plot(kprll, fgdps_chrmbeam.dps['skyvis'][0,:,0], color='blue', lw=2, ls=':')
        axs[1].plot(kprll, fgdps_chrmbeam.dps['cc_skyvis'][0,:,0], color='green', lw=2, ls=':')
        axs[1].plot(kprll, fgdps_chrmbeam.dps['cc_skyvis_res'][0,:,0], color='red', lw=2, ls=':')
        axs[1].plot(kprll, fgdps_funcbeam.dps['skyvis'][0,:,0], color='blue', lw=2, ls='--')
        axs[1].plot(kprll, fgdps_funcbeam.dps['cc_skyvis'][0,:,0], color='green', lw=2, ls='--')
        axs[1].plot(kprll, fgdps_funcbeam.dps['cc_skyvis_res'][0,:,0], color='red', lw=2, ls='--')
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[0].set_aspect('auto')
        axs[1].set_aspect('auto')        
        fig.subplots_adjust(hspace=0, wspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel(r'$P(k_\parallel)$'+ r'[K$^2$ (Mpc/h)$^3$]', fontsize=16, weight='medium', labelpad=30)
        big_ax.set_xlabel(r'$k_\parallel$ [h/Mpc]', fontsize=16, weight='medium', labelpad=20)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(bottom=0.17)
        fig.subplots_adjust(left=0.12)
        fig.subplots_adjust(right=0.98)    
        
    

PDB.set_trace()
