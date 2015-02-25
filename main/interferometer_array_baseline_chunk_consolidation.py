import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import progressbar as PGB
import interferometry as RI

project_MWA = False
project_HERA = False
project_beams = False
project_drift_scan = True
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

ground_plane = 0.3 # height of antenna element above ground plane in m
if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

delayerr = 0.0     # delay error rms in ns
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
bl_id = bl_id[sortind]
bl_length = bl_length[sortind]
total_baselines = bl_length.size

n_bl_chunks = 16
baseline_chunk_size = 128
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:baseline_bin_indices[n_bl_chunks],:]
bl_length = bl_length[:baseline_bin_indices[n_bl_chunks]]
bl_id = bl_id[:baseline_bin_indices[n_bl_chunks]]

Tsys = 440.0 # System temperature in K
freq = 150.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 80e3 # in Hz
bpass_shape = 'rect'
f_pad = 1.0
oversampling_factor = 1.0 + f_pad
n_channels = 96
nchan = n_channels

use_pfb = True

pfb_instr = ''
pfb_outstr = ''
if not use_pfb: 
    pfb_instr = 'no_pfb_'
    pfb_outstr = '_no_pfb'

obs_mode = 'drift'
# obs_mode = 'custom'
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

    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_bl_chunks).start()
    for i in range(0, n_bl_chunks):
        infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_instr+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
        # infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
        if i == 0:
            ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
        else:
            ia_next = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
            ia.concatenate(ia_next, axis=0)

        progress.update(i+1)
    progress.finish()

    outfile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_outstr
    
    ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)


