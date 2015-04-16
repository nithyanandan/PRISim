import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import progressbar as PGB
import interferometry as RI
import ipdb as PDB

rootdir = '/data3/t_nithyanandan/'
# rootdir = '/data3/MWA/lstbin_RA0/NT/'

filenaming_convention = 'new'
# filenaming_convention = 'old'

project_MWA = False
project_LSTbin = False
project_HERA = False
project_beams = False
project_drift_scan = False
project_global_EoR = True

project_dir = ''
if project_MWA: project_dir = 'project_MWA/'
if project_LSTbin:
    if rootdir == '/data3/t_nithyanandan/':
        project_dir = 'project_LSTbin/'
if project_HERA: project_dir = 'project_HERA/'
if project_beams: project_dir = 'project_beams/'
if project_drift_scan: project_dir = 'project_drift_scan/'
if project_global_EoR: project_dir = 'project_global_EoR/'

telescope_id = 'custom'
element_size = 14.0
element_shape = 'dish'
phased_array = False

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

ground_plane = None # height of antenna element above ground plane in m
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

array_layout = 'CIRC'
# array_layout = 'MWA-128T'
# array_layout = 'HERA-331'

minR = 141.0
maxR = None

if array_layout == 'MWA-128T':
    ant_info = NP.loadtxt('/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
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

n_bl_chunks = 32
baseline_chunk_size = 62

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
use_HI_monopole = False
use_HI_fluctuations = False
use_HI_cube = False

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
elif use_HI_monopole:
    fg_str = 'HI_monopole'
elif use_HI_fluctuations:
    fg_str = 'HI_fluctuations'
elif use_HI_cube:
    fg_str = 'HI_cube'
else:
    fg_str = 'other'

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
n_bl_chunks = int(NP.ceil(1.0*total_baselines/baseline_chunk_size))

bl_chunk = range(len(baseline_bin_indices)-1)
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:min(baseline_bin_indices[n_bl_chunks], total_baselines),:]
bl_length = bl_length[:min(baseline_bin_indices[n_bl_chunks], total_baselines)]
bl_id = bl_id[:min(baseline_bin_indices[n_bl_chunks], total_baselines)]

Tsys = 300.0 # System temperature in K
freq = 150.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 400e3 # in Hz
bpass_shape = 'rect'
f_pad = 1.0
oversampling_factor = 1.0 + f_pad
n_channels = 256
nchan = n_channels
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)

use_pfb = False

pfb_instr = ''
pfb_outstr = ''
if not use_pfb: 
    pfb_instr = 'no_pfb_'
    pfb_outstr = '_no_pfb'

obs_mode = 'drift'
# obs_mode = 'custom'
# obs_mode = 'lstbin'
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

duration_str = ''
if obs_mode in ['track', 'drift']:
    t_snap = 540.0    # in seconds
    n_snaps = 80
    if (t_snap is not None) and (n_snaps is not None):
        duration_str = '_{0:0d}x{1:.1f}s'.format(n_snaps, t_snap)

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

for k in range(n_sky_sectors):
    if n_sky_sectors == 1:
        sky_sector_str = '_all_sky_'
    else:
        sky_sector_str = '_sky_sector_{0:0d}_'.format(k)

    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_bl_chunks).start()
    for i in range(0, n_bl_chunks):
        if filenaming_convention == 'old':
            infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_instr+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
        else:
            infile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz_'.format(Tsys, bandpass_str, freq/1e6)+pfb_instr+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)

        # infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
        if i == 0:
            ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
        else:
            ia_next = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
            ia.concatenate(ia_next, axis=0)

        progress.update(i+1)
    progress.finish()

    if filenaming_convention == 'old':
        outfile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+pfb_outstr
    else:
        outfile = rootdir+project_dir+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+duration_str+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1}_{2:.1f}_MHz'.format(Tsys, bandpass_str, freq/1e6)+pfb_outstr        
    
    ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)


