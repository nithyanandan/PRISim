import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import progressbar as PGB
import interferometry as RI

telescope = 'mwa'
telescope_str = telescope + '_'
if telescope == 'mwa':
    telescope_str = ''
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
bl, bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
sortind = NP.argsort(bl_length, kind='mergesort')
bl = bl[sortind,:]
bl_id = bl_id[sortind]
bl_length = bl_length[sortind]
total_baselines = bl_length.size

n_bl_chunks = 200
baseline_chunk_size = 10
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)

baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]
bl = bl[:baseline_bin_indices[n_bl_chunks],:]
bl_length = bl_length[:baseline_bin_indices[n_bl_chunks]]
bl_id = bl_id[:baseline_bin_indices[n_bl_chunks]]

freq = 185.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 80e3 # in Hz
bpass_shape = 'rect'
f_pad = 1.0
oversampling_factor = 1.0 + f_pad
n_channels = 384
nchan = n_channels

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

n_sky_sectors = 1

nside = 128
use_GSM = True
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False

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
else:
    fg_str = 'other'

progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_bl_chunks).start()
for k in range(n_sky_sectors):
    if n_sky_sectors == 1:
        sky_sector_str = '_all_sky_'
    else:
        sky_sector_str = '_sky_sector_{0:0d}_'.format(k)

    for i in range(0, n_bl_chunks):
        infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[bl_chunk[i]]],bl_length[min(baseline_bin_indices[bl_chunk[i]]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6, nchan*freq_resolution/1e6)+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
        # infile = '/data3/t_nithyanandan/project_MWA/multi_baseline_visibilities_'+avg_drifts_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[i]],bl_length[min(baseline_bin_indices[i]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+'_{0:0d}_'.format(nside)+'{0:.1f}_MHz_'.format(nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_part_{0:0d}'.format(i)
        if i == 0:
            ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
        else:
            ia_next = RI.InterferometerArray(None, None, None, init_file=infile+'.fits')    
            ia.concatenate(ia_next, axis=0)
    outfile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(bl_length[baseline_bin_indices[0]],bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz'.format(freq/1e6, nchan*freq_resolution/1e6)
    
    ia.save(outfile, verbose=True, tabtype='BinTableHDU', overwrite=True)

    progress.update(i+1)
progress.finish()

