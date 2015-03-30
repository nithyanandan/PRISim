from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import numpy as NP

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
if project_MWA:
    delaygain_err_str = ''

# latitude = -26.701 
# antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

max_bl_length = None # Maximum baseline length (in m)

# ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
# ref_bl, ref_bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
# ref_bl_length = NP.sqrt(NP.sum(ref_bl**2, axis=1))
# ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True) 
# neg_ref_bl_orientation_ind = ref_bl_orientation < 0.0
# ref_bl[neg_ref_bl_orientation_ind,:] = -1.0 * ref_bl[neg_ref_bl_orientation_ind,:]
# ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True)
# sortind = NP.argsort(ref_bl_length, kind='mergesort')
# ref_bl = ref_bl[sortind,:]
# ref_bl_length = ref_bl_length[sortind]
# ref_bl_orientation = ref_bl_orientation[sortind]
# ref_bl_id = ref_bl_id[sortind]
# n_bins_baseline_orientation = 4
# nmax_baselines = 2048
# ref_bl = ref_bl[:nmax_baselines,:]
# ref_bl_length = ref_bl_length[:nmax_baselines]
# ref_bl_id = ref_bl_id[:nmax_baselines]
# ref_bl_orientation = ref_bl_orientation[:nmax_baselines]
# total_baselines = ref_bl_length.size

Tsys = 440.0 # System temperature in K
freq = 150.0e6 # center frequency in Hz
oversampling_factor = 2.0
n_sky_sectors = 1
sky_sector = None # if None, use all sky sector. Accepted values are None, 0, 1, 2, or 3
if sky_sector is None:
    sky_sector_str = '_all_sky_'
    n_sky_sectors = 1
    sky_sector = 0
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(sky_sector)

cspeed = 299792458.0  # Speed of light in m/s

# n_bl_chunks = 16
# baseline_chunk_size = 128
# baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
# bl_chunk = range(len(baseline_bin_indices))
# bl_chunk = bl_chunk[:n_bl_chunks]

# truncated_ref_bl = NP.copy(ref_bl)
# truncated_ref_bl_id = NP.copy(ref_bl_id)
# truncated_ref_bl_length = NP.sqrt(NP.sum(truncated_ref_bl[:,:2]**2, axis=1))
# # truncated_ref_bl_length = NP.copy(ref_bl_length)
# truncated_ref_bl_orientation = NP.copy(ref_bl_orientation)
# truncated_total_baselines = truncated_ref_bl_length.size
# if max_bl_length is not None:
#     truncated_ref_bl_ind = ref_bl_length <= max_bl_length
#     truncated_ref_bl = truncated_ref_bl[truncated_ref_bl_ind,:]
#     truncated_ref_bl_id = truncated_ref_bl_id[truncated_ref_bl_ind]
#     truncated_ref_bl_orientation = truncated_ref_bl_orientation[truncated_ref_bl_ind]
#     truncated_ref_bl_length = truncated_ref_bl_length[truncated_ref_bl_ind]
#     truncated_total_baselines = truncated_ref_bl_length.size

spindex_rms = 0.0
spindex_seed = None
spindex_seed_str = ''
if spindex_rms > 0.0:
    spindex_rms_str = '{0:.1f}'.format(spindex_rms)
else:
    spindex_rms = 0.0

if spindex_seed is not None:
    spindex_seed_str = '{0:0d}_'.format(spindex_seed)

use_alt_spindex = False
alt_spindex_rms = 0.3
alt_spindex_seed = 95
alt_spindex_seed_str = ''
if alt_spindex_rms > 0.0:
    alt_spindex_rms_str = '{0:.1f}'.format(alt_spindex_rms)
else:
    alt_spindex_rms = 0.0

if alt_spindex_seed is not None:
    alt_spindex_seed_str = '{0:0d}_'.format(alt_spindex_seed)

nside = 64
use_GSM = True
use_DSM = False
use_CSM = False
use_NVSS = False
use_SUMSS = False
use_MSS = False
use_GLEAM = False
use_PS = False

obs_mode = 'drift'
avg_drifts = False
beam_switch = False
snapshot_type_str = ''
if avg_drifts:
    snapshot_type_str = 'drift_averaged_'
if beam_switch:
    snapshot_type_str = 'beam_switches_'

freq_resolution = 80e3  # in kHz
nchan = 96
bw = nchan * freq_resolution

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

# Edit the following line to give the full and correct path to the simulation data file

infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(7.7, 321.7)+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)

hdulist = fits.open(infile+'.fits')

extnames = [hdulist[i].header['EXTNAME'] for i in range(1,len(hdulist))]  # Contains the names of different extensions in the FITS file
# Generally, You can access an extension by: "quantity = hdulist[name of extension].data" such as shown below

labels = hdulist['LABELS'].data['labels']
bl = hdulist['BASELINES'].data
timestamps = hdulist['TIMESTAMPS'].data['timestamps']
chans = hdulist['SPECTRAL INFO'].data['frequency']
chan_num = NP.arange(chans.size)
vis = hdulist['REAL_FREQ_OBS_VISIBILITY'].data + 1j * hdulist['IMAG_FREQ_OBS_VISIBILITY'].data
skyvis = hdulist['REAL_FREQ_SKY_VISIBILITY'].data + 1j * hdulist['IMAG_FREQ_SKY_VISIBILITY'].data
bpass = hdulist['BANDPASS'].data
blproj = hdulist['PROJ_BASELINES'].data

n_timestamps = timestamps.size
n_baselines = bl.shape[0]
nchan = chans.size

wavlen = cspeed/chans

## Uncomment the next few lines if you want coarse band edges flagged

# vis = vis * bpass
# skyvis = skyvis * bpass

## Uncomment above lines if you want coarse band edges flagged

bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
timestamps_table = NP.tile(timestamps.reshape(-1,1,1), (1,n_baselines, nchan)).ravel()
labels_table = NP.tile(labels.reshape(1,-1,1), (n_timestamps, 1, nchan)).ravel()
blproj_table = blproj[NP.newaxis,...]   # now it is 1 x n_baselines x 3 x n_timestamps
blproj_table = NP.swapaxes(blproj_table, 0, 3) / wavlen.reshape(1,1,1,-1)  # now it is n_timestamps x n_baselines x 3 x nchan
u_table = blproj_table[:,:,0,:] # now it is n_timestamps x n_baselines x nchan
v_table = blproj_table[:,:,1,:] # now it is n_timestamps x n_baselines x nchan
w_table = blproj_table[:,:,2,:] # now it is n_timestamps x n_baselines x nchan
u_table = u_table.ravel()
v_table = v_table.ravel()
w_table = w_table.ravel()
uvdist_table = NP.sqrt(u_table**2 + v_table**2 + w_table**2)
# uvdist = bl_length.reshape(1,-1,1) / wavlen.reshape(1,1,-1)
# uvdist_table = NP.repeat(uvdist, n_timestamps, axis=0).ravel()
channum_table = NP.tile(chan_num.reshape(1,1,-1), (n_timestamps, n_baselines, 1)).ravel()
vis_amp_table = NP.rollaxis(NP.abs(vis), 2, start=0).ravel()
vis_phs_table = NP.rollaxis(NP.angle(vis, deg=True), 2, start=0).ravel()
skyvis_amp_table = NP.rollaxis(NP.abs(skyvis), 2, start=0).ravel()
skyvis_phs_table = NP.rollaxis(NP.angle(skyvis, deg=True), 2, start=0).ravel()

frmts = {}
frmts['Timestamp'] = '%s12'
frmts['Label'] = '%s'
frmts['UVDist'] = '%0.2f'
frmts['Chn'] = '%0d'
frmts['Amp'] = '%0.3f'
frmts['Phs'] = '%-0.2f'
frmts['U'] = '%-0.2f'
frmts['V'] = '%-0.2f'
frmts['W'] = '%-0.2f'

## Test out first with smaller data set only first 5000 lines. Once tested, remove "[:5000]" from everywhere below to write out the entire data set

data_dict = {}
data_dict['Timestamp'] = timestamps_table[:5000]
data_dict['Label'] = labels_table[:5000]
data_dict['UVDist'] = uvdist_table[:5000]
data_dict['Chn'] = channum_table[:5000]
data_dict['U'] = u_table[:5000]
data_dict['V'] = v_table[:5000]
data_dict['W'] = w_table[:5000]

## Swap the commented and uncommented lines below if you want to switch between noisy and noiseles sky visibilities

data_dict['Amp'] = vis_amp_table[:5000]
data_dict['Phs'] = vis_phs_table[:5000]

# data_dict['Amp'] = skyvis_amp_table[:5000]
# data_dict['Phs'] = skyvis_phs_table[:5000]

## Swap the commented and uncommented lines above if you want to switch between noisy and noiseles sky visibilities

# Edit the following line to give full path to your desired output text file

outfile = infile+'.txt'

tbdata = Table(data_dict, names=['Timestamp', 'Label', 'UVDist', 'Chn', 'Amp', 'Phs', 'U', 'V', 'W'])
ascii.write(tbdata, output=outfile, format='fixed_width_two_line', formats=frmts, bookend=False, delimiter='|', delimiter_pad = ' ')

hdulist.close()


