import numpy as NP
import copy
from astropy.io import fits
from astropy.io import ascii
from astropy import coordinates as coord
from astropy.coordinates import Galactic, FK5
from astropy import units
import astropy.cosmology as CP
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
from matplotlib.ticker import FuncFormatter
import healpy as HP
from mwapy.pb import primary_beam as MWAPB
import geometry as GEOM
import interferometry as RI
import catalog as SM
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import lookup_operations as LKP
import ipdb as PDB

# 01) Plot pointings information

# 02) Plot power patterns for snapshots and ratios relative to nominal power patterns

# 03) Plot foreground models with power pattern contours for snapshots

# 04) Plot delay maps on sky for baselines of different orientations

# 05) Plot FHD data and simulations on all baselines combined

# 06) Plot FHD data to simulation ratio on all baselines combined

# 07) Plot uncertainties in FHD data to simulation ratio on all baselines combined

# 08) Plot ratio of differences between FHD data and simulation to expected error on all baselines combined

# 09) Plot histogram of fractional differences between FHD data and simulation 

# 10) Plot noiseless delay spectra from simulations for diffuse, compact and all-sky models

# 11) Plot noiseless delay spectra for all sky models broken down by baseline orientation

# 12) Plot delay spectra on northward and eastward baselines along with delay maps and sky models (with and without power pattern contours)

# 13) Plot EoR window foreground contamination when baselines are selectively removed

# 14) Plot delay spectra before and after baselines are selectively removed

# 15) Plot Fourier space

# 16) Plot average thermal noise in simulations and data as a function of baseline length

# 17) Plot delay spectra of the MWA tile power pattern using a uniform sky model

# 18) Plot delay spectra of the all-sky model with dipole, MWA tile, and HERA dish antenna shapes

# 19) Plot delay spectrum of uniform sky model with a uniform power pattern

plot_01 = False
plot_02 = True
plot_03 = False
plot_04 = False
plot_05 = False
plot_06 = False
plot_07 = False
plot_08 = False
plot_09 = False
plot_10 = False
plot_11 = False
plot_12 = False
plot_13 = False
plot_14 = False
plot_15 = False
plot_16 = False
plot_17 = False
plot_18 = False
plot_19 = False

# PLT.ioff()
PLT.ion()

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
element_shape = 'dipole'
element_orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
element_ocoords = 'altaz'
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

ground_plane = 0.278 # height of antenna element above ground plane
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

latitude = -26.701 
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

max_bl_length = 200.0 # Maximum baseline length (in m)
ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ref_bl, ref_bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
ref_bl_length = NP.sqrt(NP.sum(ref_bl**2, axis=1))
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True) 
neg_ref_bl_orientation_ind = ref_bl_orientation < 0.0
ref_bl[neg_ref_bl_orientation_ind,:] = -1.0 * ref_bl[neg_ref_bl_orientation_ind,:]
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True)
sortind = NP.argsort(ref_bl_length, kind='mergesort')
ref_bl = ref_bl[sortind,:]
ref_bl_length = ref_bl_length[sortind]
ref_bl_orientation = ref_bl_orientation[sortind]
ref_bl_id = ref_bl_id[sortind]
n_bins_baseline_orientation = 4
nmax_baselines = 2048
ref_bl = ref_bl[:nmax_baselines,:]
ref_bl_length = ref_bl_length[:nmax_baselines]
ref_bl_id = ref_bl_id[:nmax_baselines]
ref_bl_orientation = ref_bl_orientation[:nmax_baselines]
total_baselines = ref_bl_length.size
Tsys = 95.0 # System temperature in K
freq = 185.0e6 # center frequency in Hz
wavelength = FCNST.c / freq  # in meters
redshift = CNST.rest_freq_HI / freq - 1
oversampling_factor = 2.0
n_sky_sectors = 1
sky_sector = None # if None, use all sky sector. Accepted values are None, 0, 1, 2, or 3
if sky_sector is None:
    sky_sector_str = '_all_sky_'
    n_sky_sectors = 1
    sky_sector = 0
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(sky_sector)

n_bl_chunks = 32
baseline_chunk_size = 64
total_baselines = ref_bl_length.size
baseline_bin_indices = range(0,total_baselines,baseline_chunk_size)
bl_chunk = range(len(baseline_bin_indices))
bl_chunk = bl_chunk[:n_bl_chunks]

truncated_ref_bl = NP.copy(ref_bl)
truncated_ref_bl_id = NP.copy(ref_bl_id)
truncated_ref_bl_length = NP.sqrt(NP.sum(truncated_ref_bl[:,:2]**2, axis=1))
# truncated_ref_bl_length = NP.copy(ref_bl_length)
truncated_ref_bl_orientation = NP.copy(ref_bl_orientation)
truncated_total_baselines = truncated_ref_bl_length.size
if max_bl_length is not None:
    truncated_ref_bl_ind = ref_bl_length <= max_bl_length
    truncated_ref_bl = truncated_ref_bl[truncated_ref_bl_ind,:]
    truncated_ref_bl_id = truncated_ref_bl_id[truncated_ref_bl_ind]
    truncated_ref_bl_orientation = truncated_ref_bl_orientation[truncated_ref_bl_ind]
    truncated_ref_bl_length = truncated_ref_bl_length[truncated_ref_bl_ind]
    truncated_total_baselines = truncated_ref_bl_length.size

bl_orientation_str = ['South-East', 'East', 'North-East', 'North']

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

obs_mode = 'custom'
avg_drifts = False
beam_switch = False
snapshot_type_str = ''
if avg_drifts:
    snapshot_type_str = 'drift_averaged_'
if beam_switch:
    snapshot_type_str = 'beam_switches_'

freq_resolution = 80e3  # in kHz
nchan = 384
bpass_shape = 'bhw'
max_abs_delay = 1.5 # in micro seconds
coarse_channel_resolution = 1.28e6 # in Hz
bw = nchan * freq_resolution
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution) # in Hz

dsm_base_freq = 408e6 # Haslam map frequency
csm_base_freq = 1.420e9 # NVSS frequency
dsm_dalpha = 0.7/2 # Spread in spectral index in Haslam map
csm_dalpha = 0.7/2 # Spread in spectral index in NVSS
csm_jacobian_spindex = NP.abs(csm_dalpha * NP.log(freq/csm_base_freq))
dsm_jacobian_spindex = NP.abs(dsm_dalpha * NP.log(freq/dsm_base_freq))

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

# roifile = '/data3/t_nithyanandan/'+project_dir+'/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
# roi = RI.ROI_parameters(init_file=roifile)
# telescope = roi.telescope

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane
element_locs = None
if phased_array:
    phased_elements_file = '/data3/t_nithyanandan/project_MWA/MWA_tile_dipole_locations.txt'
    try:
        element_locs = NP.loadtxt(phased_elements_file, skiprows=1, comments='#', usecols=(0,1,2))
    except IOError:
        raise IOError('Could not open the specified file for phased array of antenna elements.')

if telescope_id == 'mwa':
    xlocs, ylocs = NP.meshgrid(1.1*NP.linspace(-1.5,1.5,4), 1.1*NP.linspace(1.5,-1.5,4))
    element_locs = NP.hstack((xlocs.reshape(-1,1), ylocs.reshape(-1,1), NP.zeros(xlocs.size).reshape(-1,1)))

if element_locs is not None:
    telescope['element_locs'] = element_locs

if telescope_id == 'custom':
    if element_shape == 'delta':
        telescope_id = 'delta'
    else:
        telescope_id = '{0:.1f}m_{1:}'.format(element_size, element_shape)

    if phased_array:
        telescope_id = telescope_id + '_array'
telescope_str = telescope_id+'_'

if (telescope['shape'] == 'dipole') or (telescope['shape'] == 'delta'):
    A_eff = (0.5*wavelength)**2
    if (telescope_id == 'mwa') or phased_array:
        A_eff *= 16
if telescope['shape'] == 'dish':
    A_eff = NP.pi * (0.5*element_size)**2

fhd_obsid = [1061309344, 1061316544]

pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
pointing_info_from_file = NP.loadtxt(pointing_file, comments='#', usecols=(1,2,3), delimiter=',')
obs_id = NP.loadtxt(pointing_file, comments='#', usecols=(0,), delimiter=',', dtype=str)
if (telescope_id == 'mwa') or (phased_array):
    delays_str = NP.loadtxt(pointing_file, comments='#', usecols=(4,), delimiter=',', dtype=str)
    delays_list = [NP.fromstring(delaystr, dtype=float, sep=';', count=-1) for delaystr in delays_str]
    delay_settings = NP.asarray(delays_list)
    delay_settings *= 435e-12
    delays = NP.copy(delay_settings)

pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'

h = 0.7   # Hubble constant coefficient
cosmodel100 = CP.FlatLambdaCDM(H0=100.0, Om0=0.27)  # Using H0 = 100 km/s/Mpc
cosmodel = CP.FlatLambdaCDM(H0=h*100.0, Om0=0.27)  # Using H0 = h * 100 km/s/Mpc

dr_z = (FCNST.c/1e3) * bw * (1+redshift)**2 / CNST.rest_freq_HI / cosmodel100.H0.value / cosmodel100.efunc(redshift)   # in Mpc/h
r_z = cosmodel100.comoving_transverse_distance(redshift).value   # in Mpc/h

volfactor1 = A_eff / wavelength**2 / bw
volfactor2 = r_z**2 * dr_z / bw

Jy2K = wavelength**2 * CNST.Jy / (2*FCNST.k)
mJy2mK = NP.copy(Jy2K)
Jy2mK = 1e3 * Jy2K

mK2Jy = 1/Jy2mK
mK2mJy = 1/mJy2mK
K2Jy = 1/Jy2K

dspec_min = None
dspec_max = None

def kprll(eta, z):
    return 2 * NP.pi * eta * cosmodel100.H0.value * CNST.rest_freq_HI * cosmodel100.efunc(z) / FCNST.c / (1+z)**2 * 1e3

def kperp(u, z):
    return 2 * NP.pi * u / cosmodel100.comoving_transverse_distance(z).value

##########################################

if plot_02:

    # infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
    # hdulist = fits.open(infile)
    # n_snaps = hdulist[0].header['n_acc']
    # lst = hdulist['POINTING AND PHASE CENTER INFO'].data['LST']
    # hdulist.close()
    
    backdrop_xsize = 100
    xmin = -180.0
    xmax = 180.0
    ymin = -90.0
    ymax = 90.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmax, xmin, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
    xvect = xgrid.ravel()
    yvect = ygrid.ravel()

    pinfo = []
    pb_snapshots = []
    pbx_MWA_snapshots = []
    pby_MWA_snapshots = []

    for i in xrange(n_snaps):
        havect = lst[i] - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        dircos = GEOM.altaz2dircos(altaz, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
        az = altaz[:,1] + 0.0
        az[az > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
        roi_sector_altaz = NP.asarray(NP.where(NP.logical_or(NP.logical_and(az[roi_altaz] >= -0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < -0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors), NP.logical_and(az[roi_altaz] >= 180.0 - 0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < 180.0 - 0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors)))).ravel()

        pinfo += [{}]
        if (telescope_id == 'mwa') or (phased_array):
            pinfo[i]['delays'] = delays[obs_id==str(fhd_obsid[i]),:].ravel()
            pinfo[i]['delayerr'] = delayerr
            pinfo[i]['gainerr'] = gainerr
            pinfo[i]['nrand'] = nrand
        else:
            p_altaz = pointing_info_from_file[obs_id==str(fhd_obsid[i]),:2].reshape(1,-1)
            pinfo[i]['pointing_coords'] = 'altaz'
            pinfo[i]['pointing_center'] = p_altaz

        pb = NP.empty(xvect.size)
        pb.fill(NP.nan)
        pbx_MWA_vect = NP.empty(xvect.size)
        pbx_MWA_vect.fill(NP.nan)
        pby_MWA_vect = NP.empty(xvect.size)
        pby_MWA_vect.fill(NP.nan)
    
        pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_info=pinfo[i], half_wave_dipole_approx=True)
        if (telescope_id == 'mwa') or (phased_array):
            pbx_MWA, pby_MWA = MWAPB.MWA_Tile_analytic(NP.radians(90.0-altaz[roi_altaz,0]).reshape(-1,1), NP.radians(altaz[roi_altaz,1]).reshape(-1,1), freq=185e6, delays=pinfo[i]['delays']/435e-12, power=True)
            # pbx_MWA, pby_MWA = MWAPB.MWA_Tile_advanced(NP.radians(90.0-altaz[roi_altaz,0]).reshape(-1,1), NP.radians(altaz[roi_altaz,1]).reshape(-1,1), freq=185e6, delays=pinfo[i]['delays']/435e-12, power=True)
            
            pbx_MWA_vect[roi_altaz] = pbx_MWA.ravel()
            pby_MWA_vect[roi_altaz] = pby_MWA.ravel()
    
        pb_snapshots += [pb]
        pbx_MWA_snapshots += [pbx_MWA_vect]
        pby_MWA_snapshots += [pby_MWA_vect]

    col_descriptor_str = ['off-zenith', 'zenith']
    row_descriptor_str = ['Model', 'MWA']
    fig, axs = PLT.subplots(ncols=n_snaps, nrows=2, sharex=True, sharey=True, figsize=(9,5))
    for i in range(2):
        for j in xrange(n_snaps):
            if i == 0:
                pbsky = axs[i,j].imshow(pb_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CM.jet)
            else:
                pbsky = axs[i,j].imshow(pbx_MWA_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CM.jet)                
            axs[i,j].set_xlim(xvect.max(), xvect.min())
            axs[i,j].set_ylim(yvect.min(), yvect.max())
            axs[i,j].grid(True, which='both')
            axs[i,j].set_aspect('auto')
            axs[i,j].tick_params(which='major', length=12, labelsize=12)
            axs[i,j].tick_params(which='minor', length=6)
            axs[i,j].locator_params(axis='x', nbins=5)
            axs[i,j].text(0.5, 0.9, row_descriptor_str[i]+' '+col_descriptor_str[j], transform=axs[i,j].transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    
    cbax = fig.add_axes([0.9, 0.122, 0.02, 0.84])
    cbar = fig.colorbar(pbsky, cax=cbax, orientation='vertical')

    fig.subplots_adjust(hspace=0,wspace=0)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)

    # PLT.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.subplots_adjust(top=0.98)

    PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'.eps', bbox_inches=0)




##########################################    

if plot_05 or plot_06 or plot_07 or plot_09 or plot_16:

    # 05) Plot FHD data and simulations on baselines by orientation and all combined

    fhd_obsid = [1061309344, 1061316544]

    infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)
    asm_CLEAN_infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
    if use_alt_spindex:
        alt_asm_CLEAN_infile = '/data3/t_nithyanandan/'+project_dir+'/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(alt_spindex_rms)+alt_spindex_seed_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

    ia = RI.InterferometerArray(None, None, None, init_file=infile+'.fits') 
    simdata_bl_orientation = NP.angle(ia.baselines[:,0] + 1j * ia.baselines[:,1], deg=True)
    simdata_neg_bl_orientation_ind = simdata_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    simdata_bl_orientation[simdata_neg_bl_orientation_ind] -= 180.0
    ia.baselines[simdata_neg_bl_orientation_ind,:] = -ia.baselines[simdata_neg_bl_orientation_ind,:]
    
    hdulist = fits.open(infile+'.fits')
    latitude = hdulist[0].header['latitude']
    pointing_coords = hdulist[0].header['pointing_coords']
    pointings_table = hdulist['POINTING AND PHASE CENTER INFO'].data
    lst = pointings_table['LST']
    n_snaps = lst.size
    hdulist.close()

    if pointing_coords == 'altaz':
        pointings_altaz = NP.hstack((pointings_table['pointing_latitude'].reshape(-1,1), pointings_table['pointing_longitude'].reshape(-1,1)))
        pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')
        pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    elif pointing_coords == 'radec':
        pointings_radec = NP.hstack((pointings_table['pointing_longitude'].reshape(-1,1), pointings_table['pointing_latitude'].reshape(-1,1)))
        pointings_hadec = NP.hstack(((lst-pointings_radec[:,0]).reshape(-1,1), pointings_radec[:,1].reshape(-1,1)))
        pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
        pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    elif pointing_coords == 'hadec':
        pointings_hadec = NP.hstack((pointings_table['pointing_longitude'].reshape(-1,1), pointings_table['pointing_latitude'].reshape(-1,1)))
        pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
        pointings_altaz = GEOM.hadec2altaz(pointings_hadec, latitude, units='degrees')
        pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')

    hdulist = fits.open(asm_CLEAN_infile+'.fits')
    clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    clean_lags_orig = NP.copy(clean_lags)
    asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()
    
    asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
    asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
    asm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = asm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
    asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()
    
    asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis, axis=1),axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
    asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(asm_cc_skyvis_res, axis=1),axes=1) * asm_cc_skyvis.shape[1] * freq_resolution
    asm_cc_skyvis_lag = asm_cc_skyvis_lag + asm_ccres_sky
    
    asm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis, axis=1),axes=1) * asm_cc_vis.shape[1] * freq_resolution
    asm_ccres = NP.fft.fftshift(NP.fft.ifft(asm_cc_vis_res, axis=1),axes=1) * asm_cc_vis.shape[1] * freq_resolution
    asm_cc_vis_lag = asm_cc_vis_lag + asm_ccres
    
    asm_cc_skyvis_lag = DSP.downsampler(asm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    asm_cc_vis_lag = DSP.downsampler(asm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    
    if use_alt_spindex:
        alt_hdulist = fits.open(alt_asm_CLEAN_infile+'.fits')
        alt_asm_cc_skyvis = alt_hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * alt_hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
        alt_asm_cc_skyvis_res = alt_hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * alt_hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
        alt_asm_cc_vis = alt_hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * alt_hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
        alt_asm_cc_vis_res = alt_hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * alt_hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
        alt_hdulist.close()
        
        alt_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = alt_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        alt_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = alt_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        alt_asm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = alt_asm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
        alt_asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = alt_asm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        alt_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(alt_asm_cc_skyvis, axis=1),axes=1) * alt_asm_cc_skyvis.shape[1] * freq_resolution
        alt_asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(alt_asm_cc_skyvis_res, axis=1),axes=1) * alt_asm_cc_skyvis.shape[1] * freq_resolution
        alt_asm_cc_skyvis_lag = alt_asm_cc_skyvis_lag + alt_asm_ccres_sky
        
        alt_asm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(alt_asm_cc_vis, axis=1),axes=1) * alt_asm_cc_vis.shape[1] * freq_resolution
        alt_asm_ccres = NP.fft.fftshift(NP.fft.ifft(alt_asm_cc_vis_res, axis=1),axes=1) * alt_asm_cc_vis.shape[1] * freq_resolution
        alt_asm_cc_vis_lag = alt_asm_cc_vis_lag + alt_asm_ccres
        
        alt_asm_cc_skyvis_lag = DSP.downsampler(alt_asm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
        alt_asm_cc_vis_lag = DSP.downsampler(alt_asm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)

    clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
    clean_lags = clean_lags.ravel()

    vis_noise_lag = NP.copy(ia.vis_noise_lag)
    asm_cc_skyvis_lag = asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
    asm_cc_vis_lag = asm_cc_vis_lag[truncated_ref_bl_ind,:,:]
    vis_noise_lag = vis_noise_lag[truncated_ref_bl_ind,:,:]

    delaymat = DLY.delay_envelope(ia.baselines[truncated_ref_bl_ind,:], pc, units='mks')
    min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
    max_delay = delaymat[0,:,0]-delaymat[0,:,1]
    clags = clean_lags.reshape(1,-1)
    min_delay = min_delay.reshape(-1,1)
    max_delay = max_delay.reshape(-1,1)
    thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
    thermal_noise_window = NP.repeat(thermal_noise_window, ia.baselines[truncated_ref_bl_ind,:].shape[0], axis=0)
    EoR_window = NP.logical_or(clags > max_delay+1/bw, clags < min_delay-1/bw)
    strict_EoR_window = NP.logical_and(EoR_window, NP.abs(clags) < 1/coarse_channel_resolution)
    wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
    non_wedge_window = NP.logical_not(wedge_window)
    # vis_rms_lag = OPS.rms(asm_cc_vis_lag.reshape(-1,n_snaps), mask=NP.logical_not(NP.repeat(thermal_noise_window.reshape(-1,1), n_snaps, axis=1)), axis=0)
    # vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
    # T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,-1)) * CNST.Jy
    # vis_rms_lag_theory = OPS.rms(vis_noise_lag.reshape(-1,n_snaps), mask=NP.logical_not(NP.repeat(EoR_window.reshape(-1,1), n_snaps, axis=1)), axis=0)
    # vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
    # T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff) * NP.mean(ia.eff_Q) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,-1)) * CNST.Jy
    vis_rms_lag = OPS.rms(asm_cc_vis_lag, mask=NP.logical_not(NP.repeat(thermal_noise_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
    vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
    T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
    vis_rms_lag_theory = OPS.rms(vis_noise_lag, mask=NP.logical_not(NP.repeat(EoR_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
    vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
    T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
    
    if max_abs_delay is not None:
        small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
        clean_lags = clean_lags[small_delays_ind]
        asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
        asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]
        if use_alt_spindex:
            alt_asm_cc_vis_lag = alt_asm_cc_vis_lag[:,small_delays_ind,:]
            alt_asm_cc_skyvis_lag = alt_asm_cc_skyvis_lag[:,small_delays_ind,:]
            
    ## Read in FHD data and other required information
    
    pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
    pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(1,2,3), delimiter=',')
    obs_id = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(0,), delimiter=',', dtype=str)
    obsfile_lst = 15.0 * pointing_info_from_file[:,2]
    obsfile_pointings_altaz = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
    obsfile_pointings_dircos = GEOM.altaz2dircos(obsfile_pointings_altaz, units='degrees')
    obsfile_pointings_hadec = GEOM.altaz2hadec(obsfile_pointings_altaz, latitude, units='degrees')
    
    common_bl_ind_in_ref_snapshots = []

    fhd_info = {}
    for j in range(len(fhd_obsid)):
        fhd_infile = '/data3/t_nithyanandan/project_MWA/fhd_delay_spectrum_{0:0d}_reformatted.npz'.format(fhd_obsid[j])
        fhd_data = NP.load(fhd_infile)
        fhd_vis_lag_noisy = fhd_data['fhd_vis_lag_noisy']
        fhd_C = fhd_data['fhd_C']
        valid_ind = NP.logical_and(NP.abs(NP.sum(fhd_vis_lag_noisy[:,:,0],axis=1))!=0.0, NP.abs(NP.sum(fhd_C[:,:,0],axis=1))!=0.0)
        fhd_C = fhd_C[valid_ind,:,:]
        fhd_vis_lag_noisy = fhd_vis_lag_noisy[valid_ind,:,:]
        fhd_delays = fhd_data['fhd_delays']
        fhdfile_bl_id = fhd_data['fhd_bl_id'][valid_ind]
        fhdfile_bl_length = fhd_data['fhd_bl_length'][valid_ind]
        common_bl_id = NP.intersect1d(truncated_ref_bl_id, fhdfile_bl_id, assume_unique=True)
        common_bl_ind_in_ref = NP.in1d(truncated_ref_bl_id, common_bl_id, assume_unique=True)
        common_bl_ind_in_fhd = NP.in1d(fhdfile_bl_id, common_bl_id, assume_unique=True)
        fhd_bl_id = fhdfile_bl_id[common_bl_ind_in_fhd]
        fhd_bl_length = fhdfile_bl_length[common_bl_ind_in_fhd]
        fhd_k_perp = 2 * NP.pi * fhd_bl_length / (FCNST.c/freq) / cosmodel100.comoving_transverse_distance(z=redshift).value
        fhd_bl = truncated_ref_bl[common_bl_ind_in_ref, :]
        fhd_bl_orientation = truncated_ref_bl_orientation[common_bl_ind_in_ref]
        common_bl_ind_in_ref_snapshots += [common_bl_ind_in_ref]

        fhd_neg_bl_orientation_ind = fhd_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
        fhd_bl_orientation[fhd_neg_bl_orientation_ind] -= 180.0
        fhd_bl[fhd_neg_bl_orientation_ind,:] = -fhd_bl[fhd_neg_bl_orientation_ind,:]
    
        fhd_C = fhd_C[common_bl_ind_in_fhd,:,:]
        fhd_vis_lag_noisy = fhd_vis_lag_noisy[common_bl_ind_in_fhd,:,:]*2.78*nchan*freq_resolution/fhd_C
        fhd_obsid_pointing_dircos = obsfile_pointings_dircos[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
        fhd_obsid_pointing_altaz = obsfile_pointings_altaz[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
        fhd_obsid_pointing_hadec = obsfile_pointings_hadec[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
        fhd_lst = NP.asscalar(obsfile_lst[obs_id==str(fhd_obsid[j])])
        fhd_obsid_pointing_radec = NP.copy(fhd_obsid_pointing_hadec)
        fhd_obsid_pointing_radec[0,0] = fhd_lst - fhd_obsid_pointing_hadec[0,0]
    
        fhd_delaymat = DLY.delay_envelope(fhd_bl, pc, units='mks')
    
        fhd_min_delay = -fhd_delaymat[0,:,1]-fhd_delaymat[0,:,0]
        fhd_max_delay = fhd_delaymat[0,:,0]-fhd_delaymat[0,:,1]
        fhd_min_delay = fhd_min_delay.reshape(-1,1)
        fhd_max_delay = fhd_max_delay.reshape(-1,1)
    
        fhd_thermal_noise_window = NP.abs(fhd_delays) >= max_abs_delay*1e-6
        fhd_thermal_noise_window = fhd_thermal_noise_window.reshape(1,-1)
        fhd_thermal_noise_window = NP.repeat(fhd_thermal_noise_window, fhd_bl.shape[0], axis=0)
        fhd_EoR_window = NP.logical_or(fhd_delays > fhd_max_delay+1/bw, fhd_delays < fhd_min_delay-1/bw)
        fhd_wedge_window = NP.logical_and(fhd_delays <= fhd_max_delay, fhd_delays >= fhd_min_delay)
        fhd_non_wedge_window = NP.logical_not(fhd_wedge_window)
        fhd_vis_rms_lag = OPS.rms(fhd_vis_lag_noisy[:,:,0], mask=NP.logical_not(fhd_thermal_noise_window), axis=1)
        fhd_vis_rms_freq = NP.abs(fhd_vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
    
        if max_abs_delay is not None:
            small_delays_ind = NP.abs(fhd_delays) <= max_abs_delay * 1e-6
            fhd_delays = fhd_delays[small_delays_ind]
            fhd_vis_lag_noisy = fhd_vis_lag_noisy[:,small_delays_ind,:]
        fhd_k_prll = 2 * NP.pi * fhd_delays * cosmodel100.H0.value * CNST.rest_freq_HI * cosmodel100.efunc(z=redshift) / FCNST.c / (1+redshift)**2 * 1e3
    
        fhd_info[fhd_obsid[j]] = {}
        fhd_info[fhd_obsid[j]]['bl_id'] = fhd_bl_id
        fhd_info[fhd_obsid[j]]['bl'] = fhd_bl
        fhd_info[fhd_obsid[j]]['bl_length'] = fhd_bl_length
        fhd_info[fhd_obsid[j]]['k_perp'] = fhd_k_perp
        fhd_info[fhd_obsid[j]]['bl_orientation'] = fhd_bl_orientation
        fhd_info[fhd_obsid[j]]['delays'] = fhd_delays
        fhd_info[fhd_obsid[j]]['k_prll'] = fhd_k_prll
        fhd_info[fhd_obsid[j]]['C'] = fhd_C
        fhd_info[fhd_obsid[j]]['vis_lag_noisy'] = fhd_vis_lag_noisy
        fhd_info[fhd_obsid[j]]['lst'] = fhd_lst
        fhd_info[fhd_obsid[j]]['pointing_radec'] = fhd_obsid_pointing_radec
        fhd_info[fhd_obsid[j]]['pointing_hadec'] = fhd_obsid_pointing_hadec
        fhd_info[fhd_obsid[j]]['pointing_altaz'] = fhd_obsid_pointing_altaz
        fhd_info[fhd_obsid[j]]['pointing_dircos'] = fhd_obsid_pointing_dircos
        fhd_info[fhd_obsid[j]]['min_delays'] = fhd_min_delay
        fhd_info[fhd_obsid[j]]['max_delays'] = fhd_max_delay
        fhd_info[fhd_obsid[j]]['rms_lag'] = fhd_vis_rms_lag
        fhd_info[fhd_obsid[j]]['rms_freq'] = fhd_vis_rms_freq

    if (dspec_min is None) or (dspec_max is None):
        dspec_min = min(min([NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j]).min() for j in xrange(n_snaps)]), min([NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy']).min() for j in xrange(len(fhd_obsid))]))
        dspec_max = max(max([NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j]).max() for j in xrange(n_snaps)]), max([NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy']).max() for j in xrange(len(fhd_obsid))]))
        # dspec_min = dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2
        # dspec_max = dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2

    cardinal_blo = 180.0 / n_bins_baseline_orientation * (NP.arange(n_bins_baseline_orientation)-1).reshape(-1,1)
    cardinal_bll = 100.0
    cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

    bl_orientation = simdata_bl_orientation[truncated_ref_bl_ind]
    blo = [bl_orientation[common_bl_ind_in_ref_snapshots[j]] for j in xrange(n_snaps)]

    small_delays_EoR_window = EoR_window.T
    small_delays_strict_EoR_window = strict_EoR_window.T
    small_delays_wedge_window = wedge_window.T
    if max_abs_delay is not None:
        small_delays_EoR_window = small_delays_EoR_window[small_delays_ind,:]
        small_delays_strict_EoR_window = small_delays_strict_EoR_window[small_delays_ind,:]
        small_delays_wedge_window = small_delays_wedge_window[small_delays_ind,:]

    small_delays_non_wedge_window = NP.logical_not(small_delays_wedge_window)
    
    data_sim_ratio = []
    data_sim_difference_fraction = []

    if use_alt_spindex:
        alt_data_sim_ratio = []
        alt_data_sim_difference_fraction = []

    relevant_EoR_window = []
    relevant_wedge_window = []
    relevant_non_wedge_window = []

    for j in xrange(n_snaps):
        data_sim_ratio += [NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)]
        data_sim_difference_fraction += [(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)]
        if use_alt_spindex:
            alt_data_sim_ratio += [NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(alt_asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)]
            alt_data_sim_difference_fraction += [(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) - NP.abs(alt_asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)]

        relevant_EoR_window += [small_delays_EoR_window[:,common_bl_ind_in_ref_snapshots[j]]]
        relevant_wedge_window += [small_delays_wedge_window[:,common_bl_ind_in_ref_snapshots[j]]]
        relevant_non_wedge_window += [small_delays_non_wedge_window[:,common_bl_ind_in_ref_snapshots[j]]]

    descriptor_str = ['off-zenith', 'zenith']

    if plot_05:

        # Plot FHD delay spectra

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*clean_lags, NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:-1,:-1,0].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
            horizonb = axs[j].plot(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            axs[j].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[j].set_aspect('auto')
            axs[j].text(0.5, 0.9, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')

        for j in xrange(n_snaps):
            axs_kprll = axs[j].twinx()
            axs_kprll.set_yticks(kprll(axs[j].get_yticks()*1e-6, redshift))
            axs_kprll.set_ylim(kprll(NP.asarray(axs[j].get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            axs_kprll.yaxis.set_major_formatter(yformatter)
            if j == 0:
                axs_kperp = axs[j].twiny()
                axs_kperp.set_xticks(kperp(axs[j].get_xticks()*freq/FCNST.c, redshift))
                axs_kperp.set_xlim(kperp(NP.asarray(axs[j].get_xlim())*freq/FCNST.c, redshift))
                xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
                axs_kperp.xaxis.set_major_formatter(xformatter)

        fig.subplots_adjust(hspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=30)
        big_ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=30)

        big_axr = big_ax.twinx()
        big_axr.set_axis_bgcolor('none')
        big_axr.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axr.set_xticks([])
        big_axr.set_yticks([])
        big_axr.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=40)

        big_axt = big_ax.twiny()
        big_axt.set_axis_bgcolor('none')
        big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axt.set_xticks([])
        big_axt.set_yticks([])
        big_axt.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)

        cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
        cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        cbax.set_xlabel('Jy Hz', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)
    
        PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)
    
        # Plot simulated delay spectra

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j][:-1],:-1,j].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
            horizonb = axs[j].plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            axs[j].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[j].set_aspect('auto')
            axs[j].text(0.5, 0.9, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')

        for j in xrange(n_snaps):
            axs_kprll = axs[j].twinx()
            axs_kprll.set_yticks(kprll(axs[j].get_yticks()*1e-6, redshift))
            axs_kprll.set_ylim(kprll(NP.asarray(axs[j].get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            axs_kprll.yaxis.set_major_formatter(yformatter)
            if j == 0:
                axs_kperp = axs[j].twiny()
                axs_kperp.set_xticks(kperp(axs[j].get_xticks()*freq/FCNST.c, redshift))
                axs_kperp.set_xlim(kperp(NP.asarray(axs[j].get_xlim())*freq/FCNST.c, redshift))
                xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
                axs_kperp.xaxis.set_major_formatter(xformatter)

        fig.subplots_adjust(hspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=30)
        big_ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=20)

        big_axr = big_ax.twinx()
        big_axr.set_axis_bgcolor('none')
        big_axr.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axr.set_xticks([])
        big_axr.set_yticks([])
        big_axr.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=40)

        big_axt = big_ax.twiny()
        big_axt.set_axis_bgcolor('none')
        big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axt.set_xticks([])
        big_axt.set_yticks([])
        big_axt.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)

        cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
        cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        cbax.set_xlabel('Jy Hz', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)

        PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_sim_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_sim_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    if plot_06:

        # 06) Plot FHD data to simulation ratio on all baselines combined

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.log10(data_sim_ratio[j][:-1,:-1]), vmin=-1.0, vmax=1.0)

            horizonb = axs[j].plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            axs[j].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[j].set_aspect('auto')
            axs[j].text(0.5, 0.9, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    
        for j in xrange(n_snaps):
            axs_kprll = axs[j].twinx()
            axs_kprll.set_yticks(kprll(axs[j].get_yticks()*1e-6, redshift))
            axs_kprll.set_ylim(kprll(NP.asarray(axs[j].get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            axs_kprll.yaxis.set_major_formatter(yformatter)
            if j == 0:
                axs_kperp = axs[j].twiny()
                axs_kperp.set_xticks(kperp(axs[j].get_xticks()*freq/FCNST.c, redshift))
                axs_kperp.set_xlim(kperp(NP.asarray(axs[j].get_xlim())*freq/FCNST.c, redshift))
                xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
                axs_kperp.xaxis.set_major_formatter(xformatter)

        fig.subplots_adjust(hspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=30)
        big_ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=30)

        big_axr = big_ax.twinx()
        big_axr.set_axis_bgcolor('none')
        big_axr.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axr.set_xticks([])
        big_axr.set_yticks([])
        big_axr.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=40)

        big_axt = big_ax.twiny()
        big_axt.set_axis_bgcolor('none')
        big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axt.set_xticks([])
        big_axt.set_yticks([])
        big_axt.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)

        cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
        cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)

        PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_sim_data_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_sim_data_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    if plot_07:

        # 07) Plot FHD data to simulation ratio on baselines binned by orientation

        orientation_str = ['South-East', 'East', 'North-East', 'North']

        for j in xrange(n_snaps):
            bloh, bloe, blon, blori = OPS.binned_statistic(blo[j], statistic='count', bins=n_bins_baseline_orientation, range=[(0.5*180.0/n_bins_baseline_orientation-90.0, 0.5*180.0/n_bins_baseline_orientation+90.0)])

            k = range(len(bloh))
            fig, axs = PLT.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8))
            for i,ii in enumerate(list(reversed(k))):
                blind = blori[blori[ii]:blori[ii+1]]
                bll = truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]][blind]
                sortind = NP.argsort(bll)
                rind = blind[sortind]

                imdspec_ratio = axs[i/2,i%2].pcolorfast(bll[sortind], 1e6*clean_lags, NP.log10(data_sim_ratio[j][:-1,rind[:-1]]), vmin=-1.0, vmax=1.0)
                horizonb = axs[i/2,i%2].plot(bll[sortind], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]][rind].ravel(), color='black', ls='-', lw=1.5)
                horizont = axs[i/2,i%2].plot(bll[sortind], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]][rind].ravel(), color='black', ls='-', lw=1.5)
                axs[i/2,i%2].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
                axs[i/2,i%2].text(0.5, 0.9, orientation_str[ii], transform=axs[i/2,i%2].transAxes, fontsize=14, weight='semibold', ha='center', color='black')
                axs[i/2,i%2].set_aspect('auto')                
                
            for row in range(2):
                axs_kprll = axs[row,1].twinx()
                axs_kprll.set_yticks(kprll(axs[row,1].get_yticks()*1e-6, redshift))
                axs_kprll.set_ylim(kprll(NP.asarray(axs[row,1].get_ylim())*1e-6, redshift))
                yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
                axs_kprll.yaxis.set_major_formatter(yformatter)
                if row == 0:
                    for col in range(2):
                        axs_kperp = axs[row,col].twiny()
                        axs_kperp.set_xticks(kperp(axs[row,col].get_xticks()*freq/FCNST.c, redshift))
                        axs_kperp.set_xlim(kperp(NP.asarray(axs[row,col].get_xlim())*freq/FCNST.c, redshift))
                        xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
                        axs_kperp.xaxis.set_major_formatter(xformatter)

            fig.subplots_adjust(wspace=0, hspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=30)
            big_ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium', labelpad=20)
    
            big_axr = big_ax.twinx()
            big_axr.set_axis_bgcolor('none')
            big_axr.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_axr.set_xticks([])
            big_axr.set_yticks([])
            big_axr.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=40)
    
            big_axt = big_ax.twiny()
            big_axt.set_axis_bgcolor('none')
            big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_axt.set_xticks([])
            big_axt.set_yticks([])
            big_axt.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)

            cbax = fig.add_axes([0.92, 0.125, 0.02, 0.74])
            cbar = fig.colorbar(imdspec_ratio, cax=cbax, orientation='vertical')
            
            # PLT.tight_layout()      
            fig.subplots_adjust(right=0.81)
            fig.subplots_adjust(top=0.88)

            PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_binned_baseline_CLEAN_noisy_PS_fhd_sim_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:0d}'.format(oversampling_factor, j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/'+project_dir+'/figures/'+telescope_str+'multi_binned_baseline_CLEAN_noisy_PS_fhd_sim_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:0d}'.format(oversampling_factor, j)+'.eps', bbox_inches=0)



