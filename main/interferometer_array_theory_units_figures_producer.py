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
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import lookup_operations as LKP
import ipdb as PDB

# 01) Plot pointings information

# 02) Plot power patterns for snapshots

# 03) Plot foreground models with power pattern contours for snapshots

# 04) Plot delay maps on sky for baselines of different orientations

# 05) Plot FHD data and simulations on all baselines combined

# 06) Plot FHD data to simulation ratio on all baselines combined

# 07) Plot uncertainties in FHD data to simulation ratio on all baselines combined

# 08) Plot ratio of differences between FHD data and simulation to expected error on all baselines combined

# 09) Plot histogram of fractional differences between FHD data and simulation 

# 10) Plot noiseless delay spectra from simulations for diffuse, compact and all-sky models

# 11) Plot noiseless delay spectra for all sky models broken down by baseline orientation

# 12) Plot delay spectra on northward and eastward baselines along with delay maps and sky models

# 13) Plot EoR window foreground contamination when baselines are selectively removed

# 14) Plot delay spectra before and after baselines are selectively removed

# 15) Plot Fourier space

# 16) Plot average thermal noise in simulations and data as a function of baseline length

# 17) Plot delay spectra of the MWA tile power pattern using a uniform sky model

# 18) Plot delay spectra of the all-sky model with dipole, MWA tile, and HERA dish antenna shapes

# 19) Plot delay spectrum of uniform sky model with a uniform power pattern

plot_01 = False
plot_02 = False
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
plot_19 = True

# PLT.ioff()
PLT.ion()

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

roifile = '/data3/t_nithyanandan/project_MWA/roi_info_'+telescope_str+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
roi = RI.ROI_parameters(init_file=roifile)
telescope = roi.telescope

if (telescope['shape'] == 'dipole') or (telescope['shape'] == 'delta'):
    A_eff = (0.5*wavelength)**2
    if (telescope_id == 'mwa') or phased_array:
        A_eff *= 16
if telescope['shape'] == 'dish':
    A_eff = NP.pi * (0.5*element_size)**2

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

if plot_01:
        
    # 01) Plot pointings information

    pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
    
    pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(1,2,3), delimiter=',')
    obs_id = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(0,), delimiter=',', dtype=str)
    if (telescope_id == 'mwa') or (phased_array):
        delays_str = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(4,), delimiter=',', dtype=str)
        delays_list = [NP.fromstring(delaystr, dtype=float, sep=';', count=-1) for delaystr in delays_str]
        delay_settings = NP.asarray(delays_list)
        delay_settings *= 435e-12
        delays = NP.copy(delay_settings)
        n_snaps = pointing_info_from_file.shape[0]
    pointing_info_from_file = pointing_info_from_file[:min(n_snaps, pointing_info_from_file.shape[0]),:]
    obs_id = obs_id[:min(n_snaps, pointing_info_from_file.shape[0])]
    if (telescope_id == 'mwa') or (phased_array):
        delays = delay_settings[:min(n_snaps, pointing_info_from_file.shape[0]),:]
    n_snaps = min(n_snaps, pointing_info_from_file.shape[0])
    pointings_altaz = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
    pointings_altaz_orig = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
    lst = 15.0 * pointing_info_from_file[:,2]
    lst_wrapped = lst + 0.0
    lst_wrapped[lst_wrapped > 180.0] = lst_wrapped[lst_wrapped > 180.0] - 360.0
    lst_edges = NP.concatenate((lst_wrapped, [lst_wrapped[-1]+lst_wrapped[-1]-lst_wrapped[-2]]))

    lst = 0.5*(lst_edges[1:]+lst_edges[:-1])
    t_snap = (lst_edges[1:]-lst_edges[:-1]) / 15.0 * 3.6e3

    pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
    pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')
    pointings_radec = NP.hstack(((lst-pointings_hadec[:,0]).reshape(-1,1), pointings_hadec[:,1].reshape(-1,1)))
    pointings_radec[:,0] = pointings_radec[:,0] % 360.0

    pointings_ha = pointings_hadec[:,0]
    pointings_ha[pointings_ha > 180.0] = pointings_ha[pointings_ha > 180.0] - 360.0

    pointings_ra = pointings_radec[:,0]
    pointings_ra[pointings_ra > 180.0] = pointings_ra[pointings_ra > 180.0] - 360.0

    pointings_dec = pointings_radec[:,1]

    infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
    hdulist = fits.open(infile)
    lst_select = hdulist['POINTING AND PHASE CENTER INFO'].data['LST']
    hdulist.close()
    lst_select[lst_select > 180.0] -= 360.0

    fig = PLT.figure(figsize=(6,6))
    ax1a = fig.add_subplot(111)
    ax1a.set_xlabel('Local Sidereal Time [hours]', fontsize=18, weight='medium')
    ax1a.set_ylabel('Longitude [degrees]', fontsize=18, weight='medium')
    ax1a.set_xlim((lst_wrapped.min()-1)/15.0, (lst_wrapped.max()-1)/15.0)
    ax1a.set_ylim(pointings_ha.min()-15.0, pointings_ha.max()+15.0)
    ax1a.plot(lst_wrapped/15.0, pointings_ha, 'k--', lw=2, label='HA')
    ax1a.plot(lst_wrapped/15.0, pointings_ra, 'k-', lw=2, label='RA')
    for i in xrange(lst_select.size):
        if i == 0:
            ax1a.axvline(x=lst_select[i]/15.0, color='gray', ls='-.', lw=2, label='Selected LST')
        else:
            ax1a.axvline(x=lst_select[i]/15.0, color='gray', ls='-.', lw=2)
    ax1a.tick_params(which='major', length=18, labelsize=12)
    ax1a.tick_params(which='minor', length=12, labelsize=12)
    # legend1a = ax1a.legend(loc='lower right')
    # legend1a.draw_frame(False)
    for axis in ['top','bottom','left','right']:
        ax1a.spines[axis].set_linewidth(2)
    xticklabels = PLT.getp(ax1a, 'xticklabels')
    yticklabels = PLT.getp(ax1a, 'yticklabels')
    PLT.setp(xticklabels, fontsize=15, weight='medium')
    PLT.setp(yticklabels, fontsize=15, weight='medium')    

    ax1b = ax1a.twinx()
    ax1b.set_ylabel('Declination [degrees]', fontsize=18, weight='medium')
    ax1b.set_ylim(pointings_dec.min()-5.0, pointings_dec.max()+5.0)
    ax1b.plot(lst_wrapped/15.0, pointings_dec, 'k:', lw=2, label='Dec')
    ax1b.tick_params(which='major', length=12, labelsize=12)
    # legend1b = ax1b.legend(loc='upper right')
    # legend1b.draw_frame(False)
    yticklabels = PLT.getp(ax1b, 'yticklabels')
    PLT.setp(yticklabels, fontsize=15, weight='medium')    

    decline = PLT.Line2D(range(1), range(0), color='k', ls=':', lw=2)
    haline = PLT.Line2D(range(1), range(0), color='k', ls='--', lw=2)
    raline = PLT.Line2D(range(1), range(0), color='k', ls='-', lw=2)
    lstline = PLT.Line2D(range(1), range(0), color='gray', ls='-.', lw=2)
    legend = PLT.legend((haline, raline, decline, lstline), ('HA', 'RA', 'Dec', 'Chosen LST'), loc='lower right', frameon=False)

    fig.subplots_adjust(right=0.85)

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+obs_mode+'_pointings.eps', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+obs_mode+'_pointings.png', bbox_inches=0)

#############################################################################

if plot_02 or plot_03 or plot_04 or plot_12:
    
    infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
    hdulist = fits.open(infile)
    n_snaps = hdulist[0].header['n_acc']
    lst = hdulist['POINTING AND PHASE CENTER INFO'].data['LST']
    hdulist.close()
    
    backdrop_xsize = 100
    xmin = -180.0
    xmax = 180.0
    ymin = -90.0
    ymax = 90.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmax, xmin, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
    xvect = xgrid.ravel()
    yvect = ygrid.ravel()

    pb_snapshots = []
    pbx_MWA_snapshots = []
    pby_MWA_snapshots = []

    src_ind_csm_snapshots = []
    src_ind_gsm_snapshots = []
    dsm_snapshots = []

    if plot_03 or plot_12:

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

        if spindex_seed is None:
            spindex_SUMSS = -0.83 + spindex_rms * NP.random.randn(fint.size)
        else:
            NP.random.seed(spindex_seed)
            spindex_SUMSS = -0.83 + spindex_rms * NP.random.randn(fint.size)

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
        fluxes = NP.copy(fint)

        nvss_file = '/data3/t_nithyanandan/project_MWA/foregrounds/NVSS_catalog.fits'
        freq_NVSS = 1.4 # in GHz
        hdulist = fits.open(nvss_file)
        ra_deg_NVSS = hdulist[1].data['RA(2000)']
        dec_deg_NVSS = hdulist[1].data['DEC(2000)']
        nvss_fpeak = hdulist[1].data['PEAK INT']
        nvss_majax = hdulist[1].data['MAJOR AX']
        nvss_minax = hdulist[1].data['MINOR AX']
        hdulist.close()
    
        if spindex_seed is None:
            spindex_NVSS = -0.83 + spindex_rms * NP.random.randn(nvss_fpeak.size)
        else:
            NP.random.seed(2*spindex_seed)
            spindex_NVSS = -0.83 + spindex_rms * NP.random.randn(nvss_fpeak.size)

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
        ra_deg_wrapped = ra_deg.ravel()
        ra_deg_wrapped[ra_deg_wrapped > 180.0] -= 360.0
    
        csmctlg = CTLG.Catalog(catlabel, freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, spectral_index=spindex, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

        dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq/1e6,nside)
        hdulist = fits.open(dsm_file)
        dsm_table = hdulist[1].data
        dsm_ra_deg = dsm_table['RA']
        dsm_dec_deg = dsm_table['DEC']
        dsm_temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
        dsm = HP.cartview(dsm_temperatures.ravel(), coord=['G','E'], rot=[0,0,0], xsize=backdrop_xsize, return_projected_map=True)
        dsm = dsm.ravel()

    for i in xrange(n_snaps):
        havect = lst[i] - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        dircos = GEOM.altaz2dircos(altaz, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
        az = altaz[:,1] + 0.0
        az[az > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
        roi_sector_altaz = NP.asarray(NP.where(NP.logical_or(NP.logical_and(az[roi_altaz] >= -0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < -0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors), NP.logical_and(az[roi_altaz] >= 180.0 - 0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < 180.0 - 0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors)))).ravel()
        pb = NP.empty(xvect.size)
        pb.fill(NP.nan)
        pbx_MWA_vect = NP.empty(xvect.size)
        pbx_MWA_vect.fill(NP.nan)
        pby_MWA_vect = NP.empty(xvect.size)
        pby_MWA_vect.fill(NP.nan)
    
        pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_info=roi.pinfo[i])
        if (telescope_id == 'mwa') or (phased_array):
            pbx_MWA, pby_MWA = MWAPB.MWA_Tile_advanced(NP.radians(90.0-altaz[roi_altaz,0]).reshape(-1,1), NP.radians(altaz[roi_altaz,1]).reshape(-1,1), freq=185e6, delays=roi.pinfo[i]['delays']/435e-12)
            pbx_MWA_vect[roi_altaz] = pbx_MWA.ravel()
            pby_MWA_vect[roi_altaz] = pby_MWA.ravel()
    
        pb_snapshots += [pb]
        pbx_MWA_snapshots += [pbx_MWA_vect]
        pby_MWA_snapshots += [pby_MWA_vect]

        if plot_03 or plot_12:
            csm_hadec = NP.hstack(((lst[i]-csmctlg.location[:,0]).reshape(-1,1), csmctlg.location[:,1].reshape(-1,1)))
            csm_altaz = GEOM.hadec2altaz(csm_hadec, latitude, units='degrees')
            roi_csm_altaz = NP.asarray(NP.where(csm_altaz[:,0] >= 0.0)).ravel()
            src_ind_csm_snapshots += [roi_csm_altaz]

            dsm_snapshot = NP.empty(xvect.size)
            dsm_snapshot.fill(NP.nan)
            dsm_snapshot[roi_altaz] = dsm[roi_altaz]
            dsm_snapshots += [dsm_snapshot]

    if plot_02:

        descriptor_str = ['off-zenith', 'zenith']
        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            pbsky = axs[j].imshow(pb_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CM.jet)
            axs[j].set_xlim(xvect.max(), xvect.min())
            axs[j].set_ylim(yvect.min(), yvect.max())
            axs[j].grid(True, which='both')
            axs[j].set_aspect('auto')
            axs[j].tick_params(which='major', length=12, labelsize=12)
            axs[j].tick_params(which='minor', length=6)
            axs[j].locator_params(axis='x', nbins=5)
            axs[j].text(0.5, 0.9, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='black')
        
        cbax = fig.add_axes([0.9, 0.122, 0.02, 0.84])
        cbar = fig.colorbar(pbsky, cax=cbax, orientation='vertical')

        fig.subplots_adjust(hspace=0)
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
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'.eps', bbox_inches=0)

        # Plot each snapshot separately

        for j in xrange(n_snaps):
            fig = PLT.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            pbsky = ax.imshow(pb_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-5, vmax=1.0), cmap=CM.jet)
            ax.set_xlim(xvect.max(), xvect.min())
            ax.set_ylim(yvect.min(), yvect.max())
            ax.grid(True, which='both')
            ax.set_aspect('auto')
            ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
            ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
            # ax.tick_params(which='major', length=12, labelsize=12)
            # ax.tick_params(which='minor', length=6)
            # ax.locator_params(axis='x', nbins=5)
            # ax.text(0.5, 0.9, descriptor_str[j], transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
        
            cbax = fig.add_axes([0.9, 0.15, 0.02, 0.81])
            cbar = fig.colorbar(pbsky, cax=cbax, orientation='vertical')

            # PLT.tight_layout()
            fig.subplots_adjust(right=0.89)
            fig.subplots_adjust(top=0.96)
            fig.subplots_adjust(bottom=0.15)
    
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'_snapshot_{0:1d}.png'.format(j), bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'_snapshot_{0:1d}.eps'.format(j), bbox_inches=0)

    if plot_03 or plot_12:

        csm_fluxes = csmctlg.flux_density * (freq/csmctlg.frequency)**csmctlg.spectral_index
        if plot_03:
            # 03) Plot foreground models with power pattern contours for snapshots
            
            descriptor_str = ['off-zenith', 'zenith']
            n_fg_ticks = 5
            fg_ticks = NP.round(NP.logspace(NP.log10(dsm.min()), NP.log10(dsm.max()), n_fg_ticks)).astype(NP.int)
    
            fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
            for j in xrange(n_snaps):
                dsmsky = axs[j].imshow(dsm_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=dsm.min(), vmax=dsm.max()), cmap=CM.jet)
                pbskyc = axs[j].contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[0.001953125, 0.0078125, 0.03125, 0.125, 0.5], colors='k', linewidths=1.5)
                axs[j].set_xlim(xvect.max(), xvect.min())
                axs[j].set_ylim(yvect.min(), yvect.max())
                axs[j].grid(True, which='both')
                axs[j].set_aspect('auto')
                axs[j].tick_params(which='major', length=12, labelsize=12)
                axs[j].tick_params(which='minor', length=6)
                axs[j].locator_params(axis='x', nbins=5)
                axs[j].text(0.5, 0.9, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='black')
            
            cbax = fig.add_axes([0.85, 0.125, 0.02, 0.84])
            cbar = fig.colorbar(dsmsky, cax=cbax, orientation='vertical')
            cbar.set_ticks(fg_ticks.tolist())
            cbar.set_ticklabels(fg_ticks.tolist())
            cbax.set_ylabel('Temperature [K]', labelpad=0, fontsize=14)
    
            fig.subplots_adjust(hspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
            big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)
    
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.85)
            fig.subplots_adjust(top=0.98)
    
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/dsm.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/dsm.eps', bbox_inches=0)
    
            n_fg_ticks = 5
            fg_ticks = NP.round(NP.logspace(NP.log10(csm_fluxes.min()), NP.log10(csm_fluxes.max()), n_fg_ticks)).astype(NP.int)
    
            fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
            for j in xrange(n_snaps):
                csmsky = axs[j].scatter(ra_deg_wrapped[src_ind_csm_snapshots[j]], dec_deg[src_ind_csm_snapshots[j]], c=csm_fluxes[src_ind_csm_snapshots[j]], norm=PLTC.LogNorm(vmin=csm_fluxes.min(), vmax=csm_fluxes.max()), cmap=CM.jet, edgecolor='none', s=20)
                pbskyc = axs[j].contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[0.001953125, 0.0078125, 0.03125, 0.125, 0.5], colors='k', linewidths=1.5)
                axs[j].set_xlim(xvect.max(), xvect.min())
                axs[j].set_ylim(yvect.min(), yvect.max())
                axs[j].grid(True, which='both')
                axs[j].set_aspect('auto')
                axs[j].tick_params(which='major', length=12, labelsize=12)
                axs[j].tick_params(which='minor', length=6)
                axs[j].locator_params(axis='x', nbins=5)
                axs[j].text(0.5, 0.9, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='black')
            
            cbax = fig.add_axes([0.88, 0.125, 0.02, 0.84])
            cbar = fig.colorbar(csmsky, cax=cbax, orientation='vertical')
            cbar.set_ticks(fg_ticks.tolist())
            cbar.set_ticklabels(fg_ticks.tolist())
            cbax.set_ylabel('Flux Density [Jy]', labelpad=0, fontsize=14)
    
            fig.subplots_adjust(hspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
            big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)
    
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.88)
            fig.subplots_adjust(top=0.98)
    
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/csm.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/csm.eps', bbox_inches=0)

        if plot_12:

            descriptor_str = ['Compact Emission', 'Diffuse Sky Model']
            n_fg_ticks = 5
            fg_ticks = NP.round(NP.logspace(NP.log10(dsm.min()), NP.log10(dsm.max()), n_fg_ticks)).astype(NP.int)
    
            for j in xrange(n_snaps):
                fig, axs = PLT.subplots(2, sharex=True, sharey=True, figsize=(6,6))
                csmsky = axs[0].scatter(ra_deg_wrapped[src_ind_csm_snapshots[j]], dec_deg[src_ind_csm_snapshots[j]], c=csm_fluxes[src_ind_csm_snapshots[j]], norm=PLTC.LogNorm(vmin=csm_fluxes.min(), vmax=csm_fluxes.max()), cmap=CM.jet, edgecolor='none', s=20)
                dsmsky = axs[1].imshow(dsm_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=dsm.min(), vmax=dsm.max()), cmap=CM.jet)

                for i in xrange(2):
                    pbskyc = axs[i].contour(xgrid[0,:], ygrid[:,0], pb_snapshots[j].reshape(-1,backdrop_xsize), levels=[0.001953125, 0.0078125, 0.03125, 0.125, 0.5], colors='k', linewidths=1.5)
                    axs[i].set_xlim(xvect.max(), xvect.min())
                    axs[i].set_ylim(yvect.min(), yvect.max())
                    axs[i].grid(True, which='both')
                    axs[i].set_aspect('auto')
                    axs[i].tick_params(which='major', length=12, labelsize=12)
                    axs[i].tick_params(which='minor', length=6)
                    axs[i].locator_params(axis='x', nbins=5)
                    axs[i].text(0.5, 0.9, descriptor_str[i], transform=axs[i].transAxes, fontsize=16, weight='semibold', ha='center', color='black')
            
                fg_ticks = NP.round(NP.logspace(NP.log10(dsm.min()), NP.log10(dsm.max()), n_fg_ticks)).astype(NP.int)
                cbaxbr = fig.add_axes([0.86, 0.11, 0.02, 0.4])
                cbarbr = fig.colorbar(dsmsky, cax=cbaxbr, orientation='vertical')
                cbarbr.set_ticks(fg_ticks.tolist())
                cbarbr.set_ticklabels(fg_ticks.tolist())
                cbaxbr.set_ylabel('K', labelpad=0, fontsize=14)

                fg_ticks = NP.round(NP.logspace(NP.log10(csm_fluxes.min()), NP.log10(csm_fluxes.max()), n_fg_ticks)).astype(NP.int)
                cbaxtr = fig.add_axes([0.86, 0.55, 0.02, 0.4])
                cbartr = fig.colorbar(csmsky, cax=cbaxtr, orientation='vertical')
                cbartr.set_ticks(fg_ticks.tolist())
                cbartr.set_ticklabels(fg_ticks.tolist())
                cbaxtr.set_ylabel('Jy', labelpad=0, fontsize=14)
    
                fig.subplots_adjust(hspace=0)
                big_ax = fig.add_subplot(111)
                big_ax.set_axis_bgcolor('none')
                big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
                big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)
        
                # PLT.tight_layout()
                fig.subplots_adjust(right=0.85)
                fig.subplots_adjust(top=0.98)
            
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/sky_model_snapshot_{0:0d}.png'.format(j), bbox_inches=0)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/sky_model_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)

    if plot_04 or plot_12:

        # 04) Plot delay maps on sky for baselines of different orientations

        cardinal_blo = 180.0 / n_bins_baseline_orientation * (NP.arange(n_bins_baseline_orientation)-1).reshape(-1,1)
        cardinal_bll = 100.0
        cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

        delay_map = NP.empty((n_bins_baseline_orientation, xvect.size, n_snaps))
        delay_map.fill(NP.nan)

        for i in xrange(n_snaps):
            havect = lst[i] - xvect
            altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
            dircos = GEOM.altaz2dircos(altaz, units='degrees')
            roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
            delay_map[:,roi_altaz,i] = (DLY.geometric_delay(cardinal_bl, altaz[roi_altaz,:], altaz=True, dircos=False, hadec=False, latitude=latitude)-DLY.geometric_delay(cardinal_bl, pc, altaz=False, dircos=True, hadec=False, latitude=latitude)).T

        # mindelay = NP.nanmin(delay_map)
        # maxdelay = NP.nanmax(delay_map)
        # norm_b = PLTC.Normalize(vmin=mindelay, vmax=maxdelay)

        # for i in xrange(n_snaps):
        #     fig = PLT.figure(figsize=(8,6))
        #     for j in xrange(n_bins_baseline_orientation):
        #         ax = fig.add_subplot(2,2,j+1)
        #         imdmap = ax.imshow(1e6 * delay_map[j,:,i].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), vmin=1e6*NP.nanmin(delay_map), vmax=1e6*NP.nanmax(delay_map))
        #     cbax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
        #     cbar = fig.colorbar(imdmap, cax=cbax, orientation='vertical')
        #     cbax.set_ylabel(r'$\times\,(|\mathbf{b}|/100)\,\mu$s', labelpad=-90, fontsize=18)
        #     PLT.tight_layout()
        #     fig.subplots_adjust(right=0.85)
        #     # fig.subplots_adjust(left=0.15)

        if plot_04:

            for i in xrange(n_snaps):
                fig, axs = PLT.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(12,6))
                for j in xrange(n_bins_baseline_orientation):
                    imdmap = axs[j/2,j%2].imshow(1e6 * OPS.reverse(delay_map[j,:,i].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), vmin=1e6*NP.nanmin(delay_map), vmax=1e6*NP.nanmax(delay_map))
                    axs[j/2,j%2].set_xlim(xvect.min(), xvect.max())
                    axs[j/2,j%2].set_ylim(yvect.min(), yvect.max())
                    axs[j/2,j%2].text(0.8, 0.9, 'E: {0[0]:.1f}m\nN: {0[1]:.1f}m\nZ: {0[2]:.1f}m'.format(cardinal_bl[j,:].ravel()), ha='left', va='top', transform=axs[j/2,j%2].transAxes)
                    # axs[j/2,j%2].set_aspect(1.33)
                    # ax.text(0.05, 0.7, '{0:.1f} %'.format(confidence_level*100), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    
                    # axs[j/2,j%2].set_autoscaley_on(False)
                    # imdmap = axs[j/2,j%2].imshow(1e6 * delay_map[j,:,i].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), vmin=1e6*NP.nanmin(delay_map), vmax=1e6*NP.nanmax(delay_map))
                    # axs[j/2,j%2].set_ylim(yvect.min(), yvect.max())
                    
                    # ax = axs[j/2,j%2]
                    # ax.set_ylim(yvect.min(), yvect.max())
                    # axs[j/2,j%2] = ax
                fig.subplots_adjust(wspace=0, hspace=0)
                big_ax = fig.add_subplot(111)
                big_ax.set_axis_bgcolor('none')
                big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)
                big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
    
                cbax = fig.add_axes([0.13, 0.92, 0.77, 0.02])
                cbar = fig.colorbar(imdmap, cax=cbax, orientation='horizontal')
                cbax.set_xlabel(r'delay [$\times\,(|\mathbf{b}|/100)\,\mu$s]', labelpad=-50, fontsize=18)
    
                # PLT.tight_layout()
                fig.subplots_adjust(top=0.88)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/delay_map_snapshot_{0:0d}.png'.format(i), bbox_inches=0)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/delay_map_snapshot_{0:0d}.eps'.format(i), bbox_inches=0)
                # for i in xrange(n_bins_baseline_orientation):
                #     axs[j/2,j%2].set_ylim(yvect.min(), yvect.max())
                #     axs[0,0].set_autoscaley_on(False)

        if plot_12:

            required_bl_orientation = ['North', 'East']
            for i in xrange(n_snaps):
                fig, axs = PLT.subplots(len(required_bl_orientation), sharex=True, sharey=True, figsize=(6,6))
                for k in xrange(len(required_bl_orientation)):
                    j = bl_orientation_str.index(required_bl_orientation[k])
                    imdmap = axs[k].imshow(1e6 * OPS.reverse(delay_map[j,:,i].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), vmin=1e6*NP.nanmin(delay_map), vmax=1e6*NP.nanmax(delay_map))
                    pbskyc = axs[k].contour(xgrid[0,:], ygrid[:,0], OPS.reverse(pb_snapshots[i].reshape(-1,backdrop_xsize), axis=1), levels=[0.001953125, 0.0078125, 0.03125, 0.125, 0.5], colors='k', linewidths=1.5)
                    axs[k].set_xlim(xvect.min(), xvect.max())
                    axs[k].set_ylim(yvect.min(), yvect.max())
                    axs[k].text(0.8, 0.9, required_bl_orientation[k], transform=axs[k].transAxes, fontsize=16, weight='semibold', ha='left', va='top')
                    axs[k].set_aspect('auto')
                fig.subplots_adjust(wspace=0, hspace=0)
                big_ax = fig.add_subplot(111)
                big_ax.set_axis_bgcolor('none')
                big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=20)
                big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=30)
    
                cbax = fig.add_axes([0.13, 0.92, 0.82, 0.02])
                cbar = fig.colorbar(imdmap, cax=cbax, orientation='horizontal')
                cbax.set_xlabel(r'delay [$(|\mathbf{b}|/100)\,\mu$s]', labelpad=-50, fontsize=18)
    
                # PLT.tight_layout()
                fig.subplots_adjust(top=0.88)
                fig.subplots_adjust(right=0.95)

                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/directional_delay_map_snapshot_{0:0d}.png'.format(i), bbox_inches=0)
                PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/directional_delay_map_snapshot_{0:0d}.eps'.format(i), bbox_inches=0)

##############################################################################
    
if plot_05 or plot_06 or plot_07 or plot_09 or plot_16:

    # 05) Plot FHD data and simulations on baselines by orientation and all combined

    fhd_obsid = [1061309344, 1061316544]

    infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)
    asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
    if use_alt_spindex:
        alt_asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(alt_spindex_rms)+alt_spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

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
        dspec_min = dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2
        dspec_max = dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2

    cardinal_blo = 180.0 / n_bins_baseline_orientation * (NP.arange(n_bins_baseline_orientation)-1).reshape(-1,1)
    cardinal_bll = 100.0
    cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

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
    # data_sim_difference_fraction = NP.zeros(len(fhd_obsid))

    if use_alt_spindex:
        alt_data_sim_ratio = []
        alt_data_sim_difference_fraction = []

    relevant_EoR_window = []
    relevant_wedge_window = []
    relevant_non_wedge_window = []

    if plot_05:

        descriptor_str = ['off-zenith', 'zenith']
        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*clean_lags, NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:-1,:-1,0].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
            horizonb = axs[j].plot(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            # kcontour = axs[j].contour(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*clean_lags, NP.sqrt((fhd_info[fhd_obsid[j]]['k_perp'].reshape(1,-1))**2+(fhd_info[fhd_obsid[j]]['k_prll'].reshape(-1,1))**2), levels=[0.04, 0.08, 0.16, 0.32], colors='k', linewidth=1.0)
            # axs[j].clabel(kcontour, inline=1, fontsize=8, colors='k', fmt='%0.2f')
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
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_fhd_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_fhd_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

        # fig = PLT.figure(figsize=(6,6))
        # for j in xrange(n_snaps):
        
        #     # Determine the baselines common to simulations and data
        
        #     # common_bl_ind = NP.squeeze(NP.where(NP.in1d(truncated_ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
        #     # sortind = NP.argsort(truncated_ref_bl_length[common_bl_ind], kind='heapsort')
        #     # bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])
    
        #     ax = fig.add_subplot(n_snaps,1,j+1)
        #     ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
        #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
        #     ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=18)
        #     imdspec = ax.pcolorfast(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*clean_lags, NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:-1,:-1,0].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
        #     horizonb = ax.plot(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls='-', lw=1.5)
        #     horizont = ax.plot(fhd_info[fhd_obsid[j]]['bl_length'], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls='-', lw=1.5)
        #     ax.set_aspect('auto')
    
        # cbax = fig.add_axes([0.86, 0.125, 0.02, 0.84])
        # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        # cbax.set_ylabel('Jy Hz', labelpad=0, fontsize=18)
        
        # PLT.tight_layout()
        # fig.subplots_adjust(right=0.83)
        # # fig.subplots_adjust(top=0.9)
    
        # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_visibilities_fhd_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)
    
        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j][:-1],:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
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
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)

        # fig = PLT.figure(figsize=(6,6))
        # for j in xrange(n_snaps):
        
        #     # Determine the baselines common to simulations and data
        
        #     # common_bl_ind = NP.squeeze(NP.where(NP.in1d(truncated_ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
        #     # sortind = NP.argsort(truncated_ref_bl_length[common_bl_ind], kind='heapsort')
        #     # bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])
    
        #     ax = fig.add_subplot(n_snaps,1,j+1)
        #     ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
        #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
        #     ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=18)
        #     imdspec = ax.pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j][:-1],:-1,j].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
        #     # imdspec = ax.pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.abs(asm_cc_vis_lag[common_bl_ind[sortind[:-1]],:-1,j].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
        #     horizonb = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls='-', lw=1.5)
        #     horizont = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls='-', lw=1.5)
        #     ax.set_aspect('auto')
    
        # cbax = fig.add_axes([0.86, 0.125, 0.02, 0.84])
        # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        # cbax.set_ylabel('Jy Hz', labelpad=0, fontsize=18)
        
        # PLT.tight_layout()
        # fig.subplots_adjust(right=0.83)
        # # fig.subplots_adjust(top=0.9)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

        # Plot each snapshot separately

        for j in xrange(n_snaps):
            fig = PLT.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            imdspec = ax.pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j][:-1],:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
            horizonb = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            horizont = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            ax.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            ax.set_aspect('auto')
            # ax.text(0.5, 0.9, descriptor_str[j], transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='white')

            ax_kprll = ax.twinx()
            ax_kprll.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
            ax_kprll.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            ax_kprll.yaxis.set_major_formatter(yformatter)
            ax_kperp = ax.twiny()
            ax_kperp.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
            ax_kperp.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            ax_kperp.xaxis.set_major_formatter(xformatter)

            ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
            ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
            ax_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
            ax_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
    
            cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.72)
            fig.subplots_adjust(top=0.88)

            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.eps', bbox_inches=0)


    for j in xrange(n_snaps):
    
        # Determine the baselines common to simulations and data
    
        # common_bl_ind = NP.squeeze(NP.where(NP.in1d(truncated_ref_bl_id, fhd_info[fhd_obsid[j]]['bl_id'])))
        # sortind = NP.argsort(truncated_ref_bl_length[common_bl_ind], kind='heapsort')
        # bloh, bloe, blon, blori = OPS.binned_statistic(fhd_info[fhd_obsid[j]]['bl_orientation'], statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

        # data_sim_ratio = NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)
        data_sim_ratio += [NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)]
        data_sim_difference_fraction += [(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)]
        if use_alt_spindex:
            # alt_data_sim_ratio += [NP.abs(alt_asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T) / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)]
            # alt_data_sim_difference_fraction += [(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) - NP.abs(alt_asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)]

            alt_data_sim_ratio += [NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(alt_asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)]
            alt_data_sim_difference_fraction += [(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) - NP.abs(alt_asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T)) / NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)]

        relevant_EoR_window += [small_delays_EoR_window[:,common_bl_ind_in_ref_snapshots[j]]]
        relevant_wedge_window += [small_delays_wedge_window[:,common_bl_ind_in_ref_snapshots[j]]]
        relevant_non_wedge_window += [small_delays_non_wedge_window[:,common_bl_ind_in_ref_snapshots[j]]]
    
        # data_sim_difference_fraction[j] = NP.mean(NP.abs(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T))/NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T))
        # data_sim_difference_fraction[j] = NP.mean(NP.abs(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T))/NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T))

        # data_sim_difference_fraction[j] = NP.sum(NP.abs(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T))) / NP.sum(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T))

        # data_sim_difference_fraction[j] = NP.sum(NP.abs(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T) - NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T))) / NP.sum(NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T))
        # data_sim_difference_fraction[j] = NP.abs(NP.sum(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][sortind,:,0].T)-NP.abs(NP.mean(fhd_info[fhd_obsid[j]]['rms_lag'])) - NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T)+NP.abs(NP.mean(vis_rms_lag)))) / NP.sum(NP.abs(asm_cc_vis_lag[common_bl_ind[sortind],:,j].T)-NP.abs(NP.mean(vis_rms_lag)))
   
        # mu = NP.mean(NP.log10(data_sim_ratio[relevant_EoR_window]))
        # sig= NP.std(NP.log10(data_sim_ratio[relevant_EoR_window]))
    
    if plot_06:
        # 06) Plot FHD data to simulation ratio on all baselines combined
        fig = PLT.figure(figsize=(6,6))
        for j in xrange(n_snaps):
            ax = fig.add_subplot(n_snaps,1,j+1)
            ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
            ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
            ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=18)
            imdspec = ax.pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, NP.log10(data_sim_ratio[j][:-1,:-1]), vmin=-1.0, vmax=1.0)
            # imdspec = ax.pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, data_sim_ratio[:-1,:-1], norm=PLTC.LogNorm(vmin=1e-1, vmax=1e1))
            horizonb = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            horizont = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
            ax.set_aspect('auto')
    
        cbax = fig.add_axes([0.91, 0.125, 0.02, 0.84])
        cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        # cbax.set_ylabel('Jy Hz', labelpad=0, fontsize=18)
        
        PLT.tight_layout()
        fig.subplots_adjust(right=0.88)
        # fig.subplots_adjust(top=0.9)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

        descriptor_str = ['off-zenith', 'zenith']
        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            hpdf, bins, patches = axs[j].hist(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]]), bins=50, normed=True, cumulative=False, histtype='step', linewidth=2.5, color='black')      
            mu = NP.mean(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]]))
            sig = NP.std(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]]))
            if use_alt_spindex:
                alt_hpdf, alt_bins, alt_patches = axs[j].hist(NP.log10(alt_data_sim_ratio[j][relevant_wedge_window[j]]), bins=50, normed=True, cumulative=False, histtype='step', linewidth=2.5, color='gray')      
                alt_mu = NP.mean(NP.log10(alt_data_sim_ratio[j][relevant_wedge_window[j]]))
                alt_sig = NP.std(NP.log10(alt_data_sim_ratio[j][relevant_wedge_window[j]]))
                print NP.median(NP.abs(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]]) - NP.median(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]])))), NP.median(NP.abs(NP.log10(alt_data_sim_ratio[j][relevant_wedge_window[j]]) - NP.median(NP.log10(alt_data_sim_ratio[j][relevant_wedge_window[j]]))))
            else:
                print NP.median(NP.abs(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]]) - NP.median(NP.log10(data_sim_ratio[j][relevant_wedge_window[j]]))))
            # print mu, sig
            # gauss_model = mlab.normpdf(bins, mu, sig)
            # modl = ax.plot(bins, gauss_model, 'k-')
            # axs[j].set_xlabel(r'$\log_{10}\,\rho$', fontsize=24, weight='medium')
            axs[j].set_xlim(-2, 2)
            axs[j].set_ylim(0.0, 1.1)
            axs[j].set_aspect('auto')
            axs[j].tick_params(which='major', length=12, labelsize=12)
            axs[j].tick_params(which='minor', length=6)
            axs[j].text(0.1, 0.8, descriptor_str[j], transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='left', color='black')

        fig.subplots_adjust(hspace=0)
        # PLT.tight_layout()
        fig.subplots_adjust(top=0.95)

        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xlabel(r'$\log_{10}\,\rho$', fontsize=24, weight='medium', labelpad=20)

        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'histogram_wedge_sim_data_log_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'histogram_wedge_sim_data_log_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    if plot_09:
        # 09) Plot histogram of fractional differences between FHD data and simulation 

        fig = PLT.figure(figsize=(6,6))
        for j in xrange(n_snaps):
            ax = fig.add_subplot(n_snaps,1,j+1)
            hpdf, bins, patches = ax.hist(data_sim_difference_fraction[j][relevant_wedge_window[j]], bins=50, normed=True, cumulative=False, histtype='step', linewidth=2.5, color='black', label='pdf (wedge)')        
            ax.set_xlabel(r'$\rho-1$', fontsize=24, weight='medium')
            ax.set_xlim(-2, 2)

        PLT.tight_layout()
        fig.subplots_adjust(bottom=0.15)

        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'histogram_wedge_sim_data_snr_log_fractional_diff_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'histogram_wedge_sim_data_snr_log_fractional_diff_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    if plot_07 or plot_08:
    
        dsm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_dsm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
        csm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_csm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
        
        hdulist = fits.open(dsm_CLEAN_infile+'.fits')
        dsm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
        dsm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
        dsm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
        dsm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
        hdulist.close()
        
        dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
        dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        dsm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
        dsm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis_res, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
        dsm_cc_skyvis_lag = dsm_cc_skyvis_lag + dsm_ccres_sky
        
        dsm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
        dsm_ccres = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis_res, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
        dsm_cc_vis_lag = dsm_cc_vis_lag + dsm_ccres
        
        hdulist = fits.open(csm_CLEAN_infile+'.fits')
        csm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
        csm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
        csm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
        csm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
        hdulist.close()
        
        csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        csm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
        csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        csm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
        csm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis_res, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
        csm_cc_skyvis_lag = csm_cc_skyvis_lag + csm_ccres_sky
        
        csm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
        csm_ccres = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis_res, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
        csm_cc_vis_lag = csm_cc_vis_lag + csm_ccres
        
        dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
        dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
        csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
        csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)

        dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
        dsm_cc_vis_lag = dsm_cc_vis_lag[truncated_ref_bl_ind,:,:]
        csm_cc_skyvis_lag = csm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
        csm_cc_vis_lag = csm_cc_vis_lag[truncated_ref_bl_ind,:,:]
        
        if max_abs_delay is not None:
            dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
            csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]
        
        dsm_cc_skyvis_lag_err = dsm_cc_skyvis_lag * NP.log(dsm_base_freq/freq) * dsm_dalpha
        csm_cc_skyvis_lag_err = csm_cc_skyvis_lag * NP.log(csm_base_freq/freq) * csm_dalpha
        # cc_skyvis_lag_err = NP.abs(dsm_cc_skyvis_lag_err) + NP.abs(csm_cc_skyvis_lag_err)
        # cc_skyvis_lag_err = NP.abs(dsm_cc_skyvis_lag_err + csm_cc_skyvis_lag_err)
        cc_skyvis_lag_err = NP.sqrt(NP.abs(dsm_cc_skyvis_lag_err)**2 + NP.abs(csm_cc_skyvis_lag_err)**2)

        if plot_07:

            # 07) Plot ratio of differences between FHD data and simulation to expected error on all baselines combined

            fig = PLT.figure(figsize=(6,6))
            for j in xrange(n_snaps):
                err_log_ratio_fhd = NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'].T)/NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)
                err_log_ratio_sim = NP.sqrt(cc_skyvis_lag_err[common_bl_ind_in_ref_snapshots[j],:,j]**2 + NP.abs(vis_rms_lag[common_bl_ind_in_ref_snapshots[j],:,j])**2).T / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j]).T
                err_log_ratio = NP.sqrt(err_log_ratio_sim**2 + err_log_ratio_fhd**2)
                data_sim_log_ratio = NP.log10(data_sim_ratio[j])
                # data_sim_log_ratio = NP.log10(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T))
                snr_log_ratio = data_sim_log_ratio / err_log_ratio

                ax = fig.add_subplot(n_snaps,1,j+1)
                ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
                ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
                ax.set_xlabel(r'$|\vec{mathbf{x}}|$ [m]', fontsize=18)
                imdspec = ax.pcolorfast(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*clean_lags, snr_log_ratio[:-1,:-1], vmin=-2, vmax=2)

                horizonb = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*min_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
                horizont = ax.plot(truncated_ref_bl_length[common_bl_ind_in_ref_snapshots[j]], 1e6*max_delay[common_bl_ind_in_ref_snapshots[j]].ravel(), color='white', ls=':', lw=1.5)
                ax.set_aspect('auto')
    
            cbax = fig.add_axes([0.86, 0.125, 0.02, 0.84])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')

            PLT.tight_layout()
            fig.subplots_adjust(right=0.83)

            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_snr_log_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noisy_PS_sim_data_snr_log_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

            fig = PLT.figure(figsize=(6,8))
            for j in xrange(n_snaps):
                err_log_ratio_fhd = NP.abs(fhd_info[fhd_obsid[j]]['rms_lag'].T)/NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T)
                err_log_ratio_sim = NP.sqrt(cc_skyvis_lag_err[common_bl_ind_in_ref_snapshots[j],:,j]**2 + NP.abs(vis_rms_lag[common_bl_ind_in_ref_snapshots[j],:,j])**2).T / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j]).T
                err_log_ratio = NP.sqrt(err_log_ratio_sim**2 + err_log_ratio_fhd**2)
                data_sim_log_ratio = NP.log10(data_sim_ratio[j])
                # data_sim_log_ratio = NP.log10(NP.abs(fhd_info[fhd_obsid[j]]['vis_lag_noisy'][:,:,0].T) / NP.abs(asm_cc_vis_lag[common_bl_ind_in_ref_snapshots[j],:,j].T))
                snr_data_sim_log_ratio = data_sim_log_ratio / err_log_ratio
                # relevant_EoR_window = small_delays_EoR_window[:,common_bl_ind_in_ref_snapshots[j]]
                # relevant_wedge_window = small_delays_wedge_window[:,common_bl_ind_in_ref_snapshots[j]]
                # relevant_non_wedge_window = small_delays_non_wedge_window[:,common_bl_ind_in_ref_snapshots[j]]

                ax = fig.add_subplot(n_snaps,1,j+1)
                hpdf, bins, patches = ax.hist(snr_data_sim_log_ratio[relevant_wedge_window[j]], bins=50, normed=True, cumulative=False, histtype='step', linewidth=2.5, color='black', label='pdf (wedge)')
                confidence_level = (NP.sum(hpdf[bins[:-1] <= 1.0]) - NP.sum(hpdf[bins[:-1] <= -1.0])) * (bins[1]-bins[0])
                print confidence_level
                hpdf, bins, patches = ax.hist(snr_data_sim_log_ratio[relevant_non_wedge_window[j]], bins=50, normed=True, cumulative=False, histtype='step', linewidth=2.5, color='black', label='pdf (outside)', linestyle='dashed')
                hcdf, bins, patches = ax.hist(snr_data_sim_log_ratio[relevant_wedge_window[j]], bins=50, normed=True, cumulative=True, histtype='step', linewidth=2.5, color='gray', label='cdf (outside)')
                hcdf, bins, patches = ax.hist(snr_data_sim_log_ratio[relevant_non_wedge_window[j]], bins=50, normed=True, cumulative=True, histtype='step', linewidth=2.5, color='gray', label='cdf (wedge)', linestyle='dashed')
                ax.set_xlabel(r'$\frac{\mathrm{log}_{10}\,\rho}{\Delta\,\mathrm{log}_{10}\,\rho}$', fontsize=24, weight='medium')
                # ax.set_xlabel('log(ratio) / err[log(ratio)]', fontsize=16, weight='medium')
                ax.set_xlim(-2, 2)

                ax.axvline(x=-1.0, ymax=0.67, color='black', ls=':', lw=2)
                ax.axvline(x=1.0, ymax=0.67, color='black', ls=':', lw=2)

                l1 = PLT.Line2D(range(1), range(1), color='black', linestyle='-', linewidth=2.5)
                l2 = PLT.Line2D(range(1), range(1), color='black', linestyle='--', linewidth=2.5)
                l3 = PLT.Line2D(range(1), range(1), color='gray', linestyle='-', linewidth=2.5)
                l4 = PLT.Line2D(range(1), range(1), color='gray', linestyle='--', linewidth=2.5)

                pdflegend = PLT.legend((l1, l2), ('pdf (wedge)', 'pdf (outside)'), loc='upper left', frameon=False)
                ax = PLT.gca().add_artist(pdflegend)
                cdflegend = PLT.legend((l3, l4), ('cdf (wedge)', 'cdf (outside)'), loc='upper right', frameon=False)
                # ax.text(0.05, 0.7, '{0:.1f} %'.format(confidence_level*100), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

            PLT.tight_layout()
            fig.subplots_adjust(bottom=0.15)

            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'histogram_wedge_sim_data_snr_log_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'histogram_wedge_sim_data_snr_log_ratio_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    if plot_16:
        # 16) Plot average thermal noise in simulations and data as a function of baseline length

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            axs[j].plot(truncated_ref_bl_length, NP.abs(vis_rms_lag[:,0,j]).ravel(), 'k.', label='Simulation')
            axs[j].plot(fhd_info[fhd_obsid[j]]['bl_length'], NP.abs(fhd_info[fhd_obsid[j]]['rms_lag']).ravel(), 'r.', label='MWA Data')
            axs[j].set_xlim(0.0, truncated_ref_bl_length.max())
            axs[j].set_yscale('log')
            axs[j].set_aspect('auto')
            legend = axs[j].legend(loc='upper right')
            legend.draw_frame(False)
            
        fig.subplots_adjust(hspace=0)

        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=24, weight='medium', labelpad=20)
        big_ax.set_ylabel(r'$V_{b\tau}^\mathrm{rms}(\mathbf{b})$ [Jy Hz]', fontsize=24, weight='medium', labelpad=20)
        
if plot_10 or plot_11 or plot_12 or plot_13 or plot_14:

    infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_'+fg_str+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)
    asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
    dsm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_dsm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape
    csm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/'+telescope_str+'multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_csm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+bpass_shape

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

    hdulist = fits.open(dsm_CLEAN_infile+'.fits')
    dsm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    dsm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    dsm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    dsm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()

    hdulist = fits.open(csm_CLEAN_infile+'.fits')
    csm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    csm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    csm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    csm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
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

    dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
    dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
    dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
    dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = dsm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()
    
    dsm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
    dsm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(dsm_cc_skyvis_res, axis=1),axes=1) * dsm_cc_skyvis.shape[1] * freq_resolution
    dsm_cc_skyvis_lag = dsm_cc_skyvis_lag + dsm_ccres_sky
    
    dsm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
    dsm_ccres = NP.fft.fftshift(NP.fft.ifft(dsm_cc_vis_res, axis=1),axes=1) * dsm_cc_vis.shape[1] * freq_resolution
    dsm_cc_vis_lag = dsm_cc_vis_lag + dsm_ccres

    csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
    csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
    csm_cc_vis[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis[simdata_neg_bl_orientation_ind,:,:].conj()
    csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:] = csm_cc_vis_res[simdata_neg_bl_orientation_ind,:,:].conj()
    
    csm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
    csm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(csm_cc_skyvis_res, axis=1),axes=1) * csm_cc_skyvis.shape[1] * freq_resolution
    csm_cc_skyvis_lag = csm_cc_skyvis_lag + csm_ccres_sky
    
    csm_cc_vis_lag = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
    csm_ccres = NP.fft.fftshift(NP.fft.ifft(csm_cc_vis_res, axis=1),axes=1) * csm_cc_vis.shape[1] * freq_resolution
    csm_cc_vis_lag = csm_cc_vis_lag + csm_ccres

    asm_cc_skyvis_lag = DSP.downsampler(asm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    asm_cc_vis_lag = DSP.downsampler(asm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    dsm_cc_skyvis_lag = DSP.downsampler(dsm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    dsm_cc_vis_lag = DSP.downsampler(dsm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    csm_cc_skyvis_lag = DSP.downsampler(csm_cc_skyvis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    csm_cc_vis_lag = DSP.downsampler(csm_cc_vis_lag, 1.0*clean_lags.size/ia.lags.size, axis=1)
    clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
    clean_lags = clean_lags.ravel()
    
    vis_noise_lag = NP.copy(ia.vis_noise_lag)
    vis_noise_lag = vis_noise_lag[truncated_ref_bl_ind,:,:]
    asm_cc_skyvis_lag = asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
    asm_cc_vis_lag = asm_cc_vis_lag[truncated_ref_bl_ind,:,:]
    csm_cc_skyvis_lag = csm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
    csm_cc_vis_lag = csm_cc_vis_lag[truncated_ref_bl_ind,:,:]
    dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
    dsm_cc_vis_lag = dsm_cc_vis_lag[truncated_ref_bl_ind,:,:]

    delaymat = DLY.delay_envelope(ia.baselines[truncated_ref_bl_ind,:], pc, units='mks')
    bw = nchan * freq_resolution
    min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
    max_delay = delaymat[0,:,0]-delaymat[0,:,1]
    clags = clean_lags.reshape(1,-1)
    min_delay = min_delay.reshape(-1,1)
    max_delay = max_delay.reshape(-1,1)
    thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
    thermal_noise_window = NP.repeat(thermal_noise_window, ia.baselines[truncated_ref_bl_ind,:].shape[0], axis=0)
    EoR_window = NP.logical_or(clags > max_delay+3/bw, clags < min_delay-3/bw)
    strict_EoR_window = NP.logical_and(EoR_window, NP.abs(clags) < 1/coarse_channel_resolution)
    wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
    non_wedge_window = NP.logical_not(wedge_window)
    vis_rms_lag = OPS.rms(asm_cc_vis_lag, mask=NP.logical_not(NP.repeat(thermal_noise_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
    vis_rms_freq = NP.abs(vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
    T_rms_freq = vis_rms_freq / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
    vis_rms_lag_theory = OPS.rms(vis_noise_lag, mask=NP.logical_not(NP.repeat(EoR_window[:,:,NP.newaxis], n_snaps, axis=2)), axis=1)
    vis_rms_freq_theory = NP.abs(vis_rms_lag_theory) / NP.sqrt(nchan) / freq_resolution
    T_rms_freq_theory = vis_rms_freq_theory / (2.0 * FCNST.k) * NP.mean(ia.A_eff[truncated_ref_bl_ind,:]) * NP.mean(ia.eff_Q[truncated_ref_bl_ind,:]) * NP.sqrt(2.0*freq_resolution*NP.asarray(ia.t_acc).reshape(1,1,-1)) * CNST.Jy
    
    if (dspec_min is None) or (dspec_max is None):
        dspec_max = max([NP.abs(asm_cc_skyvis_lag).max(), NP.abs(dsm_cc_skyvis_lag).max(), NP.abs(csm_cc_skyvis_lag).max()])
        dspec_min = min([NP.abs(asm_cc_skyvis_lag).min(), NP.abs(dsm_cc_skyvis_lag).min(), NP.abs(csm_cc_skyvis_lag).min()])
        dspec_max = dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2
        dspec_min = dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2

    small_delays_EoR_window = EoR_window.T
    small_delays_strict_EoR_window = strict_EoR_window.T
    small_delays_wedge_window = wedge_window.T

    if max_abs_delay is not None:
        small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
        clean_lags = clean_lags[small_delays_ind]
        asm_cc_vis_lag = asm_cc_vis_lag[:,small_delays_ind,:]
        asm_cc_skyvis_lag = asm_cc_skyvis_lag[:,small_delays_ind,:]
        dsm_cc_vis_lag = dsm_cc_vis_lag[:,small_delays_ind,:]
        dsm_cc_skyvis_lag = dsm_cc_skyvis_lag[:,small_delays_ind,:]
        csm_cc_vis_lag = csm_cc_vis_lag[:,small_delays_ind,:]
        csm_cc_skyvis_lag = csm_cc_skyvis_lag[:,small_delays_ind,:]
        small_delays_EoR_window = small_delays_EoR_window[small_delays_ind,:]
        small_delays_strict_EoR_window = small_delays_strict_EoR_window[small_delays_ind,:]
        small_delays_wedge_window = small_delays_wedge_window[small_delays_ind,:]
    
    if plot_10:
        # 10) Plot noiseless delay spectra from simulations for diffuse, compact and all-sky models

        descriptor_str = ['off-zenith', 'zenith']

        # All-sky model

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
            horizonb = axs[j].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
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
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)
    
        # fig = PLT.figure(figsize=(6,6))
        # for j in xrange(n_snaps):
        
        #     ax = fig.add_subplot(n_snaps,1,j+1)
        #     ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
        #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
        #     ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=18)
        #     imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[:-1,:-1,j].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
        #     horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls='-', lw=1.5)
        #     horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls='-', lw=1.5)
        #     ax.set_aspect('auto')
    
        # cbax = fig.add_axes([0.86, 0.125, 0.02, 0.84])
        # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        # cbax.set_ylabel('Jy Hz', labelpad=0, fontsize=18)
        
        # PLT.tight_layout()
        # fig.subplots_adjust(right=0.83)
        # # fig.subplots_adjust(top=0.9)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

        # Plot each snapshot separately

        for j in xrange(n_snaps):
            fig = PLT.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=1e0, vmax=1e12))
            horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            ax.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            ax.set_aspect('auto')
            # ax.text(0.5, 0.9, descriptor_str[j], transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    
            ax_kprll = ax.twinx()
            ax_kprll.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
            ax_kprll.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            ax_kprll.yaxis.set_major_formatter(yformatter)
            ax_kperp = ax.twiny()
            ax_kperp.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
            ax_kperp.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            ax_kperp.xaxis.set_major_formatter(xformatter)
    
            ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
            ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
            ax_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
            ax_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
        
            cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.72)
            fig.subplots_adjust(top=0.88)
        
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.eps', bbox_inches=0)

        # Diffuse foreground model

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
            horizonb = axs[j].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
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
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)
    
        # fig = PLT.figure(figsize=(6,6))
        # for j in xrange(n_snaps):
        
        #     ax = fig.add_subplot(n_snaps,1,j+1)
        #     ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
        #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
        #     ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=18)
        #     imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[:-1,:-1,j].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
        #     horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls='-', lw=1.5)
        #     horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls='-', lw=1.5)
        #     ax.set_aspect('auto')
    
        # cbax = fig.add_axes([0.86, 0.125, 0.02, 0.84])
        # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        # cbax.set_ylabel('Jy Hz', labelpad=0, fontsize=18)
        
        # PLT.tight_layout()
        # fig.subplots_adjust(right=0.83)
        # # fig.subplots_adjust(top=0.9)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_dsm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_dsm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)
            
        # Plot each snapshot separately

        for j in xrange(n_snaps):
            fig = PLT.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=1e0, vmax=1e12))
            horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            ax.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            ax.set_aspect('auto')
            # ax.text(0.5, 0.9, descriptor_str[j], transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    
            ax_kprll = ax.twinx()
            ax_kprll.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
            ax_kprll.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            ax_kprll.yaxis.set_major_formatter(yformatter)
            ax_kperp = ax.twiny()
            ax_kperp.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
            ax_kperp.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            ax_kperp.xaxis.set_major_formatter(xformatter)
    
            ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
            ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
            ax_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
            ax_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')

            cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.72)
            fig.subplots_adjust(top=0.88)
        
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_dsm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_dsm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.eps', bbox_inches=0)

        # Compact foreground model

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
            horizonb = axs[j].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
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
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)
    
        # fig = PLT.figure(figsize=(6,6))
        # for j in xrange(n_snaps):
        
        #     ax = fig.add_subplot(n_snaps,1,j+1)
        #     ax.set_ylim(NP.amin(clean_lags*1e6), NP.amax(clean_lags*1e6))
        #     ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
        #     ax.set_xlabel(r'$|\vec{mathbf{x}}|$ [m]', fontsize=18)
        #     imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[:-1,:-1,j].T), norm=PLTC.LogNorm(vmin=1e6, vmax=dspec_max))
        #     horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls='-', lw=1.5)
        #     horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls='-', lw=1.5)
        #     ax.set_aspect('auto')
    
        # cbax = fig.add_axes([0.86, 0.125, 0.02, 0.84])
        # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
        # cbax.set_ylabel('Jy Hz', labelpad=0, fontsize=18)
        
        # PLT.tight_layout()
        # fig.subplots_adjust(right=0.83)
        # # fig.subplots_adjust(top=0.9)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)
            
        # Plot each snapshot separately

        for j in xrange(n_snaps):
            fig = PLT.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=1e0, vmax=1e12))
            horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            ax.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            ax.set_aspect('auto')
            # ax.text(0.5, 0.9, descriptor_str[j], transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    
            ax_kprll = ax.twinx()
            ax_kprll.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
            ax_kprll.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            ax_kprll.yaxis.set_major_formatter(yformatter)

            ax_kperp = ax.twiny()
            ax_kperp.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
            ax_kperp.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            ax_kperp.xaxis.set_major_formatter(xformatter)
        
            ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
            ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
            ax_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
            ax_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')

            cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.72)
            fig.subplots_adjust(top=0.88)
        
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}_snapshot_{1:1d}'.format(oversampling_factor, j)+'.eps', bbox_inches=0)

        select_bl_id = ['47-21']
        # select_bl_id = ['125-124', '93-28', '95-51', '84-58', '167-166', '85-61', '94-23', '47-21', '63-58', '67-51', '68-18', '93-86']
    
        for blid in select_bl_id:
            fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
            for j in xrange(n_snaps):
                axs[j].plot(1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[ref_bl_id == blid,:,j]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'k:', lw=2, label='Diffuse')
                axs[j].plot(1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[ref_bl_id == blid,:,j]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'k--', lw=2, label='Compact')
                axs[j].plot(1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[ref_bl_id == blid,:,j]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'k-', lw=2, label='Both')
    
                dspec_ulim = NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j]).ravel()+NP.sqrt(NP.abs(csm_jacobian_spindex*csm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j])**2 + NP.abs(dsm_jacobian_spindex*dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j])**2 + NP.abs(vis_rms_lag[truncated_ref_bl_id==blid,:,j])**2).ravel()
                dspec_llim = NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j]).ravel()-NP.sqrt(NP.abs(csm_jacobian_spindex*csm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j])**2 + NP.abs(dsm_jacobian_spindex*dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j])**2 + NP.abs(vis_rms_lag[truncated_ref_bl_id==blid,:,j])**2).ravel()
                valid_ind = dspec_llim > 0.0
                dspec_llim[NP.logical_not(valid_ind)] = 10**4.8
                dspec_llim[small_delays_EoR_window[:,truncated_ref_bl_id==blid].ravel()] = 10**4.8

                dspec_llim = dspec_llim**2 * volfactor1 * volfactor2 * Jy2K**2
                dspec_ulim = dspec_ulim**2 * volfactor1 * volfactor2 * Jy2K**2
                
                axs[j].fill_between(1e6*clean_lags, dspec_ulim, dspec_llim, alpha=0.75, edgecolor='none', facecolor='gray')
    
                axs[j].axvline(x=1e6*min_delay[truncated_ref_bl_id==blid,0], ls='-.', lw=2, color='black')
                axs[j].axvline(x=1e6*max_delay[truncated_ref_bl_id==blid,0], ls='-.', lw=2, color='black')
                axs[j].set_yscale('log')
                axs[j].set_xlim(1e6*clean_lags.min(), 1e6*clean_lags.max())
                axs[j].set_ylim(dspec_llim.min(), 1.1*dspec_ulim.max())
                # axs[j].set_ylim(10**4.3, 1.1*(max([NP.abs(asm_cc_vis_lag[truncated_ref_bl_id==blid,:,j]).max(), NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j]).max(), NP.abs(dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j]).max(), NP.abs(csm_cc_vis_lag[truncated_ref_bl_id==blid,:,j]).max()])+NP.sqrt(NP.abs(csm_jacobian_spindex*csm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j])**2 + NP.abs(dsm_jacobian_spindex*dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,j])**2 + NP.abs(vis_rms_lag[truncated_ref_bl_id==blid,:,j])**2)).max())
                hl = PLT.Line2D(range(1), range(0), color='black', linestyle='-.', linewidth=2)
                csml = PLT.Line2D(range(1), range(0), color='black', linestyle='--', linewidth=2)
                dsml = PLT.Line2D(range(1), range(0), color='black', linestyle=':', linewidth=2)
                asml = PLT.Line2D(range(1), range(0), color='black', linestyle='-', linewidth=2)
                legend = axs[j].legend((dsml, csml, asml, hl), ('Diffuse', 'Compact', 'Both', 'Horizon\nLimit'), loc='upper right', frameon=False, fontsize=12)
    
                if j == 0:
                    axs[j].set_title('East: {0[0]:+.1f} m, North: {0[1]:+.1f} m, Up: {0[2]:+.1f} m'.format(truncated_ref_bl[truncated_ref_bl_id==blid].ravel()), fontsize=12, weight='medium')
            fig.subplots_adjust(hspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_xlabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=20)
            # big_ax.set_ylabel(r'$|V_{b\tau}(\mathbf{b},\tau)|$  [Jy Hz]', fontsize=16, weight='medium', labelpad=30)
            big_ax.set_ylabel(r"$P_d(k_\perp,k_\parallel)$  [K$^2$ (Mpc/$h)^3$]", fontsize=16, weight='medium', labelpad=30)
    
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_'+blid+'_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_'+blid+'_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_csm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)
    
        select_bl_id = ['85-61', '63-58', '95-51']
        fig, axs = PLT.subplots(len(select_bl_id), sharex=True, sharey=True, figsize=(6,8))
    
        for j in xrange(len(select_bl_id)):
            blid = select_bl_id[j]
            axs[j].plot(1e6*clean_lags, NP.abs(dsm_cc_skyvis_lag[ref_bl_id == blid,:,1]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'r-', lw=2, label='Diffuse')
            axs[j].plot(1e6*clean_lags, NP.abs(csm_cc_skyvis_lag[ref_bl_id == blid,:,1]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, ls='-', lw=2, label='Compact', color='cyan')
            axs[j].plot(1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[ref_bl_id == blid,:,1]).ravel()**2 * volfactor1 * volfactor2 * Jy2K**2, 'k-', lw=2, label='Both')
    
            dspec_ulim = NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1]).ravel()+NP.sqrt(NP.abs(csm_jacobian_spindex*csm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1])**2 + NP.abs(dsm_jacobian_spindex*dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1])**2 + NP.abs(vis_rms_lag[truncated_ref_bl_id==blid,:,1])**2).ravel()
            dspec_llim = NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1]).ravel()-NP.sqrt(NP.abs(csm_jacobian_spindex*csm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1])**2 + NP.abs(dsm_jacobian_spindex*dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1])**2 + NP.abs(vis_rms_lag[truncated_ref_bl_id==blid,:,1])**2).ravel()
            valid_ind = dspec_llim > 0.0
            dspec_llim[NP.logical_not(valid_ind)] = 10**4.8
            dspec_llim[small_delays_EoR_window[:,truncated_ref_bl_id==blid].ravel()] = 10**4.8
            
            dspec_llim = dspec_llim**2 * volfactor1 * volfactor2 * Jy2K**2
            dspec_ulim = dspec_ulim**2 * volfactor1 * volfactor2 * Jy2K**2

            axs[j].fill_between(1e6*clean_lags, dspec_ulim, dspec_llim, alpha=0.75, edgecolor='none', facecolor='gray')
    
            axs[j].axvline(x=1e6*min_delay[truncated_ref_bl_id==blid,0], ls=':', lw=2, color='black')
            axs[j].axvline(x=1e6*max_delay[truncated_ref_bl_id==blid,0], ls=':', lw=2, color='black')
            axs[j].set_yscale('log')
            axs[j].set_xlim(1e6*clean_lags.min(), 1e6*clean_lags.max())
            axs[j].set_ylim(dspec_llim.min(), 1.5*dspec_ulim.max())
            # axs[j].set_ylim(10**4.3, 1.1*(max([NP.abs(asm_cc_vis_lag[truncated_ref_bl_id==blid,:,1]).max(), NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1]).max(), NP.abs(dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1]).max(), NP.abs(csm_cc_vis_lag[truncated_ref_bl_id==blid,:,1]).max()])+NP.sqrt(NP.abs(csm_jacobian_spindex*csm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1])**2 + NP.abs(dsm_jacobian_spindex*dsm_cc_skyvis_lag[truncated_ref_bl_id==blid,:,1])**2 + NP.abs(vis_rms_lag[truncated_ref_bl_id==blid,:,1])**2)).max())
            hl = PLT.Line2D(range(1), range(0), color='black', linestyle=':', linewidth=2)
            dsml = PLT.Line2D(range(1), range(0), color='red', linestyle='-', linewidth=2)
            csml = PLT.Line2D(range(1), range(0), color='cyan', linestyle='-', linewidth=2)
            asml = PLT.Line2D(range(1), range(0), color='black', linestyle='-', linewidth=2)
            legend = axs[j].legend((dsml, csml, asml, hl), ('Diffuse', 'Compact', 'Both', 'Horizon\nLimit'), loc='upper right', frameon=False, fontsize=12)
            axs[j].text(0.05, 0.8, r'$|\mathbf{b}|$'+' = {0:.1f} m'.format(truncated_ref_bl_length[truncated_ref_bl_id==blid][0]), fontsize=12, weight='medium', transform=axs[j].transAxes)
            axs[j].text(0.05, 0.72, r'$\theta_b$'+' = {0:+.1f}$^\circ$'.format(truncated_ref_bl_orientation[truncated_ref_bl_id==blid][0]), fontsize=12, weight='medium', transform=axs[j].transAxes)
            # axs[j].text(0.05, 0.7, 'East: {0[0]:+.1f} m\nNorth: {0[1]:+.1f} m\nUp: {0[2]:+.1f} m'.format(truncated_ref_bl[truncated_ref_bl_id==blid].ravel()), fontsize=12, weight='medium', transform=axs[j].transAxes)
    
            if j == 0:
                axs_kprll = axs[j].twiny()
                axs_kprll.set_xticks(kprll(axs[j].get_xticks()*1e-6, redshift))
                axs_kprll.set_xlim(kprll(NP.asarray(axs[j].get_xlim())*1e-6, redshift))
                xformatter = FuncFormatter(lambda x, pos: '{0:.2f}'.format(x))
                axs_kprll.xaxis.set_major_formatter(xformatter)

        fig.subplots_adjust(hspace=0)
        big_ax = fig.add_subplot(111)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xlabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium', labelpad=20)
        # big_ax.set_ylabel(r'$|V_{b\tau}(\mathbf{b},\tau)|$  [Jy Hz]', fontsize=16, weight='medium', labelpad=30)
        big_ax.set_ylabel(r"$P_d(k_\perp,k_\parallel)$  [K$^2$ (Mpc/$h)^3$]", fontsize=16, weight='medium', labelpad=30)

        big_axt = big_ax.twiny()
        big_axt.set_axis_bgcolor('none')
        big_axt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_axt.set_xticks([])
        big_axt.set_yticks([])
        big_axt.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium', labelpad=30)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'{0:0d}_baseline_comparison'.format(len(select_bl_id))+'_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'{0:0d}_baseline_comparison'.format(len(select_bl_id))+'_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    bl_orientation = NP.copy(simdata_bl_orientation[truncated_ref_bl_ind])
    bloh, bloe, blon, blori = OPS.binned_statistic(bl_orientation, bins=n_bins_baseline_orientation, statistic='count', range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

    if plot_11:

        for j in xrange(n_snaps):
            fig, axs = PLT.subplots(n_bins_baseline_orientation, sharex=True, sharey=True, figsize=(6,9))
            for i in xrange(n_bins_baseline_orientation):
                blind = blori[blori[i]:blori[i+1]]
                sortind = NP.argsort(truncated_ref_bl_length[blind], kind='heapsort')
                imdspec = axs[n_bins_baseline_orientation-1-i].pcolorfast(truncated_ref_bl_length[blind[sortind]], 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:][blind[sortind][:-1],:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
                horizonb = axs[n_bins_baseline_orientation-1-i].plot(truncated_ref_bl_length[blind], 1e6*min_delay[blind].ravel(), color='white', ls=':', lw=1.5)
                horizont = axs[n_bins_baseline_orientation-1-i].plot(truncated_ref_bl_length[blind], 1e6*max_delay[blind].ravel(), color='white', ls=':', lw=1.5)
                axs[n_bins_baseline_orientation-1-i].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
                axs[n_bins_baseline_orientation-1-i].set_xlim(truncated_ref_bl_length.min(), truncated_ref_bl_length.max())
                axs[n_bins_baseline_orientation-1-i].set_aspect('auto')
                axs[n_bins_baseline_orientation-1-i].text(0.5, 0.1, bl_orientation_str[i]+': '+r'${0:+.1f}^\circ \leq\, \theta_b < {1:+.1f}^\circ$'.format(bloe[i], bloe[i+1]), fontsize=12, color='white', transform=axs[n_bins_baseline_orientation-1-i].transAxes, weight='bold', ha='center')

            for i in xrange(n_bins_baseline_orientation):
                axs_kprll = axs[i].twinx()
                axs_kprll.set_yticks(kprll(axs[i].get_yticks()*1e-6, redshift))
                axs_kprll.set_ylim(kprll(NP.asarray(axs[i].get_ylim())*1e-6, redshift))
                yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
                axs_kprll.yaxis.set_major_formatter(yformatter)
                if i == 0:
                    axs_kperp = axs[i].twiny()
                    axs_kperp.set_xticks(kperp(axs[i].get_xticks()*freq/FCNST.c, redshift))
                    axs_kperp.set_xlim(kperp(NP.asarray(axs[i].get_xlim())*freq/FCNST.c, redshift))
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

            # cbax = fig.add_axes([0.9, 0.1, 0.02, 0.78])
            # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            # cbax.set_xlabel('Jy Hz', labelpad=10, fontsize=12)
            # cbax.xaxis.set_label_position('top')
            
            # fig.subplots_adjust(right=0.75)
            # fig.subplots_adjust(top=0.92)
            # fig.subplots_adjust(bottom=0.07)

            cbax = fig.add_axes([0.125, 0.94, 0.72, 0.02])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='horizontal')
            cbax.xaxis.tick_top()
            cbax.set_ylabel(r'K$^2$(Mpc/h)$^3$', fontsize=12, rotation='horizontal')
            # cbax.yaxis.set_label_position('right')
            cbax.yaxis.set_label_coords(1.1, 1.0)
            
            fig.subplots_adjust(right=0.86)
            fig.subplots_adjust(top=0.85)
            fig.subplots_adjust(bottom=0.07)

            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_binned_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_binned_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)

    if plot_12:

        required_bl_orientation = ['North', 'East']
        for j in xrange(n_snaps):
            fig, axs = PLT.subplots(len(required_bl_orientation), sharex=True, sharey=True, figsize=(6,6))
            for k in xrange(len(required_bl_orientation)):
                i = bl_orientation_str.index(required_bl_orientation[k])
                blind = blori[blori[i]:blori[i+1]]
                sortind = NP.argsort(truncated_ref_bl_length[blind], kind='heapsort')
                imdspec = axs[k].pcolorfast(truncated_ref_bl_length[blind[sortind]], 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:][blind[sortind][:-1],:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=dspec_max))
                horizonb = axs[k].plot(truncated_ref_bl_length[blind], 1e6*min_delay[blind].ravel(), color='white', ls=':', lw=1.0)
                horizont = axs[k].plot(truncated_ref_bl_length[blind], 1e6*max_delay[blind].ravel(), color='white', ls=':', lw=1.0)
                axs[k].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
                axs[k].set_xlim(truncated_ref_bl_length.min(), truncated_ref_bl_length.max())
                axs[k].set_aspect('auto')
                axs[k].text(0.5, 0.1, bl_orientation_str[i]+': '+r'${0:+.1f}^\circ \leq\, \theta_b < {1:+.1f}^\circ$'.format(bloe[i], bloe[i+1]), fontsize=16, color='white', transform=axs[k].transAxes, weight='semibold', ha='center')

            for i in xrange(len(required_bl_orientation)):
                axs_kprll = axs[i].twinx()
                axs_kprll.set_yticks(kprll(axs[i].get_yticks()*1e-6, redshift))
                axs_kprll.set_ylim(kprll(NP.asarray(axs[i].get_ylim())*1e-6, redshift))
                yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
                axs_kprll.yaxis.set_major_formatter(yformatter)
                if i == 0:
                    axs_kperp = axs[i].twiny()
                    axs_kperp.set_xticks(kperp(axs[i].get_xticks()*freq/FCNST.c, redshift))
                    axs_kperp.set_xlim(kperp(NP.asarray(axs[i].get_xlim())*freq/FCNST.c, redshift))
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

            # cbax = fig.add_axes([0.9, 0.1, 0.02, 0.78])
            # cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            # cbax.set_xlabel('Jy Hz', labelpad=10, fontsize=12)
            # cbax.xaxis.set_label_position('top')
            
            # fig.subplots_adjust(right=0.75)
            # fig.subplots_adjust(top=0.92)
            # fig.subplots_adjust(bottom=0.07)

            cbax = fig.add_axes([0.125, 0.92, 0.72, 0.02])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='horizontal')
            cbax.xaxis.tick_top()
            cbax.set_ylabel(r'K$^2$(Mpc/h)$^3$', fontsize=12, rotation='horizontal')
            # cbax.yaxis.set_label_position('right')
            cbax.yaxis.set_label_coords(1.1, 1.0)
            
            fig.subplots_adjust(right=0.86)
            fig.subplots_adjust(top=0.79)
            fig.subplots_adjust(bottom=0.09)
        
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_N_E_binned_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.png'.format(j), bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_N_E_binned_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)

    if plot_13:
        # 13) Plot EoR window foreground contamination when baselines are selectively removed

        blo_target = 0.0
        n_blo_remove_range = 3
        n_inner_bll_remove_range = 20
        blo_remove_max = 0.5*180.0/n_bins_baseline_orientation*(1+NP.arange(n_blo_remove_range))/n_blo_remove_range
        inner_bll_remove_max = NP.logspace(NP.log10(truncated_ref_bl_length.min()), NP.log10(max_bl_length), n_inner_bll_remove_range)
        bl_screened_fg_contamination = NP.zeros((n_blo_remove_range, n_inner_bll_remove_range), dtype=NP.complex)
        fraction_bl_discarded = NP.zeros((n_blo_remove_range, n_inner_bll_remove_range))

        ns_blind = blori[blori[3]:blori[3+1]]
        ns_fg_contamination = OPS.rms(NP.abs(asm_cc_skyvis_lag[ns_blind,:,0])**2, mask=NP.logical_not(small_delays_strict_EoR_window[:,ns_blind]).T) * volfactor1 * volfactor2 * Jy2K**2
        ew_blind = blori[blori[1]:blori[1+1]]
        ew_fg_contamination = OPS.rms(NP.abs(asm_cc_skyvis_lag[ew_blind,:,0])**2, mask=NP.logical_not(small_delays_strict_EoR_window[:,ew_blind]).T) * volfactor1 * volfactor2 * Jy2K**2
        for i in xrange(n_blo_remove_range):
            blo_retain_ind = NP.abs(bl_orientation - blo_target) > blo_remove_max[i]
            blo_discard_ind = NP.logical_not(blo_retain_ind)
            for j in xrange(n_inner_bll_remove_range):
                bll_retain_ind = truncated_ref_bl_length > inner_bll_remove_max[j]
                bll_discard_ind = NP.logical_not(bll_retain_ind)
                retain = NP.logical_not(NP.logical_and(blo_discard_ind, bll_discard_ind))
                mask = NP.logical_not(NP.logical_and(small_delays_strict_EoR_window, retain.reshape(1,-1)))
                bl_screened_fg_contamination[i,j] = OPS.rms(NP.abs(asm_cc_skyvis_lag[:,:,0])**2, mask=mask.T) * volfactor1 * volfactor2 * Jy2K**2
                fraction_bl_discarded[i,j] = 1.0 - NP.sum(retain).astype(float)/truncated_ref_bl_length.size
            
        symbols = ['o', 's', '*', 'd', '+', 'x']
        fig = PLT.figure(figsize=(6,6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        for i in xrange(n_blo_remove_range):
            ax1.plot(inner_bll_remove_max, bl_screened_fg_contamination[i,:], marker=symbols[i], markersize=6, lw=1, color='k', ls='-', label=r'$|\theta_b|\,\leq\,${0:.1f}$^\circ$'.format(blo_remove_max[i]))
            ax2.plot(inner_bll_remove_max, fraction_bl_discarded[i,:], marker=symbols[i], markersize=5, color='k', lw=1, ls=':', label=r'$|\theta_b|\,\leq\,${0:.1f}$^\circ$'.format(blo_remove_max[i]))
        # ax1.axhline(y=NP.abs(ew_fg_contamination), color='k', ls='-.', lw=2, label='Eastward limit')
        # ax1.axhline(y=NP.abs(ns_fg_contamination), color='k', ls='--', lw=2, label='Northward limit')
        ax1.set_ylim(0.3*bl_screened_fg_contamination.min(), 1.2*bl_screened_fg_contamination.max())
        # ax1.set_ylim(0.9*NP.abs(ns_fg_contamination), 1.1*NP.abs(ew_fg_contamination))
        ax1.set_xlim(0.9*inner_bll_remove_max.min(), 1.1*inner_bll_remove_max.max())
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel(r'Eastward $|\mathbf{b}|_\mathrm{max}$ [m]', fontsize=18, weight='medium')
        ax1.set_ylabel(r'Foreground Contamination [K$^2$(Mpc/h)$^3$]', fontsize=18, weight='medium')
        # ax1.set_ylabel(r'$\langle|V_{b\tau}^\mathrm{FG}(\mathbf{b},\tau)|^2\rangle^{1/2}_{\in\,\mathrm{EoR\,window}}$ [Jy Hz]', fontsize=18, weight='medium')
        # legend = ax1.legend(loc='lower right')
        # legend = ax1.legend(loc='lower right', fancybox=True, framealpha=1.0)

        ax2.set_yscale('log') 
        ax2.set_xscale('log')
        ax2.set_ylim(1e-3, 1.0)
        ax2.set_ylabel('Baseline fraction discarded', fontsize=18, weight='medium', color='k')

        legend1_symbol = []
        legend1_text = []
        for i in xrange(n_blo_remove_range):
            legend1_symbol += [PLT.Line2D(range(1), range(0), marker=symbols[i], markersize=6, color='k', linestyle='None')]
            legend1_text += [r'$|\theta_b|\,\leq\,${0:.1f}$^\circ$'.format(blo_remove_max[i])]

        legend2_symbol = []
        legend2_text = []
        # legend2_symbol += [PLT.Line2D(range(1), range(0), linestyle='-.', lw=1.5, color='k')]
        # legend2_symbol += [PLT.Line2D(range(1), range(0), linestyle='--', lw=1.5, color='k')]
        legend2_symbol += [PLT.Line2D(range(1), range(0), linestyle='-', lw=1.5, color='k')]
        legend2_symbol += [PLT.Line2D(range(1), range(0), linestyle=':', lw=1.5, color='k')]
        # legend2_text += ['Foreground upper limit']
        # legend2_text += ['Foreground lower limit']
        legend2_text += ['Foreground in EoR window']
        legend2_text += ['Baseline fraction']

        legend1 = ax1.legend(legend1_symbol, legend1_text, loc='lower right', numpoints=1, fontsize='medium')
        legend2 = ax2.legend(legend2_symbol, legend2_text, loc='upper right', fontsize='medium')

        PLT.tight_layout()

        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_screening_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'baseline_screening_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

    if plot_14:
        # 14) Plot delay spectra before and after baselines are selectively removed        
        
        blo_target = 0.0
        blo_remove_max = 0.5*180.0/n_bins_baseline_orientation
        inner_bll_remove_max = 30.0
        blo_retain_ind = NP.abs(bl_orientation - blo_target) > blo_remove_max
        blo_discard_ind = NP.logical_not(blo_retain_ind)
        bll_retain_ind = truncated_ref_bl_length > inner_bll_remove_max
        bll_discard_ind = NP.logical_not(bll_retain_ind)
        discard = NP.logical_and(blo_discard_ind, bll_discard_ind)
        retain = NP.logical_not(discard)
        msk = NP.zeros((truncated_ref_bl_length.size, clean_lags.size))
        msk[discard,:] = 1
        colrmap = copy.copy(CM.jet)
        colrmap.set_bad(color='black', alpha=1.0)
        bl_screened_asm_cc_skyvis_lag = NP.ma.masked_array(asm_cc_skyvis_lag[:,:,0], mask=msk)
        # bl_screened_asm_cc_skyvis_lag = NP.ma.filled(bl_screened_asm_cc_skyvis_lag, fill_value=1e-5)
        # bl_screened_asm_cc_skyvis_lag = NP.ma.compress_rows(bl_screened_asm_cc_skyvis_lag)

        # bl_screened_asm_cc_skyvis_lag = NP.copy(asm_cc_skyvis_lag[:,:,0])
        # bl_screened_asm_cc_skyvis_lag[discard,:] = 1e-3

        descriptor_str = ['All baselines', 'Short eastward baselines removed']

        fig, axs = PLT.subplots(2, sharex=True, sharey=True, figsize=(6,6))
        all_imdspec = axs[0].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(asm_cc_skyvis_lag[:-1,:-1,0].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=NP.abs(asm_cc_skyvis_lag).max()**2 * volfactor1 * volfactor2 * Jy2K**2))
        screened_imdspec = axs[1].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(bl_screened_asm_cc_skyvis_lag[:-1,:-1].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e6)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=NP.abs(asm_cc_skyvis_lag).max()**2 * volfactor1 * volfactor2 * Jy2K**2), cmap=colrmap)
        for j in xrange(len(axs)):
            bll_cut = axs[j].axvline(x=inner_bll_remove_max, ymin=0.0, ymax=0.75, ls='--', color='white', lw=1.5)
            horizonb = axs[j].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            axs[j].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[j].set_aspect('auto')
            axs[j].text(0.5, 0.8, descriptor_str[j], transform=axs[j].transAxes, fontsize=12, weight='semibold', ha='center', color='white')

        for j in xrange(len(axs)):
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
        cbar = fig.colorbar(all_imdspec, cax=cbax, orientation='vertical')
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)

        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_screening_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'multi_baseline_screening_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)

##################################################################

if plot_15:
    # 15) Plot Fourier space    

    bw = nchan * freq_resolution

    fig = PLT.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    # ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), 'k-', truncated_ref_bl_length, 1e6*max_delay.ravel(), 'k-')
    # ax.plot(truncated_ref_bl_length, 1e6*(min_delay.ravel()-1/bw), 'k--', truncated_ref_bl_length, 1e6*(max_delay.ravel()+1/bw), 'k--')        
    ph_line, nh_line = ax.plot(truncated_ref_bl_length, 1e6*truncated_ref_bl_length/FCNST.c, 'k-', truncated_ref_bl_length, -1e6*truncated_ref_bl_length/FCNST.c, 'k-')
    ax.plot(truncated_ref_bl_length, -1e6*(truncated_ref_bl_length/FCNST.c + 1/bw), 'k--', truncated_ref_bl_length, 1e6*(truncated_ref_bl_length/FCNST.c + 1/bw), 'k--')
    ax.plot(truncated_ref_bl_length[truncated_ref_bl_length <= FCNST.c/coarse_channel_resolution], 1e6/coarse_channel_resolution*NP.ones(NP.sum(truncated_ref_bl_length <= FCNST.c/coarse_channel_resolution)), 'k-.')
    ax.plot(truncated_ref_bl_length[truncated_ref_bl_length <= FCNST.c/coarse_channel_resolution], -1e6/coarse_channel_resolution*NP.ones(NP.sum(truncated_ref_bl_length <= FCNST.c/coarse_channel_resolution)), 'k-.')
    ax.fill_between(truncated_ref_bl_length, -0.5/freq_resolution*1e6, -1e6*(truncated_ref_bl_length/FCNST.c + 1/bw), facecolor='0.8', edgecolor='none')
    ax.fill_between(truncated_ref_bl_length, 1e6*(truncated_ref_bl_length/FCNST.c + 1/bw), 0.5/freq_resolution*1e6, facecolor='0.8', edgecolor='none')
    ax.fill_between(truncated_ref_bl_length, -1e6/coarse_channel_resolution, -1e6*(truncated_ref_bl_length/FCNST.c + 1/bw), facecolor='0.7', edgecolor='none')
    ax.fill_between(truncated_ref_bl_length, 1e6*(truncated_ref_bl_length/FCNST.c + 1/bw), 1e6/coarse_channel_resolution, facecolor='0.7', edgecolor='none')
    ax.fill_between(truncated_ref_bl_length, -1e6*truncated_ref_bl_length/FCNST.c, 1e6*truncated_ref_bl_length/FCNST.c, facecolor='0.5', edgecolor='none')
    ax.set_xlim(truncated_ref_bl_length.min(), truncated_ref_bl_length.max())
    ax.set_ylim(-1.25, 1.25)

    ax.text(0.5, 0.5, 'Foregrounds', transform=ax.transAxes, fontsize=12, weight='semibold', ha='left', color='black')
    ax.text(100, 1e6/coarse_channel_resolution, 'Delay grating', fontsize=12, weight='semibold', ha='left', color='black', va='bottom')
    ax.text(100, -1e6/coarse_channel_resolution, 'Delay grating', fontsize=12, weight='semibold', ha='left', color='black', va='top')
    ax.text(10, 0.45, 'Maximal EoR \nsensitivity', fontsize=12, weight='semibold', ha='left', va='center')
    ax.text(10, -0.45, 'Maximal EoR \nsensitivity', fontsize=12, weight='semibold', ha='left', va='center')
    anchor_bll = 125.0
    anchor_nh_delay = -1e6 * anchor_bll/FCNST.c 
    anchor_ph_delay = 1e6 * anchor_bll/FCNST.c 
    nhp1 = ax.transData.transform_point(NP.array([nh_line.get_xdata()[0], nh_line.get_ydata()[0]]))
    nhp2 = ax.transData.transform_point(NP.array([nh_line.get_xdata()[-1], nh_line.get_ydata()[-1]]))
    nh_angle = NP.degrees(NP.arctan2(nhp2[1]-nhp1[1], nhp2[0]-nhp1[0]))
    php1 = ax.transData.transform_point(NP.array([ph_line.get_xdata()[0], ph_line.get_ydata()[0]]))
    php2 = ax.transData.transform_point(NP.array([ph_line.get_xdata()[-1], ph_line.get_ydata()[-1]]))
    ph_angle = NP.degrees(NP.arctan2(php2[1]-php1[1], php2[0]-php1[0]))
    nh_text = ax.text(anchor_bll, anchor_nh_delay, 'Horizon', fontsize=12, weight='semibold', rotation=nh_angle, ha='left')
    ph_text = ax.text(anchor_bll, anchor_ph_delay, 'Horizon', fontsize=12, weight='semibold', rotation=ph_angle, ha='left')

    # ax.set_ylim(-0.5/freq_resolution*1e6, 0.5/freq_resolution*1e6)
    ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
    ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')

    axr = ax.twinx()
    axr.set_yticks([])
    axr.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
    axr.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
    axr.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
    yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
    axr.yaxis.set_major_formatter(yformatter)

    axt = ax.twiny()
    axt.set_xticks([])
    axt.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
    axt.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
    axt.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
    xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
    axt.xaxis.set_major_formatter(xformatter)

    PLT.tight_layout()

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/fourier_space_{0:.1f}_MHz_{1:.1f}_MHz.png'.format(freq/1e6,nchan*freq_resolution/1e6))
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/fourier_space_{0:.1f}_MHz_{1:.1f}_MHz.eps'.format(freq/1e6,nchan*freq_resolution/1e6))

##################################################################

if plot_17 or plot_18 or plot_19:

    delta_array_usm_infile = '/data3/t_nithyanandan/project_MWA/delta_array_multi_baseline_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_usm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'no_pfb'
    delta_array_usm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/delta_array_multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_usm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'no_pfb_'+bpass_shape
    delta_array_asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/delta_array_multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'no_pfb_'+bpass_shape
    mwa_dipole_asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/mwa_dipole_multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'no_pfb_'+bpass_shape
    hera_asm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/hera_multi_baseline_CLEAN_visibilities_'+ground_plane_str+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_asm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'no_pfb_'+bpass_shape
    delta_usm_CLEAN_infile = '/data3/t_nithyanandan/project_MWA/delta_multi_baseline_CLEAN_visibilities_no_ground_'+snapshot_type_str+obs_mode+'_baseline_range_{0:.1f}-{1:.1f}_'.format(ref_bl_length[baseline_bin_indices[0]],ref_bl_length[min(baseline_bin_indices[n_bl_chunks-1]+baseline_chunk_size-1,total_baselines-1)])+'gaussian_FG_model_usm'+sky_sector_str+'sprms_{0:.1f}_'.format(spindex_rms)+spindex_seed_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'no_pfb_'+bpass_shape

    ia = RI.InterferometerArray(None, None, None, init_file=delta_array_usm_infile+'.fits')    
    simdata_bl_orientation = NP.angle(ia.baselines[:,0] + 1j * ia.baselines[:,1], deg=True)
    simdata_neg_bl_orientation_ind = simdata_bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    simdata_bl_orientation[simdata_neg_bl_orientation_ind] -= 180.0
    ia.baselines[simdata_neg_bl_orientation_ind,:] = -ia.baselines[simdata_neg_bl_orientation_ind,:]
    
    hdulist = fits.open(delta_array_usm_infile+'.fits')
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

    hdulist = fits.open(delta_array_usm_CLEAN_infile+'.fits')
    clean_lags = hdulist['SPECTRAL INFO'].data['lag']
    clean_lags_orig = NP.copy(clean_lags)
    delta_array_usm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    delta_array_usm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    delta_array_usm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    delta_array_usm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()

    hdulist = fits.open(delta_array_asm_CLEAN_infile+'.fits')
    delta_array_asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    delta_array_asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    delta_array_asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    delta_array_asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()

    hdulist = fits.open(mwa_dipole_asm_CLEAN_infile+'.fits')
    mwa_dipole_asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    mwa_dipole_asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    mwa_dipole_asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    mwa_dipole_asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()

    hdulist = fits.open(hera_asm_CLEAN_infile+'.fits')
    hera_asm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    hera_asm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    hera_asm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    hera_asm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()

    hdulist = fits.open(delta_usm_CLEAN_infile+'.fits')
    delta_usm_cc_skyvis = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data
    delta_usm_cc_skyvis_res = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data
    delta_usm_cc_vis = hdulist['CLEAN NOISY VISIBILITIES REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data
    delta_usm_cc_vis_res = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data + 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
    hdulist.close()

    clean_lags = DSP.downsampler(clean_lags, 1.0*clean_lags.size/ia.lags.size, axis=-1)
    clean_lags = clean_lags.ravel()

    delaymat = DLY.delay_envelope(ia.baselines[truncated_ref_bl_ind,:], pc, units='mks')
    bw = nchan * freq_resolution
    min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
    max_delay = delaymat[0,:,0]-delaymat[0,:,1]
    clags = clean_lags.reshape(1,-1)
    min_delay = min_delay.reshape(-1,1)
    max_delay = max_delay.reshape(-1,1)
    thermal_noise_window = NP.abs(clags) >= max_abs_delay*1e-6
    thermal_noise_window = NP.repeat(thermal_noise_window, ia.baselines[truncated_ref_bl_ind,:].shape[0], axis=0)
    EoR_window = NP.logical_or(clags > max_delay+3/bw, clags < min_delay-3/bw)
    strict_EoR_window = NP.logical_and(EoR_window, NP.abs(clags) < 1/coarse_channel_resolution)
    wedge_window = NP.logical_and(clags <= max_delay, clags >= min_delay)
    non_wedge_window = NP.logical_not(wedge_window)

    small_delays_EoR_window = EoR_window.T
    small_delays_strict_EoR_window = strict_EoR_window.T
    small_delays_wedge_window = wedge_window.T

    if max_abs_delay is not None:
        small_delays_ind = NP.abs(clean_lags) <= max_abs_delay * 1e-6
        clean_lags = clean_lags[small_delays_ind]
        small_delays_EoR_window = small_delays_EoR_window[small_delays_ind,:]
        small_delays_strict_EoR_window = small_delays_strict_EoR_window[small_delays_ind,:]
        small_delays_wedge_window = small_delays_wedge_window[small_delays_ind,:]

    if plot_17:
        # 17) Plot delay spectra of the MWA tile power pattern using a uniform sky model

        delta_array_usm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = delta_array_usm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        delta_array_usm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = delta_array_usm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        delta_array_usm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(delta_array_usm_cc_skyvis, axis=1),axes=1) * delta_array_usm_cc_skyvis.shape[1] * freq_resolution
        delta_array_usm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(delta_array_usm_cc_skyvis_res, axis=1),axes=1) * delta_array_usm_cc_skyvis.shape[1] * freq_resolution
        delta_array_usm_cc_skyvis_lag = delta_array_usm_cc_skyvis_lag + delta_array_usm_ccres_sky
        
        delta_array_usm_cc_skyvis_lag = DSP.downsampler(delta_array_usm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
        delta_array_usm_cc_skyvis_lag = delta_array_usm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
    
        delta_array_usm_dspec_max = NP.abs(delta_array_usm_cc_skyvis_lag).max()
        delta_array_usm_dspec_min = NP.abs(delta_array_usm_cc_skyvis_lag).min()
        # delta_array_usm_dspec_max = delta_array_usm_dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2
        # delta_array_usm_dspec_min = delta_array_usm_dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2

        if max_abs_delay is not None:
            delta_array_usm_cc_skyvis_lag = delta_array_usm_cc_skyvis_lag[:,small_delays_ind,:]

        fig, axs = PLT.subplots(n_snaps, sharex=True, sharey=True, figsize=(6,6))
        for j in xrange(n_snaps):
            imdspec = axs[j].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(delta_array_usm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e5**2) * volfactor1 * volfactor2 * Jy2K**2, vmax=(delta_array_usm_dspec_max**2) * volfactor1 * volfactor2 * Jy2K**2))
            horizonb = axs[j].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[j].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
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
        cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        
        # PLT.tight_layout()
        fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(top=0.88)
    
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/delta_array_multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_usm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_no_pfb_{0:.1f}'.format(oversampling_factor)+'.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/delta_array_multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_usm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_no_pfb_{0:.1f}'.format(oversampling_factor)+'.eps', bbox_inches=0)
        
    ##################################################################

    if plot_18:
        # 18) Plot delay spectra of the all-sky model with dipole, MWA tile, and HERA dish antenna shapes

        mwa_dipole_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = mwa_dipole_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        mwa_dipole_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = mwa_dipole_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        mwa_dipole_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(mwa_dipole_asm_cc_skyvis, axis=1),axes=1) * mwa_dipole_asm_cc_skyvis.shape[1] * freq_resolution
        mwa_dipole_asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(mwa_dipole_asm_cc_skyvis_res, axis=1),axes=1) * mwa_dipole_asm_cc_skyvis.shape[1] * freq_resolution
        mwa_dipole_asm_cc_skyvis_lag = mwa_dipole_asm_cc_skyvis_lag + mwa_dipole_asm_ccres_sky
        
        mwa_dipole_asm_cc_skyvis_lag = DSP.downsampler(mwa_dipole_asm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)

        delta_array_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = delta_array_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        delta_array_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = delta_array_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        delta_array_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(delta_array_asm_cc_skyvis, axis=1),axes=1) * delta_array_asm_cc_skyvis.shape[1] * freq_resolution
        delta_array_asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(delta_array_asm_cc_skyvis_res, axis=1),axes=1) * delta_array_asm_cc_skyvis.shape[1] * freq_resolution
        delta_array_asm_cc_skyvis_lag = delta_array_asm_cc_skyvis_lag + delta_array_asm_ccres_sky
        
        delta_array_asm_cc_skyvis_lag = DSP.downsampler(delta_array_asm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)

        hera_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = hera_asm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        hera_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = hera_asm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        hera_asm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(hera_asm_cc_skyvis, axis=1),axes=1) * hera_asm_cc_skyvis.shape[1] * freq_resolution
        hera_asm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(hera_asm_cc_skyvis_res, axis=1),axes=1) * hera_asm_cc_skyvis.shape[1] * freq_resolution
        hera_asm_cc_skyvis_lag = hera_asm_cc_skyvis_lag + hera_asm_ccres_sky
        
        hera_asm_cc_skyvis_lag = DSP.downsampler(hera_asm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)

        delta_array_asm_cc_skyvis_lag = delta_array_asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
        mwa_dipole_asm_cc_skyvis_lag = mwa_dipole_asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
        hera_asm_cc_skyvis_lag = hera_asm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]

        if max_abs_delay is not None:
            delta_array_asm_cc_skyvis_lag = delta_array_asm_cc_skyvis_lag[:,small_delays_ind,:]
            mwa_dipole_asm_cc_skyvis_lag = mwa_dipole_asm_cc_skyvis_lag[:,small_delays_ind,:]
            hera_asm_cc_skyvis_lag = hera_asm_cc_skyvis_lag[:,small_delays_ind,:]

        antelem_asm_dspec_max = max([NP.abs(mwa_dipole_asm_cc_skyvis_lag).max(), NP.abs(delta_array_asm_cc_skyvis_lag).max(), NP.abs(hera_asm_cc_skyvis_lag).max()])
        antelem_asm_dspec_min = min([NP.abs(mwa_dipole_asm_cc_skyvis_lag).min(), NP.abs(delta_array_asm_cc_skyvis_lag).min(), NP.abs(hera_asm_cc_skyvis_lag).min()])
        # antelem_asm_dspec_max = antelem_asm_dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2
        # antelem_asm_dspec_min = antelem_asm_dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2

        delta_array_roifile = '/data3/t_nithyanandan/project_MWA/roi_info_delta_array_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
        delta_array_roi = RI.ROI_parameters(init_file=delta_array_roifile)
        delta_array_telescope = delta_array_roi.telescope

        mwa_dipole_roifile = '/data3/t_nithyanandan/project_MWA/roi_info_mwa_dipole_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
        mwa_dipole_roi = RI.ROI_parameters(init_file=mwa_dipole_roifile)
        mwa_dipole_telescope = mwa_dipole_roi.telescope

        hera_roifile = '/data3/t_nithyanandan/project_MWA/roi_info_hera_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz'.format(Tsys, freq/1e6, nchan*freq_resolution/1e6)+'.fits'
        hera_roi = RI.ROI_parameters(init_file=hera_roifile)
        hera_telescope = hera_roi.telescope

        backdrop_xsize = 100
        xmin = -180.0
        xmax = 180.0
        ymin = -90.0
        ymax = 90.0
    
        xgrid, ygrid = NP.meshgrid(NP.linspace(xmax, xmin, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
        xvect = xgrid.ravel()
        yvect = ygrid.ravel()
    
        delta_array_pb_snapshots = []
        mwa_dipole_pb_snapshots = []
        hera_pb_snapshots = []

        for i in xrange(n_snaps):
            havect = lst[i] - xvect
            altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
            dircos = GEOM.altaz2dircos(altaz, units='degrees')
            roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
            az = altaz[:,1] + 0.0
            az[az > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
            roi_sector_altaz = NP.asarray(NP.where(NP.logical_or(NP.logical_and(az[roi_altaz] >= -0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < -0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors), NP.logical_and(az[roi_altaz] >= 180.0 - 0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < 180.0 - 0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors)))).ravel()
            pb = NP.empty(xvect.size)
            pb.fill(NP.nan)
        
            pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=mwa_dipole_telescope, skyunits='altaz', freq_scale='Hz', pointing_info=mwa_dipole_roi.pinfo[i])
            mwa_dipole_pb_snapshots += [pb.copy()]

            pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=delta_array_telescope, skyunits='altaz', freq_scale='Hz', pointing_info=delta_array_roi.pinfo[i])
            delta_array_pb_snapshots += [pb.copy()]

            pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=hera_telescope, skyunits='altaz', freq_scale='Hz', pointing_info=hera_roi.pinfo[i])
            hera_pb_snapshots += [pb.copy()]

        for j in xrange(n_snaps):
            fig, axs = PLT.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(10.5,2))
            mwa_dipole_pbsky = axs[0].imshow(mwa_dipole_pb_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-5, vmax=1.0), cmap=CM.jet)
            axs[0].set_xlim(xvect.max(), xvect.min())
            axs[0].set_ylim(yvect.min(), yvect.max())
            axs[0].grid(True, which='both')
            axs[0].set_aspect('auto')
            axs[0].tick_params(which='major', length=12, labelsize=12)
            axs[0].tick_params(which='minor', length=6)
            axs[0].locator_params(axis='x', nbins=5)
            axs[0].text(0.5, 0.87, 'Dipole', transform=axs[0].transAxes, fontsize=14, weight='semibold', ha='center', color='black')

            delta_array_pbsky = axs[1].imshow(delta_array_pb_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-5, vmax=1.0), cmap=CM.jet)
            axs[1].set_xlim(xvect.max(), xvect.min())
            axs[1].set_ylim(yvect.min(), yvect.max())
            axs[1].grid(True, which='both')
            axs[1].set_aspect('auto')
            axs[1].tick_params(which='major', length=12, labelsize=12)
            axs[1].tick_params(which='minor', length=6)
            axs[1].locator_params(axis='x', nbins=5)
            axs[1].text(0.5, 0.87, 'Phased Array', transform=axs[1].transAxes, fontsize=14, weight='semibold', ha='center', color='black')

            hera_pbsky = axs[2].imshow(hera_pb_snapshots[j].reshape(-1,backdrop_xsize), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), norm=PLTC.LogNorm(vmin=1e-5, vmax=1.0), cmap=CM.jet)
            axs[2].set_xlim(xvect.max(), xvect.min())
            axs[2].set_ylim(yvect.min(), yvect.max())
            axs[2].grid(True, which='both')
            axs[2].set_aspect('auto')
            axs[2].tick_params(which='major', length=12, labelsize=12)
            axs[2].tick_params(which='minor', length=6)
            axs[2].locator_params(axis='x', nbins=5)
            axs[2].text(0.5, 0.87, 'Dish', transform=axs[2].transAxes, fontsize=14, weight='semibold', ha='center', color='black')

            cbax = fig.add_axes([0.9, 0.24, 0.02, 0.7])
            cbar = fig.colorbar(hera_pbsky, cax=cbax, orientation='vertical')

            fig.subplots_adjust(hspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium', labelpad=25)
            big_ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium', labelpad=15)
    
            PLT.tight_layout()
            fig.subplots_adjust(right=0.89)
            fig.subplots_adjust(top=0.94)
            fig.subplots_adjust(bottom=0.24)
        
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'_no_delayerr_snapshot_{0:0d}.png'.format(j), bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/powerpattern_'+ground_plane_str+snapshot_type_str+obs_mode+'_no_delayerr_snapshot_{0:0d}.eps'.format(j), bbox_inches=0)

        for j in xrange(n_snaps):
            fig, axs = PLT.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(10.5,5))

            mwa_dipole_imdspec = axs[0].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(mwa_dipole_asm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=(antelem_asm_dspec_max**2) * volfactor1 * volfactor2 * Jy2K**2))
            horizonb = axs[0].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[0].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            axs[0].set_xlim(truncated_ref_bl_length.min(), truncated_ref_bl_length.max())
            axs[0].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[0].set_aspect('auto')
            axs[0].text(0.5, 0.9, 'Dipole', transform=axs[0].transAxes, fontsize=14, weight='semibold', ha='center', color='white')

            delta_array_imdspec = axs[1].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(delta_array_asm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=(antelem_asm_dspec_max**2) * volfactor1 * volfactor2 * Jy2K**2))
            horizonb = axs[1].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[1].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            axs[1].set_xlim(truncated_ref_bl_length.min(), truncated_ref_bl_length.max())
            axs[1].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[1].set_aspect('auto')
            axs[1].text(0.5, 0.9, 'Phased Array', transform=axs[1].transAxes, fontsize=14, weight='semibold', ha='center', color='white')

            hera_imdspec = axs[2].pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(hera_asm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e5)**2 * volfactor1 * volfactor2 * Jy2K**2, vmax=(antelem_asm_dspec_max**2) * volfactor1 * volfactor2 * Jy2K**2))
            horizonb = axs[2].plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = axs[2].plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            axs[2].set_xlim(truncated_ref_bl_length.min(), truncated_ref_bl_length.max())
            axs[2].set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            axs[2].set_aspect('auto')
            axs[2].text(0.5, 0.9, 'Dish', transform=axs[2].transAxes, fontsize=14, weight='semibold', ha='center', color='white')

            mwa_dipole_axs_kperp = axs[0].twiny()
            mwa_dipole_axs_kperp.set_xticks(kperp(axs[0].get_xticks()*freq/FCNST.c, redshift))
            mwa_dipole_axs_kperp.set_xlim(kperp(NP.asarray(axs[0].get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            mwa_dipole_axs_kperp.xaxis.set_major_formatter(xformatter)
                
            delta_array_axs_kperp = axs[1].twiny()
            delta_array_axs_kperp.set_xticks(kperp(axs[1].get_xticks()*freq/FCNST.c, redshift))
            delta_array_axs_kperp.set_xlim(kperp(NP.asarray(axs[1].get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            delta_array_axs_kperp.xaxis.set_major_formatter(xformatter)

            hera_axs_kperp = axs[2].twiny()
            hera_axs_kperp.set_xticks(kperp(axs[2].get_xticks()*freq/FCNST.c, redshift))
            hera_axs_kperp.set_xlim(kperp(NP.asarray(axs[2].get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            hera_axs_kperp.xaxis.set_major_formatter(xformatter)

            axs_kprll = axs[2].twinx()
            axs_kprll.set_yticks(kprll(axs[2].get_yticks()*1e-6, redshift))
            axs_kprll.set_ylim(kprll(NP.asarray(axs[2].get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            axs_kprll.yaxis.set_major_formatter(yformatter)

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
            cbar = fig.colorbar(hera_imdspec, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
    
            PLT.tight_layout()
            fig.subplots_adjust(right=0.8)
            fig.subplots_adjust(top=0.85)
            fig.subplots_adjust(left=0.1)
            fig.subplots_adjust(bottom=0.125)
        
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_antenna_multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_no_pfb_{0:.1f}_snapshot_{1:0d}'.format(oversampling_factor,j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/multi_antenna_multi_baseline_CLEAN_noiseless_PS_'+ground_plane_str+snapshot_type_str+obs_mode+'_gaussian_FG_model_asm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_no_pfb_{0:.1f}_snapshot_{1:0d}'.format(oversampling_factor,j)+'.eps', bbox_inches=0)
        
##################################################################

    if plot_19:
        # 19) Plot delay spectrum of uniform sky model with a uniform power pattern

        delta_usm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:] = delta_usm_cc_skyvis[simdata_neg_bl_orientation_ind,:,:].conj()
        delta_usm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:] = delta_usm_cc_skyvis_res[simdata_neg_bl_orientation_ind,:,:].conj()
        
        delta_usm_cc_skyvis_lag = NP.fft.fftshift(NP.fft.ifft(delta_usm_cc_skyvis, axis=1),axes=1) * delta_usm_cc_skyvis.shape[1] * freq_resolution
        delta_usm_ccres_sky = NP.fft.fftshift(NP.fft.ifft(delta_usm_cc_skyvis_res, axis=1),axes=1) * delta_usm_cc_skyvis.shape[1] * freq_resolution
        delta_usm_cc_skyvis_lag = delta_usm_cc_skyvis_lag + delta_usm_ccres_sky
        
        delta_usm_cc_skyvis_lag = DSP.downsampler(delta_usm_cc_skyvis_lag, 1.0*clean_lags_orig.size/ia.lags.size, axis=1)
        delta_usm_cc_skyvis_lag = delta_usm_cc_skyvis_lag[truncated_ref_bl_ind,:,:]
    
        delta_usm_dspec_max = NP.abs(delta_usm_cc_skyvis_lag).max()
        delta_usm_dspec_min = NP.abs(delta_usm_cc_skyvis_lag).min()
        # delta_usm_dspec_max = delta_usm_dspec_max**2 * volfactor1 * volfactor2 * Jy2K**2
        # delta_usm_dspec_min = delta_usm_dspec_min**2 * volfactor1 * volfactor2 * Jy2K**2

        if max_abs_delay is not None:
            delta_usm_cc_skyvis_lag = delta_usm_cc_skyvis_lag[:,small_delays_ind,:]

        for j in xrange(n_snaps):
            fig = PLT.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            imdspec = ax.pcolorfast(truncated_ref_bl_length, 1e6*clean_lags, NP.abs(delta_usm_cc_skyvis_lag[:-1,:-1,j].T)**2 * volfactor1 * volfactor2 * Jy2K**2, norm=PLTC.LogNorm(vmin=(1e5**2) * volfactor1 * volfactor2 * Jy2K**2, vmax=(delta_usm_dspec_max**2) * volfactor1 * volfactor2 * Jy2K**2))
            horizonb = ax.plot(truncated_ref_bl_length, 1e6*min_delay.ravel(), color='white', ls=':', lw=1.5)
            horizont = ax.plot(truncated_ref_bl_length, 1e6*max_delay.ravel(), color='white', ls=':', lw=1.5)
            ax.set_ylim(0.9*NP.amin(clean_lags*1e6), 0.9*NP.amax(clean_lags*1e6))
            ax.set_aspect('auto')
            # ax.text(0.5, 0.9, descriptor_str[j], transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    
            ax_kprll = ax.twinx()
            ax_kprll.set_yticks(kprll(ax.get_yticks()*1e-6, redshift))
            ax_kprll.set_ylim(kprll(NP.asarray(ax.get_ylim())*1e-6, redshift))
            yformatter = FuncFormatter(lambda y, pos: '{0:.2f}'.format(y))
            ax_kprll.yaxis.set_major_formatter(yformatter)

            ax_kperp = ax.twiny()
            ax_kperp.set_xticks(kperp(ax.get_xticks()*freq/FCNST.c, redshift))
            ax_kperp.set_xlim(kperp(NP.asarray(ax.get_xlim())*freq/FCNST.c, redshift))
            xformatter = FuncFormatter(lambda x, pos: '{0:.3f}'.format(x))
            ax_kperp.xaxis.set_major_formatter(xformatter)
    
            ax.set_ylabel(r'$\tau$ [$\mu$s]', fontsize=16, weight='medium')
            ax.set_xlabel(r'$|\mathbf{b}|$ [m]', fontsize=16, weight='medium')
            ax_kprll.set_ylabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')
            ax_kperp.set_xlabel(r'$k_\perp$ [$h$ Mpc$^{-1}$]', fontsize=16, weight='medium')

            cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
            cbar = fig.colorbar(imdspec, cax=cbax, orientation='vertical')
            cbax.set_xlabel(r'K$^2$(Mpc/h)$^3$', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            
            # PLT.tight_layout()
            fig.subplots_adjust(right=0.72)
            fig.subplots_adjust(top=0.88)
    
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/delta_multi_baseline_CLEAN_noiseless_PS_no_ground_'+snapshot_type_str+obs_mode+'_gaussian_FG_model_usm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_no_pfb_{0:.1f}_snapshot_{1:0d}'.format(oversampling_factor,j)+'.png', bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/delta_multi_baseline_CLEAN_noiseless_PS_no_ground_'+snapshot_type_str+obs_mode+'_gaussian_FG_model_usm'+sky_sector_str+'nside_{0:0d}_'.format(nside)+'Tsys_{0:.1f}K_{1:.1f}_MHz_{2:.1f}_MHz_'.format(Tsys, freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_no_pfb_{0:.1f}_snapshot_{1:0d}'.format(oversampling_factor,j)+'.eps', bbox_inches=0)
