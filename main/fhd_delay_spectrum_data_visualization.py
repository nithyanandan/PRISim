import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CMAP
import matplotlib.animation as MOV
from matplotlib import ticker
from scipy.interpolate import griddata
import datetime as DT
import time 
import progressbar as PGB
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

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

telescope = 'mwa'
telescope_str = telescope + '_'
if telescope == 'mwa':
    telescope_str = ''

ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ref_bl, ref_bl_id = RI.baseline_generator(ant_locs[:,1:], ant_id=ant_locs[:,0].astype(int).astype(str), auto=False, conjugate=False)
ref_bl_length = NP.sqrt(NP.sum(ref_bl**2, axis=1))
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True) 
neg_bl_orientation_ind = ref_bl_orientation < 0.0
ref_bl[neg_bl_orientation_ind,:] = -1.0 * ref_bl[neg_bl_orientation_ind,:]
ref_bl_orientation = NP.angle(ref_bl[:,0] + 1j * ref_bl[:,1], deg=True)
sortind = NP.argsort(ref_bl_length, kind='mergesort')
ref_bl = ref_bl[sortind,:]
ref_bl_length = ref_bl_length[sortind]
ref_bl_orientation = ref_bl_orientation[sortind]
ref_bl_id = ref_bl_id[sortind]
n_bins_baseline_orientation = 4

latitude = -26.701

fhd_obsid = [1061309344, 1061316544]

freq = 185.0 * 1e6 # foreground center frequency in Hz
freq_resolution = 80e3 # in Hz
bpass_shape = 'bhw'
n_channels = 384
nchan = n_channels
bw = nchan * freq_resolution

n_sky_sectors = 1
sky_sector = None # if None, use all sky sector. Accepted values are None, 0, 1, 2, or 3
if sky_sector is None:
    sky_sector_str = '_all_sky_'
    n_sky_sectors = 1
    sky_sector = 0
else:
    sky_sector_str = '_sky_sector_{0:0d}_'.format(sky_sector)

pointing_file = '/data3/t_nithyanandan/project_MWA/Aug23_obsinfo.txt'
pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(1,2,3), delimiter=',')
obs_id = NP.loadtxt(pointing_file, skiprows=2, comments='#', usecols=(0,), delimiter=',', dtype=str)
lst = 15.0 * pointing_info_from_file[:,2]
pointings_altaz = OPS.reverse(pointing_info_from_file[:,:2].reshape(-1,2), axis=1)
pointings_dircos = GEOM.altaz2dircos(pointings_altaz, units='degrees')
pointings_hadec = GEOM.altaz2hadec(pointings_altaz, latitude, units='degrees')

max_abs_delay = 2.5 # in micro seconds

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

backdrop_xsize = 100
backdrop_coords = 'radec'
if use_DSM or use_GSM:
    backdrop_coords = 'radec'

if backdrop_coords == 'radec':
    xmin = -180.0
    xmax = 180.0
    ymin = -90.0
    ymax = 90.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize/2))
    xvect = xgrid.ravel()
    yvect = ygrid.ravel()
elif backdrop_coords == 'dircos':
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0

    xgrid, ygrid = NP.meshgrid(NP.linspace(xmin, xmax, backdrop_xsize), NP.linspace(ymin, ymax, backdrop_xsize))
    nanind = (xgrid**2 + ygrid**2) > 1.0
    goodind = (xgrid**2 + ygrid**2) <= 1.0
    zgrid = NP.empty_like(xgrid)
    zgrid[nanind] = NP.nan
    zgrid[goodind] = NP.sqrt(1.0 - (xgrid[goodind]**2 + ygrid[goodind]**2))

    xvect = xgrid.ravel()
    yvect = ygrid.ravel()
    zvect = zgrid.ravel()
    xyzvect = NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1), zvect.reshape(-1,1)))

if use_DSM or use_GSM:
    dsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(freq/1e6,nside)
    hdulist = fits.open(dsm_file)
    dsm_table = hdulist[1].data
    ra_deg = dsm_table['RA']
    dec_deg = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(freq/1e6)]
    fluxes = temperatures
    backdrop = HP.cartview(temperatures.ravel(), coord=['G','E'], rot=[0,0,0], xsize=backdrop_xsize, return_projected_map=True)
elif use_GLEAM or use_SUMSS or use_NVSS or use_CSM:
    if use_GLEAM:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/mwacs_b1_131016.csv' # GLEAM catalog
        catdata = ascii.read(catalog_file, data_start=1, delimiter=',')
        dec_deg = catdata['DEJ2000']
        ra_deg = catdata['RAJ2000']
        fpeak = catdata['S150_fit']
        ferr = catdata['e_S150_fit']
        freq_catalog = 1.4 # GHz
        spindex = -0.83 + NP.zeros(fpeak.size)
        fluxes = fpeak * (freq_catalog * 1e9 / freq)**spindex
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
        fluxes = fint * (freq_catalog*1e9/freq)**spindex
    elif use_NVSS:
        pass
    else:
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

    if backdrop_coords == 'radec':
        backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='cubic')
        backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', pointing_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)
else:
    if use_PS:
        catalog_file = '/data3/t_nithyanandan/project_MWA/foregrounds/PS_catalog.txt'
        catdata = ascii.read(catalog_file, comment='#', header_start=0, data_start=1)
        ra_deg = catdata['RA'].data
        dec_deg = catdata['DEC'].data
        fluxes = catdata['F_INT'].data
        
    if backdrop_coords == 'radec':
        ra_deg_wrapped = ra_deg.ravel() + 0.0
        ra_deg_wrapped[ra_deg > 180.0] -= 360.0
        
        dxvect = xgrid[0,1]-xgrid[0,0]
        dyvect = ygrid[1,0]-ygrid[0,0]
        ibind, nnval, distNN = LKP.lookup(ra_deg_wrapped.ravel(), dec_deg.ravel(), fluxes.ravel(), xvect, yvect, distance_ULIM=NP.sqrt(dxvect**2 + dyvect**2), remove_oob=False)
        backdrop = nnval.reshape(backdrop_xsize/2, backdrop_xsize)
        # backdrop = griddata(NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fluxes, NP.hstack((xvect.reshape(-1,1), yvect.reshape(-1,1))), method='nearest')
        # backdrop = backdrop.reshape(backdrop_xsize/2, backdrop_xsize)
    elif backdrop_coords == 'dircos':
        if (telescope == 'mwa_dipole') or (obs_mode == 'drift'):
            backdrop = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, freq_scale='Hz', skyunits='dircos', pointing_center=[0.0,0.0,1.0])
            backdrop = backdrop.reshape(backdrop_xsize, backdrop_xsize)

cardinal_blo = 180.0 / n_bins_baseline_orientation * (NP.arange(n_bins_baseline_orientation)-1).reshape(-1,1)
cardinal_bll = 100.0
cardinal_bl = cardinal_bll * NP.hstack((NP.cos(NP.radians(cardinal_blo)), NP.sin(NP.radians(cardinal_blo)), NP.zeros_like(cardinal_blo)))

pc = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
pc_coords = 'dircos'

for j in range(len(fhd_obsid)):
    fhd_infile = '/data3/t_nithyanandan/project_MWA/fhd_delay_spectrum_{0:0d}_reformatted.npz'.format(fhd_obsid[j])
    fhd_data = NP.load(fhd_infile)
    fhd_bl_id = fhd_data['fhd_bl_id']
    fhd_bl_ind = NP.squeeze(NP.where(NP.in1d(ref_bl_id, fhd_bl_id)))
    bl_id = ref_bl_id[fhd_bl_ind]
    bl = ref_bl[fhd_bl_ind, :]
    bl_length = ref_bl_length[fhd_bl_ind]
    bl_orientation = ref_bl_orientation[fhd_bl_ind]
    fhd_vis_lag_noisy = fhd_data['fhd_vis_lag_noisy']
    fhd_delays = fhd_data['fhd_delays']
    fhd_C = fhd_data['fhd_C']
    valid_ind = NP.logical_and(NP.abs(NP.sum(fhd_vis_lag_noisy[:,:,0],axis=1))!=0.0, NP.abs(NP.sum(fhd_C[:,:,0],axis=1))!=0.0)
    fhd_C = fhd_C[valid_ind,:,:]
    fhd_vis_lag_noisy = fhd_vis_lag_noisy[valid_ind,:,:]
    bl_id = bl_id[valid_ind]
    bl = bl[valid_ind,:]
    bl_length = bl_length[valid_ind]
    bl_orientation = bl_orientation[valid_ind]
    neg_bl_orientation_ind = bl_orientation > 90.0 + 0.5*180.0/n_bins_baseline_orientation
    bl_orientation[neg_bl_orientation_ind] -= 180.0
    bl[neg_bl_orientation_ind,:] = -bl[neg_bl_orientation_ind,:]

    fhd_vis_lag_noisy *= 2.78*nchan*freq_resolution/fhd_C
    fhd_obsid_pointing_dircos = pointings_dircos[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
    fhd_obsid_pointing_altaz = pointings_altaz[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
    fhd_obsid_pointing_hadec = pointings_hadec[obs_id==str(fhd_obsid[j]),:].reshape(1,-1)
    fhd_lst = NP.asscalar(lst[obs_id==str(fhd_obsid[j])])
    fhd_obsid_pointing_radec = NP.copy(fhd_obsid_pointing_hadec)
    fhd_obsid_pointing_radec[0,0] = fhd_lst - fhd_obsid_pointing_hadec[0,0]
    
    delay_matrix = DLY.delay_envelope(bl, fhd_obsid_pointing_dircos, units='mks')
    delaymat = DLY.delay_envelope(bl, pc, units='mks')

    min_delay = -delaymat[0,:,1]-delaymat[0,:,0]
    max_delay = delaymat[0,:,0]-delaymat[0,:,1]
    min_delay = min_delay.reshape(-1,1)
    max_delay = max_delay.reshape(-1,1)

    thermal_noise_window = NP.abs(fhd_delays) >= max_abs_delay*1e-6
    thermal_noise_window = thermal_noise_window.reshape(1,-1)
    thermal_noise_window = NP.repeat(thermal_noise_window, bl.shape[0], axis=0)
    EoR_window = NP.logical_or(fhd_delays > max_delay+1/bw, fhd_delays < min_delay-1/bw)
    wedge_window = NP.logical_and(fhd_delays <= max_delay, fhd_delays >= min_delay)
    fhd_vis_rms_lag = OPS.rms(fhd_vis_lag_noisy[:,:,0], mask=NP.logical_not(thermal_noise_window), axis=1)
    fhd_vis_rms_freq = NP.abs(fhd_vis_rms_lag) / NP.sqrt(nchan) / freq_resolution
    PDB.set_trace()
    print fhd_vis_rms_freq
    
    if max_abs_delay is not None:
        small_delays_ind = NP.abs(fhd_delays) <= max_abs_delay * 1e-6
        fhd_delays = fhd_delays[small_delays_ind]
        fhd_vis_lag_noisy = fhd_vis_lag_noisy[:,small_delays_ind,:]

    # fig = PLT.figure(figsize=(6,8))
    
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('Baseline Index', fontsize=18)
    # ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)
    # # dspec = ax.imshow(NP.abs(fhd_vis_lag_noisy[:,:,0].T), origin='lower', extent=(0, fhd_vis_lag_noisy.shape[0]-1, NP.amin(fhd_delays*1e6), NP.amax(fhd_delays*1e6)), interpolation=None)
    # dspec = ax.imshow(NP.abs(fhd_vis_lag_noisy[:,:,0].T), origin='lower', extent=(0, fhd_vis_lag_noisy.shape[0]-1, NP.amin(fhd_delays*1e6), NP.amax(fhd_delays*1e6)), norm=PLTC.LogNorm(1.0e7, vmax=1.0e10), interpolation=None)
    # ax.set_aspect('auto')

    # cbax = fig.add_axes([0.88, 0.08, 0.03, 0.9])
    # cb = fig.colorbar(dspec, cax=cbax, orientation='vertical')
    # cbax.set_ylabel('Jy Hz', labelpad=-60, fontsize=18)
    
    # PLT.tight_layout()
    # fig.subplots_adjust(right=0.8)
    # fig.subplots_adjust(left=0.1)
    
    # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/fhd_multi_baseline_CLEAN_visibilities_{0:0d}.eps'.format(fhd_obsid[j]), bbox_inches=0)
    # PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/fhd_multi_baseline_CLEAN_visibilities_{0:0d}.png'.format(fhd_obsid[j]), bbox_inches=0)

    blo = NP.copy(bl_orientation)
    bloh, bloe, blon, blori = OPS.binned_statistic(blo, statistic='count', bins=n_bins_baseline_orientation, range=[(-90.0+0.5*180.0/n_bins_baseline_orientation, 90.0+0.5*180.0/n_bins_baseline_orientation)])

    if n_bins_baseline_orientation == 4:
        blo_ax_mapping = [7,4,1,2,3,6,9,8]

    overlay = {}
    
    if backdrop_coords == 'radec':
        havect = fhd_lst - xvect
        altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),yvect.reshape(-1,1))), latitude, units='degrees')
        dircos = GEOM.altaz2dircos(altaz, units='degrees')
        roi_altaz = NP.asarray(NP.where(altaz[:,0] >= 0.0)).ravel()
        az = altaz[:,1] + 0.0
        az[az > 360.0 - 0.5*180.0/n_sky_sectors] -= 360.0
        roi_sector_altaz = NP.asarray(NP.where(NP.logical_or(NP.logical_and(az[roi_altaz] >= -0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < -0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors), NP.logical_and(az[roi_altaz] >= 180.0 - 0.5*180.0/n_sky_sectors + sky_sector*180.0/n_sky_sectors, az[roi_altaz] < 180.0 - 0.5*180.0/n_sky_sectors + (sky_sector+1)*180.0/n_sky_sectors)))).ravel()
        pb = NP.empty(xvect.size)
        pb.fill(NP.nan)
        bd = NP.empty(xvect.size)
        bd.fill(NP.nan)
        pb[roi_altaz] = PB.primary_beam_generator(altaz[roi_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_center=fhd_obsid_pointing_altaz)
        # bd[roi_altaz] = backdrop.ravel()[roi_altaz]
        # pb[roi_altaz[roi_sector_altaz]] = PB.primary_beam_generator(altaz[roi_altaz[roi_sector_altaz],:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', phase_center=fhd_obsid_pointing_altaz)
        bd[roi_altaz[roi_sector_altaz]] = backdrop.ravel()[roi_altaz[roi_sector_altaz]]
        overlay['pbeam'] = pb
        overlay['backdrop'] = bd
        overlay['roi_obj_inds'] = roi_altaz
        overlay['roi_sector_inds'] = roi_altaz[roi_sector_altaz]
        overlay['delay_map'] = NP.empty((n_bins_baseline_orientation, xvect.size))
        overlay['delay_map'].fill(NP.nan)
        overlay['delay_map'][:,roi_altaz] = (DLY.geometric_delay(cardinal_bl, altaz[roi_altaz,:], altaz=True, dircos=False, hadec=False, latitude=latitude)-DLY.geometric_delay(cardinal_bl, pc, altaz=False, dircos=True, hadec=False, latitude=latitude)).T
        if use_CSM or use_SUMSS or use_NVSS or use_PS:
            src_hadec = NP.hstack(((fhd_lst-ctlgobj.location[:,0]).reshape(-1,1), ctlgobj.location[:,1].reshape(-1,1)))
            src_altaz = GEOM.hadec2altaz(src_hadec, latitude, units='degrees')
            roi_src_altaz = NP.asarray(NP.where(src_altaz[:,0] >= 0.0)).ravel()
            roi_pbeam = PB.primary_beam_generator(src_altaz[roi_src_altaz,:], freq, telescope=telescope, skyunits='altaz', freq_scale='Hz', pointing_center=fhd_obsid_pointing_altaz)
            overlay['src_ind'] = roi_src_altaz
            overlay['pbeam_on_src'] = roi_pbeam.ravel()

        # delay_envelope = DLY.delay_envelope(cardinal_bl, dircos[roi_altaz,:])
        # overlay['delay_map'][:,roi_altaz] = (DLY.geometric_delay(cardinal_bl, altaz[roi_altaz,:], altaz=True, dircos=False, hadec=False, latitude=latitude)-DLY.geometric_delay(cardinal_bl, fhd_obsid_pointing_altaz, altaz=True, dircos=False, hadec=False, latitude=latitude)).T
        # roi_obj_inds += [roi_altaz]
    elif backdrop_coords == 'dircos':
        havect = fhd_lst - ra_deg
        fg_altaz = GEOM.hadec2altaz(NP.hstack((havect.reshape(-1,1),dec_deg.reshape(-1,1))), latitude, units='degrees')
        fg_dircos = GEOM.altaz2dircos(fg_altaz, units='degrees')
        roi_dircos = NP.asarray(NP.where(fg_dircos[:,2] >= 0.0)).ravel()
        overlay['roi_obj_inds'] = roi_dircos
        overlay['fg_dircos'] = fg_dircos
        if obs_mode == 'track':
            pb = PB.primary_beam_generator(xyzvect, freq, telescope=telescope, skyunits='dircos', freq_scale='Hz', pointing_center=fhd_obsid_pointing_dircos)
            # pb[pb < 0.5] = NP.nan
            overlay['pbeam'] = pb.reshape(backdrop_xsize, backdrop_xsize)
        overlay['delay_map'] = NP.empty((n_bins_baseline_orientation, xyzvect.shape[0])).fill(NP.nan)

    mindelay = NP.nanmin(overlay['delay_map'])
    maxdelay = NP.nanmax(overlay['delay_map'])
    norm_b = PLTC.Normalize(vmin=mindelay, vmax=maxdelay)

    fig = PLT.figure(figsize=(10,10))
    faxs = []
    for i in xrange(n_bins_baseline_orientation):
        ax = fig.add_subplot(3,3,blo_ax_mapping[i])
        ax.set_xlim(0,bloh[i]-1)
        ax.set_ylim(NP.amin(fhd_delays*1e6), NP.amax(fhd_delays*1e6))
        ax.set_title(r'{0:+.1f} <= $\theta_b [deg]$ < {1:+.1f}'.format(bloe[i], bloe[(i)+1]), weight='medium')
        ax.set_ylabel(r'lag [$\mu$s]', fontsize=18)    
        blind = blori[blori[i]:blori[i+1]]
        sortind = NP.argsort(bl_length[blind], kind='heapsort')
        imdspec = ax.imshow(NP.abs(fhd_vis_lag_noisy[blind[sortind],:,0].T), origin='lower', extent=(0, blind.size-1, NP.amin(fhd_delays*1e6), NP.amax(fhd_delays*1e6)), norm=PLTC.LogNorm(vmin=1e5, vmax=5e10), interpolation=None)
        # norm=PLTC.LogNorm(vmin=1e-1, vmax=NP.amax(NP.abs(fhd_vis_lag_noisy))), 
        l = ax.plot([], [], 'k:', [], [], 'k:', [], [], 'k--', [], [], 'k--')
        ax.set_aspect('auto')
        faxs += [ax]
    
        ax = fig.add_subplot(3,3,blo_ax_mapping[i+n_bins_baseline_orientation])
        if backdrop_coords == 'radec':
            ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
            ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
        elif backdrop_coords == 'dircos':
            ax.set_xlabel('l')
            ax.set_ylabel('m')
        imdmap = ax.imshow(1e6 * OPS.reverse(overlay['delay_map'][i,:].reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
        # PDB.set_trace()
        imdmappbc = ax.contour(xgrid[0,:], ygrid[:,0], overlay['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
        # imdmap.set_clim(mindelay, maxdelay)
        ax.set_title(r'$\theta_b$ = {0:+.1f} [deg]'.format(cardinal_blo.ravel()[i]), fontsize=18, weight='medium')
        ax.grid(True)
        ax.tick_params(which='major', length=12, labelsize=12)
        ax.tick_params(which='minor', length=6)
        ax.locator_params(axis='x', nbins=5)
        faxs += [ax]
    
    cbmnt = NP.amin(NP.abs(fhd_vis_lag_noisy))
    cbmxt = NP.amax(NP.abs(fhd_vis_lag_noisy))
    cbaxt = fig.add_axes([0.1, 0.95, 0.8, 0.02])
    cbart = fig.colorbar(imdspec, cax=cbaxt, orientation='horizontal')
    cbaxt.set_xlabel('Jy', labelpad=-50, fontsize=18)
    
    # cbmnb = NP.nanmin(overlays[0]['delay_map']) * 1e6
    # cbmxb = NP.nanmax(overlays[0]['delay_map']) * 1e6
    # cbmnb = mindelay * 1e6
    # cbmxb = maxdelay * 1e6
    cbaxb = fig.add_axes([0.1, 0.06, 0.8, 0.02])
    cbarb = fig.colorbar(imdmap, cax=cbaxb, orientation='horizontal', norm=norm_b)
    cbaxb.set_xlabel(r'x (bl/100) $\mu$s', labelpad=-45, fontsize=18)
    
    ax = fig.add_subplot(3,3,5)
    # imsky1 = ax.imshow(backdrop, origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)))
    impbc = ax.contour(xgrid[0,:], ygrid[:,0], overlay['pbeam'].reshape(-1,backdrop_xsize), levels=[0.0078125, 0.03125, 0.125, 0.5], colors='k')
    if use_CSM or use_NVSS or use_SUMSS or use_PS:
        imsky2 = ax.scatter(ra_deg_wrapped[overlay['src_ind']].ravel(), dec_deg[overlay['src_ind']].ravel(), c=overlay['pbeam_on_src']*fluxes[overlay['src_ind']], norm=PLTC.LogNorm(vmin=1e-3, vmax=1.0), cmap=CMAP.jet, edgecolor='none', s=10)
    else:
        imsky2 = ax.imshow(OPS.reverse((overlay['pbeam']*overlay['backdrop']).reshape(-1,backdrop_xsize), axis=1), origin='lower', extent=(NP.amax(xvect), NP.amin(xvect), NP.amin(yvect), NP.amax(yvect)), alpha=0.85, norm=PLTC.LogNorm(vmin=1e-2, vmax=1e2))        
    ax.set_xlim(xvect.max(), xvect.min())
    ax.set_ylim(yvect.min(), yvect.max())
    ax.set_title('Foregrounds', fontsize=18, weight='medium')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.tick_params(which='major', length=12, labelsize=12)
    ax.tick_params(which='minor', length=6)
    if backdrop_coords == 'radec':
        ax.set_xlabel(r'$\alpha$ [degrees]', fontsize=16, weight='medium')
        ax.set_ylabel(r'$\delta$ [degrees]', fontsize=16, weight='medium')
    elif backdrop_coords == 'dircos':
        ax.set_xlabel('l')
        ax.set_ylabel('m')
    ax.locator_params(axis='x', nbins=5)
    
    cbmnc = NP.nanmin(overlay['pbeam']*overlay['backdrop'])
    cbmxc = NP.nanmax(overlay['pbeam']*overlay['backdrop'])
    cbaxc = fig.add_axes([0.4, 0.35, 0.25, 0.02])
    # cbarc = fig.colorbar(ax.images[1], cax=cbaxc, orientation='horizontal')
    cbarc = fig.colorbar(imsky2, cax=cbaxc, orientation='horizontal')
    if use_GSM or use_DSM:
        cbaxc.set_xlabel('Temperature [K]', labelpad=-50, fontsize=18, weight='medium')
    else:
        cbaxc.set_xlabel('Flux Density [Jy]', labelpad=-50, fontsize=18, weight='medium')
    # tick_locator = ticker.MaxNLocator(nbins=21)
    # cbarc.locator = tick_locator
    # cbarc.update_ticks()
    
    faxs += [ax]
    tpc = faxs[-1].text(0.5, 1.25, r' $\alpha$ = {0[0]:+.3f} deg, $\delta$ = {0[1]:+.2f} deg'.format(fhd_obsid_pointing_radec.ravel()) + '\nLST = {0:.2f} hrs'.format(fhd_lst), transform=ax.transAxes, fontsize=14, weight='medium', ha='center')
    
    PLT.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)

    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'fhd_multi_baseline_CLEAN_visibilities_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_snapshot_{0:0d}.eps'.format(fhd_obsid[j]), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MWA/figures/'+telescope_str+'fhd_multi_baseline_CLEAN_visibilities_gaussian_FG_model_'+fg_str+sky_sector_str+'nside_{0:0d}_'.format(nside)+'{0:.1f}_MHz_{1:.1f}_MHz_'.format(freq/1e6,nchan*freq_resolution/1e6)+bpass_shape+'_snapshot_{0:0d}.png'.format(fhd_obsid[j]), bbox_inches=0)


