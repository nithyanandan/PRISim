import yaml
import argparse
import numpy as NP 
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.coordinates import Galactic, FK5, SkyCoord
import scipy.constants as FCNST
import progressbar as PGB
import healpy as HP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import catalog as SM
import constants as CNST
import geometry as GEOM
import ipdb as PDB

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to manage sky models')

input_group = parser.add_argument_group('Input parameters', 'Input specifications')
input_group.add_argument('-i', '--infile', dest='infile', default='/home/t_nithyanandan/codes/mine/python/interferometry/main/skymodel_parameters.yaml', type=file, required=False, help='File specifying input parameters')

args = vars(parser.parse_args())

with args['infile'] as parms_file:
    parms = yaml.safe_load(parms_file)

rootdir = parms['directory']['rootdir']
skydir = parms['directory']['skydir']
inp_skymodels = parms['skyparm']['models']
use_csm_bright = parms['skyparm']['csm_bright']
csm_coord = parms['skyparm']['csm_coord']
dsm_coord = parms['skyparm']['dsm_coord']
dsm_type = parms['skyparm']['dsm_type']
csm_type = parms['skyparm']['csm_type']
dsm_units = parms['skyparm']['dsm_units']
csm_units = parms['skyparm']['csm_units']
dsm_pixres = parms['skyparm']['dsm_pixres']
csm_pixres = parms['skyparm']['csm_pixres']
out_coord = parms['skyparm']['out_coord']
out_nside = parms['skyparm']['out_nside']
out_units = parms['skyparm']['out_units']
DSM_dir = parms['catalog']['DSM_dir']
SUMSS_file = parms['catalog']['SUMSS_file']
NVSS_file = parms['catalog']['NVSS_file']
MWACS_file = parms['catalog']['MWACS_file']
GLEAM_file = parms['catalog']['GLEAM_file']
PS_file = parms['catalog']['PS_file']
freqs = parms['specparms']['freqs']
freq_units = parms['specparms']['freq_units']
csm_spindex = parms['specparms']['csm_spindex']
outdir = parms['outparms']['outdir']
outfile_type = parms['outparms']['out_type']
out_parse_freq = parms['outparms']['out_freq']
out_parse_model = parms['outparms']['out_model']

if not isinstance(rootdir, str):
    raise TypeError('rootdir must be a string')

if not isinstance(skydir, str):
    raise TypeError('skydir must be a string')

if not isinstance(DSM_dir, str):
    raise TypeError('DSM_dir must be a string')

if not isinstance(outdir, str):
    raise TypeError('outdir must be a string')
    
if isinstance(inp_skymodels, str):
    inp_skymodels = [inp_skymodels]
else:
    if not isinstance(inp_skymodels, list):
        raise TypeError('Input sky models must be a string or a list of strings')

inp_skymodels = list(set(inp_skymodels))
for sky_str in inp_skymodels:
    if sky_str not in ['csm', 'CSM', 'dsm', 'DSM']:
        raise ValueError('Invalid specification found for input sky model')

if ('csm' in inp_skymodels) or ('CSM' in inp_skymodels):
    if not isinstance(use_csm_bright, bool):
        raise TypeError('csm_bright parameter must be boolean')

    if not isinstance(csm_coord, str):
        raise TypeError('Point source coordinate system must be a string')
    elif csm_coord not in ['galactic', 'equatorial']:
        raise ValueError('Invalid coordinate system specified for point source model')

    if not isinstance(csm_type, str):
        raise TypeError('Point source model format must be a string')
    elif csm_type not in ['healpix', 'other']:
        raise ValueError('Invalid format specified for point source model')

    if not isinstance(csm_units, str):
        raise TypeError('Point source model units must be a string')
    elif csm_units not in ['K', 'Jy']:
        raise ValueError('Invalid units specified for point source model')
    
    if csm_type != 'healpix':
        if csm_units == 'K':
            if not isinstance(csm_pixres, (int,float)):
                raise TypeError('Point source model pixel resolution must be scalar')
            elif csm_pixres <= 0.0:
                raise ValueError('Invalid pixel resolution specified for point source model')

if ('dsm' in inp_skymodels) or ('DSM' in inp_skymodels):

    if not isinstance(dsm_coord, str):
        raise TypeError('Diffuse model coordinate system must be a string')
    elif dsm_coord not in ['galactic', 'equatorial']:
        raise ValueError('Invalid coordinate system specified for diffuse sky model')

    if not isinstance(dsm_type, str):
        raise TypeError('diffuse model format must be a string')
    elif dsm_type not in ['healpix', 'other']:
        raise ValueError('Invalid format specified for diffuse model')

    if not isinstance(dsm_units, str):
        raise TypeError('diffuse model units must be a string')
    elif dsm_units not in ['K', 'Jy']:
        raise ValueError('Invalid units specified for diffuse model')
    
    if dsm_type != 'healpix':
        if dsm_units == 'K':
            if not isinstance(dsm_pixres, (int,float)):
                raise TypeError('diffuse model pixel resolution must be scalar')
            elif dsm_pixres <= 0.0:
                raise ValueError('Invalid pixel resolution specified for diffuse model')

if not isinstance(out_coord, str):
    raise TypeError('Output coordinate system must be a string')
elif out_coord not in ['galactic', 'equatorial']:
    raise ValueError('Invalid coordinate system specified for output')

if not isinstance(out_nside, int):
    raise TypeError('nside for HEALPIX output must be an integer')
elif not HP.isnsideok(out_nside):
    raise ValueError('Invalid nside specified for HEALPIX output')

if not isinstance(out_units, str):
    raise TypeError('Output model units must be a string')
elif out_units not in ['K', 'Jy']:
    raise ValueError('Invalid units specified for output model')

if not isinstance(freqs, (int,float,list)):
    raise TypeError('Frequencies must be a scalar or a list')
else:
    freqs = NP.asarray(freqs).reshape(-1)

if not isinstance(freq_units, str):
    raise TypeError('Frequency units must be a string')
elif freq_units not in ['Hz','hz','HZ','mhz','MHZ','MHz','ghz','GHZ','GHz']:
    raise ValueError('Invalid units specified for frequency')
else:
    if freq_units in ['mhz', 'MHz', 'MHZ']:
        freqs *= 1e6
    elif freq_units in ['ghz', 'GHz', 'GHZ']:
        freqs *= 1e9

if csm_spindex is not None:
    if not isinstance(csm_spindex, (int, float)):
        raise TypeError('Point source model mean spectral index must be a scalar')

if not isinstance(outfile_type, str):
    raise TypeError('Output file format must be a string')
elif outfile_type not in ['fits', 'ascii']:
    raise ValueError('Invalid file format specified for output')

if not isinstance(out_parse_freq, str):
    raise TypeError('out_parse_freq must be a string')
elif out_parse_freq not in ['combine', 'separate']:
    raise ValueError('Invalid specification for out_parse_freq')

if not isinstance(out_parse_model, str):
    raise TypeError('out_parse_model must be a string')
elif out_parse_model not in ['combine', 'separate']:
    raise ValueError('Invalid specification for out_parse_model')

dsm_out = NP.empty((HP.nside2npix(out_nside), freqs.size))
csm_out = NP.empty((HP.nside2npix(out_nside), freqs.size))

if ('dsm' in inp_skymodels) or ('DSM' in inp_skymodels):
    if dsm_type == 'healpix':
        dsm_size = None
        dsm_equal_size = True
        for chan, freq in enumerate(freqs):
            dsm_file = rootdir+skydir+DSM_dir+'dsm_{0:.1f}_MHz.txt'.format(freq/1e6)
            indata = NP.loadtxt(dsm_file)
            in_nside = HP.get_nside(indata)
            outdata = NP.copy(indata)
            if out_nside < in_nside:
                fwhm = HP.nside2resol(out_nside)
                outdata = HP.smoothing(outdata, fwhm=fwhm, regression=False)
            if out_nside != in_nside:
                if dsm_units == 'Jy':
                    outdata = HP.ud_grade(outdata, out_nside, power=-2)
                else:
                    outdata = HP.ud_grade(outdata, out_nside)
            dsm_out[:,chan] = outdata
        if out_units != dsm_units:
            pixres = HP.nside2pixarea(out_nside)
            if out_units == 'K':
                dsm_out = dsm_out * CNST.Jy * (FCNST.c / freqs.reshape(1,-1))**2 / pixres / (2.0 * FCNST.k)
            else:
                dsm_out = 2.0 * FCNST.k * dsm_out * pixres * (freqs.reshape(1,-1) / FCNST.c)**2 / CNST.Jy

        if out_coord != dsm_coord:
            theta, phi = HP.pix2ang(out_nside, NP.arange(HP.nside2npix(out_nside)))
            if dsm_coord == 'galactic':
                gc = SkyCoord(l=NP.degrees(phi), b=90.0-NP.degrees(theta), unit='deg', frame='galactic')
                ec = gc.transform_to('icrs')
                echpx = SkyCoord(ra=NP.degrees(phi), dec=90.0-NP.degrees(theta), unit='deg', frame='icrs')
                m1, m2, d12 = GEOM.spherematch(echpx.ra.degree, echpx.dec.degree, ec.ra.degree, ec.dec.degree, matchrad=NP.degrees(HP.nside2resol(out_nside)), nnearest=1, maxmatches=1)
            else:
                ec = SkyCoord(ra=NP.degrees(phi), dec=90.0-NP.degrees(theta), unit='deg', frame='icrs')
                gc = ec.transform_to('galactic')
                gchpx = SkyCoord(l=NP.degrees(phi), b=90.0-NP.degrees(theta), unit='deg', frame='galactic')
                m1, m2, d12 = GEOM.spherematch(gchpx.l.degree, gchpx.b.degree, gc.l.degree, gc.b.degree, matchrad=NP.degrees(HP.nside2resol(out_nside)), nnearest=1, maxmatches=1)
                
            temp_dsm_out = NP.copy(dsm_out)
            temp_dsm_out[m1,:] = dsm_out[m2,:]
            dsm_out = NP.copy(temp_dsm_out)
            del temp_dsm_out
    else:
        for chan, freq in enumerate(freqs):
            dsm_file = rootdir+skydir+DSM_dir+'dsm_{0:.1f}_MHz.txt'.format(freq/1e6)
            indata = NP.loadtxt(dsm_file)
            theta, phi = HP.pix2ang(out_nside, NP.arange(indata.size))
            if dsm_coord == 'galactic':
                gc = SkyCoord(l=NP.degrees(phi), b=90.0-NP.degrees(theta), unit='deg', frame='galactic')
                ec = gc.transform_to('icrs')
            else:
                ec = SkyCoord(ra=NP.degrees(phi), dec=90.0-NP.degrees(theta), unit='deg', frame='icrs')
                
            spec_parms = None
            spec_type = 'spectrum'
            dsm_inspec = NP.asarray(dsm_inspec)
            catlabel = NP.repeat('DSM', dsm_inspec.shape[0])
            
            dsm_in = SM.SkyModel(catlabel, freqs[chan], NP.hstack((ec.ra.degree.reshape(-1,1), ec.dec.degree.reshape(-1,1))), spec_type, spectrum=indata, spec_parms=None)
            dsm_out_dict = dsm_in.to_healpix(freqs[chan], out_nside, in_units=dsm_units, dsm_out_coords=out_coord, out_units=out_units)
            dsm_out[:,chan] = dsm_out_dict['spectrum']

if ('csm' in inp_skymodels) or ('CSM' in inp_skymodels):

    freq_SUMSS = 0.843 # in GHz
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
    spindex_SUMSS = csm_spindex + NP.zeros(fint.size)

    fmajax = fmajax[PS_ind]
    fminax = fminax[PS_ind]
    fpa = fpa[PS_ind]
    dmajax = dmajax[PS_ind]
    dminax = dminax[PS_ind]
    bright_source_ind = fint >= 10.0 * (freq_SUMSS*1e9/150e6)**spindex_SUMSS
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
    freq_NVSS = 1.4 # in GHz
    hdulist = fits.open(NVSS_file)
    ra_deg_NVSS = hdulist[1].data['RA(2000)']
    dec_deg_NVSS = hdulist[1].data['DEC(2000)']
    nvss_fpeak = hdulist[1].data['PEAK INT']
    nvss_majax = hdulist[1].data['MAJOR AX']
    nvss_minax = hdulist[1].data['MINOR AX']
    hdulist.close()

    spindex_NVSS = csm_spindex + NP.zeros(nvss_fpeak.size)

    not_in_SUMSS_ind = dec_deg_NVSS > -30.0
    # not_in_SUMSS_ind = NP.logical_and(dec_deg_NVSS > -30.0, dec_deg_NVSS <= min(90.0, latitude+90.0))
    PS_ind = NP.sqrt(nvss_majax**2-(0.75/60.0)**2) < 14.0/3.6e3
    valid_ind, = NP.where(not_in_SUMSS_ind & PS_ind)
    if use_csm_bright:
        bright_source_ind = nvss_fpeak >= 10.0 * (freq_NVSS*1e9/150e6)**(spindex_NVSS)
        valid_ind, = NP.where(NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind))
    count_valid = valid_ind.size

    nvss_fpeak = nvss_fpeak[valid_ind]
    freq_catalog = NP.concatenate((freq_catalog, freq_NVSS*1e9 + NP.zeros(count_valid)))
    catlabel = NP.concatenate((catlabel, NP.repeat('NVSS',count_valid)))
    ra_deg = NP.concatenate((ra_deg, ra_deg_NVSS[valid_ind]))
    dec_deg = NP.concatenate((dec_deg, dec_deg_NVSS[valid_ind]))
    spindex = NP.concatenate((spindex, spindex_NVSS[valid_ind]))
    majax = NP.concatenate((majax, nvss_majax[valid_ind]))
    minax = NP.concatenate((minax, nvss_minax[valid_ind]))
    fluxes = NP.concatenate((fluxes, nvss_fpeak))

    csm_units = 'Jy'
    spec_parms = {}
    spec_type = 'func'
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)
    psskymod = SM.SkyModel(catlabel, freqs, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), spec_type, spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fluxes.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])
    csmhpxinfo = psskymod.to_healpix(freqs, out_nside, in_units=csm_units, out_coords=out_coord, out_units=out_units)
    csm_out = csmhpxinfo['spectrum']

if out_parse_model == 'combine':
    asm_out = dsm_out + csm_out
    outfile_prefix = 'asm'
    if out_parse_freq == 'combine':
        outfile = rootdir + skydir + outdir + outfile_prefix + '_healpix_{0:0d}_{1:.1f}-{2:.1f}_MHz'.format(out_nside, freqs.min()/1e6, freqs.max()/1e6)
        if outfile_type == 'ascii':
            out_dict = {}
            colnames = []
            colfrmts = {}
            for chan, f in enumerate(freqs):
                out_dict['{0:.1f}_MHz'.format(f/1e6)] = asm_out[:,chan]
                colnames += ['{0:.1f}_MHz'.format(f/1e6)]
                colfrmts['{0:.1f}_MHz'.format(f/1e6)] = '%0.5f'

            tbdata = Table(out_dict, names=colnames)
            ascii.write(tbdata, output=outfile+'.txt', format='fixed_width_two_line', formats=colfrmts, bookend=False, delimiter='|', delimiter_pad=' ')
        else:
            hdulist = []
            hdulist += [fits.PrimaryHDU()]
            hdulist[0].header['EXTNAME'] = 'PRIMARY'
            hdulist[0].header['NSIDE'] = out_nside
            hdulist[0].header['UNTIS'] = out_units
            hdulist[0].header['NFREQ'] = freqs.size
            hdulist[0].header['FREQUNIT'] = 'Hz'
            hdulist += [fits.ImageHDU(asm_out, name='SPECTRUM')]
            hdulist += [fits.ImageHDU(freqs, name='FREQUENCIES')]
            hdu = fits.HDUList(hdulist)
            hdu.writeto(outfile+'.fits', clobber=True)
    else:
        for chan, f in enumerate(freqs):
            outfile = rootdir + skydir + outdir + outfile_prefix + '_healpix_{0:0d}_{1:.1f}_MHz'.format(out_nside, f/1e6)
            if outfile_type == 'ascii':
                out_dict = {}
                colnames = []
                colfrmts = {}
                
                out_dict['{0:.1f}_MHz'.format(f/1e6)] = asm_out[:,chan]
                colnames += ['{0:.1f}_MHz'.format(f/1e6)]
                colfrmts['{0:.1f}_MHz'.format(f/1e6)] = '%0.5f'

                tbdata = Table(out_dict, names=colnames)
                ascii.write(tbdata, output=outfile+'.txt', format='fixed_width_two_line', formats=colfrmts, bookend=False, delimiter='|', delimiter_pad=' ')
            else:
                hdulist = []
                hdulist += [fits.PrimaryHDU()]
                hdulist[0].header['EXTNAME'] = 'PRIMARY'
                hdulist[0].header['NSIDE'] = out_nside
                hdulist[0].header['UNTIS'] = out_units
                hdulist[0].header['FREQ'] = f
                hdulist[0].header['FREQUNIT'] = 'Hz'
                hdulist += [fits.ImageHDU(asm_out[:,chan], name='SPECTRUM')]
                hdu = fits.HDUList(hdulist)
                hdu.writeto(outfile+'.fits', clobber=True)
else:
    for model in inp_skymodels:
        if model in ['csm', 'CSM']:
            outspec = csm_out
            outfile_prefix = 'csm'
        elif model in ['dsm', 'DSM']:
            outspec = dsm_out
            outfile_prefix = 'dsm'

        if out_parse_freq == 'combine':
            outfile = rootdir + skydir + outdir + outfile_prefix + '_healpix_{0:0d}_{1:.1f}-{2:.1f}_MHz'.format(out_nside, freqs.min()/1e6, freqs.max()/1e6)
            if outfile_type == 'ascii':
                out_dict = {}
                colnames = []
                colfrmts = {}
                for chan, f in enumerate(freqs):
                    out_dict['{0:.1f}_MHz'.format(f/1e6)] = outspec[:,chan]
                    colnames += ['{0:.1f}_MHz'.format(f/1e6)]
                    colfrmts['{0:.1f}_MHz'.format(f/1e6)] = '%0.5f'
    
                tbdata = Table(out_dict, names=colnames)
                ascii.write(tbdata, output=outfile+'.txt', format='fixed_width_two_line', formats=colfrmts, bookend=False, delimiter='|', delimiter_pad=' ')
            else:
                hdulist = []
                hdulist += [fits.PrimaryHDU()]
                hdulist[0].header['EXTNAME'] = 'PRIMARY'
                hdulist[0].header['NSIDE'] = out_nside
                hdulist[0].header['UNTIS'] = out_units
                hdulist[0].header['NFREQ'] = freqs.size
                hdulist[0].header['FREQUNIT'] = 'Hz'
                hdulist += [fits.ImageHDU(outspec, name='SPECTRUM')]
                hdulist += [fits.ImageHDU(freqs, name='FREQUENCIES')]
                hdu = fits.HDUList(hdulist)
                hdu.writeto(outfile+'.fits', clobber=True)
        else:
            for chan, f in enumerate(freqs):
                outfile = rootdir + skydir + outdir + outfile_prefix + '_healpix_{0:0d}_{1:.1f}_MHz'.format(out_nside, f/1e6)
                if outfile_type == 'ascii':
                    out_dict = {}
                    colnames = []
                    colfrmts = {}
                    
                    out_dict['{0:.1f}_MHz'.format(f/1e6)] = outspec[:,chan]
                    colnames += ['{0:.1f}_MHz'.format(f/1e6)]
                    colfrmts['{0:.1f}_MHz'.format(f/1e6)] = '%0.5f'
    
                    tbdata = Table(out_dict, names=colnames)
                    ascii.write(tbdata, output=outfile+'.txt', format='fixed_width_two_line', formats=colfrmts, bookend=False, delimiter='|', delimiter_pad=' ')
                else:
                    hdulist = []
                    hdulist += [fits.PrimaryHDU()]
                    hdulist[0].header['EXTNAME'] = 'PRIMARY'
                    hdulist[0].header['NSIDE'] = out_nside
                    hdulist[0].header['UNTIS'] = out_units
                    hdulist[0].header['FREQ'] = f
                    hdulist[0].header['FREQUNIT'] = 'Hz'
                    hdulist += [fits.ImageHDU(outspec[:,chan], name='SPECTRUM')]
                    hdu = fits.HDUList(hdulist)
                    hdu.writeto(outfile+'.fits', clobber=True)

dsmg = HP.cartview(dsm_out[:,0], coord='G', return_projected_map=True)
csmg = HP.cartview(csm_out[:,0], coord='G', return_projected_map=True)
asmg = HP.cartview(asm_out[:,0], coord='G', return_projected_map=True)

n_fg_ticks = 5
fg_ticks = NP.round(NP.logspace(NP.log10(dsmg.min()), NP.log10(asmg.max()), n_fg_ticks)).astype(NP.int)

fig, axs = PLT.subplots(nrows=3, sharex=True, sharey=True, figsize=(5,7))
# csmimg = axs[0].imshow(csmg, origin='lower', extent=[180,-180,-90,90], vmin=dsmg.min(), vmax=asmg.max())
# dsmimg = axs[1].imshow(dsmg, origin='lower', extent=[180,-180,-90,90], vmin=dsmg.min(), vmax=asmg.max())
# asmimg = axs[2].imshow(asmg, origin='lower', extent=[180,-180,-90,90], vmin=dsmg.min(), vmax=asmg.max())
csmimg = axs[0].imshow(csmg, origin='lower', extent=[180,-180,-90,90], norm=PLTC.LogNorm(vmin=dsmg.min(), vmax=asmg.max()))
dsmimg = axs[1].imshow(dsmg, origin='lower', extent=[180,-180,-90,90], norm=PLTC.LogNorm(vmin=dsmg.min(), vmax=asmg.max()))
asmimg = axs[2].imshow(asmg, origin='lower', extent=[180,-180,-90,90], norm=PLTC.LogNorm(vmin=dsmg.min(), vmax=asmg.max()))
axs[0].set_xlim(180,-180)
axs[1].set_xlim(180,-180)
axs[2].set_xlim(180,-180)
axs[0].set_ylim(-90,90)
axs[1].set_ylim(-90,90)
axs[2].set_ylim(-90,90)
axs[0].set_aspect('auto')
axs[1].set_aspect('auto')
axs[2].set_aspect('auto')
axs[2].set_xlabel('GLon. [degrees]')
axs[1].set_ylabel('GLat. [degrees]')
fig.subplots_adjust(hspace=0, wspace=0)
fig.subplots_adjust(top=0.9, left=0.15)
cbax = fig.add_axes([0.15, 0.95, 0.75, 0.02])
cbar = fig.colorbar(asmimg, cax=cbax, orientation='horizontal')
cbar.set_ticks(fg_ticks.tolist())
cbar.set_ticklabels(fg_ticks.tolist())
cbax.set_xlabel('Temperature [K]')
cbax.xaxis.set_label_position('top')
PLT.show()

PDB.set_trace()
