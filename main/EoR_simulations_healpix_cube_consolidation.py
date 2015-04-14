import glob
import multiprocessing as MP
import healpy as HP
import numpy as NP
from astropy.io import fits
from astropy.coordinates import Galactic, FK5
from astropy import units
import progressbar as PGB
import itertools as IT
import ipdb as PDB

def process_healpix(filename, out_nside):
    print 'Process {0} on {1}'.format(MP.current_process().name, filename)
    pixval = HP.read_map(filename, verbose=False)
    nside = HP.npix2nside(pixval.size)
    out_pixres = HP.nside2resol(out_nside)
    print 'Before: ', pixval.min(), pixval.max(), NP.mean(pixval), NP.std(pixval)
    if nside > out_nside:
        if nside/out_nside > 4:
            pixval = HP.ud_grade(pixval, 4*out_nside)
            nside = 4*out_nside
        pix_smoothed = HP.smoothing(pixval, fwhm=out_pixres, regression=False, verbose=False)
    pix_resampled = HP.ud_grade(pix_smoothed, out_nside)
    print 'After: ', pixval.min(), pixval.max(), NP.mean(pix_resampled), NP.std(pix_resampled)
    return pix_resampled

def unwrap_process_healpix(arg):
    return process_healpix(*arg)

indir = '/data3/piyanat/model/21cm/healpix/'
infile_prefix = 'hpx_interp_delta_21cm_l128_'

out_nside = 256

outdir = '/data3/t_nithyanandan/EoR_simulations/Adam_Lidz/Boom_tiles/'

freq_resolution = 80.0  # in kHz, currently the only accepted value
if freq_resolution != 80.0:
    raise ValueError('Currently frequency resolution can only be set to 80 kHz')

infiles = glob.glob(indir+infile_prefix+'*.fits')

infiles_freq = [float(infile.split('/')[-1][len(infile_prefix):len(infile_prefix)+7]) for infile in infiles]
sortind_infiles_freq = sorted(range(len(infiles_freq)), key=infiles_freq.__getitem__)
# sortind_infiles_freq = sorted(range(len(infiles_freq)), key=lambda i: infiles_freq[i])

infiles = NP.asarray(infiles)
infiles_freq = NP.asarray(infiles_freq)
infiles = infiles[sortind_infiles_freq]
infiles_freq = infiles_freq[sortind_infiles_freq]

infiles = infiles
infiles_freq = infiles_freq
nproc = max(MP.cpu_count()/2-1, 1)
chunksize = int(ceil(len(infiles)/float(nproc)))
pool = MP.Pool(processes=nproc)
resampled_pixvals = pool.map(unwrap_process_healpix, IT.izip(infiles, IT.repeat(out_nside)))
# resampled_pixvals = pool.map(unwrap_process_healpix, IT.izip(infiles, IT.repeat(out_nside)), chunksize=chunksize)
# resampled_pixvals = pool.imap(unwrap_process_healpix, IT.izip(infiles, IT.repeat(out_nside)), chunksize=chunksize)

outfile1 = outdir + 'hpxextn_{0:.3f}-{1:.3f}_MHz_{2:.1f}_kHz_nside_{3:0d}.fits'.format(infiles_freq.min(), infiles_freq.max(), freq_resolution, out_nside)
outfile2 = outdir + 'hpxcube_{0:.3f}-{1:.3f}_MHz_{2:.1f}_kHz_nside_{3:0d}.fits'.format(infiles_freq.min(), infiles_freq.max(), freq_resolution, out_nside)

theta, phi = HP.pix2ang(out_nside, NP.arange(HP.nside2npix(out_nside)))
gc = Galactic(l=NP.degrees(phi), b=90.0-NP.degrees(theta), unit=(units.degree, units.degree))
radec = gc.fk5
ra = radec.ra.degree
dec = radec.dec.degree

hdulist = []
hduprimary = fits.PrimaryHDU()
hduprimary.header.set('EXTNAME', 'PRIMARY')
hduprimary.header.set('FITSTYPE', 'BINTABLE')
hduprimary.header['NSIDE'] = (out_nside, 'NSIDE')
hduprimary.header['PIXAREA'] = (HP.nside2pixarea(out_nside), 'pixel solid angle (steradians)')
hduprimary.header['NEXTEN'] = (len(infiles)+2, 'Number of extensions')
hdulist += [hduprimary]
hdu = fits.HDUList(hdulist)
hdu.writeto(outfile1, clobber=True)

pos_cols = []
pos_cols += [fits.Column(name='l', format='D', array=gc.l.degree)]
pos_cols += [fits.Column(name='b', format='D', array=gc.b.degree)]
pos_cols += [fits.Column(name='RA', format='D', array=ra)]
pos_cols += [fits.Column(name='DEC', format='D', array=dec)]
pos_columns = fits.ColDefs(pos_cols, ascii=False)
pos_tbhdu = fits.new_table(pos_columns)
pos_tbhdu.header.set('EXTNAME', 'COORDINATE')
fits.append(outfile1, pos_tbhdu.data, pos_tbhdu.header, verify=False)

freqcol = [fits.Column(name='Frequency [MHz]', format='D', array=infiles_freq)]
freq_columns = fits.ColDefs(freqcol, ascii=False)
freq_tbhdu = fits.new_table(freq_columns)
freq_tbhdu.header.set('EXTNAME', 'FREQUENCY')
fits.append(outfile1, freq_tbhdu.data, freq_tbhdu.header, verify=False)

hdulist = []
hduprimary = fits.PrimaryHDU()
hduprimary.header.set('EXTNAME', 'PRIMARY')
hduprimary.header.set('NEXTEN', 3)
hduprimary.header.set('FITSTYPE', 'IMAGE')
hduprimary.header['NSIDE'] = (out_nside, 'NSIDE')
hduprimary.header['PIXAREA'] = (HP.nside2pixarea(out_nside), 'pixel solid angle (steradians)')
hduprimary.header['NEXTEN'] = (3, 'Number of extensions')
hdulist += [hduprimary]
hdu = fits.HDUList(hdulist)
hdu.writeto(outfile2, clobber=True)
fits.append(outfile2, pos_tbhdu.data, pos_tbhdu.header, verify=False)
fits.append(outfile2, freq_tbhdu.data, freq_tbhdu.header, verify=False)

hpxcube = None
progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=infiles.size).start()
for i, pix_resampled in enumerate(resampled_pixvals):
    cols = []
    cols += [fits.Column(name='Temperature', format='D', array=pix_resampled)]
    columns = fits.ColDefs(cols, ascii=False)
    tbhdu = fits.new_table(columns)
    tbhdu.header.set('EXTNAME', '{0:.3f} MHz'.format(infiles_freq[i]))
    fits.append(outfile1, tbhdu.data, tbhdu.header, verify=False)
    print 'Appended healpix pixels at {0:.3f} MHz'.format(infiles_freq[i])

    if i == 0:
        hpxcube = pix_resampled.reshape(-1,1)
    else:
        hpxcube = NP.hstack((hpxcube, pix_resampled.reshape(-1,1)))

    progress.update(i+1)
progress.finish()

imghdu = fits.ImageHDU(hpxcube, name='TEMPERATURE')
fits.append(outfile2, imghdu.data, imghdu.header, verify=False)



    
