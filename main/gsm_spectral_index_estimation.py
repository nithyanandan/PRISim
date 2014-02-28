import numpy as NP
import healpy as HP
from astropy.io import fits
from astropy.coordinates import Galactic, FK5
from astropy import units
import foregrounds as FG

freqs = [140.0e6, 150.0e6, 160.0e6] # frequencies in Hz
outfile = '/data3/t_nithyanandan/project_MWA/foregrounds/gsmdata.fits'

nsides = []
for i in xrange(len(freqs)):
    gsm_file = '/data3/t_nithyanandan/project_MWA/foregrounds/gsm{0:.0f}.txt'.format(freqs[i]/1e6)
    gsm_inp = NP.loadtxt(gsm_file)
    print gsm_inp.min()
    nside = HP.npix2nside(gsm_inp.size)
    gsm_smoothed = HP.smoothing(gsm_inp, fwhm=NP.radians(0.85), regression=False)
    gsm_downsampled = HP.ud_grade(gsm_smoothed, nside/8)
    # gsm_downsampled = HP.ud_grade(gsm_inp, nside/8)
    gsm_downsampled = gsm_downsampled.reshape(-1,1)
    nsides += [HP.npix2nside(gsm_downsampled.size)]
    if i == 0:
        gsm = gsm_downsampled
    else:
        gsm = NP.hstack((gsm, gsm_downsampled))

spindex = FG.power_law_spectral_index(freqs, gsm)

theta, phi = HP.pix2ang(nsides[0], NP.arange(gsm.shape[0]))
gc = Galactic(l=NP.degrees(phi), b=90.0-NP.degrees(theta), unit=(units.degree, units.degree))
radec = gc.fk5
ra = radec.ra.degree
dec = radec.dec.degree

hdulist = []
hdulist += [fits.PrimaryHDU()]
hdulist[0].header['NSIDE'] = (nsides[0], 'NSIDE')
hdulist[0].header['PIXAREA'] = (HP.nside2pixarea(nsides[0]), 'pixel solid angle (steradians)')
cols = []
cols += [fits.Column(name='l', format='D', array=gc.l.degree)]
cols += [fits.Column(name='b', format='D', array=gc.b.degree)]
cols += [fits.Column(name='RA', format='D', array=ra)]
cols += [fits.Column(name='DEC', format='D', array=dec)]
cols += [fits.Column(name='T_{0:.0f}'.format(freqs[i]/1e6), format='D', array=gsm[:,i]) for i in xrange(len(freqs))]
cols += [fits.Column(name='spindex', format='D', array=spindex)]
columns = fits.ColDefs(cols, tbtype='BinTableHDU')
tbhdu = fits.new_table(columns)
tbhdu.header.set('EXTNAME', 'GSM')
hdulist += [tbhdu]
hdu = fits.HDUList(hdulist)
hdu.writeto(outfile, clobber=True)




