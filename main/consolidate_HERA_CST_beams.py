import numpy as NP
import healpy as HP
from astropy.io import fits
import glob

rootdir = '/data3/t_nithyanandan/project_HERA/'
beams_dir = 'power_patterns/CST/mdl04/'
prefix = 'X4Y2H_4900_'
suffix = '.hmap'
pols = ['X']

colnum = 0
outdata = []
schemes = []
nsides = []
npixs = []
frequencies = []
for pol in pols:
    fullfnames = glob.glob(rootdir + beams_dir + prefix + '*' + suffix)
    fullfnames = NP.asarray(fullfnames)
    fnames = [fname.split('/')[-1] for fname in fullfnames]
    fnames = NP.asarray(fnames)
    freqs_str = [fname.split('_')[2].split('.')[0] for fname in fnames]
    freqs = NP.asarray(map(float, freqs_str)) * 1e6 # Convert to Hz
    sortind = NP.argsort(freqs)
    fullfnames = fullfnames[sortind]
    fnames = fnames[sortind]
    freqs = freqs[sortind]
    frequencies += [freqs]

    beam = None
    for fname in fullfnames:
        inhdulist = fits.open(fname)
        hdu1 = inhdulist[1]
        data = hdu1.data.field(colnum)
        data = data.flatten()
        if not data.dtype.isnative:
            data.dtype = data.dtype.newbyteorder()
            data.byteswap(True)
        scheme = hdu1.header['ORDERING'][:4]

        if beam is None:
            beam = data.reshape(-1,1)
        else:
            beam = NP.hstack((beam, data.reshape(-1,1)))

        beam = beam / NP.max(beam, axis=0, keepdims=True)

    outdata += [beam]
    schemes += [scheme]
    npixs += [beam[:,0].size]
    nsides += [HP.npix2nside(beam[:,0].size)]

outfile = rootdir + beams_dir + '{0[0]}_{0[1]}'.format(fnames[0].split('_')[:2])+suffix
hdulist = []
hdulist += [fits.PrimaryHDU()]
hdulist[0].header['EXTNAME'] = 'PRIMARY'
hdulist[0].header['NPOL'] = (len(pols), 'Number of polarizations')
hdulist[0].header['SOURCE'] = ('HERA-CST', 'Source of data')

for pi,pol in enumerate(pols):
    hdu = fits.ImageHDU(outdata[pi], name='BEAM_{0}'.format(pol))
    hdu.header['PIXTYPE'] = ('HEALPIX', 'Type of pixelization')
    hdu.header['ORDERING'] = (schemes[pi], 'Pixel ordering scheme, either RING or NESTED')
    hdu.header['NSIDE'] = (nsides[pi], 'NSIDE parameter of HEALPIX')
    hdu.header['NPIX'] = (npixs[pi], 'Number of HEALPIX pixels')
    hdu.header['FIRSTPIX'] = (0, 'First pixel # (0 based)')
    hdu.header['LASTPIX'] = (npixs[pi]-1, 'Last pixel # (0 based)')
    hdulist += [hdu]
    hdulist += [fits.ImageHDU(frequencies[pi], name='FREQS_{0}'.format(pol))]

outhdu = fits.HDUList(hdulist)
outhdu.writeto(outfile, clobber=True)
