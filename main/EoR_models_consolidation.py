import numpy as NP
import glob
from astropy.io import fits, ascii
import scipy.constants as FCNST
import constants as CNST
import ipdb as PDB

rootdir = '/data3/t_nithyanandan/EoR_models/'
model = '21cmFAST' # 'Lidz' or '21cmFAST'
modeldir = model + '/'
versiondir = 'v2/'
outfile = rootdir + modeldir + versiondir + 'PS_model.fits'

if model == 'Lidz':
    fullfnames = glob.glob(rootdir + modeldir + versiondir + '*.dat')
elif model == '21cmFAST':
    fullfnames = glob.glob(rootdir + modeldir + versiondir + 'ps_*')
else:
    raise ValueError('This EoR model is currently not implemented')
fullfnames = NP.asarray(fullfnames)
fnames = [fname.split('/')[-1] for fname in fullfnames]
fnames = NP.asarray(fnames)
if model == 'Lidz':
    redshifts_str = [fname.split('_')[2].split('z')[1].split('.dat')[0] for fname in fnames]
elif model == '21cmFAST':
    redshifts_str = [fname.split('_')[3].split('z')[1] for fname in fnames]
else:
    raise ValueError('This EoR model is currently not implemented')
redshifts = NP.asarray(map(float, redshifts_str))
sortind = NP.argsort(redshifts)
fullfnames = fullfnames[sortind]
fnames = fnames[sortind]
redshifts = redshifts[sortind]
freqs = CNST.rest_freq_HI / (1+redshifts)

hdulist = []
hdulist += [fits.PrimaryHDU()]
hdulist[0].header['EXTNAME'] = 'PRIMARY'
hdulist[0].header['NZ'] = (redshifts.size, 'Number of redshifts')
hdulist[0].header['NFREQ'] = (freqs.size, 'Number of frequencies')
hdulist[0].header['SOURCE'] = (model, 'Source of model')
hdulist[0].header['KUNIT'] = ('h/Mpc', 'Units of k')
hdulist[0].header['PKUNIT'] = ('K**2 (Mpc/h)**3', 'Units of P(k)')
hdulist[0].header['D2UNIT'] = ('K**2', 'Units of delta**2')
hdulist[0].header['FREQUNIT'] = ('Hz', 'Units of frequency')
hdulist[0].header['T0'] = (28e-3, 'T_CMB (K) at z=0, dT = T_0 ((1+z)/10)**0.5')

hdulist += [fits.ImageHDU(redshifts, name='REDSHIFT')]
hdulist += [fits.ImageHDU(freqs, name='FREQUENCY')]

for fi, fname in enumerate(fullfnames):
    ps_info = NP.loadtxt(fname, comments='#', usecols=(0,1))
    k = ps_info[:,0].astype(float)
    col2 = ps_info[:,1].astype(float)
    if model == 'Lidz':
        p_k = col2
        temp0 = 28 * NP.sqrt((1+redshifts[fi])/10.0) * 1e-3   # in K
        p_k = p_k * temp0 ** 2
        delta2 = k**3 * p_k / (2 * NP.pi**2)
    elif model == '21cmFAST':
        delta2 = col2 * 1e-6   # in K**2 since col2 is in mK**2
        p_k = 2 * NP.pi**2 * delta2 / k**3
    else:
        raise ValueError('This EoR model is currently not implemented')

    ps_arr = NP.hstack((k.reshape(-1,1), p_k.reshape(-1,1), delta2.reshape(-1,1)))
    hdulist += [fits.ImageHDU(ps_arr, name='{0:0d}'.format(fi))]
    hdulist[-1].header['FREQ'] = (freqs[fi], 'frequency (Hz)')
    hdulist[-1].header['Z'] = (redshifts[fi], 'redshift')
    hdulist[-1].header['COL1'] = ('k', 'h/Mpc')
    hdulist[-1].header['COL2'] = ('P(k)', 'K**2 (Mpc/h)**3')
    hdulist[-1].header['COL3'] = ('delta^2(k)', 'K**2')

outhdu = fits.HDUList(hdulist)
outhdu.writeto(outfile, clobber=True)
        
