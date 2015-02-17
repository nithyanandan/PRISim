import numpy as NP
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
from mwapy.pb import mwa_tile
import geometry as GEOM
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import ipdb as PDB

# freq = 182e6 # center frequency in Hz

# pointing_file = '/data3/t_nithyanandan/project_MWA/selected_obsids.txt'
# pointing_info_from_file = NP.loadtxt(pointing_file, skiprows=1, comments='#', usecols=(1,2,3), delimiter=',')
# obsids = NP.loadtxt(pointing_file, skiprows=1, comments='#', usecols=(0,), delimiter=',', dtype=str)
# delays_str = NP.loadtxt(pointing_file, skiprows=1, comments='#', usecols=(4,), delimiter=',', dtype=str)
# delays_list = [NP.fromstring(delaystr, dtype=float, sep=';', count=-1) for delaystr in delays_str]
# delay_settings = NP.asarray(delays_list)
# delay_settings *= 435e-12

# lvect = NP.linspace(-1.0, 1.0, 2001)
# mvect = NP.linspace(-1.0, 1.0, 2001)
# lgrid, mgrid = NP.meshgrid(lvect, mvect)
# lmvect = NP.hstack((lgrid.reshape(-1,1), mgrid.reshape(-1,1)))
# valid_ind = NP.sum(lmvect**2, axis=1) <= 1.0
# nvect = NP.empty(lmvect.shape[0])
# nvect.fill(NP.nan)
# nvect[valid_ind] = NP.sqrt(1.0 - NP.sum(lmvect[valid_ind,:]**2, axis=1))

# lmnvect = NP.hstack((lmvect, nvect.reshape(-1,1)))
# altaz_vect = NP.empty((lgrid.size, 2))
# altaz_vect.fill(NP.nan)

# altaz_vect[valid_ind] = GEOM.dircos2altaz(lmnvect[valid_ind], units='radians')

# tile_pbx_adv = NP.empty((lgrid.size,obsids.size))
# tile_pbx_adv.fill(NP.nan)
# tile_pby_adv = NP.empty((lgrid.size,obsids.size))
# tile_pby_adv.fill(NP.nan)

# tile_pbx_theory = NP.empty((lgrid.size,obsids.size))
# tile_pbx_theory.fill(NP.nan)
# tile_pby_theory = NP.empty((lgrid.size,obsids.size))
# tile_pby_theory.fill(NP.nan)

# dipole_pbx_adv = NP.empty(lgrid.size)
# dipole_pbx_adv.fill(NP.nan)
# dipole_pby_adv = NP.empty(lgrid.size)
# dipole_pby_adv.fill(NP.nan)

# dipole_pbx_theory = NP.empty(lgrid.size)
# dipole_pbx_theory.fill(NP.nan)
# dipole_pby_theory = NP.empty(lgrid.size)
# dipole_pby_theory.fill(NP.nan)

# adv_dipole = mwa_tile.Dipole(type='lookup')
# short_dipole = mwa_tile.Dipole(type='short')

# j = adv_dipole.getJones(NP.pi/2-altaz_vect[valid_ind,0].reshape(-1,1), altaz_vect[valid_ind,1].reshape(-1,1), freq)
# vis = mwa_tile.makeUnpolInstrumentalResponse(j,j)
# pbx, pby = (vis[:,:,0,0].real, vis[:,:,1,1].real)
# dipole_pbx_adv[valid_ind] = pbx
# dipole_pby_adv[valid_ind] = pby

# j = short_dipole.getJones(NP.pi/2-altaz_vect[valid_ind,0].reshape(-1,1), altaz_vect[valid_ind,1].reshape(-1,1), freq)
# vis = mwa_tile.makeUnpolInstrumentalResponse(j,j)
# pbx, pby = (vis[:,:,0,0].real, vis[:,:,1,1].real)
# dipole_pbx_theory[valid_ind] = pbx
# dipole_pby_theory[valid_ind] = pby

# for i in xrange(obsids.size):
#     pbx, pby = MWAPB.MWA_Tile_advanced(NP.pi/2-altaz_vect[valid_ind,0].reshape(-1,1), altaz_vect[valid_ind,1].reshape(-1,1), freq=freq, delays=delay_settings[i,:]/435e-12)
#     tile_pbx_adv[valid_ind,i] = pbx.ravel()
#     tile_pby_adv[valid_ind,i] = pby.ravel()

#     pbx, pby = MWAPB.MWA_Tile_analytic(NP.pi/2-altaz_vect[valid_ind,0].reshape(-1,1), altaz_vect[valid_ind,1].reshape(-1,1), freq=freq, delays=delay_settings[i,:]/435e-12, power=True)
#     tile_pbx_theory[valid_ind,i] = pbx.ravel()
#     tile_pby_theory[valid_ind,i] = pby.ravel()

adv_outfile = '/data3/t_nithyanandan/project_MWA/MWA_Tools_pb_advanced_{0:.1f}_MHz_for_Ian.fits'.format(freq/1e6)
theory_outfile = '/data3/t_nithyanandan/project_MWA/MWA_Tools_pb_analytic_{0:.1f}_MHz_for_Ian.fits'.format(freq/1e6)

hdulist1 = []
hdulist1 += [fits.PrimaryHDU()]
hdulist2 = []
hdulist2 += [fits.PrimaryHDU()]

cols = []
cols += [fits.Column(name='l', format='D', array=lvect)]
cols += [fits.Column(name='m', format='D', array=mvect)]
columns = fits.ColDefs(cols, tbtype='BinTableHDU')
tbhdu = fits.new_table(columns)
tbhdu.header.set('EXTNAME', 'COORDS')

hdulist1 += [tbhdu]
hdulist2 += [tbhdu]

for i in xrange(obsids.size):
    hdulist1 += [fits.ImageHDU(NP.dstack((tile_pbx_adv[:,i].reshape(lgrid.shape),tile_pby_adv[:,i].reshape(lgrid.shape))), name=obsids[i])]
    hdulist2 += [fits.ImageHDU(NP.dstack((tile_pbx_theory[:,i].reshape(lgrid.shape),tile_pby_theory[:,i].reshape(lgrid.shape))), name=obsids[i])]

hdu1 = fits.HDUList(hdulist1)
hdu2 = fits.HDUList(hdulist2)
hdu1.writeto(adv_outfile, clobber=True)
hdu2.writeto(theory_outfile, clobber=True)





