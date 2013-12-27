import numpy as NP
from astropy.io import fits
import geometry as GEOM
import interferometry as RI
import catalog as CTLG

catalog_file = '/Users/t_nithyanandan/Downloads/first_13jun05.fits'
# catalog_file = '/Users/t_nithyanandan/Downloads/catalog_13jun05.bin'

# catinfo = NP.loadtxt(catalog_file, comments='#', skiprows=2, usecols=(1,2,3,4,5,6,7,8,10,14,15,16))

# catinfo = fits.open(catalog_file)
# catdata = catinfo[1].data
# cols = catinfo[1].columns

catdata, cathdr = fits.getdata(catalog_file, 1, header=True)

sl_prob = catdata['SIDEPROB']
majax = catdata['FITTED_MAJOR']
dec_deg = catdata['DEC']

# Reliable data selection

sl_msk = sl_prob < 0.15

# Point source selection

ps_msk = ((majax <= 5.97) & (dec_deg <= 4.0)) | ((majax <= 6.87) & (dec_deg > 4.0))

# Combined data data selection

msk = sl_msk & ps_msk

catdata = catdata[msk]

ra_deg = catdata['RA']
fpeak = catdata['FPEAK']
ferr = catdata['RMS']
minax = catdata['FITTED_MINOR']
pa = catdata['FITTED_POSANG']
sl_prob = sl_prob[msk]
majax = majax[msk]
dec_deg = dec_deg[msk]

freq_catalog = 1.4 # in GHz
freq_resolution = 40.0 # in kHz
nchan = 1024*2
chans = freq_catalog + NP.arange(nchan)*freq_resolution*1e3/1e9
bpass = 1.0*NP.ones(nchan)

ctlgobj = CTLG.Catalog(freq_catalog, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), fpeak/1.0e3)

skymod = CTLG.SkyModel(ctlgobj)

intrfrmtr = RI.Interferometer('B1', [10000.0, 0.0, 0.0], chans, freq_scale='GHz')

Tsys = 100.0 # in Kelvin
t_snap = 180.0 # in seconds
# ha_range = 15.0*NP.arange(-1.0, t_snap/3.6e3, 1.0)
n_snaps = 16
lst_obs = (12.0 + (t_snap / 3.6e3) * NP.arange(n_snaps)) * 15.0 # in degrees
ha_obs = 1.0 * NP.zeros(n_snaps)
dec_obs = intrfrmtr.latitude + NP.zeros(n_snaps)

for i in xrange(n_snaps):
    intrfrmtr.observe(str(lst_obs[i]), Tsys, bpass, [ha_obs[i], dec_obs[i]], skymod, t_snap, fov_radius=0.5, lst=lst_obs[i])

intrfrmtr.delay_transform()
