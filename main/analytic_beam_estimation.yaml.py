import yaml
import argparse
import numpy as NP
from astropy.io import fits
from astropy.coordinates import Galactic, FK5, SkyCoord
from astropy import units
import progressbar as PGB
import healpy as HP
import geometry as GEOM
import primary_beams as PB
import ipdb as PDB

## Parse input arguments

parser = argparse.ArgumentParser(description='Program to estimate antenna power pattern analytically')

input_group = parser.add_argument_group('Input parameters', 'Input specifications')
input_group.add_argument('-i', '--infile', dest='infile', default='/home/t_nithyanandan/codes/mine/python/interferometry/main/pbparameters.yaml', type=file, required=False, help='File specifying input parameters')

args = vars(parser.parse_args())

with args['infile'] as parms_file:
    parms = yaml.safe_load(parms_file)

rootdir = parms['directory']['rootdir']
pbdir = parms['directory']['pbdir']
outfile = parms['directory']['outfile']
project = parms['project']
telescope_id = parms['telescope']['id']
element_shape = parms['antenna']['shape']
element_size = parms['antenna']['size']
element_ocoords = parms['antenna']['ocoords']
element_orientation = parms['antenna']['orientation']
ground_plane = parms['antenna']['ground_plane']
phased_array = parms['antenna']['phased_array']
short_dipole_approx = parms['antenna']['short_dipole']
half_wave_dipole_approx = parms['antenna']['halfwave_dipole']
phased_elements_file = parms['phasedarray']['file']
delayerr = parms['phasedarray']['delayerr']
gainerr = parms['phasedarray']['gainerr']
nrand = parms['phasedarray']['nrand']
A_eff = parms['telescope']['A_eff']
latitude = parms['telescope']['latitude']
longitude = parms['telescope']['longitude']
beam_info = parms['beam']
beam_id = beam_info['identifier']
beam_pol = beam_info['pol']
freq = parms['obsparm']['freq']
freq_resolution = parms['obsparm']['freq_resolution']
nchan = parms['obsparm']['nchan']
nside = parms['obsparm']['nside']
scheme = parms['obsparm']['ordering']
pnt_alt = parms['pointing']['alt']
pnt_az = parms['pointing']['az']
pnt_ha = parms['pointing']['ha']
pnt_dec = parms['pointing']['dec']
frequency_chunk_size = parms['processing']['freq_chunk_size']
n_freq_chunks = parms['processing']['n_freq_chunks']
nproc = parms['pp']['nproc']
pp_method = parms['pp']['method']
pp_key = parms['pp']['key']

if longitude is None:
    longitude = 0.0

if project not in ['project_MWA', 'project_global_EoR', 'project_HERA', 'project_drift_scan', 'project_beams', 'project_LSTbin']:
    raise ValueError('Invalid project specified')
else:
    project_dir = project + '/'
pbeamdir = rootdir + project_dir + pbdir + '/'

if telescope_id not in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'custom', 'paper_dipole', 'mwa_tools']:
    raise ValueError('Invalid telescope specified')

if element_shape is None:
    element_shape = 'delta'
elif element_shape not in ['dish', 'delta', 'dipole']:
    raise ValueError('Invalid antenna element shape specified')

if element_shape != 'delta':
    if element_size is None:
        raise ValueError('No antenna element size specified')
    elif element_size <= 0.0:
        raise ValueError('Antenna element size must be positive')

if not isinstance(phased_array, bool):
    raise TypeError('phased_array specification must be boolean')

if delayerr is None:
    delayerr_str = ''
    delayerr = 0.0
elif delayerr < 0.0:
    raise ValueError('delayerr must be non-negative.')
else:
    delayerr_str = 'derr_{0:.3f}ns'.format(delayerr)
delayerr *= 1e-9

if gainerr is None:
    gainerr_str = ''
    gainerr = 0.0
elif gainerr < 0.0:
    raise ValueError('gainerr must be non-negative.')
else:
    gainerr_str = '_gerr_{0:.2f}dB'.format(gainerr)

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

if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole'):
    element_size = 0.74
    element_shape = 'dipole'
    if telescope_id == 'mwa': phased_array = True
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
    if element_shape != 'delta':
        if (element_shape is None) or (element_size is None):
            raise ValueError('Both antenna element shape and size must be specified for the custom telescope type.')
        elif element_size <= 0.0:
            raise ValueError('Antenna element size must be positive.')
elif telescope_id == 'mwa_tools':
    pass
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

if element_ocoords not in ['altaz', 'dircos']:
    if element_ocoords is not None:
        raise ValueError('Antenna element orientation must be "altaz" or "dircos"')

if element_orientation is None:
    if element_ocoords is not None:
        if element_ocoords == 'altaz':
            if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole') or (element_shape == 'dipole'):
                element_orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
            else:
                element_orientation = NP.asarray([90.0, 270.0]).reshape(1,-1)
        elif element_ocoords == 'dircos':
            if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole') or (element_shape == 'dipole'):
                element_orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1)
            else:
                element_orientation = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
        else:
            raise ValueError('Invalid value specified antenna element orientation coordinate system.')
    else:
        if (telescope_id == 'mwa') or (telescope_id == 'mwa_dipole') or (element_shape == 'dipole'):
            element_orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
        else:
            element_orientation = NP.asarray([90.0, 270.0]).reshape(1,-1)
        element_ocoords = 'altaz'
else:
    if element_ocoords is None:
        raise ValueError('Antenna element orientation coordinate system must be specified to describe the specified antenna orientation.')

element_orientation = NP.asarray(element_orientation).reshape(1,-1)
if (element_orientation.size < 2) or (element_orientation.size > 3):
    raise ValueError('Antenna element orientation must be a two- or three-element vector.')
elif (element_ocoords == 'altaz') and (element_orientation.size != 2):
    raise ValueError('Antenna element orientation must be a two-element vector if using Alt-Az coordinates.')

if ground_plane is None:
    ground_plane_str = 'no_ground_'
else:
    if ground_plane > 0.0:
        ground_plane_str = '{0:.1f}m_ground_'.format(ground_plane)
    else:
        raise ValueError('Height of antenna element above ground plane must be positive.')

telescope = {}
if telescope_id in ['mwa', 'vla', 'gmrt', 'hera', 'mwa_dipole', 'mwa_tools']:
    telescope['id'] = telescope_id
telescope['shape'] = element_shape
telescope['size'] = element_size
telescope['orientation'] = element_orientation
telescope['ocoords'] = element_ocoords
telescope['groundplane'] = ground_plane
telescope['latitude'] = latitude
telescope['longitude'] = longitude

if (pnt_alt is not None) and (pnt_az is not None):
    pointing_altaz = NP.asarray([pnt_alt, pnt_az])
elif (pnt_ha is not None) and (pnt_dec is not None):
    pointing_hadec = NP.asarray([pnt_ha, pnt_dec]).reshape(1,-1)
    pointing_altaz = GEOM.hadec2altaz(pointing_hadec, latitude, units='degrees')
    pointing_altaz = pointing_altaz.reshape(-1)
else:
    raise ValueError('pointing direction not properly specified')

freq = NP.float(freq)
freq_resolution = NP.float(freq_resolution)
chans = (freq + (NP.arange(nchan) - 0.5 * nchan) * freq_resolution)/ 1e9 # in GHz
bandpass_str = '{0:0d}x{1:.1f}_kHz'.format(nchan, freq_resolution/1e3)

theta, phi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
alt = 90.0 - NP.degrees(theta)
az = NP.degrees(phi)
altaz = NP.hstack((alt.reshape(-1,1), az.reshape(-1,1)))

pinfo = {}
pinfo['pointing_center'] = pointing_altaz
pinfo['pointing_coords'] = 'altaz'

pb = PB.primary_beam_generator(altaz, chans, telescope, freq_scale='GHz', skyunits='altaz', pointing_info=pinfo, short_dipole_approx=short_dipole_approx, half_wave_dipole_approx=half_wave_dipole_approx)

colnum = 0
npix = HP.nside2npix(nside)
frequencies = chans * 1e3

if outfile is not None:
    if not isinstance(outfile, str):
        raise TypeError('outfile parameter must be a string')
else:
    outfile = telescope_str+beam_id+'_'+ground_plane_str+'nside_{0:0d}_'.format(nside)+delaygain_err_str+'{0}_{1:.1f}_MHz'.format(bandpass_str, freq/1e6)+'.fits'

hdulist = []
hdulist += [fits.PrimaryHDU()]
hdulist[0].header['EXTNAME'] = 'PRIMARY'
hdulist[0].header['NPOL'] = (1, 'Number of polarizations')
hdulist[0].header['SOURCE'] = ('ANALYTIC-{0}'.format(beam_id), 'Source of data')

hdu = fits.ImageHDU(pb, name='BEAM_{0}'.format(beam_pol))
hdu.header['PIXTYPE'] = ('HEALPIX', 'Type of pixelization')
hdu.header['ORDERING'] = (scheme, 'Pixel ordering scheme, either RING or NESTED')
hdu.header['NSIDE'] = (nside, 'NSIDE parameter of HEALPIX')
hdu.header['NPIX'] = (npix, 'Number of HEALPIX pixels')
hdu.header['FIRSTPIX'] = (0, 'First pixel # (0 based)')
hdu.header['LASTPIX'] = (npix-1, 'Last pixel # (0 based)')
hdulist += [hdu]
hdulist += [fits.ImageHDU(frequencies, name='FREQS_{0}'.format(beam_pol))]

outhdu = fits.HDUList(hdulist)
outhdu.writeto(rootdir+project_dir+pbdir+outfile, clobber=True)

PDB.set_trace()
