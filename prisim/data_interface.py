# THIS IS NOW DEPRECATED. DO NOT USE IT. WILL BE DELETED SOON

import numpy as NP
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time
from astropy import constants as FCNST
import warnings
from astroutils import geometry as GEOM
import interferometry as RI
try:
    from uvdata import UVData
except ImportError:
    uvdata_module_found = False
else:
    uvdata_module_found = True

class InterferometerData(object):

    """
    ----------------------------------------------------------------------------
    Class to act as an interface between PRISim object and external data 
    formats. 

    Attributes:

    infodict    [dictionary] Dictionary consisting of many attributes loaded 
                from the PRISim object. This will be used to convert to info
                required in external data formats

    Member functions:

    __init__()  Initialize an instance of class InterferometerData

    createUVData()
                Create an instance of class UVData

    write()     Write an instance of class InterferometerData into specified 
                formats. Currently writes in UVFITS format
    ----------------------------------------------------------------------------
    """

    def __init__(self, prisim_object, ref_point=None):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class InterferometerData.

        Class attributes initialized are:
        infodict

        Inputs:

        prisim_object   
                    [instance of class InterferometerArray] Instance of 
                    class InterferometerArray used to initialize an 
                    instance of class InterferometerData.

        ref_point   [dictionary] Contains information about the reference 
                    position to which projected baselines and rotated 
                    visibilities are to be computed. Default=None (no additional
                    phasing will be performed). It must be contain the following 
                    keys with the following values:
                    'coords'    [string] Refers to the coordinate system in
                                which value in key 'location' is specified in. 
                                Accepted values are 'radec', 'hadec', 'altaz'
                                and 'dircos'
                    'location'  [numpy array] Must be a Mx2 (if value in key 
                                'coords' is set to 'radec', 'hadec', 'altaz' or
                                'dircos') or Mx3 (if value in key 'coords' is 
                                set to 'dircos'). M can be 1 or equal to number
                                of timestamps. If M=1, the same reference point
                                in the same coordinate system will be repeated 
                                for all tiemstamps. If value under key 'coords'
                                is set to 'radec', 'hadec' or 'altaz', the 
                                value under this key 'location' must be in 
                                units of degrees.
        ------------------------------------------------------------------------
        """

        try:
            prisim_object
        except NameError:
            raise NameError('Input prisim_object not specified')
        if ref_point is not None:
            prisim_object.rotate_visibilities(ref_point)
        if not isinstance(prisim_object, RI.InterferometerArray):
            raise TypeError('Inout prisim_object must be an instance of class InterferometerArray')
        datatypes = ['noiseless', 'noisy', 'noise']
        visibilities = {key: None for key in datatypes}
        for key in visibilities:
            # Conjugate visibilities for compatibility with UVFITS and CASA imager
            if key == 'noiseless':
                visibilities[key] = prisim_object.skyvis_freq
            if key == 'noisy':
                visibilities[key] = prisim_object.vis_freq
            if key == 'noise':
                visibilities[key] = prisim_object.vis_noise_freq

        self.infodict = {}
        self.infodict['Ntimes'] = prisim_object.n_acc
        self.infodict['Nbls'] = prisim_object.baselines.shape[0]
        self.infodict['Nblts'] = self.infodict['Nbls'] * self.infodict['Ntimes']
        self.infodict['Nfreqs'] = prisim_object.channels.size
        self.infodict['Npols'] = 1
        self.infodict['Nspws'] = 1
        self.infodict['data_array'] = {'noiseless': None, 'noisy': None, 'noise': None}
        for key in visibilities:
            self.infodict['data_array'][key] = NP.transpose(NP.transpose(visibilities[key], (2,0,1)).reshape(self.infodict['Nblts'], self.infodict['Nfreqs'], self.infodict['Nspws'], self.infodict['Npols']), (0,2,1,3)) # (Nbls, Nfreqs, Ntimes) -> (Ntimes, Nbls, Nfreqs) -> (Nblts, Nfreqs, Nspws=1, Npols=1) -> (Nblts, Nspws=1, Nfreqs, Npols=1)
        self.infodict['vis_units'] = 'Jy'
        self.infodict['nsample_array'] = NP.ones((self.infodict['Nblts'], self.infodict['Nspws'], self.infodict['Nfreqs'], self.infodict['Npols']))
        self.infodict['flag_array'] = NP.zeros((self.infodict['Nblts'], self.infodict['Nspws'], self.infodict['Nfreqs'], self.infodict['Npols']), dtype=NP.bool)
        self.infodict['spw_array'] = NP.arange(self.infodict['Nspws'])
        self.infodict['uvw_array'] = NP.transpose(prisim_object.projected_baselines, (2,0,1)).reshape(self.infodict['Nblts'], 3)
        time_array = NP.asarray(prisim_object.timestamp).reshape(-1,1) + NP.zeros(self.infodict['Nbls']).reshape(1,-1)
        self.infodict['time_array'] = time_array.ravel()
        lst_array = NP.radians(NP.asarray(prisim_object.lst).reshape(-1,1)) + NP.zeros(self.infodict['Nbls']).reshape(1,-1)
        self.infodict['lst_array'] = lst_array.ravel()
        ant_1_array = prisim_object.labels['A1'].astype(NP.int)
        ant_2_array = prisim_object.labels['A2'].astype(NP.int)
        ant_1_array = ant_1_array.reshape(1,-1) + NP.zeros(self.infodict['Ntimes'], dtype=NP.int).reshape(-1,1)
        ant_2_array = ant_2_array.reshape(1,-1) + NP.zeros(self.infodict['Ntimes'], dtype=NP.int).reshape(-1,1)

        self.infodict['ant_1_array'] = ant_1_array.ravel()
        self.infodict['ant_2_array'] = ant_2_array.ravel()

        self.infodict['baseline_array'] = 2048 * (self.infodict['ant_2_array'] + 1) + (self.infodict['ant_1_array'] + 1) + 2**16
        self.infodict['freq_array'] = prisim_object.channels.reshape(self.infodict['Nspws'],-1)
        self.infodict['polarization_array'] = NP.asarray([1]).reshape(self.infodict['Npols'])
        self.infodict['integration_time'] = prisim_object.t_acc[0]
        self.infodict['channel_width'] = prisim_object.freq_resolution

        # ----- Observation information ------
        pointing_center = prisim_object.pointing_center
        pointing_coords = prisim_object.pointing_coords
        if pointing_coords == 'dircos':
            pointing_center_dircos = pointing_center
            pointing_center_altaz = GEOM.dircos2altaz(pointing_center_dircos, units='degrees')
            pointing_center_hadec = GEOM.altaz2hadec(pointing_center_altaz, prisim_object.latitude, units='degrees')
            pointing_center_ra = NP.asarray(prisim_object.lst) - pointing_center_hadec[:,0]
            pointing_center_radec = NP.hstack((pointing_center_ra.reshape(-1,1), pointing_center_hadec[:,1].reshape(-1,1)))
            pointing_coords = 'radec'
        elif pointing_coords == 'altaz':
            pointing_center_altaz = pointing_center
            pointing_center_hadec = GEOM.altaz2hadec(pointing_center_altaz, prisim_object.latitude, units='degrees')
            pointing_center_ra = NP.asarray(prisim_object.lst) - pointing_center_hadec[:,0]
            pointing_center_radec = NP.hstack((pointing_center_ra.reshape(-1,1), pointing_center_hadec[:,1].reshape(-1,1)))
            pointing_coords = 'radec'
        elif pointing_coords == 'hadec':
            pointing_center_hadec = pointing_center
            pointing_center_ra = NP.asarray(prisim_object.lst) - pointing_center_hadec[:,0]
            pointing_center_radec = NP.hstack((pointing_center_ra.reshape(-1,1), pointing_center_hadec[:,1].reshape(-1,1)))
            pointing_coords = 'radec'
        elif pointing_coords == 'radec':
            pointing_center_radec = pointing_center
        else:
            raise ValueError('Invalid pointing center coordinates')

        phase_center = prisim_object.phase_center
        phase_center_coords = prisim_object.phase_center_coords
        if phase_center_coords == 'dircos':
            phase_center_dircos = phase_center
            phase_center_altaz = GEOM.dircos2altaz(phase_center_dircos, units='degrees')
            phase_center_hadec = GEOM.altaz2hadec(phase_center_altaz, prisim_object.latitude, units='degrees')
            phase_center_ra = NP.asarray(prisim_object.lst) - phase_center_hadec[:,0]
            phase_center_radec = NP.hstack((phase_center_ra.reshape(-1,1), phase_center_hadec[:,1].reshape(-1,1)))
            phase_center_coords = 'radec'
        elif phase_center_coords == 'altaz':
            phase_center_altaz = phase_center
            phase_center_hadec = GEOM.altaz2hadec(phase_center_altaz, prisim_object.latitude, units='degrees')
            phase_center_ra = NP.asarray(prisim_object.lst) - phase_center_hadec[:,0]
            phase_center_radec = NP.hstack((phase_center_ra.reshape(-1,1), phase_center_hadec[:,1].reshape(-1,1)))
            phase_center_coords = 'radec'
        elif phase_center_coords == 'hadec':
            phase_center_hadec = phase_center
            phase_center_ra = NP.asarray(prisim_object.lst) - phase_center_hadec[:,0]
            phase_center_radec = NP.hstack((phase_center_ra.reshape(-1,1), phase_center_hadec[:,1].reshape(-1,1)))
            phase_center_coords = 'radec'
        elif phase_center_coords == 'radec':
            phase_center_radec = phase_center
        else:
            raise ValueError('Invalid phase center coordinates')

        pointing_centers = SkyCoord(ra=pointing_center_radec[:,0], dec=pointing_center_radec[:,1], frame='icrs', unit='deg')
        phase_centers = SkyCoord(ra=phase_center_radec[:,0], dec=phase_center_radec[:,1], frame='icrs', unit='deg')
        pointing_center_obscenter = pointing_centers[int(prisim_object.n_acc/2)]
        phase_center_obscenter = phase_centers[int(prisim_object.n_acc/2)]
        
        self.infodict['object_name'] = 'J{0}{1}'.format(pointing_center_obscenter.ra.to_string(sep='', precision=2, pad=True), pointing_center_obscenter.dec.to_string(sep='', precision=2, alwayssign=True, pad=True))
        if 'id' not in prisim_object.telescope:
            self.infodict['telescope_name'] = 'custom'
        else:
            self.infodict['telescope_name'] = prisim_object.telescope['id']
        self.infodict['instrument'] = self.infodict['telescope_name']
        self.infodict['telescope_location'] = NP.asarray([prisim_object.latitude, prisim_object.longitude, 0.0])
        self.infodict['history'] = 'PRISim'

        self.infodict['phase_center_epoch'] = 2000.0
        is_phased = NP.allclose(phase_centers.ra.value, phase_centers.ra.value[::-1]) and NP.allclose(phase_centers.dec.value, phase_centers.dec.value[::-1])
        self.infodict['is_phased'] = is_phased

        # ----- antenna information ------
        self.infodict['Nants_data'] = len(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2']))
        self.infodict['Nants_telescope'] = len(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2']))
        self.infodict['antenna_names'] = NP.asarray(list(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2'])))
        self.infodict['antenna_numbers'] = NP.asarray(list(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2']))).astype(NP.int)
        
        # ----- Optional information ------
        self.infodict['dateobs'] = Time(prisim_object.timestamp[0], format='jd', scale='utc').iso
        self.infodict['phase_center_ra'] = NP.radians(phase_center_obscenter.ra.value)
        self.infodict['phase_center_dec'] = NP.radians(phase_center_obscenter.dec.value)
        self.infodict['antenna_positions'] = NP.zeros((self.infodict['Nants_telescope'],3), dtype=NP.float)
        if hasattr(prisim_object, 'antenna_positions'):
            if prisim_object.antenna_positions is not None:
                if not isinstance(prisim_object.antenna_positions, NP.ndarray):
                    warnings.warn('Antenna positions must be a numpy array. Proceeding with default values.')
                else:
                    if prisim_object.antenna_positions.shape != (self.infodict['Nants_telescope'],3):
                        warnings.warn('Number of antennas in prisim_object found to be incompatible with number of unique antennas found. Proceeding with default values.')
                    else:
                        self.infodict['antenna_positions'] = prisim_object.antenna_positions
            
        self.infodict['gst0'] = 0.0
        self.infodict['rdate'] = ''
        self.infodict['earth_omega'] = 360.985
        self.infodict['dut1'] = 0.0
        self.infodict['timesys'] = 'UTC'
        
    #############################################################################

    def createUVData(self, datatype='noiseless'):

        """
        ------------------------------------------------------------------------
        Create an instance of class UVData.

        Inputs:

        datatype    [string] Specifies which visibilities are to be used in 
                    creating the UVData object. Accepted values are 'noiseless'
                    (default) for noiseless pure-sky visibilities, 'noisy' for
                    sky visibilities to which noise has been added, or 'noise'
                    for pure noise visibilities.

        Outputs:

        dataobj     [instance of class UVData] an instance of class UVData
                    containing visibilities of type specified in datatype. This
                    object can be used to write to some common external formats
                    such as UVFITS, etc.
        ------------------------------------------------------------------------
        """

        if not uvdata_module_found:
            raise ImportError('uvdata module not found')

        if datatype not in ['noiseless', 'noisy', 'noise']:
            raise ValueError('Invalid input datatype specified')

        attributes_of_uvdata = ['Ntimes', 'Nbls', 'Nblts', 'Nfreqs', 'Npols', 'Nspws', 'data_array', 'vis_units', 'nsample_array', 'flag_array', 'spw_array', 'uvw_array', 'time_array', 'lst_array', 'ant_1_array', 'ant_2_array', 'baseline_array', 'freq_array', 'polarization_array', 'integration_time', 'channel_width', 'object_name', 'telescope_name', 'instrument', 'telescope_location', 'history', 'phase_center_epoch', 'is_phased', 'Nants_data', 'Nants_telescope', 'antenna_names', 'antenna_numbers', 'dateobs', 'phase_center_ra', 'phase_center_dec']
        dataobj = UVData()
        for attrkey in attributes_of_uvdata:
            if attrkey != 'data_array':
                setattr(dataobj, attrkey, self.infodict[attrkey])
            else:
                if datatype in self.infodict[attrkey]:
                    if self.infodict[attrkey][datatype] is not None:
                        setattr(dataobj, attrkey, self.infodict[attrkey][datatype])
                    else:
                        raise KeyError('Data of specified datatype not found in InterferometerData object')
                else:
                    raise KeyError('Specified datatype not found in InterferometerData object')

        return dataobj

    #############################################################################
    
    def _blnum_to_antnums(self, blnum):
        if self.infodict['Nants_telescope'] > 2048:
            raise StandardError('error Nants={Nants}>2048 not supported'.format(Nants=self.infodict['Nants_telescope']))
        if NP.min(blnum) > 2**16:
            i = (blnum - 2**16) % 2048 - 1
            j = (blnum - 2**16 - (i + 1)) / 2048 - 1
        else:
            i = (blnum) % 256 - 1
            j = (blnum - (i + 1)) / 256 - 1
        return NP.int32(i), NP.int32(j)

    #############################################################################
    
    def _antnums_to_blnum(self, i, j, attempt256=False):
        # set the attempt256 keyword to True to (try to) use the older
        # 256 standard used in many uvfits files
        # (will use 2048 standard if there are more than 256 antennas)
        i, j = NP.int64((i, j))
        if self.infodict['Nants_telescope'] > 2048:
            raise StandardError('cannot convert i,j to a baseline index '
                                'with Nants={Nants}>2048.'
                                .format(Nants=self.infodict['Nants_telescope']))
        if attempt256:
            if (NP.max(i) < 255 and NP.max(j) < 255):
                return 256 * (j + 1) + (i + 1)
            else:
                print('Max antnums are {} and {}'.format(NP.max(i), NP.max(j)))
                message = 'antnums_to_baseline: found > 256 antennas, using ' \
                          '2048 baseline indexing. Beware compatibility ' \
                          'with CASA etc'
                warnings.warn(message)

        return NP.int64(2048 * (j + 1) + (i + 1) + 2**16)

    #############################################################################
    
    def write(self, outfile, datatype='noiseless', fmt='UVFITS', uvfits_method=None):

        """
        ------------------------------------------------------------------------
        Write an instance of class InterferometerData into specified formats.
        Currently writes in UVFITS format

        Inputs:

        outfile     [string] Filename into which data will be written

        datatype    [string] Specifies which visibilities are to be used in 
                    creating the UVData object. Accepted values are 'noiseless'
                    (default) for noiseless pure-sky visibilities, 'noisy' for
                    sky visibilities to which noise has been added, or 'noise'
                    for pure noise visibilities.

        fmt         [string] Output file format. Currently accepted values are
                    'UVFITS'

        uvfits_method
                    [string] Method using which UVFITS output is produced.
                    Accepted values are 'uvdata', 'uvfits' or None (default).
                    If set to 'uvdata', the UVFITS writer in uvdata module is
                    used. If set to 'uvfits', the in-house UVFITS writer is
                    used. If set to None, first uvdata module will be attempted
                    but if it fails then the in-house UVFITS writer will be
                    tried.
        ------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('Output filename not specified')

        if not isinstance(outfile, str):
            raise TypeError('Output filename must be a string')

        if datatype not in ['noiseless', 'noisy', 'noise']:
            raise ValueError('Invalid input datatype specified')

        if fmt.lower() not in ['uvfits']:
            raise ValueError('Output format not supported')

        if fmt.lower() == 'uvfits':
            write_successful = False
            if uvfits_method not in [None, 'uvfits', 'uvdata']:
                uvfits_method = None
            if (uvfits_method is None) or (uvfits_method == 'uvdata'):
                try:
                    uvdataobj = self.createUVData(datatype=datatype)
                    uvdataobj.write_uvfits(outfile, spoof_nonessential=True)
                except Exception as xption1:
                    write_successful = False
                    if uvfits_method == 'uvdata':
                        warnings.warn('Output through UVData module did not work due to the following exception:')
                        raise xption1
                    else:
                        warnings.warn('Output through UVData module did not work. Trying with built-in UVFITS writer')
                else:
                    write_successful = True
                    print 'Data successfully written using uvdata module to {0}'.format(outfile)
                    return

            # Try with in-house UVFITS writer
            try: 
                weights_array = self.infodict['nsample_array'] * NP.where(self.infodict['flag_array'], -1, 1)
                data_array = self.infodict['data_array'][datatype][:, NP.newaxis, NP.newaxis, :, :, :, NP.newaxis]
                weights_array = weights_array[:, NP.newaxis, NP.newaxis, :, :, :, NP.newaxis]
                # uvfits_array_data shape will be  (Nblts,1,1,[Nspws],Nfreqs,Npols,3)
                uvfits_array_data = NP.concatenate([data_array.real, data_array.imag, weights_array], axis=6)
        
                uvw_array_sec = self.infodict['uvw_array'] / FCNST.c.to('m/s').value
                # jd_midnight = NP.floor(self.infodict['time_array'][0] - 0.5) + 0.5
                tzero = NP.float32(self.infodict['time_array'][0])
            
                # uvfits convention is that time_array + relevant PZERO = actual JD
                # We are setting PZERO4 = float32(first time of observation)
                time_array = NP.float32(self.infodict['time_array'] - NP.float64(tzero))
        
                int_time_array = (NP.zeros_like((time_array), dtype=NP.float) + self.infodict['integration_time'])
                baselines_use = self._antnums_to_blnum(self.infodict['ant_1_array'], self.infodict['ant_2_array'], attempt256=True)
                # Set up dictionaries for populating hdu
                # Note that uvfits antenna arrays are 1-indexed so we add 1
                # to our 0-indexed arrays
                group_parameter_dict = {'UU      ': uvw_array_sec[:, 0],
                                        'VV      ': uvw_array_sec[:, 1],
                                        'WW      ': uvw_array_sec[:, 2],
                                        'DATE    ': time_array,
                                        'BASELINE': baselines_use,
                                        'ANTENNA1': self.infodict['ant_1_array'] + 1,
                                        'ANTENNA2': self.infodict['ant_2_array'] + 1,
                                        'SUBARRAY': NP.ones_like(self.infodict['ant_1_array']),
                                        'INTTIM': int_time_array}
                pscal_dict = {'UU      ': 1.0, 'VV      ': 1.0, 'WW      ': 1.0,
                              'DATE    ': 1.0, 'BASELINE': 1.0, 'ANTENNA1': 1.0,
                              'ANTENNA2': 1.0, 'SUBARRAY': 1.0, 'INTTIM': 1.0}
                pzero_dict = {'UU      ': 0.0, 'VV      ': 0.0, 'WW      ': 0.0,
                              'DATE    ': tzero, 'BASELINE': 0.0, 'ANTENNA1': 0.0,
                              'ANTENNA2': 0.0, 'SUBARRAY': 0.0, 'INTTIM': 0.0}

                # list contains arrays of [u,v,w,date,baseline];
                # each array has shape (Nblts)
                if (NP.max(self.infodict['ant_1_array']) < 255 and
                        NP.max(self.infodict['ant_2_array']) < 255):
                    # if the number of antennas is less than 256 then include both the
                    # baseline array and the antenna arrays in the group parameters.
                    # Otherwise just use the antenna arrays
                    parnames_use = ['UU      ', 'VV      ', 'WW      ',
                                    'DATE    ', 'BASELINE', 'ANTENNA1',
                                    'ANTENNA2', 'SUBARRAY', 'INTTIM']
                else:
                    parnames_use = ['UU      ', 'VV      ', 'WW      ', 'DATE    ',
                                    'ANTENNA1', 'ANTENNA2', 'SUBARRAY', 'INTTIM']

                group_parameter_list = [group_parameter_dict[parname] for
                                        parname in parnames_use]
                hdu = fits.GroupData(uvfits_array_data, parnames=parnames_use,
                                     pardata=group_parameter_list, bitpix=-32)
                hdu = fits.GroupsHDU(hdu)
        
                for i, key in enumerate(parnames_use):
                    hdu.header['PSCAL' + str(i + 1) + '  '] = pscal_dict[key]
                    hdu.header['PZERO' + str(i + 1) + '  '] = pzero_dict[key]
        
                # ISO string of first time in self.infodict['time_array']

                # hdu.header['DATE-OBS'] = Time(self.infodict['time_array'][0], scale='utc', format='jd').iso
                hdu.header['DATE-OBS'] = self.infodict['dateobs']
        
                hdu.header['CTYPE2  '] = 'COMPLEX '
                hdu.header['CRVAL2  '] = 1.0
                hdu.header['CRPIX2  '] = 1.0
                hdu.header['CDELT2  '] = 1.0
        
                hdu.header['CTYPE3  '] = 'STOKES  '
                hdu.header['CRVAL3  '] = self.infodict['polarization_array'][0]
                hdu.header['CRPIX3  '] = 1.0
                try:
                    hdu.header['CDELT3  '] = NP.diff(self.infodict['polarization_array'])[0]
                except(IndexError):
                    hdu.header['CDELT3  '] = 1.0
        
                hdu.header['CTYPE4  '] = 'FREQ    '
                hdu.header['CRVAL4  '] = self.infodict['freq_array'][0, 0]
                hdu.header['CRPIX4  '] = 1.0
                hdu.header['CDELT4  '] = NP.diff(self.infodict['freq_array'][0])[0]
        
                hdu.header['CTYPE5  '] = 'IF      '
                hdu.header['CRVAL5  '] = 1.0
                hdu.header['CRPIX5  '] = 1.0
                hdu.header['CDELT5  '] = 1.0
        
                hdu.header['CTYPE6  '] = 'RA'
                hdu.header['CRVAL6  '] = NP.degrees(self.infodict['phase_center_ra'])
        
                hdu.header['CTYPE7  '] = 'DEC'
                hdu.header['CRVAL7  '] = NP.degrees(self.infodict['phase_center_dec'])
        
                hdu.header['BUNIT   '] = self.infodict['vis_units']
                hdu.header['BSCALE  '] = 1.0
                hdu.header['BZERO   '] = 0.0
        
                hdu.header['OBJECT  '] = self.infodict['object_name']
                hdu.header['TELESCOP'] = self.infodict['telescope_name']
                hdu.header['LAT     '] = self.infodict['telescope_location'][0]
                hdu.header['LON     '] = self.infodict['telescope_location'][1]
                hdu.header['ALT     '] = self.infodict['telescope_location'][2]
                hdu.header['INSTRUME'] = self.infodict['instrument']
                hdu.header['EPOCH   '] = float(self.infodict['phase_center_epoch'])
        
                for line in self.infodict['history'].splitlines():
                    hdu.header.add_history(line)
            
                # ADD the ANTENNA table
                staxof = NP.zeros(self.infodict['Nants_telescope'])
        
                # 0 specifies alt-az, 6 would specify a phased array
                mntsta = NP.zeros(self.infodict['Nants_telescope'])
        
                # beware, X can mean just about anything
                poltya = NP.full((self.infodict['Nants_telescope']), 'X', dtype=NP.object_)
                polaa = [90.0] + NP.zeros(self.infodict['Nants_telescope'])
                poltyb = NP.full((self.infodict['Nants_telescope']), 'Y', dtype=NP.object_)
                polab = [0.0] + NP.zeros(self.infodict['Nants_telescope'])
        
                col1 = fits.Column(name='ANNAME', format='8A',
                                   array=self.infodict['antenna_names'])
                col2 = fits.Column(name='STABXYZ', format='3D',
                                   array=self.infodict['antenna_positions'])
                # convert to 1-indexed from 0-indexed indicies
                col3 = fits.Column(name='NOSTA', format='1J',
                                   array=self.infodict['antenna_numbers'] + 1)
                col4 = fits.Column(name='MNTSTA', format='1J', array=mntsta)
                col5 = fits.Column(name='STAXOF', format='1E', array=staxof)
                col6 = fits.Column(name='POLTYA', format='1A', array=poltya)
                col7 = fits.Column(name='POLAA', format='1E', array=polaa)
                # col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
                col9 = fits.Column(name='POLTYB', format='1A', array=poltyb)
                col10 = fits.Column(name='POLAB', format='1E', array=polab)
                # col11 = fits.Column(name='POLCALB', format='3E', array=polcalb)
                # note ORBPARM is technically required, but we didn't put it in
        
                cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col9, col10])
                ant_hdu = fits.BinTableHDU.from_columns(cols)
                ant_hdu.header['EXTNAME'] = 'AIPS AN'
                ant_hdu.header['EXTVER'] = 1

                # write XYZ coordinates if not already defined
                ant_hdu.header['ARRAYX'] = self.infodict['telescope_location'][0]
                ant_hdu.header['ARRAYY'] = self.infodict['telescope_location'][1]
                ant_hdu.header['ARRAYZ'] = self.infodict['telescope_location'][2]
                ant_hdu.header['FRAME'] = 'ITRF'
                ant_hdu.header['GSTIA0'] = self.infodict['gst0']
                ant_hdu.header['FREQ'] = self.infodict['freq_array'][0, 0]
                ant_hdu.header['RDATE'] = self.infodict['rdate']
                ant_hdu.header['UT1UTC'] = self.infodict['dut1']
        
                ant_hdu.header['TIMSYS'] = self.infodict['timesys']
                if self.infodict['timesys'] == 'IAT':
                    warnings.warn('This file has an "IAT" time system. Files of '
                                  'this type are not properly supported')
                ant_hdu.header['ARRNAM'] = self.infodict['telescope_name']
                ant_hdu.header['NO_IF'] = self.infodict['Nspws']
                ant_hdu.header['DEGPDY'] = self.infodict['earth_omega']
                # ant_hdu.header['IATUTC'] = 35.
        
                # set mandatory parameters which are not supported by this object
                # (or that we just don't understand)
                ant_hdu.header['NUMORB'] = 0
        
                # note: Bart had this set to 3. We've set it 0 after aips 117. -jph
                ant_hdu.header['NOPCAL'] = 0
        
                ant_hdu.header['POLTYPE'] = 'X-Y LIN'
        
                # note: we do not support the concept of "frequency setups"
                # -- lists of spws given in a SU table.
                ant_hdu.header['FREQID'] = -1
        
                # if there are offsets in images, this could be the culprit
                ant_hdu.header['POLARX'] = 0.0
                ant_hdu.header['POLARY'] = 0.0
        
                ant_hdu.header['DATUTC'] = 0  # ONLY UTC SUPPORTED
        
                # we always output right handed coordinates
                ant_hdu.header['XYZHAND'] = 'RIGHT'
        
                # ADD the FQ table
                # skipping for now and limiting to a single spw
        
                # write the file
                hdulist = fits.HDUList(hdus=[hdu, ant_hdu])
                hdulist.writeto(outfile, clobber=True)
            except Exception as xption2:
                print xption2
                raise IOError('Could not write to UVFITS file')
            else:
                write_successful = True
                print 'Data successfully written using in-house uvfits writer to {0}'.format(outfile)
                return

#################################################################################
