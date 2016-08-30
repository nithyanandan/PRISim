import numpy as NP
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time
from astroutils import geometry as GEOM
from prisim import interferometry as RI
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
    
