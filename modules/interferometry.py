from __future__ import division
import os as OS
import numpy as NP
import numpy.linalg as LA
import scipy.constants as FCNST
from scipy.linalg import toeplitz
import scipy.optimize as OPT
import datetime as DT
import progressbar as PGB
from astropy.io import fits
import geometry as GEOM
import primary_beams as PB
import baseline_delay_horizon as DLY
import constants as CNST
import my_DSP_modules as DSP
import catalog as CTLG

################################################################################

def baseline_generator(antenna_locations, auto=False, conjugate=False):

    """
    -------------------------------------------------------------------
    Generate baseline from antenna locations.

    Inputs:

    antenna_locations: List of tuples containing antenna coordinates, 
                       or list of instances of class Point containing
                       antenna coordinates, or Numpy array (Nx3) array
                       with each row specifying an antenna location.

    Input keywords:

    auto:              [Default=False] If True, compute zero spacings of
                       antennas with themselves.

    conjugate:         [Default=False] If True, compute conjugate 
                       baselines.

    Output:

    baseline_locations: Baseline locations in the same data type as 
                        antenna locations (list of tuples, list of 
                        instances of class Point or Numpy array of size
                        Nb x 3 with each row specifying one baseline 
                        vector)

    -------------------------------------------------------------------
    """

    try:
        antenna_locations
    except NameError:
        print 'No antenna locations supplied. Returning from baseline_generator()'
        return None

    inp_type = 'tbd'

    if not isinstance(antenna_locations, NP.ndarray):
        if isinstance(antenna_locations, list):
            if isinstance(antenna_locations[0], GEOM.Point):
                inp_type = 'loo' # list of objects
            elif isinstance(antenna_locations[0], tuple):
                inp_type = 'lot' # list of tuples
                antenna_locations = [(tuple(loc) if len(loc) == 3 else (tuple([loc[0],0.0,0.0]) if len(loc) == 1 else (tuple([loc[0],loc[1],0.0]) if len(loc) == 2 else (tuple([loc[0],loc[1],loc[2]]))))) for loc in antenna_locations if len(loc) != 0] # Remove empty tuples and validate the data range and data type for antenna locations. Force it to have three components for every antenna location.
        elif isinstance(antenna_locations, GEOM.Point):
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return GEOM.Point()
        elif isinstance(antenna_locations, tuple):
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return (0.0,0.0,0.0)
        else:
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return (0.0,0.0,0.0)
    else:
        inp_type = 'npa' # A numpy array
        if antenna_locations.shape[0] == 1:
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return NP.zeros(1,3)
        else:
            if antenna_locations.shape[1] > 3:
                antenna_locations = antenna_locations[:,:3]
            elif antenna_locations.shape[1] < 3:
                antenna_locations = NP.hstack((antenna_locations, NP.zeros(antenna_locations.shape[0],3-antenna_locations.shape[1])))

    if isinstance(antenna_locations, list):
        num_ants = len(antenna_locations)
    else:
        num_ants = antenna_locations.shape[0]

    if inp_type == 'loo':
        if auto:
            baseline_locations = [antenna_locations[j]-antenna_locations[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
        else:
            baseline_locations = [antenna_locations[j]-antenna_locations[i] for i in range(0,num_ants) for j in range(0,num_ants) if j > i]                
        if conjugate:
            baseline_locations += [antenna_locations[j]-antenna_locations[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
    elif inp_type == 'lot':
        if auto:
            baseline_locations = [tuple((antenna_locations[j][0]-antenna_locations[i][0], antenna_locations[j][1]-antenna_locations[i][1], antenna_locations[j][2]-antenna_locations[i][2])) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
        else:
            baseline_locations = [tuple((antenna_locations[j][0]-antenna_locations[i][0], antenna_locations[j][1]-antenna_locations[i][1], antenna_locations[j][2]-antenna_locations[i][2])) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
        if conjugate:
            baseline_locations += [tuple((antenna_locations[j][0]-antenna_locations[i][0], antenna_locations[j][1]-antenna_locations[i][1], antenna_locations[j][2]-antenna_locations[i][2])) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
    elif inp_type == 'npa':
        if auto:
            baseline_locations = [antenna_locations[i,:]-antenna_locations[j,:] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
        else:
            baseline_locations = [antenna_locations[i,:]-antenna_locations[j,:] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]        
        if conjugate:
            baseline_locations += [antenna_locations[i,:]-antenna_locations[j,:] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]         
        baseline_locations = NP.asarray(baseline_locations)

    return baseline_locations

#################################################################################

class Interferometer:

    """
    ----------------------------------------------------------------------------
    Class to manage information on a two-element interferometer. 

    Attributes:

    A_eff       [scalar] Effective area of the interferometer (in m^2). Default
                is pi * (25/2)^2, appropriate for a 25 m VLA dish.

    baseline:   [1 x 3 Numpy array] The baseline vector associated with the
                interferometer in SI units. The coordinate system of this vector 
                is specified by another attribute baseline_coords. 

    baseline_coords
                [string] Coordinate system for the baseline vector. Default is 
                'localenu'. Other accepted values are 'equatorial' 

    baseline_length
                [scalar] Length of the baseline in SI units

    bp          [numpy array] Inherent bandpass weights of size n_acc x nchan,
                where n_acc is the number of accumulations in the observation and
                nchan is the number of frequency channels. These weights are 
                inherent to the interferometer, although they will be kept as a
                separate quantity and applied as and when required.

    bp_wts      [numpy array] Additional bandpass frequency weights to be applied
                to the visibilities in addition to the inherent bandpass weights 
                given by bp. Default is that no additional weights are applied
                (all values in bp_wts are by default unity). bp_wts can be
                modified by passing the input parameter freq_wts in the member 
                function delay_transform()

    channels    [list or numpy vector] frequency channels in Hz

    eff_Q       [scalar] Efficiency of the interferometer. Default = 0.89,
                appropriate for the VLA. Has to be between 0 and 1. 

    freq_resolution
                [scalar] Frequency resolution (in Hz)

    label:      [Scalar] A unique identifier (preferably a string) for the 
                interferometer. 

    lags        [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as channels. This is 
                computed in member function delay_transform().

    latitude    [Scalar] Latitude of the interferometer's location. Default
                is 34.0790 degrees North corresponding to that of the VLA.

    lst         [list] List of LST (in degrees) for each timestamp

    n_acc       [scalar] Number of accumulations

    obs_catalog_indices
                [list of lists] Each element in the top list corresponds to a
                timestamp. Inside each top list is a list of indices of sources
                from the catalog which are observed inside the region of 
                interest. This is computed inside member function observe(). 

    pointing_center
                [2-column numpy array] Pointing center (latitude and 
                longitude) of the observation at a given timestamp. This is 
                where the telescopes will be phased up to as reference. 
                Coordinate system for the pointing_center is specified by another 
                attribute pointing_coords.

    pointing_coords
                [string] Coordinate system for telescope pointing. Accepted 
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz' 
                (Altitude-Azimuth). Default = 'hadec'.

    skycoords   [string] Coordinate system for the sky positions of sources.
                Accepted values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 
                'altaz' (Altitude-Azimuth). Default = 'radec'.
    
    skyvis_freq [numpy array] Complex visibility due to sky emission (in Jy or K) 
                along frequency axis estimated from the specified external
                catalog. Same size as vis_freq. Used in the member function
                observe(). Read its docstring for more details. 

    skyvis_lag  [numpy array] Complex visibility due to sky emission (in Jy Hz or
                K Hz) along the delay axis obtained by FFT of skyvis_freq along 
                frequency axis. Same size as vis_freq. Created in the member
                function delay_transform(). Read its docstring for more details. 

    telescope   [string] The name of the telescope facility. Accepted values
                are 'vla', 'gmrt', 'mwa_dipole' and 'mwa'. Default = 'vla'

    timestamp   [list] List of timestamps during the observation

    t_acc       [list] Accumulation time (sec) corresponding to each timestamp

    t_obs       [scalar] Observing duration (sec)

    Tsys        [scalar] System temperature in Kelvin

    vis_freq    [numpy array] The simulated complex visibility (in Jy or K) 
                observed by the interferometer along frequency axis for each 
                timestamp of observation per frequency channel. It is the sum of 
                skyvis_freq and vis_noise_freq. It can be either directly 
                initialized or simulated in observe(). 

    vis_lag     [numpy array] The simulated complex visibility (in Jy Hz or K Hz) 
                along delay axis obtained by FFT of vis_freq along frequency 
                axis. Same size as vis_noise_lag and skyis_lag. It is evaluated 
                in member function delay_transform(). 

    vis_noise_freq
                [numpy array] Complex visibility noise (in Jy or K) generated 
                using an rms of vis_rms_freq along frequency axis which is then 
                added to the generated sky visibility. Same size as vis_freq. 
                Used in the member function observe(). Read its docstring for 
                more details. 

    vis_noise_lag
                [numpy array] Complex visibility noise (in Jy Hz or K Hz) along 
                delay axis generated using an FFT of vis_noise_freq along 
                frequency axis. Same size as vis_noise_freq. Created in the 
                member function delay_transform(). Read its docstring for more 
                details. 

    vis_rms_freq
                [list of float] Theoretically estimated thermal noise rms (in Jy
                or K) in visibility measurements. Same size as vis_freq. This 
                will be estimated and used to inject simulated noise when a call 
                to member function observe() is made. Read the  docstring of 
                observe() for more details. The noise rms is estimated from the 
                instrument parameters as:
                (2 k T_sys / (A_eff x sqrt(2 x channel_width x t_acc))) / Jy, or
                T_sys / sqrt(2 x channel_width x t_acc)

    Member functions:

    __init__():        Initializes an instance of class Interferometer

    observe():         Simulates an observing run with the interferometer
                       specifications and an external sky catalog thus producing
                       visibilities. The simulation generates visibilities
                       observed by the interferometer for the specified
                       parameters.

    observing_run():   Simulates an observing run by repeatedly invoking 
                       observe() for each accumulation period. 

    delay_transform(): Transforms the visibilities from frequency axis onto 
                       delay (time) axis using an FFT.

    band_averaged_noise_estimate():
                       Given the attribute vis_freq, compute the thermal noise 
                       estimate (in Jy or K) on average in the data in each
                       frequency channel

    ----------------------------------------------------------------------------
    """

    ############################################################################

    def __init__(self, label, baseline, channels, telescope='vla', eff_Q=0.89,
                 latitude=34.0790, skycoords='radec', A_eff=NP.pi*(25.0/2)**2, 
                 pointing_coords='hadec', baseline_coords='localenu',
                 freq_scale=None):

        """
        ------------------------------------------------------------------------
        Intialize the Interferometer class which manages information on a 
        2-element interferometer.

        Class attributes initialized are:
        label, baseline, channels, telescope, latitude, skycoords, eff_Q, A_eff,
        pointing_coords, baseline_coords, baseline_length, channels, bp, bp_wts,
        freq_resolution, lags, lst, obs_catalog_indices, pointing_center,
        skyvis_freq, skyvis_lag, timestamp, t_acc, Tsys, vis_freq, vis_lag, 
        t_obs, n_acc, vis_noise_freq, vis_noise_lag, vis_rms_freq,
        geometric_delays.

        Read docstring of class Interferometer for details on these attributes.
        ------------------------------------------------------------------------
        """

        self.label = label
        self.baseline = NP.asarray(baseline).reshape(1,-1)
        self.baseline_length = NP.sqrt(NP.sum(self.baseline**2))
        self.telescope = telescope
        self.latitude = latitude
        self.vis_freq = None
        self.skyvis_freq = None
        # self.pb = None
        self.vis_noise_freq = None

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            self.channels = NP.asarray(channels)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            self.channels = NP.asarray(channels) * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            self.channels = NP.asarray(channels) * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            self.channels = NP.asarray(channels) * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')

        self.bp = NP.ones(self.channels.size).reshape(1,-1) # Inherent bandpass shape
        self.bp_wts = NP.ones(self.channels.size).reshape(1,-1) # Additional bandpass weights
        self.Tsys = []
        self.flux_unit = 'JY'
        self.timestamp = []
        self.t_acc = []
        self.t_obs = 0.0
        self.n_acc = 0
        self.pointing_center = NP.empty([1,2])
        self.lst = []
        self.eff_Q = eff_Q
        self.A_eff = A_eff
        self.vis_rms_freq = []
        self.freq_resolution = self.channels[1] - self.channels[0]
        self.baseline_coords = baseline_coords
        self.lags = None
        self.skyvis_lag = None
        self.vis_noise_lag = None
        self.vis_lag = None
        self.obs_catalog_indices = []
        self.geometric_delays = []

        if (pointing_coords == 'radec') or (pointing_coords == 'hadec') or (pointing_coords == 'altaz'):
            self.pointing_coords = pointing_coords
        else:
            raise ValueError('Pointing center of the interferometer must be "radec", "hadec" or "altaz". Check inputs.')

        if (skycoords == 'radec') or (skycoords == 'hadec') or (skycoords == 'altaz'):
            self.skycoords = skycoords
        else:
            raise ValueError('Sky coordinates must be "radec", "hadec" or "altaz". Check inputs.')

        if (baseline_coords == 'equatorial') or (baseline_coords == 'localenu'):
            self.baseline_coords = baseline_coords
        else:
            raise ValueError('Baseline coordinates must be "equatorial" or "local". Check inputs.')

    #############################################################################

    def observe(self, timestamp, Tsys, bandpass, pointing_center, skymodel,
                t_acc, brightness_units=None, roi_radius=None, roi_center=None,
                lst=None):

        """
        -------------------------------------------------------------------------
        Simulate a snapshot observation, by an instance of the Interferometer
        class, of the sky when a sky catalog is provided. The simulation 
        generates visibilities observed by the interferometer for the specified
        parameters. See member function observing_run() for simulating an 
        extended observing run in 'track' or 'drift' mode.

        Inputs:
        
        timestamp    [scalar] Timestamp associated with each integration in the
                     observation

        Tsys         [scalar float] System temperature associated with the 
                     timestamp for the observation

        bandpass     [numpy vector] Bandpass weights associated with the 
                     timestamp for the observation

        pointing_center
                     [2-element numpy vector or list] Pointing center (latitude 
                     and longitude) of the observation at a given timestamp. 
                     This is where the telescopes will be phased up to as 
                     reference. Coordinate system for the pointing_center is 
                     specified by the attribute pointing_coords initialized in
                     __init__(). 

        skymodel     [instance of class SkyModel] It consists of source flux
                     densities, their positions, and spectral indices. Read 
                     class SkyModel docstring for more information.

        t_acc        [scalar] Accumulation time (sec) corresponding to timestamp

        brightness_units
                     [string] Units of flux density in the catalog and for the 
                     generated visibilities. Accepted values are 'Jy' (Jansky) 
                     and 'K' (Kelvin for temperature). If None set, it defaults 
                     to 'Jy'

        Keyword Inputs:

        roi_radius   [scalar] Radius of the region of interest (degrees) inside 
                     which sources are to be observed. Default = 90 degrees, 
                     which is the entire horizon.

        roi_center   [string] Center of the region of interest around which
                     roi_radius is used. Accepted values are 'pointing_center'
                     and 'zenith'. If set to None, it defaults to 'zenith'. 

        lst          [scalar] LST (in degrees) associated with the timestamp
        ------------------------------------------------------------------------
        """

        if bandpass.size != self.bp.shape[1]:
            raise ValueError('bandpass length does not match.')

        self.Tsys = self.Tsys + [Tsys]

        if (brightness_units is None) or (brightness_units=='Jy') or (brightness_units=='JY') or (brightness_units=='jy'):
            self.vis_rms_freq = self.vis_rms_freq + [2.0*FCNST.k*Tsys/self.A_eff/self.eff_Q/NP.sqrt(2.0 * t_acc * self.freq_resolution)/CNST.Jy]
            self.flux_unit = 'JY'
        elif (brightness_units=='K') or (brightness_units=='k'):
            self.vis_rms_freq = self.vis_rms_freq + [Tsys/self.eff_Q/NP.sqrt(2.0 * t_acc * self.freq_resolution)]
            self.flux_unit = 'K'
        else:
            raise ValueError('Invalid brightness temperature units specified.')

        self.t_acc = self.t_acc + [t_acc]
        self.t_obs = t_acc
        self.n_acc = 1
        self.lst = self.lst + [lst]

        if self.timestamp == []:
            self.bp = NP.asarray(bandpass).reshape(1,-1)
            self.pointing_center = NP.asarray(pointing_center).reshape(1,-1)
        else:
            self.bp = NP.vstack((self.bp, NP.asarray(bandpass).reshape(1,-1)))
            self.pointing_center = NP.vstack((self.pointing_center, NP.asarray(pointing_center).reshape(1,-1)))
            
        self.bp_wts = NP.ones_like(self.bp) # All additional bandpass shaping weights are set to unity.

        pointing_lon = self.pointing_center[-1,0]
        pointing_lat = self.pointing_center[-1,1]

        if self.skycoords == 'radec':
            if self.pointing_coords == 'hadec':
                if lst is not None:
                    pointing_lon = lst - self.pointing_center[-1,0]
                    pointing_lat = self.pointing_center[-1,1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in RA-Dec format while pointing center is in HA-Dec format.')
            elif self.pointing_coords == 'altaz':
                pointing_lonlat = GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = lst - pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]
        elif self.skycoords == 'hadec':
            if self.pointing_coords == 'radec':
                if lst is not None:
                    pointing_lon = lst - self.pointing_center[-1,0]
                    pointing_lat = self.pointing_center[-1,1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in RA-Dec format while pointing center is in HA-Dec format.')
            elif self.pointing_coords == 'altaz':
                pointing_lonlat = lst - GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]
        else:
            if self.pointing_coords == 'radec':
                if lst is not None:
                    pointing_lonlat = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
                    pointing_lon = pointing_lonlat[0]
                    pointing_lat = pointing_lonlat[1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')
            elif self.pointing_coords == 'hadec':
                pointing_lonlat = GEOM.hadec2altaz(self.pointing_center,
                                                   self.latitude,
                                                   units='degrees')
                pointing_lon = pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]

        pointing_phase = 0.0

        baseline_in_local_frame = self.baseline
        if self.baseline_coords == 'equatorial':
            baseline_in_local_frame = GEOM.xyz2enu(self.baseline, self.latitude, 'degrees')

        pc_altaz = self.pointing_center[-1,:] # Convert pointing center to Alt-Az coordinates
        if self.pointing_coords == 'hadec':
            pc_altaz = GEOM.hadec2altaz(self.pointing_center[-1,:], self.latitude,
                                        units='degrees')
        elif self.pointing_coords == 'radec':
            if lst is not None:
                pc_altaz = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
            else:
                raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')

        pc_dircos = GEOM.altaz2dircos(pc_altaz, 'degrees') # Convert pointing center to direction cosine coordinates
        pointing_phase = 2.0 * NP.pi * NP.dot(baseline_in_local_frame.reshape(1,-1), pc_dircos.reshape(-1,1))*self.channels.reshape(1,-1)/FCNST.c

        if not isinstance(skymodel, CTLG.SkyModel):
            raise TypeError('skymodel should be an instance of class SkyModel.')

        if roi_radius is None:
            roi_radius = 90.0

        if roi_center is None:
            roi_center = 'zenith'
        elif (roi_center != 'zenith') and (roi_center != 'pointing_center'):
            raise ValueError('Center of region of interest, roi_center, must be set to "zenith" or "pointing_center".')

        if roi_center == 'pointing_center':
            m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.catalog.location[:,0], skymodel.catalog.location[:,1], roi_radius, maxmatches=0)
        else: # roi_center = 'zenith'
            if self.skycoords == 'hadec':
                skypos_altaz = GEOM.hadec2altaz(skymodel.catalog.location, self.latitude, units='degrees')
            elif self.skycoords == 'radec':
                skypos_altaz = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.catalog.location[:,0]).reshape(-1,1), skymodel.catalog.location[:,1].reshape(-1,1))), self.latitude, units='degrees')
            m2 = NP.arange(skypos_altaz.shape[0])
            m2 = m2[NP.where(skypos_altaz[:,0] >= 90.0-roi_radius)] # select sources whose altitude (angle above horizon) is 90-roi_radius

        # if roi_radius is not None:
        #     m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.catalog.location[:,0], skymodel.catalog.location[:,1], roi_radius, maxmatches=0)
        # else:
        #     m1 = [0] * skymodel.catalog.location.shape[0]
        #     m2 = xrange(skymodel.catalog.location.shape[0])
        #     d12 = GEOM.sphdist(NP.empty(skymodel.catalog.shape[0]).fill(pointing_lon), NP.empty(skymodel.catalog.shape[0]).fill(pointing_lat), skymodel.catalog.location[:,0], skymodel.catalog.location[:,1])

        if len(m2) != 0:
            pb = NP.empty((len(m2), len(self.channels)))
            fluxes = NP.empty((len(m2), len(self.channels)))
            
            if roi_center != 'zenith':
                if self.skycoords == 'altaz':
                    skypos_altaz_roi = skymodel.catalog.location[m2,:]
                elif self.skycoords == 'radec':
                    skypos_altaz_roi = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.catalog.location[m2,0]).reshape(-1,1), skymodel.catalog.location[m2,1].reshape(-1,1))), self.latitude, 'degrees')
                else:
                    skypos_altaz_roi = GEOM.hadec2altaz(skymodel.catalog.location[m2,:], self.latitude, 'degrees')
            else:
                skypos_altaz_roi = skypos_altaz[m2,:]
            coords_str = 'altaz'

            pb = PB.primary_beam_generator(skypos_altaz_roi, self.channels/1.0e9, skyunits='altaz', telescope=self.telescope, phase_center=pc_altaz)
            # fluxes = NP.repeat(skymodel.catalog.flux_density[m2].reshape(-1,1), self.channels.size, axis=1) * (NP.repeat(self.channels.reshape(1,-1), len(m2), axis=0)/skymodel.catalog.frequency)**NP.repeat(skymodel.catalog.spectral_index[m2].reshape(-1,1), self.channels.size, axis=1)
            fluxes = skymodel.catalog.flux_density[m2].reshape(-1,1) * (self.channels.reshape(1,-1)/skymodel.catalog.frequency[m2].reshape(-1,1))**skymodel.catalog.spectral_index[m2].reshape(-1,1)  # numpy array broadcasting 
            geometric_delays = DLY.geometric_delay(baseline_in_local_frame, skypos_altaz_roi, altaz=(coords_str=='altaz'), hadec=(coords_str=='hadec'), latitude=self.latitude)
            self.geometric_delays = self.geometric_delays + [geometric_delays.reshape(len(m2))]
            # phase_matrix = 2.0 * NP.pi * NP.repeat(geometric_delays.reshape(-1,1),len(self.channels),axis=1) * NP.repeat(self.channels.reshape(1,-1),len(m2),axis=0) - NP.repeat(pointing_phase, len(m2), axis=0)
            phase_matrix = 2 * NP.pi * geometric_delays.reshape(-1,1) * self.channels.reshape(1,-1) - pointing_phase.reshape(1,-1)
            # skyvis = NP.sum(pb * fluxes * NP.repeat(NP.asarray(bandpass).reshape(1,-1),len(m2),axis=0) * NP.exp(-1j*phase_matrix), axis=0)
            skyvis = NP.sum(pb * fluxes * NP.exp(-1j*phase_matrix), axis=0) # Don't apply bandpass here
            # if roi_radius is not None:
            self.obs_catalog_indices = self.obs_catalog_indices + [m2]
            # self.obs_catalog = self.obs_catalog + [skymodel.catalog.subset(m2)]
        else:
            print 'No sources found in the catalog within matching radius. Simply populating the observed visibilities with noise.'
            skyvis = NP.zeros( (1, len(self.channels)) )

        if self.timestamp == []:
            self.skyvis_freq = skyvis.reshape(1,-1)
            self.vis_noise_freq = self.vis_rms_freq[-1] / NP.sqrt(2.0) * (NP.random.randn(len(self.channels)).reshape(1,-1) + 1j * NP.random.randn(len(self.channels)).reshape(1,-1)) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts
            self.vis_freq = self.skyvis_freq + self.vis_noise_freq
        else:
            self.skyvis_freq = NP.vstack((self.skyvis_freq, skyvis.reshape(1,-1)))
            self.vis_noise_freq = NP.vstack((self.vis_noise_freq, self.vis_rms_freq[-1]/NP.sqrt(2.0) * (NP.random.randn(len(self.channels)).reshape(1,-1) + 1j * NP.random.randn(len(self.channels)).reshape(1,-1)))) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts
            self.vis_freq = NP.vstack((self.vis_freq, (self.skyvis_freq[-1,:] + self.vis_noise_freq[-1,:]).reshape(1,-1)))

        self.timestamp = self.timestamp + [timestamp]

    ############################################################################

    def observing_run(self, pointing_init, skymodel, t_acc, duration, channels, 
                      bpass, Tsys, lst_init, roi_radius=None, roi_center=None,
                      mode='track', pointing_coords=None, freq_scale=None,
                      brightness_units=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Simulate an extended observing run in 'track' or 'drift' mode, by an
        instance of the Interferometer class, of the sky when a sky catalog is
        provided. The simulation generates visibilities observed by the
        interferometer for the specified parameters. Uses member function
        observe() and builds the observation from snapshots. The timestamp for
        each snapshot is the current time at which the snapshot is generated.

        Inputs:
        
        pointing_init [2-element list or numpy array] The inital pointing
                      of the telescope at the start of the observing run. 
                      This is where the telescopes will be initially phased up to
                      as reference. Coordinate system for the pointing_center is 
                      specified by the input pointing_coords 

        skymodel      [instance of class SkyModel] It consists of source flux
                      densities, their positions, and spectral indices. Read 
                      class SkyModel docstring for more information.

        t_acc         [scalar] Accumulation time (sec) corresponding to timestamp

        brightness_units
                      [string] Units of flux density in the catalog and for the 
                      generated visibilities. Accepted values are 'Jy' (Jansky) 
                      and 'K' (Kelvin for temperature). If None set, it defaults 
                      to 'Jy'

        duration      [scalar] Duration of observation in seconds

        channels      [list or numpy vector] frequency channels in units as 
                      specified in freq_scale

        bpass         [list, list of lists or numpy array] Bandpass weights in
                      the form of M x N array or list of N-element lists. N must
                      equal the number of channels. If M=1, the same bandpass
                      will be used in all the snapshots for the entire
                      observation, otherwise M must equal the number of
                      snapshots which is int(duration/t_acc)

        Tsys          [scalar, list or numpy array] System temperature (in K). If
                      a scalar is provided, the same Tsys will be used in all the
                      snapshots for the duration of the observation. If a list or
                      numpy array is provided, the number of elements must equal 
                      the number of snapshots which is int(duration/t_int)

        lst_init      [scalar] Initial LST (in degrees) at the beginning of the 
                      observing run corresponding to pointing_init

        Keyword Inputs:

        roi_radius    [scalar] Radius of the region of interest (degrees) inside 
                      which sources are to be observed. Default = 90 degrees, 
                      which is the entire horizon.
                      
        roi_center    [string] Center of the region of interest around which
                      roi_radius is used. Accepted values are 'pointing_center'
                      and 'zenith'. If set to None, it defaults to 'zenith'. 

        freq_scale    [string] Units of frequencies specified in channels. 
                      Accepted values are 'Hz', 'hz', 'khz', 'kHz', 'mhz',
                      'MHz', 'GHz' and 'ghz'. If None provided, defaults to 'Hz'

        mode          [string] Mode of observation. Accepted values are 'track'
                      and 'drift'. If using 'track', pointing center is fixed to
                      a specific point on the sky coordinate frame. If using 
                      'drift', pointing center is fixed to a specific point on
                      the antenna's reference frame. 

        pointing_coords
                      [string] Coordinate system for pointing_init. Accepted 
                      values are 'radec', 'hadec' and 'altaz'. If None provided,
                      default is set based on observing mode. If mode='track', 
                      pointing_coords defaults to 'radec', and if mode='drift', 
                      it defaults to 'hadec'

        verbose       [boolean] If set to True, prints progress and diagnostic 
                      messages. Default = True
        ------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing an observing run...\n'
            print '\tVerifying input arguments to observing_run()...'

        try:
            pointing_init, skymodel, t_acc, duration, bpass, Tsys, lst_init
        except NameError:
            raise NameError('One or more of pointing_init, skymodel, t_acc, duration, bpass, Tsys, lst_init not specified.')

        if isinstance(pointing_init, list):
            pointing_init = NP.asarray(pointing_init)
        elif not isinstance(pointing_init, NP.ndarray):
            raise TypeError('pointing_init must be a list or numpy array.')

        if pointing_init.size != 2:
            raise ValueError('pointing_init must be a 2-element vector.')
        pointing_init = pointing_init.ravel()

        if not isinstance(skymodel, CTLG.SkyModel):
            raise TypeError('skymodel must be an instance of class SkyModel.')

        if not isinstance(t_acc, (int, float)):
            raise TypeError('t_acc must be a scalar integer or float.')

        if t_acc <= 0.0:
            raise ValueError('t_acc must be positive.')

        if not isinstance(duration, (int, float)):
            raise TypeError('duration must be a scalar integer or float.')

        if duration <= t_acc:
            if verbose:
                print '\t\tDuration specified to be shorter than t_acc. Will set it equal to t_acc'
            duration = t_acc

        n_acc = int(duration / t_acc)
        if verbose:
            print '\t\tObserving run will have {0} accumulations.'.format(n_acc)

        if isinstance(channels, list):
            channels = NP.asarray(channels)
        elif not isinstance(channels, NP.ndarray):
            raise TypeError('channels must be a list or numpy array')

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            channels = NP.asarray(channels)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            channels = channels * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            channels = channels * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            channels = channels * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')

        if isinstance(bpass, list):
            bpass = NP.asarray(bpass)
        elif not isinstance(bpass, NP.ndarray):
            raise TypeError('bpass must be a list or numpy array')
        
        if len(bpass.shape) == 1:
            bpass = bpass.reshape(1,-1)
        elif len(bpass.shape) > 2:
            raise ValueError('Too many dimensions for bandpass')

        if bpass.shape[1] == channels.size:
            if bpass.shape[0] == 1:
                bpass = NP.repeat(bpass, n_acc, axis=0)
                if verbose:
                    print '\t\tSame bandpass will be applied to all accumulations in the observing run.'
            elif bpass.shape[0] != n_acc:
                raise ValueError('Number of bandpasses specified do not match the number of accumulations.')

            self.freq_resolution = channels[1] - channels[0]
            self.channels = channels
        else:
            raise ValueError('Dimensions of bpass and channels are incompatible')

        if isinstance(Tsys, (list, NP.ndarray)):
            Tsys = NP.asarray(Tsys).ravel()
            if (Tsys.size != 1) and (Tsys.size != n_acc):
                raise ValueError('Mismatch between size of Tsys and number of accumulations.')
            if NP.any(Tsys < 0.0):
                raise ValueError('Tsys cannot be negative.')
        elif isinstance(Tsys, (int, float)):
            if Tsys < 0.0:
                raise ValueError('Tsys cannot be negative.')
            else:
                if verbose:
                    print '\t\tTsys = {0:.1f} K will be used for all accumulations.'.format(Tsys)
                Tsys = Tsys * NP.ones(n_acc)

        if not isinstance(lst_init, (int, float)):
            raise TypeError('Starting LST should be a scalar')

        if verbose:
            print '\tVerified input arguments.'
            print '\tProceeding to schedule the observing run...'

        lst = (lst_init + (t_acc/3.6e3) * NP.arange(n_acc)) * 15.0 # in degrees
        if verbose:
            print '\tCreated LST range for observing run.'

        if mode == 'track':
            if pointing_coords == 'hadec':
                pointing = NP.asarray([lst_init - pointing_init[0], pointing_init[1]])
            elif (pointing_coords == 'radec') or (pointing_coords is None):
                pointing = pointing_init
            elif pointing_coords == 'altaz':
                hadec = GEOM.altaz2hadec(pointing_init, self.latitude, units='degrees')
                pointing = NP.asarray([lst_init - hadec[0], hadec[1]])
            else:
                raise ValueError('pointing_coords can only be set to "hadec", "radec" or "altaz".')
            self.pointing_coords = 'radec'
        elif mode == 'drift':
            if pointing_coords == 'radec':
                pointing = NP.asarray([lst_init - pointing_init[0], pointing_init[1]])
            elif (pointing_coords == 'hadec') or (pointing_coords is None):
                pointing = pointing_init
            elif pointing_coords == 'altaz':
                pointing = GEOM.altaz2hadec(pointing_init, self.latitude, units='degrees')
            else:
                raise ValueError('pointing_coords can only be set to "hadec", "radec" or "altaz".')
            self.pointing_coords = 'hadec'

        if verbose:
            print '\tPreparing to observe in {0} mode'.format(mode)

        if verbose:
            milestones = range(max(1,int(n_acc/10)), int(n_acc), max(1,int(n_acc/10)))
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_acc).start()
        for i in xrange(n_acc):
            # if (verbose) and (i in milestones):
            #     print '\t\tObserving run {0:.1f} % complete...'.format(100.0*i/n_acc)
            timestamp = str(DT.datetime.now())
            self.observe(timestamp, Tsys[i], bpass[i,:], pointing, skymodel,
                         t_acc, brightness_units=brightness_units,
                         roi_radius=roi_radius, roi_center=roi_center,
                         lst=lst[i])

            if verbose:
                progress.update(i+1)

        if verbose:
            progress.finish()

        # if verbose:
        #     print '\t\tObserving run 100 % complete.'

        self.t_obs = duration
        self.n_acc = n_acc
        if verbose:
            print 'Observing run completed successfully.'

    ############################################################################

    def delay_transform(self, pad=1.0, freq_wts=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Transforms the visibilities from frequency axis onto delay (time) axis
        using an FFT. This is performed for noiseless sky visibilities, thermal
        noise in visibilities, and observed visibilities. 

        Inputs:

        pad         [scalar] Non-negative scalar indicating padding fraction 
                    relative to the number of frequency channels. For e.g., a 
                    pad of 1.0 pads the frequency axis with zeros of the same 
                    width as the number of channels. After the delay transform,
                    the transformed visibilities are downsampled by a factor of
                    1+pad. If a negative value is specified, delay transform 
                    will be performed with no padding

        freq_wts    [numpy vector or array] window shaping to be applied before
                    computing delay transform. It can either be a vector or size
                    equal to the number of channels (which will be applied to all
                    time instances) or a n_snapshots x nchan numpy array. Default
                    (None) will not apply windowing and only the inherent
                    bandpass will be used.

        verbose     [boolean] If set to True (default), print diagnostic and 
                    progress messages. If set to False, no such messages are
                    printed.
        -------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing to compute delay transform...\n\tChecking input parameters for compatibility...'

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        if freq_wts is not None:
            if freq_wts.size == len(self.channels):
                freq_wts = NP.repeat(freq_wts.reshape(1,-1), len(self.timestamp), axis=0)
            elif freq_wts.size == len(self.channels) * len(self.timestamp):
                freq_wts = freq_wts.reshape(len(self.timestamp), len(self.channels))
            else:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
            self.bp_wts = freq_wts
            if verbose:
                print '\tFrequency window weights assigned.'

        if verbose:
            print '\tInput parameters have been verified to be compatible.\n\tProceeding to compute delay transform.'

        self.lags = DSP.spectral_axis(len(self.channels), delx=self.freq_resolution, use_real=False, shift=True)
        if pad == 0.0:
            self.vis_lag = DSP.FT1D(self.vis_freq * self.bp * self.bp_wts, ax=1, use_real=False, shift=True) * self.freq_resolution
            self.skyvis_lag = DSP.FT1D(self.skyvis_freq * self.bp * self.bp_wts, ax=1, use_real=False, shift=True) * self.freq_resolution
            self.vis_noise_lag = DSP.FT1D(self.vis_noise_freq * self.bp * self.bp_wts, ax=1, use_real=False, shift=True) * self.freq_resolution
            if verbose:
                print '\tDelay transform computed without padding.'
        else:
            npad = int(len(self.channels) * pad)
            self.vis_lag = DSP.FT1D(NP.pad(self.vis_freq * self.bp * self.bp_wts, ((0,0),(0,npad)), mode='constant'), ax=1, use_real=False, shift=True) * self.freq_resolution
            self.skyvis_lag = DSP.FT1D(NP.pad(self.skyvis_freq * self.bp * self.bp_wts, ((0,0),(0,npad)), mode='constant'), ax=1, use_real=False, shift=True) * self.freq_resolution
            self.vis_noise_lag = DSP.FT1D(NP.pad(self.vis_noise_freq * self.bp * self.bp_wts, ((0,0),(0,npad)), mode='constant'), ax=1, use_real=False, shift=True) * self.freq_resolution
            if verbose:
                print '\tDelay transform computed with padding fraction {0:.1f}'.format(pad)
            self.vis_lag = DSP.downsampler(self.vis_lag, 1+pad, axis=1)
            self.skyvis_lag = DSP.downsampler(self.skyvis_lag, 1+pad, axis=1)
            self.vis_noise_lag = DSP.downsampler(self.vis_noise_lag, 1+pad, axis=1)
            if verbose:
                print '\tDelay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'delay_transform() completed successfully.'

    ############################################################################

    def band_averaged_noise_estimate(self, polydegree=4, filter_method='hpf',
                                     verbose=True):

        """
        ------------------------------------------------------------------------
        Given the attribute vis_freq, compute the thermal noise estimate (in Jy)
        in the data in each frequency channel. This uses the delay domain to 
        identify regions relatively free of foreground emission, fits a
        polynomial to remove any foreground contamination, further removes slow
        varying components in delay domain either by using a moving average
        window or a high pass filter to estimate the thermal noise form the
        resdiuals.

        Inputs:

        polydegree    [scalar integer] Positive integer denoting the degree of 
                      the polynomial to be fitted to the complex visibilities in
                      delay space beyond the horizon limit where foregrounds are
                      expected to be minimal. 

        filter_method [string] Filtering method to remove slow varying
                      components in the residuals of visibilities (after fitting
                      polynomial) along delay axis. Accepted values are 'hpf' 
                      (high pass filter) and 'ma' (moving average). 'hpf' uses a 
                      rectangular high pass filter to extract the high frequency
                      compoenents. 'ma' implements a moving average window and 
                      removes the slow varying components. Default = 'hpf' and 
                      is found to be superior in performance.
        
        verbose       [boolean] If set to True, prints progress and diagnostic 
                      messages. Default = True

        Output:

        A dictionary containing the following keys and associated information:
            'thermal_noise'       [scalar] statistical thermal noise rms estimate
                                  in visibilities in each channel averaged over 
                                  the entire bandwidth. Units = same as that of
                                  attribute vis_freq
            'foreground_noise'    [scalar] statistical foreground noise estimate 
                                  in visibilities in each channel averaged over 
                                  the entire bandwidth. Units = same as that of
                                  attribute vis_freq
            'fitted_lags'         [numpy array] Lags in delay domain (outside the
                                  horizon limit) where the visibilities were
                                  fitted using a polynomial. Number of rows =
                                  number of timestamps, number of columns =
                                  number of fitted lags
            'fitted_vis_lags'     [numpy array] visibilities for which polynomial
                                  fitting was performed for each timestamp 
                                  outside the horizon limit. Same size as the 
                                  data in the key 'fitted_lags'
            'polynomial_vis_lags' [numpy array] polynomial fitted visibilities 
                                  for the data in the key 'fitted_vis_lags' and 
                                  has the same size.
            'residuals'           [numpy array] Residuals in delay space after 
                                  polynomial fit. Same size as the data in key
                                  'fitted_lags'
            'hpf_residuals'       [numpy array] Fast varying compoenents of the
                                  residual visibilities computed as specified by 
                                  input filter_method. Same size as data in the 
                                  key 'residuals'
        ------------------------------------------------------------------------
        """

        
        if verbose:
            print 'Estimating noise in interferometer data...\n'
            print '\tChecking data compatibility...'

        if (self.lags is None) or (self.vis_lag is None):
            if self.vis_freq is None:
                raise NameError('Visiblities as a function of frequency is not available.')
            elif (self.channels is None) or (self.freq_resolution is None):
                raise NameError('Frequencies and/or frequency resolution not available')
            else:
                self.vis_lag = DSP.FT1D(self.vis_freq, ax=1, use_real=False, shift=False) * self.freq_resolution
                self.lags = DSP.spectral_axis(len(self.channels), delx=self.freq_resolution, use_real=False, shift=False)

        if polydegree < 0:
            raise ValueError('Degree of polynomial has to be non-negative.')

        if verbose:
            print '\tVerified data compatibility.'

        if self.pointing_coords == 'radec':
            pointing_center_hadec = NP.hstack(((NP.asarray(self.lst)-self.pointing_center[:,0]).reshape(-1,1),self.pointing_center[:,1].reshape(-1,1)))
            pointing_center_altaz = GEOM.hadec2altaz(pointing_center_hadec, self.latitude, units='degrees')
        elif self.pointing_coords == 'hadec':
            pointing_center_altaz = GEOM.hadec2altaz(self.pointing_center, self.latitude, units='degrees')
        else:
            pointing_center_altaz = self.pointing_center
        pointing_center_dircos = GEOM.altaz2dircos(pointing_center_altaz, units='degrees')
        
        delay_matrix = DLY.delay_envelope(self.baseline, pointing_center_dircos)
        horizon_lower_limit = delay_matrix[:,:,1] - delay_matrix[:,:,0]
        horizon_upper_limit = delay_matrix[:,:,1] + delay_matrix[:,:,0]
        horizon_limits = NP.hstack((horizon_lower_limit.reshape(-1,1), horizon_upper_limit.reshape(-1,1)))
        horizon_limit = self.baseline_length / FCNST.c

        if verbose:
            print '\tEstimated horizon limits in delay space.'

        if NP.any(NP.abs(delay_matrix[:,:,1]) > 0.5/len(self.channels)/self.freq_resolution):
            # No significant pointing center delays. All timestamps can be treated together

            right_inside_horizon_ind = NP.logical_and(self.lags >= 0.0, self.lags <= horizon_limit)
            left_inside_horizon_ind = NP.logical_and(self.lags < 0.0, self.lags >= -horizon_limit)
            right_outside_horizon_ind = self.lags > horizon_limit + 1.0/(len(self.channels) * self.freq_resolution)
            left_outside_horizon_ind = self.lags < -horizon_limit - 1.0/(len(self.channels) * self.freq_resolution)
            outside_horizon_ind = NP.abs(self.lags) > horizon_limit + 1.0/(len(self.channels) * self.freq_resolution)
            inside_horizon_ind = NP.abs(self.lags) <= horizon_limit 

            lags_outside_horizon = self.lags[outside_horizon_ind]
            vis_lag_outside_horizon = self.vis_lag[:,outside_horizon_ind]
            poly_vis_lag_outside_horizon = NP.empty_like(vis_lag_outside_horizon)

            if NP.iscomplexobj(self.vis_lag):
                right_real_polycoeffs = NP.polyfit(self.lags[right_outside_horizon_ind], self.vis_lag[:,right_outside_horizon_ind].real.T, polydegree)
                left_real_polycoeffs = NP.polyfit(self.lags[left_outside_horizon_ind], self.vis_lag[:,left_outside_horizon_ind].real.T, polydegree)
                right_imag_polycoeffs = NP.polyfit(self.lags[right_outside_horizon_ind], self.vis_lag[:,right_outside_horizon_ind].imag.T, polydegree)
                left_imag_polycoeffs = NP.polyfit(self.lags[left_outside_horizon_ind], self.vis_lag[:,left_outside_horizon_ind].imag.T, polydegree)
                if verbose:
                    print '\tFitted polynomials of degree {0:0d} to real and imaginary parts of the \n\t\tdelay spectrum outside the horizon limit'.format(polydegree)

                for timestamp in xrange(self.vis_lag.shape[0]):
                    lpr = NP.poly1d(left_real_polycoeffs[:,timestamp])
                    rpr = NP.poly1d(right_real_polycoeffs[:,timestamp])
                    lpi = NP.poly1d(left_imag_polycoeffs[:,timestamp])
                    rpi = NP.poly1d(right_imag_polycoeffs[:,timestamp])
                    poly_vis_lag_outside_horizon[timestamp, :] = NP.hstack(((lpr(self.lags[left_outside_horizon_ind]) + 1j * lpi(self.lags[left_outside_horizon_ind])).reshape(1,-1), (rpr(self.lags[right_outside_horizon_ind]) + 1j * rpi(self.lags[right_outside_horizon_ind])).reshape(1,-1)))

            else:
                right_polycoeffs = NP.polyfit(self.lags[right_outside_horizon_ind], self.vis_lag[:,right_outside_horizon_ind].T, polydegree)
                left_polycoeffs = NP.polyfit(self.lags[left_outside_horizon_ind], self.vis_lag[:,left_outside_horizon_ind].T, polydegree)
                if verbose:
                    print '\tFitted polynomials of degree {0:0d} to the delay spectrum outside the \n\t\thorizon limit'.format(polydegree)

                for timestamp in xrange(self.vis_lag.shape[0]):
                    lp = NP.poly1d(left_polycoeffs[:,timestamp])
                    rp = NP.poly1d(right_polycoeffs[:,timestamp])
                    poly_vis_lag_outside_horizon[timestamp, :] = NP.hstack((lp(self.lags[left_outside_horizon_ind]).reshape(1,-1), rp(self.lags[right_outside_horizon_ind]).reshape(1,-1)))
           
            if verbose:
                print '\tEstimated the fitted versions of the delay spectrum outside the horizon limit.'

            residuals = vis_lag_outside_horizon - poly_vis_lag_outside_horizon 
            if verbose:
                print '\tEstimated first round of residuals in the delay spectrum outside the horizon limit after polynomial fitting.'
                print '\tPreparing to remove slow varying components of residuals...'

            # wlen = NP.around(NP.sqrt(horizon_limit * self.freq_resolution * len(self.channels))) # number of delay bins as a geometric mean
            # if filter_method == 'ma':
            #     hpf_residuals = NP.empty_like(residuals)
            #     for timestamp in range(residuals.shape[0]):
            #         hpf_residuals[timestamp,:] = residuals[timestamp,:] - DSP.smooth(residuals[timestamp,:], width=wlen, stat='mean')
            # elif filter_method == 'hpf':
            #     wfrac = 1.0 - 1.0/(horizon_limit*len(self.channels)*self.freq_resolution) # High pass fraction
            #     # wfrac = 1.0/NP.sqrt(len(self.channels)*self.freq_resolution*horizon_limit) # width of high pass filter as a fraction of bandwidth as a geometric mean is equal to 1/wlen
            #     hpf_residuals = DSP.filter(residuals, width=wfrac, passband='high')

            wfrac = 1.0 - 1.0/(horizon_limit*len(self.channels)*self.freq_resolution) # High pass fraction
            # wfrac = 1.0/NP.sqrt(len(self.channels)*self.freq_resolution*horizon_limit) # width of high pass filter as a fraction of bandwidth as a geometric mean is equal to 1/wlen
            hpf_residuals = DSP.filter(residuals, width=wfrac, passband='high')

            thermal_noise_rms = NP.sqrt(NP.mean(NP.abs(hpf_residuals)**2, axis=1))
            # thermal_noise_rms = NP.sqrt(NP.median(NP.abs(hpf_residuals)**2, axis=1)) # median is used to reject outliers
            foreground_confusion_noise_rms = NP.sqrt((NP.mean(NP.abs(self.vis_lag[:,inside_horizon_ind])**2, axis=1)) - thermal_noise_rms**2)                
            thermal_noise_rms *= 1.0/(NP.sqrt(len(self.channels)) * self.freq_resolution)
            foreground_confusion_noise_rms *= 1.0/(len(self.channels) * self.freq_resolution)

            dictout = {}
            dictout['thermal_noise'] = thermal_noise_rms
            dictout['foreground_noise'] = foreground_confusion_noise_rms
            # dictout['fitted_lags'] = lags_outside_horizon
            # dictout['fitted_vis_lags'] = vis_lag_outside_horizon
            # dictout['polynomial_vis_lags'] = poly_vis_lag_outside_horizon
            # dictout['residuals'] = residuals
            # dictout['hpf_residuals'] = hpf_residuals

        else:
            # significant pointing center delays. All timestamps cannot be treated together
            
            thermal_noise_rms = NP.empty(len(self.timestamp))
            foreground_confusion_noise_rms = NP.empty(len(self.timestamp))
            for timestamp in xrange(self.vis_lag.shape[0]):

                right_outside_horizon_ind = self.lags > horizon_limits[timestamp,1] + 1.0/(len(self.channels) * self.freq_resolution)
                left_outside_horizon_ind = self.lags < horizon_limits[timestamp,0] - 1.0/(len(self.channels) * self.freq_resolution)
                outside_horizon_ind = NP.logical_or(self.lags > horizon_limits[timestamp,1] + 1.0/(len(self.channels) * self.freq_resolution), self.lags < horizon_limits[timestamp,0] - 1.0/(len(self.channels) * self.freq_resolution))
                inside_horizon_ind = NP.logical_and(self.lags > horizon_limits[timestamp,0] + 1.0/(len(self.channels) * self.freq_resolution), self.lags < horizon_limits[timestamp,1] - 1.0/(len(self.channels) * self.freq_resolution))

                lags_outside_horizon = self.lags[outside_horizon_ind]
                vis_lag_outside_horizon = self.vis_lag[timestamp,outside_horizon_ind]
                poly_vis_lag_outside_horizon = NP.empty_like(vis_lag_outside_horizon)

                if NP.iscomplexobj(self.vis_lag):

                    right_real_polycoeffs = NP.polyfit(self.lags[right_outside_horizon_ind], self.vis_lag[timestamp,right_outside_horizon_ind].real.T, polydegree)
                    left_real_polycoeffs = NP.polyfit(self.lags[left_outside_horizon_ind], self.vis_lag[timestamp,left_outside_horizon_ind].real.T, polydegree)
                    right_imag_polycoeffs = NP.polyfit(self.lags[right_outside_horizon_ind], self.vis_lag[timestamp,right_outside_horizon_ind].imag.T, polydegree)
                    left_imag_polycoeffs = NP.polyfit(self.lags[left_outside_horizon_ind], self.vis_lag[timestamp,left_outside_horizon_ind].imag.T, polydegree)

                    lpr = NP.poly1d(left_real_polycoeffs)
                    rpr = NP.poly1d(right_real_polycoeffs)
                    lpi = NP.poly1d(left_imag_polycoeffs)
                    rpi = NP.poly1d(right_imag_polycoeffs)
                    poly_vis_lag_outside_horizon = NP.hstack(((lpr(self.lags[left_outside_horizon_ind]) + 1j * lpi(self.lags[left_outside_horizon_ind])).reshape(1,-1), (rpr(self.lags[right_outside_horizon_ind]) + 1j * rpi(self.lags[right_outside_horizon_ind])).reshape(1,-1))) 

                else:

                    right_polycoeffs = NP.polyfit(self.lags[right_outside_horizon_ind], self.vis_lag[timestamp,right_outside_horizon_ind].T, polydegree)
                    left_polycoeffs = NP.polyfit(self.lags[left_outside_horizon_ind], self.vis_lag[timestamp,left_outside_horizon_ind].T, polydegree)

                    lp = NP.poly1d(left_polycoeffs)
                    rp = NP.poly1d(right_polycoeffs)
                    poly_vis_lag_outside_horizon = NP.hstack((lp(self.lags[left_outside_horizon_ind]).reshape(1,-1), rp(self.lags[right_outside_horizon_ind]).reshape(1,-1)))
                    
                residuals = vis_lag_outside_horizon - poly_vis_lag_outside_horizon

                wfrac = 1.0 - 1.0/(horizon_limit*len(self.channels)*self.freq_resolution) # High pass fraction
                # wfrac = 1.0/NP.sqrt(len(self.channels)*self.freq_resolution*horizon_limit) # width of high pass filter as a fraction of bandwidth as a geometric mean is equal to 1/wlen
                hpf_residuals = DSP.filter(residuals, width=wfrac, passband='high')

                thermal_noise_rms[timestamp] = NP.sqrt(NP.mean(NP.abs(hpf_residuals)**2))
                foreground_confusion_noise_rms[timestamp] = NP.sqrt(NP.mean(NP.abs(self.vis_lag[timestamp,inside_horizon_ind])**2) - thermal_noise_rms[timestamp]**2)                

            thermal_noise_rms *= 1.0/(NP.sqrt(len(self.channels)) * self.freq_resolution)
            foreground_confusion_noise_rms *= 1.0/(len(self.channels) * self.freq_resolution)
                
            dictout = {}
            dictout['thermal_noise'] = thermal_noise_rms
            dictout['foreground_noise'] = foreground_confusion_noise_rms

        return dictout
    
    #############################################################################

    def freq_differenced_noise_estimate(self):

        """
        -------------------------------------------------------------------------
        Estimates noise rms in each channel of frequency through frequency 
        differencing. Needs serious development.
        -------------------------------------------------------------------------
        """

        vis_diff = NP.diff(self.vis_freq, axis=1)
        band_avg_noise_info = self.band_averaged_noise_estimate()
        band_avg_noise_rms = band_avg_noise_info['thermal_noise']

        c = NP.zeros(len(self.channels))
        c[0] = 1.0
        r = NP.zeros(len(self.channels))
        r[:2] = 1.0
        matrix = toeplitz(c,r)
        matrix[-1,:] = 1.0

        stacked_matrix = NP.repeat(NP.expand_dims(matrix, axis=0), len(self.timestamp), axis=0)

        # noise_var = NP.empty_like(len(self.timestamp), len(self.channels))
        measurements = NP.hstack((NP.abs(vis_diff)**2, (len(self.channels)*band_avg_noise_rms**2).reshape(-1,1))) 

        noise_var = OPT.nnls(matrix, measurements[0,:])[0]

        # noise_var, residuals, rank, sv = LA.lstsq(matrix, measurements.T, rcond=1.0e-6)
        # noise_var = LA.solve(stacked_matrix, measurements)
        
        return noise_var

    #############################################################################
   
    def save(self, file, tabtype='BinTableHDU', overwrite=False, verbose=True):
        """
        ----------------------------------------------------------------------------
        Saves the interferometer information to disk. 

        file         [string] Filename with full path to be saved to. Will be
                     appended with '.fits' extension

        Keyword Input(s):

        tabtype      [string] indicates table type for one of the extensions in 
                     the FITS file. Allowed values are 'BinTableHDU' and 
                     'TableHDU' for binary ascii tables respectively. Default is
                     'BinTableHDU'.
                     
        overwrite    [boolean] True indicates overwrite even if a file already 
                     exists. Default = False (does not overwrite)
                     
        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        try:
            file
        except NameError:
            raise NameError('No filename provided. Aborting Interferometer.save()...')

        filename = file + '.' + self.label + '.fits' 

        if verbose:
            print '\nSaving information about interferometer...'

        hdulist = []

        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['label'] = (self.label, 'Interferometer label')
        hdulist[0].header['latitude'] = (self.latitude, 'Latitude of interferometer')
        hdulist[0].header['A_eff'] = (self.A_eff, 'Effective Area of interferometer')
        hdulist[0].header['Bx'] = (self.baseline[0,0], 'Baseline component along first axis (m)')
        hdulist[0].header['By'] = (self.baseline[0,1], 'Baseline component along second axis (m)')
        hdulist[0].header['Bz'] = (self.baseline[0,2], 'Baseline component along third axis (m)')
        hdulist[0].header['baseline_coords'] = (self.baseline_coords, 'Baseline coordinate system')
        hdulist[0].header['baseline_length'] = (self.baseline_length, 'Baseline length (m)')
        hdulist[0].header['efficiency'] = (self.eff_Q, 'Interferometer efficiency')
        hdulist[0].header['freq_resolution'] = (self.freq_resolution, 'Frequency Resolution (Hz)')
        hdulist[0].header['pointing_coords'] = (self.pointing_coords, 'Pointing coordinate system')
        hdulist[0].header['telescope'] = (self.telescope, 'Telescope')
        # hdulist[0].header['t_acc'] = (self.t_acc[0], 'Accumulation interval (s)')
        hdulist[0].header['t_obs'] = (self.t_obs, 'Observing duration (s)')
        hdulist[0].header['n_acc'] = (self.n_acc, 'Number of accumulations')        
        hdulist[0].header['flux_unit'] = (self.flux_unit, 'Unit of flux density')
        hdulist[0].header.set('EXTNAME', 'Interferometer ({0})'.format(self.label))

        if verbose:
            print '\tCreated a primary HDU.'

        cols = []
        if self.lst: 
            cols += [fits.Column(name='LST', format='D', array=NP.asarray(self.lst).ravel())]
            cols += [fits.Column(name='pointing_longitude', format='D', array=self.pointing_center[:,0])]
            cols += [fits.Column(name='pointing_latitude', format='D', array=self.pointing_center[:,1])]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'POINTING INFO')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated pointing information table.'

        cols = []
        cols += [fits.Column(name='frequency', format='D', array=self.channels)]
        if self.lags is not None:
            cols += [fits.Column(name='lag', format='D', array=self.lags)]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'SPECTRAL INFO')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated spectral information table.'

        if self.t_acc:
            hdulist += [fits.ImageHDU(self.t_acc, name='t_acc')]
            if verbose:
                print '\tCreated an extension for accumulation times.'

        if self.vis_rms_freq:
            hdulist += [fits.ImageHDU(self.vis_rms_freq, name='freq_channel_noise_rms_visibility')]
            if verbose:
                print '\tCreated an extension for simulated visibility noise rms per channel.'
        
        if self.vis_freq is not None:
            hdulist += [fits.ImageHDU(self.vis_freq.real, name='real_freq_obs_visibility')]
            hdulist += [fits.ImageHDU(self.vis_freq.imag, name='imag_freq_obs_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of observed visibility frequency spectrum of size {0[0]} x {0[1]} '.format(self.vis_freq.shape)

        if self.skyvis_freq is not None:
            hdulist += [fits.ImageHDU(self.skyvis_freq.real, name='real_freq_sky_visibility')]
            hdulist += [fits.ImageHDU(self.skyvis_freq.imag, name='imag_freq_sky_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of noiseless sky visibility frequency spectrum of size {0[0]} x {0[1]} '.format(self.skyvis_freq.shape)

        if self.vis_noise_freq is not None:
            hdulist += [fits.ImageHDU(self.vis_noise_freq.real, name='real_freq_noise_visibility')]
            hdulist += [fits.ImageHDU(self.vis_noise_freq.imag, name='imag_freq_noise_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of visibility noise frequency spectrum of size {0[0]} x {0[1]} '.format(self.vis_noise_freq.shape)

        hdulist += [fits.ImageHDU(self.bp, name='bandpass')]
        if verbose:
            print '\tCreated an extension for bandpass functions of size {0[0]} x {0[1]} as a function of snapshot instance and frequency'.format(self.bp.shape)

        if self.vis_lag is not None:
            hdulist += [fits.ImageHDU(self.vis_lag.real, name='real_lag_visibility')]
            hdulist += [fits.ImageHDU(self.vis_lag.imag, name='imag_lag_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of observed visibility delay spectrum of size {0[0]} x {0[1]} '.format(self.vis_lag.shape)

        if self.skyvis_lag is not None:
            hdulist += [fits.ImageHDU(self.skyvis_lag.real, name='real_lag_sky_visibility')]
            hdulist += [fits.ImageHDU(self.skyvis_lag.imag, name='imag_lag_sky_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of noiseless sky visibility delay spectrum of size {0[0]} x {0[1]} '.format(self.skyvis_lag.shape)

        if self.vis_noise_lag is not None:
            hdulist += [fits.ImageHDU(self.vis_noise_lag.real, name='real_lag_noise_visibility')]
            hdulist += [fits.ImageHDU(self.vis_noise_lag.imag, name='imag_lag_noise_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of visibility noise delay spectrum of size {0[0]} x {0[1]} '.format(self.vis_noise_lag.shape)

        if verbose:
            print '\tNow writing FITS file to disk...'

        hdu = fits.HDUList(hdulist)
        hdu.writeto(filename, clobber=overwrite)

        if verbose:
            print '\tInterferometer information written successfully to FITS file on disk:\n\t\t{0}\n'.format(filename)

#################################################################################

class InterferometerArray:

    """
    ----------------------------------------------------------------------------
    Class to manage information on a multi-element interferometer array. 

    Attributes:

    A_eff       [scalar, list or numpy vector] Effective area of the
                interferometers (in m^2). If a scalar is provided, it is assumed
                to be identical for all interferometers. Otherwise, one value
                must be specified for each interferometer. Default is
                 pi * (25/2)^2, appropriate for a 25 m VLA dish.

    baselines:  [M x 3 Numpy array] The baseline vectors associated with the
                M interferometers in SI units. The coordinate system of these
                vectors is specified by another attribute baseline_coords. 

    baseline_coords
                [string] Coordinate system for the baseline vectors. Default is 
                'localenu'. Other accepted values are 'equatorial' 

    baseline_lengths
                [M-element numpy array] Lengths of the baseline in SI units

    bp          [numpy array] Bandpass weights of size n_baselines x nchan x
                n_acc, where n_acc is the number of accumulations in the
                observation, nchan is the number of frequency channels, and
                n_baselines is the number of baselines

    channels    [list or numpy vector] frequency channels in Hz

    eff_Q       [scalar, list or numpy vector] Efficiency of the interferometers, 
                one value for each interferometer. Default = 0.89, appropriate for
                the VLA. Has to be between 0 and 1. If only a scalar value
                provided, it will be assumed to be identical for all the 
                interferometers. Otherwise, one value must be provided for each
                of the interferometers.

    freq_resolution
                [scalar] Frequency resolution (in Hz)

    labels:     [list] A unique identifier (preferably a string) for each of the 
                interferometers. 

    lags        [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as channels. This is 
                computed in member function delay_transform().

    latitude    [Scalar] Latitude of the interferometer's location. Default
                is 34.0790 degrees North corresponding to that of the VLA.

    lst         [list] List of LST (in degrees) for each timestamp

    n_acc       [scalar] Number of accumulations

    obs_catalog_indices
                [list of lists] Each element in the top list corresponds to a
                timestamp. Inside each top list is a list of indices of sources
                from the catalog which are observed inside the region of 
                interest. This is computed inside member function observe(). 

    pointing_center
                [2-column numpy array] Pointing center (latitude and 
                longitude) of the observation at a given timestamp. This is 
                where the telescopes will be phased up to as reference. 
                Coordinate system for the pointing_center is specified by another 
                attribute pointing_coords.

    phase_center
                [2-column numpy array] Phase center (latitude and 
                longitude) of the observation at a given timestamp. This is 
                where the telescopes will be phased up to as reference. 
                Coordinate system for the phase_center is specified by another 
                attribute phase_center_coords.

    pointing_coords
                [string] Coordinate system for telescope pointing. Accepted 
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz' 
                (Altitude-Azimuth). Default = 'hadec'.

    phase_center_coords
                [string] Coordinate system for array phase center. Accepted 
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz' 
                (Altitude-Azimuth). Default = 'hadec'.

    skycoords   [string] Coordinate system for the sky positions of sources.
                Accepted values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 
                'altaz' (Altitude-Azimuth). Default = 'radec'.
    
    skyvis_freq [numpy array] Complex visibility due to sky emission (in Jy or K) 
                along frequency axis for each interferometer estimated from the
                specified external catalog. Same size as vis_freq. Used in the
                member function observe(). Read its docstring for more details. 

    skyvis_lag  [numpy array] Complex visibility due to sky emission (in Jy Hz or
                K Hz) along the delay axis for each interferometer obtained by
                FFT of skyvis_freq along frequency axis. Same size as vis_freq.
                Created in the member function delay_transform(). Read its
                docstring for more details. 

    telescope   [string] The name of the telescope facility. Accepted values
                are 'vla', 'gmrt', 'mwa_dipole' and 'mwa'. Default = 'vla'

    timestamp   [list] List of timestamps during the observation

    t_acc       [list] Accumulation time (sec) corresponding to each timestamp

    t_obs       [scalar] Observing duration (sec)

    Tsys        [scalar, list or numpy vector] System temperature in Kelvin

    vis_freq    [numpy array] The simulated complex visibility (in Jy or K) 
                observed by each of the interferometers along frequency axis for 
                each timestamp of observation per frequency channel. It is the
                sum of skyvis_freq and vis_noise_freq. It can be either directly
                initialized or simulated in observe(). 

    vis_lag     [numpy array] The simulated complex visibility (in Jy Hz or K Hz) 
                along delay axis for each interferometer obtained by FFT of
                vis_freq along frequency axis. Same size as vis_noise_lag and
                skyis_lag. It is evaluated in member function delay_transform(). 

    vis_noise_freq
                [numpy array] Complex visibility noise (in Jy or K) generated 
                using an rms of vis_rms_freq along frequency axis for each 
                interferometer which is then added to the generated sky
                visibility. Same size as vis_freq. Used in the member function
                observe(). Read its docstring for more details. 

    vis_noise_lag
                [numpy array] Complex visibility noise (in Jy Hz or K Hz) along 
                delay axis for each interferometer generated using an FFT of
                vis_noise_freq along frequency axis. Same size as vis_noise_freq.
                Created in the member function delay_transform(). Read its
                docstring for more details. 

    vis_rms_freq
                [list of float] Theoretically estimated thermal noise rms (in Jy
                or K) in visibility measurements. Same size as vis_freq. This 
                will be estimated and used to inject simulated noise when a call 
                to member function observe() is made. Read the  docstring of 
                observe() for more details. The noise rms is estimated from the 
                instrument parameters as:
                (2 k T_sys / (A_eff x sqrt(2 x channel_width x t_acc))) / Jy, or
                T_sys / sqrt(2 x channel_width x t_acc)

    Member functions:

    __init__():        Initializes an instance of class InterferometerArray

    observe():         Simulates an observing run with the interferometer
                       specifications and an external sky catalog thus producing
                       visibilities. The simulation generates visibilities
                       observed by the interferometer for the specified
                       parameters.

    observing_run():   Simulate an extended observing run in 'track' or 'drift'
                       mode, by an instance of the InterferometerArray class, of
                       the sky when a sky catalog is provided. The simulation
                       generates visibilities observed by the interferometer
                       array for the specified parameters. Uses member function
                       observe() and builds the observation from snapshots. The
                       timestamp for each snapshot is the current time at which
                       the snapshot is generated.

    delay_transform(): Transforms the visibilities from frequency axis onto 
                       delay (time) axis using an FFT.

    noise_estimate():  Given the attribute vis_freq, compute the thermal noise 
                       estimate (in Jy or K) in the data in each frequency
                       channel

    ----------------------------------------------------------------------------
    """

    def __init__(self, labels, baselines, channels, telescope='vla', eff_Q=0.89,
                 latitude=34.0790, skycoords='radec', A_eff=NP.pi*(25.0/2)**2, 
                 pointing_coords='hadec', baseline_coords='localenu',
                 freq_scale=None, init_file=None):
        
        """
        ------------------------------------------------------------------------
        Intialize the InterferometerArray class which manages information on a 
        multi-element interferometer.

        Class attributes initialized are:
        labels, baselines, channels, telescope, latitude, skycoords, eff_Q, A_eff,
        pointing_coords, baseline_coords, baseline_lengths, channels, bp, bp_wts,
        freq_resolution, lags, lst, obs_catalog_indices, pointing_center,
        skyvis_freq, skyvis_lag, timestamp, t_acc, Tsys, vis_freq, vis_lag, 
        t_obs, n_acc, vis_noise_freq, vis_noise_lag, vis_rms_freq,
        geometric_delays.

        Read docstring of class InterferometerArray for details on these
        attributes.
        ------------------------------------------------------------------------
        """

        argument_init = False
        init_file_success = False
        if init_file is not None:
            try:
                hdulist = fits.open(init_file)
            except IOError:
                argument_init = True
                print '\tinit_file provided but could not open the initialization file. Attempting to initialize with input parameters...'

            extnames = [hdulist[i].header['EXTNAME'] for i in xrange(1,len(hdulist))]
            try:
                self.freq_resolution = hdulist[0].header['freq_resolution']
            except KeyError:
                hdulist.close()
                raise KeyError('Keyword "freq_resolution" not found in header.')

            try:
                self.latitude = hdulist[0].header['latitude']
            except KeyError:
                print '\tKeyword "latitude" not found in header. Assuming 34.0790 degrees for attribute latitude.'
                self.latitude = 34.0790
                
            try:
                self.telescope = hdulist[0].header['telescope']
            except KeyError:
                print '\tKeyword "telescope" not found in header. Assuming "vla" for attribute telescope.'
                self.telescope = 'vla'

            try:
                self.baseline_coords = hdulist[0].header['baseline_coords']
            except KeyError:
                print '\tKeyword "baseline_coords" not found in header. Assuming "localenu" for attribute baseline_coords.'
                self.baseline_coords = 'localenu'

            try:
                self.pointing_coords = hdulist[0].header['pointing_coords']
            except KeyError:
                print '\tKeyword "pointing_coords" not found in header. Assuming "hadec" for attribute pointing_coords.'
                self.pointing_coords = 'hadec'

            try:
                self.phase_center_coords = hdulist[0].header['phase_center_coords']
            except KeyError:
                print '\tKeyword "phase_center_coords" not found in header. Assuming "hadec" for attribute phase_center_coords.'
                self.phase_center_coords = 'hadec'

            try:
                self.skycoords = hdulist[0].header['skycoords']
            except KeyError:
                print '\tKeyword "skycoords" not found in header. Assuming "radec" for attribute skycoords.'
                self.skycoords = 'radec'

            try:
                self.flux_unit = hdulist[0].header['flux_unit']
            except KeyError:
                print '\tKeyword "flux_unit" not found in header. Assuming "jy" for attribute flux_unit.'
                self.skycoords = 'JY'

            if 'POINTING AND PHASE CENTER INFO' not in extnames:
                raise KeyError('No extension table found containing pointing information.')
            else:
                self.lst = hdulist['POINTING AND PHASE CENTER INFO'].data['LST'].tolist()
                self.pointing_center = NP.hstack((hdulist['POINTING AND PHASE CENTER INFO'].data['pointing_longitude'].reshape(-1,1), hdulist['POINTING AND PHASE CENTER INFO'].data['pointing_latitude'].reshape(-1,1)))
                self.phase_center = NP.hstack((hdulist['POINTING AND PHASE CENTER INFO'].data['phase_center_longitude'].reshape(-1,1), hdulist['POINTING AND PHASE CENTER INFO'].data['phase_center_latitude'].reshape(-1,1)))

            if 'TIMESTAMPS' in extnames:
                self.timestamp = hdulist['TIMESTAMPS'].data['timestamps'].tolist()
            else:
                raise KeyError('Extension named "TIMESTAMPS" not found in init_file.')

            if 'BASELINES' in extnames:
                self.baselines = hdulist['BASELINES'].data.reshape(-1,3)
                self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
            else:
                raise KeyError('Extension named "BASELINES" not found in init_file.')

            if 'LABELS' in extnames:
                self.labels = hdulist['LABELS'].data.tolist()
            else:
                self.labels = ['B{0:0d}'.format(i+1) for i in range(self.baseline_lengths.size)]

            if 'EFFECTIVE AREA' in extnames:
                self.A_eff = hdulist['EFFECTIVE AREA'].data
            else:
                raise KeyError('Extension named "EFFECTIVE AREA" not found in init_file.')

            if 'INTERFEROMETER EFFICIENCY' in extnames:
                self.eff_Q = hdulist['INTERFEROMETER EFFICIENCY'].data
            else:
                raise KeyError('Extension named "INTERFEROMETER EFFICIENCY" not found in init_file.')

            if 'SPECTRAL INFO' not in extnames:
                raise KeyError('No extension table found containing spectral information.')
            else:
                self.channels = hdulist['SPECTRAL INFO'].data['frequency']
                try:
                    self.lags = hdulist['SPECTRAL INFO'].data['lag']
                except KeyError:
                    self.lags = None

            if 'BANDPASS' in extnames:
                self.bp = hdulist['BANDPASS'].data
            else:
                raise KeyError('Extension named "BANDPASS" not found in init_file.')

            if 'BANDPASS_WEIGHTS' in extnames:
                self.bp_wts = hdulist['BANDPASS_WEIGHTS'].data
            else:
                self.bp_wts = NP.ones_like(self.bp)

            if 'T_ACC' in extnames:
                self.t_acc = hdulist['t_acc'].data.tolist()
                self.n_acc = len(self.t_acc)
                self.t_obs = sum(self.t_acc)
            else:
                raise KeyError('Extension named "T_ACC" not found in init_file.')
            
            if 'FREQ_CHANNEL_NOISE_RMS_VISIBILITY' in extnames:
                self.vis_rms_freq = hdulist['freq_channel_noise_rms_visibility'].data
            else:
                raise KeyError('Extension named "FREQ_CHANNEL_NOISE_RMS_VISIBILITY" not found in init_file.')

            if 'REAL_FREQ_OBS_VISIBILITY' in extnames:
                self.vis_freq = hdulist['real_freq_obs_visibility'].data
                if 'IMAG_FREQ_OBS_VISIBILITY' in extnames:
                    self.vis_freq = self.vis_freq.astype(NP.complex64)
                    self.vis_freq += 1j * hdulist['imag_freq_obs_visibility'].data
            else:
                raise KeyError('Extension named "REAL_FREQ_OBS_VISIBILITY" not found in init_file.')

            if 'REAL_FREQ_SKY_VISIBILITY' in extnames:
                self.skyvis_freq = hdulist['real_freq_sky_visibility'].data
                if 'IMAG_FREQ_SKY_VISIBILITY' in extnames:
                    self.skyvis_freq = self.skyvis_freq.astype(NP.complex64)
                    self.skyvis_freq += 1j * hdulist['imag_freq_sky_visibility'].data
            else:
                raise KeyError('Extension named "REAL_FREQ_SKY_VISIBILITY" not found in init_file.')

            if 'REAL_FREQ_NOISE_VISIBILITY' in extnames:
                self.vis_noise_freq = hdulist['real_freq_noise_visibility'].data
                if 'IMAG_FREQ_NOISE_VISIBILITY' in extnames:
                    self.vis_noise_freq = self.vis_noise_freq.astype(NP.complex64)
                    self.vis_noise_freq += 1j * hdulist['imag_freq_noise_visibility'].data
            else:
                raise KeyError('Extension named "REAL_FREQ_NOISE_VISIBILITY" not found in init_file.')

            if 'REAL_LAG_VISIBILITY' in extnames:
                self.vis_lag = hdulist['real_lag_visibility'].data
                if 'IMAG_LAG_VISIBILITY' in extnames:
                    self.vis_lag = self.vis_lag.astype(NP.complex64)
                    self.vis_lag += 1j * hdulist['imag_lag_visibility'].data
            else:
                self.vis_lag = None

            if 'REAL_LAG_SKY_VISIBILITY' in extnames:
                self.skyvis_lag = hdulist['real_lag_sky_visibility'].data
                if 'IMAG_LAG_SKY_VISIBILITY' in extnames:
                    self.skyvis_lag = self.skyvis_lag.astype(NP.complex64)
                    self.skyvis_lag += 1j * hdulist['imag_lag_sky_visibility'].data
            else:
                self.skyvis_lag = None

            if 'REAL_LAG_NOISE_VISIBILITY' in extnames:
                self.vis_noise_lag = hdulist['real_lag_noise_visibility'].data
                if 'IMAG_LAG_NOISE_VISIBILITY' in extnames:
                    self.vis_noise_lag = self.vis_noise_lag.astype(NP.complex64)
                    self.vis_noise_lag += 1j * hdulist['imag_lag_noise_visibility'].data
            else:
                self.vis_noise_lag = None

            hdulist.close()
            init_file_success = True
            return
        else:
            argument_init = True
            
        if (not argument_init) and (not init_file_success):
            raise ValueError('Initialization failed with the use of init_file.')

        self.baselines = NP.asarray(baselines)
        if len(self.baselines.shape) == 1:
            if self.baselines.size == 2:
                self.baselines = NP.hstack((self.baselines.reshape(1,-1), NP.zeros(1)))
            elif self.baselines.size == 3:
                self.baselines = self.baselines.reshape(1,-1)
            else:
                raise ValueError('Baseline(s) must be a 2- or 3-column array.')
        elif len(self.baselines.shape) == 2:
            if self.baselines.shape[1] == 2:
                self.baselines = NP.hstack((self.baselines, NP.zeros(self.baselines.shape[0]).reshape(-1,1)))
            elif self.baselines.shape[1] != 3:
                raise ValueError('Baseline(s) must be a 2- or 3-column array')
        else:
            raise ValueError('Baseline(s) array contains more than 2 dimensions.')

        self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
        self.baseline_orientations = NP.angle(self.baselines[:,0] + 1j * self.baselines[:,1])

        if not isinstance(labels, (list, tuple)):
            raise TypeError('Interferometer array labels must be a list or tuple of unique identifiers')
        elif len(labels) != self.baselines.shape[0]:
            raise ValueError('Number of labels do not match the number of baselines specified.')
        else:
            self.labels = labels

        self.telescope = telescope
        self.latitude = latitude
        self.vis_freq = None
        self.skyvis_freq = None
        # self.pb = None
        self.vis_noise_freq = None

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            self.channels = NP.asarray(channels)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            self.channels = NP.asarray(channels) * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            self.channels = NP.asarray(channels) * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            self.channels = NP.asarray(channels) * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')

        self.bp = NP.ones((self.baselines.shape[0],self.channels.size)) # Inherent bandpass shape
        self.bp_wts = NP.ones((self.baselines.shape[0],self.channels.size)) # Additional bandpass weights
        self.Tsys = NP.zeros((self.baselines.shape[0],self.channels.size))
        self.flux_unit = 'JY'
        self.timestamp = []
        self.t_acc = []
        self.t_obs = 0.0
        self.n_acc = 0
        self.pointing_center = NP.empty([1,2])
        self.phase_center = NP.empty([1,2])
        self.lst = []

        if isinstance(eff_Q, (int, float)):
            if (eff_Q >= 0.0) or (eff_Q <= 1.0):
                self.eff_Q = eff_Q * NP.ones((self.baselines.shape[0], self.channels.size))
            else:
                raise ValueError('Efficiency value of interferometer is invalid.')
        elif isinstance(eff_Q, (list, tuple, NP.ndarray)):
            eff_Q = NP.asarray(eff_Q)
            if (NP.any(eff_Q < 0.0)) or (NP.any(eff_Q > 1.0)):
                raise ValueError('One or more values of eff_Q found to be outside the range [0,1].')
            if eff_Q.size == self.baselines.shape[0]:
                self.eff_Q = NP.repeat(eff_Q.reshape(-1,1), self.channels.size, axis=1)
            elif eff_Q.size == self.channels.size:
                self.eff_Q = NP.repeat(eff_Q.reshape(1,-1), self.channels.size, axis=0)
            elif eff_Q.size == self.baselines.shape[0]*self.channels.size:
                self.eff_Q = eff_Q.reshape(-1,self.channels.size)
            else:
                raise ValueError('Efficiency values of interferometers incompatible with the number of interferometers and/or frequency channels.')
        else:
            raise TypeError('Efficiency values of interferometers must be provided as a scalar, list, tuple or numpy array.')

        if isinstance(A_eff, (int, float)):
            if A_eff >= 0.0:
                self.A_eff = A_eff * NP.ones((self.baselines.shape[0], self.channels.size))
            else:
                raise ValueError('Negative value for effective area is invalid.')
        elif isinstance(A_eff, (list, tuple, NP.ndarray)):
            A_eff = NP.asarray(A_eff)
            if NP.any(A_eff < 0.0):
                raise ValueError('One or more values of A_eff found to be negative.')
            if A_eff.size == self.baselines.shape[0]:
                self.A_eff = NP.repeat(A_eff.reshape(-1,1), self.channels.size, axis=1)
            elif A_eff.size == self.channels.size:
                self.A_eff = NP.repeat(A_eff.reshape(1,-1), self.channels.size, axis=0)
            elif A_eff.size == self.baselines.shape[0]*self.channels.size:
                self.A_eff = A_eff.reshape(-1,self.channels.size)
            else:
                raise ValueError('Effective area(s) of interferometers incompatible with the number of interferometers and/or frequency channels.')
        else:
            raise TypeError('Effective area(s) of interferometers must be provided as a scalar, list, tuple or numpy array.')

        self.vis_rms_freq = None
        self.freq_resolution = self.channels[1] - self.channels[0]
        self.baseline_coords = baseline_coords
        self.lags = None
        self.skyvis_lag = None
        self.vis_noise_lag = None
        self.vis_lag = None
        self.obs_catalog_indices = []
        self.geometric_delays = []

        if (pointing_coords == 'radec') or (pointing_coords == 'hadec') or (pointing_coords == 'altaz'):
            self.pointing_coords = pointing_coords
            self.phase_center_coords = pointing_coords
        else:
            raise ValueError('Pointing center of the interferometer must be "radec", "hadec" or "altaz". Check inputs.')

        if (skycoords == 'radec') or (skycoords == 'hadec') or (skycoords == 'altaz'):
            self.skycoords = skycoords
        else:
            raise ValueError('Sky coordinates must be "radec", "hadec" or "altaz". Check inputs.')

        if (baseline_coords == 'equatorial') or (baseline_coords == 'localenu'):
            self.baseline_coords = baseline_coords
        else:
            raise ValueError('Baseline coordinates must be "equatorial" or "local". Check inputs.')

    #############################################################################

    def observe(self, timestamp, Tsys, bandpass, pointing_center, skymodel,
                t_acc, brightness_units=None, roi_radius=None, roi_center=None,
                lst=None, memsave=False):

        """
        -------------------------------------------------------------------------
        Simulate a snapshot observation, by an instance of the 
        InterferometerArray class, of the sky when a sky catalog is provided. The 
        simulation generates visibilities observed by the interferometers for the 
        specified parameters. See member function observing_run() for simulating 
        an extended observing run in 'track' or 'drift' mode.

        Inputs:
        
        timestamp    [scalar] Timestamp associated with each integration in the
                     observation

        Tsys         [scalar, list, tuple or numpy array] System temperature(s)
                     associated with the interferometers for the specified
                     timestamp of observation. If a scalar value is provided, it 
                     will be assumed to be identical for all interferometers and
                     all frequencies. If a vector is provided whose length is
                     equal to the number of interferoemters, it will be assumed 
                     identical for all frequencies. If a vector is provided whose
                     length is equal to the number of frequency channels, it will
                     be assumed identical for all interferometers. If a 2D array
                     is provided, it should be of size n_baselines x nchan

        bandpass     [numpy array] Bandpass weights associated with the 
                     interferometers for the specified timestamp of observation

        pointing_center
                     [2-element numpy vector or list] Pointing center (latitude 
                     and longitude) of the observation at a given timestamp. 
                     This is where the telescopes will be phased up to as 
                     reference. Coordinate system for the pointing_center is 
                     specified by the attribute pointing_coords initialized in
                     __init__(). 

        skymodel     [instance of class SkyModel] It consists of source flux
                     densities, their positions, and spectral indices. Read 
                     class SkyModel docstring for more information.

        t_acc        [scalar] Accumulation time (sec) corresponding to timestamp

        brightness_units
                     [string] Units of flux density in the catalog and for the 
                     generated visibilities. Accepted values are 'Jy' (Jansky) 
                     and 'K' (Kelvin for temperature). If None set, it defaults 
                     to 'Jy'

        Keyword Inputs:

        roi_radius   [scalar] Radius of the region of interest (degrees) inside 
                     which sources are to be observed. Default = 90 degrees, 
                     which is the entire horizon.

        roi_center   [string] Center of the region of interest around which
                     roi_radius is used. Accepted values are 'pointing_center'
                     and 'zenith'. If set to None, it defaults to 'zenith'. 

        lst          [scalar] LST (in degrees) associated with the timestamp
        ------------------------------------------------------------------------
        """

        if len(bandpass.shape) == 1:
            if bandpass.size != self.channels.size:
                raise ValueError('Specified bandpass incompatible with the number of frequency channels')

            if len(self.bp.shape) == 2:
                self.bp = NP.expand_dims(NP.repeat(bandpass.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
            else:
                self.bp = NP.dstack((self.bp, NP.repeat(bandpass.reshape(1,-1), self.baselines.shape[0], axis=0)))
        elif len(bandpass.shape) == 2:
            if bandpass.shape[1] != self.channels.size:
                raise ValueError('Specified bandpass incompatible with the number of frequency channels')
            elif bandpass.shape[0] != self.baselines.shape[0]:
                raise ValueError('Specified bandpass incompatible with the number of interferometers')

            if len(self.bp.shape) == 2:
                self.bp = NP.expand_dims(bandpass, axis=2)
            else:
                self.bp = NP.dstack((self.bp, bandpass))
        elif len(bandpass.shape) == 3:
            if bandpass.shape[1] != self.channels.size:
                raise ValueError('Specified bandpass incompatible with the number of frequency channels')
            elif bandpass.shape[0] != self.baselines.shape[0]:
                raise ValueError('Specified bandpass incompatible with the number of interferometers')
            elif bandpass.shape[2] != 1:
                raise ValueError('Bandpass can have only one layer for this instance of accumulation.')

            if len(self.bp.shape) == 2:
                self.bp = bandpass
            else:
                self.bp = NP.dstack((self.bp, bandpass))

        self.bp_wts = NP.ones_like(self.bp) # All additional bandpass shaping weights are set to unity.

        if isinstance(Tsys, (int,float)):
            if Tsys < 0.0:
                raise ValueError('Tsys found to be negative.')
            
            if len(self.Tsys.shape) == 2:
                self.Tsys = Tsys + NP.zeros((self.baselines.shape[0], self.channels.size, 1))
            else:
                self.Tsys = NP.dstack((self.Tsys, Tsys + NP.zeros((self.baselines.shape[0], self.channels.size, 1))))
        elif isinstance(Tsys, (list, tuple, NP.ndarray)):
            Tsys = NP.asarray(Tsys)
            if NP.any(Tsys < 0.0):
                raise ValueError('Tsys should be non-negative.')

            if Tsys.size == self.baselines.shape[0]:
                if len(self.Tsys.shape) == 2:
                    self.Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(-1,1), self.channels.size, axis=1), axis=2)
                elif len(self.Tsys.shape) == 3:
                    self.Tsys = NP.dstack((self.Tsys, NP.expand_dims(NP.repeat(Tsys.reshape(-1,1), self.channels.size, axis=1), axis=2)))
            elif Tsys.size == self.channels.size:
                if len(self.Tsys.shape) == 2:
                    self.Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
                elif len(self.Tsys.shape) == 3:
                    self.Tsys = NP.dstack((self.Tsys, NP.expand_dims(NP.repeat(Tsys.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)))
            elif Tsys.size == self.baselines.shape[0]*self.channels.size:
                if len(self.Tsys.shape) == 2:
                    self.Tsys = NP.expand_dims(Tsys.reshape(-1,self.channels.size), axis=2)
                elif len(self.Tsys.shape) == 3:
                    self.Tsys = NP.dstack((self.Tsys, NP.expand_dims(Tsys.reshape(-1,self.channels.size), axis=2)))
            else:
                raise ValueError('Specified Tsys has incompatible dimensions with the number of baselines and/or number of frequency channels.')
        else:
            raise TypeError('Tsys should be a scalar, list, tuple, or numpy array')

        # if (brightness_units is None) or (brightness_units=='Jy') or (brightness_units=='JY') or (brightness_units=='jy'):
        #     if self.vis_rms_freq is None:
        #         self.vis_rms_freq = 2.0 * FCNST.k / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.A_eff/self.eff_Q, axis=2) / CNST.Jy
        #     elif len(self.vis_rms_freq.shape) == 3:
        #         self.vis_rms_freq = NP.dstack((self.vis_rms_freq, 2.0 * FCNST.k / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.A_eff/self.eff_Q, axis=2)/CNST.Jy))
        #     self.flux_unit = 'JY'
        # elif (brightness_units=='K') or (brightness_units=='k'):
        #     if len(self.vis_rms_freq.shape) == 2:
        #         self.vis_rms_freq = 1 / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.eff_Q, axis=2)
        #     elif len(self.vis_rms_freq.shape) == 3:
        #         self.vis_rms_freq = NP.dstack((self.vis_rms_freq, 1 / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.eff_Q, axis=2)))
        #     self.flux_unit = 'K'
        # else:
        #     raise ValueError('Invalid brightness temperature units specified.')

        self.t_acc = self.t_acc + [t_acc]
        self.t_obs = t_acc
        self.n_acc = 1
        self.lst = self.lst + [lst]

        if not self.timestamp:
            self.pointing_center = NP.asarray(pointing_center).reshape(1,-1)
            self.phase_center = NP.asarray(pointing_center).reshape(1,-1)
        else:
            self.pointing_center = NP.vstack((self.pointing_center, NP.asarray(pointing_center).reshape(1,-1)))
            self.phase_center = NP.vstack((self.phase_center, NP.asarray(pointing_center).reshape(1,-1)))

        pointing_lon = self.pointing_center[-1,0]
        pointing_lat = self.pointing_center[-1,1]

        if self.skycoords == 'radec':
            if self.pointing_coords == 'hadec':
                if lst is not None:
                    pointing_lon = lst - self.pointing_center[-1,0]
                    pointing_lat = self.pointing_center[-1,1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in RA-Dec format while pointing center is in HA-Dec format.')
            elif self.pointing_coords == 'altaz':
                pointing_lonlat = GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = lst - pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]
        elif self.skycoords == 'hadec':
            if self.pointing_coords == 'radec':
                if lst is not None:
                    pointing_lon = lst - self.pointing_center[-1,0]
                    pointing_lat = self.pointing_center[-1,1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in RA-Dec format while pointing center is in HA-Dec format.')
            elif self.pointing_coords == 'altaz':
                pointing_lonlat = lst - GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]
        else:
            if self.pointing_coords == 'radec':
                if lst is not None:
                    pointing_lonlat = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
                    pointing_lon = pointing_lonlat[0]
                    pointing_lat = pointing_lonlat[1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')
            elif self.pointing_coords == 'hadec':
                pointing_lonlat = GEOM.hadec2altaz(self.pointing_center,
                                                   self.latitude,
                                                   units='degrees')
                pointing_lon = pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]

        baselines_in_local_frame = self.baselines
        if self.baseline_coords == 'equatorial':
            baselines_in_local_frame = GEOM.xyz2enu(self.baselines, self.latitude, 'degrees')

        pc_altaz = self.pointing_center[-1,:] # Convert pointing center to Alt-Az coordinates
        if self.pointing_coords == 'hadec':
            pc_altaz = GEOM.hadec2altaz(self.pointing_center[-1,:], self.latitude, units='degrees')
        elif self.pointing_coords == 'radec':
            if lst is not None:
                pc_altaz = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
            else:
                raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')

        pc_dircos = GEOM.altaz2dircos(pc_altaz, 'degrees') # Convert pointing center to direction cosine coordinates
        pc_delay_offsets = DLY.geometric_delay(baselines_in_local_frame, pc_dircos, altaz=False, hadec=False, dircos=True, latitude=self.latitude)
        if memsave:
            pc_delay_offsets = pc_delay_offsets.astype(NP.float32)

        # pointing_phase = 2.0 * NP.pi * NP.repeat(NP.dot(baselines_in_local_frame, pc_dircos.reshape(-1,1)), self.channels.size, axis=1) * NP.repeat(self.channels.reshape(1,-1), self.baselines.shape[0], axis=0)/FCNST.c

        if not isinstance(skymodel, CTLG.SkyModel):
            raise TypeError('skymodel should be an instance of class SkyModel.')

        if roi_radius is None:
            roi_radius = 90.0

        if roi_center is None:
            roi_center = 'zenith'
        elif (roi_center != 'zenith') and (roi_center != 'pointing_center'):
            raise ValueError('Center of region of interest, roi_center, must be set to "zenith" or "pointing_center".')

        if roi_center == 'pointing_center':
            m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.catalog.location[:,0], skymodel.catalog.location[:,1], roi_radius, maxmatches=0)
        else: # roi_center = 'zenith'
            if self.skycoords == 'hadec':
                skypos_altaz = GEOM.hadec2altaz(skymodel.catalog.location, self.latitude, units='degrees')
            elif self.skycoords == 'radec':
                skypos_altaz = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.catalog.location[:,0]).reshape(-1,1), skymodel.catalog.location[:,1].reshape(-1,1))), self.latitude, units='degrees')
            m2 = NP.arange(skypos_altaz.shape[0])
            m2 = m2[NP.where(skypos_altaz[:,0] >= 90.0-roi_radius)] # select sources whose altitude (angle above horizon) is 90-roi_radius

        # if roi_radius is not None:
        #     m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.catalog.location[:,0], skymodel.catalog.location[:,1], roi_radius, maxmatches=0)
        # else:
        #     m1 = [0] * skymodel.catalog.location.shape[0]
        #     m2 = xrange(skymodel.catalog.location.shape[0])
        #     d12 = GEOM.sphdist(NP.empty(skymodel.catalog.shape[0]).fill(pointing_lon), NP.empty(skymodel.catalog.shape[0]).fill(pointing_lat), skymodel.catalog.location[:,0], skymodel.catalog.location[:,1])

        if len(m2) != 0:
            pb = NP.empty((len(m2), len(self.channels)))
            fluxes = NP.empty((len(m2), len(self.channels)))
            
            if roi_center != 'zenith':
                if self.skycoords == 'altaz':
                    skypos_altaz_roi = skymodel.catalog.location[m2,:]
                elif self.skycoords == 'radec':
                    skypos_altaz_roi = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.catalog.location[m2,0]).reshape(-1,1), skymodel.catalog.location[m2,1].reshape(-1,1))), self.latitude, 'degrees')
                else:
                    skypos_altaz_roi = GEOM.hadec2altaz(skymodel.catalog.location[m2,:], self.latitude, 'degrees')
            else:
                skypos_altaz_roi = skypos_altaz[m2,:]
            coords_str = 'altaz'

            pb = PB.primary_beam_generator(skypos_altaz_roi, self.channels/1.0e9, skyunits='altaz', telescope=self.telescope, phase_center=pc_altaz)
            # fluxes = NP.repeat(skymodel.catalog.flux_density[m2].reshape(-1,1), self.channels.size, axis=1) * (NP.repeat(self.channels.reshape(1,-1), len(m2), axis=0)/skymodel.catalog.frequency)**NP.repeat(skymodel.catalog.spectral_index[m2].reshape(-1,1), self.channels.size, axis=1)
            fluxes = skymodel.catalog.flux_density[m2].reshape(-1,1) * (self.channels.reshape(1,-1)/skymodel.catalog.frequency[m2].reshape(-1,1))**skymodel.catalog.spectral_index[m2].reshape(-1,1) # numpy array broadcasting

            pbfluxes = pb * fluxes
            geometric_delays = DLY.geometric_delay(baselines_in_local_frame, skypos_altaz_roi, altaz=(coords_str=='altaz'), hadec=(coords_str=='hadec'), latitude=self.latitude)

            vis_wts = None
            if skymodel.catalog.src_shape is not None:
                eps = 1.0e-13
                f0 = self.channels[self.channels.size/2]
                wl0 = FCNST.c / f0
                skypos_dircos_roi = GEOM.altaz2dircos(skypos_altaz_roi, units='degrees')
                # projected_spatial_frequencies = NP.sqrt(NP.repeat(self.baseline_lengths.reshape(1,-1)**2, len(m2), axis=0) - (FCNST.c * geometric_delays)**2) / wl0
                projected_spatial_frequencies = NP.sqrt(self.baseline_lengths.reshape(1,-1)**2 - (FCNST.c * geometric_delays)**2) / wl0
                src_FWHM = NP.sqrt(skymodel.catalog.src_shape[m2,0] * skymodel.catalog.src_shape[m2,1])
                src_FWHM_dircos = 2.0 * NP.sin(0.5*NP.radians(src_FWHM)).reshape(-1,1)
                # src_FWHM_dircos = NP.repeat(src_FWHM_dircos.reshape(-1,1), self.baselines.shape[0], axis=1)
                src_sigma_spatial_frequencies = 2.0 * NP.sqrt(2.0 * NP.log(2.0)) / (2 * NP.pi * src_FWHM_dircos)
                extended_sources_flag = 1/NP.clip(projected_spatial_frequencies, 0.5, NP.amax(projected_spatial_frequencies)) < src_FWHM_dircos
                vis_wts = NP.ones_like(projected_spatial_frequencies)
                # vis_wts[extended_sources_flag] = NP.exp(-0.5 * (projected_spatial_frequencies[extended_sources_flag]/src_sigma_spatial_frequencies[extended_sources_flag])**2)
                vis_wts = NP.exp(-0.5 * (projected_spatial_frequencies/src_sigma_spatial_frequencies)**2)
            
            if memsave:
                pbfluxes = pbfluxes.astype(NP.float32, copy=False)
                self.geometric_delays = self.geometric_delays + [geometric_delays.astype(NP.float32)]
                if vis_wts is not None:
                    vis_wts = vis_wts.astype(NP.float32, copy=False)
            else:
                self.geometric_delays = self.geometric_delays + [geometric_delays]

            if memsave:
                skyvis = NP.zeros((self.baselines.shape[0], self.channels.size), dtype=NP.complex64)
                memory_required = len(m2) * self.channels.size * self.baselines.shape[0] * 4.0 * 2 # bytes, 4 bytes per float, factor 2 is because the phase involves complex values
                # memory_required = len(m2) * self.channels.size * self.baselines.shape[0] * 4.0 * 2 * 2 # bytes, 4 bytes per float, factor 2 is because the phase involves complex values and another factor 2 for visibility weights
            else:
                skyvis = NP.zeros((self.baselines.shape[0], self.channels.size), dtype=NP.complex_)
                # memory_required = len(m2) * self.channels.size * self.baselines.shape[0] * 8.0 * 2 * 2 # bytes, 8 bytes per float, factor 2 is because the phase involves complex values and another factor 2 for visibility weights
                memory_required = len(m2) * self.channels.size * self.baselines.shape[0] * 8.0 * 2 # bytes, 8 bytes per float, factor 2 is because the phase involves complex values

            memory_available = OS.popen("free -b").readlines()[2].split()[3]
            if float(memory_available) > memory_required:
                if memsave:
                    # phase_matrix = NP.exp(-1j * NP.asarray(2.0 * NP.pi).astype(NP.float32) * NP.repeat(NP.expand_dims(self.geometric_delays[-1] - NP.repeat(pc_delay_offsets, len(m2), axis=0), axis=2), self.channels.size, axis=2) * NP.repeat(NP.expand_dims(NP.repeat(self.channels.astype(NP.float32).reshape(1,-1), self.baselines.shape[0], axis=0), axis=0), len(m2), axis=0)).astype(NP.complex64)
                    phase_matrix = NP.exp(-1j * NP.asarray(2.0 * NP.pi).astype(NP.float32) *  (self.geometric_delays[-1][:,:,NP.newaxis] - pc_delay_offsets.reshape(1,-1,1)) * self.channels.astype(NP.float32).reshape(1,1,-1)).astype(NP.complex64)
                    if vis_wts is not None:
                        # phase_matrix *= NP.repeat(NP.expand_dims(vis_wts, axis=2), self.channels.size, axis=2)
                        phase_matrix *= vis_wts[:,:,NP.newaxis]
                    # skyvis = NP.sum(NP.repeat(NP.expand_dims(pbfluxes, axis=1), self.baselines.shape[0], axis=1) * phase_matrix, axis=0) # Don't apply bandpass here
                    skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * phase_matrix, axis=0) # Don't apply bandpass here
                else:
                    # phase_matrix = 2.0 * NP.pi * NP.repeat(NP.expand_dims(self.geometric_delays[-1] - NP.repeat(pc_delay_offsets, len(m2), axis=0), axis=2), self.channels.size, axis=2) * NP.repeat(NP.expand_dims(NP.repeat(self.channels.reshape(1,-1), self.baselines.shape[0], axis=0), axis=0), len(m2), axis=0)
                    phase_matrix = 2.0 * NP.pi * (self.geometric_delays[-1][:,:,NP.newaxis] - pc_delay_offsets.reshape(1,-1,1)) * self.channels.reshape(1,1,-1)
                    if vis_wts is not None:
                        # skyvis = NP.sum(NP.repeat(NP.expand_dims(pbfluxes, axis=1), self.baselines.shape[0], axis=1) * NP.exp(-1j*phase_matrix) * NP.repeat(NP.expand_dims(vis_wts, axis=2), self.channels.size, axis=2), axis=0) # Don't apply bandpass here
                        skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * NP.exp(-1j*phase_matrix) * vis_wts[:,:,NP.newaxis], axis=0) # Don't apply bandpass here
                    else:
                        # skyvis = NP.sum(NP.repeat(NP.expand_dims(pbfluxes, axis=1), self.baselines.shape[0], axis=1) * NP.exp(-1j*phase_matrix), axis=0) # Don't apply bandpass here    
                        skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * NP.exp(-1j*phase_matrix), axis=0) # Don't apply bandpass here    
            else:
                print '\t\tDetecting memory shortage. Enforcing single precision computations.'
                # downsize_factor = 2*NP.ceil(memory_required/float(memory_available))
                downsize_factor = NP.ceil(memory_required/float(memory_available))
                n_src_stepsize = int(len(m2)/downsize_factor)
                src_indices = range(0,len(m2),n_src_stepsize)
                for i in xrange(len(src_indices)):
                    # phase_matrix = NP.exp(NP.asarray(-1j * 2.0 * NP.pi).astype(NP.complex64) * NP.repeat(NP.expand_dims(self.geometric_delays[-1][src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:].astype(NP.float32) - NP.repeat(pc_delay_offsets.astype(NP.float32), min(n_src_stepsize,len(m2)-src_indices[i]), axis=0), axis=2), self.channels.size, axis=2) * NP.repeat(NP.expand_dims(NP.repeat(self.channels.astype(NP.float32).reshape(1,-1), self.baselines.shape[0], axis=0), axis=0), min(n_src_stepsize,len(m2)-src_indices[i]), axis=0)).astype(NP.complex64, copy=False)
                    phase_matrix = NP.exp(NP.asarray(-1j * 2.0 * NP.pi).astype(NP.complex64) * (self.geometric_delays[-1][src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis].astype(NP.float32) - pc_delay_offsets.astype(NP.float32).reshape(1,-1,1)) * self.channels.astype(NP.float32).reshape(1,1,-1)).astype(NP.complex64, copy=False)
                    if vis_wts is not None:
                        # phase_matrix *= NP.repeat(NP.expand_dims(vis_wts[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:].astype(NP.float32), axis=2), self.channels.size, axis=2)
                        phase_matrix *= vis_wts[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis].astype(NP.float32)
                    # phase_matrix *= NP.repeat(NP.expand_dims(pbfluxes[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:].astype(NP.float32), axis=1), self.baselines.shape[0], axis=1)
                    phase_matrix *= pbfluxes[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),NP.newaxis,:].astype(NP.float32)
                    skyvis += NP.sum(phase_matrix, axis=0)
                    # skyvis += NP.sum(NP.repeat(NP.expand_dims(pbfluxes[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:], axis=1), self.baselines.shape[0], axis=1) * phase_matrix, axis=0)

            self.obs_catalog_indices = self.obs_catalog_indices + [m2]
        else:
            print 'No sources found in the catalog within matching radius. Simply populating the observed visibilities with noise.'
            skyvis = NP.zeros( (self.baselines.shape[0], self.channels.size) )

        if self.timestamp == []:
            self.skyvis_freq = skyvis[:,:,NP.newaxis]
            # self.vis_noise_freq = self.vis_rms_freq / NP.sqrt(2.0) * (NP.random.randn(self.baselines.shape[0], self.channels.size, 1) + 1j * NP.random.randn(self.baselines.shape[0], self.channels.size, 1)) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts
            # self.vis_freq = self.skyvis_freq + self.vis_noise_freq
        else:
            self.skyvis_freq = NP.dstack((self.skyvis_freq, skyvis[:,:,NP.newaxis]))
            # self.vis_noise_freq = NP.dstack((self.vis_noise_freq, NP.expand_dims(self.vis_rms_freq[:,:,-1],axis=2) / NP.sqrt(2.0) * (NP.random.randn(self.baselines.shape[0], self.channels.size, 1) + 1j * NP.random.randn(self.baselines.shape[0], self.channels.size, 1)) )) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts 
            # self.vis_freq = NP.dstack((self.vis_freq, NP.expand_dims(self.skyvis_freq[:,:,-1] + self.vis_noise_freq[:,:,-1], axis=2)))

        self.timestamp = self.timestamp + [timestamp]

    ############################################################################

    def observing_run(self, pointing_init, skymodel, t_acc, duration, channels, 
                      bpass, Tsys, lst_init, roi_radius=None, roi_center=None,
                      mode='track', pointing_coords=None, freq_scale=None,
                      brightness_units=None, verbose=True, memsave=False):

        """
        -------------------------------------------------------------------------
        Simulate an extended observing run in 'track' or 'drift' mode, by an
        instance of the InterferometerArray class, of the sky when a sky catalog 
        is provided. The simulation generates visibilities observed by the
        interferometer array for the specified parameters. Uses member function
        observe() and builds the observation from snapshots. The timestamp for
        each snapshot is the current time at which the snapshot is generated.

        Inputs:
        
        pointing_init [2-element list or numpy array] The inital pointing
                      of the telescope at the start of the observing run. 
                      This is where the telescopes will be initially phased up to
                      as reference. Coordinate system for the pointing_center is 
                      specified by the input pointing_coords 

        skymodel      [instance of class SkyModel] It consists of source flux
                      densities, their positions, and spectral indices. Read 
                      class SkyModel docstring for more information.

        t_acc         [scalar] Accumulation time (sec) corresponding to timestamp

        brightness_units
                      [string] Units of flux density in the catalog and for the 
                      generated visibilities. Accepted values are 'Jy' (Jansky) 
                      and 'K' (Kelvin for temperature). If None set, it defaults 
                      to 'Jy'

        duration      [scalar] Duration of observation in seconds

        channels      [list or numpy vector] frequency channels in units as 
                      specified in freq_scale

        bpass         [list, list of lists or numpy array] Bandpass weights in
                      the form of M x N array or list of N-element lists. N must
                      equal the number of channels. If M=1, the same bandpass
                      will be used in all the snapshots for the entire
                      observation, otherwise M must equal the number of
                      snapshots which is int(duration/t_acc)

        Tsys          [scalar, list or numpy array] System temperature (in K). If
                      a scalar is provided, the same Tsys will be used in all the
                      snapshots for the duration of the observation. If a list or
                      numpy array is provided, the number of elements must equal 
                      the number of snapshots which is int(duration/t_int)

        lst_init      [scalar] Initial LST (in degrees) at the beginning of the 
                      observing run corresponding to pointing_init

        Keyword Inputs:

        roi_radius    [scalar] Radius of the region of interest (degrees) inside 
                      which sources are to be observed. Default = 90 degrees, 
                      which is the entire horizon.
                      
        roi_center    [string] Center of the region of interest around which
                      roi_radius is used. Accepted values are 'pointing_center'
                      and 'zenith'. If set to None, it defaults to 'zenith'. 

        freq_scale    [string] Units of frequencies specified in channels. 
                      Accepted values are 'Hz', 'hz', 'khz', 'kHz', 'mhz',
                      'MHz', 'GHz' and 'ghz'. If None provided, defaults to 'Hz'

        mode          [string] Mode of observation. Accepted values are 'track'
                      and 'drift'. If using 'track', pointing center is fixed to
                      a specific point on the sky coordinate frame. If using 
                      'drift', pointing center is fixed to a specific point on
                      the antenna's reference frame. 

        pointing_coords
                      [string] Coordinate system for pointing_init. Accepted 
                      values are 'radec', 'hadec' and 'altaz'. If None provided,
                      default is set based on observing mode. If mode='track', 
                      pointing_coords defaults to 'radec', and if mode='drift', 
                      it defaults to 'hadec'

        verbose       [boolean] If set to True, prints progress and diagnostic 
                      messages. Default = True
        ------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing an observing run...\n'
            print '\tVerifying input arguments to observing_run()...'

        try:
            pointing_init, skymodel, t_acc, duration, bpass, Tsys, lst_init
        except NameError:
            raise NameError('One or more of pointing_init, skymodel, t_acc, duration, bpass, Tsys, lst_init not specified.')

        if isinstance(pointing_init, list):
            pointing_init = NP.asarray(pointing_init)
        elif not isinstance(pointing_init, NP.ndarray):
            raise TypeError('pointing_init must be a list or numpy array.')

        if pointing_init.size != 2:
            raise ValueError('pointing_init must be a 2-element vector.')
        pointing_init = pointing_init.ravel()

        if not isinstance(skymodel, CTLG.SkyModel):
            raise TypeError('skymodel must be an instance of class SkyModel.')

        if not isinstance(t_acc, (int, float)):
            raise TypeError('t_acc must be a scalar integer or float.')

        if t_acc <= 0.0:
            raise ValueError('t_acc must be positive.')

        if not isinstance(duration, (int, float)):
            raise TypeError('duration must be a scalar integer or float.')

        if duration <= t_acc:
            if verbose:
                print '\t\tDuration specified to be shorter than t_acc. Will set it equal to t_acc'
            duration = t_acc

        n_acc = int(duration / t_acc)
        if verbose:
            print '\t\tObserving run will have {0} accumulations.'.format(n_acc)

        if isinstance(channels, list):
            channels = NP.asarray(channels)
        elif not isinstance(channels, NP.ndarray):
            raise TypeError('channels must be a list or numpy array')

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            channels = NP.asarray(channels)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            channels = channels * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            channels = channels * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            channels = channels * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')

        if isinstance(bpass, (list, tuple, NP.ndarray)):
            bpass = NP.asarray(bpass)
        else:
            raise TypeError('bpass must be a list, tuple or numpy array')
        
        if bpass.size == self.channels.size:
            bpass = NP.expand_dims(NP.repeat(bpass.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
            if verbose:
                print '\t\tSame bandpass will be applied to all baselines and all accumulations in the observing run.'
        elif bpass.size == self.baselines.shape[0] * self.channels.size:
            bpass = NP.expand_dims(bpass.reshape(-1,self.channels.size), axis=2)
            if verbose:
                print '\t\tSame bandpass will be applied to all accumulations in the observing run.'
        elif bpass.size == self.baselines.shape[0] * self.channels.size * n_acc:
            bpass = bpass.reshape(-1,self.channels.size,n_acc)
        else:
            raise ValueError('Dimensions of bpass incompatible with the number of frequency channels, baselines and number of accumulations.')

        if isinstance(Tsys, (int, float, list, tuple, NP.ndarray)):
            Tsys = NP.asarray(Tsys).reshape(-1)
        else:
            raise TypeError('Tsys must be a scalar, list, tuple or numpy array')
        
        if Tsys.size == 1:
            if verbose:
                print '\t\tTsys = {0:.1f} K will be assumed for all frequencies, baselines, and accumulations.'.format(Tsys[0])
            Tsys = Tsys + NP.zeros((self.baselines.shape[0], self.channels.size, 1))
        elif Tsys.size == self.channels.size:
            Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
            if verbose:
                print '\t\tSame Tsys will be assumed for all baselines and all accumulations in the observing run.'
        elif Tsys.size == self.baselines.shape[0]:
            Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(-1,1), self.channels.size, axis=1), axis=2)
            if verbose:
                print '\t\tSame Tsys will be assumed for all frequency channels and all accumulations in the observing run.'
        elif Tsys.size == self.baselines.shape[0] * self.channels.size:
            Tsys = NP.expand_dims(Tsys.reshape(-1,self.channels.size), axis=2)
            if verbose:
                print '\t\tSame Tsys will be assumed for all accumulations in the observing run.'
        elif Tsys.size == self.baselines.shape[0] * self.channels.size * n_acc:
            Tsys = Tsys.reshape(-1,self.channels.size,n_acc)
        else:
            raise ValueError('Dimensions of Tsys incompatible with the number of frequency channels, baselines and number of accumulations.')

        if not isinstance(lst_init, (int, float)):
            raise TypeError('Starting LST should be a scalar')

        if verbose:
            print '\tVerified input arguments.'
            print '\tProceeding to schedule the observing run...'

        lst = (lst_init + (t_acc/3.6e3) * NP.arange(n_acc)) * 15.0 # in degrees
        if verbose:
            print '\tCreated LST range for observing run.'

        if mode == 'track':
            if pointing_coords == 'hadec':
                pointing = NP.asarray([lst_init - pointing_init[0], pointing_init[1]])
            elif (pointing_coords == 'radec') or (pointing_coords is None):
                pointing = pointing_init
            elif pointing_coords == 'altaz':
                hadec = GEOM.altaz2hadec(pointing_init, self.latitude, units='degrees')
                pointing = NP.asarray([lst_init - hadec[0], hadec[1]])
            else:
                raise ValueError('pointing_coords can only be set to "hadec", "radec" or "altaz".')
            self.pointing_coords = 'radec'
            self.phase_center_coords = 'radec'
        elif mode == 'drift':
            if pointing_coords == 'radec':
                pointing = NP.asarray([lst_init - pointing_init[0], pointing_init[1]])
            elif (pointing_coords == 'hadec') or (pointing_coords is None):
                pointing = pointing_init
            elif pointing_coords == 'altaz':
                pointing = GEOM.altaz2hadec(pointing_init, self.latitude, units='degrees')
            else:
                raise ValueError('pointing_coords can only be set to "hadec", "radec" or "altaz".')
            self.pointing_coords = 'hadec'
            self.phase_center_coords = 'hadec'

        if verbose:
            print '\tPreparing to observe in {0} mode'.format(mode)
            
        if verbose:
            milestones = range(max(1,int(n_acc/10)), int(n_acc), max(1,int(n_acc/10)))
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_acc).start()
        for i in range(n_acc):
            # if (verbose) and (i in milestones):
            #     print '\t\tObserving run {0:.1f} % complete...'.format(100.0*i/n_acc)
            timestamp = str(DT.datetime.now())
            self.observe(timestamp, Tsys[:,:,i%Tsys.shape[2]],
                         bpass[:,:,i%bpass.shape[2]], pointing, skymodel,
                         t_acc, brightness_units=brightness_units,
                         roi_radius=roi_radius, roi_center=roi_center,
                         lst=lst[i], memsave=memsave)
            if verbose:
                progress.update(i+1)

        if verbose:
            progress.finish()

        # if verbose:
        #     print '\t\tObserving run 100 % complete.'

        self.t_obs = duration
        self.n_acc = n_acc
        if verbose:
            print 'Observing run completed successfully.'

    #############################################################################

    def generate_noise(self):
        
        """
        -------------------------------------------------------------------------
        Generates thermal noise from attributes that describe system parameters 
        which can be added to sky visibilities
        -------------------------------------------------------------------------
        """

        eff_Q = self.eff_Q
        A_eff = self.A_eff
        t_acc = NP.asarray(self.t_acc)

        if len(eff_Q.shape) == 2:
            eff_Q = eff_Q[:,:,NP.newaxis]
        if len(A_eff.shape) == 2:
            A_eff = A_eff[:,:,NP.newaxis]
        t_acc = t_acc[NP.newaxis,NP.newaxis,:]

        if (self.flux_unit == 'JY') or (self.flux_unit == 'jy') or (self.flux_unit == 'Jy'):
            self.vis_rms_freq = 2.0 * FCNST.k / NP.sqrt(2.0*t_acc*self.freq_resolution) * (self.Tsys/A_eff/eff_Q) / CNST.Jy
        elif (self.flux_unit == 'K') or (self.flux_unit == 'k'):
            self.vis_rms_freq = 1 / NP.sqrt(2.0*t_acc*self.freq_resolution) * self.Tsys/eff_Q
        else:
            raise ValueError('Flux density units can only be in Jy or K.')

        self.vis_noise_freq = self.vis_rms_freq / NP.sqrt(2.0) * (NP.random.randn(self.baselines.shape[0], self.channels.size, len(self.timestamp)) + 1j * NP.random.randn(self.baselines.shape[0], self.channels.size, len(self.timestamp))) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts

    #############################################################################

    def add_noise(self):

        """
        -------------------------------------------------------------------------
        Adds the thermal noise generated in member function generate_noise() to
        the sky visibilities
        -------------------------------------------------------------------------
        """
        
        self.vis_freq = self.skyvis_freq + self.vis_noise_freq

    #############################################################################

    def phase_centering(self, phase_center=None, phase_center_coords=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Centers the phase of visibilities around any given phase center.

        Inputs:
        
        phase_center  [numpy array] Mx2 or Mx3 numpy array specifying phase
                      centers for each timestamp in the observation. Deafault is 
                      None (No phase rotation of visibilities). M can be 1 
                      or equal to the number of timestamps in the observation. If
                      M=1, the same phase center is assumed for all the
                      timestamps in the observation and visibility phases are
                      centered accordingly. If M = number of timestamps, each 
                      timestamp is rotated by the corresponding phase center. If
                      phase center coordinates are specified in 'altaz', 'hadec'
                      or 'radec' coordinates, it is a 2-column array. If
                      specified in 'dircos' coordinates, it can be a 2-column or 
                      3-column array following rules of direction cosines. If a
                      2-column array of direction cosines is provided, the third
                      column is automatically generated. The coordinates of phase
                      center are provided by the other input phase_center_coords.
        
        phase_center_coords
                      [string scalar] Coordinate system of phase cneter. It can 
                      be 'altaz', 'radec', 'hadec' or 'dircos'. Default = None.
                      phase_center_coords must be provided.

        verbose:      [boolean] If set to True (default), prints progress and
                      diagnostic messages.
        -------------------------------------------------------------------------
        """

        if phase_center is None:
            print 'No Phase center provided.'
            return
        elif not isinstance(phase_center, NP.ndarray):
            raise TypeError('Phase center must be a numpy array')
        elif phase_center.shape[0] == 1:
            phase_center = NP.repeat(phase_center, len(self.lst), axis=0)
        elif phase_center.shape[0] != len(self.lst):
            raise ValueError('One phase center must be provided for every timestamp.')

        phase_center_current = self.phase_center + 0.0
        phase_center_new = phase_center + 0.0
        phase_center_coords_current = self.phase_center_coords + ''
        phase_center_coords_new = phase_center_coords + ''
        phase_center_temp = phase_center_new + 0.0
        phase_center_coords_temp = phase_center_coords_new + ''

        if phase_center_coords_new is None:
            raise NameError('Coordinates of phase center not provided.')
        elif phase_center_coords_new == 'dircos':
            if (phase_center_new.shape[1] < 2) or (phase_center_new.shape[1] > 3):
                raise ValueError('Dimensions incompatible for direction cosine positions')
            if NP.any(NP.sqrt(NP.sum(phase_center_new**2, axis=1)) > 1.0):
                raise ValueError('direction cosines found to be exceeding unit magnitude.')
            if phase_center_new.shape[1] == 2:
                n = 1.0 - NP.sqrt(NP.sum(phase_center_new**2, axis=1))
                phase_center_new = NP.hstack((phase_center_new, n.reshape(-1,1)))
            phase_center_temp = phase_center_new + 0.0
            phase_center_coords_temp = 'dircos'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.dircos2altaz(phase_center_temp, units='degrees')
                phase_center_coords_temp = 'altaz'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.altaz2hadec(phase_center_temp, self.latitude, units='degrees')
                phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'radec'
            if phase_center_coords_temp != phase_center_coords_current:
                raise ValueError('Pointing coordinates of interferometer array instance invalid.')
        elif phase_center_coords_new == 'altaz':
            phase_center_temp = phase_center_new + 0.0
            phase_center_coords_temp = 'altaz'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.altaz2hadec(phase_center_temp, self.latitude, units='degrees')
                phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'radec'
            if phase_center_coords_temp != phase_center_coords_current:
                raise ValueError('Pointing coordinates of interferometer array instance invalid.')
            phase_center_coords_temp = phase_center_coords_current + ''
            phase_center_new = GEOM.altaz2dircos(phase_center_new, units='degrees')
        elif phase_center_coords_new == 'hadec':
            phase_center_temp = phase_center_new + 0.0
            phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                if self.pointing_coords == 'radec':
                    phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                    phase_center_coords_temp = 'radec'
                else:
                    phase_center_temp = GEOM.hadec2altaz(phase_center_temp, self.latitude, units='degrees')
                    phase_center_coords_temp = 'altaz'
                    if phase_center_coords_temp != phase_center_coords_current:
                        phase_center_temp = GEOM.altaz2dircos(phase_center_temp, units='degrees')
                        phase_center_coords_temp = 'dircos'
                        if phase_center_coords_temp != phase_center_coords_current:
                            raise ValueError('Pointing coordinates of interferometer array instance invalid.')
            phase_center_new = GEOM.hadec2altaz(phase_center_new, self.latitude, units='degrees')
            phase_center_new = GEOM.altaz2dircos(phase_center_new, units='degrees')
        elif phase_center_coords_new == 'radec':
            phase_center_temp = phase_center_new + 0.0
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'hadec'

            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.hadec2altaz(phase_center_temp, self.latitude, units='degrees')
                phase_center_coords_temp = 'altaz'

            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.altaz2dircos(phase_center_temp, units='degrees')
                phase_center_coords_temp = 'dircos'

            if phase_center_coords_temp != phase_center_coords_current:
                raise ValueError('Pointing coordinates of interferometer array instance invalid.')

            phase_center_new[:,0] = self.lst - phase_center_new[:,0]
            phase_center_new = GEOM.hadec2altaz(phase_center_new, self.latitude, units='degrees')
            phase_center_new = GEOM.altaz2dircos(phase_center_new, units='degrees')
        else:
            raise ValueError('Invalid phase center coordinate system specified')

        phase_center_current_temp = phase_center_current + 0.0
        phase_center_coords_current_temp = phase_center_coords_current + ''
        if phase_center_coords_current_temp == 'radec':
            phase_center_current_temp[:,0] = self.lst - phase_center_current_temp[:,0]
            phase_center_coords_current_temp = 'hadec'
        if phase_center_coords_current_temp == 'hadec':
            phase_center_current_temp = GEOM.hadec2altaz(phase_center_current_temp, self.latitude, units='degrees')
            phase_center_coords_current_temp = 'altaz'
        if phase_center_coords_current_temp == 'altaz':
            phase_center_current_temp = GEOM.altaz2dircos(phase_center_current_temp, units='degrees')
            phase_center_coords_current_temp = 'dircos'

        pos_diff_dircos = phase_center_current_temp - phase_center_new 
        b_dot_l = NP.dot(self.baselines, pos_diff_dircos.T)

        self.phase_center = phase_center_temp + 0.0
        self.phase_center_coords = phase_center_coords_temp + ''

        self.vis_freq = self.vis_freq * NP.exp(-1j * 2 * NP.pi * b_dot_l[:,NP.newaxis,:] * self.channels.reshape(1,-1,1) / FCNST.c)
        self.skyvis_freq = self.skyvis_freq * NP.exp(-1j * 2 * NP.pi * b_dot_l[:,NP.newaxis,:] * self.channels.reshape(1,-1,1) / FCNST.c)
        self.vis_noise_freq = self.vis_noise_freq * NP.exp(-1j * 2 * NP.pi * b_dot_l[:,NP.newaxis,:] * self.channels.reshape(1,-1,1) / FCNST.c)
        self.delay_transform()
        print 'Running delay_transform() with defaults inside phase_centering() after rotating visibility phases. Run delay_transform() again with appropriate inputs.'

    #############################################################################

    def delay_transform(self, pad=1.0, freq_wts=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Transforms the visibilities from frequency axis onto delay (time) axis
        using an IFFT. This is performed for noiseless sky visibilities, thermal
        noise in visibilities, and observed visibilities. 

        Inputs:

        pad         [scalar] Non-negative scalar indicating padding fraction 
                    relative to the number of frequency channels. For e.g., a 
                    pad of 1.0 pads the frequency axis with zeros of the same 
                    width as the number of channels. After the delay transform,
                    the transformed visibilities are downsampled by a factor of
                    1+pad. If a negative value is specified, delay transform 
                    will be performed with no padding

        freq_wts    [numpy vector or array] window shaping to be applied before
                    computing delay transform. It can either be a vector or size
                    equal to the number of channels (which will be applied to all
                    time instances for all baselines), or a nchan x n_snapshots 
                    numpy array which will be applied to all baselines, or a 
                    n_baselines x nchan numpy array which will be applied to all 
                    timestamps, or a n_baselines x nchan x n_snapshots numpy 
                    array. Default (None) will not apply windowing and only the
                    inherent bandpass will be used.

        verbose     [boolean] If set to True (default), print diagnostic and 
                    progress messages. If set to False, no such messages are
                    printed.
        ------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing to compute delay transform...\n\tChecking input parameters for compatibility...'

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        if freq_wts is not None:
            if freq_wts.size == self.channels.size:
                freq_wts = NP.repeat(NP.expand_dims(NP.repeat(freq_wts.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.channels.size * self.n_acc:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(self.channels.size, -1), axis=0), self.baselines.shape[0], axis=0)
            elif freq_wts.size == self.channels.size * self.baselines.shape[0]:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(-1, self.channels.size), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.channels.size * self.baselines.shape[0] * self.n_acc:
                freq_wts = freq_wts.reshape(self.baselines.shape[0], self.channels.size, self.n_acc)
            else:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
            self.bp_wts = freq_wts
            if verbose:
                print '\tFrequency window weights assigned.'

        if verbose:
            print '\tInput parameters have been verified to be compatible.\n\tProceeding to compute delay transform.'
            
        self.lags = DSP.spectral_axis(self.channels.size, delx=self.freq_resolution, use_real=False, shift=True)
        if pad == 0.0:
            self.vis_lag = DSP.FT1D(self.vis_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True)
            self.skyvis_lag = DSP.FT1D(self.skyvis_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True)
            self.vis_noise_lag = DSP.FT1D(self.vis_noise_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True)
            if verbose:
                print '\tDelay transform computed without padding.'
        else:
            npad = int(self.channels.size * pad)
            self.vis_lag = DSP.FT1D(NP.pad(self.vis_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True)
            self.skyvis_lag = DSP.FT1D(NP.pad(self.skyvis_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True)
            self.vis_noise_lag = DSP.FT1D(NP.pad(self.vis_noise_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True)
            if verbose:
                print '\tDelay transform computed with padding fraction {0:.1f}'.format(pad)
            self.vis_lag = (1+npad*1.0/self.channels.size) * DSP.downsampler(self.vis_lag, 1+pad, axis=1)
            self.skyvis_lag = (1+npad*1.0/self.channels.size) * DSP.downsampler(self.skyvis_lag, 1+pad, axis=1)
            self.vis_noise_lag = (1+npad*1.0/self.channels.size) * DSP.downsampler(self.vis_noise_lag, 1+pad, axis=1)
            if verbose:
                print '\tDelay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'delay_transform() completed successfully.'

    #############################################################################

    def concatenate(self, others, axis):
        """
        -------------------------------------------------------------------------
        Concatenates different visibility data sets from instances of class
        InterferometerArray along baseline, frequency or time axis.

        Inputs:

        others       [instance of class Interferometer Array or list of such 
                     instances] Instance or list of instances of class
                     InterferometerArray whose visibility data have to be 
                     concatenated to the current instance.

        axis         [scalar] Axis along which visibility data sets are to be
                     concatenated. Accepted values are 0 (concatenate along
                     baseline axis), 1 (concatenate frequency channels), or 2 
                     (concatenate along time/snapshot axis). Default=None
        -------------------------------------------------------------------------
        """

        try:
            others, axis
        except NameError:
            raise NameError('An instance of class InterferometerArray or a list of such instances and the axis along which they are to be concatenated must be provided.')

        if isinstance(others, list):
            for other in others:
                if not isinstance(other, InterferometerArray):
                    raise TypeError('The interferometer array data to be concatenated must be an instance of class InterferometerArray or a list of such instances')
            loo = [self]+others
        elif isinstance(others, InterferometerArray):
            loo = [self, others]
        elif not isinstance(other, InterferometerArray):
            raise TypeError('The interferometer array data to be concatenated must be an instance of class InterferometerArray or a list of such instances')
            
        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')

        self_shape = self.skyvis_freq.shape

        if axis >= len(self_shape):
            raise ValueError('Specified axis not found in the visibility data.')
        elif axis == -1:
            axis = len(self_shape) - 1
        elif axis < -1:
            raise ValueError('Specified axis not found in the visibility data.')

        self.skyvis_freq = NP.concatenate(tuple([elem.skyvis_freq for elem in loo]), axis=axis)
        self.vis_freq = NP.concatenate(tuple([elem.vis_freq for elem in loo]), axis=axis)
        self.vis_noise_freq = NP.concatenate(tuple([elem.vis_noise_freq for elem in loo]), axis=axis)
        self.vis_rms_freq  = NP.concatenate(tuple([elem.vis_rms_freq for elem in loo]), axis=axis)
        self.bp = NP.concatenate(tuple([elem.bp for elem in loo]), axis=axis)
        self.bp_wts = NP.concatenate(tuple([elem.bp_wts for elem in loo]), axis=axis)
        # self.Tsys = NP.concatenate(tuple([elem.Tsys for elem in loo]), axis=axis)
        if axis != 1:
            self.skyvis_lag = NP.concatenate(tuple([elem.skyvis_lag for elem in loo]), axis=axis)
            self.vis_lag = NP.concatenate(tuple([elem.vis_lag for elem in loo]), axis=axis)
            self.vis_noise_lag = NP.concatenate(tuple([elem.vis_noise_lag for elem in loo]), axis=axis)

        if axis == 0: # baseline axis
            for elem in loo:
                if elem.baseline_coords != self.baseline_coords:
                    raise ValueError('Coordinate systems for the baseline vectors are mismatched.')
            self.baselines = NP.vstack(tuple([elem.baselines for elem in loo]))
            self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
            self.baseline_orientations = NP.angle(self.baselines[:,0] + 1j * self.baselines[:,1])
            self.labels = [label for elem in loo for label in elem.labels]
            self.A_eff = NP.vstack(tuple([elem.A_eff for elem in loo]))
            self.eff_Q = NP.vstack(tuple([elem.eff_Q for elem in loo]))
        elif axis == 1: # Frequency axis
            self.channels = NP.hstack(tuple([elem.channels for elem in loo]))
            self.A_eff = NP.hstack(tuple([elem.A_eff for elem in loo]))
            self.eff_Q = NP.hstack(tuple([elem.eff_Q for elem in loo]))
            # self.delay_transform()
        elif axis == 2: # time axis
            self.timestamp = [timestamp for elem in loo for timestamp in elem.timestamp]
            self.t_acc = [t_acc for elem in loo for t_acc in elem.t_acc]
            self.n_acc = len(self.t_acc)
            self.t_obs = sum(self.t_acc)
            self.pointing_center = NP.vstack(tuple([elem.pointing_center for elem in loo]))
            self.phase_center = NP.vstack(tuple([elem.phase_center for elem in loo]))
            self.lst = [lst for elem in loo for lst in elem.lst]
            self.timestamp = [timestamp for elem in loo for timestamp in elem.timestamp]
            # self.obs_catalog_indices = [elem.obs_catalog_indices for elem in loo]

    #############################################################################

    def save(self, file, tabtype='BinTableHDU', overwrite=False, verbose=True):
        """
        ----------------------------------------------------------------------------
        Saves the interferometer array information to disk. 

        file         [string] Filename with full path to be saved to. Will be
                     appended with '.fits' extension

        Keyword Input(s):

        tabtype      [string] indicates table type for one of the extensions in 
                     the FITS file. Allowed values are 'BinTableHDU' and 
                     'TableHDU' for binary ascii tables respectively. Default is
                     'BinTableHDU'.
                     
        overwrite    [boolean] True indicates overwrite even if a file already 
                     exists. Default = False (does not overwrite)
                     
        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        try:
            file
        except NameError:
            raise NameError('No filename provided. Aborting InterferometerArray.save()...')

        filename = file + '.fits' 

        if verbose:
            print '\nSaving information about interferometer...'

        hdulist = []

        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['latitude'] = (self.latitude, 'Latitude of interferometer')
        hdulist[0].header['baseline_coords'] = (self.baseline_coords, 'Baseline coordinate system')
        hdulist[0].header['freq_resolution'] = (self.freq_resolution, 'Frequency Resolution (Hz)')
        hdulist[0].header['pointing_coords'] = (self.pointing_coords, 'Pointing coordinate system')
        hdulist[0].header['phase_center_coords'] = (self.phase_center_coords, 'Phase center coordinate system')
        hdulist[0].header['skycoords'] = (self.skycoords, 'Sky coordinate system')
        hdulist[0].header['telescope'] = (self.telescope, 'Telescope')
        hdulist[0].header['t_obs'] = (self.t_obs, 'Observing duration (s)')
        hdulist[0].header['n_acc'] = (self.n_acc, 'Number of accumulations')        
        hdulist[0].header['flux_unit'] = (self.flux_unit, 'Unit of flux density')

        if verbose:
            print '\tCreated a primary HDU.'

        cols = []
        if self.lst: 
            cols += [fits.Column(name='LST', format='D', array=NP.asarray(self.lst).ravel())]
            cols += [fits.Column(name='pointing_longitude', format='D', array=self.pointing_center[:,0])]
            cols += [fits.Column(name='pointing_latitude', format='D', array=self.pointing_center[:,1])]
            cols += [fits.Column(name='phase_center_longitude', format='D', array=self.phase_center[:,0])]
            cols += [fits.Column(name='phase_center_latitude', format='D', array=self.phase_center[:,1])]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'POINTING AND PHASE CENTER INFO')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated pointing and phase center information table.'

        cols = []
        cols += [fits.Column(name='labels', format='5A', array=NP.asarray(self.labels))]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'LABELS')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated extension table containing baseline labels.'

        hdulist += [fits.ImageHDU(self.baselines, name='baselines')]
        if verbose:
            print '\tCreated an extension for baseline vectors.'

        hdulist += [fits.ImageHDU(self.A_eff, name='Effective area')]
        if verbose:
            print '\tCreated an extension for effective area.'

        hdulist += [fits.ImageHDU(self.eff_Q, name='Interferometer efficiency')]
        if verbose:
            print '\tCreated an extension for interferometer efficiency.'

        cols = []
        cols += [fits.Column(name='frequency', format='D', array=self.channels)]
        if self.lags is not None:
            cols += [fits.Column(name='lag', format='D', array=self.lags)]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'SPECTRAL INFO')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated spectral information table.'

        if self.t_acc:
            hdulist += [fits.ImageHDU(self.t_acc, name='t_acc')]
            if verbose:
                print '\tCreated an extension for accumulation times.'

        cols = []
        cols += [fits.Column(name='timestamps', format='12A', array=NP.asarray(self.timestamp))]
        columns = fits.ColDefs(cols, tbtype=tabtype)
        tbhdu = fits.new_table(columns)
        tbhdu.header.set('EXTNAME', 'TIMESTAMPS')
        hdulist += [tbhdu]
        if verbose:
            print '\tCreated extension table containing timestamps.'

        if (self.Tsys is not None) and (self.Tsys != []):
            hdulist += [fits.ImageHDU(self.Tsys, name='Tsys')]
            if verbose:
                print '\tCreated an extension for Tsys.'

        if self.vis_rms_freq is not None:
            hdulist += [fits.ImageHDU(self.vis_rms_freq, name='freq_channel_noise_rms_visibility')]
            if verbose:
                print '\tCreated an extension for simulated visibility noise rms per channel.'
        
        if self.vis_freq is not None:
            hdulist += [fits.ImageHDU(self.vis_freq.real, name='real_freq_obs_visibility')]
            hdulist += [fits.ImageHDU(self.vis_freq.imag, name='imag_freq_obs_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of observed visibility frequency spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_freq.shape)

        if self.skyvis_freq is not None:
            hdulist += [fits.ImageHDU(self.skyvis_freq.real, name='real_freq_sky_visibility')]
            hdulist += [fits.ImageHDU(self.skyvis_freq.imag, name='imag_freq_sky_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of noiseless sky visibility frequency spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.skyvis_freq.shape)

        if self.vis_noise_freq is not None:
            hdulist += [fits.ImageHDU(self.vis_noise_freq.real, name='real_freq_noise_visibility')]
            hdulist += [fits.ImageHDU(self.vis_noise_freq.imag, name='imag_freq_noise_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of visibility noise frequency spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_noise_freq.shape)

        hdulist += [fits.ImageHDU(self.bp, name='bandpass')]
        if verbose:
            print '\tCreated an extension for bandpass functions of size {0[0]} x {0[1]} x {0[2]} as a function of baseline,  frequency, and snapshot instance'.format(self.bp.shape)

        hdulist += [fits.ImageHDU(self.bp_wts, name='bandpass_weights')]
        if verbose:
            print '\tCreated an extension for bandpass weights of size {0[0]} x {0[1]} x {0[2]} as a function of baseline,  frequency, and snapshot instance'.format(self.bp_wts.shape)

        if self.vis_lag is not None:
            hdulist += [fits.ImageHDU(self.vis_lag.real, name='real_lag_visibility')]
            hdulist += [fits.ImageHDU(self.vis_lag.imag, name='imag_lag_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of observed visibility delay spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_lag.shape)

        if self.skyvis_lag is not None:
            hdulist += [fits.ImageHDU(self.skyvis_lag.real, name='real_lag_sky_visibility')]
            hdulist += [fits.ImageHDU(self.skyvis_lag.imag, name='imag_lag_sky_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of noiseless sky visibility delay spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.skyvis_lag.shape)

        if self.vis_noise_lag is not None:
            hdulist += [fits.ImageHDU(self.vis_noise_lag.real, name='real_lag_noise_visibility')]
            hdulist += [fits.ImageHDU(self.vis_noise_lag.imag, name='imag_lag_noise_visibility')]
            if verbose:
                print '\tCreated extensions for real and imaginary parts of visibility noise delay spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_noise_lag.shape)

        if verbose:
            print '\tNow writing FITS file to disk...'

        hdu = fits.HDUList(hdulist)
        hdu.writeto(filename, clobber=overwrite)

        if verbose:
            print '\tInterferometer array information written successfully to FITS file on disk:\n\t\t{0}\n'.format(filename)

#################################################################################
