from __future__ import division
import numpy as NP
import numpy.linalg as LA
import scipy.constants as FCNST
from scipy.linalg import toeplitz
import scipy.optimize as OPT
import datetime as DT
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

    bp          [numpy vector] Bandpass weights (same size as channels)

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

    lst         [list] List of LST for each timestamp

    n_acc       [scalar] Number of accumulations

    obs_catalog_indices
                [list of lists] Each element in the top list corresponds to a
                timestamp. Inside each top list is a list of indices of sources
                from the catalog which are observed inside the field of view.
                This is computed inside member function observe(). 

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
    
    skyvis_freq [numpy array] Complex visibility due to sky emission (in Jy) 
                along frequency axis estimated from the specified external
                catalog. Same size as vis_freq. Used in the member function
                observe(). Read its docstring for more details. 

    skyvis_lag  [numpy array] Complex visibility due to sky emission (in Jy Hz)
                along the delay axis obtained by FFT of skyvis_freq along 
                frequency axis. Same size as vis_freq. Created in the member
                function delay_transform(). Read its docstring for more details. 

    telescope   [string] The name of the telescope facility. Accepted values
                are 'vla', 'gmrt' and 'mwa'. Default = 'vla'

    timestamp   [list] List of timestamps during the observation

    t_acc       [list] Accumulation time (sec) corresponding to each timestamp

    t_obs       [scalar] Observing duration (sec)

    Tsys        [scalar] System temperature in Kelvin

    vis_freq    [numpy array] The simulated complex visibility (in Jy) observed 
                by the interferometer along frequency axis for each timestamp of 
                observation per frequency channel. It is the sum of skyvis_freq 
                and vis_noise_freq. It can be either directly initialized or
                simulated in observe(). 

    vis_lag     [numpy array] The simulated complex visibility (in Jy Hz) along
                delay axis obtained by FFT of vis_freq along frequency axis.
                Same size as vis_noise_lag and skyis_lag. It is evaluated in
                member function delay_transform(). 

    vis_noise_freq
                [numpy array] Complex visibility noise (in Jy) generated using 
                an rms of vis_rms_freq along frequency axis which is then added 
                to the generated sky visibility. Same size as vis_freq. Used in
                the member function observe(). Read its docstring for more
                details. 

    vis_noise_lag
                [numpy array] Complex visibility noise (in Jy Hz) along delay
                axisgenerated using an FFT of vis_noise_freq along frequency 
                axis. Same size as vis_noise_freq. Created in the member function
                delay_transform(). Read its docstring for more details. 

    vis_rms_freq
                [list of float] Theoretically estimated thermal noise rms (in Jy) 
                in visibility measurements. Same size as vis_freq. This will be 
                estimated and used to inject simulated noise when a call to 
                member function observe() is made. Read the  docstring of 
                observe() for more details. The noise rms is estimated from the 
                instrument parameters as:
                (2 k T_sys / (A_eff x sqrt(channel_width x t_obs))) / Jy

    Member functions:

    __init__():        Initializes an instance of class Interferometer

    observe():         Simulates an observing run with the interferometer
                       specifications and an external sky catalog thus producing
                       visibilities. The simulation generates visibilities
                       observed by the interferometer for the specified
                       parameters.

    delay_transform(): Transforms the visibilities from frequency axis onto 
                       delay (time) axis using an FFT.

    noise_estimate():  Given the attribute vis_freq, compute the thermal noise 
                       estimate (in Jy) in the data in each frequency channel

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
        pointing_coords, baseline_coords, freq_scale, baseline_length, channels,
        bp, freq_resolution, lags, lst, obs_catalog_indices, pointing_center,
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

        self.bp = NP.asarray(NP.ones(self.channels.size)).reshape(1,-1)
        self.Tsys = []
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

    ############################################################################

    def observe(self, timestamp, Tsys, bandpass, pointing_center, skymodel,
                t_acc, fov_radius=None, lst=None):

        """
        ------------------------------------------------------------------------
        Simulate an observation of the sky in the form of an external catalog
        by an instance of the Interferometer class. The simulation generates
        visibilities observed by the interferometer for the specified parameters.
        

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

        t_acc         [scalar] Accumulation time corresponding to timestamp

        Keyword Inputs:

        fov_radius   [scalar] Radius of the field of view (degrees) inside which 
                     sources are to be observed. Default = 90 degrees, which is
                     the entire horizon.

        lst          [scalar] LST associated with the timestamp
        ------------------------------------------------------------------------
        """

        if bandpass.size != self.bp.shape[1]:
            raise ValueError('bandpass length does not match.')

        self.Tsys = self.Tsys + [Tsys]
        self.vis_rms_freq = self.vis_rms_freq + [2.0*FCNST.k*Tsys/self.A_eff/self.eff_Q/NP.sqrt(2.0 * t_acc * self.freq_resolution)/CNST.Jy]
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

        ptmp = self.pointing_center[-1,:] # Convert pointing center to Alt-Az coordinates
        if self.pointing_coords == 'hadec':
            ptmp = GEOM.hadec2altaz(self.pointing_center[-1,:], self.latitude,
                                    units='degrees')
        elif self.pointing_coords == 'radec':
            if lst is not None:
                ptmp = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
            else:
                raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')

        ptmp = GEOM.altaz2dircos(ptmp, 'degrees') # Convert pointing center to direction cosine coordinates
        pointing_phase = 2.0 * NP.pi * NP.dot(baseline_in_local_frame.reshape(1,-1), ptmp.reshape(-1,1))*self.channels.reshape(1,-1)/FCNST.c

        if not isinstance(skymodel, CTLG.SkyModel):
            raise TypeError('skymodel should be an instance of class SkyModel.')

        if fov_radius is None:
            fov_radius = 90.0

        m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.catalog.location[:,0], skymodel.catalog.location[:,1], fov_radius, maxmatches=0)

        # if fov_radius is not None:
        #     m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.catalog.location[:,0], skymodel.catalog.location[:,1], fov_radius, maxmatches=0)
        # else:
        #     m1 = [0] * skymodel.catalog.location.shape[0]
        #     m2 = xrange(skymodel.catalog.location.shape[0])
        #     d12 = GEOM.sphdist(NP.empty(skymodel.catalog.shape[0]).fill(pointing_lon), NP.empty(skymodel.catalog.shape[0]).fill(pointing_lat), skymodel.catalog.location[:,0], skymodel.catalog.location[:,1])

        if len(d12) != 0:
            pb = NP.empty((len(d12), len(self.channels)))
            fluxes = NP.empty((len(d12), len(self.channels)))
            
            if self.skycoords == 'altaz':
                source_positions = skymodel.catalog.location[m2,:]
            elif self.skycoords == 'radec':
                source_positions = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.catalog.location[m2,0]).reshape(-1,1), skymodel.catalog.location[m2,1].reshape(-1,1))), self.latitude, 'degrees')
            else:
                source_positions = GEOM.hadec2altaz(skymodel.catalog.location[m2,:], self.latitude, 'degrees')
                
            coords_str = 'altaz'

            if self.pointing_coords == 'altaz':
                phase_center = pointing_center
            elif self.pointing_coords == 'radec':
                phase_center = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-self.pointing_center[-1,0]).reshape(-1,1), self.pointing_center[-1,1].reshape(-1,1))), self.latitude, 'degrees')
            else:
                phase_center = GEOM.hadec2altaz(self.pointing_center[-1,:], self.latitude, 'degrees')

            for i in xrange(len(self.channels)):
                # pb[:,i] = PB.primary_beam_generator(d12, self.channels[i]/1.0e9, 'degrees', self.telescope)
                pb[:,i] = PB.primary_beam_generator(source_positions, self.channels[i]/1.0e9, skyunits='altaz', telescope=self.telescope, phase_center=phase_center)
                fluxes[:,i] = skymodel.catalog.flux_density[m2] * (self.channels[i]/skymodel.catalog.frequency)**skymodel.catalog.spectral_index[m2]

            geometric_delays = DLY.geometric_delay(baseline_in_local_frame, source_positions, altaz=(coords_str=='altaz'), hadec=(coords_str=='hadec'), latitude=self.latitude)
            self.geometric_delays = self.geometric_delays + [geometric_delays.reshape(len(source_positions))]

            phase_matrix = 2.0 * NP.pi * NP.repeat(geometric_delays.reshape(-1,1),len(self.channels),axis=1) * NP.repeat(self.channels.reshape(1,-1),len(d12),axis=0) - NP.repeat(pointing_phase, len(d12), axis=0)

            skyvis = NP.sum(pb * fluxes * NP.repeat(NP.asarray(bandpass).reshape(1,-1),len(d12),axis=0) * NP.exp(-1j*phase_matrix), axis=0)
            if fov_radius is not None:
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
                      bpass, Tsys, lst_init, fov_radius=None, mode='track', 
                      pointing_coords=None, freq_scale=None, verbose=True):

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
            channels *= 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            channels *= 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            channels *= 1.0e3
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

        for i in xrange(n_acc):
            if (verbose) and (i in milestones):
                print '\t\tObserving run {0:.1f} % complete...'.format(100.0*i/n_acc)
            timestamp = str(DT.datetime.now())
            self.observe(timestamp, Tsys[i], bpass[i,:], pointing, skymodel, t_acc, fov_radius, lst[i])

        if verbose:
            print '\t\tObserving run 100 % complete.'

        self.t_obs = duration
        self.n_acc = n_acc
        if verbose:
            print 'Observing run completed successfully.'

    ############################################################################

    def delay_transform(self):

        """
        ------------------------------------------------------------------------
        Transforms the visibilities from frequency axis onto delay (time) axis
        using an FFT. This is performed for noiseless sky visibilities, thermal
        noise in visibilities, and observed visibilities. 
        ------------------------------------------------------------------------
        """

        self.vis_lag = DSP.FT1D(self.vis_freq, ax=1, use_real=False, shift=True) * self.freq_resolution
        self.skyvis_lag = DSP.FT1D(self.skyvis_freq, ax=1, use_real=False, shift=True) * self.freq_resolution
        self.vis_noise_lag = DSP.FT1D(self.vis_noise_freq, ax=1, use_real=False, shift=True) * self.freq_resolution
        self.lags = DSP.spectral_axis(len(self.channels), delx=self.freq_resolution,
                                      use_real=False, shift=True)

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

    ############################################################################    
    
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
        hdulist[0].header.set('EXTNAME', 'Interferometer ({0})'.format(self.label))

        if verbose:
            print '\tCreated a primary HDU.'

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

        if not self.t_acc:
            hdulist += [fits.ImageHDU(self.t_acc, name='t_acc')]
            if verbose:
                print '\tCreated an extension for accumulation times.'

        if not self.vis_rms_freq:
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

    ############################################################################
