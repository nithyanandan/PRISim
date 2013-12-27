from __future__ import division
import numpy as NP
import scipy.constants as FCNST
import geometry as GEOM
import primary_beams as PB
import baseline_delay_horizon as DLY
import constants as CNST
import my_DSP_modules as DSP

class Interferometer:

    def __init__(self, label, baseline, channels, telescope='vla',
                 latitude=34.0790, skycoords='radec', eff_Q=0.89,
                 A_eff=NP.pi*(25.0/2)**2, pointing_coords='hadec',
                 baseline_coords='localenu', freq_scale=None):

        self.label = label
        self.baseline = NP.asarray(baseline).reshape(-1,3)
        self.telescope = telescope
        self.latitude = latitude
        self.vis_freq = None
        self.skyvis_freq = None
        self.pb = None
        self.vis_noise_freq = None

        if (freq_scale is None) or (freq_scale == 'Hz') and (freq_scale == 'hz'):
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
        self.tobs = []
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
        self.obs_catalog = []
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


    def observe(self, timestamp, Tsys, bandpass, pointing_center, skymodel,
                tobs, pb_min=0.1, fov_radius=None, lst=None):

        if bandpass.size != self.bp.shape[1]:
            raise ValueError('bandpass length does not match.')

        self.Tsys = self.Tsys + [Tsys]
        self.vis_rms_freq = self.vis_rms_freq + [2.0*FCNST.k*Tsys/self.A_eff/self.eff_Q/NP.sqrt(2)/tobs/self.freq_resolution/CNST.Jy]
        self.tobs = self.tobs + [tobs]
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
                pointing_lonlat = lst - GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = pointing_lonlat[0]
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
            
            coords_str = self.skycoords
            if self.skycoords == 'radec':
                coords_str = 'altaz'
                source_positions = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.catalog.location[m2,0]).reshape(-1,1),skymodel.catalog.location[m2,1].reshape(-1,1))), self.latitude, 'degrees')

            for i in xrange(len(self.channels)):
                # pb[:,i] = PB.primary_beam_generator(d12, self.channels[i]/1.0e9, 'degrees', self.telescope)
                pb[:,i] = PB.primary_beam_generator(source_positions, self.channels[i]/1.0e9, 'altaz', self.telescope)
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
            self.vis_noise_freq = self.vis_rms_freq[-1] * (NP.random.randn(len(self.channels)).reshape(1,-1) + 1j * NP.random.randn(len(self.channels)).reshape(1,-1))
            self.vis_freq = self.skyvis_freq + self.vis_noise_freq
        else:
            self.skyvis_freq = NP.vstack((self.skyvis_freq, skyvis.reshape(1,-1)))
            self.vis_noise_freq = NP.vstack((self.vis_noise_freq, self.vis_rms_freq[-1] * (NP.random.randn(len(self.channels)).reshape(1,-1) + 1j * NP.random.randn(len(self.channels)).reshape(1,-1))))
            self.vis_freq = NP.vstack((self.vis_freq, (self.skyvis_freq[-1,:] + self.vis_noise_freq[-1,:]).reshape(1,-1)))

        self.timestamp = self.timestamp + [timestamp]


    def delay_transform(self):
        self.vis_lag = DSP.FT1D(self.vis_freq, ax=1, use_real=False, shift=False) * self.freq_resolution
        self.skyvis_lag = DSP.FT1D(self.skyvis_freq, ax=1, use_real=False, shift=False) * self.freq_resolution
        self.vis_noise_lag = DSP.FT1D(self.vis_noise_freq, ax=1, use_real=False, shift=False) * self.freq_resolution
        self.lags = DSP.spectral_axis(len(self.channels), delx=self.freq_resolution, use_real=False, shift=False)


