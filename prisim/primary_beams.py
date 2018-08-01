import numpy as NP 
import scipy.constants as FCNST
import scipy.special as SPS
import h5py
from astroutils import geometry as GEOM

#################################################################################

def primary_beam_generator(skypos, frequency, telescope, freq_scale='GHz',
                           skyunits='degrees', east2ax1=0.0, pointing_info=None,
                           pointing_center=None, short_dipole_approx=False,
                           half_wave_dipole_approx=False):

    """
    -----------------------------------------------------------------------------
    A wrapper for estimating the power patterns of different telescopes such as
    the VLA, GMRT, MWA, HERA, PAPER, HIRAX, CHIME, etc. For the VLA and GMRT, 
    polynomial power patterns are estimated as specified in AIPS task PBCOR. For 
    MWA, it is based on theoretical expressions for dipole (element) pattern 
    multiplied with the array pattern of isotropic radiators.

    Inputs:

    skypos      [numpy array] Sky positions at which the power pattern is to be
                estimated. Size is M x N where M is the number of locations and 
                N = 1 (if skyunits = degrees, for azimuthally symmetric
                telescopes such as VLA and GMRT which have parabolic dishes), 
                N = 2 (if skyunits = altaz denoting Alt-Az coordinates), or N = 3
                (if skyunits = dircos denoting direction cosine coordinates)

    frequency   [list or numpy vector] frequencies at which the power pattern is 
                to be estimated. Units can be GHz, MHz or kHz (see input
                freq_scale)

    telescope   [dictionary] dictionary that specifies the type of element,
                element size and orientation. It consists of the following keys
                and values:
                'id'          [string] If set, will ignore the other keys and use
                              telescope details for known telescopes. Accepted 
                              values are 'mwa', 'vla', 'gmrt', 'hera', 'paper', 
                              'hirax', and 'chime' 
                'shape'       [string] Shape of antenna element. Accepted values
                              are 'dipole', 'delta', 'dish', 'gaussian', 'rect' 
                              and 'square'. Will be ignored if key 'id' is set. 
                              'delta' denotes a delta function for the antenna 
                              element which has an isotropic radiation pattern. 
                              'delta' is the default when keys 'id' and 'shape' 
                              are not set.
                'size'        [scalar or 2-element list/numpy array] Diameter of 
                              the telescope dish (in meters) if the key 'shape' 
                              is set to 'dish', side of the square aperture (in 
                              meters) if the key 'shape' is set to 'square', 
                              2-element sides if key 'shape' is set to 'rect', 
                              or length of the dipole if key 'shape' is set to 
                              'dipole'. Will be ignored if key 'shape' is set to 
                              'delta'. Will be ignored if key 'id' is set and a 
                              preset value used for the diameter or dipole.
                'orientation' [list or numpy array] If key 'shape' is set to 
                              dipole, it refers to the orientation of the dipole 
                              element unit vector whose magnitude is specified by 
                              length. If key 'shape' is set to 'dish', it refers 
                              to the position on the sky to which the dish is
                              pointed. For a dipole, this unit vector must be
                              provided in the local ENU coordinate system aligned 
                              with the direction cosines coordinate system or in
                              the Alt-Az coordinate system. This will be
                              used only when key 'shape' is set to 'dipole'.
                              This could be a 2-element vector (transverse 
                              direction cosines) where the third (line-of-sight) 
                              component is determined, or a 3-element vector
                              specifying all three direction cosines or a two-
                              element coordinate in Alt-Az system. If not provided 
                              it defaults to an eastward pointing dipole. If key
                              'shape' is set to 'dish' or 'gaussian', the 
                              orientation refers to the pointing center of the 
                              dish on the sky. It can be provided in Alt-Az 
                              system as a two-element vector or in the direction 
                              cosine coordinate system as a two- or three-element 
                              vector. If not set in the case of a dish element, 
                              it defaults to zenith. This is not to be confused 
                              with the key 'pointing_center' in dictionary 
                              'pointing_info' which refers to the beamformed 
                              pointing center of the array. The coordinate system 
                              is specified by the key 'ocoords'
                'ocoords'     [scalar string] specifies the coordinate system 
                              for key 'orientation'. Accepted values are 'altaz'
                              and 'dircos'. 
                'element_locs'
                              [2- or 3-column array] Element locations that
                              constitute the tile. Each row specifies
                              location of one element in the tile. The
                              locations must be specified in local ENU
                              coordinate system. First column specifies along
                              local east, second along local north and the
                              third along local up. If only two columns are 
                              specified, the third column is assumed to be 
                              zeros. If 'elements_locs' is not provided, it
                              assumed to be a one-element system and not a
                              phased array as far as determination of primary 
                              beam is concerned.
                'groundplane' [scalar] height of telescope element above the 
                              ground plane (in meteres). Default = None will
                              denote no ground plane effects.
                'ground_modify'
                              [dictionary] contains specifications to modify
                              the analytically computed ground plane pattern. If
                              absent, the ground plane computed will not be
                              modified. If set, it may contain the following 
                              keys:
                              'scale' [scalar] positive value to scale the 
                                      modifying factor with. If not set, the 
                                      scale factor to the modification is unity.
                              'max'   [scalar] positive value to clip the 
                                      modified and scaled values to. If not set, 
                                      there is no upper limit

    freq_scale  [scalar] string specifying the units of frequency. Accepted
                values are 'GHz', 'MHz' and 'Hz'. Default = 'GHz'

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'degrees', 'altaz', and 'dircos'.
                Default = 'degrees'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up

    east2ax1    [scalar] Angle (in degrees) the primary axis of the aperture 
                makes with the local East (positive anti-clockwise). 

    pointing_info 
              [dictionary] A dictionary consisting of information relating to 
              pointing center in case of a phased array. The pointing center 
              can be specified either via element delay compensation or by 
              directly specifying the pointing center in a certain coordinate 
              system. Default = None (pointing centered at zenith). This 
              dictionary consists of the following tags and values:
              'gains'           [numpy array] Complex element gains. Must be of 
                                size equal to the number of elements as 
                                specified by the number of rows in antpos. If 
                                set to None (default), all element gains are 
                                assumed to be unity. Used only in phased array
                                mode.
              'gainerr'         [int, float] RMS error in voltage amplitude in 
                                dB to be used in the beamformer. Random jitters 
                                are drawn from a normal distribution in 
                                logarithm units which are then converted to 
                                linear units. Must be a non-negative scalar. 
                                If not provided, it defaults to 0 (no jitter).
                                Used only in phased array mode.
              'delays'          [numpy array] Delays (in seconds) to be applied 
                                to the tile elements. Size should be equal to 
                                number of tile elements (number of rows in
                                antpos). Default = None will set all element
                                delays to zero phasing them to zenith. Used only
                                in phased array mode. 
              'pointing_center' [numpy array] This will apply in the absence of 
                                key 'delays'. This can be specified as a row 
                                vector. Should have two-columns if using Alt-Az
                                coordinates, or two or three columns if using
                                direction cosines. There is no default. The
                                coordinate system must be specified in
                                'pointing_coords' if 'pointing_center' is to be
                                used.
              'pointing_coords' [string scalar] Coordinate system in which the
                                pointing_center is specified. Accepted values 
                                are 'altaz' or 'dircos'. Must be provided if
                                'pointing_center' is to be used. No default.
              'delayerr'        [int, float] RMS jitter in delays used in the
                                beamformer. Random jitters are drawn from a 
                                normal distribution with this rms. Must be
                                a non-negative scalar. If not provided, it
                                defaults to 0 (no jitter). Used only in phased 
                                array mode.
              'nrand'           [int] number of random realizations of gainerr 
                                and/or delayerr to be averaged. Must be 
                                positive. If none provided, it defaults to 1.
                                Used only in phased array mode.

    pointing_center
                [list or numpy array] coordinates of pointing center (in the same
                coordinate system as that of sky coordinates specified by
                skyunits). 2-element vector if skyunits='altaz'. 2- or 3-element
                vector if skyunits='dircos'. Only used with phased array primary
                beams, dishes excluding VLA and GMRT, or uniform rectangular or 
                square apertures. For all telescopes except MWA, pointing_center 
                is used in place of pointing_info. For MWA, this is used if 
                pointing_info is not provided.

    short_dipole_approx
                [boolean] if True, indicates short dipole approximation
                is to be used. Otherwise, a more accurate expression is used
                for the dipole pattern. Default=False. Both
                short_dipole_approx and half_wave_dipole_approx cannot be set 
                to True at the same time


    half_wave_dipole_approx
                [boolean] if True, indicates half-wave dipole approximation
                is to be used. Otherwise, a more accurate expression is used
                for the dipole pattern. Default=False

    Output:

    [Numpy array] Power pattern at the specified sky positions. 
    -----------------------------------------------------------------------------
    """

    try:
        skypos, frequency, telescope
    except NameError:
        raise NameError('Sky positions, frequency and telescope inputs must be specified.')

    if (freq_scale == 'ghz') or (freq_scale == 'GHz'):
        frequency = frequency * 1.0e9
    elif (freq_scale == 'mhz') or (freq_scale == 'MHz'):
        frequency = frequency * 1.0e6
    elif (freq_scale == 'khz') or (freq_scale == 'kHz'):
        frequency = frequency * 1.0e3

    frequency = NP.asarray(frequency)

    if (telescope is None) or (not isinstance(telescope, dict)):
        raise TypeError('telescope must be specified as a dictionary')

    if 'id' in telescope:
        if (telescope['id'] == 'vla') or (telescope['id'] == 'gmrt'):
            if skyunits == 'altaz':
                angles = 90.0 - skypos[:,0]
            elif skyunits == 'dircos':
                angles = NP.arccos(NP.sqrt(1.0 - NP.sum(skypos[:,2]**2, axis=1)))
            elif skyunits == 'degrees':
                angles = skypos
            else:
                raise ValueError('skyunits must be "altaz", "dircos" or "degrees".')
    
            if telescope['id'] == 'vla':
                pb = VLA_primary_beam_PBCOR(angles, frequency/1e9, 'degrees')
            elif telescope['id'] == 'gmrt':
                pb = GMRT_primary_beam(angles, frequency/1e9, 'degrees')
        elif (telescope['id'] == 'hera') or (telescope['id'] == 'hirax'):
            if telescope['id'] == 'hera':
                dish_dia = 14.0
            else:
                dish_dia = 6.0
            pb = airy_disk_pattern(dish_dia, skypos, frequency, skyunits=skyunits,
                                   peak=1.0, pointing_center=telescope['orientation'], 
                                   pointing_coords=telescope['ocoords'],
                                   gaussian=False, power=True, small_angle_tol=1e-10)
        elif telescope['id'] == 'mwa':
            if (skyunits == 'altaz') or (skyunits == 'dircos'):
                if ('orientation' in telescope) and ('ocoords' in telescope):
                    orientation = NP.asarray(telescope['orientation']).reshape(1,-1)
                    ocoords = telescope['ocoords']
                elif ('orientation' not in telescope) and ('ocoords' in telescope):
                    ocoords = telescope['ocoords']
                    if telescope['ocoords'] == 'altaz':
                        orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
                    elif telescope['ocoords'] == 'dircos':
                        orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1)
                    else:
                        raise ValueError('key "ocoords" in telescope dictionary contains invalid value')
                elif ('orientation' in telescope) and ('ocoords' not in telescope):
                    raise KeyError('key "ocoords" in telescope dictionary not specified.')
                else:
                    ocoords = 'dircos'
                    orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1)

                ep = dipole_field_pattern(0.74, skypos, dipole_coords=ocoords,
                                          dipole_orientation=orientation,
                                          skycoords=skyunits, wavelength=FCNST.c/frequency, 
                                          short_dipole_approx=short_dipole_approx,
                                          half_wave_dipole_approx=half_wave_dipole_approx,
                                          power=False)
                ep = ep[:,:,NP.newaxis]  # add an axis to be compatible with random ralizations
                if pointing_info is None: # Use analytical formula
                    if skyunits == 'altaz':
                        pointing_center = NP.asarray([90.0, 270.0]).reshape(1,-1)
                    elif skyunits == 'dircos':
                        pointing_center = NP.asarray([0.0, 0.0, 1.0]).reshape(1,-1)
                    else:
                        raise ValueError('skyunits for MWA must be "altaz" or "dircos"')
                    
                    irap = isotropic_radiators_array_field_pattern(4, 4, 1.1, 1.1, skypos,
                                                                   FCNST.c/frequency, east2ax1=east2ax1,
                                                                   pointing_center=pointing_center,
                                                                   skycoords=skyunits, power=False)
                    irap = irap[:,:,NP.newaxis]  # add an axis to be compatible with random ralizations

                else: # Call the beamformer
                    if 'element_locs' not in telescope:
                        nrand = 1
                        xlocs, ylocs = NP.meshgrid(1.1*NP.linspace(-1.5,1.5,4), 1.1*NP.linspace(1.5,-1.5,4))
                        element_locs = NP.hstack((xlocs.reshape(-1,1), ylocs.reshape(-1,1), NP.zeros(xlocs.size).reshape(-1,1)))
                    else:
                        element_locs = telescope['element_locs']
                    pinfo = {}
                    gains = None
                    if 'delays' in pointing_info:
                        pinfo['delays'] = pointing_info['delays']
                    if 'delayerr' in pointing_info:
                        pinfo['delayerr'] = pointing_info['delayerr']
                    if 'pointing_center' in pointing_info:
                        pinfo['pointing_center'] = pointing_info['pointing_center']
                    if 'pointing_coords' in pointing_info:
                        pinfo['pointing_coords'] = pointing_info['pointing_coords']
                    if 'gains' in pointing_info:
                        pinfo['gains'] = pointing_info['gains']
                    if 'gainerr' in pointing_info:
                        pinfo['gainerr'] = pointing_info['gainerr']
                    if 'nrand' in pointing_info:
                        pinfo['nrand'] = pointing_info['nrand']
                    irap = array_field_pattern(element_locs, skypos, 
                                               skycoords=skyunits,
                                               pointing_info=pinfo,
                                               wavelength=FCNST.c/frequency,
                                               power=False)
                    nrand = irap.shape[-1]
                pb = NP.mean(NP.abs(ep * irap)**2, axis=2) # Power pattern is square of the field pattern
            else:
                raise ValueError('skyunits must be in Alt-Az or direction cosine coordinates for MWA.')
        elif (telescope['id'] == 'mwa_dipole') or (telescope['id'] == 'paper'):
            if telescope['id'] == 'mwa_dipole':
                dipole_size = 0.74
            else:
                dipole_size = 2.0
            if (skyunits == 'altaz') or (skyunits == 'dircos'):
                if ('orientation' in telescope) and ('ocoords' in telescope):
                    orientation = NP.asarray(telescope['orientation']).reshape(1,-1)
                    ocoords = telescope['ocoords']
                elif ('orientation' not in telescope) and ('ocoords' in telescope):
                    ocoords = telescope['ocoords']
                    if telescope['ocoords'] == 'altaz':
                        orientation = NP.asarray([0.0, 90.0]).reshape(1,-1)
                    elif telescope['ocoords'] == 'dircos':
                        orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1)
                    else:
                        raise ValueError('key "ocoords" in telescope dictionary contains invalid value')
                elif ('orientation' in telescope) and ('ocoords' not in telescope):
                    raise KeyError('key "ocoords" in telescope dictionary not specified.')
                else:
                    ocoords = 'dircos'
                    orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1)

                ep = dipole_field_pattern(dipole_size, skypos, dipole_coords=ocoords,
                                          dipole_orientation=orientation,
                                          skycoords=skyunits, wavelength=FCNST.c/frequency, 
                                          short_dipole_approx=short_dipole_approx,
                                          half_wave_dipole_approx=half_wave_dipole_approx,
                                          power=False)
                pb = NP.abs(ep)**2 # Power pattern is square of the field pattern
            else:
                raise ValueError('skyunits must be in Alt-Az or direction cosine coordinates for MWA dipole.')
        else:
            raise ValueError('No presets available for the specified telescope ID. Set custom parameters instead in input parameter telescope.')
    else:
        if 'shape' not in telescope:
            telescope['shape'] = 'delta'
            ep = 1.0
        elif telescope['shape'] == 'delta':
            ep = 1.0
        elif telescope['shape'] == 'dipole':
            ep = dipole_field_pattern(telescope['size'], skypos,
                                      dipole_coords=telescope['ocoords'],
                                      dipole_orientation=telescope['orientation'],
                                      skycoords=skyunits, wavelength=FCNST.c/frequency, 
                                      short_dipole_approx=short_dipole_approx,
                                      half_wave_dipole_approx=half_wave_dipole_approx,
                                      power=False)
            ep = ep[:,:,NP.newaxis]   # add an axis to be compatible with random ralizations
        elif telescope['shape'] == 'dish':
            ep = airy_disk_pattern(telescope['size'], skypos, frequency, skyunits=skyunits,
                                   peak=1.0, pointing_center=pointing_center, 
                                   gaussian=False, power=False, small_angle_tol=1e-10)
            ep = ep[:,:,NP.newaxis]   # add an axis to be compatible with random ralizations
        elif telescope['shape'] == 'gaussian':
            ep = gaussian_beam(telescope['size'], skypos, frequency, skyunits=skyunits,
                                   pointing_center=pointing_center, power=False)
            ep = ep[:,:,NP.newaxis]   # add an axis to be compatible with random ralizations
        elif telescope['shape'] == 'rect':
            ep = uniform_rectangular_aperture(telescope['size'], skypos, frequency, skyunits=skyunits, east2ax1=east2ax1, pointing_center=pointing_center, power=False)
        elif telescope['shape'] == 'square':
            ep = uniform_square_aperture(telescope['size'], skypos, frequency, skyunits=skyunits, east2ax1=east2ax1, pointing_center=pointing_center, power=False)
        else:
            raise ValueError('Value in key "shape" of telescope dictionary invalid.')

        if pointing_info is not None: # Call the beamformer

            if 'element_locs' not in telescope:
                nrand = 1
                irap = NP.ones(skypos.shape[0]*frequency.size).reshape(skypos.shape[0],frequency.size,nrand)
            else:
                element_locs = telescope['element_locs']
                pinfo = {}
                gains = None
                gainerr = None
                if 'delays' in pointing_info:
                    pinfo['delays'] = pointing_info['delays']
                if 'delayerr' in pointing_info:
                    pinfo['delayerr'] = pointing_info['delayerr']
                if 'pointing_center' in pointing_info:
                    pinfo['pointing_center'] = pointing_info['pointing_center']
                if 'pointing_coords' in pointing_info:
                    pinfo['pointing_coords'] = pointing_info['pointing_coords']
                if 'gains' in pointing_info:
                    pinfo['gains'] = pointing_info['gains']
                if 'gainerr' in pointing_info:
                    pinfo['gainerr'] = pointing_info['gainerr']
                if 'nrand' in pointing_info:
                    pinfo['nrand'] = pointing_info['nrand']
                irap = array_field_pattern(element_locs, skypos, skycoords=skyunits,
                                           pointing_info=pinfo,
                                           wavelength=FCNST.c/frequency, power=False)
                nrand = irap.shape[-1]
        else:
            nrand = 1
            irap = NP.ones(skypos.shape[0]*frequency.size).reshape(skypos.shape[0],frequency.size,nrand)  # Last axis indicates number of random realizations
        pb = NP.mean(NP.abs(ep * irap)**2, axis=2) # Power pattern is square of the field pattern averaged over all random realizations of delays and gains if specified
       
    if 'groundplane' in telescope:
        gp = 1.0
        if telescope['groundplane'] is not None:
            if 'shape' in telescope:
                if telescope['shape'] != 'dish':  # If shape is not dish, compute ground plane pattern
                    modifier = None
                    if 'ground_modify' in telescope:
                        modifier = telescope['ground_modify']
        
                    gp = ground_plane_field_pattern(telescope['groundplane'], skypos, skycoords=skyunits,
                                                    wavelength=FCNST.c/frequency, angle_units='degrees', 
                                                    modifier=modifier, power=False)
            else:
                modifier = None
                if 'ground_modify' in telescope:
                    modifier = telescope['ground_modify']
        
                gp = ground_plane_field_pattern(telescope['groundplane'], skypos, skycoords=skyunits,
                                                wavelength=FCNST.c/frequency, angle_units='degrees', 
                                                modifier=modifier, power=False)
                
        pb *= gp**2

    return pb
    
#################################################################################

def VLA_primary_beam_PBCOR(skypos, frequency, skyunits='degrees'):

    """
    -----------------------------------------------------------------------------
    Primary beam power pattern for the VLA dishes based on the polynomial formula
    in AIPS task PBCOR

    Inputs:

    skypos      [list or numpy vector] Sky positions at which the power pattern 
                is to be estimated. Size is M x N where M is the number of 
                locations and N = 1 (if skyunits = degrees), N = 2 (if
                skyunits = altaz denoting Alt-Az coordinates), or N = 3 (if
                skyunits = dircos denoting direction cosine coordinates)

    frequency   [list or numpy vector] frequencies (in GHz) at which the power 
                pattern is to be estimated. Frequencies differing by too much
                and extending over the usual bands cannot be given. 

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'degrees', 'altaz', and 'dircos'.
                Default = 'degrees'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up

    Output:

    [Numpy array] Power pattern at the specified sky positions. 
    -----------------------------------------------------------------------------
    """

    try:
        skypos, frequency
    except NameError:
        raise NameError('skypos and frequency are required in VLA_primary_beam_PBCOR().')

    frequency = NP.asarray(frequency).ravel()

    freq_ref = NP.asarray([0.0738, 0.3275, 1.465, 4.885, 8.435, 14.965, 22.485, 43.315]).reshape(-1,1)
    parms_ref = NP.asarray([[-0.897,   2.71,  -0.242], 
                            [-0.935,   3.23,  -0.378], 
                            [-1.343,  6.579,  -1.186], 
                            [-1.372,  6.940,  -1.309], 
                            [-1.306,  6.253,  -1.100], 
                            [-1.305,  6.155,  -1.030], 
                            [-1.417,  7.332,  -1.352], 
                            [-1.321,  6.185,  -0.983]])
    
    idx = NP.argmin(NP.abs(freq_ref - frequency[0])) # Index of closest value

    skypos = NP.asarray(skypos)

    if skyunits == 'degrees':
        x = (NP.repeat(skypos.reshape(-1,1), frequency.size, axis=1) * 60.0 * NP.repeat(frequency.reshape(1,-1), skypos.size, axis=0))**2
    elif skyunits == 'altaz':
        x = ((90.0-NP.repeat(skypos[:,0].reshape(-1,1), frequency.size, axis=1)) * 60.0 * NP.repeat(frequency.reshape(1,-1), skypos.size, axis=0))**2
    elif skyunits == 'dircos':
        x = (NP.degrees(NP.arccos(NP.repeat(skypos[:,-1].reshape(-1,1), frequency.size, axis=1))) * 60.0 * NP.repeat(frequency.reshape(1,-1), skypos.size, axis=0))**2
    else:
        raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in GMRT_primary_beam().')

    pb = 1.0 + parms_ref[idx,0]*x/1e3 + parms_ref[idx,1]*(x**2)/1e7 + \
         parms_ref[idx,2]*(x**3)/1e10

    return pb

##########################################################################

def airy_disk_pattern(diameter, skypos, frequency, skyunits='altaz', peak=1.0, 
                      pointing_center=None, pointing_coords=None,
                      small_angle_tol=1e-10, power=True, gaussian=False):

    """
    -----------------------------------------------------------------------------
    Field pattern of a uniformly illuminated dish

    Inputs:

    diameter    [scalar] Diameter of the dish (in m)

    skypos      [list or numpy vector] Sky positions at which the power pattern 
                is to be estimated. Size is M x N where M is the number of 
                locations and N = 1 (if skyunits = degrees), N = 2 (if
                skyunits = altaz denoting Alt-Az coordinates), or N = 3 (if
                skyunits = dircos denoting direction cosine coordinates). If
                skyunits = altaz, then altitude and azimuth must be in degrees

    frequency   [list or numpy vector] frequencies (in GHz) at which the power 
                pattern is to be estimated. Frequencies differing by too much
                and extending over the usual bands cannot be given. 

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'degrees', 'altaz', and 'dircos'.
                Default = 'degrees'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up. If 'altaz', then 
                altitude and azimuth must be in degrees.

    pointing_center
                [numpy array] 1xN numpy array, where N is the same as in skypos.
                If None specified, pointing_center is assumed to be at zenith.
                
    pointing_coords
                [string] Coordiantes of the pointing center. If None specified, 
                it is assumed to be same as skyunits. Same allowed values as 
                skyunits. Default = None.

    gaussian    [boolean] If set to True, use a gaussian shape to approximate
                the power pattern. If False, use the standard airy pattern.
                Default = False

    power       [boolean] If set to True (default), compute power pattern,
                otherwise compute field pattern.

    small_angle_tol
                [scalar] Small angle limit (in radians) below which division by 
                zero is to be avoided. Default = 1e-10

    Output:

    [Numpy array] Field or Power pattern at the specified sky positions. 
    -----------------------------------------------------------------------------
    """
    try:
        diameter, skypos, frequency
    except NameError:
        raise NameError('diameter, skypos and frequency are required in airy_disk_pattern().')

    skypos = NP.asarray(skypos)
    frequency = NP.asarray(frequency).ravel()

    if pointing_center is None:
        if skyunits == 'degrees':
            x = NP.radians(skypos)
        elif skyunits == 'altaz':
            x = NP.radians(90.0 - skypos[:,0])
        elif skyunits == 'dircos':
            x = NP.arcsin(NP.sqrt(skypos[:,0]**2 + skypos[:,1]**2))
        else:
            raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in GMRT_primary_beam().')
        zero_ind = x >= NP.pi/2   # Determine positions beyond the horizon
    else:
        if pointing_coords is None:
            pointing_coords = skyunits
        if skyunits == 'degrees':
            x = NP.radians(skypos)
        else:
            pc_altaz = pointing_center.reshape(1,-1)
            if pointing_coords == 'altaz':
                if pc_altaz.size != 2:
                    raise IndexError('Pointing center in Alt-Az coordinates must contain exactly two elements.')
            elif pointing_coords == 'dircos':
                if pc_altaz.size != 3:
                    raise IndexError('Pointing center in direction cosine coordinates must contain exactly three elements.')
                pc_altaz = GEOM.dircos2altaz(pc_altaz, units='degrees')

            skypos_altaz = NP.copy(skypos)
            if skyunits == 'dircos':
                skypos_altaz = GEOM.dircos2altaz(skypos, units='degrees')
            elif skyunits != 'altaz':
                raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in GMRT_primary_beam().')
            x = GEOM.sphdist(skypos_altaz[:,1], skypos_altaz[:,0], pc_altaz[0,1], pc_altaz[0,0])
            x = NP.radians(x)
            zero_ind = NP.logical_or(x >= NP.pi/2, skypos_altaz[:,0] <= 0.0)   # Determine positions beyond the horizon of the sky as well as those beyond the horizon of the dish, if it is pointed away from the horizon

    k = 2*NP.pi*frequency/FCNST.c
    k = k.reshape(1,-1)
    small_angles_ind = x < small_angle_tol
    x = NP.where(small_angles_ind, small_angle_tol, x)
    x = x.reshape(-1,1)
    pattern = 2 * SPS.j1(k*0.5*diameter*NP.sin(x)) / (k*0.5*diameter*NP.sin(x))

    pattern[zero_ind,:] = 0.0   # Blank all values beyond the horizon

    maxval = 2 * SPS.j1(k*0.5*diameter*NP.sin(small_angle_tol)) / (k*0.5*diameter*NP.sin(small_angle_tol))
    if power:
        pattern = NP.abs(pattern)**2
        maxval = maxval**2
        
    pattern *= peak / maxval 
    
    return pattern

##########################################################################

def gaussian_beam(diameter, skypos, frequency, skyunits='altaz', 
                  pointing_center=None, pointing_coords=None, power=True):

    """
    -----------------------------------------------------------------------------
    Field/power pattern of a Gaussian illumination

    Inputs:

    diameter    [scalar] FWHM diameter of the dish (in m)

    skypos      [list or numpy vector] Sky positions at which the power pattern 
                is to be estimated. Size is M x N where M is the number of 
                locations and N = 1 (if skyunits = degrees), N = 2 (if
                skyunits = altaz denoting Alt-Az coordinates), or N = 3 (if
                skyunits = dircos denoting direction cosine coordinates). If
                skyunits = altaz, then altitude and azimuth must be in degrees

    frequency   [list or numpy vector] frequencies (in GHz) at which the power 
                pattern is to be estimated. Frequencies differing by too much
                and extending over the usual bands cannot be given. 

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'degrees', 'altaz', and 'dircos'.
                Default = 'degrees'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up. If 'altaz', then 
                altitude and azimuth must be in degrees.

    pointing_center
                [numpy array] 1xN numpy array, where N is the same as in skypos.
                If None specified, pointing_center is assumed to be at zenith.
                
    pointing_coords
                [string] Coordiantes of the pointing center. If None specified, 
                it is assumed to be same as skyunits. Same allowed values as 
                skyunits. Default = None.

    power       [boolean] If set to True (default), compute power pattern,
                otherwise compute field pattern.

    Output:

    [Numpy array] Field or Power pattern at the specified sky positions. 
    -----------------------------------------------------------------------------
    """
    try:
        diameter, skypos, frequency
    except NameError:
        raise NameError('diameter, skypos and frequency are required in airy_disk_pattern().')

    skypos = NP.asarray(skypos)
    frequency = NP.asarray(frequency).ravel()

    if pointing_center is None:
        if skyunits == 'degrees':
            x = NP.radians(skypos)
        elif skyunits == 'altaz':
            x = NP.radians(90.0 - skypos[:,0])
        elif skyunits == 'dircos':
            x = NP.arcsin(NP.sqrt(skypos[:,0]**2 + skypos[:,1]**2))
        else:
            raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in GMRT_primary_beam().')
        zero_ind = x >= NP.pi/2   # Determine positions beyond the horizon
    else:
        if pointing_coords is None:
            pointing_coords = skyunits
        if skyunits == 'degrees':
            x = NP.radians(skypos)
        else:
            pc_altaz = pointing_center.reshape(1,-1)
            if pointing_coords == 'altaz':
                if pc_altaz.size != 2:
                    raise IndexError('Pointing center in Alt-Az coordinates must contain exactly two elements.')
            elif pointing_coords == 'dircos':
                if pc_altaz.size != 3:
                    raise IndexError('Pointing center in direction cosine coordinates must contain exactly three elements.')
                pc_altaz = GEOM.dircos2altaz(pc_altaz, units='degrees')

            skypos_altaz = NP.copy(skypos)
            if skyunits == 'dircos':
                skypos_altaz = GEOM.dircos2altaz(skypos, units='degrees')
            elif skyunits != 'altaz':
                raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in GMRT_primary_beam().')
            x = GEOM.sphdist(skypos_altaz[:,1], skypos_altaz[:,0], pc_altaz[0,1], pc_altaz[0,0])
            x = NP.radians(x)
            zero_ind = NP.logical_or(x >= NP.pi/2, skypos_altaz[:,0] <= 0.0)   # Determine positions beyond the horizon of the sky as well as those beyond the horizon of the dish, if it is pointed away from the horizon

    x = x.reshape(-1,1) # nsrc x 1
    sigma_aprtr = diameter / (2.0 * NP.sqrt(2.0 * NP.log(2.0))) / (FCNST.c/frequency) # in units of "u"
    # exp(-a t**2) <--> exp(-(pi*f)**2/a)
    # 2 x sigma_aprtr**2 = 1/a
    # 2 x sigma_dircos**2 = a / pi**2 = 1 / (2 * pi**2 * sigma_aprtr**2)
    sigma_dircos = 1.0 / (2 * NP.pi * sigma_aprtr)
    sigma_dircos = sigma_dircos.reshape(1,-1) # 1 x nchan
    dircos = NP.sin(x)
    pattern = NP.exp(-0.5 * (dircos/sigma_dircos)**2)
    pattern[zero_ind,:] = 0.0   # Blank all values beyond the horizon

    if power:
        pattern = NP.abs(pattern)**2
        
    return pattern

##########################################################################

def GMRT_primary_beam(skypos, frequency, skyunits='degrees'):

    """
    -----------------------------------------------------------------------------
    Primary beam power pattern for the GMRT dishes based on the polynomial 
    formula in AIPS task PBCOR

    Inputs:

    skypos      [list or numpy vector] Sky positions at which the power pattern 
                is to be estimated. Size is M x N where M is the number of 
                locations and N = 1 (if skyunits = degrees), N = 2 (if
                skyunits = altaz denoting Alt-Az coordinates), or N = 3 (if
                skyunits = dircos denoting direction cosine coordinates)

    frequency   [list or numpy vector] frequencies (in GHz) at which the power 
                pattern is to be estimated. Frequencies differing by too much
                and extending over the usual bands cannot be given. 

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'degrees', 'altaz', and 'dircos'.
                Default = 'degrees'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up

    Output:

    [Numpy array] Power pattern at the specified sky positions. 
    -----------------------------------------------------------------------------
    """

    try:
        skypos, frequency
    except NameError:
        raise NameError('skypos and frequency are required in GMRT_primary_beam().')

    frequency = NP.asarray(frequency).ravel()

    freq_ref = NP.asarray([0.235, 0.325, 0.610, 1.420]).reshape(-1,1)
    parms_ref = NP.asarray([[-3.366  , 46.159 , -29.963 ,  7.529  ], 
                            [-3.397  , 47.192 , -30.931 ,  7.803  ], 
                            [-3.486  , 47.749 , -35.203 , 10.399  ], 
                            [-2.27961, 21.4611,  -9.7929,  1.80153]])
    
    idx = NP.argmin(NP.abs(freq_ref - frequency[0])) # Index of closest value

    skypos = NP.asarray(skypos)

    if skyunits == 'degrees':
        x = (NP.repeat(skypos.reshape(-1,1), frequency.size, axis=1) * 60.0 * NP.repeat(frequency.reshape(1,-1), skypos.size, axis=0))**2
    elif skyunits == 'altaz':
        x = ((90.0-NP.repeat(skypos[:,0].reshape(-1,1), frequency.size, axis=1)) * 60.0 * NP.repeat(frequency.reshape(1,-1), skypos.size, axis=0))**2
    elif skyunits == 'dircos':
        x = (NP.degrees(NP.arccos(NP.repeat(skypos[:,-1].reshape(-1,1), frequency.size, axis=1))) * 60.0 * NP.repeat(frequency.reshape(1,-1), skypos.size, axis=0))**2
    else:
        raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in GMRT_primary_beam().')

    pb = 1.0 + parms_ref[idx,0]*x/1e3 + parms_ref[idx,1]*(x**2)/1e7 + \
         parms_ref[idx,2]*(x**3)/1e10 + parms_ref[idx,3]*(x**4)/1e13

    return pb

#################################################################################

def ground_plane_field_pattern(height, skypos, skycoords=None, wavelength=1.0,
                               angle_units=None, modifier=None, power=True):

    """
    -----------------------------------------------------------------------------
    Compute the field pattern of ground plane of specified height at the 
    specified sky positions at the specified wavelength. 

    Inputs:

    height           [scalar] height of the dipole above ground plane (in meters)

    skypos           [numpy array] Sky positions at which the field pattern is to 
                     be estimated. Size is M x N where M is the number of 
                     locations and N = 2 (if skycoords = 'altaz'), N = 2 or 3 (if 
                     skycoords = 'dircos'). If only transverse direction cosines 
                     are provided (N=2, skycoords='dircos'), the line-of-sight 
                     component will be determined appropriately.

    Keyword Inputs:

    skycoords        [string] string specifying the coordinate system of the sky 
                     positions. Accepted values are 'degrees', 'altaz', and 
                     'dircos'. Default = 'degrees'. If 'dircos', the direction 
                     cosines are aligned with the local East, North, and Up

    wavelength       [scalar, list or numpy vector] Wavelengths at which the field 
                     dipole pattern is to be estimated. Must be in the same units as 
                     the dipole length

    angle_units      [string] Units of angles used when Alt-Az coordinates are 
                     used in case of skypos or dipole_orientation. Accepted 
                     values are 'degrees' and 'radians'. If none given,
                     default='degrees' is used.

    modifier         [dictionary] Dictionary specifying modifications to the 
                     ground plane. If modifier is set to None, the ground plane 
                     is not modified from the analytical value. If not set to 
                     None, it may contain the following two keys:
                     'scale'   [scalar] positive value to scale the modifying
                               factor with. If not set, the scale factor to the
                               modification is unity.
                     'max'     [scalar] positive value to clip the modified and
                               scaled values to. If not set, there is no upper
                               limit

    power            [boolean] If set to True (default), compute power pattern,
                     otherwise compute field pattern.

    Output:

    Ground plane electric field or power pattern, a numpy array with number of 
    rows equal to the number of sky positions (which is equal to the number of 
    rows in skypos) and number of columns equal to number of wavelengths 
    specified. 
    -----------------------------------------------------------------------------
    """

    try:
        height, skypos
    except NameError:
        raise NameError('Dipole height above ground plane and sky positions must be specified. Check inputs.')

    if not isinstance(height, (int,float)):
        raise TypeError('Dipole height above ground plane should be a scalar.')

    if height <= 0.0:
        raise ValueError('Dipole height above ground plane should be positive.')

    if isinstance(wavelength, list):
        wavelength = NP.asarray(wavelength)
    elif isinstance(wavelength, (int, float)):
        wavelength = NP.asarray(wavelength).reshape(-1)
    elif not isinstance(wavelength, NP.ndarray):
        raise TypeError('Wavelength should be a scalar, list or numpy array.')
 
    if NP.any(wavelength <= 0.0):
        raise ValueError('Wavelength(s) should be positive.')

    if skycoords is not None:
        if not isinstance(skycoords, str):
            raise TypeError('skycoords must be a string. Allowed values are "altaz" and "dircos"')
        elif (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos".')
    else:
        raise ValueError('skycoords must be specified. Allowed values are "altaz" and "dircos"')

    if skycoords == 'altaz':
        if angle_units is None:
            angle_units = 'degrees'
        elif not isinstance(angle_units, str):
            raise TypeError('angle_units must be a string. Allowed values are "degrees" and "radians".')
        elif (angle_units != 'degrees') and (angle_units != 'radians'):
            raise ValueError('angle_units must be "degrees" or "radians".')

        skypos = NP.asarray(skypos)
        if angle_units == 'radians':
            skypos = NP.degrees(skypos)

        if skypos.ndim < 2:
            if len(skypos) == 2:
                skypos = NP.asarray(skypos).reshape(1,2)
            else:
                raise ValueError('skypos must be a Nx2 Numpy array.')
        elif skypos.ndim > 2:
            raise ValueError('skypos must be a Nx2 Numpy array.')
        else:
            if skypos.shape[1] != 2:
                raise ValueError('skypos must be a Nx2 Numpy array.')
            elif NP.any(skypos[:,0] < 0.0) or NP.any(skypos[:,0] > 90.0):
                raise ValueError('Altitudes in skypos have to be positive and <= 90 degrees')
        
        skypos_dircos = GEOM.altaz2dircos(skypos, units='degrees')
    else:
        if skypos.ndim < 2:
            if (len(skypos) == 2) or (len(skypos) == 3):
                skypos = NP.asarray(skypos).reshape(1,-1)
            else:
                raise ValueError('skypos must be a Nx2 Nx3 Numpy array.')
        elif skypos.ndim > 2:
            raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
        else:
            if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
                raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
            else:
                if NP.any(NP.abs(skypos[:,0]) > 1.0) or NP.any(NP.abs(skypos[:,1]) > 1.0):
                    raise ValueError('skypos in transverse direction cosine coordinates found to be exceeding unity.')
                else:
                    if skypos.shape[1] == 3:
                        eps = 1.0e-10
                        if NP.any(NP.abs(NP.sqrt(NP.sum(skypos**2, axis=1)) - 1.0) > eps) or NP.any(skypos[:,2] < 0.0):
                            print 'Warning: skypos in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                            skypos[:,2] = 1.0 - NP.sqrt(NP.sum(skypos[:,:2]**2,axis=1))
                    else:
                        skypos = NP.hstack((skypos, 1.0 - NP.asarray(NP.sqrt(NP.sum(skypos[:,:2]**2,axis=1))).reshape(-1,1)))
                        
        skypos_dircos = skypos

    k = 2 * NP.pi / wavelength

    skypos_altaz = GEOM.dircos2altaz(skypos_dircos, units='radians')
    ground_pattern = 2 * NP.sin(k.reshape(1,-1) * height * NP.sin(skypos_altaz[:,0].reshape(-1,1))) # array broadcasting

    if modifier is not None:
        if isinstance(modifier, dict):
            val = 1.0 / NP.sqrt(NP.abs(skypos_dircos[:,2]))
            if 'scale' in modifier:
                val *= modifier['scale']
            if 'max' in modifier:
                val = NP.clip(val, 0.0, modifier['max'])
            val = val[:,NP.newaxis]
            ground_pattern *= val

    max_pattern = 2 * NP.sin(k.reshape(1,-1) * height * NP.sin(NP.pi/2).reshape(-1,1)) # array broadcasting
    ground_pattern = ground_pattern / max_pattern

    if power:
        return NP.abs(ground_pattern)**2
    else:
        return ground_pattern

#################################################################################

def dipole_field_pattern(length, skypos, dipole_coords=None, skycoords=None, 
                         dipole_orientation=None, wavelength=1.0, angle_units=None, 
                         short_dipole_approx=False, half_wave_dipole_approx=True,
                         power=True):

    """
    -----------------------------------------------------------------------------
    Compute the dipole field pattern of specified length at the specified sky
    positions at the specified wavelength. 

    Inputs:

    length           [scalar] length of the dipole 

    skypos           [numpy array] Sky positions at which the field pattern is to 
                     be estimated. Size is M x N where M is the number of 
                     locations and N = 2 (if skycoords = 'altaz'), N = 2 or 3 (if 
                     skycoords = 'dircos'). If only transverse direction cosines 
                     are provided (N=2, skycoords='dircos'), the line-of-sight 
                     component will be determined appropriately.

    Keyword Inputs:

    dipole_coords    [string] specifies coordinate system for the unit vector of
                     the dipole element specified in dipole_orientation. Accepted
                     values are 'altaz' (Alt-Az) and 'dircos' (direction cosines).
                     If none provided, default='dircos' is used.

    dipole_orientation
                     [list or numpy array] Orientation of the dipole element
                     unit vector and magnitude specified by length. This unit
                     vector could be provided in a coordinate system specified
                     by dipole_coords. If dipole_coords='altaz', then the 
                     dipole_orientation should be a 2-element vector. If 'dircos'
                     is used, this could be a 2-element vector (transverse
                     direction cosines) where the third (line-of-sight) component
                     is determined, or a 3-element vector specifying all three
                     direction cosines. If set to None, defaults to eastward 
                     pointing dipole ([0.0, 90.0] if dipole_coords = 'altaz', or
                     [1.0, 0.0, 0.0]) if dipole_coords = 'dircos'

    skycoords        [string] string specifying the coordinate system of the sky 
                     positions. Accepted values are 'degrees', 'altaz', and 
                     'dircos'. Default = 'degrees'. If 'dircos', the direction 
                     cosines are aligned with the local East, North, and Up

    wavelength       [scalar, list or numpy vector] Wavelengths at which the field 
                     dipole pattern is to be estimated. Must be in the same units 
                     as the dipole length

    angle_units      [string] Units of angles used when Alt-Az coordinates are 
                     used in case of skypos or dipole_orientation. Accepted 
                     values are 'degrees' and 'radians'. If none given,
                     default='degrees' is used.

    short_dipole_approx
                     [boolean] if True, indicates short dipole approximation
                     is to be used. Otherwise, a more accurate expression is used
                     for the dipole pattern. Default=False. Both
                     short_dipole_approx and half_wave_dipole_approx cannot be set 
                     to True at the same time

    half_wave_dipole_approx
                     [boolean] if True, indicates half-wave dipole approximation
                     is to be used. Otherwise, a more accurate expression is used
                     for the dipole pattern. Default=True. Both
                     short_dipole_approx and half_wave_dipole_approx cannot be set 
                     to True at the same time

    power            [boolean] If set to True (default), compute power pattern,
                     otherwise compute field pattern.

    Output:

    Dipole Electric field or power pattern, a numpy array with number of rows 
    equal to the number of sky positions (which is equal to the number of rows 
    in skypos) and number of columns equal to number of wavelengths specified. 
    -----------------------------------------------------------------------------
    """

    try:
        length, skypos
    except NameError:
        raise NameError('Dipole length and sky positions must be specified. Check inputs.')

    if not isinstance(length, (int,float)):
        raise TypeError('Dipole length should be a scalar.')

    if length <= 0.0:
        raise ValueError('Dipole length should be positive.')

    if short_dipole_approx and half_wave_dipole_approx:
        raise ValueError('Both short dipole and half-wave dipole approximations cannot be made at the same time')

    if isinstance(wavelength, list):
        wavelength = NP.asarray(wavelength)
    elif isinstance(wavelength, (int, float)):
        wavelength = NP.asarray(wavelength).reshape(-1)
    elif not isinstance(wavelength, NP.ndarray):
        raise TypeError('Wavelength should be a scalar, list or numpy array.')
 
    if NP.any(wavelength <= 0.0):
        raise ValueError('Wavelength(s) should be positive.')

    # if ground_plane is not None:
    #     if not isinstance(ground_plane, (int,float)):
    #         raise TypeError('Height above ground plane should be a scalar.')

    #     if ground_plane <= 0.0:
    #         raise ValueError('Height above ground plane should be positive.')

    if dipole_coords is not None:
        if not isinstance(dipole_coords, str):
            raise TypeError('dipole_coords must be a string. Allowed values are "altaz" and "dircos"')
        elif (dipole_coords != 'altaz') and (dipole_coords != 'dircos'):
            raise ValueError('dipole_coords must be "altaz" or "dircos".')

    if skycoords is not None:
        if not isinstance(skycoords, str):
            raise TypeError('skycoords must be a string. Allowed values are "altaz" and "dircos"')
        elif (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos".')

    if (dipole_coords is None) and (skycoords is None):
        raise ValueError('At least one of dipole_coords and skycoords must be specified. Allowed values are "altaz" and "dircos"')
    elif (dipole_coords is not None) and (skycoords is None):
        skycoords = dipole_coords
    elif (dipole_coords is None) and (skycoords is not None):
        dipole_coords = skycoords    
    
    if (skycoords == 'altaz') or (dipole_coords == 'altaz'):
        if angle_units is None:
            angle_units = 'degrees'
        elif not isinstance(angle_units, str):
            raise TypeError('angle_units must be a string. Allowed values are "degrees" and "radians".')
        elif (angle_units != 'degrees') and (angle_units != 'radians'):
            raise ValueError('angle_units must be "degrees" or "radians".')

    if skycoords == 'altaz':
        skypos = NP.asarray(skypos)
        if angle_units == 'radians':
            skypos = NP.degrees(skypos)

        if skypos.ndim < 2:
            if len(skypos) == 2:
                skypos = NP.asarray(skypos).reshape(1,2)
            else:
                raise ValueError('skypos must be a Nx2 Numpy array.')
        elif skypos.ndim > 2:
            raise ValueError('skypos must be a Nx2 Numpy array.')
        else:
            if skypos.shape[1] != 2:
                raise ValueError('skypos must be a Nx2 Numpy array.')
            elif NP.any(skypos[:,0] < 0.0) or NP.any(skypos[:,0] > 90.0):
                raise ValueError('Altitudes in skypos have to be positive and <= 90 degrees')
        
        skypos_dircos = GEOM.altaz2dircos(skypos, units='degrees')
    else:
        if skypos.ndim < 2:
            if (len(skypos) == 2) or (len(skypos) == 3):
                skypos = NP.asarray(skypos).reshape(1,-1)
            else:
                raise ValueError('skypos must be a Nx2 Nx3 Numpy array.')
        elif skypos.ndim > 2:
            raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
        else:
            if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
                raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
            else:
                if NP.any(NP.abs(skypos[:,0]) > 1.0) or NP.any(NP.abs(skypos[:,1]) > 1.0):
                    raise ValueError('skypos in transverse direction cosine coordinates found to be exceeding unity.')
                else:
                    if skypos.shape[1] == 3:
                        eps = 1.0e-10
                        if NP.any(NP.abs(NP.sqrt(NP.sum(skypos**2, axis=1)) - 1.0) > eps) or NP.any(skypos[:,2] < 0.0):
                            print 'Warning: skypos in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                            skypos[:,2] = 1.0 - NP.sqrt(NP.sum(skypos[:,:2]**2,axis=1))
                    else:
                        skypos = NP.hstack((skypos, 1.0 - NP.asarray(NP.sqrt(NP.sum(skypos[:,:2]**2,axis=1))).reshape(-1,1)))
                        
        skypos_dircos = skypos

    if dipole_coords == 'altaz':
        if dipole_orientation is not None:
            dipole_orientation = NP.asarray(dipole_orientation)
            if angle_units == 'radians':
                dipole_orientation = NP.degrees(dipole_orientation)
    
            if dipole_orientation.ndim < 2:
                if len(dipole_orientation) == 2:
                    dipole_orientation = NP.asarray(dipole_orientation).reshape(1,2)
                else:
                    raise ValueError('dipole_orientation must be a Nx2 Numpy array.')
            elif dipole_orientation.ndim > 2:
                raise ValueError('dipole_orientation must be a Nx2 Numpy array.')
            else:
                if dipole_orientation.shape[1] != 2:
                    raise ValueError('dipole_orientation must be a Nx2 Numpy array.')
                elif NP.any(dipole_orientation[:,0] < 0.0) or NP.any(dipole_orientation[:,0] > 90.0):
                    raise ValueError('Altitudes in dipole_orientation have to be positive and <= 90 degrees')
        else:
            dipole_orietnation = NP.asarray([0.0, 90.0]).reshape(1,-1) # # Default dipole orientation points towards east

        dipole_orientation_dircos = GEOM.altaz2dircos(dipole_orientation, units='degrees')
    else:
        if dipole_orientation is not None:
            if dipole_orientation.ndim < 2:
                if (len(dipole_orientation) == 2) or (len(dipole_orientation) == 3):
                    dipole_orientation = NP.asarray(dipole_orientation).reshape(1,-1)
                else:
                    raise ValueError('dipole_orientation must be a Nx2 Nx3 Numpy array.')
            elif dipole_orientation.ndim > 2:
                raise ValueError('dipole_orientation must be a Nx2 or Nx3 Numpy array.')
            else:
                if (dipole_orientation.shape[1] < 2) or (dipole_orientation.shape[1] > 3):
                    raise ValueError('dipole_orientation must be a Nx2 or Nx3 Numpy array.')
                else:
                    if NP.any(NP.abs(dipole_orientation[:,0]) > 1.0) or NP.any(NP.abs(dipole_orientation[:,1]) > 1.0):
                        raise ValueError('dipole_orientation in transverse direction cosine coordinates found to be exceeding unity.')
                    else:
                        if dipole_orientation.shape[1] == 3:
                            eps = 1.0e-10
                            if NP.any(NP.abs(NP.sqrt(NP.sum(dipole_orientation**2, axis=1)) - 1.0) > eps) or NP.any(dipole_orientation[:,2] < 0.0):
                                print 'Warning: dipole_orientation in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                                dipole_orientation[:,2] = 1.0 - NP.sqrt(NP.sum(dipole_orientation[:,:2]**2,axis=1))
                        else:
                            dipole_orientation = NP.hstack((dipole_orientation, 1.0 - NP.asarray(NP.sqrt(NP.sum(dipole_orientation[:,:2]**2,axis=1))).reshape(-1,1)))
        else:
            dipole_orientation = NP.asarray([1.0, 0.0, 0.0]).reshape(1,-1) # Default dipole orientation points towards east
            
        dipole_orientation_dircos = dipole_orientation

    k = 2 * NP.pi / wavelength.reshape(1,-1)
    h = 0.5 * length
    dot_product = NP.dot(dipole_orientation_dircos, skypos_dircos.T).reshape(-1,1)
    angles = NP.arccos(dot_product)

    eps = 1.e-10
    zero_angles_ind = NP.abs(NP.abs(dot_product) - 1.0) < eps
    n_zero_angles = NP.sum(zero_angles_ind)
    reasonable_angles_ind = NP.abs(NP.abs(dot_product) - 1.0) > eps

    max_pattern = 1.0 # Normalization factor
    if short_dipole_approx:
        field_pattern = NP.sin(angles)
        field_pattern = NP.repeat(field_pattern.reshape(-1,1), wavelength.size, axis=1) # Repeat along wavelength axis
    else:
        if half_wave_dipole_approx:
            field_pattern = NP.cos(0.5 * NP.pi * NP.cos(angles)) / NP.sin(angles)
            field_pattern = NP.repeat(field_pattern.reshape(-1,1), wavelength.size, axis=1) # Repeat along wavelength axis
        else:
            max_pattern = 1.0 - NP.cos(k * h) # Maximum occurs at angle = NP.pi / 2
            field_pattern = (NP.cos(k*h*NP.cos(angles)) - NP.cos(k*h)) / NP.sin(angles)

        if n_zero_angles > 0:
            field_pattern[zero_angles_ind.ravel(),:] = k*h * NP.sin(k*h * NP.cos(angles[zero_angles_ind])) * NP.tan(angles[zero_angles_ind]) # Correct expression from L' Hospital rule
    
    if power:
        return NP.abs(field_pattern / max_pattern)**2
    else:
        return field_pattern / max_pattern

#################################################################################

def isotropic_radiators_array_field_pattern(nax1, nax2, sep1, sep2=None,
                                            skypos=None, wavelength=1.0,
                                            east2ax1=None, skycoords='altaz',
                                            pointing_center=None, power=True):

    """
    -----------------------------------------------------------------------------
    Compute the electric field pattern at the specified sky positions due to an 
    array of antennas.

    Inputs:

    nax1          [scalar] Number of radiator elements along axis #1

    nax2          [scalar] Number of radiator elements along axis #2

    sep1          [scalar] Distance along axis #1 between two adjacent radiator
                  elements along axis #1

    Keyword Inputs:

    sep2          [scalar] Distance along axis #2 between two adjacent radiator
                  elements along axis #2. If none specified, sep2 is set equal to
                  sep1. Same units as sep1.

    skypos        [numpy array] Sky positions at which the field pattern is to be
                  estimated. Size is M x N where M is the number of locations and 
                  N = 1 (if skycoords = degrees, for azimuthally symmetric
                  telescopes such as VLA and GMRT which have parabolic dishes), 
                  N = 2 (if skycoords = altaz denoting Alt-Az coordinates), or 
                  N = 3 (if skycoords = dircos denoting direction cosine
                  coordinates)

    skycoords     [string] string specifying the coordinate system of the sky 
                  positions. Accepted values are 'degrees', 'altaz', and 
                  'dircos'. Default = 'degrees'. If 'dircos', the direction 
                  cosines are aligned with the local East, North, and Up

    wavelength    [scalar, list or numpy vector] Wavelengths at which the field 
                  dipole pattern is to be estimated. Must be in the same units as 
                  the dipole length

    east2ax1      [scalar] Angle (in degrees) the primary axis of the array makes 
                  with the local East (positive anti-clockwise). 
                  
    pointing_center  [list or numpy array] coordinates of pointing center (in the same
                  coordinate system as that of sky coordinates specified by
                  skycoords). 2-element vector if skycoords='altaz'. 2- or 
                  3-element vector if skycoords='dircos'. Only used with phased 
                  array primary beams or dishes excluding those of VLA and GMRT.

    power         [boolean] If set to True (default), compute power pattern,
                  otherwise compute field pattern.

    Output:

    Array Electric field or power pattern, number of rows equal to the number of 
    sky positions (which is equal to the number of rows in skypos), and number of 
    columns equal to the number of wavelengths. The array pattern is the product 
    of the array patterns along each axis.
    -----------------------------------------------------------------------------
    """

    try:
        nax1, nax2, sep1, skypos
    except NameError:
        raise NameError('Number of radiators along axis 1 and 2 and their separation must be specified. Check inputs.')

    if skypos is None:
        raise NameError('skypos must be specified in Alt-Az or direction cosine units as a Numpy array. Check inputs.')

    if not isinstance(nax1, int):
        raise TypeError('nax1 must be a positive integer.')
    elif nax1 <= 0:
        raise ValueError('nax1 must be a positive integer.')

    if not isinstance(nax2, int):
        raise TypeError('nax2 must be a positive integer.')
    elif nax2 <= 0:
        raise ValueError('nax2 must be a positive integer.')

    if not isinstance(sep1, (int,float)):
        raise TypeError('sep1 must be a positive scalar.')
    elif sep1 <= 0:
        raise ValueError('sep1 must be a positive value.')

    if sep2 is None:
        sep2 = sep1

    if isinstance(wavelength, list):
        wavelength = NP.asarray(wavelength)
    elif isinstance(wavelength, (int, float)):
        wavelength = NP.asarray(wavelength).reshape(-1)
    elif not isinstance(wavelength, NP.ndarray):
        raise TypeError('Wavelength should be a scalar, list or numpy array.')
 
    if NP.any(wavelength <= 0.0):
        raise ValueError('Wavelength(s) should be positive.')

    # if not isinstance(wavelength, (int,float)):
    #     raise TypeError('wavelength must be a positive scalar.')
    # elif wavelength <= 0:
    #     raise ValueError('wavelength must be a positive value.')

    if not isinstance(east2ax1, (int,float)):
        raise TypeError('east2ax1 must be a scalar.')

    if not isinstance(skypos, NP.ndarray):
        raise TypeError('skypos must be a Numpy array.')
    
    if skycoords is not None:
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
            if skypos.ndim < 2:
                if skypos.size == 2:
                    skypos = NP.asarray(skypos).reshape(1,2)
                else:
                    raise ValueError('skypos must be a Nx2 Numpy array.')
            elif skypos.ndim > 2:
                raise ValueError('skypos must be a Nx2 Numpy array.')
            else:
                if skypos.shape[1] != 2:
                    raise ValueError('skypos must be a Nx2 Numpy array.')
                elif NP.any(skypos[:,0] < 0.0) or NP.any(skypos[:,0] > 90.0):
                    raise ValueError('Altitudes in skypos have to be positive and <= 90 degrees')
        else:
            if skypos.ndim < 2:
                if (skypos.size == 2) or (skypos.size == 3):
                    skypos = NP.asarray(skypos).reshape(1,-1)
                else:
                    raise ValueError('skypos must be a Nx2 Nx3 Numpy array.')
            elif skypos.ndim > 2:
                raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
            else:
                if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
                    raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
                elif skypos.shape[1] == 2:
                    if NP.any(NP.sum(skypos**2, axis=1) > 1.0):
                        raise ValueError('skypos in direction cosine coordinates are invalid.')
                
                    skypos = NP.hstack((skypos, NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1)))
                else:
                    eps = 1.0e-10
                    if NP.any(NP.abs(NP.sum(skypos**2, axis=1) - 1.0) > eps) or NP.any(skypos[:,2] < 0.0):
                        if verbose:
                            print '\tWarning: skypos in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                        skypos[:,2] = NP.sqrt(1.0 - NP.sum(skypos[:2]**2, axis=1))
    else:
        raise ValueError('skycoords has not been set.')

    if pointing_center is None:
        if skycoords == 'altaz':
            pointing_center = NP.asarray([90.0, 0.0]) # Zenith in Alt-Az coordinates
        else:
            pointing_center = NP.asarray([0.0, 0.0, 1.0]) # Zenith in direction-cosine coordinates
    else:
        if not isinstance(pointing_center, (list, NP.ndarray)):
            raise TypeError('pointing_center must be a list or numpy array')
        
        pointing_center = NP.asarray(pointing_center)
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
            if pointing_center.size != 2:
                raise ValueError('pointing_center must be a 2-element vector in Alt-Az coordinates.')
            else:
                pointing_center = pointing_center.ravel()

            if NP.any(pointing_center[0] < 0.0) or NP.any(pointing_center[0] > 90.0):
                raise ValueError('Altitudes in pointing_center have to be positive and <= 90 degrees')
        else:
            if (pointing_center.size < 2) or (pointing_center.size > 3):
                raise ValueError('pointing_center must be a 2- or 3-element vector in direction cosine coordinates')
            else:
                pointing_center = pointing_center.ravel()

            if pointing_center.size == 2:
                if NP.sum(pointing_center**2) > 1.0:
                    raise ValueError('pointing_center in direction cosine coordinates are invalid.')
                
                pointing_center = NP.hstack((pointing_center, NP.sqrt(1.0-NP.sum(pointing_center**2))))
            else:
                eps = 1.0e-10
                if (NP.abs(NP.sum(pointing_center**2) - 1.0) > eps) or (pointing_center[2] < 0.0):
                    if verbose:
                        print '\tWarning: pointing_center in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                    pointing_center[2] = NP.sqrt(1.0 - NP.sum(pointing_center[:2]**2))

    # skypos_dircos_relative = NP.empty((skypos.shape[0],3))

    if east2ax1 is not None:
        if not isinstance(east2ax1, (int, float)):
            raise TypeError('east2ax1 must be a scalar value.')
        else:
            if skycoords == 'altaz':
                # skypos_dircos_rotated = GEOM.altaz2dircos(NP.hstack((skypos[:,0].reshape(-1,1),NP.asarray(skypos[:,1]-east2ax1).reshape(-1,1))), units='degrees')
                # pointing_center_dircos_rotated = GEOM.altaz2dircos([pointing_center[0], pointing_center[1]-east2ax1], units='degrees')

                # Rotate in Az. Remember Az is measured clockwise from North
                # whereas east2ax1 is measured anti-clockwise from East.
                # Therefore, newAz = Az + East2ax1 wrt to principal axis
                skypos_dircos_rotated = GEOM.altaz2dircos(NP.hstack((skypos[:,0].reshape(-1,1),NP.asarray(skypos[:,1]+east2ax1).reshape(-1,1))), units='degrees')
                pointing_center_dircos_rotated = GEOM.altaz2dircos([pointing_center[0], pointing_center[1]+east2ax1], units='degrees')
            else:
                angle = NP.radians(east2ax1)
                rotation_matrix = NP.asarray([[NP.cos(angle), NP.sin(angle), 0.0],
                                              [-NP.sin(angle), NP.cos(angle),  0.0],
                                              [0.0,            0.0,           1.0]])
                skypos_dircos_rotated = NP.dot(skypos, rotation_matrix.T)
                pointing_center_dircos_rotated = NP.dot(pointing_center, rotation_matrix.T)

            skypos_dircos_relative = skypos_dircos_rotated - NP.repeat(pointing_center_dircos_rotated.reshape(1,-1), skypos.shape[0], axis=0)
    else:
        if skycoords == 'altaz':
            skypos_dircos = GEOM.altaz2dircos(skypos, units='degrees')
            pointing_center_dircos = GEOM.altaz2dircos([pointing_center[0], pointing_center[1]-east2ax1], units='degrees')
        else:
            skypos_dircos_rotated = skypos
        skypos_dircos_relative = skypos_dircos - NP.repeat(pointing_center_dircos, skypos.shape[0], axis=0)

    phi = 2 * NP.pi * sep1 * NP.repeat(skypos_dircos_relative[:,0].reshape(-1,1), wavelength.size, axis=1) / NP.repeat(wavelength.reshape(1,-1), skypos.shape[0], axis=0) 
    psi = 2 * NP.pi * sep2 * NP.repeat(skypos_dircos_relative[:,1].reshape(-1,1), wavelength.size, axis=1) / NP.repeat(wavelength.reshape(1,-1), skypos.shape[0], axis=0) 

    eps = 1.0e-10
    zero_phi = NP.abs(phi) < eps
    zero_psi = NP.abs(psi) < eps

    term1 = NP.sin(0.5*nax1*phi) / NP.sin(0.5*phi) / nax1
    term1_zero_phi = NP.cos(0.5*nax1*phi[zero_phi]) / NP.cos(0.5*phi[zero_phi]) # L'Hospital rule
    term1[zero_phi] = term1_zero_phi.ravel()

    term2 = NP.sin(0.5*nax1*psi) / NP.sin(0.5*psi) / nax1
    term2_zero_psi = NP.cos(0.5*nax1*psi[zero_psi]) / NP.cos(0.5*psi[zero_psi]) # L'Hospital rule
    term2[zero_psi] = term2_zero_psi.ravel()

    pb =  term1 * term2
    if power:
        pb = NP.abs(pb)**2
    return pb

#################################################################################

def array_field_pattern(antpos, skypos, skycoords='altaz', pointing_info=None, 
                        wavelength=1.0, power=True):

    """
    -----------------------------------------------------------------------------
    A routine to generate field pattern from an array of generic shape made of
    isotropic radiator elements. This can supercede the functionality of 
    isotropic_radiators_array_field_pattern() because the latter can only handle
    rectangular or square arrays with equally spaced elements. Secondly, this 
    routine can handle beam pointing through specification of pointing center or
    beamformer delays. Effect of jitter in the delay settings of the beamformer
    can also be taken into account.

    Inputs:

    antpos    [2- or 3-column numpy array] The position of elements in tile. The
              coordinates are assumed to be in the local ENU coordinate system  
              in meters. If a 2-column array is provided, the third column is 
              assumed to be made of zeros. Each row is for one element. No 
              default. 

    skypos    [2- or 3-column numpy array] The positions on the sky for which 
              the array field pattern is to be estimated. The coordinate system 
              specified using the keyword input skycoords. If skycoords is set
              to 'altaz', skypos must be a 2-column array that obeys Alt-Az 
              conventions with altitude in the first column and azimuth in the 
              second column. Both altitude and azimuth must be in degrees. If 
              skycoords is set to 'dircos', a 3- or 2-column (the
              third column is automatically determined from direction cosine 
              rules), it must obey conventions of direction cosines. The first 
              column is l (east), the second is m (north) and third is n (up).
              Default will be set to zenith position in the coordinate system 
              specified.

    skycoords [string scalar] Coordinate system of sky positions specified in 
              skypos. Accepted values are 'altaz' (Alt-Az) or 'dircos' (direction
              cosines)

    pointing_info 
              [dictionary] A dictionary consisting of information relating to 
              pointing center. The pointing center can be specified either via
              element delay compensation or by directly specifying the pointing
              center in a certain coordinate system. Default = None (pointing 
              centered at zenith).This dictionary consists of the following tags 
              and values:
              'delays'          [numpy array] Delays (in seconds) to be applied 
                                to the tile elements. Size should be equal to 
                                number of tile elements (number of rows in
                                antpos). Default = None will set all element
                                delays to zero phasing them to zenith. 
              'pointing_center' [numpy array] This will apply in the absence of 
                                key 'delays'. This can be specified as a row 
                                vector. Should have two-columns if using Alt-Az
                                coordinates, or two or three columns if using
                                direction cosines. There is no default. The
                                coordinate system must be specified in
                                'pointing_coords' if 'pointing_center' is to be
                                used.
              'pointing_coords' [string scalar] Coordinate system in which the
                                pointing_center is specified. Accepted values 
                                are 'altaz' or 'dircos'. Must be provided if
                                'pointing_center' is to be used. No default.
              'delayerr'        [int, float] RMS jitter in delays used in the
                                beamformer. Random jitters are drawn from a 
                                normal distribution with this rms. Must be
                                a non-negative scalar. If not provided, it
                                defaults to 0 (no jitter). 
              'gains'           [numpy array] Complex element gains. Must be of 
                                size equal to the number of elements as 
                                specified by the number of rows in antpos. If
                                set to None (default), all element gains are 
                                assumed to be unity.
              'gainerr'         [int, float] RMS error in voltage amplitude in 
                                dB to be used in the beamformer. Random jitters 
                                are drawn from a normal distribution in 
                                logarithm units which are then converted to 
                                linear units. Must be a non-negative scalar. If 
                                not provided, it defaults to 0 (no jitter). 
              'nrand'           [int] number of random realizations of gainerr 
                                and/or delayerr to be generated. Must be 
                                positive. If none provided, it defaults to 1.
              
    wavelength [scalar, list or numpy vector] Wavelengths at which the field 
               dipole pattern is to be estimated. Must be in the same units as 
               element positions in antpos.
                               
    power      [boolean] If set to True (default), compute power pattern,
               otherwise compute field pattern.

    Output:

    Returns a complex electric field or power pattern as a MxN numpy array, 
    M=number of sky positions, N=number of wavelengths.
    -----------------------------------------------------------------------------
    """

    try:
        antpos, skypos
    except NameError:
        raise NameError('antpos and skypos must be provided for array_beamformer().')
        
    if not isinstance(antpos, NP.ndarray):
        raise TypeError('antenna positions in antpos must be a numpy array.')
    else:
        if (len(antpos.shape) != 2):
            raise ValueError('antpos must be a 2-dimensional 2- or 3-column numpy array')
        else:
            if antpos.shape[1] == 2:
                antpos = NP.hstack((antpos, NP.zeros(antpos.shape[0]).reshape(-1,1)))
            elif antpos.shape[1] != 3:
                raise ValueError('antpos must be a 2- or 3-column array')
            antpos = antpos.astype(NP.float32)

    if pointing_info is None:
        delays = NP.zeros(antpos.shape[0])
        gains = NP.ones(antpos.shape[0])
        nrand = 1
    else:
        if 'nrand' in pointing_info:
            nrand = pointing_info['nrand']
            if nrand is None: 
                nrand = 1
            elif not isinstance(nrand, int):
                raise TypeError('nrand must be an integer')
            elif nrand < 1:
                raise ValueError('nrand must be positive')
        else:
            nrand = 1

        if 'delays' in pointing_info:
            delays = pointing_info['delays']
            if delays is None:
                delays = NP.zeros(antpos.shape[0])
            elif not isinstance(delays, NP.ndarray):
                raise TypeError('delays must be a numpy array')
            else:
                if delays.size != antpos.shape[0]:
                    raise ValueError('size of delays must be equal to the number of antennas')
            delays = delays.ravel()
        elif 'pointing_center' in pointing_info:
            if 'pointing_coords' not in pointing_info:
                raise KeyError('pointing_coords not specified.')
            elif pointing_info['pointing_coords'] == 'altaz':
                pointing_center = GEOM.altaz2dircos(pointing_info['pointing_center'].reshape(1,-1), units='degrees')
            elif pointing_info['pointing_coords'] == 'dircos':            
                if NP.sum(pointing_info['pointing_center']**2 > 1.0):
                    raise ValueError('Invalid direction cosines specified in pointing_center')
                pointing_center = pointing_info['pointing_center'].reshape(1,-1)
            else:
                raise ValueError('pointing_coords must be set to "dircos" or "altaz"')
            delays = NP.dot(antpos, pointing_center.T) / FCNST.c # Opposite sign as that used for determining geometric delays later because this is delay compensation
        else:
            delays = NP.zeros(antpos.shape[0], dtype=NP.float32)

        if 'gains' in pointing_info:
            gains = pointing_info['gains']
            if gains is None:
                gains = NP.ones(antpos.shape[0])
            elif not isinstance(gains, NP.ndarray):
                raise TypeError('gains must be a numpy array')
            else:
                if gains.size != antpos.shape[0]:
                    raise ValueError('size of gains must be equal to the number of antennas')
            gains = gains.ravel()
        else:
            gains = NP.ones(antpos.shape[0], dtype=NP.float32)

        if 'delayerr' in pointing_info:
            delayerr = pointing_info['delayerr']
            if delayerr is not None:
                if isinstance(delayerr, (int, float)):
                    if delayerr < 0.0:
                        raise ValueError('delayerr must be non-negative')
                    delays = delays.reshape(antpos.shape[0],1) + delayerr * NP.random.standard_normal((antpos.shape[0],nrand))
                else:
                    raise TypeError('delayerr must be an integer or float')

        if 'gainerr' in pointing_info:
            gainerr = pointing_info['gainerr']
            if gainerr is not None:
                if isinstance(gainerr, (int, float)):
                    if gainerr < 0.0:
                        raise ValueError('gainerr must be non-negative')
                    gainerr /= 10.0         # Convert from dB to logarithmic units
                    gains = gains.reshape(antpos.shape[0],1) * 10**(gainerr * NP.random.standard_normal((antpos.shape[0],nrand)))
                else:
                    raise TypeError('gainerr must be an integer or float')

    gains = gains.astype(NP.float32)        
    delays = delays.astype(NP.float32)

    if not isinstance(skypos, NP.ndarray):
        raise TypeError('skypos must be a Numpy array.')
    
    if skycoords is not None:
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
            if skypos.ndim < 2:
                if skypos.size == 2:
                    skypos = NP.asarray(skypos).reshape(1,2)
                else:
                    raise ValueError('skypos must be a Nx2 Numpy array.')
            elif skypos.ndim > 2:
                raise ValueError('skypos must be a Nx2 Numpy array.')
            else:
                if skypos.shape[1] != 2:
                    raise ValueError('skypos must be a Nx2 Numpy array.')
                elif NP.any(skypos[:,0] < 0.0) or NP.any(skypos[:,0] > 90.0):
                    raise ValueError('Altitudes in skypos have to be positive and <= 90 degrees')
            skypos = GEOM.altaz2dircos(skypos, 'degrees') # Convert sky positions to direction cosines
        else:
            if skypos.ndim < 2:
                if (skypos.size == 2) or (skypos.size == 3):
                    skypos = NP.asarray(skypos).reshape(1,-1)
                else:
                    raise ValueError('skypos must be a Nx2 Nx3 Numpy array.')
            elif skypos.ndim > 2:
                raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
            else:
                if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
                    raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
                elif skypos.shape[1] == 2:
                    if NP.any(NP.sum(skypos**2, axis=1) > 1.0):
                        raise ValueError('skypos in direction cosine coordinates are invalid.')
                
                    skypos = NP.hstack((skypos, NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1)))
                else:
                    eps = 1.0e-10
                    if NP.any(NP.abs(NP.sum(skypos**2, axis=1) - 1.0) > eps) or NP.any(skypos[:,2] < 0.0):
                        if verbose:
                            print '\tWarning: skypos in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                        skypos[:,2] = NP.sqrt(1.0 - NP.sum(skypos[:2]**2, axis=1))
    else:
        raise ValueError('skycoords has not been set.')
    
    skypos = skypos.astype(NP.float32, copy=False)
    
    if isinstance(wavelength, list):
        wavelength = NP.asarray(wavelength)
    elif isinstance(wavelength, (int, float)):
        wavelength = NP.asarray(wavelength).reshape(-1)
    elif not isinstance(wavelength, NP.ndarray):
        raise TypeError('Wavelength should be a scalar, list or numpy array.')
 
    if NP.any(wavelength <= 0.0):
        raise ValueError('Wavelength(s) should be positive.')

    wavelength = wavelength.astype(NP.float32)

    geometric_delays = -NP.dot(antpos, skypos.T) / FCNST.c
    geometric_delays = geometric_delays[:,:,NP.newaxis,NP.newaxis].astype(NP.float32, copy=False) # Add an axis for wavelengths, and random realizations of beamformer settings
    
    gains = gains.reshape(antpos.shape[0],1,1,nrand).astype(NP.complex64, copy=False)
    delays = delays.reshape(antpos.shape[0],1,1,nrand)
    wavelength = wavelength.reshape(1,1,-1,1).astype(NP.float32, copy=False)

    retvalue = geometric_delays + delays
    retvalue = retvalue.astype(NP.complex64, copy=False)
    # retvalue *= 1j * 2*NP.pi * FCNST.c
    # retvalue = retvalue.astype(NP.complex64, copy=False)
    # retvalue = retvalue/wavelength
    retvalue = NP.exp(1j * 2*NP.pi * FCNST.c/wavelength * retvalue).astype(NP.complex64, copy=False)
    retvalue *= gains/antpos.shape[0]
    retvalue = NP.sum(retvalue.astype(NP.complex64), axis=0)

    # field_pattern = NP.sum(gains * NP.exp(1j * 2*NP.pi * (geometric_delays+delays) * FCNST.c / wavelength), axis=0) / antpos.shape[0]

    # return field_pattern

    if power:
        retvalue = NP.abs(retvalue)**2
    return retvalue
                
#################################################################################

def uniform_rectangular_aperture(sides, skypos, frequency, skyunits='altaz', 
                                 east2ax1=None, pointing_center=None, 
                                 power=True):

    """
    -----------------------------------------------------------------------------
    Compute the electric field or power pattern at the specified sky positions 
    due to a uniformly illuminated rectangular aperture

    Inputs:

    sides       [scalar, list or numpy array]  Sides of the rectangle (in m). If
                scalar, it will be assumed to be identical for both sides which
                is a square. If a list or numpy array, it must have two 
                elements

    skypos      [list or numpy vector] Sky positions at which the power pattern 
                is to be estimated. Size is M x N where M is the number of 
                locations, N = 2 (if skyunits = altaz denoting Alt-Az 
                coordinates), or N = 3 (if skyunits = dircos denoting direction 
                cosine coordinates). If skyunits = altaz, then altitude and 
                azimuth must be in degrees

    frequency   [list or numpy vector] frequencies (in GHz) at which the power 
                pattern is to be estimated. Frequencies differing by too much
                and extending over the usual bands cannot be given. 

    Keyword Inputs:

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'altaz', and 'dircos'.
                Default = 'altaz'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up. If 'altaz', then 
                altitude and azimuth must be in degrees.

    east2ax1    [scalar] Angle (in degrees) the primary axis of the array makes 
                with the local East (positive anti-clockwise). 
                  
    pointing_center  
                [list or numpy array] coordinates of pointing center (in the same
                coordinate system as that of sky coordinates specified by
                skycoords). 2-element vector if skycoords='altaz'. 2- or 
                3-element vector if skycoords='dircos'. 

    power       [boolean] If set to True (default), compute power pattern, 
                otherwise compute field pattern 

    Output:

    Electric field pattern or power pattern, number of rows equal to the number 
    of sky positions (which is equal to the number of rows in skypos), and 
    number of columns equal to the number of wavelengths. 
    -----------------------------------------------------------------------------
    """

    try:
        sides, skypos, frequency
    except NameError:
        raise NameError('Rectangular antenna sides, skypos, frequency must be specified')

    if isinstance(sides, (int,float)):
        sides = NP.asarray([sides]*2, dtype=NP.float)
    elif isinstance(sides, list):
        sides = NP.asarray(sides).astype(NP.float)
    elif not isinstance(sides, NP.ndarray):
        raise TypeError('Antenna sides must be a scalar, list or numpy array')
    
    sides = sides.astype(NP.float)
    if sides.size == 1:
        sides = sides.ravel() + NP.zeros(2)
    elif sides.size == 2:
        sides = sides.ravel()
        sides= sides.astype(NP.float)
    else:
        raise ValueError('Antenna sides must not have more than 2 elements')

    if NP.any(sides < 0.0):
        raise ValueError('Antenna sides must not be negative')

    if isinstance(frequency, list):
        frequency = NP.asarray(frequency)
    elif isinstance(frequency, (int, float)):
        frequency = NP.asarray(frequency).reshape(-1)
    elif not isinstance(frequency, NP.ndarray):
        raise TypeError('Frequency should be a scalar, list or numpy array.')
 
    if NP.any(frequency <= 0.0):
        raise ValueError('Frequency(s) should be positive.')

    if not isinstance(east2ax1, (int,float)):
        raise TypeError('east2ax1 must be a scalar.')

    if not isinstance(skypos, NP.ndarray):
        raise TypeError('skypos must be a Numpy array.')

    frequency = NP.asarray(frequency).ravel()
    wavelength = FCNST.c / frequency

    if skycoords is not None:
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
            if skypos.ndim < 2:
                if skypos.size == 2:
                    skypos = NP.asarray(skypos).reshape(1,2)
                else:
                    raise ValueError('skypos must be a Nx2 Numpy array.')
            elif skypos.ndim > 2:
                raise ValueError('skypos must be a Nx2 Numpy array.')
            else:
                if skypos.shape[1] != 2:
                    raise ValueError('skypos must be a Nx2 Numpy array.')
                elif NP.any(skypos[:,0] < 0.0) or NP.any(skypos[:,0] > 90.0):
                    raise ValueError('Altitudes in skypos have to be positive and <= 90 degrees')
        else:
            if skypos.ndim < 2:
                if (skypos.size == 2) or (skypos.size == 3):
                    skypos = NP.asarray(skypos).reshape(1,-1)
                else:
                    raise ValueError('skypos must be a Nx2 Nx3 Numpy array.')
            elif skypos.ndim > 2:
                raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
            else:
                if (skypos.shape[1] < 2) or (skypos.shape[1] > 3):
                    raise ValueError('skypos must be a Nx2 or Nx3 Numpy array.')
                elif skypos.shape[1] == 2:
                    if NP.any(NP.sum(skypos**2, axis=1) > 1.0):
                        raise ValueError('skypos in direction cosine coordinates are invalid.')
                
                    skypos = NP.hstack((skypos, NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1)))
                else:
                    eps = 1.0e-10
                    if NP.any(NP.abs(NP.sum(skypos**2, axis=1) - 1.0) > eps) or NP.any(skypos[:,2] < 0.0):
                        if verbose:
                            print '\tWarning: skypos in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                        skypos[:,2] = NP.sqrt(1.0 - NP.sum(skypos[:2]**2, axis=1))
    else:
        raise ValueError('skycoords has not been set.')
    
    if pointing_center is None:
        if skycoords == 'altaz':
            pointing_center = NP.asarray([90.0, 0.0]) # Zenith in Alt-Az coordinates
        else:
            pointing_center = NP.asarray([0.0, 0.0, 1.0]) # Zenith in direction-cosine coordinates
    else:
        if not isinstance(pointing_center, (list, NP.ndarray)):
            raise TypeError('pointing_center must be a list or numpy array')
        
        pointing_center = NP.asarray(pointing_center)
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
            if pointing_center.size != 2:
                raise ValueError('pointing_center must be a 2-element vector in Alt-Az coordinates.')
            else:
                pointing_center = pointing_center.ravel()

            if NP.any(pointing_center[0] < 0.0) or NP.any(pointing_center[0] > 90.0):
                raise ValueError('Altitudes in pointing_center have to be positive and <= 90 degrees')
        else:
            if (pointing_center.size < 2) or (pointing_center.size > 3):
                raise ValueError('pointing_center must be a 2- or 3-element vector in direction cosine coordinates')
            else:
                pointing_center = pointing_center.ravel()

            if pointing_center.size == 2:
                if NP.sum(pointing_center**2) > 1.0:
                    raise ValueError('pointing_center in direction cosine coordinates are invalid.')
                
                pointing_center = NP.hstack((pointing_center, NP.sqrt(1.0-NP.sum(pointing_center**2))))
            else:
                eps = 1.0e-10
                if (NP.abs(NP.sum(pointing_center**2) - 1.0) > eps) or (pointing_center[2] < 0.0):
                    if verbose:
                        print '\tWarning: pointing_center in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                    pointing_center[2] = NP.sqrt(1.0 - NP.sum(pointing_center[:2]**2))

    if east2ax1 is not None:
        if not isinstance(east2ax1, (int, float)):
            raise TypeError('east2ax1 must be a scalar value.')
        else:
            if skycoords == 'altaz':
                # skypos_dircos_rotated = GEOM.altaz2dircos(NP.hstack((skypos[:,0].reshape(-1,1),NP.asarray(skypos[:,1]-east2ax1).reshape(-1,1))), units='degrees')
                # pointing_center_dircos_rotated = GEOM.altaz2dircos([pointing_center[0], pointing_center[1]-east2ax1], units='degrees')

                # Rotate in Az. Remember Az is measured clockwise from North
                # whereas east2ax1 is measured anti-clockwise from East.
                # Therefore, newAz = Az + East2ax1 wrt to principal axis
                skypos_dircos_rotated = GEOM.altaz2dircos(NP.hstack((skypos[:,0].reshape(-1,1),NP.asarray(skypos[:,1]+east2ax1).reshape(-1,1))), units='degrees')
                pointing_center_dircos_rotated = GEOM.altaz2dircos([pointing_center[0], pointing_center[1]+east2ax1], units='degrees')
            else:
                angle = NP.radians(east2ax1)
                rotation_matrix = NP.asarray([[NP.cos(angle), NP.sin(angle), 0.0],
                                              [-NP.sin(angle), NP.cos(angle),  0.0],
                                              [0.0,            0.0,           1.0]])
                skypos_dircos_rotated = NP.dot(skypos, rotation_matrix.T)
                pointing_center_dircos_rotated = NP.dot(pointing_center, rotation_matrix.T)

            skypos_dircos_relative = skypos_dircos_rotated - NP.repeat(pointing_center_dircos_rotated.reshape(1,-1), skypos.shape[0], axis=0)
    else:
        if skycoords == 'altaz':
            skypos_dircos = GEOM.altaz2dircos(skypos, units='degrees')
            pointing_center_dircos = GEOM.altaz2dircos([pointing_center[0], pointing_center[1]-east2ax1], units='degrees')
        else:
            skypos_dircos_rotated = skypos
        skypos_dircos_relative = skypos_dircos - NP.repeat(pointing_center_dircos, skypos.shape[0], axis=0)

    arg1 = sides[0] * skypos_dircos_relative[:,0].reshape(-1,1) / wavelength.reshape(1,-1)
    arg2 = sides[1] * skypos_dircos_relative[:,1].reshape(-1,1) / wavelength.reshape(1,-1)
    ab = NP.sinc(arg1) * NP.sinc(arg2)
    if power:
        ab = NP.abs(ab)**2

    return ab
    
################################################################################

def uniform_square_aperture(side, skypos, frequency, skyunits='altaz', 
                            east2ax1=None, pointing_center=None, 
                            power=True):

    """
    -----------------------------------------------------------------------------
    Compute the electric field or power pattern at the specified sky positions 
    due to a uniformly illuminated square aperture

    Inputs:

    side        [scalar] Sides of the square (in m)

    skypos      [list or numpy vector] Sky positions at which the power pattern 
                is to be estimated. Size is M x N where M is the number of 
                locations, N = 2 (if skyunits = altaz denoting Alt-Az 
                coordinates), or N = 3 (if skyunits = dircos denoting direction 
                cosine coordinates). If skyunits = altaz, then altitude and 
                azimuth must be in degrees

    frequency   [list or numpy vector] frequencies (in GHz) at which the power 
                pattern is to be estimated. Frequencies differing by too much
                and extending over the usual bands cannot be given. 

    Keyword Inputs:

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'altaz', and 'dircos'.
                Default = 'altaz'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up. If 'altaz', then 
                altitude and azimuth must be in degrees.

    east2ax1    [scalar] Angle (in degrees) the primary axis of the array makes 
                with the local East (positive anti-clockwise). 
                  
    pointing_center  
                [list or numpy array] coordinates of pointing center (in the same
                coordinate system as that of sky coordinates specified by
                skycoords). 2-element vector if skycoords='altaz'. 2- or 
                3-element vector if skycoords='dircos'. 

    power       [boolean] If set to True (default), compute power pattern, 
                otherwise compute field pattern

    Output:

    Electric field pattern or power pattern, number of rows equal to the number 
    of sky positions (which is equal to the number of rows in skypos), and number 
    of columns equal to the number of wavelengths. 
    -----------------------------------------------------------------------------
    """

    try:
        side, skypos, frequency
    except NameError:
        raise NameError('Square antenna side, skypos, frequency must be specified')

    if not isinstance(sides, (int,float)):
        raise TypeError('Antenna sides must be a scalar')
    sides = NP.asarray([side]*2, dtype=NP.float)

    ab = uniform_rectangular_aperture(sides, skypos, frequency,
                                      skyunits=skyunits, 
                                      east2ax1=east2ax1,
                                      pointing_center=pointing_center, 
                                      power=power)
    return ab
    
################################################################################
