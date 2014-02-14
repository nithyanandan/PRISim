import numpy as NP
import geometry as GEOM
import scipy.constants as FCNST
import pdb as PDB

#################################################################################

def primary_beam_generator(skypos, frequency, telescope='vla', freq_scale='GHz',
                           skyunits='degrees', east2ax1=0.0, phase_center=None):

    """
    -----------------------------------------------------------------------------
    A wrapper for estimating the power patterns of different telescopes such as
    the VLA, GMRT, MWA, etc. For the VLA and GMRT, polynomial power patterns are
    estimated as specified in AIPS task PBCOR. For MWA, it is based on
    theoretical expressions for dipole (element) pattern multiplied with the 
    array pattern of isotropic radiators.

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

    telescope   [scalar] String specifying the name of the telescope. Currently
                accepted values are 'vla', 'gmrt', 'mwa_dipole' and 'mwa'.
                Default = 'vla'. In case of 'mwa_dipole', the array layout of
                dipoles in a tile is ignored power pattern only due to the dipole
                is considered.

    freq_scale  [scalar] string specifying the units of frequency. Accepted
                values are 'GHz', 'MHz' and 'Hz'. Default = 'GHz'

    skyunits    [string] string specifying the coordinate system of the sky 
                positions. Accepted values are 'degrees', 'altaz', and 'dircos'.
                Default = 'degrees'. If 'dircos', the direction cosines are 
                aligned with the local East, North, and Up

    east2ax1    [scalar] Angle (in degrees) the primary axis of the array makes 
                with the local East (positive anti-clockwise). 

    phase_center
                [list or numpy array] coordinates of phase center (in the same
                coordinate system as that of sky coordinates specified by
                skyunits). 2-element vector if skyunits='altaz'. 2- or 3-element
                vector if skyunits='dircos'. Only used with phased array primary
                beams. 

    Output:

    [Numpy array] Power pattern at the specified sky positions. 
    -----------------------------------------------------------------------------
    """

    if (freq_scale == 'ghz') or (freq_scale == 'GHz'):
        frequency = frequency * 1.0e9
    elif (freq_scale == 'mhz') or (freq_scale == 'MHz'):
        frequency = frequency * 1.0e6
    elif (freq_scale == 'khz') or (freq_scale == 'kHz'):
        frequency = frequency * 1.0e3

    if (telescope == 'vla') or (telescope == 'gmrt'):
        if skyunits == 'altaz':
            angles = 90.0 - skypos[:,0]
        elif skyunits == 'dircos':
            angles = NP.arccos(NP.sqrt(1.0 - NP.sum(skypos[:,2]**2, axis=1)))
        elif skyunits == 'degrees':
            angles = skypos
        else:
            raise ValueError('skyunits must be "altaz", "dircos" or "degrees".')

        if telescope == 'vla':
            pb = VLA_primary_beam_PBCOR(angles, frequency/1e9, 'degrees')
        else:
            pb = GMRT_primary_beam(angles, frequency/1e9, 'degrees')
    elif telescope == 'mwa':
        if (skyunits == 'altaz') or (skyunits == 'dircos'):
            irap = isotropic_radiators_array_field_pattern(4, 4, 1.1, 1.1, skypos,
                                                           FCNST.c/frequency, east2ax1=east2ax1,
                                                           phase_center=phase_center,
                                                           skycoords=skyunits)
            dp = dipole_field_pattern(1.1, skypos, dipole_coords='dircos',
                                      dipole_orientation=NP.asarray([1.0,0.0,0.0]).reshape(1,-1),
                                      skycoords=skyunits, wavelength=FCNST.c/frequency)
            pb = NP.abs(dp * irap)**2 # Power pattern is square of the field pattern
        else:
            raise ValueError('skyunits must be in Alt-Az or direction cosine coordinates for MWA.')
    elif telescope == 'mwa_dipole':
        if (skyunits == 'altaz') or (skyunits == 'dircos'):
            dp = dipole_field_pattern(1.1, skypos, dipole_coords='dircos',
                                      dipole_orientation=NP.asarray([1.0,0.0,0.0]).reshape(1,-1),
                                      skycoords=skyunits, wavelength=FCNST.c/frequency)
            pb = NP.abs(dp)**2 # Power pattern is square of the field pattern
        else:
            raise ValueError('skyunits must be in Alt-Az or direction cosine coordinates for MWA dipole.')

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

def dipole_field_pattern(length, skypos, dipole_coords=None, skycoords=None, 
                         dipole_orientation=None, wavelength=1.0, angle_units=None, 
                         ground_plane=None, half_wave_dipole_approx=True):

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
                     compoenent will be determined appropriately.

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
                     direction cosines.

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

    ground_plane     [scalar] Height of the dipole element above the ground
                     plane (same units as dipole length and wavelength)

    half_wave_dipole_approx
                     [boolean] if True, indicates half-wave dipole approximation
                     is to be used. Otherwise, a more accurate expression is used
                     for the dipole pattern. Default=True

    Output:

    Dipole Electric field pattern, a numpy array with number of rows equal to the
    number of sky positions (which is equal to the number of rows in skypos) and
    number of columns equal to number of wavelengths specified. 
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

    if isinstance(wavelength, list):
        wavelength = NP.asarray(wavelength)
    elif isinstance(wavelength, (int, float)):
        wavelength = NP.asarray(wavelength).reshape(-1)
    elif not isinstance(wavelength, NP.ndarray):
        raise TypeError('Wavelength should be a scalar, list or numpy array.')
 
    if NP.any(wavelength <= 0.0):
        raise ValueError('Wavelength(s) should be positive.')

    if ground_plane is not None:
        if not isinstance(ground_plane, (int,float)):
            raise TypeError('Height above ground plane should be a scalar.')

        if ground_plane <= 0.0:
            raise ValueError('Height above ground plane should be positive.')

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
        
        skypos_dircos = GEOM.altaz2dircos(skypos, 'degrees')
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
        
        dipole_orientation_dircos = GEOM.altaz2dircos(dipole_orientation, 'degrees')
    else:
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
            
        dipole_orientation_dircos = dipole_orientation

    k = 2 * NP.pi / wavelength
    h = 0.5 * length
    dot_product = NP.dot(dipole_orientation_dircos, skypos_dircos.T)
    angles = NP.arccos(dot_product)

    eps = 1.e-10
    zero_angles_ind = NP.abs(NP.abs(dot_product) - 1.0) < eps
    n_zero_angles = NP.sum(zero_angles_ind)
    reasonable_angles_ind = NP.abs(NP.abs(dot_product) - 1.0) > eps

    # field_pattern = NP.empty_like(angles)

    max_pattern = 1.0 # Normalization factor
    if half_wave_dipole_approx:
        field_pattern = NP.cos(0.5 * NP.pi * NP.cos(angles)) / NP.sin(angles)
        field_pattern = NP.repeat(field_pattern.reshape(-1,1), wavelength.size, axis=1) # new stuff
    else:
        max_pattern = 1.0 - NP.cos(k * h) # Maximum occurs at angle = NP.pi / 2
        max_pattern = NP.repeat(max_pattern.reshape(1,-1), angles.size, axis=0)
        arg1 = NP.repeat(k.reshape(1,-1), angles.size, axis=0) * h 
        arg2 = NP.repeat(angles.reshape(-1,1), k.size, axis=1)
        field_pattern = (NP.cos(arg1*arg2) - NP.cos(arg1)) / NP.sin(arg2) # new stuff
        # field_pattern = (NP.cos(k * h * NP.cos(angles)) - NP.cos(k * h)) / NP.sin(angles) # old stuff

    # field_pattern[zero_angles_ind] = k * h * NP.tan(0.5 * angles[zero_angles_ind]) * NP.sin(0.5 * k * h * (1.0 + NP.cos(angles[zero_angles_ind]))) # old stuff L'Hospital rule
    if n_zero_angles > 0:
        field_pattern[zero_angles_ind, :] = NP.repeat(k.reshape(1,-1), n_zero_angles, axis=0) * h * NP.tan(0.5 * NP.repeat(angles[zero_angles_ind].reshape(-1,1), k.size, axis=1)) * NP.sin(0.5 * NP.repeat(k.reshape(1,-1), n_zero_angles, axis=0) * h * (1.0 + NP.cos(NP.repeat(angles[zero_angles_ind].reshape(-1,1), k.size, axis=1)))) # new stuff L'Hospital rule
    
    # # field_pattern[zero_angles_ind] = 0.0

    if ground_plane is not None: # Ground plane formulas to be verified. Use with caution
        skypos_altaz = GEOM.dircos2altaz(skypos_dircos, 'radians')
        # ground_pattern = 2 * NP.cos(k * ground_plane * NP.sin(skypos_altaz[:,0])) # old stuff
        ground_pattern = 2 * NP.cos(NP.repeat(k.reshape(1,-1), angles.size, axis=0) * ground_plane * NP.sin(NP.repeat(skypos_altaz[:,0].reshape(-1,1), k.size, axis=1))) # new stuff
    else:
        ground_pattern = 1.0

    return (field_pattern / max_pattern) * ground_pattern

#################################################################################

def isotropic_radiators_array_field_pattern(nax1, nax2, sep1, sep2=None,
                                            skypos=None, wavelength=1.0,
                                            east2ax1=None, skycoords='altaz',
                                            phase_center=None):

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
                  N = 1 (if skyunits = degrees, for azimuthally symmetric
                  telescopes such as VLA and GMRT which have parabolic dishes), 
                  N = 2 (if skyunits = altaz denoting Alt-Az coordinates), or 
                  N = 3 (if skyunits = dircos denoting direction cosine
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
                  
    phase_center  [list or numpy array] coordinates of phase center (in the same
                  coordinate system as that of sky coordinates specified by
                  skyunits). 2-element vector if skyunits='altaz'. 2- or 
                  3-element vector if skyunits='dircos'. Only used with phased 
                  array primary beams. 

    Output:

    Array Electric field pattern, number of rows equal to the number of sky 
    positions (which is equal to the number of rows in skypos), and number of 
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

    if phase_center is None:
        if skycoords == 'altaz':
            phase_center = NP.asarray([90.0, 0.0]) # Zenith in Alt-Az coordinates
        else:
            phase_center = NP.asarray([0.0, 0.0, 1.0]) # Zenith in direction-cosine coordinates
    else:
        if not isinstance(phase_center, (list, NP.ndarray)):
            raise TypeError('phase_center must be a list or numpy array')
        
        phase_center = NP.asarray(phase_center)
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
            if phase_center.size != 2:
                raise ValueError('phase_center must be a 2-element vector in Alt-Az coordinates.')
            else:
                phase_center = phase_center.ravel()

            if NP.any(phase_center[0] < 0.0) or NP.any(phase_center[0] > 90.0):
                raise ValueError('Altitudes in phase_center have to be positive and <= 90 degrees')
        else:
            if (phase_center.size < 2) or (phase_center.size > 3):
                raise ValueError('phase_center must be a 2- or 3-element vector in direction cosine coordinates')
            else:
                phase_center = phase_center.ravel()

            if phase_center.size == 2:
                if NP.sum(phase_center**2) > 1.0:
                    raise ValueError('phase_center in direction cosine coordinates are invalid.')
                
                phase_center = NP.hstack((phase_center, NP.sqrt(1.0-NP.sum(phase_center**2))))
            else:
                eps = 1.0e-10
                if (NP.abs(NP.sum(phase_center**2) - 1.0) > eps) or (phase_center[2] < 0.0):
                    if verbose:
                        print '\tWarning: phase_center in direction cosine coordinates along line of sight found to be negative or some direction cosines are not unit vectors. Resetting to correct values.'
                    phase_center[2] = NP.sqrt(1.0 - NP.sum(phase_center[:2]**2))

    # skypos_dircos_relative = NP.empty((skypos.shape[0],3))

    if east2ax1 is not None:
        if not isinstance(east2ax1, (int, float)):
            raise TypeError('east2ax1 must be a scalar value.')
        else:
            if skycoords == 'altaz':
                skypos_dircos_rotated = GEOM.altaz2dircos(NP.hstack((skypos[:,0].reshape(-1,1),NP.asarray(skypos[:,1]-east2ax1).reshape(-1,1))), 'degrees')
                phase_center_dircos_rotated = GEOM.altaz2dircos([phase_center[0], phase_center[1]-east2ax1], 'degrees')
            else:
                angle = NP.radians(east2ax1)
                rotation_matrix = NP.asarray([[NP.cos(angle), NP.sin(angle), 0.0],
                                              [-NP.sin(angle), NP.cos(angle),  0.0],
                                              [0.0,            0.0,           1.0]])
                skypos_dircos_rotated = NP.dot(skypos, rotation_matrix.T)
                phase_center_dircos_rotated = NP.dot(phase_center, rotation_matrix.T)

            skypos_dircos_relative = skypos_dircos_rotated - NP.repeat(phase_center_dircos_rotated.reshape(1,-1), skypos.shape[0], axis=0)
    else:
        if skycoords == 'altaz':
            skypos_dircos = GEOM.altaz2dircos(skypos, 'degrees')
            phase_center_dircos = GEOM.altaz2dircos([phase_center[0], phase_center[1]-east2ax1], 'degrees')
        else:
            skypos_dircos_rotated = skypos
        skypos_dircos_relative = skypos_dircos - NP.repeat(phase_center_dircos, skypos.shape[0], axis=0)

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
    return pb

#################################################################################

    
    


                
