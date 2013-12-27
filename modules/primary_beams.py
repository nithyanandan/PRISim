import numpy as NP
import geometry as GEOM
import scipy.constants as FCNST

##########################################################################

def primary_beam_generator(skypos, frequency, skyunits='degrees',
                           telescope='vla', freq_scale='GHz', east2ax1=0.0):

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
                                                           skycoords=skyunits)
            dp = dipole_field_pattern(1.1, skypos, dipole_coords='dircos',
                                      dipole_orientation=NP.asarray([1.0,0.0,0.0]).reshape(1,-1),
                                      skycoords=skyunits, wavelength=FCNST.c/frequency)
            pb = dp * irap
        else:
            raise ValueError('skyunits must be in Alt-Az or direction cosine coordinates for MWA.')

    return pb
    
##########################################################################

def VLA_primary_beam_PBCOR(skypos, frequency, skyunits='degrees'):

    try:
        skypos, frequency
    except NameError:
        raise NameError('skypos and frequency are required in VLA_primary_beams().')

    freq_ref = NP.asarray([0.0738, 0.3275, 1.465, 4.885, 8.435, 14.965, 22.485, 43.315]).reshape(-1,1)
    parms_ref = NP.asarray([[-0.897,   2.71,  -0.242], 
                            [-0.935,   3.23,  -0.378], 
                            [-1.343,  6.579,  -1.186], 
                            [-1.372,  6.940,  -1.309], 
                            [-1.306,  6.253,  -1.100], 
                            [-1.305,  6.155,  -1.030], 
                            [-1.417,  7.332,  -1.352], 
                            [-1.321,  6.185,  -0.983]])
    
    idx = NP.argmin(NP.abs(freq_ref - frequency)) # Index of closest value

    skypos = NP.asarray(skypos)

    if skyunits == 'degrees':
        x = (skypos*60.0*frequency)**2
    elif skyunits == 'altaz':
        x = ((90.0-skypos[:,0])*60.0*frequency)**2
    elif skyunits == 'dircos':
        x = (NP.degrees(NP.arccos(skypos[:,-1]))*60.0*frequency)**2
    else:
        raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in VLA_primary_beam().')

    pb = 1.0 + parms_ref[idx,0]*x/1e3 + parms_ref[idx,1]*(x**2)/1e7 + \
         parms_ref[idx,2]*(x**3)/1e10

    return pb


##########################################################################

def GMRT_primary_beam(skypos, frequency, skyunits='degrees'):

    try:
        skypos, frequency
    except NameError:
        raise NameError('skypos and frequency are required in VLA_primary_beams().')

    freq_ref = NP.asarray([0.235, 0.325, 0.610, 1.420]).reshape(-1,1)
    parms_ref = NP.asarray([[-3.366  , 46.159 , -29.963 ,  7.529  ], 
                            [-3.397  , 47.192 , -30.931 ,  7.803  ], 
                            [-3.486  , 47.749 , -35.203 , 10.399  ], 
                            [-2.27961, 21.4611,  -9.7929,  1.80153]])
    
    idx = NP.argmin(NP.abs(freq_ref - frequency)) # Index of closest value

    skypos = NP.asarray(skypos)

    if skyunits == 'degrees':
        x = (skypos*60.0*frequency)**2
    elif skyunits == 'altaz':
        x = ((90.0-skypos[:,0])*60.0*frequency)**2
    elif skyunits == 'dircos':
        x = (NP.degrees(NP.arccos(skypos[:,-1]))*60.0*frequency)**2
    else:
        raise ValueError('skyunits must be "degrees", "altaz" or "dircos" in VLA_primary_beam().')

    pb = 1.0 + parms_ref[idx,0]*x/1e3 + parms_ref[idx,1]*(x**2)/1e7 + \
         parms_ref[idx,2]*(x**3)/1e10 + parms_ref[idx,3]*(x**4)/1e13

    return pb

##########################################################################

def dipole_field_pattern(length, skypos, dipole_coords=None, dipole_orientation=None,
                         skycoords=None, wavelength=1.0, ground_plane=None,
                         angle_units=None, half_wave_dipole_approx=True):

    try:
        length, skypos
    except NameError:
        raise NameError('Dipole length and sky positions must be specified. Check inputs.')

    if not isinstance(length, (int,float)):
        raise TypeError('Dipole length should be a scalar.')

    if length <= 0.0:
        raise ValueError('Dipole length should be positive.')

    if not isinstance(wavelength, (int,float)):
        raise TypeError('Wavelength should be a scalar.')
 
    if wavelength <= 0.0:
        raise ValueError('Wavelength should be positive.')

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
    reasonable_angles_ind = NP.abs(NP.abs(dot_product) - 1.0) > eps

    # field_pattern = NP.empty_like(angles)

    max_pattern = 1.0 # Normalization factor
    if half_wave_dipole_approx:
        field_pattern = NP.cos(0.5 * NP.pi * NP.cos(angles)) / NP.sin(angles)
    else:
        max_pattern = 1.0 - NP.cos(k * h) # Maximum occurs at angles = NP.pi / 2
        field_pattern = (NP.cos(k * h * NP.cos(angles)) - NP.cos(k * h)) / NP.sin(angles)

    field_pattern[zero_angles_ind] = k * h * NP.tan(0.5 * angles[zero_angles_ind]) * NP.sin(0.5 * k * h * (1.0 + NP.cos(angles[zero_angles_ind])))
    # field_pattern[zero_angles_ind] = 0.0

    if ground_plane is not None: # Ground plane formulas to be verified. Use with caution
        skypos_altaz = GEOM.dircos2altaz(skypos_dircos, 'radians')
        ground_pattern = 2 * NP.cos(k * ground_plane * NP.sin(skypos_altaz[:,0]))
    else:
        ground_pattern = 1.0

    return (field_pattern / max_pattern) * ground_pattern

##########################################################################

def isotropic_radiators_array_field_pattern(nax1, nax2, sep1, sep2=None,
                                            skypos=None, wavelength=1.0,
                                            east2ax1=None, skycoords='altaz'):
    try:
        nax1, nax2, sep1
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

    if not isinstance(wavelength, (int,float)):
        raise TypeError('wavelength must be a positive scalar.')
    elif wavelength <= 0:
        raise ValueError('wavelength must be a positive value.')

    if not isinstance(east2ax1, (int,float)):
        raise TypeError('east2ax1 must be a scalar.')

    if not isinstance(skypos, NP.ndarray):
        raise TypeError('skypos must be a Numpy array.')
    
    if skycoords is not None:
        if (skycoords != 'altaz') and (skycoords != 'dircos'):
            raise ValueError('skycoords must be "altaz" or "dircos" or None (default).')
        elif skycoords == 'altaz':
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
    else:
        raise ValueError('skycoords has not been set.')

    skypos_dircos_relative = NP.empty((skypos.shape[0],3))
    if east2ax1 is not None:
        if not isinstance(east2ax1, (int, float)):
            raise TypeError('east2ax1 must be a scalar value.')
        else:
            if skycoords == 'altaz':
                skypos_dircos_relative = GEOM.altaz2dircos(NP.hstack((skypos[:,0].reshape(-1,1),NP.asarray(skypos[:,1]-east2ax1).reshape(-1,1))), 'degrees')
            else:
                angle = NP.radians(east2ax1)
                rotation_matrix = NP.asarray([[NP.cos(angle),  NP.sin(angle), 0.0],
                                              [-NP.sin(angle), NP.cos(angle), 0.0],
                                              [0.0,            0.0,           1.0]])
                skypos_dircos_relative = NP.dot(skypos, rotation_matrix.T)
    else:
        if skycoords == 'altaz':
            skypos_dircos_relative = GEOM.altaz2dircos(skypos, 'degrees')
        else:
            skypos_dircos_relative = skypos

    phi = 2 * NP.pi * sep1 * skypos_dircos_relative[:,0] / wavelength 
    psi = 2 * NP.pi * sep2 * skypos_dircos_relative[:,1] / wavelength 

    pb = NP.sin(0.5*nax1*phi)/NP.sin(0.5*phi) * NP.sin(0.5*nax2*psi)/NP.sin(0.5*psi) / (nax1*nax2)
    return pb


    
    


                
