import numpy as NP
import geometry as GEOM
import scipy.constants as FCNST
import ipdb as PDB
#################################################################################

def delay_envelope(bl, dircos, units='mks'):

    """
    ---------------------------------------------------------------------------
    Estimates the delay envelope determined by the sky horizon for given 
    baseline(s) for the phase centers specified by sky positions in direction
    cosines.

    Inputs:

    bl:     E, N, and U components of baseline vectors in a Mx3 numpy 
            array in local ENU coordinates

    dircos: Nx3 (direction cosines) numpy array of sky positions

    units:  'mks' or 'cgs' units. Default='mks'

    Outputs:
    
    delaymatrix: NxMx2 matrix. delaymatrix[:,:,0] contains the maximum delay if
                 there was no shift due to non-zenith phase center.
                 delaymatrix[:,:,1] contains the delay shift. To determine the 
                 minimum delay, use -delaymatrix[:,:,1]-delaymatrix[:,:,0]. To 
                 determine effective maximum delay, use
                 delaymatrix[:,:,0]-delaymatrix[:,:,1]. Minimum delay without
                 shift is -delaymatrix[:,:,0]

    ---------------------------------------------------------------------------
    """

    try:
        bl
    except NameError:
        raise NameError('No baseline(s) provided. Aborting delay_envelope().')

    try:
        dircos
    except NameError:
        print 'No sky position in direction cosine units provided. Assuming zenith for phase center in delay_envelope().'
        dircos = NP.zeros(3).reshape(1,3)

    try:
        units
    except NameError:
        print 'No units provided. Assuming MKS units.'
        units = 'mks'

    if (units != 'mks') and (units != 'cgs'):
        print 'Units should be specified to be one of MKS or CGS. Default=MKS'
        print 'Proceeding with MKS units.'
        units = 'mks'

    # Set the speed of light in MKS or CGS units
    if units == 'mks': c = FCNST.c
    elif units == 'cgs': c = FCNST.c * 1e2

    if len(bl.shape) == 1: bl = bl.reshape(1,len(bl))
    if len(dircos.shape) == 1: dircos = dircos.reshape(1,len(dircos))
    blshape = bl.shape
    dcshape = dircos.shape

    bl = bl[:,:min(blshape[1],dcshape[1])]
    dircos = dircos[:,:min(blshape[1],dcshape[1])]

    if blshape[1] > min(3,blshape[1],dcshape[1]):
        bl = bl[:,:min(3,blshape[1],dcshape[1])]
    if dcshape[1] > min(3,blshape[1],dcshape[1]):
        dircos = dircos[:,:min(3,blshape[1],dcshape[1])]

    blshape = bl.shape
    dcshape = dircos.shape

    eps = 1.0e-10
    if NP.any(NP.sqrt(NP.sum(dircos**2,axis=1)) > 1.0+eps):
        raise ValueError('Certain direction cosines exceed unit magnitude. Check inputs.')
    elif dcshape[1] == 3:
        if NP.any(NP.absolute(NP.sqrt(NP.sum(dircos**2,axis=1)) - 1.0) > eps):
            raise ValueError('Magnitude of vector of direction cosines have to equal unity. Check inputs.')
        # if NP.any(NP.sqrt(NP.sum(dircos**2,axis=1)) > 1.0+eps)):
        #     raise ValueError('Magnitude of vector of direction cosines have to equal unity. Check inputs.')
        if NP.any(dircos[:,2] < 0.0):
            raise ValueError('Direction cosines should lie on the upper hemisphere. Check inputs.')

    delaymatrix_max = NP.repeat(NP.sqrt(NP.sum(bl.T**2,axis=0)).reshape(1,blshape[0]), dcshape[0], axis=0)/c
    delaymatrix_shift = NP.dot(dircos, bl.T)/c

    delaymatrix = NP.dstack((delaymatrix_max, delaymatrix_shift))

    return delaymatrix

#################################################################################

def geometric_delay(baselines, skypos, altaz=False, dircos=False, hadec=True,
                    units='mks', latitude=None):

    """
    ---------------------------------------------------------------------
    Estimates the geometric delays matrix for different baselines from 
    different sky positions. 

    Inputs:

    baselines: x, y, and z components of baseline vectors in a Mx3 numpy 
               array

    skypos: Nx2 (Alt-Az or HA-Dec) or Nx3 (direction cosines) numpy array
            of sky positions

    altaz: [Boolean flag, default=False] If True, skypos is in Alt-Az 
           coordinates system

    hadec: [Boolean flag, default=True] If True, skypos is in HA-Dec 
           coordinates system

    dircos: [Boolean flag, default=False] If True, skypos is in direction 
           cosines coordinates system

    units: Units of baselines. Default='mks'. Alternative is 'cgs'.

    latitude: Latitude of the observatory. Required if hadec is True.

    Outputs:

    geometric delays [NxM numpy array] Geometric delay for every combination
                     of baselines and skypos.

    ---------------------------------------------------------------------
    """

    try:
        baselines, skypos
    except NameError:
        raise NameError('baselines and/or skypos not defined in geometric_delay().')

    if (altaz)+(dircos)+(hadec) != 1:
        raise ValueError('One and only one of altaz, dircos, hadec must be set to True.')

    if hadec and (latitude is None):
        raise ValueError('Latitude must be specified when skypos is in HA-Dec format.')

    try:
        units
    except NameError:
        print 'No units provided. Assuming MKS units.'
        units = 'mks'

    if (units != 'mks') and (units != 'cgs'):
        print 'Units should be specified to be one of MKS or CGS. Default=MKS'
        print 'Proceeding with MKS units.'
        units = 'mks'

    if not isinstance(baselines, NP.ndarray):
        raise TypeError('baselines should be a Nx3 numpy array in geometric_delay().')

    if len(baselines.shape) == 1:
        baselines = baselines.reshape(1,-1)

    if baselines.shape[1] == 1:
        baselines = NP.hstack(baselines, NP.zeros((baselines.size,2)))
    elif baselines.shape[1] == 2:
        baselines = NP.hstack(baselines, NP.zeros((baselines.size,1)))
    elif baselines.shape[1] > 3:
        baselines = baselines[:,:3]

    if altaz or hadec:
        if len(skypos.shape) < 2:
            if skypos.size != 2:
                raise ValueError('Sky position in altitude-azimuth or HA-Dec should consist of 2 elements.')
            else:
                skypos = skypos.reshape(1,-1)
        elif len(skypos.shape) > 2:
            raise ValueError('Sky positions should be a Nx2 numpy array if using altitude-azimuth of HA-Dec.')
        else:
            if skypos.shape[1] != 2:
                raise ValueError('Sky positions should be a Nx2 numpy array if using altitude-azimuth of HA-Dec.')

        if altaz:
            dc = GEOM.altaz2dircos(skypos, 'degrees')
        else:
            dc = GEOM.altaz2dircos(GEOM.hadec2altaz(skypos, latitude, 'degrees'), 'degrees')
    else:
        if len(skypos.shape) < 2:
            if skypos.size != 3:
                raise ValueError('Sky position in direction cosines should consist of 3 elements.')
            else:
                skypos = skypos.reshape(1,-1)
        elif len(skypos.shape) > 2:
            raise ValueError('Sky positions should be a Nx3 numpy array if using direction cosines.')
        else:
            if skypos.shape[1] != 3:
                raise ValueError('Sky positions should be a Nx3 numpy array if using direction cosines.')
        
        dc = skypos

    # Set the speed of light in MKS or CGS units
    if units == 'mks': c = FCNST.c
    elif units == 'cgs': c = FCNST.c * 1e2

    # geometric_delays = delay_envelope(baselines, dc, units)[:,:,-1]
    geometric_delays = NP.dot(dc, baselines.T)/c
    return geometric_delays

##########################################################################
