from __future__ import division
import glob
import numpy as NP
from functools import reduce
import numpy.ma as MA
import progressbar as PGB
import h5py
import healpy as HP
import warnings
import copy
import astropy.cosmology as CP
from astropy.time import Time
from astropy.io import fits
from astropy import units as U
from astropy import constants as FCNST
from scipy import interpolate
from astroutils import DSP_modules as DSP
from astroutils import constants as CNST
from astroutils import nonmathops as NMO
from astroutils import mathops as OPS
from astroutils import lookup_operations as LKP
import prisim
from prisim import interferometry as RI
from prisim import delay_spectrum as DS
try:
    from pyuvdata import UVBeam
except ImportError:
    uvbeam_module_found = False
else:
    uvbeam_module_found = True

prisim_path = prisim.__path__[0]+'/'

cosmoPlanck15 = CP.Planck15 # Planck 2015 cosmology
cosmo100 = cosmoPlanck15.clone(name='Modified Planck 2015 cosmology with h=1.0', H0=100.0) # Modified Planck 2015 cosmology with h=1.0, H= 100 km/s/Mpc

################################################################################

def write_PRISim_bispectrum_phase_to_npz(infile_prefix, outfile_prefix,
                                         triads=None, bltriplet=None,
                                         hdf5file_prefix=None, infmt='npz',
                                         datakey='noisy', blltol=0.1):

    """
    ----------------------------------------------------------------------------
    Write closure phases computed in a PRISim simulation to a NPZ file with
    appropriate format for further analysis.

    Inputs:

    infile_prefix 
                [string] HDF5 file or NPZ file created by a PRISim simulation or
                its replication respectively. If infmt is specified as 'hdf5', 
                then hdf5file_prefix will be ignored and all the observing

                info will be read from here. If infmt is specified as 'npz', 
                then hdf5file_prefix needs to be specified in order to read the
                observing parameters.

    triads      [list or numpy array or None] Antenna triads given as a list of 
                3-element lists or a ntriads x 3 array. Each element in the
                inner list is an antenna label. They will be converted to
                strings internally. If set to None, then all triads determined
                by bltriplet will be used. If specified, then inputs in blltol
                and bltriplet will be ignored. 

    bltriplet   [numpy array or None] 3x3 numpy array containing the 3 baseline 
                vectors. The first axis denotes the three baselines, the second 
                axis denotes the East, North, Up coordinates of the baseline 
                vector. Units are in m. Will be used only if triads is set to 
                None.

    outfile_prefix
                [string] Prefix of the NPZ file. It will be appended by 
                '_noiseless', '_noisy', and '_noise' and further by extension 
                '.npz'

    infmt       [string] Format of the input file containing visibilities. 
                Accepted values are 'npz' (default), and 'hdf5'. If infmt is
                specified as 'npz', then hdf5file_prefix also needs to be 
                specified for reading the observing parameters

    datakey     [string] Specifies which -- 'noiseless', 'noisy' (default), or
                'noise' -- visibilities are to be written to the output. If set
                to None, and infmt is 'hdf5', then all three sets of 
                visibilities are written. The datakey string will also be added
                as a suffix in the output file. 

    blltol      [scalar] Baseline length tolerance (in m) for matching baseline
                vectors in triads. It must be a scalar. Default = 0.1 m. Will
                be used only if triads is set to None and bltriplet is to be
                used.
    ----------------------------------------------------------------------------
    """

    if not isinstance(infile_prefix, str):
        raise TypeError('Input infile_prefix must be a string')
    if not isinstance(outfile_prefix, str):
        raise TypeError('Input outfile_prefix must be a string')
    if (triads is None) and (bltriplet is None):
        raise ValueError('One of triads or bltriplet must be set')
    if triads is None:
        if not isinstance(bltriplet, NP.ndarray):
            raise TypeError('Input bltriplet must be a numpy array')
        if not isinstance(blltol, (int,float)):
            raise TypeError('Input blltol must be a scalar')
        if bltriplet.ndim != 2:
            raise ValueError('Input bltriplet must be a 2D numpy array')
        if bltriplet.shape[0] != 3:
            raise ValueError('Input bltriplet must contain three baseline vectors')
        if bltriplet.shape[1] != 3:
            raise ValueError('Input bltriplet must contain baseline vectors along three corrdinates in the ENU frame')
    else:
        if not isinstance(triads, (list, NP.ndarray)):
            raise TypeError('Input triads must be a list or numpy array')
        triads = NP.asarray(triads).astype(str)

    if not isinstance(infmt, str):
        raise TypeError('Input infmt must be a string')
    if infmt.lower() not in ['npz', 'hdf5']:
        raise ValueError('Input file format must be npz or hdf5')
    if infmt.lower() == 'npz':
        if not isinstance(hdf5file_prefix, str):
            raise TypeError('If infmt is npz, then hdf5file_prefix needs to be specified for observing parameters information')
        if datakey is None:
            datakey = ['noisy']

    if isinstance(datakey, str):
        datakey = [datakey]
    elif not isinstance(datakey, list):
        raise TypeError('Input datakey must be a list')
    for dkey in datakey:
        if dkey.lower() not in ['noiseless', 'noisy', 'noise']:
            raise ValueError('Invalid input found in datakey')

    if infmt.lower() == 'hdf5':
        fullfnames_with_extension = glob.glob(infile_prefix + '*' + infmt.lower())
        fullfnames_without_extension = [fname.split('.hdf5')[0] for fname in fullfnames_with_extension]
    else:
        fullfnames_without_extension = [infile_prefix]
    if len(fullfnames_without_extension) == 0:
        raise IOError('No input files found with pattern {0}'.format(infile_prefix))

    try:
        if infmt.lower() == 'hdf5':
            simvis = RI.InterferometerArray(None, None, None, init_file=fullfnames_without_extension[0])
        else:
            simvis = RI.InterferometerArray(None, None, None, init_file=hdf5file_prefix)
    except:
        raise IOError('Input PRISim file does not contain a valid PRISim output')

    latitude = simvis.latitude
    longitude = simvis.longitude
    location = ('{0:.5f}d'.format(longitude), '{0:.5f}d'.format(latitude))
    last = simvis.lst / 15.0 / 24.0 # from degrees to fraction of day
    last = last.reshape(-1,1)
    daydata = NP.asarray(simvis.timestamp[0]).ravel()

    if infmt.lower() == 'npz':
        simvisinfo = NP.load(fullfnames_without_extension[0]+'.'+infmt.lower())
        skyvis = simvisinfo['noiseless'][0,...]
        vis = simvisinfo['noisy']
        noise = simvisinfo['noise']
        n_realize = vis.shape[0]
    else:
        n_realize = len(fullfnames_without_extension)

    cpdata = {}
    outfile = {}
    for fileind in range(n_realize):
        if infmt.lower() == 'npz':
            simvis.vis_freq = vis[fileind,...]
            simvis.vis_noise_freq = noise[fileind,...]
        else:
            simvis = RI.InterferometerArray(None, None, None, init_file=fullfnames_without_extension[fileind])
        if fileind == 0:
            if triads is None:
                triads, bltriplets = simvis.getThreePointCombinations(unique=False)
                # triads = NP.asarray(prisim_BSP_info['antenna_triplets']).reshape(-1,3)
                # bltriplets = NP.asarray(prisim_BSP_info['baseline_triplets'])
                triads = NP.asarray(triads).reshape(-1,3)
                bltriplets = NP.asarray(bltriplets)
            
                blinds = []
                matchinfo = LKP.find_NN(bltriplet, bltriplets.reshape(-1,3), distance_ULIM=blltol)
                revind = []
                for blnum in NP.arange(bltriplet.shape[0]):
                    if len(matchinfo[0][blnum]) == 0:
                        revind += [blnum]
                if len(revind) > 0:
                    flip_factor = NP.ones(3, dtype=NP.float)
                    flip_factor[NP.array(revind)] = -1
                    rev_bltriplet = bltriplet * flip_factor.reshape(-1,1)
                    matchinfo = LKP.find_NN(rev_bltriplet, bltriplets.reshape(-1,3), distance_ULIM=blltol)
                    for blnum in NP.arange(bltriplet.shape[0]):
                        if len(matchinfo[0][blnum]) == 0:
                            raise ValueError('Some baselines in the triplet are not found in the model triads')
                triadinds = []
                for blnum in NP.arange(bltriplet.shape[0]):
                    triadind, blind = NP.unravel_index(NP.asarray(matchinfo[0][blnum]), (bltriplets.shape[0], bltriplets.shape[1]))
                    triadinds += [triadind]
                
                triadind_intersection = NP.intersect1d(triadinds[0], NP.intersect1d(triadinds[1], triadinds[2]))
                if triadind_intersection.size == 0:
                    raise ValueError('Specified triad not found in the PRISim model. Try other permutations of the baseline vectors and/or reverse individual baseline vectors in the triad before giving up.')
            
                triads = triads[triadind_intersection,:]
                selected_bltriplets = bltriplets[triadind_intersection,:,:].reshape(-1,3,3)

        prisim_BSP_info = simvis.getClosurePhase(antenna_triplets=triads.tolist(),
                                                 delay_filter_info=None,
                                                 specsmooth_info=None,
                                                 spectral_window_info=None,
                                                 unique=False)
        if fileind == 0:
            triads = NP.asarray(prisim_BSP_info['antenna_triplets']).reshape(-1,3)  # Re-establish the triads returned after the first iteration (to accunt for any order flips)

        for outkey in datakey:
            if fileind == 0:
                outfile[outkey] = outfile_prefix + '_{0}.npz'.format(outkey)
            if outkey == 'noiseless':
                if fileind == 0:
                    # cpdata = prisim_BSP_info['closure_phase_skyvis'][triadind_intersection,:,:][NP.newaxis,...]
                    cpdata[outkey] = prisim_BSP_info['closure_phase_skyvis'][NP.newaxis,...]
                else:
                    # cpdata = NP.concatenate((cpdata, prisim_BSP_info['closure_phase_skyvis'][triadind_intersection,:,:][NP.newaxis,...]), axis=0)
                    cpdata[outkey] = NP.concatenate((cpdata[outkey], prisim_BSP_info['closure_phase_skyvis'][NP.newaxis,...]), axis=0)
            if outkey == 'noisy':
                if fileind == 0:
                    # cpdata = prisim_BSP_info['closure_phase_vis'][triadind_intersection,:,:][NP.newaxis,...]
                    cpdata[outkey] = prisim_BSP_info['closure_phase_vis'][NP.newaxis,...]
                else:
                    # cpdata = NP.concatenate((cpdata, prisim_BSP_info['closure_phase_vis'][triadind_intersection,:,:][NP.newaxis,...]), axis=0)
                    cpdata[outkey] = NP.concatenate((cpdata[outkey], prisim_BSP_info['closure_phase_vis'][NP.newaxis,...]), axis=0)
            if outkey == 'noise':
                if fileind == 0:
                    # cpdata = prisim_BSP_info['closure_phase_noise'][triadind_intersection,:,:]
                    cpdata[outkey] = prisim_BSP_info['closure_phase_noise'][NP.newaxis,:,:]
                else:
                    # cpdata = NP.concatenate((cpdata, prisim_BSP_info['closure_phase_noise'][triadind_intersection,:,:][NP.newaxis,...]), axis=0)
                    cpdata[outkey] = NP.concatenate((cpdata[outkey], prisim_BSP_info['closure_phase_noise'][NP.newaxis,...]), axis=0)
    for outkey in datakey:
        cpdata[outkey] = NP.rollaxis(cpdata[outkey], 3, start=0)
        flagsdata = NP.zeros(cpdata[outkey].shape, dtype=NP.bool)
        NP.savez_compressed(outfile[outkey], closures=cpdata[outkey],
                            flags=flagsdata, triads=triads,
                            last=last+NP.zeros((1,n_realize)),
                            days=daydata+NP.arange(n_realize))

################################################################################

def loadnpz(npzfile, longitude=0.0, latitude=0.0):

    """
    ----------------------------------------------------------------------------
    Read an input NPZ file containing closure phase data output from CASA and
    return a dictionary

    Inputs:

    npzfile     [string] Input NPZ file including full path containing closure 
                phase data. It must have the following files/keys inside:
                'closures'  [numpy array] Closure phase (radians). It is of 
                            shape (nlst,ndays,ntriads,nchan)
                'triads'    [numpy array] Array of triad tuples, of shape 
                            (ntriads,3)
                'flags'     [numpy array] Array of flags (boolean), of shape
                            (nlst,ndays,ntriads,nchan)
                'last'      [numpy array] Array of LST for each day (CASA units 
                            which is MJD+6713). Shape is (nlst,ndays)
                'days'      [numpy array] Array of days, shape is (ndays,)
                'averaged_closures'
                            [numpy array] optional array of closure phases
                            averaged across days. Shape is (nlst,ntriads,nchan)
                'std_dev_lst'
                            [numpy array] optional array of standard deviation
                            of closure phases across days. Shape is 
                            (nlst,ntriads,nchan)
                'std_dev_triads'
                            [numpy array] optional array of standard deviation
                            of closure phases across triads. Shape is 
                            (nlst,ndays,nchan)

    latitude    [scalar int or float] Latitude of site (in degrees). 
                Default=0.0 deg. 

    longitude   [scalar int or float] Longitude of site (in degrees). 
                Default=0.0 deg.

    Output:

    cpinfo          [dictionary] Contains one top level keys, namely, 'raw' 

                    Under key 'raw' which holds a dictionary, the subkeys 
                    include 'cphase' (nlst,ndays,ntriads,nchan), 
                    'triads' (ntriads,3), 'lst' (nlst,ndays), and 'flags' 
                    (nlst,ndays,ntriads,nchan), and some other optional keys
    ----------------------------------------------------------------------------
    """

    npzdata = NP.load(npzfile)
    cpdata = npzdata['closures']
    triadsdata = npzdata['triads']
    flagsdata = npzdata['flags']
    location = ('{0:.5f}d'.format(longitude), '{0:.5f}d'.format(latitude))
    # lstdata = Time(npzdata['last'].astype(NP.float64) - 6713.0, scale='utc', format='mjd', location=('+21.4278d', '-30.7224d')).sidereal_time('apparent') # Subtract 6713 based on CASA convention to obtain MJD
    lstfrac, lstint = NP.modf(npzdata['last'])
    lstday = Time(lstint.astype(NP.float64) - 6713.0, scale='utc', format='mjd', location=location) # Subtract 6713 based on CASA convention to obtain MJD
    lstHA = lstfrac * 24.0 # in hours
    daydata = Time(npzdata['days'].astype(NP.float64), scale='utc', format='jd', location=location)

    cp = cpdata.astype(NP.float64)
    flags = flagsdata.astype(NP.bool)

    cpinfo = {}
    datapool = ['raw']
    for dpool in datapool:
        cpinfo[dpool] = {}
        if dpool == 'raw':
            qtys = ['cphase', 'triads', 'flags', 'lst', 'lst-day', 'days', 'dayavg', 'std_triads', 'std_lst']
        for qty in qtys:
            if qty == 'cphase':
                cpinfo[dpool][qty] = NP.copy(cp)
            elif qty == 'triads':
                cpinfo[dpool][qty] = NP.copy(triadsdata)
            elif qty == 'flags':
                cpinfo[dpool][qty] = NP.copy(flags)
            elif qty == 'lst':
                cpinfo[dpool][qty] = NP.copy(lstHA)
            elif qty == 'lst-day':
                cpinfo[dpool][qty] = NP.copy(lstday.value)
            elif qty == 'days':
                cpinfo[dpool][qty] = NP.copy(daydata.jd)
            elif qty == 'dayavg':
                if 'averaged_closures' in npzdata:
                    cpinfo[dpool][qty] = NP.copy(cp_dayavg)
            elif qty == 'std_triads':
                if 'std_dev_triad' in npzdata:
                    cpinfo[dpool][qty] = NP.copy(cp_std_triads)
            elif qty == 'std_lst':
                if 'std_dev_lst' in npzdata:
                    cpinfo[dpool][qty] = NP.copy(cp_std_lst)
    
    return cpinfo

################################################################################

def npz2hdf5(npzfile, hdf5file, longitude=0.0, latitude=0.0):

    """
    ----------------------------------------------------------------------------
    Read an input NPZ file containing closure phase data output from CASA and
    save it to HDF5 format

    Inputs:

    npzfile     [string] Input NPZ file including full path containing closure 
                phase data. It must have the following files/keys inside:
                'closures'  [numpy array] Closure phase (radians). It is of 
                            shape (nlst,ndays,ntriads,nchan)
                'triads'    [numpy array] Array of triad tuples, of shape 
                            (ntriads,3)
                'flags'     [numpy array] Array of flags (boolean), of shape
                            (nlst,ndays,ntriads,nchan)
                'last'      [numpy array] Array of LST for each day (CASA units 
                            ehich is MJD+6713). Shape is (nlst,ndays)
                'days'      [numpy array] Array of days, shape is (ndays,)
                'averaged_closures'
                            [numpy array] optional array of closure phases
                            averaged across days. Shape is (nlst,ntriads,nchan)
                'std_dev_lst'
                            [numpy array] optional array of standard deviation
                            of closure phases across days. Shape is 
                            (nlst,ntriads,nchan)
                'std_dev_triads'
                            [numpy array] optional array of standard deviation
                            of closure phases across triads. Shape is 
                            (nlst,ndays,nchan)

    hdf5file    [string] Output HDF5 file including full path.

    latitude    [scalar int or float] Latitude of site (in degrees). 
                Default=0.0 deg. 

    longitude   [scalar int or float] Longitude of site (in degrees). 
                Default=0.0 deg.
    ----------------------------------------------------------------------------
    """

    npzdata = NP.load(npzfile)
    cpdata = npzdata['closures']
    triadsdata = npzdata['triads']
    flagsdata = npzdata['flags']
    location = ('{0:.5f}d'.format(longitude), '{0:.5f}d'.format(latitude))
    # lstdata = Time(npzdata['last'].astype(NP.float64) - 6713.0, scale='utc', format='mjd', location=('+21.4278d', '-30.7224d')).sidereal_time('apparent') # Subtract 6713 based on CASA convention to obtain MJD
    lstfrac, lstint = NP.modf(npzdata['last'])
    lstday = Time(lstint.astype(NP.float64) - 6713.0, scale='utc', format='mjd', location=location) # Subtract 6713 based on CASA convention to obtain MJD
    lstHA = lstfrac * 24.0 # in hours
    daydata = Time(npzdata['days'].astype(NP.float64), scale='utc', format='jd', location=location)

    cp = cpdata.astype(NP.float64)
    flags = flagsdata.astype(NP.bool)
    if 'averaged_closures' in npzdata:
        day_avg_cpdata = npzdata['averaged_closures']
        cp_dayavg = day_avg_cpdata.astype(NP.float64)
    if 'std_dev_triad' in npzdata:
        std_triads_cpdata = npzdata['std_dev_triad']
        cp_std_triads = std_triads_cpdata.astype(NP.float64)
    if 'std_dev_lst' in npzdata:
        std_lst_cpdata = npzdata['std_dev_lst']
        cp_std_lst = std_lst_cpdata.astype(NP.float64)

    with h5py.File(hdf5file, 'w') as fobj:
        datapool = ['raw']
        for dpool in datapool:
            if dpool == 'raw':
                qtys = ['cphase', 'triads', 'flags', 'lst', 'lst-day', 'days', 'dayavg', 'std_triads', 'std_lst']
            for qty in qtys:
                data = None
                if qty == 'cphase':
                    data = NP.copy(cp)
                elif qty == 'triads':
                    data = NP.copy(triadsdata)
                elif qty == 'flags':
                    data = NP.copy(flags)
                elif qty == 'lst':
                    data = NP.copy(lstHA)
                elif qty == 'lst-day':
                    data = NP.copy(lstday.value)
                elif qty == 'days':
                    data = NP.copy(daydata.jd)
                elif qty == 'dayavg':
                    if 'averaged_closures' in npzdata:
                        data = NP.copy(cp_dayavg)
                elif qty == 'std_triads':
                    if 'std_dev_triad' in npzdata:
                        data = NP.copy(cp_std_triads)
                elif qty == 'std_lst':
                    if 'std_dev_lst' in npzdata:
                        data = NP.copy(cp_std_lst)
                if data is not None:
                    dset = fobj.create_dataset('{0}/{1}'.format(dpool, qty), data=data, compression='gzip', compression_opts=9)
            
################################################################################

def save_CPhase_cross_power_spectrum(xcpdps, outfile):

    """
    ----------------------------------------------------------------------------
    Save cross-power spectrum information in a dictionary to a HDF5 file

    Inputs:

    xcpdps      [dictionary] This dictionary is essentially an output of the 
                member function compute_power_spectrum() of class 
                ClosurePhaseDelaySpectrum. It has the following key-value 
                structure:
                'triads' ((ntriads,3) array), 'triads_ind', 
                ((ntriads,) array), 'lstXoffsets' ((ndlst_range,) array), 'lst' 
                ((nlst,) array), 'dlst' ((nlst,) array), 'lst_ind' ((nlst,) 
                array), 'days' ((ndays,) array), 'day_ind' ((ndays,) array), 
                'dday' ((ndays,) array), 'oversampled' and 'resampled' 
                corresponding to whether resample was set to False or True in 
                call to member function FT(). Values under keys 'triads_ind' 
                and 'lst_ind' are numpy array corresponding to triad and time 
                indices used in selecting the data. Values under keys 
                'oversampled' and 'resampled' each contain a dictionary with 
                the following keys and values:
                'z'     [numpy array] Redshifts corresponding to the band 
                        centers in 'freq_center'. It has shape=(nspw,)
                'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,)
                'kprll' [numpy array] k_parallel modes (in h/Mpc) corresponding 
                        to 'lags'. It has shape=(nspw,nlags)
                'freq_center'   
                        [numpy array] contains the center frequencies (in Hz) 
                        of the frequency subbands of the subband delay spectra. 
                        It is of size n_win. It is roughly equivalent to 
                        redshift(s)
                'freq_wts'      
                        [numpy array] Contains frequency weights applied on 
                        each frequency sub-band during the subband delay 
                        transform. It is of size n_win x nchan. 
                'bw_eff'        
                        [numpy array] contains the effective bandwidths (in Hz) 
                        of the subbands being delay transformed. It is of size 
                        n_win. It is roughly equivalent to width in redshift or 
                        along line-of-sight
                'shape' [string] shape of the frequency window function applied. 
                        Usual values are 'rect' (rectangular), 'bhw' 
                        (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'fftpow'
                        [scalar] the power to which the FFT of the window was 
                        raised. The value is be a positive scalar with 
                        default = 1.0
                'lag_corr_length' 
                        [numpy array] It is the correlation timescale (in 
                        pixels) of the subband delay spectra. It is proportional 
                        to inverse of effective bandwidth. It is of size n_win. 
                        The unit size of a pixel is determined by the difference
                        between adjacent pixels in lags under key 'lags' which 
                        in turn is effectively inverse of the effective 
                        bandwidth of the subband specified in bw_eff
                
                It further contains one or more of the following keys named 
                'whole', 'submodel', 'residual', and 'errinfo' each of which is 
                a dictionary. 'whole' contains power spectrum info about the 
                input closure phases. 'submodel' contains power spectrum info 
                about the model that will have been subtracted (as closure 
                phase) from the 'whole' model. 'residual' contains power 
                spectrum info about the closure phases obtained as a difference 
                between 'whole' and 'submodel'. It contains the following keys 
                and values:
                'mean'  [numpy array] Delay power spectrum incoherently 
                        estimated over the axes specified in xinfo['axes'] 
                        using the 'mean' key in input cpds or attribute 
                        cPhaseDS['processed']['dspec']. It has shape that 
                        depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are 
                        not set, those axes will be replaced with square 
                        covariance matrices. If collapse_axes is provided but 
                        avgcov is False, those axes will be of shape 2*Naxis-1. 
                'median'
                        [numpy array] Delay power spectrum incoherently averaged 
                        over the axes specified in incohax using the 'median' 
                        key in input cpds or attribute 
                        cPhaseDS['processed']['dspec']. It has shape that 
                        depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are not 
                        set, those axes will be replaced with square covariance 
                        matrices. If collapse_axes is provided bu avgcov is 
                        False, those axes will be of shape 2*Naxis-1. 
                'diagoffsets' 
                        [dictionary] Same keys corresponding to keys under 
                        'collapse_axes' in input containing the diagonal 
                        offsets for those axes. If 'avgcov' was set, those 
                        entries will be removed from 'diagoffsets' since all the 
                        leading diagonal elements have been collapsed (averaged) 
                        further. Value under each key is a numpy array where 
                        each element in the array corresponds to the index of 
                        that leading diagonal. This should match the size of the 
                        output along that axis in 'mean' or 'median' above. 
                'diagweights'
                        [dictionary] Each key is an axis specified in 
                        collapse_axes and the value is a numpy array of weights 
                        corresponding to the diagonal offsets in that axis.
                'axesmap'
                        [dictionary] If covariance in cross-power is calculated 
                        but is not collapsed, the number of dimensions in the 
                        output will have changed. This parameter tracks where 
                        the original axis is now placed. The keys are the 
                        original axes that are involved in incoherent 
                        cross-power, and the values are the new locations of 
                        those original axes in the output. 
                'nsamples_incoh'
                        [integer] Number of incoherent samples in producing the 
                        power spectrum
                'nsamples_coh'
                        [integer] Number of coherent samples in producing the 
                        power spectrum

    outfile     [string] Full path to the external HDF5 file where the cross-
                power spectrum information provided in xcpdps will be saved    ----------------------------------------------------------------------------
    """

    if not isinstance(xcpdps, dict):
        raise TypeError('Input xcpdps must be a dictionary')

    with h5py.File(outfile, 'w') as fileobj:
        hdrgrp = fileobj.create_group('header')
        hdrkeys = ['triads', 'triads_ind', 'lst', 'lst_ind', 'dlst', 'days', 'day_ind', 'dday']
        for key in hdrkeys:
            dset = hdrgrp.create_dataset(key, data=xcpdps[key])

        sampling = ['oversampled', 'resampled']
        sampling_keys = ['z', 'kprll', 'lags', 'freq_center', 'bw_eff', 'shape', 'freq_wts', 'lag_corr_length']
        dpool_keys = ['whole', 'submodel', 'residual', 'errinfo']
        for smplng in sampling:
            if smplng in xcpdps:
                smplgrp = fileobj.create_group(smplng)
                for key in sampling_keys:
                    dset = smplgrp.create_dataset(key, data=xcpdps[smplng][key])
                for dpool in dpool_keys:
                    if dpool in xcpdps[smplng]:
                        dpoolgrp = smplgrp.create_group(dpool)
                        keys = ['diagoffsets', 'diagweights', 'axesmap', 'nsamples_incoh', 'nsamples_coh']
                        for key in keys:
                            if isinstance(xcpdps[smplng][dpool][key], dict):
                                subgrp = dpoolgrp.create_group(key)
                                for subkey in xcpdps[smplng][dpool][key]:
                                    dset = subgrp.create_dataset(str(subkey), data=xcpdps[smplng][dpool][key][subkey])
                            else:
                                dset = dpoolgrp.create_dataset(key, data=xcpdps[smplng][dpool][key])
                        for stat in ['mean', 'median']:
                            if stat in xcpdps[smplng][dpool]:
                                dset = dpoolgrp.create_dataset(stat, data=xcpdps[smplng][dpool][stat].si.value)
                                dset.attrs['units'] = str(xcpdps[smplng][dpool][stat].si.unit)

################################################################################

def read_CPhase_cross_power_spectrum(infile):

    """
    ----------------------------------------------------------------------------
    Read information about cross power spectrum from an external HDF5 file into
    a dictionary. This is the counterpart to save_CPhase_corss_power_spectrum()

    Input:

    infile      [string] Full path to the external HDF5 file that contains info
                about cross-power spectrum. 

    Output:

    xcpdps      [dictionary] This dictionary has structure the same as output 
                of the member function compute_power_spectrum() of class 
                ClosurePhaseDelaySpectrum. It has the following key-value 
                structure:
                'triads' ((ntriads,3) array), 'triads_ind', 
                ((ntriads,) array), 'lstXoffsets' ((ndlst_range,) array), 'lst' 
                ((nlst,) array), 'dlst' ((nlst,) array), 'lst_ind' ((nlst,) 
                array), 'days' ((ndays,) array), 'day_ind' ((ndays,) array), 
                'dday' ((ndays,) array), 'oversampled' and 'resampled' 
                corresponding to whether resample was set to False or True in 
                call to member function FT(). Values under keys 'triads_ind' 
                and 'lst_ind' are numpy array corresponding to triad and time 
                indices used in selecting the data. Values under keys 
                'oversampled' and 'resampled' each contain a dictionary with 
                the following keys and values:
                'z'     [numpy array] Redshifts corresponding to the band 
                        centers in 'freq_center'. It has shape=(nspw,)
                'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,)
                'kprll' [numpy array] k_parallel modes (in h/Mpc) corresponding 
                        to 'lags'. It has shape=(nspw,nlags)
                'freq_center'   
                        [numpy array] contains the center frequencies (in Hz) 
                        of the frequency subbands of the subband delay spectra. 
                        It is of size n_win. It is roughly equivalent to 
                        redshift(s)
                'freq_wts'      
                        [numpy array] Contains frequency weights applied on 
                        each frequency sub-band during the subband delay 
                        transform. It is of size n_win x nchan. 
                'bw_eff'        
                        [numpy array] contains the effective bandwidths (in Hz) 
                        of the subbands being delay transformed. It is of size 
                        n_win. It is roughly equivalent to width in redshift or 
                        along line-of-sight
                'shape' [string] shape of the frequency window function applied. 
                        Usual values are 'rect' (rectangular), 'bhw' 
                        (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'fftpow'
                        [scalar] the power to which the FFT of the window was 
                        raised. The value is be a positive scalar with 
                        default = 1.0
                'lag_corr_length' 
                        [numpy array] It is the correlation timescale (in 
                        pixels) of the subband delay spectra. It is proportional 
                        to inverse of effective bandwidth. It is of size n_win. 
                        The unit size of a pixel is determined by the difference
                        between adjacent pixels in lags under key 'lags' which 
                        in turn is effectively inverse of the effective 
                        bandwidth of the subband specified in bw_eff
                
                It further contains one or more of the following keys named 
                'whole', 'submodel', 'residual', and 'errinfo' each of which is 
                a dictionary. 'whole' contains power spectrum info about the 
                input closure phases. 'submodel' contains power spectrum info 
                about the model that will have been subtracted (as closure 
                phase) from the 'whole' model. 'residual' contains power 
                spectrum info about the closure phases obtained as a difference 
                between 'whole' and 'submodel'. It contains the following keys 
                and values:
                'mean'  [numpy array] Delay power spectrum incoherently 
                        estimated over the axes specified in xinfo['axes'] 
                        using the 'mean' key in input cpds or attribute 
                        cPhaseDS['processed']['dspec']. It has shape that 
                        depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are 
                        not set, those axes will be replaced with square 
                        covariance matrices. If collapse_axes is provided but 
                        avgcov is False, those axes will be of shape 2*Naxis-1. 
                'median'
                        [numpy array] Delay power spectrum incoherently averaged 
                        over the axes specified in incohax using the 'median' 
                        key in input cpds or attribute 
                        cPhaseDS['processed']['dspec']. It has shape that 
                        depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are not 
                        set, those axes will be replaced with square covariance 
                        matrices. If collapse_axes is provided bu avgcov is 
                        False, those axes will be of shape 2*Naxis-1. 
                'diagoffsets' 
                        [dictionary] Same keys corresponding to keys under 
                        'collapse_axes' in input containing the diagonal 
                        offsets for those axes. If 'avgcov' was set, those 
                        entries will be removed from 'diagoffsets' since all the 
                        leading diagonal elements have been collapsed (averaged) 
                        further. Value under each key is a numpy array where 
                        each element in the array corresponds to the index of 
                        that leading diagonal. This should match the size of the 
                        output along that axis in 'mean' or 'median' above. 
                'diagweights'
                        [dictionary] Each key is an axis specified in 
                        collapse_axes and the value is a numpy array of weights 
                        corresponding to the diagonal offsets in that axis.
                'axesmap'
                        [dictionary] If covariance in cross-power is calculated 
                        but is not collapsed, the number of dimensions in the 
                        output will have changed. This parameter tracks where 
                        the original axis is now placed. The keys are the 
                        original axes that are involved in incoherent 
                        cross-power, and the values are the new locations of 
                        those original axes in the output. 
                'nsamples_incoh'
                        [integer] Number of incoherent samples in producing the 
                        power spectrum
                'nsamples_coh'
                        [integer] Number of coherent samples in producing the 
                        power spectrum

    outfile     [string] Full path to the external HDF5 file where the cross-
                power spectrum information provided in xcpdps will be saved    
    ----------------------------------------------------------------------------
    """

    if not isinstance(infile, str):
        raise TypeError('Input infile must be a string')
        
    xcpdps = {}
    with h5py.File(infile, 'r') as fileobj:
        hdrgrp = fileobj['header']
        hdrkeys = ['triads', 'triads_ind', 'lst', 'lst_ind', 'dlst', 'days', 'day_ind', 'dday']
        for key in hdrkeys:
            xcpdps[key] = hdrgrp[key].value
        sampling = ['oversampled', 'resampled']
        sampling_keys = ['z', 'kprll', 'lags', 'freq_center', 'bw_eff', 'shape', 'freq_wts', 'lag_corr_length']
        dpool_keys = ['whole', 'submodel', 'residual', 'errinfo']
        for smplng in sampling:
            if smplng in fileobj:
                smplgrp = fileobj[smplng]
                xcpdps[smplng] = {}
                for key in sampling_keys:
                    xcpdps[smplng][key] = smplgrp[key].value
                for dpool in dpool_keys:
                    if dpool in smplgrp:                    
                        xcpdps[smplng][dpool] = {}
                        dpoolgrp = smplgrp[dpool]
                        keys = ['diagoffsets', 'diagweights', 'axesmap', 'nsamples_incoh', 'nsamples_coh']
                        for key in keys:  
                            if isinstance(dpoolgrp[key], h5py.Group):
                                xcpdps[smplng][dpool][key] = {}
                                for subkey in dpoolgrp[key]:
                                    xcpdps[smplng][dpool][key][subkey] = dpoolgrp[key][subkey].value                                
                            elif isinstance(dpoolgrp[key], h5py.Dataset):
                                xcpdps[smplng][dpool][key] = dpoolgrp[key].value
                            else:
                                raise TypeError('Invalid h5py data type encountered')
                        for stat in ['mean', 'median']:
                            if stat in dpoolgrp:
                                valunits = dpoolgrp[stat].attrs['units']
                                xcpdps[smplng][dpool][stat] = dpoolgrp[stat].value * U.Unit(valunits)
    return xcpdps  

################################################################################

def incoherent_cross_power_spectrum_average(xcpdps, excpdps=None, diagoffsets=None):
    """
    ----------------------------------------------------------------------------
    Perform incoherent averaging of cross power spectrum along specified axes
    
    Inputs:

    xcpdps      [dictionary or list of dictionaries] If provided as a list of 
                dictionaries, each dictionary consists of cross power spectral
                information coming possible from different sources, and they 
                will be averaged be averaged incoherently. If a single 
                dictionary is provided instead of a list of dictionaries, the 
                said averaging does not take place. Each dictionary is 
                essentially an output of the member function 
                compute_power_spectrum() of class ClosurePhaseDelaySpectrum. It 
                has the following key-value structure:
                'triads' ((ntriads,3) array), 'triads_ind', 
                ((ntriads,) array), 'lstXoffsets' ((ndlst_range,) array), 'lst' 
                ((nlst,) array), 'dlst' ((nlst,) array), 'lst_ind' ((nlst,) 
                array), 'days' ((ndays,) array), 'day_ind' ((ndays,) array), 
                'dday' ((ndays,) array), 'oversampled' and 'resampled' 
                corresponding to whether resample was set to False or True in 
                call to member function FT(). Values under keys 'triads_ind' 
                and 'lst_ind' are numpy array corresponding to triad and time 
                indices used in selecting the data. Values under keys 
                'oversampled' and 'resampled' each contain a dictionary with 
                the following keys and values:
                'z'     [numpy array] Redshifts corresponding to the band 
                        centers in 'freq_center'. It has shape=(nspw,)
                'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,)
                'kprll' [numpy array] k_parallel modes (in h/Mpc) corresponding 
                        to 'lags'. It has shape=(nspw,nlags)
                'freq_center'   
                        [numpy array] contains the center frequencies (in Hz) 
                        of the frequency subbands of the subband delay spectra. 
                        It is of size n_win. It is roughly equivalent to 
                        redshift(s)
                'freq_wts'      
                        [numpy array] Contains frequency weights applied on 
                        each frequency sub-band during the subband delay 
                        transform. It is of size n_win x nchan. 
                'bw_eff'        
                        [numpy array] contains the effective bandwidths (in Hz) 
                        of the subbands being delay transformed. It is of size 
                        n_win. It is roughly equivalent to width in redshift or 
                        along line-of-sight
                'shape' [string] shape of the frequency window function applied. 
                        Usual values are 'rect' (rectangular), 'bhw' 
                        (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'fftpow'
                        [scalar] the power to which the FFT of the window was 
                        raised. The value is be a positive scalar with 
                        default = 1.0
                'lag_corr_length' 
                        [numpy array] It is the correlation timescale (in 
                        pixels) of the subband delay spectra. It is proportional 
                        to inverse of effective bandwidth. It is of size n_win. 
                        The unit size of a pixel is determined by the difference
                        between adjacent pixels in lags under key 'lags' which 
                        in turn is effectively inverse of the effective 
                        bandwidth of the subband specified in bw_eff
                
                It further contains 3 keys named 'whole', 'submodel', and 
                'residual' each of which is a dictionary. 'whole' contains power 
                spectrum info about the input closure phases. 'submodel' 
                contains power spectrum info about the model that will have been 
                subtracted (as closure phase) from the 'whole' model. 'residual' 
                contains power spectrum info about the closure phases obtained 
                as a difference between 'whole' and 'submodel'. It contains the 
                following keys and values:
                'mean'  [numpy array] Delay power spectrum incoherently 
                        estimated over the axes specified in xinfo['axes'] 
                        using the 'mean' key in input cpds or attribute 
                        cPhaseDS['processed']['dspec']. It has shape that 
                        depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are 
                        not set, those axes will be replaced with square 
                        covariance matrices. If collapse_axes is provided but 
                        avgcov is False, those axes will be of shape 2*Naxis-1. 
                'median'
                        [numpy array] Delay power spectrum incoherently averaged 
                        over the axes specified in incohax using the 'median' 
                        key in input cpds or attribute 
                        cPhaseDS['processed']['dspec']. It has shape that 
                        depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are not 
                        set, those axes will be replaced with square covariance 
                        matrices. If collapse_axes is provided bu avgcov is 
                        False, those axes will be of shape 2*Naxis-1. 
                'diagoffsets' 
                        [dictionary] Same keys corresponding to keys under 
                        'collapse_axes' in input containing the diagonal 
                        offsets for those axes. If 'avgcov' was set, those 
                        entries will be removed from 'diagoffsets' since all the 
                        leading diagonal elements have been collapsed (averaged) 
                        further. Value under each key is a numpy array where 
                        each element in the array corresponds to the index of 
                        that leading diagonal. This should match the size of the 
                        output along that axis in 'mean' or 'median' above. 
                'diagweights'
                        [dictionary] Each key is an axis specified in 
                        collapse_axes and the value is a numpy array of weights 
                        corresponding to the diagonal offsets in that axis.
                'axesmap'
                        [dictionary] If covariance in cross-power is calculated 
                        but is not collapsed, the number of dimensions in the 
                        output will have changed. This parameter tracks where 
                        the original axis is now placed. The keys are the 
                        original axes that are involved in incoherent 
                        cross-power, and the values are the new locations of 
                        those original axes in the output. 
                'nsamples_incoh'
                        [integer] Number of incoherent samples in producing the 
                        power spectrum
                'nsamples_coh'
                        [integer] Number of coherent samples in producing the 
                        power spectrum

    excpdps     [dictionary or list of dictionaries] If provided as a list of 
                dictionaries, each dictionary consists of cross power spectral
                information of subsample differences coming possible from 
                different sources, and they will be averaged be averaged 
                incoherently. This is optional. If not set (default=None), no 
                incoherent averaging happens. If a single dictionary is provided 
                instead of a list of dictionaries, the said averaging does not 
                take place. Each dictionary is essentially an output of the 
                member function compute_power_spectrum_uncertainty() of class 
                ClosurePhaseDelaySpectrum. It has the following key-value 
                structure:
                'triads' ((ntriads,3) array), 'triads_ind', 
                ((ntriads,) array), 'lstXoffsets' ((ndlst_range,) array), 'lst' 
                ((nlst,) array), 'dlst' ((nlst,) array), 'lst_ind' ((nlst,) 
                array), 'days' ((ndaycomb,) array), 'day_ind' ((ndaycomb,) 
                array), 'dday' ((ndaycomb,) array), 'oversampled' and 
                'resampled' corresponding to whether resample was set to False 
                or True in call to member function FT(). Values under keys 
                'triads_ind' and 'lst_ind' are numpy array corresponding to 
                triad and time indices used in selecting the data. Values under 
                keys 'oversampled' and 'resampled' each contain a dictionary 
                with the following keys and values:
                'z'     [numpy array] Redshifts corresponding to the band 
                        centers in 'freq_center'. It has shape=(nspw,)
                'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,)
                'kprll' [numpy array] k_parallel modes (in h/Mpc) corresponding
                        to 'lags'. It has shape=(nspw,nlags)
                'freq_center'   
                        [numpy array] contains the center frequencies (in Hz) of 
                        the frequency subbands of the subband delay spectra. It 
                        is of size n_win. It is roughly equivalent to 
                        redshift(s)
                'freq_wts'      
                        [numpy array] Contains frequency weights applied on each 
                        frequency sub-band during the subband delay transform. 
                        It is of size n_win x nchan. 
                'bw_eff'        
                        [numpy array] contains the effective bandwidths (in Hz) 
                        of the subbands being delay transformed. It is of size 
                        n_win. It is roughly equivalent to width in redshift or 
                        along line-of-sight
                'shape' [string] shape of the frequency window function applied. 
                        Usual values are 'rect' (rectangular), 'bhw' 
                        (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'fftpow'
                        [scalar] the power to which the FFT of the window was 
                        raised. The value is be a positive scalar with 
                        default = 1.0
                'lag_corr_length' 
                        [numpy array] It is the correlation timescale (in 
                        pixels) of the subband delay spectra. It is proportional 
                        to inverse of effective bandwidth. It is of size n_win. 
                        The unit size of a pixel is determined by the difference 
                        between adjacent pixels in lags under key 'lags' which 
                        in turn is effectively inverse of the effective 
                        bandwidth of the subband specified in bw_eff
                
                It further contains a key named 'errinfo' which is a dictionary. 
                It contains information about power spectrum uncertainties 
                obtained from subsample differences. It contains the following 
                keys and values:
                'mean'  [numpy array] Delay power spectrum uncertainties 
                        incoherently estimated over the axes specified in 
                        xinfo['axes'] using the 'mean' key in input cpds or 
                        attribute cPhaseDS['errinfo']['dspec']. It has shape 
                        that depends on the combination of input parameters. See 
                        examples below. If both collapse_axes and avgcov are not 
                        set, those axes will be replaced with square covariance 
                        matrices. If collapse_axes is provided but avgcov is 
                        False, those axes will be of shape 2*Naxis-1. 
                'median'
                        [numpy array] Delay power spectrum uncertainties 
                        incoherently averaged over the axes specified in incohax 
                        using the 'median' key in input cpds or attribute 
                        cPhaseDS['errinfo']['dspec']. It has shape that depends 
                        on the combination of input parameters. See examples 
                        below. If both collapse_axes and avgcov are not set, 
                        those axes will be replaced with square covariance 
                        matrices. If collapse_axes is provided but avgcov is 
                        False, those axes will be of shape 2*Naxis-1. 
                'diagoffsets' 
                        [dictionary] Same keys corresponding to keys under 
                        'collapse_axes' in input containing the diagonal offsets 
                        for those axes. If 'avgcov' was set, those entries will 
                        be removed from 'diagoffsets' since all the leading 
                        diagonal elements have been collapsed (averaged) further. 
                        Value under each key is a numpy array where each element 
                        in the array corresponds to the index of that leading 
                        diagonal. This should match the size of the output along 
                        that axis in 'mean' or 'median' above. 
                'diagweights'
                        [dictionary] Each key is an axis specified in 
                        collapse_axes and the value is a numpy array of weights 
                        corresponding to the diagonal offsets in that axis.
                'axesmap'
                        [dictionary] If covariance in cross-power is calculated 
                        but is not collapsed, the number of dimensions in the 
                        output will have changed. This parameter tracks where 
                        the original axis is now placed. The keys are the 
                        original axes that are involved in incoherent 
                        cross-power, and the values are the new locations of 
                        those original axes in the output. 
                'nsamples_incoh'
                        [integer] Number of incoherent samples in producing the 
                        power spectrum
                'nsamples_coh'
                        [integer] Number of coherent samples in producing the 
                        power spectrum

    diagoffsets [NoneType or dictionary or list of dictionaries] This info is
                used for incoherent averaging along specified diagonals along
                specified axes. This incoherent averaging is performed after
                incoherently averaging multiple cross-power spectra (if any). 
                If set to None, this incoherent averaging is not performed. 
                Many combinations of axes and diagonals can be specified as 
                individual dictionaries in a list. If only one dictionary is
                specified, then it assumed that only one combination of axes
                and diagonals is requested. If a list of dictionaries is given,
                each dictionary in the list specifies a different combination
                for incoherent averaging. Each dictionary should have the 
                following key-value pairs. The key is the axis number (allowed 
                values are 1, 2, 3) that denote the axis type (1=LST, 2=Days, 
                3=Triads to be averaged), and the value under they keys is a
                list or numpy array of diagonals to be averaged incoherently. 
                These axes-diagonal combinations apply to both the inputs 
                xcpdps and excpdps, except axis=2 does not apply to excpdps 
                (since it is made of subsample differences already) and will be
                skipped. 

    Outputs:

    A tuple consisting of two dictionaries. The first dictionary contains the
    incoherent averaging of xcpdps as specified by the inputs, while the second
    consists of incoherent of excpdps as specified by the inputs. The structure
    of these dictionaries are practically the same as the dictionary inputs 
    xcpdps and excpdps respectively. The only differences in dictionary 
    structure are:
    * Under key ['oversampled'/'resampled']['whole'/'submodel'/'residual'
      /'effinfo']['mean'/'median'] is a list of numpy arrays, where each
      array in the list corresponds to the dictionary in the list in input
      diagoffsets that defines the axes-diagonal combination. 
    ----------------------------------------------------------------------------
    """

    if isinstance(xcpdps, dict):
        xcpdps = [xcpdps]
    if not isinstance(xcpdps, list):
        raise TypeError('Invalid data type provided for input xcpdps')

    if excpdps is not None:
        if isinstance(excpdps, dict):
            excpdps = [excpdps]
        if not isinstance(excpdps, list):
            raise TypeError('Invalid data type provided for input excpdps')
        if len(xcpdps) != len(excpdps):
            raise ValueError('Inputs xcpdps and excpdps found to have unequal number of values')

    out_xcpdps = {'triads': xcpdps[0]['triads'], 'triads_ind': xcpdps[0]['triads_ind'], 'lst': xcpdps[0]['lst'], 'lst_ind': xcpdps[0]['lst_ind'], 'dlst': xcpdps[0]['dlst'], 'days': xcpdps[0]['days'], 'day_ind': xcpdps[0]['day_ind'], 'dday': xcpdps[0]['dday']}
    
    out_excpdps = None
    if excpdps is not None:
        out_excpdps = {'triads': excpdps[0]['triads'], 'triads_ind': excpdps[0]['triads_ind'], 'lst': excpdps[0]['lst'], 'lst_ind': excpdps[0]['lst_ind'], 'dlst': excpdps[0]['dlst'], 'days': excpdps[0]['days'], 'day_ind': excpdps[0]['day_ind'], 'dday': excpdps[0]['dday']}

    for smplng in ['oversampled', 'resampled']:
        if smplng in xcpdps[0]:
            out_xcpdps[smplng] = {'z': xcpdps[0][smplng]['z'], 'kprll': xcpdps[0][smplng]['kprll'], 'lags': xcpdps[0][smplng]['lags'], 'freq_center': xcpdps[0][smplng]['freq_center'], 'bw_eff': xcpdps[0][smplng]['bw_eff'], 'shape': xcpdps[0][smplng]['shape'], 'freq_wts': xcpdps[0][smplng]['freq_wts'], 'lag_corr_length': xcpdps[0][smplng]['lag_corr_length']}
            if excpdps is not None:
                out_excpdps[smplng] = {'z': excpdps[0][smplng]['z'], 'kprll': excpdps[0][smplng]['kprll'], 'lags': excpdps[0][smplng]['lags'], 'freq_center': excpdps[0][smplng]['freq_center'], 'bw_eff': excpdps[0][smplng]['bw_eff'], 'shape': excpdps[0][smplng]['shape'], 'freq_wts': excpdps[0][smplng]['freq_wts'], 'lag_corr_length': excpdps[0][smplng]['lag_corr_length']}

            for dpool in ['whole', 'submodel', 'residual']:
                if dpool in xcpdps[0][smplng]:
                    out_xcpdps[smplng][dpool] = {'diagoffsets': xcpdps[0][smplng][dpool]['diagoffsets'], 'axesmap': xcpdps[0][smplng][dpool]['axesmap']}
                    for stat in ['mean', 'median']:
                        if stat in xcpdps[0][smplng][dpool]:
                            out_xcpdps[smplng][dpool][stat] = {}
                            arr = []
                            diagweights = []
                            diagwts = 1.0
                            for i in range(len(xcpdps)):
                                arr += [xcpdps[i][smplng][dpool][stat].si.value]
                                arr_units = xcpdps[i][smplng][dpool][stat].si.unit
                                diagwts_shape = NP.ones(xcpdps[i][smplng][dpool][stat].ndim, dtype=NP.int)
                                for ax in xcpdps[i][smplng][dpool]['diagweights']:
                                    tmp_shape = NP.copy(diagwts_shape)
                                    tmp_shape[xcpdps[i][smplng][dpool]['axesmap'][ax]] = xcpdps[i][smplng][dpool]['diagweights'][ax].size
                                    diagwts = diagwts * xcpdps[i][smplng][dpool]['diagweights'][ax].reshape(tuple(tmp_shape))
                                diagweights += [diagwts]
                            diagweights = NP.asarray(diagweights)
                            arr = NP.asarray(arr)
                            arr = NP.nansum(arr * diagweights, axis=0) / NP.nansum(diagweights, axis=0) * arr_units
                            diagweights = NP.nansum(diagweights, axis=0)
                        out_xcpdps[smplng][dpool][stat] = arr
                    out_xcpdps[smplng][dpool]['diagweights'] = diagweights

            for dpool in ['errinfo']:
                if dpool in excpdps[0][smplng]:
                    out_excpdps[smplng][dpool] = {'diagoffsets': excpdps[0][smplng][dpool]['diagoffsets'], 'axesmap': excpdps[0][smplng][dpool]['axesmap']}
                    for stat in ['mean', 'median']:
                        if stat in excpdps[0][smplng][dpool]:
                            out_excpdps[smplng][dpool][stat] = {}
                            arr = []
                            diagweights = []
                            diagwts = 1.0
                            for i in range(len(excpdps)):
                                arr += [excpdps[i][smplng][dpool][stat].si.value]
                                arr_units = excpdps[i][smplng][dpool][stat].si.unit
                                diagwts_shape = NP.ones(excpdps[i][smplng][dpool][stat].ndim, dtype=NP.int)
                                for ax in excpdps[i][smplng][dpool]['diagweights']:
                                    tmp_shape = NP.copy(diagwts_shape)
                                    tmp_shape[excpdps[i][smplng][dpool]['axesmap'][ax]] = excpdps[i][smplng][dpool]['diagweights'][ax].size
                                    diagwts = diagwts * excpdps[i][smplng][dpool]['diagweights'][ax].reshape(tuple(tmp_shape))
                                diagweights += [diagwts]
                            diagweights = NP.asarray(diagweights)
                            arr = NP.asarray(arr)
                            arr = NP.nansum(arr * diagweights, axis=0) / NP.nansum(diagweights, axis=0) * arr_units
                            diagweights = NP.nansum(diagweights, axis=0)
                        out_excpdps[smplng][dpool][stat] = arr
                    out_excpdps[smplng][dpool]['diagweights'] = diagweights

    if diagoffsets is not None:
        if isinstance(diagoffsets, dict):
            diagoffsets = [diagoffsets]
        if not isinstance(diagoffsets, list):
            raise TypeError('Input diagoffsets must be a list of dictionaries')
        for ind in range(len(diagoffsets)):
            for ax in diagoffsets[ind]:
                if not isinstance(diagoffsets[ind][ax], (list, NP.ndarray)):
                    raise TypeError('Values in input dictionary diagoffsets must be a list or numpy array')
                diagoffsets[ind][ax] = NP.asarray(diagoffsets[ind][ax])

        for smplng in ['oversampled', 'resampled']:
            if smplng in out_xcpdps:
                for dpool in ['whole', 'submodel', 'residual']:
                    if dpool in out_xcpdps[smplng]:
                        masks = []
                        for ind in range(len(diagoffsets)):
                            mask_ones = NP.ones(out_xcpdps[smplng][dpool]['diagweights'].shape, dtype=NP.bool)
                            mask_agg = None
                            for ax in diagoffsets[ind]:
                                mltdim_slice = [slice(None)] * mask_ones.ndim
                                mltdim_slice[out_xcpdps[smplng][dpool]['axesmap'][ax].squeeze()] = NP.where(NP.isin(out_xcpdps[smplng][dpool]['diagoffsets'][ax], diagoffsets[ind][ax]))[0]
                                mask_tmp = NP.copy(mask_ones)
                                mask_tmp[tuple(mltdim_slice)] = False
                                if mask_agg is None:
                                    mask_agg = NP.copy(mask_tmp)
                                else:
                                    mask_agg = NP.logical_or(mask_agg, mask_tmp)
                            masks += [NP.copy(mask_agg)]
                        diagwts = NP.copy(out_xcpdps[smplng][dpool]['diagweights'])
                        out_xcpdps[smplng][dpool]['diagweights'] = []
                        for stat in ['mean', 'median']:
                            if stat in out_xcpdps[smplng][dpool]:
                                arr = NP.copy(out_xcpdps[smplng][dpool][stat].si.value)
                                arr_units = out_xcpdps[smplng][dpool][stat].si.unit
                                out_xcpdps[smplng][dpool][stat] = []
                                for ind in range(len(diagoffsets)):
                                    masked_diagwts = MA.array(diagwts, mask=masks[ind])
                                    axes_to_avg = tuple([out_xcpdps[smplng][dpool]['axesmap'][ax][0] for ax in diagoffsets[ind]])
                                    out_xcpdps[smplng][dpool][stat] += [MA.sum(arr * masked_diagwts, axis=axes_to_avg, keepdims=True) / MA.sum(masked_diagwts, axis=axes_to_avg, keepdims=True) * arr_units]
                                    if len(out_xcpdps[smplng][dpool]['diagweights']) < len(diagoffsets):
                                        out_xcpdps[smplng][dpool]['diagweights'] += [MA.sum(masked_diagwts, axis=axes_to_avg, keepdims=True)]

        if excpdps is not None:
            for smplng in ['oversampled', 'resampled']:
                if smplng in out_excpdps:
                    for dpool in ['errinfo']:
                        if dpool in out_excpdps[smplng]:
                            masks = []
                            for ind in range(len(diagoffsets)):
                                mask_ones = NP.ones(out_excpdps[smplng][dpool]['diagweights'].shape, dtype=NP.bool)
                                mask_agg = None
                                for ax in diagoffsets[ind]:
                                    if ax != 2:
                                        mltdim_slice = [slice(None)] * mask_ones.ndim
                                        mltdim_slice[out_excpdps[smplng][dpool]['axesmap'][ax].squeeze()] = NP.where(NP.isin(out_excpdps[smplng][dpool]['diagoffsets'][ax], diagoffsets[ind][ax]))[0]
                                        mask_tmp = NP.copy(mask_ones)
                                        mask_tmp[tuple(mltdim_slice)] = False
                                        if mask_agg is None:
                                            mask_agg = NP.copy(mask_tmp)
                                        else:
                                            mask_agg = NP.logical_or(mask_agg, mask_tmp)
                                masks += [NP.copy(mask_agg)]
                            diagwts = NP.copy(out_excpdps[smplng][dpool]['diagweights'])
                            out_excpdps[smplng][dpool]['diagweights'] = []
                            for stat in ['mean', 'median']:
                                if stat in out_excpdps[smplng][dpool]:
                                    arr = NP.copy(out_excpdps[smplng][dpool][stat].si.value)
                                    arr_units = out_excpdps[smplng][dpool][stat].si.unit
                                    out_excpdps[smplng][dpool][stat] = []
                                    for ind in range(len(diagoffsets)):
                                        masked_diagwts = MA.array(diagwts, mask=masks[ind])
                                        axes_to_avg = tuple([out_excpdps[smplng][dpool]['axesmap'][ax][0] for ax in diagoffsets[ind]])
                                        out_excpdps[smplng][dpool][stat] += [MA.sum(arr * masked_diagwts, axis=axes_to_avg, keepdims=True) / MA.sum(masked_diagwts, axis=axes_to_avg, keepdims=True) * arr_units]
                                        if len(out_excpdps[smplng][dpool]['diagweights']) < len(diagoffsets):
                                            out_excpdps[smplng][dpool]['diagweights'] += [MA.sum(masked_diagwts, axis=axes_to_avg, keepdims=True)]
                                        
    return (out_xcpdps, out_excpdps)

################################################################################

def incoherent_kbin_averaging(xcpdps, kbins=None, num_kbins=None, kbintype='log'):

    """
    ----------------------------------------------------------------------------
    Averages the power spectrum incoherently by binning in bins of k. Returns
    the power spectrum in units of both standard power spectrum and \Delta^2

    Inputs:

    xcpdps      [dictionary] A dictionary that contains the incoherent averaged
                power spectrum along LST and/or triads axes. This dictionary is
                essentially the one(s) returned as the output of the function
                incoherent_cross_power_spectrum_average()

    kbins       [NoneType, list or numpy array] Bins in k. If set to None 
                (default), it will be determined automatically based on the 
                inputs in num_kbins, and kbintype. If num_kbins is None and 
                kbintype='linear', the negative and positive values of k are
                folded into a one-sided power spectrum. In this case, the 
                bins will approximately have the same resolution as the k-values 
                in the input power spectrum for all the spectral windows. 
                

    num_kbins   [NoneType or integer] Number of k-bins. Used only if kbins is 
                set to None. If kbintype is set to 'linear', the negative and 
                positive values of k are folded into a one-sided power spectrum.
                In this case, the bins will approximately have the same 
                resolution as the k-values in the input power spectrum for all
                the spectral windows. 

    kbintype    [string] Specifies the type of binning, used only if kbins is 
                set to None. Accepted values are 'linear' and 'log' for linear 
                and logarithmic bins respectively.

    Outputs:

    Dictionary containing the power spectrum information. At the top level, it
    contains keys specifying the sampling to be 'oversampled' or 'resampled'. 
    Under each of these keys is another dictionary containing the following 
    keys:
    'z'     [numpy array] Redshifts corresponding to the band centers in 
            'freq_center'. It has shape=(nspw,)
    'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,).
    'freq_center'   
            [numpy array] contains the center frequencies (in Hz) of the 
            frequency subbands of the subband delay spectra. It is of size 
            n_win. It is roughly equivalent to redshift(s)
    'freq_wts'      
            [numpy array] Contains frequency weights applied on each 
            frequency sub-band during the subband delay transform. It is 
            of size n_win x nchan. 
    'bw_eff'        
            [numpy array] contains the effective bandwidths (in Hz) of the 
            subbands being delay transformed. It is of size n_win. It is 
            roughly equivalent to width in redshift or along line-of-sight
    'shape' [string] shape of the frequency window function applied. Usual
            values are 'rect' (rectangular), 'bhw' (Blackman-Harris), 
            'bnw' (Blackman-Nuttall). 
    'fftpow'
            [scalar] the power to which the FFT of the window was raised. 
            The value is be a positive scalar with default = 1.0
    'lag_corr_length' 
            [numpy array] It is the correlation timescale (in pixels) of 
            the subband delay spectra. It is proportional to inverse of 
            effective bandwidth. It is of size n_win. The unit size of a 
            pixel is determined by the difference between adjacent pixels 
            in lags under key 'lags' which in turn is effectively inverse 
            of the effective bandwidth of the subband specified in bw_eff
        It further contains 3 keys named 'whole', 'submodel', and 'residual'
        or one key named 'errinfo' each of which is a dictionary. 'whole' 
        contains power spectrum info about the input closure phases. 'submodel' 
        contains power spectrum info about the model that will have been 
        subtracted (as closure phase) from the 'whole' model. 'residual' 
        contains power spectrum info about the closure phases obtained as a 
        difference between 'whole' and 'submodel'. 'errinfo' contains power
        spectrum information about the subsample differences. There is also 
        another dictionary under key 'kbininfo' that contains information about
        k-bins. These dictionaries contain the following keys and values:
        'whole'/'submodel'/'residual'/'errinfo'
            [dictionary] It contains the following keys and values:
            'mean'  [dictionary] Delay power spectrum information under the 
                    'mean' statistic incoherently obtained by averaging the 
                    input power spectrum in bins of k. It contains output power 
                    spectrum expressed as two quantities each of which is a 
                    dictionary with the following key-value pairs:
                    'PS'    [list of numpy arrays] Standard power spectrum in
                            units of 'K2 Mpc3'. Each numpy array in the list 
                            maps to a specific combination of axes and axis 
                            diagonals chosen for incoherent averaging in 
                            earlier processing such as in the function 
                            incoherent_cross_power_spectrum_average(). The 
                            numpy array has a shape similar to the input power
                            spectrum, but that last axis (k-axis) will have a
                            different size that depends on the k-bins that
                            were used in the incoherent averaging along that
                            axis. 
                    'Del2'  [list of numpy arrays] power spectrum in Delta^2
                            units of 'K2'. Each numpy array in the list 
                            maps to a specific combination of axes and axis 
                            diagonals chosen for incoherent averaging in 
                            earlier processing such as in the function 
                            incoherent_cross_power_spectrum_average(). The 
                            numpy array has a shape similar to the input power
                            spectrum, but that last axis (k-axis) will have a
                            different size that depends on the k-bins that
                            were used in the incoherent averaging along that
                            axis. 
            'median'
                    [dictionary] Delay power spectrum information under the 
                    'median' statistic incoherently obtained by averaging the 
                    input power spectrum in bins of k. It contains output power 
                    spectrum expressed as two quantities each of which is a 
                    dictionary with the following key-value pairs:
                    'PS'    [list of numpy arrays] Standard power spectrum in
                            units of 'K2 Mpc3'. Each numpy array in the list 
                            maps to a specific combination of axes and axis 
                            diagonals chosen for incoherent averaging in 
                            earlier processing such as in the function 
                            incoherent_cross_power_spectrum_average(). The 
                            numpy array has a shape similar to the input power
                            spectrum, but that last axis (k-axis) will have a
                            different size that depends on the k-bins that
                            were used in the incoherent averaging along that
                            axis. 
                    'Del2'  [list of numpy arrays] power spectrum in Delta^2
                            units of 'K2'. Each numpy array in the list 
                            maps to a specific combination of axes and axis 
                            diagonals chosen for incoherent averaging in 
                            earlier processing such as in the function 
                            incoherent_cross_power_spectrum_average(). The 
                            numpy array has a shape similar to the input power
                            spectrum, but that last axis (k-axis) will have a
                            different size that depends on the k-bins that
                            were used in the incoherent averaging along that
                            axis. 
        'kbininfo'  
            [dictionary] Contains the k-bin information. It contains the 
            following key-value pairs:
            'counts'    
                [list] List of numpy arrays where each numpy array in the stores 
                the counts in the determined k-bins. Each numpy array in the 
                list corresponds to a spectral window (redshift subband). The 
                shape of each numpy array is (nkbins,)
            'kbin_edges'
                [list] List of numpy arrays where each numpy array contains the 
                k-bin edges. Each array in the list corresponds to a spectral 
                window (redshift subband). The shape of each array is 
                (nkbins+1,). 
            'kbinnum'
                [list] List of numpy arrays containing the bin number under 
                which the k value falls. Each array in the list corresponds to 
                a spectral window (redshift subband). The shape of each array 
                is (nlags,).
            'ri'
                [list] List of numpy arrays containing the reverse indices for 
                each k-bin. Each array in the list corresponds to a spectral 
                window (redshift subband). The shape of each array is 
                (nlags+nkbins+1,).
            'whole'/'submodel'/'residual' or 'errinfo' [dictionary] k-bin info 
                estimated for the different datapools under different stats 
                and PS definitions. It has the keys 'mean' and 'median' for the
                mean and median statistic respectively. Each of them contain a
                dictionary with the following key-value pairs:
                'PS'    [list] List of numpy arrays where each numpy array 
                        contains a standard power spectrum typically in units of
                        'K2 Mpc3'. Its shape is the same as input power spectrum
                        except the k-axis which now has nkbins number of 
                        elements. 
                'Del2'  [list] List of numpy arrays where each numpy array 
                        contains a Delta^2 power spectrum typically in units of
                        'K2'. Its shape is the same as input power spectrum
                        except the k-axis which now has nkbins number of 
                        elements. 
    ----------------------------------------------------------------------------
    """

    if not isinstance(xcpdps, dict):
        raise TypeError('Input xcpdps must be a dictionary')
    if kbins is not None:
        if not isinstance(kbins, (list,NP.ndarray)):
            raise TypeError('Input kbins must be a list or numpy array')
    else:
        if not isinstance(kbintype, str):
            raise TypeError('Input kbintype must be a string')
        if kbintype.lower() not in ['linear', 'log']:
            raise ValueError('Input kbintype must be set to "linear" or "log"')
        if kbintype.lower() == 'log':
            if num_kbins is None:
                num_kbins = 10
    psinfo = {}
    keys = ['triads', 'triads_ind', 'lst', 'lst_ind', 'dlst', 'days', 'day_ind', 'dday']
    for key in keys:
        psinfo[key] = xcpdps[key]
    sampling = ['oversampled', 'resampled']
    sampling_keys = ['z', 'freq_center', 'bw_eff', 'shape', 'freq_wts', 'lag_corr_length']
    dpool_keys = ['whole', 'submodel', 'residual', 'errinfo']
    for smplng in sampling:
        if smplng in xcpdps:
            psinfo[smplng] = {}
            for key in sampling_keys:
                psinfo[smplng][key] = xcpdps[smplng][key]
            kprll = xcpdps[smplng]['kprll']
            lags = xcpdps[smplng]['lags']
            eps = 1e-10
            if kbins is None:
                dkprll = NP.max(NP.mean(NP.diff(kprll, axis=-1), axis=-1))
                if kbintype.lower() == 'linear':
                    bins_kprll = NP.linspace(eps, NP.abs(kprll).max()+eps, num=kprll.shape[1]/2+1, endpoint=True)
                else:
                    bins_kprll = NP.geomspace(eps, NP.abs(kprll).max()+eps, num=num_kbins+1, endpoint=True)
                bins_kprll = NP.insert(bins_kprll, 0, -eps)
            else:
                bins_kprll = NP.asarray(kbins)
            num_kbins = bins_kprll.size - 1
            psinfo[smplng]['kbininfo'] = {'counts': [], 'kbin_edges': [], 'kbinnum': [], 'ri': []}
            for spw in range(kprll.shape[0]):
                counts, kbin_edges, kbinnum, ri = OPS.binned_statistic(NP.abs(kprll[spw,:]), statistic='count', bins=bins_kprll)
                counts = counts.astype(NP.int)           
                psinfo[smplng]['kbininfo']['counts'] += [NP.copy(counts)]
                psinfo[smplng]['kbininfo']['kbin_edges'] += [kbin_edges / U.Mpc]
                psinfo[smplng]['kbininfo']['kbinnum'] += [NP.copy(kbinnum)]
                psinfo[smplng]['kbininfo']['ri'] += [NP.copy(ri)]
            for dpool in dpool_keys:
                if dpool in xcpdps[smplng]:
                    psinfo[smplng][dpool] = {}
                    psinfo[smplng]['kbininfo'][dpool] = {}
                    keys = ['diagoffsets', 'diagweights', 'axesmap']
                    for key in keys:
                        psinfo[smplng][dpool][key] = xcpdps[smplng][dpool][key]
                    for stat in ['mean', 'median']:
                        if stat in xcpdps[smplng][dpool]:
                            psinfo[smplng][dpool][stat] = {'PS': [], 'Del2': []}
                            psinfo[smplng]['kbininfo'][dpool][stat] = []
                            for combi in range(len(xcpdps[smplng][dpool][stat])):
                                outshape = NP.asarray(xcpdps[smplng][dpool][stat][combi].shape)
                                outshape[-1] = num_kbins
                                tmp_dps = NP.full(tuple(outshape), NP.nan, dtype=NP.complex) * U.Unit(xcpdps[smplng][dpool][stat][combi].unit)
                                tmp_Del2 = NP.full(tuple(outshape), NP.nan, dtype=NP.complex) * U.Unit(xcpdps[smplng][dpool][stat][combi].unit / U.Mpc**3)
                                tmp_kprll = NP.full(tuple(outshape), NP.nan, dtype=NP.float) / U.Mpc
                                for spw in range(kprll.shape[0]):
                                    counts = NP.copy(psinfo[smplng]['kbininfo']['counts'][spw])
                                    ri = NP.copy(psinfo[smplng]['kbininfo']['ri'][spw])
                                    for binnum in range(num_kbins):
                                        if counts[binnum] > 0:
                                            ind_kbin = ri[ri[binnum]:ri[binnum+1]]
                                            tmp_dps[spw,...,binnum] = NP.nanmean(NP.take(xcpdps[smplng][dpool][stat][combi][spw], ind_kbin, axis=-1), axis=-1)
                                            k_shape = NP.ones(NP.take(xcpdps[smplng][dpool][stat][combi][spw], ind_kbin, axis=-1).ndim, dtype=NP.int)
                                            k_shape[-1] = -1
                                            tmp_Del2[spw,...,binnum] = NP.nanmean(NP.abs(kprll[spw,ind_kbin].reshape(tuple(k_shape))/U.Mpc)**3 * NP.take(xcpdps[smplng][dpool][stat][combi][spw], ind_kbin, axis=-1), axis=-1) / (2*NP.pi**2)
                                            tmp_kprll[spw,...,binnum] = NP.nansum(NP.abs(kprll[spw,ind_kbin].reshape(tuple(k_shape))/U.Mpc) * NP.abs(NP.take(xcpdps[smplng][dpool][stat][combi][spw], ind_kbin, axis=-1)), axis=-1) / NP.nansum(NP.abs(NP.take(xcpdps[smplng][dpool][stat][combi][spw], ind_kbin, axis=-1)), axis=-1)
                                psinfo[smplng][dpool][stat]['PS'] += [copy.deepcopy(tmp_dps)]
                                psinfo[smplng][dpool][stat]['Del2'] += [copy.deepcopy(tmp_Del2)]
                                psinfo[smplng]['kbininfo'][dpool][stat] += [copy.deepcopy(tmp_kprll)]

    return psinfo

################################################################################

class ClosurePhase(object):

    """
    ----------------------------------------------------------------------------
    Class to hold and operate on Closure Phase information. 

    It has the following attributes and member functions.

    Attributes:

    extfile         [string] Full path to external file containing information
                    of ClosurePhase instance. The file is in HDF5 format

    cpinfo          [dictionary] Contains the following top level keys, 
                    namely, 'raw', 'processed', and 'errinfo'

                    Under key 'raw' which holds a dictionary, the subkeys 
                    include 'cphase' (nlst,ndays,ntriads,nchan), 
                    'triads' (ntriads,3), 'lst' (nlst,ndays), and 'flags' 
                    (nlst,ndays,ntriads,nchan). 

                    Under the 'processed' key are more subkeys, namely, 
                    'native', 'prelim', and optionally 'submodel' and 'residual' 
                    each holding a dictionary. 
                        Under 'native' dictionary, the subsubkeys for further 
                        dictionaries are 'cphase' (masked array: 
                        (nlst,ndays,ntriads,nchan)), 'eicp' (complex masked 
                        array: (nlst,ndays,ntriads,nchan)), and 'wts' (masked 
                        array: (nlst,ndays,ntriads,nchan)).

                        Under 'prelim' dictionary, the subsubkeys for further 
                        dictionaries are 'tbins' (numpy array of tbin centers 
                        after smoothing), 'dtbins' (numpy array of tbin 
                        intervals), 'wts' (masked array: 
                        (ntbins,ndays,ntriads,nchan)), 'eicp' and 'cphase'. 
                        The dictionaries under 'eicp' are indexed by keys 
                        'mean' (complex masked array: 
                        (ntbins,ndays,ntriads,nchan)), and 'median' (complex
                        masked array: (ntbins,ndays,ntriads,nchan)). 
                        The dictionaries under 'cphase' are indexed by keys
                        'mean' (masked array: (ntbins,ndays,ntriads,nchan)), 
                        'median' (masked array: (ntbins,ndays,ntriads,nchan)),
                        'rms' (masked array: (ntbins,ndays,ntriads,nchan)), and
                        'mad' (masked array: (ntbins,ndays,ntriads,nchan)). The
                        last one denotes Median Absolute Deviation.

                        Under 'submodel' dictionary, the subsubkeys for further
                        dictionaries are 'cphase' (masked array: 
                        (nlst,ndays,ntriads,nchan)), and 'eicp' (complex masked 
                        array: (nlst,ndays,ntriads,nchan)). 

                        Under 'residual' dictionary, the subsubkeys for further
                        dictionaries are 'cphase' and 'eicp'. These are 
                        dictionaries too. The dictionaries under 'eicp' are 
                        indexed by keys 'mean' (complex masked array: 
                        (ntbins,ndays,ntriads,nchan)), and 'median' (complex
                        masked array: (ntbins,ndays,ntriads,nchan)). 
                        The dictionaries under 'cphase' are indexed by keys
                        'mean' (masked array: (ntbins,ndays,ntriads,nchan)), 
                        and 'median' (masked array: 
                        (ntbins,ndays,ntriads,nchan)).

                    Under key 'errinfo', it contains the following keys and
                    values:
                    'list_of_pair_of_pairs' 
                            List of pair of pairs for which differences of
                            complex exponentials have been computed, where the
                            elements are bins of days. The number of elements
                            in the list is ncomb. And each element is a smaller 
                            (4-element) list of pair of pairs
                         
                    'eicp_diff'
                            Difference of complex exponentials between pairs
                            of day bins. This will be used in evaluating noise
                            properties in power spectrum. It is a dictionary 
                            with two keys '0' and '1' where each contains the
                            difference from a pair of subsamples. Each of these
                            keys contains a numpy array of shape
                            (nlstbins,ncomb,2,ntriads,nchan)
                    'wts'   Weights in difference of complex exponentials 
                            obtained by sum of squares of weights that are
                            associated with the pair that was used in the
                            differencing. It is a dictionary with two keys '0' 
                            and '1' where each contains the weights associated 
                            It is of shape (nlstbins,ncomb,2,ntriads,nchan)

    Member functions:

    __init__()      Initialize an instance of class ClosurePhase

    expicp()        Compute and return complex exponential of the closure phase 
                    as a masked array

    smooth_in_tbins()
                    Smooth the complex exponentials of closure phases in LST  
                    bins. Both mean and median smoothing is produced.

    subtract()      Subtract complex exponential of the bispectrum phase 
                    from the current instance and updates the cpinfo attribute

    subsample_differencing()
                    Create subsamples and differences between subsamples to 
                    evaluate noise properties from the data set.

    save()          Save contents of attribute cpinfo in external HDF5 file
    ----------------------------------------------------------------------------
    """
    
    def __init__(self, infile, freqs, infmt='npz'):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class ClosurePhase

        Inputs:

        infile      [string] Input file including full path. It could be a NPZ
                    with raw data, or a HDF5 file that could contain raw or 
                    processed data. The input file format is specified in the 
                    input infmt. If it is a NPZ file, it must contain the 
                    following keys/files:
                    'closures'  [numpy array] Closure phase (radians). It is of 
                                shape (nlst,ndays,ntriads,nchan)
                    'triads'    [numpy array] Array of triad tuples, of shape 
                                (ntriads,3)
                    'flags'     [numpy array] Array of flags (boolean), of shape
                                (nlst,ndays,ntriads,nchan)
                    'last'      [numpy array] Array of LST for each day (CASA 
                                units which is MJD+6713). Shape is (nlst,ndays)
                    'days'      [numpy array] Array of days, shape is (ndays,)
                    'averaged_closures'
                                [numpy array] optional array of closure phases
                                averaged across days. Shape is 
                                (nlst,ntriads,nchan)
                    'std_dev_lst'
                                [numpy array] optional array of standard 
                                deviation of closure phases across days. Shape 
                                is (nlst,ntriads,nchan)
                    'std_dev_triads'
                                [numpy array] optional array of standard 
                                deviation of closure phases across triads. 
                                Shape is (nlst,ndays,nchan)

        freqs       [numpy array] Frequencies (in Hz) in the input. Size is 
                    nchan.

        infmt       [string] Input file format. Accepted values are 'npz' 
                    (default) and 'hdf5'.
        ------------------------------------------------------------------------
        """

        if not isinstance(infile, str):
            raise TypeError('Input infile must be a string')

        if not isinstance(freqs, NP.ndarray):
            raise TypeError('Input freqs must be a numpy array')
        freqs = freqs.ravel()

        if not isinstance(infmt, str):
            raise TypeError('Input infmt must be a string')

        if infmt.lower() not in ['npz', 'hdf5']:
            raise ValueError('Input infmt must be "npz" or "hdf5"')

        if infmt.lower() == 'npz':
            infilesplit = infile.split('.npz')
            infile_noext = infilesplit[0]
            self.cpinfo = loadnpz(infile)
            # npz2hdf5(infile, infile_noext+'.hdf5')
            self.extfile = infile_noext + '.hdf5'
        else:
            # if not isinstance(infile, h5py.File):
            #     raise TypeError('Input infile is not a valid HDF5 file')
            self.extfile = infile
            self.cpinfo = NMO.load_dict_from_hdf5(self.extfile)

        if freqs.size != self.cpinfo['raw']['cphase'].shape[-1]:
            raise ValueError('Input frequencies do not match with dimensions of the closure phase data')
        self.f = freqs
        self.df = freqs[1] - freqs[0]

        force_expicp = False
        if 'processed' not in self.cpinfo:
            force_expicp = True
        else:
            if 'native' not in self.cpinfo['processed']:
                force_expicp = True

        self.expicp(force_action=force_expicp)

        if 'prelim' not in self.cpinfo['processed']:
            self.cpinfo['processed']['prelim'] = {}

        self.cpinfo['errinfo'] = {}

    ############################################################################

    def expicp(self, force_action=False):

        """
        ------------------------------------------------------------------------
        Compute the complex exponential of the closure phase as a masked array

        Inputs:

        force_action    [boolean] If set to False (default), the complex 
                        exponential is computed only if it has not been done so
                        already. Otherwise the computation is forced.
        ------------------------------------------------------------------------
        """

        if 'processed' not in self.cpinfo:
            self.cpinfo['processed'] = {}
            force_action = True
        if 'native' not in self.cpinfo['processed']:
            self.cpinfo['processed']['native'] = {}
            force_action = True
        if 'cphase' not in self.cpinfo['processed']['native']:
            self.cpinfo['processed']['native']['cphase'] = MA.array(self.cpinfo['raw']['cphase'].astype(NP.float64), mask=self.cpinfo['raw']['flags'])
            force_action = True
        if not force_action:
            if 'eicp' not in self.cpinfo['processed']['native']:
                self.cpinfo['processed']['native']['eicp'] = NP.exp(1j * self.cpinfo['processed']['native']['cphase'])
                self.cpinfo['processed']['native']['wts'] = MA.array(NP.logical_not(self.cpinfo['raw']['flags']).astype(NP.float), mask=self.cpinfo['raw']['flags'])
        else:
            self.cpinfo['processed']['native']['eicp'] = NP.exp(1j * self.cpinfo['processed']['native']['cphase'])
            self.cpinfo['processed']['native']['wts'] = MA.array(NP.logical_not(self.cpinfo['raw']['flags']).astype(NP.float), mask=self.cpinfo['raw']['flags'])

    ############################################################################

    def smooth_in_tbins(self, daybinsize=None, ndaybins=None, lstbinsize=None):

        """
        ------------------------------------------------------------------------
        Smooth the complex exponentials of closure phases in time bins. Both
        mean and median smoothing is produced.

        Inputs:

        daybinsize  [Nonetype or scalar] Day bin size (in days) over which mean
                    and median are estimated across different days for a fixed
                    LST bin. If set to None, it will look for value in input
                    ndaybins. If both are None, no smoothing is performed. Only
                    one of daybinsize or ndaybins must be set to non-None value.

        ndaybins    [NoneType or integer] Number of bins along day axis. Only 
                    if daybinsize is set to None. It produces bins that roughly 
                    consist of equal number of days in each bin regardless of
                    how much the days in each bin are separated from each other. 
                    If both are None, no smoothing is performed. Only one of 
                    daybinsize or ndaybins must be set to non-None value.

        lstbinsize  [NoneType or scalar] LST bin size (in seconds) over which
                    mean and median are estimated across the LST. If set to 
                    None, no smoothing is performed
        ------------------------------------------------------------------------
        """

        if (ndaybins is not None) and (daybinsize is not None):
            raise ValueError('Only one of daybinsize or ndaybins should be set')

        if (daybinsize is not None) or (ndaybins is not None):
            if daybinsize is not None:
                if not isinstance(daybinsize, (int,float)):
                    raise TypeError('Input daybinsize must be a scalar')
                dres = NP.diff(self.cpinfo['raw']['days']).min() # in days
                dextent = self.cpinfo['raw']['days'].max() - self.cpinfo['raw']['days'].min() + dres # in days
                if daybinsize > dres:
                    daybinsize = NP.clip(daybinsize, dres, dextent)
                    eps = 1e-10
                    daybins = NP.arange(self.cpinfo['raw']['days'].min(), self.cpinfo['raw']['days'].max() + dres + eps, daybinsize)
                    ndaybins = daybins.size
                    daybins = NP.concatenate((daybins, [daybins[-1]+daybinsize+eps]))
                    if ndaybins > 1:
                        daybinintervals = daybins[1:] - daybins[:-1]
                        daybincenters = daybins[:-1] + 0.5 * daybinintervals
                    else:
                        daybinintervals = NP.asarray(daybinsize).reshape(-1)
                        daybincenters = daybins[0] + 0.5 * daybinintervals
                    counts, daybin_edges, daybinnum, ri = OPS.binned_statistic(self.cpinfo['raw']['days'], statistic='count', bins=daybins)
                    counts = counts.astype(NP.int)
    
                    # if 'prelim' not in self.cpinfo['processed']:
                    #     self.cpinfo['processed']['prelim'] = {}
                    # self.cpinfo['processed']['prelim']['eicp'] = {}
                    # self.cpinfo['processed']['prelim']['cphase'] = {}
                    # self.cpinfo['processed']['prelim']['daybins'] = daybincenters
                    # self.cpinfo['processed']['prelim']['diff_dbins'] = daybinintervals
    
                    wts_daybins = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]))
                    eicp_dmean = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]), dtype=NP.complex128)
                    eicp_dmedian = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]), dtype=NP.complex128)
                    cp_drms = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]))
                    cp_dmad = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]))
                    for binnum in xrange(counts.size):
                        ind_daybin = ri[ri[binnum]:ri[binnum+1]]
                        wts_daybins[:,binnum,:,:] = NP.sum(self.cpinfo['processed']['native']['wts'][:,ind_daybin,:,:].data, axis=1)
                        eicp_dmean[:,binnum,:,:] = NP.exp(1j*NP.angle(MA.mean(self.cpinfo['processed']['native']['eicp'][:,ind_daybin,:,:], axis=1)))
                        eicp_dmedian[:,binnum,:,:] = NP.exp(1j*NP.angle(MA.median(self.cpinfo['processed']['native']['eicp'][:,ind_daybin,:,:].real, axis=1) + 1j * MA.median(self.cpinfo['processed']['native']['eicp'][:,ind_daybin,:,:].imag, axis=1)))
                        cp_drms[:,binnum,:,:] = MA.std(self.cpinfo['processed']['native']['cphase'][:,ind_daybin,:,:], axis=1).data
                        cp_dmad[:,binnum,:,:] = MA.median(NP.abs(self.cpinfo['processed']['native']['cphase'][:,ind_daybin,:,:] - NP.angle(eicp_dmedian[:,binnum,:,:][:,NP.newaxis,:,:])), axis=1).data
                    # mask = wts_daybins <= 0.0
                    # self.cpinfo['processed']['prelim']['wts'] = MA.array(wts_daybins, mask=mask)
                    # self.cpinfo['processed']['prelim']['eicp']['mean'] = MA.array(eicp_dmean, mask=mask)
                    # self.cpinfo['processed']['prelim']['eicp']['median'] = MA.array(eicp_dmedian, mask=mask)
                    # self.cpinfo['processed']['prelim']['cphase']['mean'] = MA.array(NP.angle(eicp_dmean), mask=mask)
                    # self.cpinfo['processed']['prelim']['cphase']['median'] = MA.array(NP.angle(eicp_dmedian), mask=mask)
                    # self.cpinfo['processed']['prelim']['cphase']['rms'] = MA.array(cp_drms, mask=mask)
                    # self.cpinfo['processed']['prelim']['cphase']['mad'] = MA.array(cp_dmad, mask=mask)
            else:
                if not isinstance(ndaybins, int):
                    raise TypeError('Input ndaybins must be an integer')
                if ndaybins <= 0:
                    raise ValueError('Input ndaybins must be positive')
                days_split = NP.array_split(self.cpinfo['raw']['days'], ndaybins)
                daybincenters = NP.asarray([NP.mean(days) for days in days_split])
                daybinintervals = NP.asarray([days.max()-days.min() for days in days_split])
                counts = NP.asarray([days.size for days in days_split])
    
                wts_split = NP.array_split(self.cpinfo['processed']['native']['wts'].data, ndaybins, axis=1)
                # mask_split = NP.array_split(self.cpinfo['processed']['native']['wts'].mask, ndaybins, axis=1)
                wts_daybins = NP.asarray([NP.sum(wtsitem, axis=1) for wtsitem in wts_split]) # ndaybins x nlst x ntriads x nchan
                wts_daybins = NP.moveaxis(wts_daybins, 0, 1) # nlst x ndaybins x ntriads x nchan
                mask_split = NP.array_split(self.cpinfo['processed']['native']['eicp'].mask, ndaybins, axis=1)
                eicp_split = NP.array_split(self.cpinfo['processed']['native']['eicp'].data, ndaybins, axis=1)
                eicp_dmean = MA.array([MA.mean(MA.array(eicp_split[i], mask=mask_split[i]), axis=1) for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                eicp_dmean = NP.exp(1j * NP.angle(eicp_dmean))
                eicp_dmean = NP.moveaxis(eicp_dmean, 0, 1) # nlst x ndaybins x ntriads x nchan
    
                eicp_dmedian = MA.array([MA.median(MA.array(eicp_split[i].real, mask=mask_split[i]), axis=1) + 1j * MA.median(MA.array(eicp_split[i].imag, mask=mask_split[i]), axis=1) for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                eicp_dmedian = NP.exp(1j * NP.angle(eicp_dmedian))
                eicp_dmedian = NP.moveaxis(eicp_dmedian, 0, 1) # nlst x ndaybins x ntriads x nchan
                
                cp_split = NP.array_split(self.cpinfo['processed']['native']['cphase'].data, ndaybins, axis=1)
                cp_drms = NP.array([MA.std(MA.array(cp_split[i], mask=mask_split[i]), axis=1).data for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                cp_drms = NP.moveaxis(cp_drms, 0, 1) # nlst x ndaybins x ntriads x nchan
    
                cp_dmad = NP.array([MA.median(NP.abs(cp_split[i] - NP.angle(eicp_dmedian[:,[i],:,:])), axis=1).data for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                cp_dmad = NP.moveaxis(cp_dmad, 0, 1) # nlst x ndaybins x ntriads x nchan

            if 'prelim' not in self.cpinfo['processed']:
                self.cpinfo['processed']['prelim'] = {}
            self.cpinfo['processed']['prelim']['eicp'] = {}
            self.cpinfo['processed']['prelim']['cphase'] = {}
            self.cpinfo['processed']['prelim']['daybins'] = daybincenters
            self.cpinfo['processed']['prelim']['diff_dbins'] = daybinintervals

            mask = wts_daybins <= 0.0
            self.cpinfo['processed']['prelim']['wts'] = MA.array(wts_daybins, mask=mask)
            self.cpinfo['processed']['prelim']['eicp']['mean'] = MA.array(eicp_dmean, mask=mask)
            self.cpinfo['processed']['prelim']['eicp']['median'] = MA.array(eicp_dmedian, mask=mask)
            self.cpinfo['processed']['prelim']['cphase']['mean'] = MA.array(NP.angle(eicp_dmean), mask=mask)
            self.cpinfo['processed']['prelim']['cphase']['median'] = MA.array(NP.angle(eicp_dmedian), mask=mask)
            self.cpinfo['processed']['prelim']['cphase']['rms'] = MA.array(cp_drms, mask=mask)
            self.cpinfo['processed']['prelim']['cphase']['mad'] = MA.array(cp_dmad, mask=mask)
            
        if lstbinsize is not None:
            if not isinstance(lstbinsize, (int,float)):
                raise TypeError('Input lstbinsize must be a scalar')
            rawlst = NP.degrees(NP.unwrap(NP.radians(self.cpinfo['raw']['lst'] * 15.0), discont=NP.pi, axis=0)) / 15.0 # in hours but unwrapped to have no discontinuities
            if NP.any(rawlst > 24.0):
                rawlst -= 24.0
            lstbinsize = lstbinsize / 3.6e3 # in hours
            tres = NP.diff(rawlst[:,0]).min() # in hours
            textent = rawlst[:,0].max() - rawlst[:,0].min() + tres # in hours
            if lstbinsize > tres:
                lstbinsize = NP.clip(lstbinsize, tres, textent)
                eps = 1e-10
                lstbins = NP.arange(rawlst[:,0].min(), rawlst[:,0].max() + tres + eps, lstbinsize)
                nlstbins = lstbins.size
                lstbins = NP.concatenate((lstbins, [lstbins[-1]+lstbinsize+eps]))
                if nlstbins > 1:
                    lstbinintervals = lstbins[1:] - lstbins[:-1]
                    lstbincenters = lstbins[:-1] + 0.5 * lstbinintervals
                else:
                    lstbinintervals = NP.asarray(lstbinsize).reshape(-1)
                    lstbincenters = lstbins[0] + 0.5 * lstbinintervals
                counts, lstbin_edges, lstbinnum, ri = OPS.binned_statistic(rawlst[:,0], statistic='count', bins=lstbins)
                counts = counts.astype(NP.int)

                if 'prelim' not in self.cpinfo['processed']:
                    self.cpinfo['processed']['prelim'] = {}
                self.cpinfo['processed']['prelim']['lstbins'] = lstbincenters
                self.cpinfo['processed']['prelim']['dlstbins'] = lstbinintervals

                if 'wts' not in self.cpinfo['processed']['prelim']:
                    outshape = (counts.size, self.cpinfo['processed']['native']['eicp'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3])
                else:
                    outshape = (counts.size, self.cpinfo['processed']['prelim']['wts'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3])
                wts_lstbins = NP.zeros(outshape)
                eicp_tmean = NP.zeros(outshape, dtype=NP.complex128)
                eicp_tmedian = NP.zeros(outshape, dtype=NP.complex128)
                cp_trms = NP.zeros(outshape)
                cp_tmad = NP.zeros(outshape)
                    
                for binnum in xrange(counts.size):
                    ind_lstbin = ri[ri[binnum]:ri[binnum+1]]
                    if 'wts' not in self.cpinfo['processed']['prelim']:
                        indict = self.cpinfo['processed']['native']
                    else:
                        indict = self.cpinfo['processed']['prelim']
                    wts_lstbins[binnum,:,:,:] = NP.sum(indict['wts'][ind_lstbin,:,:,:].data, axis=0)
                    if 'wts' not in self.cpinfo['processed']['prelim']:
                        eicp_tmean[binnum,:,:,:] = NP.exp(1j*NP.angle(MA.mean(indict['eicp'][ind_lstbin,:,:,:], axis=0)))
                        eicp_tmedian[binnum,:,:,:] = NP.exp(1j*NP.angle(MA.median(indict['eicp'][ind_lstbin,:,:,:].real, axis=0) + 1j * MA.median(self.cpinfo['processed']['native']['eicp'][ind_lstbin,:,:,:].imag, axis=0)))
                        cp_trms[binnum,:,:,:] = MA.std(indict['cphase'][ind_lstbin,:,:,:], axis=0).data
                        cp_tmad[binnum,:,:,:] = MA.median(NP.abs(indict['cphase'][ind_lstbin,:,:,:] - NP.angle(eicp_tmedian[binnum,:,:,:][NP.newaxis,:,:,:])), axis=0).data
                    else:
                        eicp_tmean[binnum,:,:,:] = NP.exp(1j*NP.angle(MA.mean(NP.exp(1j*indict['cphase']['mean'][ind_lstbin,:,:,:]), axis=0)))
                        eicp_tmedian[binnum,:,:,:] = NP.exp(1j*NP.angle(MA.median(NP.cos(indict['cphase']['median'][ind_lstbin,:,:,:]), axis=0) + 1j * MA.median(NP.sin(indict['cphase']['median'][ind_lstbin,:,:,:]), axis=0)))
                        cp_trms[binnum,:,:,:] = MA.std(indict['cphase']['mean'][ind_lstbin,:,:,:], axis=0).data
                        cp_tmad[binnum,:,:,:] = MA.median(NP.abs(indict['cphase']['median'][ind_lstbin,:,:,:] - NP.angle(eicp_tmedian[binnum,:,:,:][NP.newaxis,:,:,:])), axis=0).data
                        
                mask = wts_lstbins <= 0.0
                self.cpinfo['processed']['prelim']['wts'] = MA.array(wts_lstbins, mask=mask)
                if 'eicp' not in self.cpinfo['processed']['prelim']:
                    self.cpinfo['processed']['prelim']['eicp'] = {}
                if 'cphase' not in self.cpinfo['processed']['prelim']:
                    self.cpinfo['processed']['prelim']['cphase'] = {}
                self.cpinfo['processed']['prelim']['eicp']['mean'] = MA.array(eicp_tmean, mask=mask)
                self.cpinfo['processed']['prelim']['eicp']['median'] = MA.array(eicp_tmedian, mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['mean'] = MA.array(NP.angle(eicp_tmean), mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['median'] = MA.array(NP.angle(eicp_tmedian), mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['rms'] = MA.array(cp_trms, mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['mad'] = MA.array(cp_tmad, mask=mask)

    ############################################################################

    def subtract(self, cphase):

        """
        ------------------------------------------------------------------------
        Subtract complex exponential of the bispectrum phase from the current 
        instance and updates the cpinfo attribute

        Inputs:

        cphase      [masked array] Bispectrum phase array as a maked array. It 
                    must be of same size as freqs along the axis specified in 
                    input axis.

        Action:     Updates 'submodel' and 'residual' keys under attribute
                    cpinfo under key 'processed'
        ------------------------------------------------------------------------
        """

        if not isinstance(cphase, NP.ndarray):
            raise TypeError('Input cphase must be a numpy array')
        
        if not isinstance(cphase, MA.MaskedArray):
            cphase = MA.array(cphase, mask=NP.isnan(cphase))
    
        if not OPS.is_broadcastable(cphase.shape, self.cpinfo['processed']['prelim']['cphase']['median'].shape):
            raise ValueError('Input cphase has shape incompatible with that in instance attribute')
        else:
            minshape = tuple(NP.ones(self.cpinfo['processed']['prelim']['cphase']['median'].ndim - cphase.ndim, dtype=NP.int)) + cphase.shape
            cphase = cphase.reshape(minshape)
            # cphase = NP.broadcast_to(cphase, minshape)
    
        eicp = NP.exp(1j*cphase)
        
        self.cpinfo['processed']['submodel'] = {}
        self.cpinfo['processed']['submodel']['cphase'] = cphase
        self.cpinfo['processed']['submodel']['eicp'] = eicp
        self.cpinfo['processed']['residual'] = {'eicp': {}, 'cphase': {}}
        for key in ['mean', 'median']:
            eicpdiff = self.cpinfo['processed']['prelim']['eicp'][key] - eicp
            eicpratio = self.cpinfo['processed']['prelim']['eicp'][key] / eicp
            self.cpinfo['processed']['residual']['eicp'][key] = eicpdiff
            self.cpinfo['processed']['residual']['cphase'][key] = MA.array(NP.angle(eicpratio.data), mask=self.cpinfo['processed']['residual']['eicp'][key].mask)
        
    ############################################################################

    def subsample_differencing(self, daybinsize=None, ndaybins=4, lstbinsize=None):

        """
        ------------------------------------------------------------------------
        Create subsamples and differences between subsamples to evaluate noise
        properties from the data set.

        Inputs:

        daybinsize  [Nonetype or scalar] Day bin size (in days) over which mean
                    and median are estimated across different days for a fixed
                    LST bin. If set to None, it will look for value in input
                    ndaybins. If both are None, no smoothing is performed. Only
                    one of daybinsize or ndaybins must be set to non-None value.
                    Must yield greater than or equal to 4 bins

        ndaybins    [NoneType or integer] Number of bins along day axis. Only 
                    if daybinsize is set to None. It produces bins that roughly 
                    consist of equal number of days in each bin regardless of
                    how much the days in each bin are separated from each other. 
                    If both are None, no smoothing is performed. Only one of 
                    daybinsize or ndaybins must be set to non-None value. If set,
                    it must be set to greater than or equal to 4

        lstbinsize  [NoneType or scalar] LST bin size (in seconds) over which
                    mean and median are estimated across the LST. If set to 
                    None, no smoothing is performed
        ------------------------------------------------------------------------
        """

        if (ndaybins is not None) and (daybinsize is not None):
            raise ValueError('Only one of daybinsize or ndaybins should be set')

        if (daybinsize is not None) or (ndaybins is not None):
            if daybinsize is not None:
                if not isinstance(daybinsize, (int,float)):
                    raise TypeError('Input daybinsize must be a scalar')
                dres = NP.diff(self.cpinfo['raw']['days']).min() # in days
                dextent = self.cpinfo['raw']['days'].max() - self.cpinfo['raw']['days'].min() + dres # in days
                if daybinsize > dres:
                    daybinsize = NP.clip(daybinsize, dres, dextent)
                    eps = 1e-10
                    daybins = NP.arange(self.cpinfo['raw']['days'].min(), self.cpinfo['raw']['days'].max() + dres + eps, daybinsize)
                    ndaybins = daybins.size
                    daybins = NP.concatenate((daybins, [daybins[-1]+daybinsize+eps]))
                    if ndaybins >= 4:
                        daybinintervals = daybins[1:] - daybins[:-1]
                        daybincenters = daybins[:-1] + 0.5 * daybinintervals
                    else:
                        raise ValueError('Could not find at least 4 bins along repeating days. Adjust binning interval.')
                    counts, daybin_edges, daybinnum, ri = OPS.binned_statistic(self.cpinfo['raw']['days'], statistic='count', bins=daybins)
                    counts = counts.astype(NP.int)
    
                    wts_daybins = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]))
                    eicp_dmean = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]), dtype=NP.complex128)
                    eicp_dmedian = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]), dtype=NP.complex128)
                    cp_drms = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]))
                    cp_dmad = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], counts.size, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3]))
                    for binnum in xrange(counts.size):
                        ind_daybin = ri[ri[binnum]:ri[binnum+1]]
                        wts_daybins[:,binnum,:,:] = NP.sum(self.cpinfo['processed']['native']['wts'][:,ind_daybin,:,:].data, axis=1)
                        eicp_dmean[:,binnum,:,:] = NP.exp(1j*NP.angle(MA.mean(self.cpinfo['processed']['native']['eicp'][:,ind_daybin,:,:], axis=1)))
                        eicp_dmedian[:,binnum,:,:] = NP.exp(1j*NP.angle(MA.median(self.cpinfo['processed']['native']['eicp'][:,ind_daybin,:,:].real, axis=1) + 1j * MA.median(self.cpinfo['processed']['native']['eicp'][:,ind_daybin,:,:].imag, axis=1)))
                        cp_drms[:,binnum,:,:] = MA.std(self.cpinfo['processed']['native']['cphase'][:,ind_daybin,:,:], axis=1).data
                        cp_dmad[:,binnum,:,:] = MA.median(NP.abs(self.cpinfo['processed']['native']['cphase'][:,ind_daybin,:,:] - NP.angle(eicp_dmedian[:,binnum,:,:][:,NP.newaxis,:,:])), axis=1).data
            else:
                if not isinstance(ndaybins, int):
                    raise TypeError('Input ndaybins must be an integer')
                if ndaybins < 4:
                    raise ValueError('Input ndaybins must be greater than or equal to 4')
                days_split = NP.array_split(self.cpinfo['raw']['days'], ndaybins)
                daybincenters = NP.asarray([NP.mean(days) for days in days_split])
                daybinintervals = NP.asarray([days.max()-days.min() for days in days_split])
                counts = NP.asarray([days.size for days in days_split])
    
                wts_split = NP.array_split(self.cpinfo['processed']['native']['wts'].data, ndaybins, axis=1)
                # mask_split = NP.array_split(self.cpinfo['processed']['native']['wts'].mask, ndaybins, axis=1)
                wts_daybins = NP.asarray([NP.sum(wtsitem, axis=1) for wtsitem in wts_split]) # ndaybins x nlst x ntriads x nchan
                wts_daybins = NP.moveaxis(wts_daybins, 0, 1) # nlst x ndaybins x ntriads x nchan
                mask_split = NP.array_split(self.cpinfo['processed']['native']['eicp'].mask, ndaybins, axis=1)
                eicp_split = NP.array_split(self.cpinfo['processed']['native']['eicp'].data, ndaybins, axis=1)
                eicp_dmean = MA.array([MA.mean(MA.array(eicp_split[i], mask=mask_split[i]), axis=1) for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                eicp_dmean = NP.exp(1j * NP.angle(eicp_dmean))
                eicp_dmean = NP.moveaxis(eicp_dmean, 0, 1) # nlst x ndaybins x ntriads x nchan
    
                eicp_dmedian = MA.array([MA.median(MA.array(eicp_split[i].real, mask=mask_split[i]), axis=1) + 1j * MA.median(MA.array(eicp_split[i].imag, mask=mask_split[i]), axis=1) for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                eicp_dmedian = NP.exp(1j * NP.angle(eicp_dmedian))
                eicp_dmedian = NP.moveaxis(eicp_dmedian, 0, 1) # nlst x ndaybins x ntriads x nchan
                
                cp_split = NP.array_split(self.cpinfo['processed']['native']['cphase'].data, ndaybins, axis=1)
                cp_drms = NP.array([MA.std(MA.array(cp_split[i], mask=mask_split[i]), axis=1).data for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                cp_drms = NP.moveaxis(cp_drms, 0, 1) # nlst x ndaybins x ntriads x nchan
    
                cp_dmad = NP.array([MA.median(NP.abs(cp_split[i] - NP.angle(eicp_dmedian[:,[i],:,:])), axis=1).data for i in range(daybincenters.size)]) # ndaybins x nlst x ntriads x nchan
                cp_dmad = NP.moveaxis(cp_dmad, 0, 1) # nlst x ndaybins x ntriads x nchan

        mask = wts_daybins <= 0.0
        wts_daybins = MA.array(wts_daybins, mask=mask)
        cp_dmean = MA.array(NP.angle(eicp_dmean), mask=mask)
        cp_dmedian = MA.array(NP.angle(eicp_dmedian), mask=mask)
        self.cpinfo['errinfo']['daybins'] = daybincenters
        self.cpinfo['errinfo']['diff_dbins'] = daybinintervals
        self.cpinfo['errinfo']['wts'] = {'{0}'.format(ind): None for ind in range(2)}
        self.cpinfo['errinfo']['eicp_diff'] = {'{0}'.format(ind): {} for ind in range(2)}
        if lstbinsize is not None:
            if not isinstance(lstbinsize, (int,float)):
                raise TypeError('Input lstbinsize must be a scalar')
            rawlst = NP.degrees(NP.unwrap(NP.radians(self.cpinfo['raw']['lst'] * 15.0), discont=NP.pi, axis=0)) / 15.0 # in hours but unwrapped to have no discontinuities
            lstbinsize = lstbinsize / 3.6e3 # in hours
            tres = NP.diff(rawlst[:,0]).min() # in hours
            textent = rawlst[:,0].max() - rawlst[:,0].min() + tres # in hours
            if lstbinsize > tres:
                lstbinsize = NP.clip(lstbinsize, tres, textent)
                eps = 1e-10
                lstbins = NP.arange(rawlst[:,0].min(), rawlst[:,0].max() + tres + eps, lstbinsize)
                nlstbins = lstbins.size
                lstbins = NP.concatenate((lstbins, [lstbins[-1]+lstbinsize+eps]))
                if nlstbins > 1:
                    lstbinintervals = lstbins[1:] - lstbins[:-1]
                    lstbincenters = lstbins[:-1] + 0.5 * lstbinintervals
                else:
                    lstbinintervals = NP.asarray(lstbinsize).reshape(-1)
                    lstbincenters = lstbins[0] + 0.5 * lstbinintervals
                counts, lstbin_edges, lstbinnum, ri = OPS.binned_statistic(rawlst[:,0], statistic='count', bins=lstbins)
                counts = counts.astype(NP.int)
                self.cpinfo['errinfo']['lstbins'] = lstbincenters
                self.cpinfo['errinfo']['dlstbins'] = lstbinintervals
                outshape = (counts.size, wts_daybins.shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3])
                wts_lstbins = NP.zeros(outshape)
                eicp_tmean = NP.zeros(outshape, dtype=NP.complex128)
                eicp_tmedian = NP.zeros(outshape, dtype=NP.complex128)
                cp_trms = NP.zeros(outshape)
                cp_tmad = NP.zeros(outshape)
                for binnum in xrange(counts.size):
                    ind_lstbin = ri[ri[binnum]:ri[binnum+1]]
                    wts_lstbins[binnum,:,:,:] = NP.sum(wts_daybins[ind_lstbin,:,:,:].data, axis=0)
                    eicp_tmean[binnum,:,:,:] = NP.exp(1j*NP.angle(MA.mean(NP.exp(1j*cp_dmean[ind_lstbin,:,:,:]), axis=0)))
                    eicp_tmedian[binnum,:,:,:] = NP.exp(1j*NP.angle(MA.median(NP.cos(cp_dmedian[ind_lstbin,:,:,:]), axis=0) + 1j * MA.median(NP.sin(cp_dmedian[ind_lstbin,:,:,:]), axis=0)))
                mask = wts_lstbins <= 0.0
                wts_lstbins = MA.array(wts_lstbins, mask=mask)
                eicp_tmean = MA.array(eicp_tmean, mask=mask)
                eicp_tmedian = MA.array(eicp_tmedian, mask=mask)

        ncomb = NP.sum(NP.asarray([(ndaybins-i-1)*(ndaybins-i-2)*(ndaybins-i-3)/2 for i in range(ndaybins-3)])).astype(int)
        diff_outshape = (counts.size, ncomb, self.cpinfo['processed']['native']['eicp'].shape[2], self.cpinfo['processed']['native']['eicp'].shape[3])
        for diffind in range(2):
            self.cpinfo['errinfo']['eicp_diff']['{0}'.format(diffind)]['mean'] = MA.empty(diff_outshape, dtype=NP.complex)
            self.cpinfo['errinfo']['eicp_diff']['{0}'.format(diffind)]['median'] = MA.empty(diff_outshape, dtype=NP.complex)
            self.cpinfo['errinfo']['wts']['{0}'.format(diffind)] = MA.empty(diff_outshape, dtype=NP.float)
        ind = -1
        self.cpinfo['errinfo']['list_of_pair_of_pairs'] = []
        list_of_pair_of_pairs = []
        for i in range(ndaybins-1):
            for j in range(i+1,ndaybins):
                for k in range(ndaybins-1):
                    if (k != i) and (k != j):
                        for m in range(k+1,ndaybins):
                            if (m != i) and (m != j):
                                pair_of_pairs = [set([i,j]), set([k,m])]
                                if (pair_of_pairs not in list_of_pair_of_pairs) and (pair_of_pairs[::-1] not in list_of_pair_of_pairs):
                                    ind += 1
                                    list_of_pair_of_pairs += [copy.deepcopy(pair_of_pairs)]
                                    self.cpinfo['errinfo']['list_of_pair_of_pairs'] += [[i,j,k,m]]
                                    for stat in ['mean', 'median']:
                                        if stat == 'mean':
                                            self.cpinfo['errinfo']['eicp_diff']['0'][stat][:,ind,:,:] = MA.array(0.5 * (eicp_tmean[:,j,:,:].data - eicp_tmean[:,i,:,:].data), mask=NP.logical_or(eicp_tmean[:,j,:,:].mask, eicp_tmean[:,i,:,:].mask))
                                            self.cpinfo['errinfo']['eicp_diff']['1'][stat][:,ind,:,:] = MA.array(0.5 * (eicp_tmean[:,m,:,:].data - eicp_tmean[:,k,:,:].data), mask=NP.logical_or(eicp_tmean[:,m,:,:].mask, eicp_tmean[:,k,:,:].mask))
                                            self.cpinfo['errinfo']['wts']['0'][:,ind,:,:] = MA.array(NP.sqrt(wts_lstbins[:,j,:,:].data**2 + wts_lstbins[:,i,:,:].data**2), mask=NP.logical_or(wts_lstbins[:,j,:,:].mask, wts_lstbins[:,i,:,:].mask))
                                            self.cpinfo['errinfo']['wts']['1'][:,ind,:,:] = MA.array(NP.sqrt(wts_lstbins[:,m,:,:].data**2 + wts_lstbins[:,k,:,:].data**2), mask=NP.logical_or(wts_lstbins[:,m,:,:].mask, wts_lstbins[:,k,:,:].mask))
                                            # self.cpinfo['errinfo']['eicp_diff']['0'][stat][:,ind,:,:] = 0.5 * (eicp_tmean[:,j,:,:] - eicp_tmean[:,i,:,:])
                                            # self.cpinfo['errinfo']['eicp_diff']['1'][stat][:,ind,:,:] = 0.5 * (eicp_tmean[:,m,:,:] - eicp_tmean[:,k,:,:])
                                            # self.cpinfo['errinfo']['wts']['0'][:,ind,:,:] = NP.sqrt(wts_lstbins[:,j,:,:]**2 + wts_lstbins[:,i,:,:]**2)
                                            # self.cpinfo['errinfo']['wts']['1'][:,ind,:,:] = NP.sqrt(wts_lstbins[:,m,:,:]**2 + wts_lstbins[:,k,:,:]**2)
                                        else:
                                            self.cpinfo['errinfo']['eicp_diff']['0'][stat][:,ind,:,:] = MA.array(0.5 * (eicp_tmedian[:,j,:,:].data - eicp_tmedian[:,i,:,:].data), mask=NP.logical_or(eicp_tmedian[:,j,:,:].mask, eicp_tmedian[:,i,:,:].mask))
                                            self.cpinfo['errinfo']['eicp_diff']['1'][stat][:,ind,:,:] = MA.array(0.5 * (eicp_tmedian[:,m,:,:].data - eicp_tmedian[:,k,:,:].data), mask=NP.logical_or(eicp_tmedian[:,m,:,:].mask, eicp_tmedian[:,k,:,:].mask))
                                            # self.cpinfo['errinfo']['eicp_diff']['0'][stat][:,ind,:,:] = 0.5 * (eicp_tmedian[:,j,:,:] - eicp_tmedian[:,i,:,:])
                                            # self.cpinfo['errinfo']['eicp_diff']['1'][stat][:,ind,:,:] = 0.5 * (eicp_tmedian[:,m,:,:] - eicp_tmedian[:,k,:,:])
                                        mask0 = self.cpinfo['errinfo']['wts']['0'] <= 0.0
                                        mask1 = self.cpinfo['errinfo']['wts']['1'] <= 0.0
                                        self.cpinfo['errinfo']['eicp_diff']['0'][stat] = MA.array(self.cpinfo['errinfo']['eicp_diff']['0'][stat], mask=mask0)
                                        self.cpinfo['errinfo']['eicp_diff']['1'][stat] = MA.array(self.cpinfo['errinfo']['eicp_diff']['1'][stat], mask=mask1)
                                        self.cpinfo['errinfo']['wts']['0'] = MA.array(self.cpinfo['errinfo']['wts']['0'], mask=mask0)
                                        self.cpinfo['errinfo']['wts']['1'] = MA.array(self.cpinfo['errinfo']['wts']['1'], mask=mask1)

    ############################################################################

    def save(self, outfile=None):

        """
        ------------------------------------------------------------------------
        Save contents of attribute cpinfo in external HDF5 file

        Inputs:

        outfile     [NoneType or string] Output file (HDF5) to save contents to.
                    If set to None (default), it will be saved in the file 
                    pointed to by the extfile attribute of class ClosurePhase
        ------------------------------------------------------------------------
        """
        
        if outfile is None:
            outfile = self.extfile
        
        NMO.save_dict_to_hdf5(self.cpinfo, outfile, compressinfo={'compress_fmt': 'gzip', 'compress_opts': 9})
        
################################################################################

class ClosurePhaseDelaySpectrum(object):

    """
    ----------------------------------------------------------------------------
    Class to hold and operate on Closure Phase information.

    It has the following attributes and member functions.

    Attributes:

    cPhase          [instance of class ClosurePhase] Instance of class
                    ClosurePhase

    f               [numpy array] Frequencies (in Hz) in closure phase spectra

    df              [float] Frequency resolution (in Hz) in closure phase 
                    spectra

    cPhaseDS        [dictionary] Possibly oversampled Closure Phase Delay 
                    Spectrum information.

    cPhaseDS_resampled
                    [dictionary] Resampled Closure Phase Delay Spectrum 
                    information.

    Member functions:

    __init__()      Initialize instance of class ClosurePhaseDelaySpectrum

    FT()            Fourier transform of complex closure phase spectra mapping 
                    from frequency axis to delay axis.

    subset()        Return triad and time indices to select a subset of 
                    processed data

    compute_power_spectrum()
                    Compute power spectrum of closure phase data. It is in units 
                    of Mpc/h. 

    rescale_power_spectrum()
                    Rescale power spectrum to dimensional quantity by converting 
                    the ratio given visibility amplitude information

    average_rescaled_power_spectrum()
                    Average the rescaled power spectrum with physical units 
                    along certain axes with inverse variance or regular 
                    averaging

    beam3Dvol()     Compute three-dimensional volume of the antenna power 
                    pattern along two transverse axes and one LOS axis. 
    ----------------------------------------------------------------------------
    """
    
    def __init__(self, cPhase):

        """
        ------------------------------------------------------------------------
        Initialize instance of class ClosurePhaseDelaySpectrum

        Inputs:

        cPhase      [class ClosurePhase] Instance of class ClosurePhase
        ------------------------------------------------------------------------
        """

        if not isinstance(cPhase, ClosurePhase):
            raise TypeError('Input cPhase must be an instance of class ClosurePhase')
        self.cPhase = cPhase
        self.f = self.cPhase.f
        self.df = self.cPhase.df
        self.cPhaseDS = None
        self.cPhaseDS_resampled = None

    ############################################################################

    def FT(self, bw_eff, freq_center=None, shape=None, fftpow=None, pad=None,
           datapool='prelim', visscaleinfo=None, method='fft', resample=True,
           apply_flags=True):

        """
        ------------------------------------------------------------------------
        Fourier transform of complex closure phase spectra mapping from 
        frequency axis to delay axis.

        Inputs:

        bw_eff      [scalar or numpy array] effective bandwidths (in Hz) on the 
                    selected frequency windows for subband delay transform of 
                    closure phases. If a scalar value is provided, the same 
                    will be applied to all frequency windows

        freq_center [scalar, list or numpy array] frequency centers (in Hz) of 
                    the selected frequency windows for subband delay transform 
                    of closure phases. The value can be a scalar, list or numpy 
                    array. If a scalar is provided, the same will be applied to 
                    all frequency windows. Default=None uses the center 
                    frequency from the class attribute named channels

        shape       [string] frequency window shape for subband delay transform 
                    of closure phases. Accepted values for the string are 
                    'rect' or 'RECT' (for rectangular), 'bnw' and 'BNW' (for 
                    Blackman-Nuttall), and 'bhw' or 'BHW' (for 
                    Blackman-Harris). Default=None sets it to 'rect' 
                    (rectangular window)

        fftpow      [scalar] the power to which the FFT of the window will be 
                    raised. The value must be a positive scalar. Default = 1.0

        pad         [scalar] padding fraction relative to the number of 
                    frequency channels for closure phases. Value must be a 
                    non-negative scalar. For e.g., a pad of 1.0 pads the 
                    frequency axis with zeros of the same width as the number 
                    of channels. After the delay transform, the transformed 
                    closure phases are downsampled by a factor of 1+pad. If a 
                    negative value is specified, delay transform will be 
                    performed with no padding. Default=None sets to padding 
                    factor to 1.0

        datapool    [string] Specifies which data set is to be Fourier 
                    transformed

        visscaleinfo
                    [dictionary] Dictionary containing reference visibilities
                    based on which the closure phases will be scaled to units
                    of visibilities. It contains the following keys and values:
                    'vis'   [numpy array or instance of class 
                            InterferometerArray] Reference visibilities from the 
                            baselines that form the triad. It can be an instance
                            of class RI.InterferometerArray or a numpy array. 
                            If an instance of class InterferometerArray, the 
                            baseline triplet must be set in key 'bltriplet' 
                            and value in key 'lst' will be ignored. If the 
                            value under this key 'vis' is set to a numpy array, 
                            it must be of shape (nbl=3, nlst_vis, nchan). In 
                            this case the value under key 'bltriplet' will be
                            ignored. The nearest LST will be looked up and 
                            applied after smoothing along LST based on the 
                            smoothing parameter 'smooth'
                    'bltriplet'
                            [Numpy array] Will be used in searching for matches
                            to these three baseline vectors if the value under
                            key 'vis' is set to an instance of class
                            InterferometerArray. However, if value under key
                            'vis' is a numpy array, this key 'bltriplet' will
                            be ignored. 
                    'lst'   [numpy array] Reference LST (in hours). It is of
                            shape (nlst_vis,). It will be used only if value 
                            under key 'vis' is a numpy array, otherwise it will
                            be ignored and read from the instance of class
                            InterferometerArray passed under key 'vis'. If the
                            specified LST range does not cover the data LST
                            range, those LST will contain NaN in the delay
                            spectrum
                    'smoothinfo'
                            [dictionary] Dictionary specifying smoothing and/or 
                            interpolation parameters. It has the following keys
                            and values:
                            'op_type'       [string] Specifies the interpolating 
                                            operation. Must be specified (no 
                                            default). Accepted values are 
                                            'interp1d' (scipy.interpolate), 
                                            'median' (skimage.filters), 'tophat' 
                                            (astropy.convolution) and 'gaussian' 
                                            (astropy.convolution)
                            'interp_kind'   [string (optional)] Specifies the 
                                            interpolation kind (if 'op_type' is 
                                            set to 'interp1d'). For accepted 
                                            values, see 
                                            scipy.interpolate.interp1d()
                            'window_size'   [integer (optional)] Specifies the 
                                            size of the interpolating/smoothing 
                                            kernel. Only applies when 'op_type' 
                                            is set to 'median', 'tophat' or 
                                            'gaussian' The kernel is a tophat 
                                            function when 'op_type' is set to 
                                            'median' or 'tophat'. If refers to 
                                            FWHM when 'op_type' is set to 
                                            'gaussian'
                            
        resample    [boolean] If set to True (default), resample the delay 
                    spectrum axis to independent samples along delay axis. If
                    set to False, return the results as is even if they may be
                    be oversampled and not all samples may be independent

        method      [string] Specifies the Fourier transform method to be used.
                    Accepted values are 'fft' (default) for FFT and 'nufft' for 
                    non-uniform FFT

        apply_flags [boolean] If set to True (default), weights determined from
                    flags will be applied. If False, no weights from flagging 
                    will be applied, and thus even flagged data will be included

        Outputs:

        A dictionary that contains the oversampled (if resample=False) or 
        resampled (if resample=True) delay spectrum information. It has the 
        following keys and values:
        'freq_center'   [numpy array] contains the center frequencies 
                        (in Hz) of the frequency subbands of the subband
                        delay spectra. It is of size n_win. It is roughly 
                        equivalent to redshift(s)
        'freq_wts'      [numpy array] Contains frequency weights applied 
                        on each frequency sub-band during the subband delay 
                        transform. It is of size n_win x nchan. 
        'bw_eff'        [numpy array] contains the effective bandwidths 
                        (in Hz) of the subbands being delay transformed. It
                        is of size n_win. It is roughly equivalent to width 
                        in redshift or along line-of-sight
        'shape'         [string] shape of the window function applied. 
                        Accepted values are 'rect' (rectangular), 'bhw'
                        (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
        'fftpow'        [scalar] the power to which the FFT of the window was 
                        raised. The value is be a positive scalar with 
                        default = 1.0
        'npad'          [scalar] Numbber of zero-padded channels before
                        performing the subband delay transform. 
        'lags'          [numpy array] lags of the subband delay spectra 
                        after padding in frequency during the transform. It
                        is of size nlags=nchan+npad if resample=True, where 
                        npad is the number of frequency channels padded 
                        specified under the key 'npad'. If resample=False, 
                        nlags = number of delays after resampling only 
                        independent delays. The lags roughly correspond to 
                        k_parallel.
        'lag_kernel'    [numpy array] delay transform of the frequency 
                        weights under the key 'freq_wts'. It is of size
                        n_win x nlst x ndays x ntriads x nlags. 
                        nlags=nchan+npad if resample=True, where npad is the 
                        number of frequency channels padded specified under 
                        the key 'npad'. If resample=False, nlags = number of 
                        delays after resampling only independent delays. 
        'lag_corr_length' 
                        [numpy array] It is the correlation timescale (in 
                        pixels) of the subband delay spectra. It is 
                        proportional to inverse of effective bandwidth. It
                        is of size n_win. The unit size of a pixel is 
                        determined by the difference between adjacent pixels 
                        in lags under key 'lags' which in turn is 
                        effectively inverse of the effective bandwidth of 
                        the subband specified in bw_eff
        'whole'         [dictionary] Delay spectrum results corresponding to 
                        bispectrum phase in 'prelim' key of attribute cpinfo. 
                        Contains the following keys and values:
                        'dspec' [dictionary] Contains the following keys and 
                                values:
                                'twts'  [numpy array] Weights from time-based
                                        flags that went into time-averaging.
                                        Shape=(nlst,ndays,ntriads,nchan)
                                'mean'  [numpy array] Delay spectrum of closure
                                        phases based on their mean across time
                                        intervals. 
                                        Shape=(nspw,nlst,ndays,ntriads,nlags)
                                'median'
                                        [numpy array] Delay spectrum of closure
                                        phases based on their median across time
                                        intervals. 
                                        Shape=(nspw,nlst,ndays,ntriads,nlags)
        'submodel'      [dictionary] Delay spectrum results corresponding to 
                        bispectrum phase in 'submodel' key of attribute cpinfo. 
                        Contains the following keys and values:
                        'dspec' [numpy array] Delay spectrum of closure phases 
                                Shape=(nspw,nlst,ndays,ntriads,nlags)
        'residual'      [dictionary] Delay spectrum results corresponding to 
                        bispectrum phase in 'residual' key of attribute cpinfo
                        after subtracting 'submodel' bispectrum phase from that
                        of 'prelim'. It contains the following keys and values:
                        'dspec' [dictionary] Contains the following keys and 
                                values:
                                'twts'  [numpy array] Weights from time-based
                                        flags that went into time-averaging.
                                        Shape=(nlst,ndays,ntriads,nchan)
                                'mean'  [numpy array] Delay spectrum of closure
                                        phases based on their mean across time
                                        intervals. 
                                        Shape=(nspw,nlst,ndays,ntriads,nlags)
                                'median'
                                        [numpy array] Delay spectrum of closure
                                        phases based on their median across time
                                        intervals. 
                                        Shape=(nspw,nlst,ndays,ntriads,nlags)
        'errinfo'       [dictionary] It has two keys 'dspec0' and 'dspec1' each
                        of which are dictionaries with the following keys and
                        values:
                        'twts'  [numpy array] Weights for the subsample 
                                difference. It is of shape (nlst, ndays, 
                                ntriads, nchan)
                        'mean'  [numpy array] Delay spectrum of the 
                                subsample difference obtained by using the 
                                mean statistic. It is of shape (nspw, nlst, 
                                ndays, ntriads, nlags)
                        'median'
                                [numpy array] Delay spectrum of the subsample 
                                difference obtained by using the median 
                                statistic. It is of shape (nspw, nlst, ndays, 
                                ntriads, nlags)
                      
        ------------------------------------------------------------------------
        """
        
        try:
            bw_eff
        except NameError:
            raise NameError('Effective bandwidth must be specified')
        else:
            if not isinstance(bw_eff, (int, float, list, NP.ndarray)):
                raise TypeError('Value of effective bandwidth must be a scalar, list or numpy array')
            bw_eff = NP.asarray(bw_eff).reshape(-1)
            if NP.any(bw_eff <= 0.0):
                raise ValueError('All values in effective bandwidth must be strictly positive')
        if freq_center is None:
            freq_center = NP.asarray(self.f[self.f.size/2]).reshape(-1)
        elif isinstance(freq_center, (int, float, list, NP.ndarray)):
            freq_center = NP.asarray(freq_center).reshape(-1)
            if NP.any((freq_center <= self.f.min()) | (freq_center >= self.f.max())):
                raise ValueError('Value(s) of frequency center(s) must lie strictly inside the observing band')
        else:
            raise TypeError('Values(s) of frequency center must be scalar, list or numpy array')

        if (bw_eff.size == 1) and (freq_center.size > 1):
            bw_eff = NP.repeat(bw_eff, freq_center.size)
        elif (bw_eff.size > 1) and (freq_center.size == 1):
            freq_center = NP.repeat(freq_center, bw_eff.size)
        elif bw_eff.size != freq_center.size:
            raise ValueError('Effective bandwidth(s) and frequency center(s) must have same number of elements')
            
        if shape is not None:
            if not isinstance(shape, str):
                raise TypeError('Window shape must be a string')
            if shape not in ['rect', 'bhw', 'bnw', 'RECT', 'BHW', 'BNW']:
                raise ValueError('Invalid value for window shape specified.')
        else:
            shape = 'rect'

        if fftpow is None:
            fftpow = 1.0
        else:
            if not isinstance(fftpow, (int, float)):
                raise TypeError('Power to raise window FFT by must be a scalar value.')
            if fftpow < 0.0:
                raise ValueError('Power for raising FFT of window by must be positive.')

        if pad is None:
            pad = 1.0
        else:
            if not isinstance(pad, (int, float)):
                raise TypeError('pad fraction must be a scalar value.')
            if pad < 0.0:
                pad = 0.0
                if verbose:
                    print('\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).')

        if not isinstance(datapool, str):
            raise TypeError('Input datapool must be a string')

        if datapool.lower() not in ['prelim']:
            raise ValueError('Specified datapool not supported')

        if visscaleinfo is not None:
            if not isinstance(visscaleinfo, dict):
                raise TypeError('Input visscaleinfo must be a dictionary')
            if 'vis' not in visscaleinfo:
                raise KeyError('Input visscaleinfo does not contain key "vis"')
            if not isinstance(visscaleinfo['vis'], RI.InterferometerArray):
                if 'lst' not in visscaleinfo:
                    raise KeyError('Input visscaleinfo does not contain key "lst"')
                lst_vis = visscaleinfo['lst'] * 15.0
                if not isinstance(visscaleinfo['vis'], (NP.ndarray,MA.MaskedArray)):
                    raise TypeError('Input visibilities must be a numpy or a masked array')
                if not isinstance(visscaleinfo['vis'], MA.MaskedArray):
                    visscaleinfo['vis'] = MA.array(visscaleinfo['vis'], mask=NP.isnan(visscaleinfo['vis']))
                vistriad = MA.copy(visscaleinfo['vis'])
            else:
                if 'bltriplet' not in visscaleinfo:
                    raise KeyError('Input dictionary visscaleinfo does not contain key "bltriplet"')
                blind, blrefind, dbl = LKP.find_1NN(visscaleinfo['vis'].baselines, visscaleinfo['bltriplet'], distance_ULIM=0.2, remove_oob=True)
                if blrefind.size != 3:
                    blind_missing = NP.setdiff1d(NP.arange(3), blind, assume_unique=True)
                    blind_next, blrefind_next, dbl_next = LKP.find_1NN(visscaleinfo['vis'].baselines, -1*visscaleinfo['bltriplet'][blind_missing,:], distance_ULIM=0.2, remove_oob=True)
                    if blind_next.size + blind.size != 3:
                        raise ValueError('Exactly three baselines were not found in the reference baselines')
                    else:
                        blind = NP.append(blind, blind_missing[blind_next])
                        blrefind = NP.append(blrefind, blrefind_next)
                else:
                    blind_missing = []
                    
                vistriad = NP.transpose(visscaleinfo['vis'].skyvis_freq[blrefind,:,:], (0,2,1))
                if len(blind_missing) > 0:
                    vistriad[-blrefind_next.size:,:,:] = vistriad[-blrefind_next.size:,:,:].conj()
                vistriad = MA.array(vistriad, mask=NP.isnan(vistriad))
                lst_vis = visscaleinfo['vis'].lst
                
            viswts = MA.array(NP.ones_like(vistriad.data), mask=vistriad.mask, dtype=NP.float)
            lst_out = self.cPhase.cpinfo['processed']['prelim']['lstbins'] * 15.0
            vis_ref, wts_ref = OPS.interpolate_masked_array_1D(vistriad, viswts, 1, visscaleinfo['smoothinfo'], inploc=lst_vis, outloc=lst_out)

        if not isinstance(method, str):
            raise TypeError('Input method must be a string')

        if method.lower() not in ['fft', 'nufft']:
            raise ValueError('Specified FFT method not supported')

        if not isinstance(apply_flags, bool):
            raise TypeError('Input apply_flags must be boolean')

        flagwts = 1.0
        visscale = 1.0

        if datapool.lower() == 'prelim':
            if method.lower() == 'fft':
                freq_wts = NP.empty((bw_eff.size, self.f.size), dtype=NP.float_) # nspw x nchan
                frac_width = DSP.window_N2width(n_window=None, shape=shape, fftpow=fftpow, area_normalize=False, power_normalize=True)
                window_loss_factor = 1 / frac_width
                n_window = NP.round(window_loss_factor * bw_eff / self.df).astype(NP.int)
                ind_freq_center, ind_channels, dfrequency = LKP.find_1NN(self.f.reshape(-1,1), freq_center.reshape(-1,1), distance_ULIM=0.51*self.df, remove_oob=True)
                sortind = NP.argsort(ind_channels)
                ind_freq_center = ind_freq_center[sortind]
                ind_channels = ind_channels[sortind]
                dfrequency = dfrequency[sortind]
                n_window = n_window[sortind]
        
                for i,ind_chan in enumerate(ind_channels):
                    window = NP.sqrt(frac_width * n_window[i]) * DSP.window_fftpow(n_window[i], shape=shape, fftpow=fftpow, centering=True, peak=None, area_normalize=False, power_normalize=True)
                    window_chans = self.f[ind_chan] + self.df * (NP.arange(n_window[i]) - int(n_window[i]/2))
                    ind_window_chans, ind_chans, dfreq = LKP.find_1NN(self.f.reshape(-1,1), window_chans.reshape(-1,1), distance_ULIM=0.51*self.df, remove_oob=True)
                    sind = NP.argsort(ind_window_chans)
                    ind_window_chans = ind_window_chans[sind]
                    ind_chans = ind_chans[sind]
                    dfreq = dfreq[sind]
                    window = window[ind_window_chans]
                    window = NP.pad(window, ((ind_chans.min(), self.f.size-1-ind_chans.max())), mode='constant', constant_values=((0.0,0.0)))
                    freq_wts[i,:] = window
        
                npad = int(self.f.size * pad)
                lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=True)
                result = {'freq_center': freq_center, 'shape': shape, 'freq_wts': freq_wts, 'bw_eff': bw_eff, 'fftpow': fftpow, 'npad': npad, 'lags': lags, 'lag_corr_length': self.f.size / NP.sum(freq_wts, axis=-1), 'whole': {'dspec': {'twts': self.cPhase.cpinfo['processed'][datapool]['wts']}}, 'residual': {'dspec': {'twts': self.cPhase.cpinfo['processed'][datapool]['wts']}}, 'errinfo': {'dspec0': {'twts': self.cPhase.cpinfo['errinfo']['wts']['0']}, 'dspec1': {'twts': self.cPhase.cpinfo['errinfo']['wts']['1']}}, 'submodel': {}}
    
                if visscaleinfo is not None:
                    visscale = NP.nansum(NP.transpose(vis_ref[NP.newaxis,NP.newaxis,:,:,:], axes=(0,3,1,2,4)) * freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:], axis=-1, keepdims=True) / NP.nansum(freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:], axis=-1, keepdims=True) # nspw x nlst x (ndays=1) x (nbl=3) x (nchan=1)
                    visscale = NP.sqrt(1.0/NP.nansum(1/NP.abs(visscale)**2, axis=-2, keepdims=True)) # nspw x nlst x (ndays=1) x (ntriads=1) x (nchan=1)

                for dpool in ['errinfo', 'prelim', 'submodel', 'residual']:
                    if dpool.lower() == 'errinfo':
                        for diffind in range(2):
                            if apply_flags:
                                flagwts = NP.copy(self.cPhase.cpinfo['errinfo']['wts']['{0}'.format(diffind)].data)
                                flagwts = flagwts[NP.newaxis,...] # nlst x ndays x ntriads x nchan --> (nspw=1) x nlst x ndays x ntriads x nchan
                                flagwts = 1.0 * flagwts / NP.mean(flagwts, axis=-1, keepdims=True) # (nspw=1) x nlst x ndays x ntriads x nchan
                            for stat in self.cPhase.cpinfo[dpool]['eicp_diff']['{0}'.format(diffind)]:
                                eicp = NP.copy(self.cPhase.cpinfo[dpool]['eicp_diff']['{0}'.format(diffind)][stat].data) # Minimum shape as stored
                                # eicp = NP.copy(self.cPhase.cpinfo[dpool]['eicp_diff']['{0}'.format(diffind)][stat].filled(0.0)) # Minimum shape as stored
                                eicp = NP.broadcast_to(eicp, self.cPhase.cpinfo[dpool]['eicp_diff']['{0}'.format(diffind)][stat].shape) # Broadcast to final shape
                                eicp = eicp[NP.newaxis,...] # nlst x ndayscomb x ntriads x nchan --> (nspw=1) x nlst x ndayscomb x ntriads x nchan
                                ndim_padtuple = [(0,0)]*(eicp.ndim-1) + [(0,npad)] # [(0,0), (0,0), (0,0), (0,0), (0,npad)]
                                result[dpool]['dspec{0}'.format(diffind)][stat] = DSP.FT1D(NP.pad(eicp*flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:]*visscale.filled(NP.nan), ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                    else:
                        if dpool in self.cPhase.cpinfo['processed']:
                            if apply_flags:
                                flagwts = NP.copy(self.cPhase.cpinfo['processed'][datapool]['wts'].data)
                                flagwts = flagwts[NP.newaxis,...] # nlst x ndays x ntriads x nchan --> (nspw=1) x nlst x ndays x ntriads x nchan
                                flagwts = 1.0 * flagwts / NP.mean(flagwts, axis=-1, keepdims=True) # (nspw=1) x nlst x ndays x ntriads x nchan
                        
                            if dpool == 'submodel':
                                eicp = NP.copy(self.cPhase.cpinfo['processed'][dpool]['eicp'].data) # Minimum shape as stored
                                # eicp = NP.copy(self.cPhase.cpinfo['processed'][dpool]['eicp'].filled(1.0)) # Minimum shape as stored
                                eicp = NP.broadcast_to(eicp, self.cPhase.cpinfo['processed'][datapool]['eicp']['mean'].shape) # Broadcast to final shape
                                eicp = eicp[NP.newaxis,...] # nlst x ndays x ntriads x nchan --> (nspw=1) x nlst x ndays x ntriads x nchan
                                ndim_padtuple = [(0,0)]*(eicp.ndim-1) + [(0,npad)] # [(0,0), (0,0), (0,0), (0,0), (0,npad)]
                                result[dpool]['dspec'] = DSP.FT1D(NP.pad(eicp*flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:]*visscale.filled(NP.nan), ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                            else:
                                for key in self.cPhase.cpinfo['processed'][dpool]['eicp']:
                                    eicp = NP.copy(self.cPhase.cpinfo['processed'][dpool]['eicp'][key].data)
                                    # eicp = NP.copy(self.cPhase.cpinfo['processed'][dpool]['eicp'][key].filled(1.0))
                                    eicp = eicp[NP.newaxis,...] # nlst x ndays x ntriads x nchan --> (nspw=1) x nlst x ndays x ntriads x nchan
                                    ndim_padtuple = [(0,0)]*(eicp.ndim-1) + [(0,npad)] # [(0,0), (0,0), (0,0), (0,0), (0,npad)]
                                    if dpool == 'prelim':
                                        result['whole']['dspec'][key] = DSP.FT1D(NP.pad(eicp*flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:]*visscale.filled(NP.nan), ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                                    else:
                                        result[dpool]['dspec'][key] = DSP.FT1D(NP.pad(eicp*flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:]*visscale.filled(NP.nan), ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                result['lag_kernel'] = DSP.FT1D(NP.pad(flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:], ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df

            self.cPhaseDS = result
            if resample:
                result_resampled = copy.deepcopy(result)
                downsample_factor = NP.min((self.f.size + npad) * self.df / bw_eff)
                result_resampled['lags'] = DSP.downsampler(result_resampled['lags'], downsample_factor, axis=-1, method='interp', kind='linear')
                result_resampled['lag_kernel'] = DSP.downsampler(result_resampled['lag_kernel'], downsample_factor, axis=-1, method='interp', kind='linear')

                for dpool in ['errinfo', 'prelim', 'submodel', 'residual']:
                    if dpool.lower() == 'errinfo':
                        for diffind in self.cPhase.cpinfo[dpool]['eicp_diff']:
                            for key in self.cPhase.cpinfo[dpool]['eicp_diff'][diffind]:
                                result_resampled[dpool]['dspec'+diffind][key] = DSP.downsampler(result_resampled[dpool]['dspec'+diffind][key], downsample_factor, axis=-1, method='FFT')
                    if dpool in self.cPhase.cpinfo['processed']:
                        if dpool == 'submodel':
                            result_resampled[dpool]['dspec'] = DSP.downsampler(result_resampled[dpool]['dspec'], downsample_factor, axis=-1, method='FFT')
                        else:
                            for key in self.cPhase.cpinfo['processed'][datapool]['eicp']:
                                if dpool == 'prelim':
                                    result_resampled['whole']['dspec'][key] = DSP.downsampler(result_resampled['whole']['dspec'][key], downsample_factor, axis=-1, method='FFT')
                                else:
                                    result_resampled[dpool]['dspec'][key] = DSP.downsampler(result_resampled[dpool]['dspec'][key], downsample_factor, axis=-1, method='FFT')

                self.cPhaseDS_resampled = result_resampled
                return result_resampled
            else:
                return result

    ############################################################################

    def subset(self, selection=None):

        """
        ------------------------------------------------------------------------
        Return triad and time indices to select a subset of processed data

        Inputs:

        selection   [NoneType or dictionary] Selection parameters based on which
                    triad, LST, and day indices will be returned. If set to None
                    (default), all triad, LST, and day indices will be returned. 
                    Otherwise it must be a dictionary with the following keys 
                    and values:
                    'triads'    [NoneType or list of 3-element tuples] If set
                                to None (default), indices of all triads are
                                returned. Otherwise, the specific triads must
                                be specified such as [(1,2,3), (1,2,4), ...] 
                                and their indices will be returned
                    'lst'       [NoneType, list or numpy array] If set to None
                                (default), indices of all LST are returned. 
                                Otherwise must be a list or numpy array 
                                containing indices to LST.
                    'days'      [NoneType, list or numpy array] If set to None
                                (default), indices of all days are returned. 
                                Otherwise must be a list or numpy array 
                                containing indices to days. 

        Outputs:

        Tuple (triad_ind, lst_ind, day_ind, day_ind_eicpdiff) containing the 
        triad, LST, day, and day-pair (for subsample differences) indices, 
        each as a numpy array
        ------------------------------------------------------------------------
        """

        if selection is None:
            selsection = {}
        else:
            if not isinstance(selection, dict):
                raise TypeError('Input selection must be a dictionary')

        triads = map(tuple, self.cPhase.cpinfo['raw']['triads'])

        if 'triads' not in selection:
            selection['triads'] = triads
        if selection['triads'] is None:
            selection['triads'] = triads

        triad_ind = [triads.index(triad) for triad in selection['triads']]
        triad_ind = NP.asarray(triad_ind)

        lst_ind = None
        if 'lst' not in selection:
            if 'prelim' in self.cPhase.cpinfo['processed']:
                lst_ind = NP.arange(self.cPhase.cpinfo['processed']['prelim']['wts'].shape[0])
        else:
            if selection['lst'] is None:
                if 'prelim' in self.cPhase.cpinfo['processed']:
                    lst_ind = NP.arange(self.cPhase.cpinfo['processed']['prelim']['wts'].shape[0])
            elif isinstance(selection['lst'], (list,NP.ndarray)):
                if 'prelim' in self.cPhase.cpinfo['processed']:
                    lst_ind = selection['lst']
                    if NP.any(NP.logical_or(lst_ind < 0, lst_ind >= self.cPhase.cpinfo['processed']['prelim']['wts'].shape[0])):
                        raise ValueError('Input processed lst indices out of bounds')
            else:
                raise TypeError('Wrong type for processed lst indices')

        if lst_ind is None:
            raise ValueError('LST index selection could not be performed')
                
        day_ind = None
        day_ind_eicpdiff = None
        if 'days' not in selection:
            if 'prelim' in self.cPhase.cpinfo['processed']:
                day_ind = NP.arange(self.cPhase.cpinfo['processed']['prelim']['wts'].shape[1])
            if 'errinfo' in self.cPhase.cpinfo:
                day_ind_eicpdiff = NP.arange(len(self.cPhase.cpinfo['errinfo']['list_of_pair_of_pairs']))
        else:
            if selection['days'] is None:
                if 'prelim' in self.cPhase.cpinfo['processed']:
                    day_ind = NP.arange(self.cPhase.cpinfo['processed']['prelim']['wts'].shape[1])
                if 'errinfo' in self.cPhase.cpinfo:
                    day_ind_eicpdiff = NP.arange(len(self.cPhase.cpinfo['errinfo']['list_of_pair_of_pairs']))
            elif isinstance(selection['days'], (list,NP.ndarray)):
                if 'prelim' in self.cPhase.cpinfo['processed']:
                    day_ind = selection['days']
                    if NP.any(NP.logical_or(day_ind < 0, day_ind >= self.cPhase.cpinfo['processed']['prelim']['wts'].shape[1])):
                        raise ValueError('Input processed day indices out of bounds')
                if 'errinfo' in self.cPhase.cpinfo:
                    day_ind_eicpdiff = [i for i,item in enumerate(self.cPhase.cpinfo['errinfo']['list_of_pair_of_pairs']) if len(set(item)-set(selection['days']))==0]
            else:
                raise TypeError('Wrong type for processed day indices')

        if day_ind is None:
            raise ValueError('Day index selection could not be performed')
                
        return (triad_ind, lst_ind, day_ind, day_ind_eicpdiff)

    ############################################################################

    def compute_power_spectrum(self, cpds=None, selection=None, autoinfo=None,
                               xinfo=None, cosmo=cosmo100, units='K', beamparms=None):

        """
        ------------------------------------------------------------------------
        Compute power spectrum of closure phase data. It is in units of Mpc/h

        Inputs:

        cpds    [dictionary] A dictionary that contains the 'oversampled' (if 
                resample=False) and/or 'resampled' (if resample=True) delay 
                spectrum information. If it is not specified the attributes 
                cPhaseDS['processed'] and cPhaseDS_resampled['processed'] are 
                used. Under each of these keys, it holds a dictionary that has 
                the following keys and values:
                'freq_center'   [numpy array] contains the center frequencies 
                                (in Hz) of the frequency subbands of the subband
                                delay spectra. It is of size n_win. It is 
                                roughly equivalent to redshift(s)
                'freq_wts'      [numpy array] Contains frequency weights applied 
                                on each frequency sub-band during the subband 
                                delay transform. It is of size n_win x nchan. 
                'bw_eff'        [numpy array] contains the effective bandwidths 
                                (in Hz) of the subbands being delay transformed. 
                                It is of size n_win. It is roughly equivalent to 
                                width in redshift or along line-of-sight
                'shape'         [string] shape of the window function applied. 
                                Accepted values are 'rect' (rectangular), 'bhw'
                                (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'fftpow'        [scalar] the power to which the FFT of the window 
                                was raised. The value is be a positive scalar 
                                with default = 1.0
                'npad'          [scalar] Numbber of zero-padded channels before
                                performing the subband delay transform. 
                'lags'          [numpy array] lags of the subband delay spectra 
                                after padding in frequency during the transform. 
                                It is of size nlags. The lags roughly correspond 
                                to k_parallel.
                'lag_kernel'    [numpy array] delay transform of the frequency 
                                weights under the key 'freq_wts'. It is of size
                                n_bl x n_win x nlags x n_t. 
                'lag_corr_length' 
                                [numpy array] It is the correlation timescale 
                                (in pixels) of the subband delay spectra. It is 
                                proportional to inverse of effective bandwidth. 
                                It is of size n_win. The unit size of a pixel is 
                                determined by the difference between adjacent 
                                pixels in lags under key 'lags' which in turn is 
                                effectively inverse of the effective bandwidth 
                                of the subband specified in bw_eff
                'processed'     [dictionary] Contains the following keys and 
                                values:
                                'dspec' [dictionary] Contains the following keys 
                                        and values:
                                        'twts'  [numpy array] Weights from 
                                                time-based flags that went into 
                                                time-averaging. 
                                                Shape=(ntriads,npol,nchan,nt)
                                        'mean'  [numpy array] Delay spectrum of 
                                                closure phases based on their 
                                                mean across time intervals. 
                                                Shape=(nspw,npol,nt,ntriads,nlags)
                                        'median'
                                                [numpy array] Delay spectrum of 
                                                closure phases based on their 
                                                median across time intervals. 
                                                Shape=(nspw,npol,nt,ntriads,nlags)

        selection   [NoneType or dictionary] Selection parameters based on which
                    triad, LST, and day indices will be returned. If set to None
                    (default), all triad, LST, and day indices will be returned. 
                    Otherwise it must be a dictionary with the following keys 
                    and values:
                    'triads'    [NoneType or list of 3-element tuples] If set
                                to None (default), indices of all triads are
                                returned. Otherwise, the specific triads must
                                be specified such as [(1,2,3), (1,2,4), ...] 
                                and their indices will be returned
                    'lst'       [NoneType, list or numpy array] If set to None
                                (default), indices of all LST are returned. 
                                Otherwise must be a list or numpy array 
                                containing indices to LST.
                    'days'      [NoneType, list or numpy array] If set to None
                                (default), indices of all days are returned. 
                                Otherwise must be a list or numpy array 
                                containing indices to days. 

        autoinfo
                [NoneType or dictionary] Specifies parameters for processing 
                before power spectrum in auto or cross modes. If set to None, 
                a dictionary will be created with the default values as 
                described below. The dictionary must have the following keys
                and values:
                'axes'  [NoneType/int/list/tuple/numpy array] Axes that will
                        be averaged coherently before squaring (for auto) or
                        cross-multiplying (for cross) power spectrum. If set 
                        to None (default), no axes are averaged coherently. 
                        If set to int, list, tuple or numpy array, those axes
                        will be averaged coherently after applying the weights
                        specified under key 'wts' along those axes. 1=lst, 
                        2=days, 3=triads.
                'wts'   [NoneType/list/numpy array] If not provided (equivalent
                        to setting it to None) or set to None (default), it is
                        set to a one element list which is a one element numpy
                        array of unity. Otherwise, it must be a list of same
                        number of elements as in key 'axes' and each of these
                        must be a numpy broadcast compatible array corresponding
                        to each of the axis specified in 'axes'

        xinfo   [NoneType or dictionary] Specifies parameters for processing 
                cross power spectrum. If set to None, a dictionary will be 
                created with the default values as described below. The 
                dictionary must have the following keys and values:
                'axes'  [NoneType/int/list/tuple/numpy array] Axes over which 
                        power spectrum will be computed incoherently by cross-
                        multiplication. If set to None (default), no cross-
                        power spectrum is computed. If set to int, list, tuple 
                        or numpy array, cross-power over those axes will be 
                        computed incoherently by cross-multiplication. The 
                        cross-spectrum over these axes will be computed after
                        applying the pre- and post- cross-multiplication 
                        weights specified in key 'wts'. 1=lst, 2=days,
                        3=triads.
                'collapse_axes'
                        [list] The axes that will be collpased after the
                        cross-power matrix is produced by cross-multiplication.
                        If this key is not set, it will be initialized to an
                        empty list (default), in which case none of the axes 
                        is collapsed and the full cross-power matrix will be
                        output. it must be a subset of values under key 'axes'.
                        This will reduce it from a square matrix along that axis
                        to collapsed values along each of the leading diagonals.
                        1=lst, 2=days, 3=triads.
                'dlst'  [scalar] LST interval (in mins) or difference between LST
                        pairs which will be determined and used for 
                        cross-power spectrum. Will only apply if values under 
                        'axes' contains the LST axis(=1). 
                'dlst_range'
                        [scalar, numpy array, or NoneType] Specifies the LST 
                        difference(s) in minutes that are to be used in the 
                        computation of cross-power spectra. If a scalar, only 
                        the diagonal consisting of pairs with that LST 
                        difference will be computed. If a numpy array, those
                        diagonals consisting of pairs with that LST difference
                        will be computed. If set to None (default), the main
                        diagonal (LST difference of 0) and the first off-main 
                        diagonal (LST difference of 1 unit) corresponding to
                        pairs with 0 and 1 unit LST difference are computed.
                        Applies only if key 'axes' contains LST axis (=1).
                'avgcov'
                        [boolean] It specifies if the collapse of square 
                        covariance matrix is to be collapsed further to a single
                        number after applying 'postX' weights. If not set or
                        set to False (default), this late stage collapse will
                        not be performed. Otherwise, it will be averaged in a 
                        weighted average sense where the 'postX' weights would
                        have already been applied during the collapsing 
                        operation
                'wts'   [NoneType or Dictionary] If not set, a default 
                        dictionary (see default values below) will be created. 
                        It must have the follwoing keys and values:
                        'preX'  [list of numpy arrays] It contains pre-cross-
                                multiplication weights. It is a list where 
                                each element in the list is a numpy array, and
                                the number of elements in the list must match 
                                the number of entries in key 'axes'. If 'axes'
                                is set None, 'preX' may be set to a list 
                                with one element which is a numpy array of ones.
                                The number of elements in each of the numpy 
                                arrays must be numpy broadcastable into the 
                                number of elements along that axis in the 
                                delay spectrum.
                        'preXnorm'
                                [boolean] If False (default), no normalization
                                is done after the application of weights. If 
                                set to True, the delay spectrum will be 
                                normalized by the sum of the weights. 
                        'postX' [list of numpy arrays] It contains post-cross-
                                multiplication weights. It is a list where 
                                each element in the list is a numpy array, and
                                the number of elements in the list must match 
                                the number of entries in key 'axes'. If 'axes'
                                is set None, 'preX' may be set to a list 
                                with one element which is a numpy array of ones.
                                The number of elements in each of the numpy 
                                arrays must be numpy broadcastable into the 
                                number of elements along that axis in the 
                                delay spectrum. 
                        'preXnorm'
                                [boolean] If False (default), no normalization
                                is done after the application of 'preX' weights. 
                                If set to True, the delay spectrum will be 
                                normalized by the sum of the weights. 
                        'postXnorm'
                                [boolean] If False (default), no normalization
                                is done after the application of postX weights. 
                                If set to True, the delay cross power spectrum 
                                will be normalized by the sum of the weights. 

        cosmo   [instance of cosmology class from astropy] An instance of class
                FLRW or default_cosmology of astropy cosmology module. Default
                uses Planck 2015 cosmology, with H0=100 h km/s/Mpc

        units   [string] Specifies the units of output power spectum. Accepted
                values are 'Jy' and 'K' (default)) and the power spectrum will 
                be in corresponding squared units.

        Output:

        Dictionary with the keys 'triads' ((ntriads,3) array), 'triads_ind', 
        ((ntriads,) array), 'lstXoffsets' ((ndlst_range,) array), 'lst' 
        ((nlst,) array), 'dlst' ((nlst,) array), 'lst_ind' ((nlst,) array), 
        'days' ((ndays,) array), 'day_ind' ((ndays,) array), 'dday' 
        ((ndays,) array), 'oversampled' and 'resampled' corresponding to whether 
        resample was set to False or True in call to member function FT(). 
        Values under keys 'triads_ind' and 'lst_ind' are numpy array 
        corresponding to triad and time indices used in selecting the data. 
        Values under keys 'oversampled' and 'resampled' each contain a 
        dictionary with the following keys and values:
        'z'     [numpy array] Redshifts corresponding to the band centers in 
                'freq_center'. It has shape=(nspw,)
        'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,).
        'kprll' [numpy array] k_parallel modes (in h/Mpc) corresponding to 
                'lags'. It has shape=(nspw,nlags)
        'freq_center'   
                [numpy array] contains the center frequencies (in Hz) of the 
                frequency subbands of the subband delay spectra. It is of size 
                n_win. It is roughly equivalent to redshift(s)
        'freq_wts'      
                [numpy array] Contains frequency weights applied on each 
                frequency sub-band during the subband delay transform. It is 
                of size n_win x nchan. 
        'bw_eff'        
                [numpy array] contains the effective bandwidths (in Hz) of the 
                subbands being delay transformed. It is of size n_win. It is 
                roughly equivalent to width in redshift or along line-of-sight
        'shape' [string] shape of the frequency window function applied. Usual
                values are 'rect' (rectangular), 'bhw' (Blackman-Harris), 
                'bnw' (Blackman-Nuttall). 
        'fftpow'
                [scalar] the power to which the FFT of the window was raised. 
                The value is be a positive scalar with default = 1.0
        'lag_corr_length' 
                [numpy array] It is the correlation timescale (in pixels) of 
                the subband delay spectra. It is proportional to inverse of 
                effective bandwidth. It is of size n_win. The unit size of a 
                pixel is determined by the difference between adjacent pixels 
                in lags under key 'lags' which in turn is effectively inverse 
                of the effective bandwidth of the subband specified in bw_eff

        It further contains 3 keys named 'whole', 'submodel', and 'residual'
        each of which is a dictionary. 'whole' contains power spectrum info 
        about the input closure phases. 'submodel' contains power spectrum info
        about the model that will have been subtracted (as closure phase) from 
        the 'whole' model. 'residual' contains power spectrum info about the 
        closure phases obtained as a difference between 'whole' and 'submodel'.
        It contains the following keys and values:
        'mean'  [numpy array] Delay power spectrum incoherently estiamted over 
                the axes specified in xinfo['axes'] using the 'mean' key in input 
                cpds or attribute cPhaseDS['processed']['dspec']. It has shape 
                that depends on the combination of input parameters. See 
                examples below. If both collapse_axes and avgcov are not set, 
                those axes will be replaced with square covariance matrices. If 
                collapse_axes is provided but avgcov is False, those axes will be 
                of shape 2*Naxis-1. 
        'median'
                [numpy array] Delay power spectrum incoherently averaged over 
                the axes specified in incohax using the 'median' key in input 
                cpds or attribute cPhaseDS['processed']['dspec']. It has shape 
                that depends on the combination of input parameters. See 
                examples below. If both collapse_axes and avgcov are not set, 
                those axes will be replaced with square covariance matrices. If 
                collapse_axes is provided bu avgcov is False, those axes will be 
                of shape 2*Naxis-1. 
        'diagoffsets' 
                [dictionary] Same keys corresponding to keys under 
                'collapse_axes' in input containing the diagonal offsets for
                those axes. If 'avgcov' was set, those entries will be removed 
                from 'diagoffsets' since all the leading diagonal elements have 
                been collapsed (averaged) further. Value under each key is a 
                numpy array where each element in the array corresponds to the 
                index of that leading diagonal. This should match the size of 
                the output along that axis in 'mean' or 'median' above. 
        'diagweights'
                [dictionary] Each key is an axis specified in collapse_axes and
                the value is a numpy array of weights corresponding to the 
                diagonal offsets in that axis.
        'axesmap'
                [dictionary] If covariance in cross-power is calculated but is 
                not collapsed, the number of dimensions in the output will have
                changed. This parameter tracks where the original axis is now 
                placed. The keys are the original axes that are involved in 
                incoherent cross-power, and the values are the new locations of 
                those original axes in the output. 
        'nsamples_incoh'
                [integer] Number of incoherent samples in producing the power
                spectrum
        'nsamples_coh'
                [integer] Number of coherent samples in producing the power
                spectrum

        Examples: 

        (1)
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': 2, 'wts': None}
        xinfo = {'axes': None, 'avgcov': False, 'collapse_axes': [], 
                 'wts':{'preX': None, 'preXnorm': False, 
                        'postX': None, 'postXnorm': False}}
        Output delay power spectrum has shape (Nspw, Nlst, 1, Ntriads, Nlags)

        (2) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': 2, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': False, 'collapse_axes': [], 
                 'wts':{'preX': None, 'preXnorm': False, 
                        'postX': None, 'postXnorm': False}, 
                 'dlst_range': None}
        Output delay power spectrum has shape 
        (Nspw, 2, Nlst, 1, Ntriads, Ntriads, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range)}, 
        axesmap = {1: [1,2], 3: [4,5]}

        (3) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': 2, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': False, 'collapse_axes': [3], 
                 'dlst_range': [0.0, 1.0, 2.0]}
        Output delay power spectrum has shape 
        (Nspw, 3, Nlst, 1, 2*Ntriads-1, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range), 
                       3: NP.arange(-Ntriads,Ntriads)}, 
        axesmap = {1: [1,2], 3: [4]}

        (4) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': None, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': False, 'collapse_axes': [1,3], 
                 'dlst_range': [1.0, 2.0, 3.0, 4.0]}
        Output delay power spectrum has shape 
        (Nspw, 4, Ndays, 2*Ntriads-1, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range), 
                       3: NP.arange(-Ntriads,Ntriads)}, 
        axesmap = {1: [1], 3: [3]}

        (5) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': None, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': True, 'collapse_axes': [3], 
                 'dlst_range': None}
        Output delay power spectrum has shape 
        (Nspw, 2, Nlst, Ndays, 1, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range)}, axesmap = {1: [1,2], 3: [4]}

        (6) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': None, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': True, 'collapse_axes': []}
        Output delay power spectrum has shape 
        (Nspw, 1, Ndays, 1, Nlags)
        diagoffsets = {}, axesmap = {1: [1], 3: [3]}
        ------------------------------------------------------------------------
        """

        if not isinstance(units,str):
            raise TypeError('Input parameter units must be a string')
        if units.lower() == 'k':
            if not isinstance(beamparms, dict):
                raise TypeError('Input beamparms must be a dictionary')
            if 'freqs' not in beamparms:
                beamparms['freqs'] = self.f
            beamparms_orig = copy.deepcopy(beamparms)

        if autoinfo is None:
            autoinfo = {'axes': None, 'wts': [NP.ones(1, dtpye=NP.float)]}
        elif not isinstance(autoinfo, dict):
            raise TypeError('Input autoinfo must be a dictionary')

        if 'axes' not in autoinfo:
            autoinfo['axes'] = None
        else:
            if autoinfo['axes'] is not None:
                if not isinstance(autoinfo['axes'], (list,tuple,NP.ndarray,int)):
                    raise TypeError('Value under key axes in input autoinfo must be an integer, list, tuple or numpy array')
                else:
                    autoinfo['axes'] = NP.asarray(autoinfo['axes']).reshape(-1)

        if 'wts' not in autoinfo:
            if autoinfo['axes'] is not None:
                autoinfo['wts'] = [NP.ones(1, dtype=NP.float)] * len(autoinfo['axes'])
            else:
                autoinfo['wts'] = [NP.ones(1, dtype=NP.float)]
        else:
            if autoinfo['axes'] is not None:
                if not isinstance(autoinfo['wts'], list):
                    raise TypeError('wts in input autoinfo must be a list of numpy arrays')
                else:
                    if len(autoinfo['wts']) != len(autoinfo['axes']):
                        raise ValueError('Input list of wts must be same as length of autoinfo axes')
            else:
                autoinfo['wts'] = [NP.ones(1, dtype=NP.float)]

        if xinfo is None:
            xinfo = {'axes': None, 'wts': {'preX': [NP.ones(1, dtpye=NP.float)], 'postX': [NP.ones(1, dtpye=NP.float)], 'preXnorm': False, 'postXnorm': False}}
        elif not isinstance(xinfo, dict):
            raise TypeError('Input xinfo must be a dictionary')

        if 'axes' not in xinfo:
            xinfo['axes'] = None
        else:
            if not isinstance(xinfo['axes'], (list,tuple,NP.ndarray,int)):
                raise TypeError('Value under key axes in input xinfo must be an integer, list, tuple or numpy array')
            else:
                xinfo['axes'] = NP.asarray(xinfo['axes']).reshape(-1)

        if 'wts' not in xinfo:
            xinfo['wts'] = {}
            for xkey in ['preX', 'postX']:
                if xinfo['axes'] is not None:
                    xinfo['wts'][xkey] = [NP.ones(1, dtype=NP.float)] * len(xinfo['axes'])
                else:
                    xinfo['wts'][xkey] = [NP.ones(1, dtype=NP.float)]
            xinfo['wts']['preXnorm'] = False
            xinfo['wts']['postXnorm'] = False
        else:
            if xinfo['axes'] is not None:
                if not isinstance(xinfo['wts'], dict):
                    raise TypeError('wts in input xinfo must be a dictionary')
                for xkey in ['preX', 'postX']:
                    if not isinstance(xinfo['wts'][xkey], list):
                        raise TypeError('{0} wts in input xinfo must be a list of numpy arrays'.format(xkey))
                    else:
                        if len(xinfo['wts'][xkey]) != len(xinfo['axes']):
                            raise ValueError('Input list of {0} wts must be same as length of xinfo axes'.format(xkey))
            else:
                for xkey in ['preX', 'postX']:
                    xinfo['wts'][xkey] = [NP.ones(1, dtype=NP.float)]

            if 'preXnorm' not in xinfo['wts']:
                xinfo['wts']['preXnorm'] = False
            if 'postXnorm' not in xinfo['wts']:
                xinfo['wts']['postXnorm'] = False
            if not isinstance(xinfo['wts']['preXnorm'], NP.bool):
                raise TypeError('preXnorm in input xinfo must be a boolean')
            if not isinstance(xinfo['wts']['postXnorm'], NP.bool):
                raise TypeError('postXnorm in input xinfo must be a boolean')

        if 'avgcov' not in xinfo:
            xinfo['avgcov'] = False
        if not isinstance(xinfo['avgcov'], NP.bool):
            raise TypeError('avgcov under input xinfo must be boolean')

        if 'collapse_axes' not in xinfo:
            xinfo['collapse_axes'] = []
        if not isinstance(xinfo['collapse_axes'], (int,list,tuple,NP.ndarray)):
            raise TypeError('collapse_axes under input xinfo must be an integer, tuple, list or numpy array')
        else:
            xinfo['collapse_axes'] = NP.asarray(xinfo['collapse_axes']).reshape(-1)

        if (autoinfo['axes'] is not None) and (xinfo['axes'] is not None):
            if NP.intersect1d(autoinfo['axes'], xinfo['axes']).size > 0:
                raise ValueError("Inputs autoinfo['axes'] and xinfo['axes'] must have no intersection")

        cohax = autoinfo['axes']
        if cohax is None:
            cohax = []
        incohax = xinfo['axes']
        if incohax is None:
            incohax = []

        if selection is None:
            selection = {'triads': None, 'lst': None, 'days': None}
        else:
            if not isinstance(selection, dict):
                raise TypeError('Input selection must be a dictionary')

        if cpds is None:
            cpds = {}
            sampling = ['oversampled', 'resampled']
            for smplng in sampling:
                if smplng == 'oversampled':
                    cpds[smplng] = copy.deepcopy(self.cPhaseDS)
                else:
                    cpds[smplng] = copy.deepcopy(self.cPhaseDS_resampled)

        triad_ind, lst_ind, day_ind, day_ind_eicpdiff = self.subset(selection=selection)

        result = {'triads': self.cPhase.cpinfo['raw']['triads'][triad_ind], 'triads_ind': triad_ind, 'lst': self.cPhase.cpinfo['processed']['prelim']['lstbins'][lst_ind], 'lst_ind': lst_ind, 'dlst': self.cPhase.cpinfo['processed']['prelim']['dlstbins'][lst_ind], 'days': self.cPhase.cpinfo['processed']['prelim']['daybins'][day_ind], 'day_ind': day_ind, 'dday': self.cPhase.cpinfo['processed']['prelim']['diff_dbins'][day_ind]}

        dlstbin = NP.mean(self.cPhase.cpinfo['processed']['prelim']['dlstbins'])
        if 'dlst_range' in xinfo:
            if xinfo['dlst_range'] is None:
                dlst_range = None
                lstshifts = NP.arange(2) # LST index offsets of 0 and 1 are only estimated
            else:
                dlst_range = NP.asarray(xinfo['dlst_range']).ravel() / 60.0 # Difference in LST between a pair of LST (in hours)
                if dlst_range.size == 1:
                    dlst_range = NP.insert(dlst_range, 0, 0.0)
                lstshifts = NP.arange(max([0, NP.ceil(1.0*dlst_range.min()/dlstbin).astype(NP.int)]), min([NP.ceil(1.0*dlst_range.max()/dlstbin).astype(NP.int), result['lst'].size]))
        else:
            dlst_range = None
            lstshifts = NP.arange(2) # LST index offsets of 0 and 1 are only estimated
        result['lstXoffsets'] = lstshifts * dlstbin # LST interval corresponding to diagonal offsets created by the LST covariance

        for smplng in sampling:
            result[smplng] = {}
                
            wl = FCNST.c / (cpds[smplng]['freq_center'] * U.Hz)
            z = CNST.rest_freq_HI / cpds[smplng]['freq_center'] - 1
            dz = CNST.rest_freq_HI / cpds[smplng]['freq_center']**2 * cpds[smplng]['bw_eff']
            dkprll_deta = DS.dkprll_deta(z, cosmo=cosmo)
            kprll = dkprll_deta.reshape(-1,1) * cpds[smplng]['lags']

            rz_los = cosmo.comoving_distance(z) # in Mpc/h
            drz_los = FCNST.c * cpds[smplng]['bw_eff']*U.Hz * (1+z)**2 / (CNST.rest_freq_HI * U.Hz) / (cosmo.H0 * cosmo.efunc(z))   # in Mpc/h
            if units == 'Jy':
                jacobian1 = 1 / (cpds[smplng]['bw_eff'] * U.Hz)
                jacobian2 = drz_los / (cpds[smplng]['bw_eff'] * U.Hz)
                temperature_from_fluxdensity = 1.0
            elif units == 'K':
                beamparms = copy.deepcopy(beamparms_orig)
                omega_bw = self.beam3Dvol(beamparms, freq_wts=cpds[smplng]['freq_wts'])
                jacobian1 = 1 / (omega_bw * U.Hz) # The steradian is present but not explicitly assigned
                jacobian2 = rz_los**2 * drz_los / (cpds[smplng]['bw_eff'] * U.Hz)
                temperature_from_fluxdensity = wl**2 / (2*FCNST.k_B)
            else:
                raise ValueError('Input value for units invalid')

            factor = jacobian1 * jacobian2 * temperature_from_fluxdensity**2

            result[smplng]['z'] = z
            result[smplng]['kprll'] = kprll
            result[smplng]['lags'] = NP.copy(cpds[smplng]['lags'])
            result[smplng]['freq_center'] = cpds[smplng]['freq_center']
            result[smplng]['bw_eff'] = cpds[smplng]['bw_eff']
            result[smplng]['shape'] = cpds[smplng]['shape']
            result[smplng]['freq_wts'] = cpds[smplng]['freq_wts']
            result[smplng]['lag_corr_length'] = cpds[smplng]['lag_corr_length']

            for dpool in ['whole', 'submodel', 'residual']:
                if dpool in cpds[smplng]:
                    result[smplng][dpool] = {}
                    inpshape = list(cpds[smplng]['whole']['dspec']['mean'].shape)
                    inpshape[1] = lst_ind.size
                    inpshape[2] = day_ind.size
                    inpshape[3] = triad_ind.size
                    if len(cohax) > 0:
                        nsamples_coh = NP.prod(NP.asarray(inpshape)[NP.asarray(cohax)])
                    else:
                        nsamples_coh = 1
                    if len(incohax) > 0:
                        nsamples = NP.prod(NP.asarray(inpshape)[NP.asarray(incohax)])
                        nsamples_incoh = nsamples * (nsamples - 1)
                    else:
                        nsamples_incoh = 1
                    twts_multidim_idx = NP.ix_(lst_ind,day_ind,triad_ind,NP.arange(1)) # shape=(nlst,ndays,ntriads,1)
                    dspec_multidim_idx = NP.ix_(NP.arange(wl.size),lst_ind,day_ind,triad_ind,NP.arange(inpshape[4])) # shape=(nspw,nlst,ndays,ntriads,nchan)
                    max_wt_in_chan = NP.max(NP.sum(cpds[smplng]['whole']['dspec']['twts'].data, axis=(0,1,2)))
                    select_chan = NP.argmax(NP.sum(cpds[smplng]['whole']['dspec']['twts'].data, axis=(0,1,2)))
                    twts = NP.copy(cpds[smplng]['whole']['dspec']['twts'].data[:,:,:,[select_chan]]) # shape=(nlst,ndays,ntriads,nlags=1)

                    if nsamples_coh > 1:
                        awts_shape = tuple(NP.ones(cpds[smplng]['whole']['dspec']['mean'].ndim, dtype=NP.int))
                        awts = NP.ones(awts_shape, dtype=NP.complex)
                        awts_shape = NP.asarray(awts_shape)
                        for caxind,caxis in enumerate(cohax):
                            curr_awts_shape = NP.copy(awts_shape)
                            curr_awts_shape[caxis] = -1
                            awts = awts * autoinfo['wts'][caxind].reshape(tuple(curr_awts_shape))

                    for stat in ['mean', 'median']:
                        if dpool == 'submodel':
                            dspec = NP.copy(cpds[smplng][dpool]['dspec'][dspec_multidim_idx])
                        else:
                            dspec = NP.copy(cpds[smplng][dpool]['dspec'][stat][dspec_multidim_idx])
                        if nsamples_coh > 1:
                            if stat == 'mean':
                                dspec = NP.sum(twts[twts_multidim_idx][NP.newaxis,...] * awts * dspec[dspec_multidim_idx], axis=cohax, keepdims=True) / NP.sum(twts[twts_multidim_idx][NP.newaxis,...] * awts, axis=cohax, keepdims=True)
                            else:
                                dspec = NP.median(dspec[dspec_multidim_idx], axis=cohax, keepdims=True)
                        if nsamples_incoh > 1:
                            expandax_map = {}
                            wts_shape = tuple(NP.ones(dspec.ndim, dtype=NP.int))
                            preXwts = NP.ones(wts_shape, dtype=NP.complex)
                            wts_shape = NP.asarray(wts_shape)
                            for incaxind,incaxis in enumerate(xinfo['axes']):
                                curr_wts_shape = NP.copy(wts_shape)
                                curr_wts_shape[incaxis] = -1
                                preXwts = preXwts * xinfo['wts']['preX'][incaxind].reshape(tuple(curr_wts_shape))
                            dspec1 = NP.copy(dspec)
                            dspec2 = NP.copy(dspec)
                            preXwts1 = NP.copy(preXwts)
                            preXwts2 = NP.copy(preXwts)
                            for incax in NP.sort(incohax)[::-1]:
                                dspec1 = NP.expand_dims(dspec1, axis=incax)
                                preXwts1 = NP.expand_dims(preXwts1, axis=incax)
                                if incax == 1:
                                    preXwts1_outshape = list(preXwts1.shape)
                                    preXwts1_outshape[incax+1] = dspec1.shape[incax+1]
                                    preXwts1_outshape = tuple(preXwts1_outshape)
                                    preXwts1 = NP.broadcast_to(preXwts1, preXwts1_outshape).copy() # For some strange reason the NP.broadcast_to() creates a "read-only" immutable array which is changed to writeable by copy()
                                    
                                    preXwts2_tmp = NP.expand_dims(preXwts2, axis=incax)
                                    preXwts2_shape = NP.asarray(preXwts2_tmp.shape)
                                    preXwts2_shape[incax] = lstshifts.size
                                    preXwts2_shape[incax+1] = preXwts1_outshape[incax+1]
                                    preXwts2_shape = tuple(preXwts2_shape)
                                    preXwts2 = NP.broadcast_to(preXwts2_tmp, preXwts2_shape).copy() # For some strange reason the NP.broadcast_to() creates a "read-only" immutable array which is changed to writeable by copy()

                                    dspec2_tmp = NP.expand_dims(dspec2, axis=incax)
                                    dspec2_shape = NP.asarray(dspec2_tmp.shape)
                                    dspec2_shape[incax] = lstshifts.size
                                    # dspec2_shape = NP.insert(dspec2_shape, incax, lstshifts.size)
                                    dspec2_shape = tuple(dspec2_shape)
                                    dspec2 = NP.broadcast_to(dspec2_tmp, dspec2_shape).copy() # For some strange reason the NP.broadcast_to() creates a "read-only" immutable array which is changed to writeable by copy()
                                    for lstshiftind, lstshift in enumerate(lstshifts):
                                        dspec2[:,lstshiftind,...] = NP.roll(dspec2_tmp[:,0,...], lstshift, axis=incax)
                                        dspec2[:,lstshiftind,:lstshift,...] = NP.nan
                                        preXwts2[:,lstshiftind,...] = NP.roll(preXwts2_tmp[:,0,...], lstshift, axis=incax)
                                        preXwts2[:,lstshiftind,:lstshift,...] = NP.nan
                                else:
                                    dspec2 = NP.expand_dims(dspec2, axis=incax+1)
                                    preXwts2 = NP.expand_dims(preXwts2, axis=incax+1)
                                expandax_map[incax] = incax + NP.arange(2)
                                for ekey in expandax_map:
                                    if ekey > incax:
                                        expandax_map[ekey] += 1
                                        
                            result[smplng][dpool][stat] = factor.reshape((-1,)+tuple(NP.ones(dspec1.ndim-1, dtype=NP.int))) * (dspec1*U.Unit('Jy Hz') * preXwts1) * (dspec2*U.Unit('Jy Hz') * preXwts2).conj()
                            if xinfo['wts']['preXnorm']:
                                result[smplng][dpool][stat] = result[smplng][dpool][stat] / NP.nansum(preXwts1 * preXwts2.conj(), axis=NP.union1d(NP.where(logical_or(NP.asarray(preXwts1.shape)>1, NP.asarray(preXwts2.shape)>1))), keepdims=True) # Normalize by summing the weights over the expanded axes
        
                            if (len(xinfo['collapse_axes']) > 0) or (xinfo['avgcov']):
        
                                # if any one of collapsing of incoherent axes or 
                                # averaging of full covariance is requested
        
                                diagoffsets = {} # Stores the correlation index difference along each axis.
                                diagweights = {} # Stores the number of points summed in the trace along the offset diagonal
                                for colaxind, colax in enumerate(xinfo['collapse_axes']):
                                    if colax == 1:
                                        shp = NP.ones(dspec.ndim, dtype=NP.int)
                                        shp[colax] = lst_ind.size
                                        multdim_idx = tuple([NP.arange(axdim) for axdim in shp])
                                        diagweights[colax] = NP.sum(NP.logical_not(NP.isnan(dspec[multdim_idx]))) - lstshifts
                                        # diagweights[colax] = result[smplng][dpool][stat].shape[expandax_map[colax][-1]] - lstshifts

                                        if stat == 'mean':
                                            result[smplng][dpool][stat] = NP.nanmean(result[smplng][dpool][stat], axis=expandax_map[colax][-1])
                                        else:
                                            result[smplng][dpool][stat] = NP.nanmedian(result[smplng][dpool][stat], axis=expandax_map[colax][-1])
                                        diagoffsets[colax] = lstshifts
                                    else:
                                        pspec_unit = result[smplng][dpool][stat].si.unit
                                        result[smplng][dpool][stat], offsets, diagwts = OPS.array_trace(result[smplng][dpool][stat].si.value, offsets=None, axis1=expandax_map[colax][0], axis2=expandax_map[colax][1], outaxis='axis1')
                                        diagwts_shape = NP.ones(result[smplng][dpool][stat].ndim, dtype=NP.int)
                                        diagwts_shape[expandax_map[colax][0]] = diagwts.size
                                        diagoffsets[colax] = offsets
                                        diagweights[colax] = NP.copy(diagwts)
                                        result[smplng][dpool][stat] = result[smplng][dpool][stat] * pspec_unit / diagwts.reshape(diagwts_shape)
                                    for ekey in expandax_map:
                                        if ekey > colax:
                                            expandax_map[ekey] -= 1
                                    expandax_map[colax] = NP.asarray(expandax_map[colax][0]).ravel()
        
                                wts_shape = tuple(NP.ones(result[smplng][dpool][stat].ndim, dtype=NP.int))
                                postXwts = NP.ones(wts_shape, dtype=NP.complex)
                                wts_shape = NP.asarray(wts_shape)
                                for colaxind, colax in enumerate(xinfo['collapse_axes']):
                                    curr_wts_shape = NP.copy(wts_shape)
                                    curr_wts_shape[expandax_map[colax]] = -1
                                    postXwts = postXwts * xinfo['wts']['postX'][colaxind].reshape(tuple(curr_wts_shape))
                                    
                                result[smplng][dpool][stat] = result[smplng][dpool][stat] * postXwts
        
                                axes_to_sum = tuple(NP.asarray([expandax_map[colax] for colax in xinfo['collapse_axes']]).ravel()) # for post-X normalization and collapse of covariance matrix
        
                                if xinfo['wts']['postXnorm']:
                                    result[smplng][dpool][stat] = result[smplng][dpool][stat] / NP.nansum(postXwts, axis=axes_to_sum, keepdims=True) # Normalize by summing the weights over the collapsed axes
                                if xinfo['avgcov']:
        
                                    # collapse the axes further (postXwts have already
                                    # been applied)
        
                                    diagoffset_weights = 1.0
                                    for colaxind in zip(*sorted(zip(NP.arange(xinfo['collapse_axes'].size), xinfo['collapse_axes']), reverse=True))[0]:
                                        # It is important to sort the collapsable axes in
                                        # reverse order before deleting elements below,
                                        # otherwise the axes ordering may be get messed up
        
                                        diagoffset_weights_shape = NP.ones(result[smplng][dpool][stat].ndim, dtype=NP.int)
                                        diagoffset_weights_shape[expandax_map[xinfo['collapse_axes'][colaxind]][0]] = diagweights[xinfo['collapse_axes'][colaxind]].size
                                        diagoffset_weights = diagoffset_weights * diagweights[xinfo['collapse_axes'][colaxind]].reshape(diagoffset_weights_shape)
                                        del diagoffsets[xinfo['collapse_axes'][colaxind]]
                                    result[smplng][dpool][stat] = NP.nansum(result[smplng][dpool][stat]*diagoffset_weights, axis=axes_to_sum, keepdims=True) / NP.nansum(diagoffset_weights, axis=axes_to_sum, keepdims=True)
                        else:
                            result[smplng][dpool][stat] = factor.reshape((-1,)+tuple(NP.ones(dspec.ndim-1, dtype=NP.int))) * NP.abs(dspec * U.Jy)**2
                            diagoffsets = {}
                            expandax_map = {}                            
                            
                    if units == 'Jy':
                        result[smplng][dpool][stat] = result[smplng][dpool][stat].to('Jy2 Mpc')
                    elif units == 'K':
                        result[smplng][dpool][stat] = result[smplng][dpool][stat].to('K2 Mpc3')
                    else:
                        raise ValueError('Input value for units invalid')
                    result[smplng][dpool]['diagoffsets'] = diagoffsets
                    result[smplng][dpool]['diagweights'] = diagweights
                    result[smplng][dpool]['axesmap'] = expandax_map

                    result[smplng][dpool]['nsamples_incoh'] = nsamples_incoh
                    result[smplng][dpool]['nsamples_coh'] = nsamples_coh

        return result

    ############################################################################

    def compute_power_spectrum_uncertainty(self, cpds=None, selection=None,
                                           autoinfo=None,xinfo=None,
                                           cosmo=cosmo100, units='K',
                                           beamparms=None):

        """
        ------------------------------------------------------------------------
        Compute uncertainty in the power spectrum of closure phase data. It is 
        in units of Mpc/h

        Inputs:

        cpds    [dictionary] A dictionary that contains the 'oversampled' (if 
                resample=False) and/or 'resampled' (if resample=True) delay 
                spectrum information on the key 'errinfo'. If it is not 
                specified the attributes cPhaseDS['errinfo'] and 
                cPhaseDS_resampled['errinfo'] are used. Under each of these 
                sampling keys, it holds a dictionary that has the following 
                keys and values:
                'freq_center'   [numpy array] contains the center frequencies 
                                (in Hz) of the frequency subbands of the subband
                                delay spectra. It is of size n_win. It is 
                                roughly equivalent to redshift(s)
                'freq_wts'      [numpy array] Contains frequency weights applied 
                                on each frequency sub-band during the subband 
                                delay transform. It is of size n_win x nchan. 
                'bw_eff'        [numpy array] contains the effective bandwidths 
                                (in Hz) of the subbands being delay transformed. 
                                It is of size n_win. It is roughly equivalent to 
                                width in redshift or along line-of-sight
                'shape'         [string] shape of the window function applied. 
                                Accepted values are 'rect' (rectangular), 'bhw'
                                (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'fftpow'        [scalar] the power to which the FFT of the window 
                                was raised. The value is be a positive scalar 
                                with default = 1.0
                'npad'          [scalar] Numbber of zero-padded channels before
                                performing the subband delay transform. 
                'lags'          [numpy array] lags of the subband delay spectra 
                                after padding in frequency during the transform. 
                                It is of size nlags. The lags roughly correspond 
                                to k_parallel.
                'lag_kernel'    [numpy array] delay transform of the frequency 
                                weights under the key 'freq_wts'. It is of size
                                n_bl x n_win x nlags x n_t. 
                'lag_corr_length' 
                                [numpy array] It is the correlation timescale 
                                (in pixels) of the subband delay spectra. It is 
                                proportional to inverse of effective bandwidth. 
                                It is of size n_win. The unit size of a pixel is 
                                determined by the difference between adjacent 
                                pixels in lags under key 'lags' which in turn is 
                                effectively inverse of the effective bandwidth 
                                of the subband specified in bw_eff
                'errinfo'       [dictionary] It has two keys 'dspec0' and 
                                'dspec1' each of which are dictionaries with 
                                the following keys and values:
                                'twts'  [numpy array] Weights for the subsample 
                                        difference. It is of shape (nlst, ndays, 
                                        ntriads, nchan)
                                'mean'  [numpy array] Delay spectrum of the 
                                        subsample difference obtained by using 
                                        the mean statistic. It is of shape 
                                        (nspw, nlst, ndays, ntriads, nlags)
                                'median'
                                        [numpy array] Delay spectrum of the 
                                        subsample difference obtained by using 
                                        the median statistic. It is of shape 
                                        (nspw, nlst, ndays, ntriads, nlags)

        selection   [NoneType or dictionary] Selection parameters based on which
                    triad, LST, and day indices will be returned. If set to None
                    (default), all triad, LST, and day indices will be returned. 
                    Otherwise it must be a dictionary with the following keys 
                    and values:
                    'triads'    [NoneType or list of 3-element tuples] If set
                                to None (default), indices of all triads are
                                returned. Otherwise, the specific triads must
                                be specified such as [(1,2,3), (1,2,4), ...] 
                                and their indices will be returned
                    'lst'       [NoneType, list or numpy array] If set to None
                                (default), indices of all LST are returned. 
                                Otherwise must be a list or numpy array 
                                containing indices to LST.
                    'days'      [NoneType, list or numpy array] If set to None
                                (default), indices of all days are returned. 
                                Otherwise must be a list or numpy array 
                                containing indices to days. 

        autoinfo
                [NoneType or dictionary] Specifies parameters for processing 
                before power spectrum in auto or cross modes. If set to None, 
                a dictionary will be created with the default values as 
                described below. The dictionary must have the following keys
                and values:
                'axes'  [NoneType/int/list/tuple/numpy array] Axes that will
                        be averaged coherently before squaring (for auto) or
                        cross-multiplying (for cross) power spectrum. If set 
                        to None (default), no axes are averaged coherently. 
                        If set to int, list, tuple or numpy array, those axes
                        will be averaged coherently after applying the weights
                        specified under key 'wts' along those axes. 1=lst, 
                        3=triads. Value of 2 for axes is not allowed since 
                        that denotes repeated days and it is along this axis
                        that cross-power is computed regardless. 
                'wts'   [NoneType/list/numpy array] If not provided (equivalent
                        to setting it to None) or set to None (default), it is
                        set to a one element list which is a one element numpy
                        array of unity. Otherwise, it must be a list of same
                        number of elements as in key 'axes' and each of these
                        must be a numpy broadcast compatible array corresponding
                        to each of the axis specified in 'axes'

        xinfo   [NoneType or dictionary] Specifies parameters for processing 
                cross power spectrum. If set to None, a dictionary will be 
                created with the default values as described below. The 
                dictionary must have the following keys and values:
                'axes'  [NoneType/int/list/tuple/numpy array] Axes over which 
                        power spectrum will be computed incoherently by cross-
                        multiplication. If set to None (default), no cross-
                        power spectrum is computed. If set to int, list, tuple 
                        or numpy array, cross-power over those axes will be 
                        computed incoherently by cross-multiplication. The 
                        cross-spectrum over these axes will be computed after
                        applying the pre- and post- cross-multiplication 
                        weights specified in key 'wts'. 1=lst, 3=triads. Value 
                        of 2 for axes is not allowed since that denotes 
                        repeated days and it is along this axis that 
                        cross-power is computed regardless. 
                'collapse_axes'
                        [list] The axes that will be collpased after the
                        cross-power matrix is produced by cross-multiplication.
                        If this key is not set, it will be initialized to an
                        empty list (default), in which case none of the axes 
                        is collapsed and the full cross-power matrix will be
                        output. it must be a subset of values under key 'axes'.
                        This will reduce it from a square matrix along that axis
                        to collapsed values along each of the leading diagonals.
                        1=lst, 3=triads.
                'dlst'  [scalar] LST interval (in mins) or difference between LST
                        pairs which will be determined and used for 
                        cross-power spectrum. Will only apply if values under 
                        'axes' contains the LST axis(=1). 
                'dlst_range'
                        [scalar, numpy array, or NoneType] Specifies the LST 
                        difference(s) in minutes that are to be used in the 
                        computation of cross-power spectra. If a scalar, only 
                        the diagonal consisting of pairs with that LST 
                        difference will be computed. If a numpy array, those
                        diagonals consisting of pairs with that LST difference
                        will be computed. If set to None (default), the main
                        diagonal (LST difference of 0) and the first off-main 
                        diagonal (LST difference of 1 unit) corresponding to
                        pairs with 0 and 1 unit LST difference are computed.
                        Applies only if key 'axes' contains LST axis (=1).
                'avgcov'
                        [boolean] It specifies if the collapse of square 
                        covariance matrix is to be collapsed further to a single
                        number after applying 'postX' weights. If not set or
                        set to False (default), this late stage collapse will
                        not be performed. Otherwise, it will be averaged in a 
                        weighted average sense where the 'postX' weights would
                        have already been applied during the collapsing 
                        operation
                'wts'   [NoneType or Dictionary] If not set, a default 
                        dictionary (see default values below) will be created. 
                        It must have the follwoing keys and values:
                        'preX'  [list of numpy arrays] It contains pre-cross-
                                multiplication weights. It is a list where 
                                each element in the list is a numpy array, and
                                the number of elements in the list must match 
                                the number of entries in key 'axes'. If 'axes'
                                is set None, 'preX' may be set to a list 
                                with one element which is a numpy array of ones.
                                The number of elements in each of the numpy 
                                arrays must be numpy broadcastable into the 
                                number of elements along that axis in the 
                                delay spectrum.
                        'preXnorm'
                                [boolean] If False (default), no normalization
                                is done after the application of weights. If 
                                set to True, the delay spectrum will be 
                                normalized by the sum of the weights. 
                        'postX' [list of numpy arrays] It contains post-cross-
                                multiplication weights. It is a list where 
                                each element in the list is a numpy array, and
                                the number of elements in the list must match 
                                the number of entries in key 'axes'. If 'axes'
                                is set None, 'preX' may be set to a list 
                                with one element which is a numpy array of ones.
                                The number of elements in each of the numpy 
                                arrays must be numpy broadcastable into the 
                                number of elements along that axis in the 
                                delay spectrum. 
                        'preXnorm'
                                [boolean] If False (default), no normalization
                                is done after the application of 'preX' weights. 
                                If set to True, the delay spectrum will be 
                                normalized by the sum of the weights. 
                        'postXnorm'
                                [boolean] If False (default), no normalization
                                is done after the application of postX weights. 
                                If set to True, the delay cross power spectrum 
                                will be normalized by the sum of the weights. 

        cosmo   [instance of cosmology class from astropy] An instance of class
                FLRW or default_cosmology of astropy cosmology module. Default
                uses Planck 2015 cosmology, with H0=100 h km/s/Mpc

        units   [string] Specifies the units of output power spectum. Accepted
                values are 'Jy' and 'K' (default)) and the power spectrum will 
                be in corresponding squared units.

        Output:

        Dictionary with the keys 'triads' ((ntriads,3) array), 'triads_ind', 
        ((ntriads,) array), 'lstXoffsets' ((ndlst_range,) array), 'lst' 
        ((nlst,) array), 'dlst' ((nlst,) array), 'lst_ind' ((nlst,) array), 
        'days' ((ndaycomb,) array), 'day_ind' ((ndaycomb,) array), 'dday' 
        ((ndaycomb,) array), 'oversampled' and 'resampled' corresponding to 
        whether resample was set to False or True in call to member function 
        FT(). Values under keys 'triads_ind' and 'lst_ind' are numpy array 
        corresponding to triad and time indices used in selecting the data. 
        Values under keys 'oversampled' and 'resampled' each contain a 
        dictionary with the following keys and values:
        'z'     [numpy array] Redshifts corresponding to the band centers in 
                'freq_center'. It has shape=(nspw,)
        'lags'  [numpy array] Delays (in seconds). It has shape=(nlags,).
        'kprll' [numpy array] k_parallel modes (in h/Mpc) corresponding to 
                'lags'. It has shape=(nspw,nlags)
        'freq_center'   
                [numpy array] contains the center frequencies (in Hz) of the 
                frequency subbands of the subband delay spectra. It is of size 
                n_win. It is roughly equivalent to redshift(s)
        'freq_wts'      
                [numpy array] Contains frequency weights applied on each 
                frequency sub-band during the subband delay transform. It is 
                of size n_win x nchan. 
        'bw_eff'        
                [numpy array] contains the effective bandwidths (in Hz) of the 
                subbands being delay transformed. It is of size n_win. It is 
                roughly equivalent to width in redshift or along line-of-sight
        'shape' [string] shape of the frequency window function applied. Usual
                values are 'rect' (rectangular), 'bhw' (Blackman-Harris), 
                'bnw' (Blackman-Nuttall). 
        'fftpow'
                [scalar] the power to which the FFT of the window was raised. 
                The value is be a positive scalar with default = 1.0
        'lag_corr_length' 
                [numpy array] It is the correlation timescale (in pixels) of 
                the subband delay spectra. It is proportional to inverse of 
                effective bandwidth. It is of size n_win. The unit size of a 
                pixel is determined by the difference between adjacent pixels 
                in lags under key 'lags' which in turn is effectively inverse 
                of the effective bandwidth of the subband specified in bw_eff

        It further contains a key named 'errinfo' which is a dictionary. It 
        contains information about power spectrum uncertainties obtained from 
        subsample differences. It contains the following keys and values:
        'mean'  [numpy array] Delay power spectrum uncertainties incoherently 
                estimated over the axes specified in xinfo['axes'] using the 
                'mean' key in input cpds or attribute 
                cPhaseDS['errinfo']['dspec']. It has shape that depends on the 
                combination of input parameters. See examples below. If both 
                collapse_axes and avgcov are not set, those axes will be 
                replaced with square covariance matrices. If collapse_axes is 
                provided but avgcov is False, those axes will be of shape 
                2*Naxis-1. 
        'median'
                [numpy array] Delay power spectrum uncertainties incoherently 
                averaged over the axes specified in incohax using the 'median' 
                key in input cpds or attribute cPhaseDS['errinfo']['dspec']. 
                It has shape that depends on the combination of input 
                parameters. See examples below. If both collapse_axes and 
                avgcov are not set, those axes will be replaced with square 
                covariance matrices. If collapse_axes is provided but avgcov is 
                False, those axes will be of shape 2*Naxis-1. 
        'diagoffsets' 
                [dictionary] Same keys corresponding to keys under 
                'collapse_axes' in input containing the diagonal offsets for
                those axes. If 'avgcov' was set, those entries will be removed 
                from 'diagoffsets' since all the leading diagonal elements have 
                been collapsed (averaged) further. Value under each key is a 
                numpy array where each element in the array corresponds to the 
                index of that leading diagonal. This should match the size of 
                the output along that axis in 'mean' or 'median' above. 
        'diagweights'
                [dictionary] Each key is an axis specified in collapse_axes and
                the value is a numpy array of weights corresponding to the 
                diagonal offsets in that axis.
        'axesmap'
                [dictionary] If covariance in cross-power is calculated but is 
                not collapsed, the number of dimensions in the output will have
                changed. This parameter tracks where the original axis is now 
                placed. The keys are the original axes that are involved in 
                incoherent cross-power, and the values are the new locations of 
                those original axes in the output. 
        'nsamples_incoh'
                [integer] Number of incoherent samples in producing the power
                spectrum
        'nsamples_coh'
                [integer] Number of coherent samples in producing the power
                spectrum

        Examples: 

        (1)
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': 2, 'wts': None}
        xinfo = {'axes': None, 'avgcov': False, 'collapse_axes': [], 
                 'wts':{'preX': None, 'preXnorm': False, 
                        'postX': None, 'postXnorm': False}}
        This will not do anything because axes cannot include value 2 which 
        denote the 'days' axis and the uncertainties are obtained through 
        subsample differencing along days axis regardless. 
        Output delay power spectrum has shape (Nspw, Nlst, Ndaycomb, Ntriads, 
        Nlags)

        (2) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': 2, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': False, 'collapse_axes': [], 
                 'wts':{'preX': None, 'preXnorm': False, 
                        'postX': None, 'postXnorm': False}, 
                 'dlst_range': None}
        This will not do anything about coherent averaging along axis=2 because 
        axes cannot include value 2 which denote the 'days' axis and the 
        uncertainties are obtained through subsample differencing along days 
        axis regardless.         
        Output delay power spectrum has shape 
        (Nspw, 2, Nlst, Ndaycomb, Ntriads, Ntriads, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range)}, 
        axesmap = {1: [1,2], 3: [4,5]}

        (3) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': 2, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': False, 'collapse_axes': [3], 
                 'dlst_range': [0.0, 1.0, 2.0]}
        This will not do anything about coherent averaging along axis=2 because 
        axes cannot include value 2 which denote the 'days' axis and the 
        uncertainties are obtained through subsample differencing along days 
        axis regardless.         
        Output delay power spectrum has shape 
        (Nspw, 3, Nlst, 1, 2*Ntriads-1, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range), 
                       3: NP.arange(-Ntriads,Ntriads)}, 
        axesmap = {1: [1,2], 3: [4]}

        (4) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': None, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': False, 'collapse_axes': [1,3], 
                 'dlst_range': [1.0, 2.0, 3.0, 4.0]}
        Output delay power spectrum has shape 
        (Nspw, 4, Ndaycomb, 2*Ntriads-1, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range), 
                       3: NP.arange(-Ntriads,Ntriads)}, 
        axesmap = {1: [1], 3: [3]}

        (5) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': None, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': True, 'collapse_axes': [3], 
                 'dlst_range': None}
        Output delay power spectrum has shape 
        (Nspw, 2, Nlst, Ndays, 1, Nlags)
        diagoffsets = {1: NP.arange(n_dlst_range)}, axesmap = {1: [1,2], 3: [4]}

        (6) 
        Input delay spectrum of shape (Nspw, Nlst, Ndays, Ntriads, Nlags)
        autoinfo = {'axes': None, 'wts': None}
        xinfo = {'axes': [1,3], 'avgcov': True, 'collapse_axes': []}
        Output delay power spectrum has shape 
        (Nspw, 1, Ndays, 1, Nlags)
        diagoffsets = {}, axesmap = {1: [1], 3: [3]}
        ------------------------------------------------------------------------
        """

        if not isinstance(units,str):
            raise TypeError('Input parameter units must be a string')
        if units.lower() == 'k':
            if not isinstance(beamparms, dict):
                raise TypeError('Input beamparms must be a dictionary')
            if 'freqs' not in beamparms:
                beamparms['freqs'] = self.f
            beamparms_orig = copy.deepcopy(beamparms)

        if autoinfo is None:
            autoinfo = {'axes': None, 'wts': [NP.ones(1, dtpye=NP.float)]}
        elif not isinstance(autoinfo, dict):
            raise TypeError('Input autoinfo must be a dictionary')

        if 'axes' not in autoinfo:
            autoinfo['axes'] = None
        else:
            if autoinfo['axes'] is not None:
                if not isinstance(autoinfo['axes'], (list,tuple,NP.ndarray,int)):
                    raise TypeError('Value under key axes in input autoinfo must be an integer, list, tuple or numpy array')
                else:
                    autoinfo['axes'] = NP.asarray(autoinfo['axes']).reshape(-1)

        if 'wts' not in autoinfo:
            if autoinfo['axes'] is not None:
                autoinfo['wts'] = [NP.ones(1, dtype=NP.float)] * len(autoinfo['axes'])
            else:
                autoinfo['wts'] = [NP.ones(1, dtype=NP.float)]
        else:
            if autoinfo['axes'] is not None:
                if not isinstance(autoinfo['wts'], list):
                    raise TypeError('wts in input autoinfo must be a list of numpy arrays')
                else:
                    if len(autoinfo['wts']) != len(autoinfo['axes']):
                        raise ValueError('Input list of wts must be same as length of autoinfo axes')
            else:
                autoinfo['wts'] = [NP.ones(1, dtype=NP.float)]

        if xinfo is None:
            xinfo = {'axes': None, 'wts': {'preX': [NP.ones(1, dtpye=NP.float)], 'postX': [NP.ones(1, dtpye=NP.float)], 'preXnorm': False, 'postXnorm': False}}
        elif not isinstance(xinfo, dict):
            raise TypeError('Input xinfo must be a dictionary')

        if 'axes' not in xinfo:
            xinfo['axes'] = None
        else:
            if not isinstance(xinfo['axes'], (list,tuple,NP.ndarray,int)):
                raise TypeError('Value under key axes in input xinfo must be an integer, list, tuple or numpy array')
            else:
                xinfo['axes'] = NP.asarray(xinfo['axes']).reshape(-1)

        if 'wts' not in xinfo:
            xinfo['wts'] = {}
            for xkey in ['preX', 'postX']:
                if xinfo['axes'] is not None:
                    xinfo['wts'][xkey] = [NP.ones(1, dtype=NP.float)] * len(xinfo['axes'])
                else:
                    xinfo['wts'][xkey] = [NP.ones(1, dtype=NP.float)]
            xinfo['wts']['preXnorm'] = False
            xinfo['wts']['postXnorm'] = False
        else:
            if xinfo['axes'] is not None:
                if not isinstance(xinfo['wts'], dict):
                    raise TypeError('wts in input xinfo must be a dictionary')
                for xkey in ['preX', 'postX']:
                    if not isinstance(xinfo['wts'][xkey], list):
                        raise TypeError('{0} wts in input xinfo must be a list of numpy arrays'.format(xkey))
                    else:
                        if len(xinfo['wts'][xkey]) != len(xinfo['axes']):
                            raise ValueError('Input list of {0} wts must be same as length of xinfo axes'.format(xkey))
            else:
                for xkey in ['preX', 'postX']:
                    xinfo['wts'][xkey] = [NP.ones(1, dtype=NP.float)]

            if 'preXnorm' not in xinfo['wts']:
                xinfo['wts']['preXnorm'] = False
            if 'postXnorm' not in xinfo['wts']:
                xinfo['wts']['postXnorm'] = False
            if not isinstance(xinfo['wts']['preXnorm'], NP.bool):
                raise TypeError('preXnorm in input xinfo must be a boolean')
            if not isinstance(xinfo['wts']['postXnorm'], NP.bool):
                raise TypeError('postXnorm in input xinfo must be a boolean')

        if 'avgcov' not in xinfo:
            xinfo['avgcov'] = False
        if not isinstance(xinfo['avgcov'], NP.bool):
            raise TypeError('avgcov under input xinfo must be boolean')

        if 'collapse_axes' not in xinfo:
            xinfo['collapse_axes'] = []
        if not isinstance(xinfo['collapse_axes'], (int,list,tuple,NP.ndarray)):
            raise TypeError('collapse_axes under input xinfo must be an integer, tuple, list or numpy array')
        else:
            xinfo['collapse_axes'] = NP.asarray(xinfo['collapse_axes']).reshape(-1)

        if (autoinfo['axes'] is not None) and (xinfo['axes'] is not None):
            if NP.intersect1d(autoinfo['axes'], xinfo['axes']).size > 0:
                raise ValueError("Inputs autoinfo['axes'] and xinfo['axes'] must have no intersection")

        cohax = autoinfo['axes']
        if cohax is None:
            cohax = []
        if 2 in cohax: # Remove axis=2 from cohax
            if isinstance(cohax, list):
                cohax.remove(2)
            if isinstance(cohax, NP.ndarray):
                cohax = cohax.tolist()
                cohax.remove(2)
                cohax = NP.asarray(cohax)
        incohax = xinfo['axes']
        if incohax is None:
            incohax = []
        if 2 in incohax: # Remove axis=2 from incohax
            if isinstance(incohax, list):
                incohax.remove(2)
            if isinstance(incohax, NP.ndarray):
                incohax = incohax.tolist()
                incohax.remove(2)
                incohax = NP.asarray(incohax)

        if selection is None:
            selection = {'triads': None, 'lst': None, 'days': None}
        else:
            if not isinstance(selection, dict):
                raise TypeError('Input selection must be a dictionary')

        if cpds is None:
            cpds = {}
            sampling = ['oversampled', 'resampled']
            for smplng in sampling:
                if smplng == 'oversampled':
                    cpds[smplng] = copy.deepcopy(self.cPhaseDS)
                else:
                    cpds[smplng] = copy.deepcopy(self.cPhaseDS_resampled)

        triad_ind, lst_ind, day_ind, day_ind_eicpdiff = self.subset(selection=selection)

        result = {'triads': self.cPhase.cpinfo['raw']['triads'][triad_ind], 'triads_ind': triad_ind, 'lst': self.cPhase.cpinfo['errinfo']['lstbins'][lst_ind], 'lst_ind': lst_ind, 'dlst': self.cPhase.cpinfo['errinfo']['dlstbins'][lst_ind], 'days': self.cPhase.cpinfo['errinfo']['daybins'][day_ind], 'day_ind': day_ind_eicpdiff, 'dday': self.cPhase.cpinfo['errinfo']['diff_dbins'][day_ind]}

        dlstbin = NP.mean(self.cPhase.cpinfo['errinfo']['dlstbins'])
        if 'dlst_range' in xinfo:
            if xinfo['dlst_range'] is None:
                dlst_range = None
                lstshifts = NP.arange(2) # LST index offsets of 0 and 1 are only estimated
            else:
                dlst_range = NP.asarray(xinfo['dlst_range']).ravel() / 60.0 # Difference in LST between a pair of LST (in hours)
                if dlst_range.size == 1:
                    dlst_range = NP.insert(dlst_range, 0, 0.0)
                lstshifts = NP.arange(max([0, NP.ceil(1.0*dlst_range.min()/dlstbin).astype(NP.int)]), min([NP.ceil(1.0*dlst_range.max()/dlstbin).astype(NP.int), result['lst'].size]))
        else:
            dlst_range = None
            lstshifts = NP.arange(2) # LST index offsets of 0 and 1 are only estimated
        result['lstXoffsets'] = lstshifts * dlstbin # LST interval corresponding to diagonal offsets created by the LST covariance

        for smplng in sampling:
            result[smplng] = {}
                
            wl = FCNST.c / (cpds[smplng]['freq_center'] * U.Hz)
            z = CNST.rest_freq_HI / cpds[smplng]['freq_center'] - 1
            dz = CNST.rest_freq_HI / cpds[smplng]['freq_center']**2 * cpds[smplng]['bw_eff']
            dkprll_deta = DS.dkprll_deta(z, cosmo=cosmo)
            kprll = dkprll_deta.reshape(-1,1) * cpds[smplng]['lags']

            rz_los = cosmo.comoving_distance(z) # in Mpc/h
            drz_los = FCNST.c * cpds[smplng]['bw_eff']*U.Hz * (1+z)**2 / (CNST.rest_freq_HI * U.Hz) / (cosmo.H0 * cosmo.efunc(z))   # in Mpc/h
            if units == 'Jy':
                jacobian1 = 1 / (cpds[smplng]['bw_eff'] * U.Hz)
                jacobian2 = drz_los / (cpds[smplng]['bw_eff'] * U.Hz)
                temperature_from_fluxdensity = 1.0
            elif units == 'K':
                beamparms = copy.deepcopy(beamparms_orig)
                omega_bw = self.beam3Dvol(beamparms, freq_wts=cpds[smplng]['freq_wts'])
                jacobian1 = 1 / (omega_bw * U.Hz) # The steradian is present but not explicitly assigned
                jacobian2 = rz_los**2 * drz_los / (cpds[smplng]['bw_eff'] * U.Hz)
                temperature_from_fluxdensity = wl**2 / (2*FCNST.k_B)
            else:
                raise ValueError('Input value for units invalid')

            factor = jacobian1 * jacobian2 * temperature_from_fluxdensity**2

            result[smplng]['z'] = z
            result[smplng]['kprll'] = kprll
            result[smplng]['lags'] = NP.copy(cpds[smplng]['lags'])
            result[smplng]['freq_center'] = cpds[smplng]['freq_center']
            result[smplng]['bw_eff'] = cpds[smplng]['bw_eff']
            result[smplng]['shape'] = cpds[smplng]['shape']
            result[smplng]['freq_wts'] = cpds[smplng]['freq_wts']
            result[smplng]['lag_corr_length'] = cpds[smplng]['lag_corr_length']

            dpool = 'errinfo'
            if dpool in cpds[smplng]:
                result[smplng][dpool] = {}
                inpshape = list(cpds[smplng][dpool]['dspec0']['mean'].shape)
                inpshape[1] = lst_ind.size
                inpshape[2] = day_ind_eicpdiff.size
                inpshape[3] = triad_ind.size
                if len(cohax) > 0:
                    nsamples_coh = NP.prod(NP.asarray(inpshape)[NP.asarray(cohax)])
                else:
                    nsamples_coh = 1
                if len(incohax) > 0:
                    nsamples = NP.prod(NP.asarray(inpshape)[NP.asarray(incohax)])
                    nsamples_incoh = nsamples * (nsamples - 1)
                else:
                    nsamples_incoh = 1
                twts_multidim_idx = NP.ix_(lst_ind,day_ind_eicpdiff,triad_ind,NP.arange(1)) # shape=(nlst,ndays,ntriads,1)
                dspec_multidim_idx = NP.ix_(NP.arange(wl.size),lst_ind,day_ind_eicpdiff,triad_ind,NP.arange(inpshape[4])) # shape=(nspw,nlst,ndays,ntriads,nchan)
                max_wt_in_chan = NP.max(NP.sum(cpds[smplng]['errinfo']['dspec0']['twts'].data, axis=(0,1,2,3)))
                select_chan = NP.argmax(NP.sum(cpds[smplng]['errinfo']['dspec0']['twts'].data, axis=(0,1,2,3)))
                twts = {'0': NP.copy(cpds[smplng]['errinfo']['dspec0']['twts'].data[:,:,:,[select_chan]]), '1': NP.copy(cpds[smplng]['errinfo']['dspec1']['twts'].data[:,:,:,[select_chan]])}

                if nsamples_coh > 1:
                    awts_shape = tuple(NP.ones(cpds[smplng]['errinfo']['dspec']['mean'].ndim, dtype=NP.int))
                    awts = NP.ones(awts_shape, dtype=NP.complex)
                    awts_shape = NP.asarray(awts_shape)
                    for caxind,caxis in enumerate(cohax):
                        curr_awts_shape = NP.copy(awts_shape)
                        curr_awts_shape[caxis] = -1
                        awts = awts * autoinfo['wts'][caxind].reshape(tuple(curr_awts_shape))

                for stat in ['mean', 'median']:
                    dspec0 = NP.copy(cpds[smplng][dpool]['dspec0'][stat][dspec_multidim_idx])
                    dspec1 = NP.copy(cpds[smplng][dpool]['dspec1'][stat][dspec_multidim_idx])
                    if nsamples_coh > 1:
                        if stat == 'mean':
                            dspec0 = NP.sum(twts['0'][NP.newaxis,...] * awts * dspec0, axis=cohax, keepdims=True) / NP.sum(twts['0'][twts_multidim_idx][NP.newaxis,...] * awts, axis=cohax, keepdims=True)
                            dspec1 = NP.sum(twts['1'][NP.newaxis,...] * awts * dspec1, axis=cohax, keepdims=True) / NP.sum(twts['1'][twts_multidim_idx][NP.newaxis,...] * awts, axis=cohax, keepdims=True)
                        else:
                            dspec0 = NP.median(dspec0, axis=cohax, keepdims=True)
                            dspec1 = NP.median(dspec1, axis=cohax, keepdims=True)
                    if nsamples_incoh > 1:
                        expandax_map = {}
                        wts_shape = tuple(NP.ones(dspec0.ndim, dtype=NP.int))
                        preXwts = NP.ones(wts_shape, dtype=NP.complex)
                        wts_shape = NP.asarray(wts_shape)
                        for incaxind,incaxis in enumerate(xinfo['axes']):
                            curr_wts_shape = NP.copy(wts_shape)
                            curr_wts_shape[incaxis] = -1
                            preXwts = preXwts * xinfo['wts']['preX'][incaxind].reshape(tuple(curr_wts_shape))
                        preXwts0 = NP.copy(preXwts)
                        preXwts1 = NP.copy(preXwts)
                        for incax in NP.sort(incohax)[::-1]:
                            dspec0 = NP.expand_dims(dspec0, axis=incax)
                            preXwts0 = NP.expand_dims(preXwts0, axis=incax)
                            if incax == 1:
                                preXwts0_outshape = list(preXwts0.shape)
                                preXwts0_outshape[incax+1] = dspec0.shape[incax+1]
                                preXwts0_outshape = tuple(preXwts0_outshape)
                                preXwts0 = NP.broadcast_to(preXwts0, preXwts0_outshape).copy() # For some strange reason the NP.broadcast_to() creates a "read-only" immutable array which is changed to writeable by copy()
                                
                                preXwts1_tmp = NP.expand_dims(preXwts1, axis=incax)
                                preXwts1_shape = NP.asarray(preXwts1_tmp.shape)
                                preXwts1_shape[incax] = lstshifts.size
                                preXwts1_shape[incax+1] = preXwts0_outshape[incax+1]
                                preXwts1_shape = tuple(preXwts1_shape)
                                preXwts1 = NP.broadcast_to(preXwts1_tmp, preXwts1_shape).copy() # For some strange reason the NP.broadcast_to() creates a "read-only" immutable array which is changed to writeable by copy()

                                dspec1_tmp = NP.expand_dims(dspec1, axis=incax)
                                dspec1_shape = NP.asarray(dspec1_tmp.shape)
                                dspec1_shape[incax] = lstshifts.size
                                # dspec1_shape = NP.insert(dspec1_shape, incax, lstshifts.size)
                                dspec1_shape = tuple(dspec1_shape)
                                dspec1 = NP.broadcast_to(dspec1_tmp, dspec1_shape).copy() # For some strange reason the NP.broadcast_to() creates a "read-only" immutable array which is changed to writeable by copy()
                                for lstshiftind, lstshift in enumerate(lstshifts):
                                    dspec1[:,lstshiftind,...] = NP.roll(dspec1_tmp[:,0,...], lstshift, axis=incax)
                                    dspec1[:,lstshiftind,:lstshift,...] = NP.nan
                                    preXwts1[:,lstshiftind,...] = NP.roll(preXwts1_tmp[:,0,...], lstshift, axis=incax)
                                    preXwts1[:,lstshiftind,:lstshift,...] = NP.nan
                            else:
                                dspec1 = NP.expand_dims(dspec1, axis=incax+1)
                                preXwts1 = NP.expand_dims(preXwts1, axis=incax+1)
                            expandax_map[incax] = incax + NP.arange(2)
                            for ekey in expandax_map:
                                if ekey > incax:
                                    expandax_map[ekey] += 1
                                    
                        result[smplng][dpool][stat] = factor.reshape((-1,)+tuple(NP.ones(dspec0.ndim-1, dtype=NP.int))) * (dspec0*U.Unit('Jy Hz') * preXwts0) * (dspec1*U.Unit('Jy Hz') * preXwts1).conj()
                        if xinfo['wts']['preXnorm']:
                            result[smplng][dpool][stat] = result[smplng][dpool][stat] / NP.nansum(preXwts0 * preXwts1.conj(), axis=NP.union1d(NP.where(logical_or(NP.asarray(preXwts0.shape)>1, NP.asarray(preXwts1.shape)>1))), keepdims=True) # Normalize by summing the weights over the expanded axes
        
                        if (len(xinfo['collapse_axes']) > 0) or (xinfo['avgcov']):
                            # Remove axis=2 if present
                            if 2 in xinfo['collapse_axes']:
                                # Remove axis=2 from cohax
                                if isinstance(xinfo['collapse_axes'], list):
                                    xinfo['collapse_axes'].remove(2)
                                if isinstance(xinfo['collapse_axes'], NP.ndarray):
                                    xinfo['collapse_axes'] = xinfo['collapse_axes'].tolist()
                                    xinfo['collapse_axes'].remove(2)
                                    xinfo['collapse_axes'] = NP.asarray(xinfo['collapse_axes'])

                        if (len(xinfo['collapse_axes']) > 0) or (xinfo['avgcov']):
                            # if any one of collapsing of incoherent axes or 
                            # averaging of full covariance is requested
        
                            diagoffsets = {} # Stores the correlation index difference along each axis.
                            diagweights = {} # Stores the number of points summed in the trace along the offset diagonal
                            for colaxind, colax in enumerate(xinfo['collapse_axes']):
                                if colax == 1:
                                    shp = NP.ones(cpds[smplng][dpool]['dspec0'][stat].ndim, dtype=NP.int)
                                    shp[colax] = lst_ind.size
                                    multdim_idx = tuple([NP.arange(axdim) for axdim in shp])
                                    diagweights[colax] = NP.sum(NP.logical_not(NP.isnan(cpds[smplng][dpool]['dspec0'][stat][dspec_multidim_idx][multdim_idx]))) - lstshifts
                                    # diagweights[colax] = result[smplng][dpool][stat].shape[expandax_map[colax][-1]] - lstshifts

                                    if stat == 'mean':
                                        result[smplng][dpool][stat] = NP.nanmean(result[smplng][dpool][stat], axis=expandax_map[colax][-1])
                                    else:
                                        result[smplng][dpool][stat] = NP.nanmedian(result[smplng][dpool][stat], axis=expandax_map[colax][-1])
                                    diagoffsets[colax] = lstshifts
                                else:
                                    pspec_unit = result[smplng][dpool][stat].si.unit
                                    result[smplng][dpool][stat], offsets, diagwts = OPS.array_trace(result[smplng][dpool][stat].si.value, offsets=None, axis1=expandax_map[colax][0], axis2=expandax_map[colax][1], outaxis='axis1')
                                    diagwts_shape = NP.ones(result[smplng][dpool][stat].ndim, dtype=NP.int)
                                    diagwts_shape[expandax_map[colax][0]] = diagwts.size
                                    diagoffsets[colax] = offsets
                                    diagweights[colax] = NP.copy(diagwts)
                                    result[smplng][dpool][stat] = result[smplng][dpool][stat] * pspec_unit / diagwts.reshape(diagwts_shape)
                                for ekey in expandax_map:
                                    if ekey > colax:
                                        expandax_map[ekey] -= 1
                                expandax_map[colax] = NP.asarray(expandax_map[colax][0]).ravel()
        
                            wts_shape = tuple(NP.ones(result[smplng][dpool][stat].ndim, dtype=NP.int))
                            postXwts = NP.ones(wts_shape, dtype=NP.complex)
                            wts_shape = NP.asarray(wts_shape)
                            for colaxind, colax in enumerate(xinfo['collapse_axes']):
                                curr_wts_shape = NP.copy(wts_shape)
                                curr_wts_shape[expandax_map[colax]] = -1
                                postXwts = postXwts * xinfo['wts']['postX'][colaxind].reshape(tuple(curr_wts_shape))
                                
                            result[smplng][dpool][stat] = result[smplng][dpool][stat] * postXwts
        
                            axes_to_sum = tuple(NP.asarray([expandax_map[colax] for colax in xinfo['collapse_axes']]).ravel()) # for post-X normalization and collapse of covariance matrix
        
                            if xinfo['wts']['postXnorm']:
                                result[smplng][dpool][stat] = result[smplng][dpool][stat] / NP.nansum(postXwts, axis=axes_to_sum, keepdims=True) # Normalize by summing the weights over the collapsed axes
                            if xinfo['avgcov']:
        
                                # collapse the axes further (postXwts have already
                                # been applied)
        
                                diagoffset_weights = 1.0
                                result[smplng][dpool][stat] = NP.nanmean(result[smplng][dpool][stat], axis=axes_to_sum, keepdims=True)
                                for colaxind in zip(*sorted(zip(NP.arange(xinfo['collapse_axes'].size), xinfo['collapse_axes']), reverse=True))[0]:
        
                                    # It is import to sort the collapsable axes in
                                    # reverse order before deleting elements below,
                                    # otherwise the axes ordering may be get messed up
        
                                    diagoffset_weights_shape = NP.ones(result[smplng][dpool][stat].ndim, dtype=NP.int)
                                    diagoffset_weights_shape[expandax_map[xinfo['collapse_axes'][colaxind]][0]] = diagweights[xinfo['collapse_axes'][colaxind]].size
                                    diagoffset_weights = diagoffset_weights * diagweights[xinfo['collapse_axes'][colaxind]].reshape(diagoffset_weights_shape)
                                    del diagoffsets[xinfo['collapse_axes'][colaxind]]
                                result[smplng][dpool][stat] = NP.nansum(result[smplng][dpool][stat]*diagoffset_weights, axis=axes_to_sum, keepdims=True) / NP.nansum(diagoffset_weights, axis=axes_to_sum, keepdims=True)
                    else:
                        result[smplng][dpool][stat] = factor.reshape((-1,)+tuple(NP.ones(dspec.ndim-1, dtype=NP.int))) * NP.abs(dspec * U.Jy)**2
                        diagoffsets = {}
                        expandax_map = {}                            
                        
                if units == 'Jy':
                    result[smplng][dpool][stat] = result[smplng][dpool][stat].to('Jy2 Mpc')
                elif units == 'K':
                    result[smplng][dpool][stat] = result[smplng][dpool][stat].to('K2 Mpc3')
                else:
                    raise ValueError('Input value for units invalid')
                result[smplng][dpool]['diagoffsets'] = diagoffsets
                result[smplng][dpool]['diagweights'] = diagweights
                result[smplng][dpool]['axesmap'] = expandax_map

                result[smplng][dpool]['nsamples_incoh'] = nsamples_incoh
                result[smplng][dpool]['nsamples_coh'] = nsamples_coh

        return result

    ############################################################################

    def rescale_power_spectrum(self, cpdps, visfile, blindex, visunits='Jy'):

        """
        ------------------------------------------------------------------------
        Rescale power spectrum to dimensional quantity by converting the ratio
        given visibility amplitude information

        Inputs:

        cpdps       [dictionary] Dictionary with the keys 'triads', 
                    'triads_ind', 'lstbins', 'lst', 'dlst', 'lst_ind', 
                    'oversampled' and 'resampled' corresponding to whether 
                    resample was set to False or True in call to member function 
                    FT(). Values under keys 'triads_ind' and 'lst_ind' are numpy 
                    array corresponding to triad and time indices used in 
                    selecting the data. Values under keys 'oversampled' and 
                    'resampled' each contain a dictionary with the following keys 
                    and values:
                    'z'     [numpy array] Redshifts corresponding to the band 
                            centers in 'freq_center'. It has shape=(nspw,)
                    'lags'  [numpy array] Delays (in seconds). It has 
                            shape=(nlags,).
                    'kprll' [numpy array] k_parallel modes (in h/Mpc) 
                            corresponding to 'lags'. It has shape=(nspw,nlags)
                    'freq_center'   
                            [numpy array] contains the center frequencies (in 
                            Hz) of the frequency subbands of the subband delay 
                            spectra. It is of size n_win. It is roughly 
                            equivalent to redshift(s)
                    'freq_wts'      
                            [numpy array] Contains frequency weights applied on 
                            each frequency sub-band during the subband delay 
                            transform. It is of size n_win x nchan. 
                    'bw_eff'        
                            [numpy array] contains the effective bandwidths (in 
                            Hz) of the subbands being delay transformed. It is 
                            of size n_win. It is roughly equivalent to width in 
                            redshift or along line-of-sight
                    'shape' [string] shape of the frequency window function 
                            applied. Usual values are 'rect' (rectangular), 
                            'bhw' (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                    'fftpow'
                            [scalar] the power to which the FFT of the window 
                            was raised. 
                            The value is be a positive scalar with default = 1.0
                    'mean'  [numpy array] Delay power spectrum incoherently 
                            averaged over the axes specified in incohax using 
                            the 'mean' key in input cpds or attribute 
                            cPhaseDS['processed']['dspec']. It has 
                            shape=(nspw,nlst,ndays,ntriads,nchan). It has units 
                            of Mpc/h. If incohax was set, those axes will be set 
                            to 1.
                    'median'
                            [numpy array] Delay power spectrum incoherently 
                            averaged over the axes specified in incohax using 
                            the 'median' key in input cpds or attribute 
                            cPhaseDS['processed']['dspec']. It has 
                            shape=(nspw,nlst,ndays,ntriads,nchan). It has units 
                            of Mpc/h. If incohax was set, those axes will be set 
                            to 1.

        visfile     [string] Full path to the visibility file in NPZ format that
                    consists of the following keys and values:
                    'vis'   [numpy array] Complex visibilities averaged over 
                            all redundant baselines of different classes of 
                            baselines. It is of shape (nlst,nbl,nchan)
                    'last'  [numpy array] Array of LST in units of days where
                            the fractional part is LST in days.
                            
        blindex     [numpy array] 3-element array of baseline indices to use in
                    selecting the triad corresponding to closure phase power
                    spectrum in cpdps. It will index into the 'vis' array in
                    NPZ file visfile 

        visunits    [string] Units of visibility in visfile. Accepted values
                    are 'Jy' (default; for Jansky) and 'K' (for Kelvin)

        Outputs:

        Same dictionary as input cpdps except it has the following additional
        keys and values. Under 'resampled' and 'oversampled' keys, there are 
        now new keys called 'mean-absscale' and 'median-absscale' keys which 
        are each dictionaries with the following keys and values: 
        'converted' [numpy array] Values of power (in units of visunits^2) with
                    same shape as the values under 'mean' and 'median' keys --
                    (nspw,nlst,ndays,ntriads,nchan) unless some of those axes
                    have already been averaged coherently or incoherently
        'units'     [string] Units of power in key 'converted'. Its values are
                    square of the input visunits -- 'Jy^2' or 'K^2'
        ------------------------------------------------------------------------
        """

        if not isinstance(cpdps, dict):
            raise TypeError('Input cpdps must be a dictionary')
        if not isinstance(visfile, str):
            raise TypeError('Input visfile must be a string containing full file path')
        if isinstance(blindex, NP.ndarray):
            raise TypeError('Input blindex must be a numpy array')
        if blindex.size != 3:
            raise ValueError('Input blindex must be a 3-element array')
        if not isinstance(visunits, str):
            raise TypeError('Input visunits must be a string')
        if visunits not in ['Jy', 'K']:
            raise ValueError('Input visunits currently not accepted')

        datapool = []
        for dpool in ['resampled', 'oversampled']:
            if dpool in cpdps:
                datapool += [dpool]
        scaleinfo = NP.load(visfile)
        
        vis = scaleinfo['vis'][:,blindex,:] # shape=(nlst,nbl,nchan)
        vis_lstfrac, vis_lstint = NP.modf(scaleinfo['last']) # shape=(nlst,)
        vis_lstHA = vis_lstfrac * 24.0 # in hours
        vis_lstdeg = vis_lstHA * 15.0 # in degrees
        cpdps_lstdeg = 15.0*cpdps['lst'] # in degrees
        lstmatrix = cpdps_lstdeg.reshape(-1,1) - vis_lstdeg.reshape(1,-1)
        lstmatrix[NP.abs(lstmatrix) > 180.0] -= 360.0
        ind_minlstsep = NP.argmin(NP.abs(lstmatrix), axis=1)

        vis_nearestLST = vis[blindex,ind_minlstsep,:] # nlst x nbl x nchan
        for dpool in datapool:
            freq_wts = cpdps[dpool]['freq_wts'] # nspw x nchan
            freqwtd_avgvis_nearestLST = NP.sum(freq_wts[:,NP.newaxis,NP.newaxis,:] * vis_nearestLST[NP.newaxis,:,:,:], axis=-1, keepdims=True) / NP.sum(freq_wts[:,NP.newaxis,NP.newaxis,:], axis=-1, keepdims=True) # nspw x nlst x nbl x (nchan=1)
            vis_square_multscalar = 1 / NP.sum(1/NP.abs(freqwtd_avgvis_nearestLST)**2, axis=2, keepdims=True) # nspw x nlst x (nbl=1) x (nchan=1)
            for stat in ['mean', 'median']:
                cpdps[dpool][stat+'-absscale'] = {}
                cpdps[dpool][stat+'-absscale']['converted'] = cpdps[dpool][stat] * vis_square_multscalar[:,:,NP.newaxis,:,:] # nspw x nlst x ndays x ntriads x nlags
                cpdps[dpool][stat+'-absscale']['units'] = '{0}^2'.format(visunits)

        return cpdps
            
    ############################################################################

    def average_rescaled_power_spectrum(rcpdps, avgax, kprll_llim=None):

        """
        ------------------------------------------------------------------------
        Average the rescaled power spectrum with physical units along certain 
        axes with inverse variance or regular averaging

        Inputs:

        rcpdps      [dictionary] Dictionary with the keys 'triads', 
                    'triads_ind', 'lstbins', 'lst', 'dlst', 'lst_ind', 
                    'oversampled' and 'resampled' corresponding to whether 
                    resample was set to False or True in call to member function 
                    FT(). Values under keys 'triads_ind' and 'lst_ind' are numpy 
                    array corresponding to triad and time indices used in 
                    selecting the data. Values under keys 'oversampled' and 
                    'resampled' each contain a dictionary with the following keys 
                    and values:
                    'z'     [numpy array] Redshifts corresponding to the band 
                            centers in 'freq_center'. It has shape=(nspw,)
                    'lags'  [numpy array] Delays (in seconds). It has 
                            shape=(nlags,).
                    'kprll' [numpy array] k_parallel modes (in h/Mpc) 
                            corresponding to 'lags'. It has shape=(nspw,nlags)
                    'freq_center'   
                            [numpy array] contains the center frequencies (in 
                            Hz) of the frequency subbands of the subband delay 
                            spectra. It is of size n_win. It is roughly 
                            equivalent to redshift(s)
                    'freq_wts'      
                            [numpy array] Contains frequency weights applied on 
                            each frequency sub-band during the subband delay 
                            transform. It is of size n_win x nchan. 
                    'bw_eff'        
                            [numpy array] contains the effective bandwidths (in 
                            Hz) of the subbands being delay transformed. It is 
                            of size n_win. It is roughly equivalent to width in 
                            redshift or along line-of-sight
                    'shape' [string] shape of the frequency window function 
                            applied. Usual values are 'rect' (rectangular), 
                            'bhw' (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                    'fftpow'
                            [scalar] the power to which the FFT of the window 
                            was raised. 
                            The value is be a positive scalar with default = 1.0
                    'mean'  [numpy array] Delay power spectrum incoherently 
                            averaged over the axes specified in incohax using 
                            the 'mean' key in input cpds or attribute 
                            cPhaseDS['processed']['dspec']. It has 
                            shape=(nspw,nlst,ndays,ntriads,nchan). It has units 
                            of Mpc/h. If incohax was set, those axes will be set 
                            to 1.
                    'median'
                            [numpy array] Delay power spectrum incoherently 
                            averaged over the axes specified in incohax using 
                            the 'median' key in input cpds or attribute 
                            cPhaseDS['processed']['dspec']. It has 
                            shape=(nspw,nlst,ndays,ntriads,nchan). It has units 
                            of Mpc/h. If incohax was set, those axes will be set 
                            to 1.
                    'mean-absscale' and 'median-absscale'
                            [dictionary] Each dictionary consists of the 
                            following keys and values:
                            'converted' [numpy array] Values of power (in units 
                                        of value in key 'units') with same shape 
                                        as the values under 'mean' and 'median' 
                                        keys -- (nspw,nlst,ndays,ntriads,nchan) 
                                        unless some of those axes have already 
                                        been averaged coherently or incoherently
                            'units'     [string] Units of power in key 
                                        'converted'. Its values are square of 
                                        either 'Jy^2' or 'K^2'

        avgax       [int, list, tuple] Specifies the axes over which the power
                    in absolute scale (with physical units) should be averaged.
                    This counts as incoherent averaging. The averaging is done
                    with inverse-variance weighting if the input kprll_llim is
                    set to choose the range of kprll from which the variance 
                    and inverse variance will be determined. Otherwise, a 
                    regular averaging is performed. 

        kprll_llim  [float] Lower limit of absolute value of kprll (in Mpc/h) 
                    beyond which the variance will be determined in order to 
                    estimate the inverse variance weights. If set to None, the 
                    weights are uniform. If set to a value, values beyond this 
                    kprll_llim are used to estimate the variance and hence the
                    inverse-variance weights. 

        Outputs:

        Dictionary with the same structure as the input dictionary rcpdps except
        with the following additional keys and values. Under the dictionaries 
        under keys 'mean-absscale' and 'median-absscale', there is an additional
        key-value pair:
        'avg'   [numpy array] Values of power (in units of value in key 'units') 
                with same shape as the values under 'converted' -- 
                (nspw,nlst,ndays,ntriads,nchan) except those axes which were 
                averaged in this member function, and those axes will be 
                retained but with axis size=1. 
        ------------------------------------------------------------------------
        """

        if not isinstance(rcpdps, dict):
            raise TypeError('Input rcpdps must be a dictionary')

        if isinstance(avgax, int):
            if avgax >= 4:
                raise ValueError('Input avgax has a value greater than the maximum axis number over which averaging can be performed')
            avgax = NP.asarray(avgax)
        elif isinstance(avgax, (list,tuple)):
            avgax = NP.asarray(avgax)
            if NP.any(avgax >= 4):
                raise ValueError('Input avgax contains a value greater than the maximum axis number over which averaging can be performed')
        else:
            raise TypeError('Input avgax must be an integer, list, or tuple')

        if kprll_llim is not None:
            if not isinstance(kprll_llim, (int,float)):
                raise TypeError('Input kprll_llim must be a scalar')
            kprll_llim = NP.abs(kprll_llim)

        for dpool in datapool:
            for stat in ['mean', 'median']:
                wts = NP.ones((1,1,1,1,1))
                if kprll_llim is not None:
                    kprll_ind = NP.abs(rcpdps[dpool]['kprll']) >= kprll_llim # nspw x nlags
                    
                    if NP.any(kprll_ind):
                        if rcpdps[dpool]['z'].size > 1:
                            indsets = [NP.where(kprll_ind[i,:])[0] for i in range(rcpdps[dpool]['z'].size)]
                            common_kprll_ind = reduce(NP.intersect1d(indsets))
                            multidim_idx = NP.ix_(NP.arange(rcpdps[dpool]['freq_center'].size), NP.arange(rcpdps['lst'].size), NP.arange(rcpdps['days'].size), NP.arange(rcpdps['triads'].size), common_kprll_ind)
                        else:
                            multidim_idx = NP.ix_(NP.arange(rcpdps[dpool]['freq_center'].size), NP.arange(rcpdps['lst'].size), NP.arange(rcpdps['days'].size), NP.arange(rcpdps['triads'].size), kprll_ind[0,:])
                    else:
                        multidim_idx = NP.ix_(NP.arange(rcpdps[dpool]['freq_center'].size), NP.arange(rcpdps['lst'].size), NP.arange(rcpdps['days'].size), NP.arange(rcpdps['triads'].size), rcpdps[dpool]['lags'].size)
                    wts = 1 / NP.var(rcpdps[dpool][stat]['absscale']['rescale'][multidim_idx], axis=avgax, keepdims=True)
                rcpdps[dpool][stat]['absscale']['avg'] = NP.sum(wts * rcpdps[dpool][stat]['absscale']['rescale'], axis=avgax, keepdims=True) / NP.sum(wts, axis=avgax, keepdims=True)

        return rcpdps
        
    ############################################################################

    def beam3Dvol(self, beamparms, freq_wts=None):

        """
        ------------------------------------------------------------------------
        Compute three-dimensional (transverse-LOS) volume of the beam in units
        of "Sr Hz".
 
        Inputs:

        beamparms   [dictionary] Contains beam information. It contains the
                    following keys and values:
                    'beamfile'  [string] If set to string, should contain the
                                filename relative to default path or absolute 
                                path containing the power pattern. If both 
                                'beamfile' and 'telescope' are set, the 
                                'beamfile' will be used. The latter is used for 
                                determining analytic beam.
                    'filepathtype'
                                [string] Specifies if the beamfile is to be 
                                found at the 'default' location or a 'custom' 
                                location. If set to 'default', the PRISim path 
                                is searched for the beam file. Only applies if 
                                'beamfile' key is set.
                    'filefmt'   [string] External file format of the beam. 
                                Accepted values are 'uvbeam', 'fits' and 'hdf5'
                    'telescope' [dictionary] Information used to analytically 
                                determine the power pattern. used only if 
                                'beamfile' is not set or set to None. This 
                                specifies the type of element, its size and 
                                orientation. It consists of the following keys 
                                and values:
                        'id'        [string] If set, will ignore the other keys 
                                    and use telescope details for known 
                                    telescopes. Accepted values are 'mwa', 
                                    'vla', 'gmrt', 'hera', 'paper', 'hirax', 
                                    and 'chime' 
                        'shape'     [string] Shape of antenna element. Accepted 
                                    values are 'dipole', 'delta', 'dish', 
                                    'gaussian', 'rect' and 'square'. Will be 
                                    ignored if key 'id' is set. 'delta' denotes 
                                    a delta function for the antenna element 
                                    which has an isotropic radiation pattern. 
                                    'delta' is the default when keys 'id' and 
                                    'shape' are not set.
                        'size'      [scalar or 2-element list/numpy array] 
                                    Diameter of the telescope dish (in meters) 
                                    if the key 'shape' is set to 'dish', side 
                                    of the square aperture (in meters) if the 
                                    key 'shape' is set to 'square', 2-element 
                                    sides if key 'shape' is set to 'rect', or 
                                    length of the dipole if key 'shape' is set 
                                    to 'dipole'. Will be ignored if key 'shape' 
                                    is set to 'delta'. Will be ignored if key 
                                    'id' is set and a preset value used for the 
                                    diameter or dipole.
                        'orientation' 
                                    [list or numpy array] If key 'shape' is set 
                                    to dipole, it refers to the orientation of 
                                    the dipole element unit vector whose 
                                    magnitude is specified by length. If key 
                                    'shape' is set to 'dish', it refers to the 
                                    position on the sky to which the dish is 
                                    pointed. For a dipole, this unit vector must 
                                    be provided in the local ENU coordinate 
                                    system aligned with the direction cosines 
                                    coordinate system or in the Alt-Az 
                                    coordinate system. This will be used only 
                                    when key 'shape' is set to 'dipole'. This 
                                    could be a 2-element vector (transverse 
                                    direction cosines) where the third 
                                    (line-of-sight) component is determined, or 
                                    a 3-element vector specifying all three 
                                    direction cosines or a two-element 
                                    coordinate in Alt-Az system. If not provided 
                                    it defaults to an eastward pointing dipole. 
                                    If key 'shape' is set to 'dish' or 
                                    'gaussian', the orientation refers to the 
                                    pointing center of the dish on the sky. It 
                                    can be provided in Alt-Az system as a 
                                    two-element vector or in the direction 
                                    cosine coordinate system as a two- or 
                                    three-element vector. If not set in the case 
                                    of a dish element, it defaults to zenith. 
                                    This is not to be confused with the key 
                                    'pointing_center' in dictionary 
                                    'pointing_info' which refers to the 
                                    beamformed pointing center of the array. The 
                                    coordinate system is specified by the key 
                                    'ocoords'
                        'ocoords'   [string] specifies the coordinate system 
                                    for key 'orientation'. Accepted values are 
                                    'altaz' and 'dircos'. 
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
                                    phased array as far as determination of 
                                    primary beam is concerned.
                        'groundplane' 
                                    [scalar] height of telescope element above 
                                    the ground plane (in meteres). Default=None 
                                    will denote no ground plane effects.
                        'ground_modify'
                                    [dictionary] contains specifications to 
                                    modify the analytically computed ground 
                                    plane pattern. If absent, the ground plane 
                                    computed will not be modified. If set, it 
                                    may contain the following keys:
                            'scale'     [scalar] positive value to scale the 
                                        modifying factor with. If not set, the 
                                        scale factor to the modification is 
                                        unity.
                            'max'       [scalar] positive value to clip the 
                                        modified and scaled values to. If not 
                                        set, there is no upper limit

                    'freqs'     [numpy array] Numpy array denoting frequencies 
                                (in Hz) at which beam integrals are to be 
                                evaluated. If set to None, it will automatically 
                                be set from the class attribute. 
                    'nside'     [integer] NSIDE parameter for determining and
                                interpolating the beam. If not set, it will be
                                set to 64 (default).
                    'chromatic' [boolean] If set to true, a chromatic power 
                                pattern is used. If false, an achromatic power
                                pattern is used based on a reference frequency 
                                specified in 'select_freq'.
                    'select_freq'
                                [scalar] Selected frequency for the achromatic
                                beam. If not set, it will be determined to be
                                mean of the array in 'freqs'
                    'spec_interp'
                                [string] Method to perform spectral 
                                interpolation. Accepted values are those 
                                accepted in scipy.interpolate.interp1d() and
                                'fft'. Default='cubic'.
                                
        freq_wts    [numpy array] Frequency weights centered on different 
                    spectral windows or redshifts. Its shape is (nwin,nchan) 
                    and should match the number of spectral channels in input
                    parameter 'freqs' under 'beamparms' dictionary

        Output:

        omega_bw    [numpy array] Integral of the square of the power pattern
                    over transverse and spectral axes. Its shape is (nwin,)
        ------------------------------------------------------------------------
        """

        if not isinstance(beamparms, dict):
            raise TypeError('Input beamparms must be a dictionary')
        if ('beamfile' not in beamparms) and ('telescope' not in beamparms):
            raise KeyError('Input beamparms does not contain either "beamfile" or "telescope" keys')
        if 'freqs' not in beamparms:
            raise KeyError('Key "freqs" not found in input beamparms')
        if not isinstance(beamparms['freqs'], NP.ndarray):
            raise TypeError('Key "freqs" in input beamparms must contain a numpy array')
        if 'nside' not in beamparms:
            beamparms['nside'] = 64
        if not isinstance(beamparms['nside'], int):
            raise TypeError('"nside" parameter in input beamparms must be an integer')
        if 'chromatic' not in beamparms:
            beamparms['chromatic'] = True
        else:
            if not isinstance(beamparms['chromatic'], bool):
                raise TypeError('Beam chromaticity parameter in input beamparms must be a boolean')

        if beamparms['beamfile'] is not None:
            if 'filepathtype' in beamparms:
                if beamparms['filepathtype'] == 'default':
                    beamparms['beamfile'] = prisim_path+'data/beams/'+beamparms['beamfile']
            if 'filefmt' not in beamparms:
                raise KeyError('Input beam file format must be specified for an external beam')
            if beamparms['filefmt'].lower() in ['hdf5', 'fits', 'uvbeam']:
                beamparms['filefmt'] = beamparms['filefmt'].lower()
            else:
                raise ValueError('Invalid beam file format specified')
            
            if 'pol' not in beamparms:
                raise KeyError('Beam polarization must be specified')
            if not beamparms['chromatic']:
                if 'select_freq' not in beamparms:
                    raise KeyError('Input reference frequency for achromatic behavior must be specified')
                if beamparms['select_freq'] is None:
                    beamparms['select_freq'] = NP.mean(beamparms['freqs'])
                if 'spec_interp' not in beamparms:
                    beamparms['spec_interp'] = 'cubic'
            theta, phi = HP.pix2ang(beamparms['nside'], NP.arange(HP.nside2npix(beamparms['nside'])))
            theta_phi = NP.hstack((theta.reshape(-1,1), phi.reshape(-1,1)))
            if beamparms['beamfile'] :
                if beamparms['filefmt'] == 'fits':
                    external_beam = fits.getdata(beamparms['beamfile'], extname='BEAM_{0}'.format(beamparms['pol']))
                    external_beam_freqs = fits.getdata(beamparms['beamfile'], extname='FREQS_{0}'.format(beamparms['pol'])) # in MHz
                    external_beam = external_beam.reshape(-1,external_beam_freqs.size) # npix x nfreqs
                elif beamparms['filefmt'] == 'uvbeam':
                    if uvbeam_module_found:
                        uvbm = UVBeam()
                        uvbm.read_beamfits(beamparms['beamfile'])
                        axis_vec_ind = 0 # for power beam
                        spw_ind = 0 # spectral window index
                        if beamparms['pol'].lower() in ['x', 'e']:
                            beam_pol_ind = 0
                        else:
                            beam_pol_ind = 1
                        external_beam = uvbm.data_array[axis_vec_ind,spw_ind,beam_pol_ind,:,:].T # npix x nfreqs
                        external_beam_freqs = uvbm.freq_array.ravel() # nfreqs (in Hz)
                    else:
                        raise ImportError('uvbeam module not installed/found')
            
                    if NP.abs(NP.abs(external_beam).max() - 1.0) > 1e-10:
                        external_beam /= NP.abs(external_beam).max()
                else:
                    raise ValueError('Specified beam file format not currently supported')
                if beamparms['chromatic']:
                    if beamparms['spec_interp'] == 'fft':
                        external_beam = external_beam[:,:-1]
                        external_beam_freqs = external_beam_freqs[:-1]
                    interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(external_beam), theta_phi=theta_phi, inloc_axis=external_beam_freqs, outloc_axis=beamparms['freqs'], axis=1, kind=beamparms['spec_interp'], assume_sorted=True)
                else:
                    nearest_freq_ind = NP.argmin(NP.abs(external_beam_freqs - beamparms['select_freq']))
                    interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(NP.repeat(external_beam[:,nearest_freq_ind].reshape(-1,1), beamparms['freqs'].size, axis=1)), theta_phi=theta_phi, inloc_axis=beamparms['freqs'], outloc_axis=beamparms['freqs'], axis=1, assume_sorted=True)
                interp_logbeam_max = NP.nanmax(interp_logbeam, axis=0)
                interp_logbeam_max[interp_logbeam_max <= 0.0] = 0.0
                interp_logbeam_max = interp_logbeam_max.reshape(1,-1)
                interp_logbeam = interp_logbeam - interp_logbeam_max
                beam = 10**interp_logbeam
            else:
                altaz = NP.array([90.0, 0.0]).reshape(1,-1) + NP.array([-1,1]).reshape(1,-1) * NP.degrees(theta_phi)
                if beamparms['chromatic']:
                    beam = PB.primary_beam_generator(altaz, beamparms['freqs'], beamparms['telescope'], skyunits='altaz', pointing_info=None, pointing_center=None, freq_scle='Hz', east2ax1=0.0)
                else:
                    beam = PB.primary_beam_generator(altaz, beamparms['select_freq'], beamparms['telescope'], skyunits='altaz', pointing_info=None, pointing_center=None, freq_scle='Hz', east2ax1=0.0)
                    beam = beam.reshape(-1,1) * NP.ones(beamparms['freqs'].size).reshape(1,-1)
            omega_bw = DS.beam3Dvol(beam, beamparms['freqs'], freq_wts=freq_wts, hemisphere=True)
            return omega_bw

    ############################################################################


        
