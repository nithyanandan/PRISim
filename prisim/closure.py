from __future__ import division
import numpy as NP
import numpy.ma as MA
import progressbar as PGB
import h5py
import warnings
import copy
from skimage import img_as_float
import skimage.morphology as morphology
from skimage.filters import median
from skimage.filters.rank import mean
import astropy.cosmology as CP
import astropy.convolution as CONV
from astropy.time import Time
import scipy.constants as FCNST
from scipy import interpolate
from astroutils import DSP_modules as DSP
from astroutils import constants as CNST
from astroutils import nonmathops as NMO
from astroutils import mathops as OPS
from astroutils import lookup_operations as LKP
import prisim
from prisim import delay_spectrum as DS

prisim_path = prisim.__path__[0]+'/'

cosmoPlanck15 = CP.Planck15 # Planck 2015 cosmology
cosmo100 = cosmoPlanck15.clone(name='Modified Planck 2015 cosmology with h=1.0', H0=100.0) # Modified Planck 2015 cosmology with h=1.0, H= 100 km/s/Mpc

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
    day_avg_cpdata = npzdata['averaged_closures']
    std_triads_cpdata = npzdata['std_dev_triad']
    std_lst_cpdata = npzdata['std_dev_lst']

    cp = cpdata.astype(NP.float64)
    flags = flagsdata.astype(NP.bool)
    cp_dayavg = day_avg_cpdata.astype(NP.float64)
    cp_std_triads = std_triads_cpdata.astype(NP.float64)
    cp_std_lst = std_lst_cpdata.astype(NP.float64)

    with h5py.File(hdf5file, 'w') as fobj:
        datapool = ['raw']
        for dpool in datapool:
            if dpool == 'raw':
                qtys = ['cphase', 'triads', 'flags', 'lst', 'lst-day', 'days', 'dayavg', 'std_triads', 'std_lst']
            for qty in qtys:
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
                    data = NP.copy(cp_dayavg)
                elif qty == 'std_triads':
                    data = NP.copy(cp_std_triads)
                elif qty == 'std_lst':
                    data = NP.copy(cp_std_lst)
                dset = fobj.create_dataset('{0}/{1}'.format(dpool, qty), data=data, compression='gzip', compression_opts=9)
            
################################################################################

def interpolate_eicp_spectrum(eicp, wts, spec_ax, interp_parms, 
                              collapse_axes=None, collapse_stat='median'):

    """
    ----------------------------------------------------------------------------
    Interpolate complex exponential of closure phase and fill values where 
    values are not available. 

    Inputs:

    eicp    [Masked array] Complex masked array containing complex exponential 
            of the closure phases. Usually, it has shape (nlst, ndays, ntriads, 
            nchan)

    wts     [Maksed array] Maksed array containing weights corresponding to
            number of measurements. It has same shape as input eicp, which is 
            usually (nlst, ndays, ntriads, nchan)

    spec_ax [integer] Axis containing frequency spectra corresponding to nchan. 
            Must be an integer.

    interp_parms
            [dictionary] Dictionary specifying interpolation parameters. It has
            the following keys and values:
            'op_type'       [string] Specifies the interpolating operation.
                            Must be specified (no default). Accepted values are
                            'interp1d' (scipy.interpolate), 'median' 
                            (skimage.filters), 'tophat' (astropy.convolution) 
                            and 'gaussian' (astropy.convolution)
            'interp_kind'   [string (optional)] Specifies the interpolation 
                            kind (if 'op_type' is set to 'interp1d'). For
                            accepted values, see scipy.interpolate.interp1d()
            'window_size'   [integer (optional)] Specifies the size of the
                            interpolating/smoothing kernel. Only applies when
                            'op_type' is set to 'median', 'tophat' or 'gaussian'
                            The kernel is a tophat function when 'op_type' is 
                            set to 'median' or 'tophat'. If refers to FWHM when
                            'op_type' is set to 'gaussian'
    collapse_axes
            [Nonetype, int, tuple, list, or numpy array] Axes to collapse the
            data along before interpolation. If set to None (default), no
            collapse is performed. Otherwise, the axes specified here will be
            collapsed in the data using statistic specified in input
            collapse_stat. Usually, these axes are those along which closure 
            phases can be assumed to be coherent.

    collapse_stat
            [string (optional)] Statistic used to collapse the input data along
            the axes specified in input collapse_axes. Only applies if input
            collapse_axes is not set to None. Accepted values are 'mean' and
            'median' (default)

    Outputs:

    Tuple consisting of two elements. First element is a masked array of 
    interpolated complex exponential of closure phases of shape (nlst, ndays, 
    ntriads, nchan) except along the axes which were collapsed. Second element
    is a masked array of interpolated weights of same shape as the interpolated
    complex exponentials of closure phases.
    ----------------------------------------------------------------------------
    """

    if not isinstance(eicp, MA.MaskedArray):
        raise TypeError('Input eicp must be a numpy masked array')
    if not isinstance(wts, MA.MaskedArray):
        raise TypeError('Input wts must be a numpy masked array')
    if eicp.shape != wts.shape:
        raise ValueError('Inputs eicp and wts must have the same shape')
    if not isinstance(spec_ax, int):
        raise TypeError('Input spec_ax must be an integer')
    if spec_ax >= eicp.ndim:
        raise ValueError('Input spec_ax out of bounds')
    nchan = eicp.shape[spec_ax]
    if collapse_axes is not None:
        if not isinstance(collapse_axes, (int,tuple,list,NP.ndarray)):
            raise TypeError('Input collapse_axes must be an integer, list, tuple, or numpy array')
        collapse_axes = NP.asarray(collapse_axes).ravel()
        if NP.sum(NP.in1d(spec_ax, collapse_axes)) > 0:
            raise ValueError('spec_ax must not be included in collapse_axes')
        if not isinstance(collapse_stat, str):
            raise TypeError('Input collapse_stat must be a string')
        if collapse_stat.lower() not in ['mean', 'median']:
            raise ValueError('Invalid input for collapse_stat') 
        if collapse_stat.lower() == 'mean':
            eicp_collapsed = MA.sum(eicp*wts, axis=tuple(collapse_axes), keepdims=True) / MA.sum(wts, axis=tuple(collapse_axes), keepdims=True)
        elif collapse_stat.lower() == 'median':
            eicp_collapsed = MA.median(eicp.real, axis=tuple(collapse_axes), keepdims=True) +1j * MA.median(eicp.imag, axis=tuple(collapse_axes), keepdims=True)
        wts_collapsed = MA.sum(wts, axis=tuple(collapse_axes), keepdims=True)
        eicp_collapsed /= NP.abs(eicp_collapsed) # Renormalize to unit amplitude
    else:
        eicp_collapsed = MA.copy(eicp)
        wts_collapsed = MA.copy(wts)

    if not isinstance(interp_parms, dict):
        raise TypeError('Input interp_parms must be a dictionary')
    if 'op_type' not in interp_parms:
        raise KeyError('Key "op_type" not found in input interp_parms')
    if interp_parms['op_type'].lower() not in ['median', 'gaussian', 'tophat', 'interp1d']:
        raise ValueError('op_type specified in interp_parms currently not supported')
    if interp_parms['op_type'].lower() in ['median', 'gaussian', 'tophat']:
        if 'window_size' not in interp_parms:
            raise KeyError('Input "window_size" not found in interp_parms')
        if interp_parms['window_size'] <= 0:
            raise ValueError('Spectral filter window size must be positive')
    if interp_parms['op_type'].lower() == 'interp1d':
        if 'interp_kind' not in interp_parms:
            interp_parms['interp_kind'] = 'linear'
    mask_in = eicp.mask
    eicp_filled = MA.filled(eicp_collapsed.real, fill_value=NP.nan) + 1j * MA.filled(eicp_collapsed.imag, fill_value=NP.nan) # Both real and imaginary parts need to contain NaN for interpolation to work later separately on these parts
    wts_filled = MA.filled(wts_collapsed, fill_value=0.0)
    if interp_parms['op_type'].lower() == 'interp1d':
        non_spec_ax = NP.where(NP.arange(eicp_collapsed.ndim) != spec_ax)[0]
        collapsed_freq_mask = NP.sum(mask_in, axis=tuple(non_spec_ax)) # shape=(nchan,)
        if NP.sum(collapsed_freq_mask.astype(NP.bool)) > 1.0/3 * collapsed_freq_mask.size:
            raise ValueError('More than 1/3 of channels are flagged at some point or another. This will lead to failure of interp1d method. Try other interpolation options.')
        masked_chans = NP.arange(eicp_collapsed.shape[spec_ax])[collapsed_freq_mask.astype(NP.bool)]
        unmasked_chans = NP.arange(eicp_collapsed.shape[spec_ax])[NP.logical_not(collapsed_freq_mask.astype(NP.bool))]
        unmasked_eicp = NP.take(eicp_filled, unmasked_chans, axis=spec_ax, mode='clip')
        unmasked_wts = NP.take(wts_filled, unmasked_chans, axis=spec_ax, mode='clip')
        eicp_interpfunc_real = interpolate.interp1d(unmasked_chans, unmasked_eicp.real, kind=interp_parms['interp_kind'], axis=spec_ax, bounds_error=False, fill_value=NP.nan)
        eicp_interpfunc_imag = interpolate.interp1d(unmasked_chans, unmasked_eicp.imag, kind=interp_parms['interp_kind'], axis=spec_ax, bounds_error=False, fill_value=NP.nan)
        wts_interpfunc = interpolate.interp1d(unmasked_chans, unmasked_wts, kind=interp_parms['interp_kind'], axis=spec_ax, bounds_error=False, fill_value=0.0)
        wts_interped = wts_interpfunc(NP.arange(eicp_collapsed.shape[spec_ax]))
        eicp_interped = eicp_interpfunc_real(NP.arange(eicp_collapsed.shape[spec_ax])) + 1j * eicp_interpfunc_imag(NP.arange(eicp_collapsed.shape[spec_ax]))
    else:
        wts_reshaped = NP.moveaxis(wts_filled, spec_ax, wts_collapsed.ndim-1) # spec_ax is the last axis
        wts_reshaped_shape = wts_reshaped.shape
        mask_reshaped = NP.moveaxis(wts_collapsed.mask, spec_ax, wts_collapsed.ndim-1) # spec_ax is the last axis
        eicp_reshaped = NP.moveaxis(eicp_filled, spec_ax, eicp_collapsed.ndim-1) # spec_ax is the last axis
        
        if interp_parms['op_type'].lower() == 'median': # Always typecasts to int which is a problem!!! Needs to be fixed.
            kernel = morphology.rectangle(1, interp_parms['window_size'], dtype=NP.float64)
            maxval = NP.nanmax(NP.abs(wts_reshaped)) 
            wts_interped = maxval * mean(img_as_float(wts_reshaped.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) # shape=(-1,nchan), use mean not median for weights, array must be normalized to lie inside [-1,1]
            maxval = NP.nanmax(NP.abs(eicp_reshaped))
            eicp_interped = maxval * (median(img_as_float(eicp_reshaped.real.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1])) + 1j * median(img_as_float(eicp_reshaped.imag.reshape(-1,wts_reshaped_shape[-1])/maxval), kernel, mask=mask_reshaped.reshape(-1,wts_reshaped_shape[-1]))) # array must be normalized to lie inside [-1,1]
        else:
            wts_filled = MA.filled(wts_collapsed, fill_value=NP.nan)
            wts_reshaped = NP.moveaxis(wts_filled, spec_ax, wts_collapsed.ndim-1) # spec_ax is the last axis
            if interp_parms['op_type'].lower() == 'gaussian':
                fwhm = interp_parms['window_size']
                x_sigma = fwhm / (2.0 * NP.sqrt(2.0 * NP.log(2.0)))
                kernel1D = CONV.Gaussian1DKernel(x_sigma)
            elif interp_parms['op_type'].lower() == 'tophat':
                if interp_parms['window_size'] % 2 == 0:
                    interp_parms['window_size'] += 1
                kernel1D = CONV.Box1DKernel(interp_parms['window_size'])
            kernel = CONV.CustomKernel(kernel1D.array[NP.newaxis,:]) # Make a 2D kernel from the 1D kernel where it spans only one element in the new axis
            wts_interped = CONV.interpolate_replace_nans(wts_reshaped.reshape(-1,wts_reshaped_shape[-1]), kernel)
            eicp_interped = CONV.interpolate_replace_nans(eicp_reshaped.real.reshape(-1,wts_reshaped_shape[-1]), kernel) + 1j * CONV.interpolate_replace_nans(eicp_reshaped.imag.reshape(-1,wts_reshaped_shape[-1]), kernel)
        
        wts_interped = wts_interped.reshape(wts_reshaped_shape) # back to intermediate shape with spec_ax as the last axis
        wts_interped = NP.moveaxis(wts_interped, wts_collapsed.ndim-1, spec_ax) # Original shape
        eicp_interped = eicp_interped.reshape(wts_reshaped_shape) # back to intermediate shape with spec_ax as the last axis
        eicp_interped = NP.moveaxis(eicp_interped, eicp_collapsed.ndim-1, spec_ax) # Original shape
        eicp_interped /= NP.abs(eicp_interped)
    eps = 1e-10
    mask_out = NP.logical_or(wts_interped < eps, NP.isnan(wts_interped))
    wts_interped = MA.array(wts_interped, mask=mask_out)
    eicp_interped = MA.array(eicp_interped, mask=mask_out)

    return (eicp_interped, wts_interped)

################################################################################
        
class ClosurePhase(object):

    """
    ----------------------------------------------------------------------------
    Class to hold and operate on Closure Phase information. 

    It has the following attributes and member functions.

    Attributes:

    extfile         [string] Full path to external file containing information
                    of ClosurePhase instance. The file is in HDF5 format

    cpinfo          [dictionary] Contains two top level keys, namely, 'raw' and
                    'processed'. 

                    Under key 'raw' which holds a dictionary, the subkeys 
                    include 'cphase' (nlst,ndays,ntriads,nchan), 
                    'triads' (ntriads,3), 'lst' (nlst,ndays), and 'flags' 
                    (nlst,ndays,ntriads,nchan). 

                    Under the 'processed' key are two subkeys, namely, 'native' 
                    and 'prelim' each holding a dictionary. 
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

    Member functions:

    __init__()      Initialize an instance of class ClosurePhase

    expicp()        Compute and return complex exponential of the closure phase 
                    as a masked array

    smooth_in_tbins()
                    Smooth the complex exponentials of closure phases in LST  
                    bins. Both mean and median smoothing is produced.

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
            npz2hdf5(infile, infile_noext+'.hdf5')
            self.extfile = infile_noext + '.hdf5'
            self.cpinfo = NMO.load_dict_from_hdf5(self.extfile)
        else:
            if not isinstance(infile, h5py.File):
                raise TypeError('Input infile is not a valid HDF5 file')
            self.extfile = infile

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

    def smooth_in_tbins(self, daybinsize=None, lstbinsize=None):

        """
        ------------------------------------------------------------------------
        Smooth the complex exponentials of closure phases in time bins. Both
        mean and median smoothing is produced.

        Inputs:

        daybinsize  [Nonetype or scalar] Day bin size (in days) over which mean
                    and median are estimated across different days for a fixed
                    LST bin. If set to None, no smoothing is performed

        lstbinsize  [NoneType or scalar] LST bin size (in seconds) over which
                    mean and median are estimated across the LST. If set to 
                    None, no smoothing is performed
        ------------------------------------------------------------------------
        """

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

                if 'prelim' not in self.cpinfo['processed']:
                    self.cpinfo['processed']['prelim'] = {}
                self.cpinfo['processed']['prelim']['eicp'] = {}
                self.cpinfo['processed']['prelim']['cphase'] = {}
                self.cpinfo['processed']['prelim']['daybins'] = daybincenters
                self.cpinfo['processed']['prelim']['diff_dbins'] = daybinintervals

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
            lstbinsize = lstbinsize / 3.6e3 # in hours
            tres = NP.diff(self.cpinfo['raw']['lst'][:,0]).min() # in hours
            textent = self.cpinfo['raw']['lst'][:,0].max() - self.cpinfo['raw']['lst'][:,0].min() + tres # in hours
            if lstbinsize > tres:
                lstbinsize = NP.clip(lstbinsize, tres, textent)
                eps = 1e-10
                lstbins = NP.arange(self.cpinfo['raw']['lst'][:,0].min(), self.cpinfo['raw']['lst'][:,0].max() + tres + eps, lstbinsize)
                nlstbins = lstbins.size
                lstbins = NP.concatenate((lstbins, [lstbins[-1]+lstbinsize+eps]))
                if nlstbins > 1:
                    lstbinintervals = lstbins[1:] - lstbins[:-1]
                    lstbincenters = lstbins[:-1] + 0.5 * lstbinintervals
                else:
                    lstbinintervals = NP.asarray(lstbinsize).reshape(-1)
                    lstbincenters = lstbins[0] + 0.5 * lstbinintervals
                counts, lstbin_edges, lstbinnum, ri = OPS.binned_statistic(self.cpinfo['raw']['lst'][:,0], statistic='count', bins=lstbins)
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
                        eicp_tmean[binnum,:,:,:] = MA.mean(NP.exp(1j*indict['cphase']['mean'][ind_lstbin,:,:,:]), axis=0)
                        eicp_tmedian[binnum,:,:,:] = MA.median(NP.cos(indict['cphase']['median'][ind_lstbin,:,:,:]), axis=0) + 1j * MA.median(NP.sin(indict['cphase']['median'][ind_lstbin,:,:,:]), axis=0)
                        cp_trms[binnum,:,:,:] = MA.std(indict['cphase']['mean'][ind_lstbin,:,:,:], axis=0).data
                        cp_tmad[binnum,:,:,:] = MA.median(NP.abs(indict['cphase']['median'][ind_lstbin,:,:,:] - NP.angle(eicp_tmedian[binnum,:,:,:][NP.newaxis,:,:,:])), axis=0).data
                        
                mask = wts_lstbins <= 0.0
                self.cpinfo['processed']['prelim']['wts'] = MA.array(wts_lstbins, mask=mask)
                self.cpinfo['processed']['prelim']['eicp']['mean'] = MA.array(eicp_tmean, mask=mask)
                self.cpinfo['processed']['prelim']['eicp']['median'] = MA.array(eicp_tmedian, mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['mean'] = MA.array(NP.angle(eicp_tmean), mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['median'] = MA.array(NP.angle(eicp_tmedian), mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['rms'] = MA.array(cp_trms, mask=mask)
                self.cpinfo['processed']['prelim']['cphase']['mad'] = MA.array(cp_tmad, mask=mask)

    ############################################################################

    def save(outfile=None):

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
                    of Mpc/h
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
           datapool='prelim', method='fft', resample=True, apply_flags=True):

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
        'processed'     [dictionary] Contains the following keys and values:
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
                    print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        if not isinstance(datapool, str):
            raise TypeError('Input datapool must be a string')

        if datapool.lower() not in ['prelim']:
            raise ValueError('Specified datapool not supported')

        if not isinstance(method, str):
            raise TypeError('Input method must be a string')

        if method.lower() not in ['fft', 'nufft']:
            raise ValueError('Specified FFT method not supported')

        if not isinstance(apply_flags, bool):
            raise TypeError('Input apply_flags must be boolean')

        flagwts = 1.0

        if datapool.lower() == 'prelim':
            if method.lower() == 'fft':
                freq_wts = NP.empty((bw_eff.size, self.f.size), dtype=NP.float_) # nspw x nchan
                frac_width = DSP.window_N2width(n_window=None, shape=shape, fftpow=fftpow, area_normalize=False, power_normalize=True)
                window_loss_factor = 1 / frac_width
                n_window = NP.round(window_loss_factor * bw_eff / self.df).astype(NP.int)
                ind_freq_center, ind_channels, dfrequency = LKP.find_1NN(self.f.reshape(-1,1), freq_center.reshape(-1,1), distance_ULIM=0.5*self.df, remove_oob=True)
                sortind = NP.argsort(ind_channels)
                ind_freq_center = ind_freq_center[sortind]
                ind_channels = ind_channels[sortind]
                dfrequency = dfrequency[sortind]
                n_window = n_window[sortind]
        
                for i,ind_chan in enumerate(ind_channels):
                    window = NP.sqrt(frac_width * n_window[i]) * DSP.window_fftpow(n_window[i], shape=shape, fftpow=fftpow, centering=True, peak=None, area_normalize=False, power_normalize=True)
                    window_chans = self.f[ind_chan] + self.df * (NP.arange(n_window[i]) - int(n_window[i]/2))
                    ind_window_chans, ind_chans, dfreq = LKP.find_1NN(self.f.reshape(-1,1), window_chans.reshape(-1,1), distance_ULIM=0.5*self.df, remove_oob=True)
                    sind = NP.argsort(ind_window_chans)
                    ind_window_chans = ind_window_chans[sind]
                    ind_chans = ind_chans[sind]
                    dfreq = dfreq[sind]
                    window = window[ind_window_chans]
                    window = NP.pad(window, ((ind_chans.min(), self.f.size-1-ind_chans.max())), mode='constant', constant_values=((0.0,0.0)))
                    freq_wts[i,:] = window
        
                npad = int(self.f.size * pad)
                lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=True)
                result = {'freq_center': freq_center, 'shape': shape, 'freq_wts': freq_wts, 'bw_eff': bw_eff, 'npad': npad, 'lags': lags, 'lag_corr_length': self.f.size / NP.sum(freq_wts, axis=-1), 'processed': {'dspec': {'twts': self.cPhase.cpinfo['processed'][datapool]['wts']}}}
    
                for key in self.cPhase.cpinfo['processed'][datapool]['eicp']:
                    eicp = NP.copy(self.cPhase.cpinfo['processed'][datapool]['eicp'][key].data)
                    eicp = eicp[NP.newaxis,...] # nlst x ndays x ntriads x nchan --> (nspw=1) x nlst x ndays x ntriads x nchan

                    if apply_flags:
                        flagwts = NP.copy(self.cPhase.cpinfo['processed'][datapool]['wts'].data)
                        flagwts = flagwts[NP.newaxis,...] # nlst x ndays x ntriads x nchan --> (nspw=1) x nlst x ndays x ntriads x nchan
                        flagwts = 1.0 * flagwts / NP.mean(flagwts, axis=-1, keepdims=True) # (nspw=1) x nlst x ndays x ntriads x nchan

                    ndim_padtuple = [(0,0)]*(eicp.ndim-1) + [(0,npad)] # [(0,0), (0,0), (0,0), (0,0), (0,npad)]
                    result['processed']['dspec'][key] = DSP.FT1D(NP.pad(eicp*flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:], ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                # result['lag_kernel'] = DSP.FT1D(NP.pad(freq_wts, [(0,0), (0,npad)], mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                result['lag_kernel'] = DSP.FT1D(NP.pad(flagwts*freq_wts[:,NP.newaxis,NP.newaxis,NP.newaxis,:], ndim_padtuple, mode='constant'), ax=-1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df

            self.cPhaseDS = result
            if resample:
                result_resampled = copy.deepcopy(result)
                downsample_factor = NP.min((self.f.size + npad) * self.df / bw_eff)
                result_resampled['lags'] = DSP.downsampler(result_resampled['lags'], downsample_factor, axis=-1, method='interp', kind='linear')
                result_resampled['lag_kernel'] = DSP.downsampler(result_resampled['lag_kernel'], downsample_factor, axis=-1, method='interp', kind='linear')
                for key in self.cPhase.cpinfo['processed'][datapool]['eicp']:
                    result_resampled['processed']['dspec'][key] = DSP.downsampler(result_resampled['processed']['dspec'][key], downsample_factor, axis=-1, method='FFT')
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

        Tuple (triad_ind, lst_ind, day_ind) containing the triad, LST, and day 
        indices, each as a numpy array
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
        if 'days' not in selection:
            if 'prelim' in self.cPhase.cpinfo['processed']:
                day_ind = NP.arange(self.cPhase.cpinfo['processed']['prelim']['wts'].shape[1])
        else:
            if selection['days'] is None:
                if 'prelim' in self.cPhase.cpinfo['processed']:
                    day_ind = NP.arange(self.cPhase.cpinfo['processed']['prelim']['wts'].shape[1])
            elif isinstance(selection['days'], (list,NP.ndarray)):
                if 'prelim' in self.cPhase.cpinfo['processed']:
                    day_ind = selection['days']
                    if NP.any(NP.logical_or(day_ind < 0, day_ind >= self.cPhase.cpinfo['processed']['prelim']['wts'].shape[1])):
                        raise ValueError('Input processed day indices out of bounds')
            else:
                raise TypeError('Wrong type for processed day indices')

        if day_ind is None:
            raise ValueError('Day index selection could not be performed')
                
        return (triad_ind, lst_ind, day_ind)

    ############################################################################

    def compute_power_spectrum(self, cpds=None, selection=None, cohax=None,
                               incohax=None, cosmo=cosmo100):

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

        cohax   [NoneType or tuple] Specifies a tuple of axes over which the 
                delay spectra will be coherently averaged before computing power
                spectra. If set to None (default), it is set to (2,3) 
                (corresponding to Days, and triads respectively). 

        incohax [NoneType or tuple] Specifies a tuple of axes over which the 
                delay power spectra will be incoherently averaged. If set to 
                None (default), it is set to (1,) (corresponding to LST).

        cosmo   [instance of cosmology class from astropy] An instance of class
                FLRW or default_cosmology of astropy cosmology module. Default
                uses Planck 2015 cosmology, with H0=100 h km/s/Mpc

        Output:

        Dictionary with the keys 'triads', 'triads_ind', 'lstbins', 'lst', 'dlst',
        'lst_ind', 'oversampled' and 'resampled' corresponding to whether 
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
        'mean'  [numpy array] Delay power spectrum incoherently averaged over 
                the axes specified in incohax using the 'mean' key in input 
                cpds or attribute cPhaseDS['processed']['dspec']. It has
                shape=(nspw,npol,1,1,nlags) if incohax=(2,3). It has units of
                Mpc/h.
        'median'
                [numpy array] Delay power spectrum incoherently averaged over 
                the axes specified in incohax using the 'median' key in input 
                cpds or attribute cPhaseDS['processed']['dspec']. It has
                shape=(nspw,npol,1,1,nlags) if incohax=(2,3). It has units of
                Mpc/h.
        ------------------------------------------------------------------------
        """

        if cohax is None:
            cohax = (2, 3) # (ndays,ntriads)
        else:
            cohax = tuple(cohax)

        if incohax is None:
            incohax = (1,) # (nlst,)
        else:
            incohax = tuple(incohax)

        if NP.intersect1d(cohax, incohax).size > 0:
            raise ValueError('Inputs cohax and incohax must have no intersection')

        if selection is None:
            selection = {'triads': None, 'lst': None, 'days': None}
        else:
            if not isinstance(selection, dict):
                raise TypeError('Input selection must be a dictionary')

        if cpds is None:
            cpds = {}
            datapool = ['oversampled', 'resampled']
            for dpool in datapool:
                if dpool == 'oversampled':
                    cpds[dpool] = copy.deepcopy(self.cPhaseDS)
                else:
                    cpds[dpool] = copy.deepcopy(self.cPhaseDS_resampled)

        triad_ind, lst_ind, day_ind = self.subset(selection=selection)

        result = {'triads': self.cPhase.cpinfo['raw']['triads'][triad_ind], 'triads_ind': triad_ind, 'lst': self.cPhase.cpinfo['processed']['prelim']['lstbins'][lst_ind], 'lst_ind': lst_ind, 'dlst': self.cPhase.cpinfo['processed']['prelim']['dlstbins'][lst_ind], 'days': self.cPhase.cpinfo['processed']['prelim']['daybins'][day_ind], 'day_ind': day_ind, 'dday': self.cPhase.cpinfo['processed']['prelim']['diff_dbins'][day_ind]}
        for dpool in datapool:
            result[dpool] = {}
                
            wl = FCNST.c / cpds[dpool]['freq_center']
            z = CNST.rest_freq_HI / cpds[dpool]['freq_center'] - 1
            dz = CNST.rest_freq_HI / cpds[dpool]['freq_center']**2 * cpds[dpool]['bw_eff']
            dkprll_deta = DS.dkprll_deta(z, cosmo=cosmo)
            kprll = dkprll_deta.reshape(-1,1) * cpds[dpool]['lags']

            drz_los = (FCNST.c/1e3) * cpds[dpool]['bw_eff'] * (1+z)**2 / CNST.rest_freq_HI / cosmo.H0.value / cosmo.efunc(z)   # in Mpc/h
            jacobian1 = 1 / cpds[dpool]['bw_eff']
            jacobian2 = drz_los / cpds[dpool]['bw_eff']
            factor = jacobian1 * jacobian2

            result[dpool]['z'] = z
            result[dpool]['kprll'] = kprll
            result[dpool]['lags'] = NP.copy(cpds[dpool]['lags'])

            for stat in ['mean', 'median']:
                inpshape = list(cpds[dpool]['processed']['dspec'][stat].shape)
                inpshape[1] = lst_ind.size
                inpshape[2] = day_ind.size
                inpshape[3] = triad_ind.size
                nsamples_coh = NP.prod(NP.asarray(inpshape)[NP.asarray(cohax)])
                nsamples = NP.prod(NP.asarray(inpshape)[NP.asarray(incohax)])
                nsamples_incoh = nsamples * (nsamples - 1)
                twts_multidim_idx = NP.ix_(lst_ind,day_ind,triad_ind,NP.arange(1)) # shape=(nlst,ndays,ntriads,nchan)
                dspec_multidim_idx = NP.ix_(NP.arange(wl.size),lst_ind,day_ind,triad_ind,NP.arange(inpshape[4])) # shape=(nspw,nlst,ndays,ntriads,nchan)
                max_wt_in_chan = NP.max(NP.sum(cpds[dpool]['processed']['dspec']['twts'].data, axis=(0,1,2)))
                select_chan = NP.argmax(NP.sum(cpds[dpool]['processed']['dspec']['twts'].data, axis=(0,1,2)))
                twts = NP.copy(cpds[dpool]['processed']['dspec']['twts'].data[:,:,:,[select_chan]]) # shape=(nspw=1,nlst,ndays,ntriads,nlags=1)
                dspec = NP.copy(cpds[dpool]['processed']['dspec'][stat][dspec_multidim_idx])
                if nsamples_coh > 1:
                    if stat == 'mean':
                        dspec = NP.sum(twts[twts_multidim_idx][NP.newaxis,...] * cpds[dpool]['processed']['dspec'][stat][dspec_multidim_idx], axis=cohax, keepdims=True) / NP.sum(twts[twts_multidim_idx][NP.newaxis,...], axis=cohax, keepdims=True)
                    else:
                        dspec = NP.median(cpds[dpool]['processed']['dspec'][stat][dspec_multidim_idx], axis=cohax, keepdims=True)
                if nsamples_incoh > 1:
                    result[dpool][stat] = factor.reshape(-1,1,1,1,1) / nsamples_incoh * (NP.abs(NP.sum(dspec, axis=incohax, keepdims=True))**2 - NP.sum(NP.abs(dspec)**2, axis=incohax, keepdims=True))
                else:
                    result[dpool][stat] = factor.reshape(-1,1,1,1,1) * NP.abs(dspec)**2
            result[dpool]['nsamples_incoh'] = nsamples_incoh
            result[dpool]['nsamples_coh'] = nsamples_coh
            
        return result

    ############################################################################

            
