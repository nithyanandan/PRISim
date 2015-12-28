from __future__ import division
import numpy as NP
import multiprocessing as MP
import itertools as IT
import statsmodels.robust.scale as stats
import progressbar as PGB
import writer_module as WM
import aipy as AP
import astropy 
from astropy.io import fits
import astropy.cosmology as CP
import scipy.constants as FCNST
from distutils.version import LooseVersion
import constants as CNST
import my_DSP_modules as DSP 
import baseline_delay_horizon as DLY
import geometry as GEOM
import interferometry as RI
import lookup_operations as LKP
import ipdb as PDB

cosmo100 = CP.FlatLambdaCDM(H0=100.0, Om0=0.27)  # Using H0 = 100 km/s/Mpc

#################################################################################

def _astropy_columns(cols, tabtype='BinTableHDU'):
    
    """
    ----------------------------------------------------------------------------
    !!! FOR INTERNAL USE ONLY !!!
    This internal routine checks for Astropy version and produces the FITS 
    columns based on the version

    Inputs:

    cols    [list of Astropy FITS columns] These are a list of Astropy FITS 
            columns

    tabtype [string] specifies table type - 'BinTableHDU' (default) for binary
            tables and 'TableHDU' for ASCII tables

    Outputs:

    columns [Astropy FITS column data] 
    ----------------------------------------------------------------------------
    """

    try:
        cols
    except NameError:
        raise NameError('Input cols not specified')

    if tabtype not in ['BinTableHDU', 'TableHDU']:
        raise ValueError('tabtype specified is invalid.')

    use_ascii = False
    if tabtype == 'TableHDU':
        use_ascii = True
    if astropy.__version__ == '0.4':
        columns = fits.ColDefs(cols, tbtype=tabtype)
    elif LooseVersion(astropy.__version__)>=LooseVersion('0.4.2'):
        columns = fits.ColDefs(cols, ascii=use_ascii)
    return columns    

################################################################################

def _gentle_clean(dd, _w, tol=1e-1, area=None, stop_if_div=True, maxiter=100,
                  verbose=False, autoscale=True):

    if verbose:
        print "Performing gentle clean..."

    scale_factor = 1.0
    if autoscale:
        scale_factor = NP.nanmax(NP.abs(_w))
    dd /= scale_factor
    _w /= scale_factor

    cc, info = AP.deconv.clean(dd, _w, tol=tol, area=area, stop_if_div=False,
                               maxiter=maxiter, verbose=verbose)
    #dd = info['res']

    cc = NP.zeros_like(dd)
    inside_res = NP.std(dd[area!=0])
    outside_res = NP.std(dd[area==0])
    initial_res = inside_res
    #print inside_res,'->',
    ncycle=0
    if verbose:
        print "inside_res outside_res"
        print inside_res, outside_res
    inside_res = 2*outside_res #just artifically bump up the inside res so the loop runs at least once
    while(inside_res>outside_res and maxiter>0):
        if verbose: print '.',
        _d_cl, info = AP.deconv.clean(dd, _w, tol=tol, area=area, stop_if_div=stop_if_div, maxiter=maxiter, verbose=verbose, pos_def=True)
        res = info['res']
        inside_res = NP.std(res[area!=0])
        outside_res = NP.std(res[area==0])
        dd = info['res']
        cc += _d_cl
        ncycle += 1
        if verbose: print inside_res*scale_factor, outside_res*scale_factor
        if ncycle>1000: break

    info['ncycle'] = ncycle-1

    dd *= scale_factor
    _w *= scale_factor
    cc *= scale_factor
    info['initial_residual'] = initial_res * scale_factor
    info['final_residual'] = inside_res * scale_factor
    
    return cc, info

#################################################################################

def complex1dClean_arg_splitter(args, **kwargs):
    return complex1dClean(*args, **kwargs)

def complex1dClean(inp, kernel, cbox=None, gain=0.1, maxiter=10000,
                   threshold=5e-3, threshold_type='relative', verbose=False,
                   progressbar=False, pid=None, progressbar_yloc=0):

    """
    ----------------------------------------------------------------------------
    Hogbom CLEAN algorithm applicable to 1D complex array

    Inputs:

    inp      [numpy vector] input 1D array to be cleaned. Can be complex.

    kernel   [numpy vector] 1D array that acts as the deconvolving kernel. Can 
             be complex. Must be of same size as inp

    cbox     [boolean array] 1D boolean array that acts as a mask for pixels 
             which should be cleaned. Same size as inp. Only pixels with values 
             True are to be searched for maxima in residuals for cleaning and 
             the rest are not searched for. Default=None (means all pixels are 
             to be searched for maxima while cleaning)

    gain     [scalar] gain factor to be applied while subtracting clean 
             component from residuals. This is the fraction of the maximum in 
             the residuals that will be subtracted. Must lie between 0 and 1.
             A lower value will have a smoother convergence but take a longer 
             time to converge. Default=0.1

    maxiter  [scalar] maximum number of iterations for cleaning process. Will 
             terminate if the number of iterations exceed maxiter. Default=10000

    threshold 
             [scalar] represents the cleaning depth either as a fraction of the
             maximum in the input (when thershold_type is set to 'relative') or
             the absolute value (when threshold_type is set to 'absolute') in 
             same units of input down to which inp should be cleaned. Value must 
             always be positive. When threshold_type is set to 'relative', 
             threshold mu st lie between 0 and 1. Default=5e-3 (found to work 
             well and converge fast) assuming threshold_type is set to 'relative'

    threshold_type
             [string] represents the type of threshold specified by value in 
             input threshold. Accepted values are 'relative' and 'absolute'. If
             set to 'relative' the threshold value is the fraction (between 0
             and 1) of maximum in input down to which it should be cleaned. If 
             set to 'asbolute' it is the actual value down to which inp should 
             be cleaned. Default='relative'

    verbose  [boolean] If set to True (default), print diagnostic and progress 
             messages. If set to False, no such messages are printed.

    progressbar 
             [boolean] If set to False (default), no progress bar is displayed

    pid      [string or integer] process identifier (optional) relevant only in
             case of parallel processing and if progressbar is set to True. If
             pid is not specified, it defaults to the Pool process id

    progressbar_yloc
             [integer] row number where the progressbar is displayed on the
             terminal. Default=0

    Output:

    outdict  [dictionary] It consists of the following keys and values at
             termination:
             'termination' [dictionary] consists of information on the 
                           conditions for termination with the following keys 
                           and values:
                           'threshold' [boolean] If True, the cleaning process
                                       terminated because the threshold was 
                                       reached
                           'maxiter'   [boolean] If True, the cleaning process
                                       terminated because the number of 
                                       iterations reached maxiter
                           'inrms<outrms'
                                       [boolean] If True, the cleaning process
                                       terminated because the rms inside the 
                                       clean box is below the rms outside of it
             'iter'        [scalar] number of iterations performed before 
                           termination
             'rms'         [numpy vector] rms of the residuals as a function of
                           iteration
             'inrms'       [numpy vector] rms of the residuals inside the clean 
                           box as a function of iteration
             'outrms'      [numpy vector] rms of the residuals outside the clean 
                           box as a function of iteration
             'res'         [numpy array] uncleaned residuals at the end of the
                           cleaning process. Complex valued and same size as 
                           inp
             'cc'          [numpy array] clean components at the end of the
                           cleaning process. Complex valued and same size as 
                           inp
    ----------------------------------------------------------------------------
    """

    try:
        inp, kernel
    except NameError:
        raise NameError('Inputs inp and kernel not specified')

    if not isinstance(inp, NP.ndarray):
        raise TypeError('inp must be a numpy array')
    if not isinstance(kernel, NP.ndarray):
        raise TypeError('kernel must be a numpy array')

    if threshold_type not in ['relative', 'absolute']:
        raise ValueError('invalid specification for threshold_type')

    if not isinstance(threshold, (int,float)):
        raise TypeError('input threshold must be a scalar')
    else:
        threshold = float(threshold)
        if threshold <= 0.0:
            raise ValueError('input threshold must be positive')

    inp = inp.flatten()
    kernel = kernel.flatten()
    kernel /= NP.abs(kernel).max()
    kmaxind = NP.argmax(NP.abs(kernel))

    if inp.size != kernel.size:
        raise ValueError('inp and kernel must have same size')

    if cbox is None:
        cbox = NP.ones(inp.size, dtype=NP.bool)
    elif isinstance(cbox, NP.ndarray):
        cbox = cbox.flatten()
        if cbox.size != inp.size:
            raise ValueError('Clean box must be of same size as input')
        cbox = NP.where(cbox > 0.0, True, False)
        # cbox = cbox.astype(NP.int)
    else:
        raise TypeError('cbox must be a numpy array')
    cbox = cbox.astype(NP.bool)

    if threshold_type == 'relative':
        lolim = threshold
    else:
        lolim = threshold / NP.abs(inp).max()

    if lolim >= 1.0:
        raise ValueError('incompatible value specified for threshold')

    # inrms = [NP.std(inp[cbox])]
    inrms = [NP.median(NP.abs(inp[cbox] - NP.median(inp[cbox])))]
    if inp.size - NP.sum(cbox) <= 2:
        outrms = None
    else:
        # outrms = [NP.std(inp[NP.invert(cbox)])]
        outrms = [NP.median(NP.abs(inp[NP.invert(cbox)] - NP.median(inp[NP.invert(cbox)])))]

    if not isinstance(gain, float):
        raise TypeError('gain must be a floating point number')
    else:
        if (gain <= 0.0) or (gain >= 1.0):
            raise TypeError('gain must lie between 0 and 1')

    if not isinstance(maxiter, int):
        raise TypeError('maxiter must be an integer')
    else:
        if maxiter <= 0:
            raise ValueError('maxiter must be positive')

    cc = NP.zeros_like(inp)
    res = NP.copy(inp)
    cond4 = False
    # prevrms = NP.std(res)
    # currentrms = [NP.std(res)]
    prevrms = NP.median(NP.abs(res - NP.median(res)))
    currentrms = [NP.median(NP.abs(res - NP.median(res)))]
    itr = 0
    terminate = False

    if progressbar:
        if pid is None:
            pid = MP.current_process().name
        else:
            pid = '{0:0d}'.format(pid)
        progressbar_loc = (0, progressbar_yloc)
        writer=WM.Writer(progressbar_loc)
        progress = PGB.ProgressBar(widgets=[pid+' ', PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Iterations '.format(maxiter), PGB.ETA()], maxval=maxiter, fd=writer).start()
    while not terminate:
        itr += 1
        indmaxres = NP.argmax(NP.abs(res*cbox))
        maxres = res[indmaxres]
        
        ccval = gain * maxres
        cc[indmaxres] += ccval
        res = res - ccval * NP.roll(kernel, indmaxres-kmaxind)
        
        prevrms = NP.copy(currentrms[-1])
        # currentrms += [NP.std(res)]
        currentrms += [NP.median(NP.abs(res - NP.median(res)))]

        # inrms += [NP.std(res[cbox])]
        inrms += [NP.median(NP.abs(res[cbox] - NP.median(res[cbox])))]
            
        # cond1 = NP.abs(maxres) <= inrms[-1]
        cond1 = NP.abs(maxres) <= lolim * NP.abs(inp).max()
        cond2 = itr >= maxiter
        terminate = cond1 or cond2
        if outrms is not None:
            # outrms += [NP.std(res[NP.invert(cbox)])]
            outrms += [NP.median(NP.abs(res[NP.invert(cbox)] - NP.median(res[NP.invert(cbox)])))]
            cond3 = inrms[-1] <= outrms[-1]
            terminate = terminate or cond3

        if progressbar:
            progress.update(itr)
    if progressbar:
        progress.finish()

    inrms = NP.asarray(inrms)
    currentrms = NP.asarray(currentrms)
    if outrms is not None:
        outrms = NP.asarray(outrms)
        
    outdict = {'termination':{'threshold': cond1, 'maxiter': cond2, 'inrms<outrms': cond3}, 'iter': itr, 'rms': currentrms, 'inrms': inrms, 'outrms': outrms, 'cc': cc, 'res': res}

    return outdict

#################################################################################

class DelaySpectrum(object):

    """
    ----------------------------------------------------------------------------
    Class to manage delay spectrum information on a multi-element interferometer 
    array. 

    Attributes:

    ia          [instance of class InterferometerArray] An instance of class
                InterferometerArray that contains the results of the simulated
                interferometer visibilities

    bp          [numpy array] Bandpass weights of size n_baselines x nchan x
                n_acc, where n_acc is the number of accumulations in the
                observation, nchan is the number of frequency channels, and
                n_baselines is the number of baselines

    bp_wts      [numpy array] Additional weighting to be applied to the bandpass
                shapes during the application of the member function 
                delay_transform(). Same size as attribute bp. 

    f           [list or numpy vector] frequency channels in Hz

    cc_freq     [list or numpy vector] frequency channels in Hz associated with 
                clean components of delay spectrum. Same size as cc_lags. This 
                computed inside member function delayClean()

    df          [scalar] Frequency resolution (in Hz)

    lags        [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as channels. This is 
                computed in member function delay_transform().

    cc_lags     [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as cc_freq. This is computed in 
                member function delayClean().

    lag_kernel  [numpy array] Inverse Fourier Transform of the frequency 
                bandpass shape. In other words, it is the impulse response 
                corresponding to frequency bandpass. Same size as attributes
                bp and bp_wts. It is initialized in __init__() member function
                but effectively computed in member functions delay_transform()
                and delayClean()

    cc_lag_kernel  
                [numpy array] Inverse Fourier Transform of the frequency 
                bandpass shape. In other words, it is the impulse response 
                corresponding to frequency bandpass shape used in complex delay 
                clean routine. It is initialized in __init__() member function
                but effectively computed in member function delayClean()

    n_acc       [scalar] Number of accumulations

    horizon_delay_limits
                [numpy array] NxMx2 numpy array denoting the neagtive and 
                positive horizon delay limits where N is the number of 
                timestamps, M is the number of baselines. The 0 index in the 
                third dimenstion denotes the negative horizon delay limit while 
                the 1 index denotes the positive horizon delay limit

    skyvis_lag  [numpy array] Complex visibility due to sky emission (in Jy Hz or
                K Hz) along the delay axis for each interferometer obtained by
                FFT of skyvis_freq along frequency axis. Same size as vis_freq.
                Created in the member function delay_transform(). Read its
                docstring for more details. Same dimensions as skyvis_freq

    vis_lag     [numpy array] The simulated complex visibility (in Jy Hz or K Hz) 
                along delay axis for each interferometer obtained by FFT of
                vis_freq along frequency axis. Same size as vis_noise_lag and
                skyis_lag. It is evaluated in member function delay_transform(). 

    vis_noise_lag
                [numpy array] Complex visibility noise (in Jy Hz or K Hz) along 
                delay axis for each interferometer generated using an FFT of
                vis_noise_freq along frequency axis. Same size as vis_noise_freq.
                Created in the member function delay_transform(). Read its
                docstring for more details. 

    cc_skyvis_lag
                [numpy array] Complex cleaned visibility delay spectra (in 
                Jy Hz or K Hz) of noiseless simulated sky visibilities for each 
                baseline at each LST. Size is nbl x nlags x nlst

    cc_skyvis_res_lag
                [numpy array] Complex residuals from cleaned visibility delay 
                spectra (in Jy Hz or K Hz) of noiseless simulated sky 
                visibilities for each baseline at each LST. Size is 
                nbl x nlags x nlst

    cc_skyvis_net_lag
                [numpy array] Sum of complex cleaned visibility delay spectra
                and residuals (in Jy Hz or K Hz) of noiseless simulated sky 
                visibilities for each baseline at each LST. Size is 
                nbl x nlags x nlst. cc_skyvis_net_lag = cc_skyvis_lag + 
                cc_skyvis_res_lag

    cc_vis_lag
                [numpy array] Complex cleaned visibility delay spectra (in 
                Jy Hz or K Hz) of noisy simulated sky visibilities for each 
                baseline at each LST. Size is nbl x nlags x nlst

    cc_vis_res_lag
                [numpy array] Complex residuals from cleaned visibility delay 
                spectra (in Jy Hz or K Hz) of noisy simulated sky 
                visibilities for each baseline at each LST. Size is 
                nbl x nlags x nlst

    cc_vis_net_lag
                [numpy array] Sum of complex cleaned visibility delay spectra
                and residuals (in Jy Hz or K Hz) of noisy simulated sky 
                visibilities for each baseline at each LST. Size is 
                nbl x nlags x nlst. cc_vis_net_lag = cc_vis_lag + 
                cc_vis_res_lag

    cc_skyvis_freq
                [numpy array] Complex cleaned visibility delay spectra 
                transformed to frequency domain (in Jy or K.Sr) obtained from 
                noiseless simulated sky visibilities for each baseline at each 
                LST. Size is nbl x nlags x nlst

    cc_skyvis_res_freq
                [numpy array] Complex residuals from cleaned visibility delay 
                spectra transformed to frequency domain (in Jy or K.Sr) obtained 
                from noiseless simulated sky visibilities for each baseline at 
                each LST. Size is nbl x nlags x nlst

    cc_skyvis_net_freq
                [numpy array] Sum of complex cleaned visibility delay spectra
                and residuals transformed to frequency domain (in Jy or K.Sr) 
                obtained from noiseless simulated sky visibilities for each 
                baseline at each LST. Size is nbl x nlags x nlst. 
                cc_skyvis_net_freq = cc_skyvis_freq + cc_skyvis_res_freq

    cc_vis_freq
                [numpy array] Complex cleaned visibility delay spectra 
                transformed to frequency domain (in Jy or K.Sr) obtained from 
                noisy simulated sky visibilities for each baseline at each LST. 
                Size is nbl x nlags x nlst

    cc_vis_res_freq
                [numpy array] Complex residuals from cleaned visibility delay 
                spectra transformed to frequency domain (in Jy or K.Sr) of noisy 
                simulated sky visibilities for each baseline at each LST. Size 
                is nbl x nlags x nlst

    cc_vis_net_freq
                [numpy array] Sum of complex cleaned visibility delay spectra
                and residuals transformed to frequency domain (in Jy or K.Sr) 
                obtained from noisy simulated sky visibilities for each baseline 
                at each LST. Size is nbl x nlags x nlst. 
                cc_vis_net_freq = cc_vis_freq + cc_vis_res_freq

    clean_window_buffer
                [scalar] number of inverse bandwidths to extend beyond the 
                horizon delay limit to include in the CLEAN deconvolution. 

    pad         [scalar] Non-negative scalar indicating padding fraction 
                relative to the number of frequency channels. For e.g., a 
                pad of 1.0 pads the frequency axis with zeros of the same 
                width as the number of channels. After the delay transform,
                the transformed visibilities are downsampled by a factor of
                1+pad. If a negative value is specified, delay transform 
                will be performed with no padding

    subband_delay_spectra
                [dictionary] contains two top level keys, namely, 'cc' and 'sim' 
                denoting information about CLEAN and simulated visibilities 
                respectively. Under each of these keys is information about delay 
                spectra of different frequency sub-bands (n_win in number) in the 
                form of a dictionary under the following keys:
                'freq_center' 
                           [numpy array] contains the center frequencies 
                           (in Hz) of the frequency subbands of the subband
                           delay spectra. It is of size n_win. It is roughly 
                           equivalent to redshift(s)
                'freq_wts'  [numpy array] Contains frequency weights applied 
                            on each frequency sub-band during the subband delay 
                            transform. It is of size n_win x nchan. 
                'bw_eff'    [numpy array] contains the effective bandwidths 
                            (in Hz) of the subbands being delay transformed. It
                            is of size n_win. It is roughly equivalent to width 
                            in redshift or along line-of-sight
                'shape'     [string] shape of the window function applied. 
                            Accepted values are 'rect' (rectangular), 'bhw'
                            (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
                'bpcorrect' [boolean] If True (default), correct for frequency
                            weights that were applied during the original 
                            delay transform using which the delay CLEAN was 
                            done. This would flatten the bandpass after delay
                            CLEAN. If False, do not apply the correction, 
                            namely, inverse of bandpass weights. This applies 
                            only CLEAned visibilities under the 'cc' key and 
                            hence is present only if the top level key is 'cc' 
                            and absent for key 'sim'
                'npad'      [scalar] Numbber of zero-padded channels before
                            performing the subband delay transform. 
                'lags'      [numpy array] lags of the subband delay spectra 
                            after padding in frequency during the transform. It
                            is of size nchan+npad where npad is the number of 
                            frequency channels padded specified under the key 
                            'npad'
                'lag_kernel'
                            [numpy array] delay transform of the frequency 
                            weights under the key 'freq_wts'. It is of size
                            n_bl x n_win x (nchan+npad) x n_t.
                'lag_corr_length' 
                            [numpy array] It is the correlation timescale (in 
                            pixels) of the subband delay spectra. It is 
                            proportional to inverse of effective bandwidth. It
                            is of size n_win. The unit size of a pixel is 
                            determined by the difference between adjacent pixels 
                            in lags under key 'lags' which in turn is 
                            effectively inverse of the total bandwidth 
                            (nchan x df) simulated.
                'skyvis_lag'
                            [numpy array] subband delay spectra of simulated 
                            or CLEANed noiseless visibilities, depending on 
                            whether the top level key is 'cc' or 'sim' 
                            respectively, after applying the frequency weights 
                            under the key 'freq_wts'. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t. 
                'vis_lag'   [numpy array] subband delay spectra of simulated 
                            or CLEANed noisy visibilities, depending on whether
                            the top level key is 'cc' or 'sim' respectively,
                            after applying the frequency weights under the key 
                            'freq_wts'. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t. 
                'vis_noise_lag'   
                            [numpy array] subband delay spectra of simulated 
                            noise after applying the frequency weights under 
                            the key 'freq_wts'. Only present if top level key is 
                            'sim' and absent for 'cc'. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t. 
                'skyvis_res_lag'
                            [numpy array] subband delay spectra of residuals
                            after delay CLEAN of simualted noiseless 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x (nchan+npad) x n_t
                'vis_res_lag'
                            [numpy array] subband delay spectra of residuals
                            after delay CLEAN of simualted noisy 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x (nchan+npad) x n_t

    Member functions:

    __init__()  Initializes an instance of class DelaySpectrum
                        
    delay_transform()  
                Transforms the visibilities from frequency axis onto 
                delay (time) axis using an IFFT. This is performed for 
                noiseless sky visibilities, thermal noise in visibilities, 
                and observed visibilities. 

    clean()     Transforms the visibilities from frequency axis onto delay 
                (time) axis using an IFFT and deconvolves the delay transform 
                quantities along the delay axis. This is performed for noiseless 
                sky visibilities, thermal noise in visibilities, and observed 
                visibilities. 

    delayClean()
                Transforms the visibilities from frequency axis onto delay 
                (time) axis using an IFFT and deconvolves the delay transform 
                quantities along the delay axis. This is performed for noiseless 
                sky visibilities, thermal noise in visibilities, and observed 
                visibilities. This calls an in-house module complex1dClean 
                instead of the clean routine in AIPY module. It can utilize 
                parallelization

    subband_delay_transform()
                Computes delay transform on multiple frequency sub-bands with 
                specified weights

    get_horizon_delay_limits()
                Estimates the delay envelope determined by the sky horizon 
                for the baseline(s) for the phase centers 

    set_horizon_delay_limits()
                Estimates the delay envelope determined by the sky horizon for 
                the baseline(s) for the phase centers of the DelaySpectrum 
                instance. No output is returned. Uses the member function 
                get_horizon_delay_limits()

    save()      Saves the interferometer array delay spectrum information to 
                disk. 
    ----------------------------------------------------------------------------
    """

    def __init__(self, interferometer_array=None, init_file=None):

        """
        ------------------------------------------------------------------------
        Intialize the DelaySpectrum class which manages information on delay
        spectrum of a multi-element interferometer.

        Class attributes initialized are:
        f, bp, bp_wts, df, lags, skyvis_lag, vis_lag, n_acc, vis_noise_lag, ia, 
        pad, lag_kernel, horizon_delay_limits, cc_skyvis_lag, cc_skyvis_res_lag, 
        cc_skyvis_net_lag, cc_vis_lag, cc_vis_res_lag, cc_vis_net_lag, 
        cc_skyvis_freq, cc_skyvis_res_freq, cc_sktvis_net_freq, cc_vis_freq,
        cc_vis_res_freq, cc_vis_net_freq, clean_window_buffer, cc_freq, cc_lags,
        cc_lag_kernel, multiwin_delay_transform

        Read docstring of class DelaySpectrum for details on these
        attributes.

        Input(s):

        interferometer_array
                     [instance of class InterferometerArray] An instance of 
                     class InterferometerArray from which certain attributes 
                     will be obtained and used

        init_file    [string] full path to filename in FITS format containing 
                     delay spectrum information of interferometer array

        Other input parameters have their usual meanings. Read the docstring of
        class DelaySpectrum for details on these inputs.
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
                self.df = hdulist[0].header['freq_resolution']
            except KeyError:
                hdulist.close()
                raise KeyError('Keyword "freq_resolution" not found in header')

            try:
                self.n_acc = hdulist[0].header['N_ACC']
            except KeyError:
                hdulist.close()
                raise KeyError('Keyword "N_ACC" not found in header')
            
            try:
                self.pad = hdulist[0].header['PAD']
            except KeyError:
                hdulist.close()
                raise KeyError('Keyword "PAD" not found in header')

            try:
                self.clean_window_buffer = hdulist[0].header['DBUFFER']
            except KeyError:
                hdulist.close()
                raise KeyError('Keyword "DBUFFER" not found in header')

            try:
                iarray_init_file = hdulist[0].header['IARRAY']
            except KeyError:
                hdulist.close()
                raise KeyError('Keyword "IARRAY" not found in header')
            self.ia = RI.InterferometerArray(None, None, None, init_file=iarray_init_file)
            
            # if 'SPECTRAL INFO' not in extnames:
            #     raise KeyError('No extension table found containing spectral information.')
            # else:
            #     self.f = hdulist['SPECTRAL INFO'].data['frequency']
            #     try:
            #         self.lags = hdulist['SPECTRAL INFO'].data['lag']
            #     except KeyError:
            #         self.lags = None

            try:
                self.f = hdulist['FREQUENCIES'].data
            except KeyError:
                hdulist.close()
                raise KeyError('Extension "FREQUENCIES" nout found in header')

            self.lags = None
            if 'LAGS' in extnames:
                self.lags = hdulist['LAGS'].data

            self.cc_lags = None
            if 'CC_LAGS' in extnames:
                self.cc_lags = hdulist['CC_LAGS'].data

            self.cc_freq = None
            if 'CC_FREQ' in extnames:
                self.cc_freq = hdulist['CC_FREQ'].data
                
            if 'BANDPASS' in extnames:
                self.bp = hdulist['BANDPASS'].data
            else:
                raise KeyError('Extension named "BANDPASS" not found in init_file.')

            if 'BANDPASS WEIGHTS' in extnames:
                self.bp_wts = hdulist['BANDPASS WEIGHTS'].data
            else:
                self.bp_wts = NP.ones_like(self.bp)

            if 'HORIZON LIMITS' in extnames:
                self.horizon_delay_limits = hdulist['HORIZON LIMITS'].data
            else:
                self.set_horizon_delay_limits()

            self.lag_kernel = None
            if 'LAG KERNEL REAL' in extnames:
                self.lag_kernel = hdulist['LAG KERNEL REAL'].data
            if 'LAG KERNEL IMAG' in extnames:
                self.lag_kernel = self.lag_kernel.astype(NP.complex)
                self.lag_kernel += 1j * hdulist['LAG KERNEL IMAG'].data

            self.cc_lag_kernel = None
            if 'CLEAN LAG KERNEL REAL' in extnames:
                self.cc_lag_kernel = hdulist['CLEAN LAG KERNEL REAL'].data
            if 'CLEAN LAG KERNEL IMAG' in extnames:
                self.cc_lag_kernel = self.cc_lag_kernel.astype(NP.complex)
                self.cc_lag_kernel += 1j * hdulist['CLEAN LAG KERNEL IMAG'].data
                
            self.skyvis_lag = None
            if 'NOISELESS DELAY SPECTRA REAL' in extnames:
                self.skyvis_lag = hdulist['NOISELESS DELAY SPECTRA REAL'].data
            if 'NOISELESS DELAY SPECTRA IMAG' in extnames:
                self.skyvis_lag = self.skyvis_lag.astype(NP.complex)
                self.skyvis_lag += 1j * hdulist['NOISELESS DELAY SPECTRA IMAG'].data

            self.vis_lag = None
            if 'NOISY DELAY SPECTRA REAL' in extnames:
                self.vis_lag = hdulist['NOISY DELAY SPECTRA REAL'].data
            if 'NOISY DELAY SPECTRA IMAG' in extnames:
                self.vis_lag = self.vis_lag.astype(NP.complex)
                self.vis_lag += 1j * hdulist['NOISY DELAY SPECTRA IMAG'].data

            self.vis_noise_lag = None
            if 'DELAY SPECTRA NOISE REAL' in extnames:
                self.vis_noise_lag = hdulist['DELAY SPECTRA NOISE REAL'].data
            if 'DELAY SPECTRA NOISE IMAG' in extnames:
                self.vis_noise_lag = self.vis_noise_lag.astype(NP.complex)
                self.vis_noise_lag += 1j * hdulist['DELAY SPECTRA NOISE IMAG'].data
                
            self.cc_skyvis_lag = None
            if 'CLEAN NOISELESS DELAY SPECTRA REAL' in extnames:
                self.cc_skyvis_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA REAL'].data
            if 'CLEAN NOISELESS DELAY SPECTRA IMAG' in extnames:
                self.cc_skyvis_lag = self.cc_skyvis_lag.astype(NP.complex)
                self.cc_skyvis_lag += 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA IMAG'].data

            self.cc_vis_lag = None
            if 'CLEAN NOISY DELAY SPECTRA REAL' in extnames:
                self.cc_vis_lag = hdulist['CLEAN NOISY DELAY SPECTRA REAL'].data
            if 'CLEAN NOISY DELAY SPECTRA IMAG' in extnames:
                self.cc_vis_lag = self.cc_vis_lag.astype(NP.complex)
                self.cc_vis_lag += 1j * hdulist['CLEAN NOISY DELAY SPECTRA IMAG'].data

            self.cc_skyvis_res_lag = None
            if 'CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL' in extnames:
                self.cc_skyvis_res_lag = hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL'].data
            if 'CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG' in extnames:
                self.cc_skyvis_res_lag = self.cc_skyvis_res_lag.astype(NP.complex)
                self.cc_skyvis_res_lag += 1j * hdulist['CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG'].data

            self.cc_vis_res_lag = None
            if 'CLEAN NOISY DELAY SPECTRA RESIDUALS REAL' in extnames:
                self.cc_vis_res_lag = hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS REAL'].data
            if 'CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG' in extnames:
                self.cc_vis_res_lag = self.cc_vis_res_lag.astype(NP.complex)
                self.cc_vis_res_lag += 1j * hdulist['CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG'].data
                
            self.cc_skyvis_freq = None
            if 'CLEAN NOISELESS VISIBILITIES REAL' in extnames:
                self.cc_skyvis_freq = hdulist['CLEAN NOISELESS VISIBILITIES REAL'].data
            if 'CLEAN NOISELESS VISIBILITIES IMAG' in extnames:
                self.cc_skyvis_freq = self.cc_skyvis_freq.astype(NP.complex)
                self.cc_skyvis_freq += 1j * hdulist['CLEAN NOISELESS VISIBILITIES IMAG'].data

            self.cc_vis_freq = None
            if 'CLEAN NOISY VISIBILITIES REAL' in extnames:
                self.cc_vis_freq = hdulist['CLEAN NOISY VISIBILITIES REAL'].data
            if 'CLEAN NOISY VISIBILITIES IMAG' in extnames:
                self.cc_vis_freq = self.cc_vis_freq.astype(NP.complex)
                self.cc_vis_freq += 1j * hdulist['CLEAN NOISY VISIBILITIES IMAG'].data

            self.cc_skyvis_res_freq = None
            if 'CLEAN NOISELESS VISIBILITIES RESIDUALS REAL' in extnames:
                self.cc_skyvis_res_freq = hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS REAL'].data
            if 'CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG' in extnames:
                self.cc_skyvis_res_freq = self.cc_skyvis_res_freq.astype(NP.complex)
                self.cc_skyvis_res_freq += 1j * hdulist['CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG'].data

            self.cc_vis_res_freq = None
            if 'CLEAN NOISY VISIBILITIES RESIDUALS REAL' in extnames:
                self.cc_vis_res_freq = hdulist['CLEAN NOISY VISIBILITIES RESIDUALS REAL'].data
            if 'CLEAN NOISY VISIBILITIES RESIDUALS IMAG' in extnames:
                self.cc_vis_res_freq = self.cc_vis_res_freq.astype(NP.complex)
                self.cc_vis_res_freq += 1j * hdulist['CLEAN NOISY VISIBILITIES RESIDUALS IMAG'].data
                
            self.cc_skyvis_net_lag = None
            if (self.cc_skyvis_lag is not None) and (self.cc_skyvis_res_lag is not None):
                self.cc_skyvis_net_lag = self.cc_skyvis_lag + self.cc_skyvis_res_lag

            self.cc_vis_net_lag = None
            if (self.cc_vis_lag is not None) and (self.cc_vis_res_lag is not None):
                self.cc_vis_net_lag = self.cc_vis_lag + self.cc_vis_res_lag

            self.cc_skyvis_net_freq = None
            if (self.cc_skyvis_freq is not None) and (self.cc_skyvis_res_freq is not None):
                self.cc_skyvis_net_freq = self.cc_skyvis_freq + self.cc_skyvis_res_freq

            self.cc_vis_net_freq = None
            if (self.cc_vis_freq is not None) and (self.cc_vis_res_freq is not None):
                self.cc_vis_net_freq = self.cc_vis_freq + self.cc_vis_res_freq
                
            self.subband_delay_spectra = {}
            self.subband_delay_spectra['cc'] = {}
            self.subband_delay_spectra['sim'] = {}
            if 'SBDS' in hdulist[0].header:
                self.subband_delay_spectra['shape'] = hdulist[0].header['SBDS-WSHAPE']
                self.subband_delay_spectra['bpcorrect'] = bool(hdulist[0].header['SBDS-BPCORR'])
                self.subband_delay_spectra['npad'] = hdulist[0].header['SBDS-NPAD']
                self.subband_delay_spectra['datapool'] = hdulist[0].header['SBDS-DPOOL']
                self.subband_delay_spectra['freq_center'] = hdulist['SBDS-F0'].data
                self.subband_delay_spectra['freq_wts'] = hdulist['SBDS-FWTS'].data
                self.subband_delay_spectra['bw_eff'] = hdulist['SBDS-BWEFF'].data
                self.subband_delay_spectra['lags'] = hdulist['SBDS-LAGS'].data
                self.subband_delay_spectra['lag_kernel'] = hdulist['SBDS-LAGKERN-REAL'].data + 1j * hdulist['SBDS-LAGKERN-IMAG'].data
                self.subband_delay_spectra['lag_corr_length'] = hdulist['SBDS-LAGCORR'].data
                self.subband_delay_spectra['skyvis_lag'] = hdulist['SBDS-SKYVISLAG-REAL'].data + 1j * hdulist['SBDS-SKYVISLAG-IMAG'].data
                self.subband_delay_spectra['vis_lag'] = hdulist['SBDS-VISLAG-REAL'].data + 1j * hdulist['SBDS-VISLAG-IMAG'].data
                self.subband_delay_spectra['vis_noise_lag'] = hdulist['SBDS-NOISELAG-REAL'].data + 1j * hdulist['SBDS-NOISELAG-IMAG'].data
                if self.subband_delay_spectra['datapool'] == 'ccvis':
                    self.subband_delay_spectra['skyvis_res_lag'] = hdulist['SBDS-CCSKYVISRESLAG-REAL'].data + 1j * hdulist['SBDS-CCSKYVISRESLAG-IMAG'].data
                    self.subband_delay_spectra['vis_res_lag'] = hdulist['SBDS-CCVISRESLAG-REAL'].data + 1j * hdulist['SBDS-CCVISRESLAG-IMAG'].data

            hdulist.close()
            init_file_success = True
            return
        else:
            argument_init = True

        if (not argument_init) and (not init_file_success):
            raise ValueError('Initialization failed with the use of init_file.')

        if not isinstance(interferometer_array, RI.InterferometerArray):
            raise TypeError('Input interferometer_array must be an instance of class InterferometerArray')

        self.ia = interferometer_array
        self.f = interferometer_array.channels
        self.df = interferometer_array.freq_resolution
        self.n_acc = interferometer_array.n_acc
        self.horizon_delay_limits = self.get_horizon_delay_limits()

        self.bp = interferometer_array.bp # Inherent bandpass shape
        self.bp_wts = interferometer_array.bp_wts # Additional bandpass weights

        self.pad = 0.0
        self.lags = DSP.spectral_axis(self.f.size, delx=self.df, use_real=False, shift=True)
        self.lag_kernel = None

        self.skyvis_lag = None
        self.vis_lag = None
        self.vis_noise_lag = None

        self.clean_window_buffer = 1.0

        self.cc_lags = None
        self.cc_freq = None
        self.cc_lag_kernel = None
        self.cc_skyvis_lag = None
        self.cc_skyvis_res_lag = None
        self.cc_vis_lag = None
        self.cc_vis_res_lag = None

        self.cc_skyvis_net_lag = None
        self.cc_vis_net_lag = None

        self.cc_skyvis_freq = None
        self.cc_skyvis_res_freq = None
        self.cc_vis_freq = None
        self.cc_vis_res_freq = None

        self.cc_skyvis_net_freq = None
        self.cc_vis_net_freq = None

        self.subband_delay_spectra = {}
        self.subband_delay_spectra['cc'] = {}
        self.subband_delay_spectra['sim'] = {}

    #############################################################################

    def delay_transform(self, pad=1.0, freq_wts=None, downsample=True,
                        verbose=True):

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

        downsample  [boolean] If set to True (default), the delay transform
                    quantities will be downsampled by exactly the same factor
                    that was used in padding. For instance, if pad is set to 
                    1.0, the downsampling will be by a factor of 2. If set to 
                    False, no downsampling will be done even if the original 
                    quantities were padded 

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
            if freq_wts.size == self.f.size:
                freq_wts = NP.repeat(NP.expand_dims(NP.repeat(freq_wts.reshape(1,-1), self.ia.baselines.shape[0], axis=0), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.f.size * self.n_acc:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(self.f.size, -1), axis=0), self.ia.baselines.shape[0], axis=0)
            elif freq_wts.size == self.f.size * self.ia.baselines.shape[0]:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(-1, self.f.size), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.f.size * self.ia.baselines.shape[0] * self.n_acc:
                freq_wts = freq_wts.reshape(self.ia.baselines.shape[0], self.f.size, self.n_acc)
            else:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
            self.bp_wts = freq_wts
            if verbose:
                print '\tFrequency window weights assigned.'

        if not isinstance(downsample, bool):
            raise TypeError('Input downsample must be of boolean type')

        if verbose:
            print '\tInput parameters have been verified to be compatible.\n\tProceeding to compute delay transform.'
            
        self.lags = DSP.spectral_axis(int(self.f.size*(1+pad)), delx=self.df, use_real=False, shift=True)
        if pad == 0.0:
            self.vis_lag = DSP.FT1D(self.ia.vis_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            self.skyvis_lag = DSP.FT1D(self.ia.skyvis_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            self.vis_noise_lag = DSP.FT1D(self.ia.vis_noise_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            self.lag_kernel = DSP.FT1D(self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            if verbose:
                print '\tDelay transform computed without padding.'
        else:
            npad = int(self.f.size * pad)
            self.vis_lag = DSP.FT1D(NP.pad(self.ia.vis_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            self.skyvis_lag = DSP.FT1D(NP.pad(self.ia.skyvis_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            self.vis_noise_lag = DSP.FT1D(NP.pad(self.ia.vis_noise_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            self.lag_kernel = DSP.FT1D(NP.pad(self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df

            if verbose:
                print '\tDelay transform computed with padding fraction {0:.1f}'.format(pad)
        if downsample:
            self.vis_lag = DSP.downsampler(self.vis_lag, 1+pad, axis=1)
            self.skyvis_lag = DSP.downsampler(self.skyvis_lag, 1+pad, axis=1)
            self.vis_noise_lag = DSP.downsampler(self.vis_noise_lag, 1+pad, axis=1)
            self.lag_kernel = DSP.downsampler(self.lag_kernel, 1+pad, axis=1)
            self.lags = DSP.downsampler(self.lags, 1+pad)
            self.lags = self.lags.flatten()
            if verbose:
                print '\tDelay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'delay_transform() completed successfully.'

        self.pad = pad

    #############################################################################
        
    def clean(self, pad=1.0, freq_wts=None, clean_window_buffer=1.0,
              verbose=True):

        """
        ------------------------------------------------------------------------
        TO BE DEPRECATED!!! USE MEMBER FUNCTION delayClean()

        Transforms the visibilities from frequency axis onto delay (time) axis
        using an IFFT and deconvolves the delay transform quantities along the 
        delay axis. This is performed for noiseless sky visibilities, thermal
        noise in visibilities, and observed visibilities. 

        Inputs:

        pad         [scalar] Non-negative scalar indicating padding fraction 
                    relative to the number of frequency channels. For e.g., a 
                    pad of 1.0 pads the frequency axis with zeros of the same 
                    width as the number of channels. If a negative value is 
                    specified, delay transform will be performed with no padding

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

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'
    
        if freq_wts is not None:
            if freq_wts.size == self.f.size:
                freq_wts = NP.repeat(NP.expand_dims(NP.repeat(freq_wts.reshape(1,-1), self.ia.baselines.shape[0], axis=0), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.f.size * self.n_acc:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(self.f.size, -1), axis=0), self.ia.baselines.shape[0], axis=0)
            elif freq_wts.size == self.f.size * self.ia.baselines.shape[0]:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(-1, self.f.size), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.f.size * self.ia.baselines.shape[0] * self.n_acc:
                freq_wts = freq_wts.reshape(self.ia.baselines.shape[0], self.f.size, self.n_acc)
            else:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
            self.bp_wts = freq_wts
            if verbose:
                print '\tFrequency window weights assigned.'

        bw = self.df * self.f.size
        pc = self.ia.phase_center
        pc_coords = self.ia.phase_center_coords
        if pc_coords == 'hadec':
            pc_altaz = GEOM.hadec2altaz(pc, self.ia.latitude, units='degrees')
            pc_dircos = GEOM.altaz2dircos(pc_altaz, units='degrees')
        elif pc_coords == 'altaz':
            pc_dircos = GEOM.altaz2dircos(pc, units='degrees')
        
        npad = int(self.f.size * pad)
        lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=False)
        dlag = lags[1] - lags[0]
    
        clean_area = NP.zeros(self.f.size + npad, dtype=int)
        skyvis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.skyvis_freq*self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        vis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.vis_freq*self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        lag_kernel = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.bp, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        
        ccomponents_noiseless = NP.zeros_like(skyvis_lag)
        ccres_noiseless = NP.zeros_like(skyvis_lag)
    
        ccomponents_noisy = NP.zeros_like(vis_lag)
        ccres_noisy = NP.zeros_like(vis_lag)
        
        for snap_iter in xrange(self.n_acc):
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baselines '.format(self.ia.baselines.shape[0]), PGB.ETA()], maxval=self.ia.baselines.shape[0]).start()
            for bl_iter in xrange(self.ia.baselines.shape[0]):
                clean_area[NP.logical_and(lags <= self.horizon_delay_limits[snap_iter,bl_iter,1]+clean_window_buffer/bw, lags >= self.horizon_delay_limits[snap_iter,bl_iter,0]-clean_window_buffer/bw)] = 1
    
                cc_noiseless, info_noiseless = _gentle_clean(skyvis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=False, autoscale=True)
                ccomponents_noiseless[bl_iter,:,snap_iter] = cc_noiseless
                ccres_noiseless[bl_iter,:,snap_iter] = info_noiseless['res']
    
                cc_noisy, info_noisy = _gentle_clean(vis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=False, autoscale=True)
                ccomponents_noisy[bl_iter,:,snap_iter] = cc_noisy
                ccres_noisy[bl_iter,:,snap_iter] = info_noisy['res']
    
                progress.update(bl_iter+1)
            progress.finish()
    
        deta = lags[1] - lags[0]
        cc_skyvis = NP.fft.fft(ccomponents_noiseless, axis=1) * deta
        cc_skyvis_res = NP.fft.fft(ccres_noiseless, axis=1) * deta
    
        cc_vis = NP.fft.fft(ccomponents_noisy, axis=1) * deta
        cc_vis_res = NP.fft.fft(ccres_noisy, axis=1) * deta
    
        self.skyvis_lag = NP.fft.fftshift(skyvis_lag, axes=1)
        self.vis_lag = NP.fft.fftshift(vis_lag, axes=1)
        self.lag_kernel = NP.fft.fftshift(lag_kernel, axes=1)
        self.cc_skyvis_lag = NP.fft.fftshift(ccomponents_noiseless, axes=1)
        self.cc_skyvis_res_lag = NP.fft.fftshift(ccres_noiseless, axes=1)
        self.cc_vis_lag = NP.fft.fftshift(ccomponents_noisy, axes=1)
        self.cc_vis_res_lag = NP.fft.fftshift(ccres_noisy, axes=1)

        self.cc_skyvis_net_lag = self.cc_skyvis_lag + self.cc_skyvis_res_lag
        self.cc_vis_net_lag = self.cc_vis_lag + self.cc_vis_res_lag
        self.lags = NP.fft.fftshift(lags)

        self.cc_skyvis_freq = cc_skyvis
        self.cc_skyvis_res_freq = cc_skyvis_res
        self.cc_vis_freq = cc_vis
        self.cc_vis_res_freq = cc_vis_res

        self.cc_skyvis_net_freq = cc_skyvis + cc_skyvis_res
        self.cc_vis_net_freq = cc_vis + cc_vis_res

        self.clean_window_buffer = clean_window_buffer
        
    #############################################################################
        
    def delayClean(self, pad=1.0, freq_wts=None, clean_window_buffer=1.0,
                   gain=0.1, maxiter=10000, threshold=5e-3,
                   threshold_type='relative', parallel=False, nproc=None,
                   verbose=True):

        """
        ------------------------------------------------------------------------
        Transforms the visibilities from frequency axis onto delay (time) axis
        using an IFFT and deconvolves the delay transform quantities along the 
        delay axis. This is performed for noiseless sky visibilities, thermal
        noise in visibilities, and observed visibilities. This calls an in-house
        module complex1dClean instead of the clean routine in AIPY module. It
        can utilize parallelization

        Inputs:

        pad      [scalar] Non-negative scalar indicating padding fraction 
                 relative to the number of frequency channels. For e.g., a 
                 pad of 1.0 pads the frequency axis with zeros of the same 
                 width as the number of channels. If a negative value is 
                 specified, delay transform will be performed with no padding

        freq_wts [numpy vector or array] window shaping to be applied before
                 computing delay transform. It can either be a vector or size
                 equal to the number of channels (which will be applied to all
                 time instances for all baselines), or a nchan x n_snapshots 
                 numpy array which will be applied to all baselines, or a 
                 n_baselines x nchan numpy array which will be applied to all 
                 timestamps, or a n_baselines x nchan x n_snapshots numpy 
                 array. Default (None) will not apply windowing and only the
                 inherent bandpass will be used.

        gain     [scalar] gain factor to be applied while subtracting clean 
                 component from residuals. This is the fraction of the maximum in 
                 the residuals that will be subtracted. Must lie between 0 and 1.
                 A lower value will have a smoother convergence but take a longer 
                 time to converge. Default=0.1

        maxiter  [scalar] maximum number of iterations for cleaning process. Will 
                 terminate if the number of iterations exceed maxiter. 
                 Default=10000

        threshold 
                 [scalar] represents the cleaning depth either as a fraction of 
                 the maximum in the input (when thershold_type is set to 
                 'relative') or the absolute value (when threshold_type is set 
                 to 'absolute') in same units of input down to which inp should 
                 be cleaned. Value must always be positive. When threshold_type 
                 is set to 'relative', threshold mu st lie between 0 and 1. 
                 Default=5e-3 (found to work well and converge fast) assuming 
                 threshold_type is set to 'relative'

        threshold_type
                 [string] represents the type of threshold specified by value in 
                 input threshold. Accepted values are 'relative' and 'absolute'. 
                 If set to 'relative' the threshold value is the fraction 
                 (between 0 and 1) of maximum in input down to which it should 
                 be cleaned. If set to 'asbolute' it is the actual value down to 
                 which inp should be cleaned. Default='relative'

        parallel [boolean] specifies if parallelization is to be invoked. 
                 False (default) means only serial processing

        nproc    [integer] specifies number of independent processes to spawn.
                 Default = None, means automatically determines the number of 
                 process cores in the system and use one less than that to 
                 avoid locking the system for other processes. Applies only 
                 if input parameter 'parallel' (see above) is set to True. 
                 If nproc is set to a value more than the number of process
                 cores in the system, it will be reset to number of process 
                 cores in the system minus one to avoid locking the system out 
                 for other processes

        verbose  [boolean] If set to True (default), print diagnostic and 
                 progress messages. If set to False, no such messages are
                 printed.
        ------------------------------------------------------------------------
        """

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'
    
        if freq_wts is not None:
            if freq_wts.size == self.f.size:
                freq_wts = NP.repeat(NP.expand_dims(NP.repeat(freq_wts.reshape(1,-1), self.ia.baselines.shape[0], axis=0), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.f.size * self.n_acc:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(self.f.size, -1), axis=0), self.ia.baselines.shape[0], axis=0)
            elif freq_wts.size == self.f.size * self.ia.baselines.shape[0]:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(-1, self.f.size), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.f.size * self.ia.baselines.shape[0] * self.n_acc:
                freq_wts = freq_wts.reshape(self.ia.baselines.shape[0], self.f.size, self.n_acc)
            else:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
            self.bp_wts = freq_wts
            if verbose:
                print '\tFrequency window weights assigned.'

        bw = self.df * self.f.size
        pc = self.ia.phase_center
        pc_coords = self.ia.phase_center_coords
        if pc_coords == 'hadec':
            pc_altaz = GEOM.hadec2altaz(pc, self.ia.latitude, units='degrees')
            pc_dircos = GEOM.altaz2dircos(pc_altaz, units='degrees')
        elif pc_coords == 'altaz':
            pc_dircos = GEOM.altaz2dircos(pc, units='degrees')
        
        npad = int(self.f.size * pad)
        lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=False)
        dlag = lags[1] - lags[0]
    
        clean_area = NP.zeros(self.f.size + npad, dtype=int)

        skyvis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.skyvis_freq*self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        vis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.vis_freq*self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        lag_kernel = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)

        # skyvis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.skyvis_freq, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        # vis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.vis_freq, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        # lag_kernel = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.bp, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        
        # lag_kernel = lag_kernel * NP.exp(-1j * 2 * NP.pi * self.f[0] * lags).reshape(1,-1,1)

        ccomponents_noiseless = NP.zeros_like(skyvis_lag)
        ccres_noiseless = NP.zeros_like(skyvis_lag)
    
        ccomponents_noisy = NP.zeros_like(vis_lag)
        ccres_noisy = NP.zeros_like(vis_lag)
        
        if parallel:
            if nproc is None:
                nproc = min(max(MP.cpu_count()-1, 1), self.ia.baselines.shape[0]*self.n_acc)
            else:
                nproc = min(max(MP.cpu_count()-1, 1), self.ia.baselines.shape[0]*self.n_acc, nproc)

            list_of_skyvis_lag = []
            list_of_vis_lag = []
            list_of_dkern = []
            list_of_cboxes = []
            for bli in xrange(self.ia.baselines.shape[0]):
                for ti in xrange(self.n_acc):
                    list_of_skyvis_lag += [skyvis_lag[bli,:,ti]]
                    list_of_vis_lag += [vis_lag[bli,:,ti]]
                    list_of_dkern += [lag_kernel[bli,:,ti]]
                    clean_area = NP.zeros(self.f.size + npad, dtype=int)
                    clean_area[NP.logical_and(lags <= self.horizon_delay_limits[ti,bli,1]+clean_window_buffer/bw, lags >= self.horizon_delay_limits[ti,bli,0]-clean_window_buffer/bw)] = 1
                    list_of_cboxes += [clean_area]
            list_of_gains = [gain] * self.ia.baselines.shape[0]*self.n_acc
            list_of_maxiter = [maxiter] * self.ia.baselines.shape[0]*self.n_acc
            list_of_thresholds = [threshold] * self.ia.baselines.shape[0]*self.n_acc
            list_of_threshold_types = [threshold_type] * self.ia.baselines.shape[0]*self.n_acc
            list_of_verbosity = [verbose] * self.ia.baselines.shape[0]*self.n_acc
            list_of_pid = range(self.ia.baselines.shape[0]*self.n_acc)
            # list_of_pid = [None] * self.ia.baselines.shape[0]*self.n_acc
            list_of_progressbars = [True] * self.ia.baselines.shape[0]*self.n_acc
            list_of_progressbar_ylocs = NP.arange(self.ia.baselines.shape[0]*self.n_acc) % min(nproc, WM.term.height)
            list_of_progressbar_ylocs = list_of_progressbar_ylocs.tolist()

            pool = MP.Pool(processes=nproc)
            list_of_noiseless_cleanstates = pool.map(complex1dClean_arg_splitter, IT.izip(list_of_skyvis_lag, list_of_dkern, list_of_cboxes, list_of_gains, list_of_maxiter, list_of_thresholds, list_of_threshold_types, list_of_verbosity, list_of_progressbars, list_of_pid, list_of_progressbar_ylocs))
            list_of_noisy_cleanstates = pool.map(complex1dClean_arg_splitter, IT.izip(list_of_vis_lag, list_of_dkern, list_of_cboxes, list_of_gains, list_of_maxiter, list_of_thresholds, list_of_threshold_types, list_of_verbosity, list_of_progressbars, list_of_pid, list_of_progressbar_ylocs))
                
            for bli in xrange(self.ia.baselines.shape[0]):
                for ti in xrange(self.n_acc):
                    ind = bli * self.n_acc + ti
                    noiseless_cleanstate = list_of_noiseless_cleanstates[ind]
                    ccomponents_noiseless[bli,:,ti] = noiseless_cleanstate['cc']
                    ccres_noiseless[bli,:,ti] = noiseless_cleanstate['res']

                    noisy_cleanstate = list_of_noisy_cleanstates[ind]
                    ccomponents_noisy[bli,:,ti] = noisy_cleanstate['cc']
                    ccres_noisy[bli,:,ti] = noisy_cleanstate['res']
        else:
            for snap_iter in xrange(self.n_acc):
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baselines '.format(self.ia.baselines.shape[0]), PGB.ETA()], maxval=self.ia.baselines.shape[0]).start()
                for bl_iter in xrange(self.ia.baselines.shape[0]):
                    clean_area[NP.logical_and(lags <= self.horizon_delay_limits[snap_iter,bl_iter,1]+clean_window_buffer/bw, lags >= self.horizon_delay_limits[snap_iter,bl_iter,0]-clean_window_buffer/bw)] = 1
        
                    cleanstate = complex1dClean(skyvis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], cbox=clean_area, gain=gain, maxiter=maxiter, threshold=threshold, threshold_type=threshold_type, verbose=verbose)
                    ccomponents_noiseless[bl_iter,:,snap_iter] = cleanstate['cc']
                    ccres_noiseless[bl_iter,:,snap_iter] = cleanstate['res']
    
                    cleanstate = complex1dClean(vis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], cbox=clean_area, gain=gain, maxiter=maxiter, threshold=threshold, threshold_type=threshold_type, verbose=verbose)
                    ccomponents_noisy[bl_iter,:,snap_iter] = cleanstate['cc']
                    ccres_noisy[bl_iter,:,snap_iter] = cleanstate['res']
                    
                    progress.update(bl_iter+1)
                progress.finish()
    
        deta = lags[1] - lags[0]
        cc_skyvis = NP.fft.fft(ccomponents_noiseless, axis=1) * deta
        cc_skyvis_res = NP.fft.fft(ccres_noiseless, axis=1) * deta
    
        cc_vis = NP.fft.fft(ccomponents_noisy, axis=1) * deta
        cc_vis_res = NP.fft.fft(ccres_noisy, axis=1) * deta
    
        self.skyvis_lag = NP.fft.fftshift(skyvis_lag, axes=1)
        self.vis_lag = NP.fft.fftshift(vis_lag, axes=1)
        self.lag_kernel = NP.fft.fftshift(lag_kernel, axes=1)
        self.cc_lag_kernel = NP.fft.fftshift(lag_kernel, axes=1)        
        self.cc_skyvis_lag = NP.fft.fftshift(ccomponents_noiseless, axes=1)
        self.cc_skyvis_res_lag = NP.fft.fftshift(ccres_noiseless, axes=1)
        self.cc_vis_lag = NP.fft.fftshift(ccomponents_noisy, axes=1)
        self.cc_vis_res_lag = NP.fft.fftshift(ccres_noisy, axes=1)

        self.cc_skyvis_net_lag = self.cc_skyvis_lag + self.cc_skyvis_res_lag
        self.cc_vis_net_lag = self.cc_vis_lag + self.cc_vis_res_lag
        self.cc_lags = NP.fft.fftshift(lags)

        self.cc_skyvis_freq = cc_skyvis
        self.cc_skyvis_res_freq = cc_skyvis_res
        self.cc_vis_freq = cc_vis
        self.cc_vis_res_freq = cc_vis_res

        self.cc_skyvis_net_freq = cc_skyvis + cc_skyvis_res
        self.cc_vis_net_freq = cc_vis + cc_vis_res

        self.clean_window_buffer = clean_window_buffer
        
    #############################################################################
        
    def subband_delay_transform(self, bw_eff, freq_center=None, shape=None,
                                pad=None, bpcorrect=False, action=None,
                                verbose=True):

        """
        ------------------------------------------------------------------------
        Computes delay transform on multiple frequency sub-bands with specified
        weights

        Inputs:

        bw_eff       [dictionary] dictionary with two keys 'cc' and 'sim' to
                     specify effective bandwidths (in Hz) on the elected 
                     frequency windows for subband delay 
                     transform of CLEANed and simulated visibilities 
                     respectively. The values under these keys can be a scalar, 
                     list or numpy array and are independent of each other. If 
                     a scalar value is provided, the same will be applied to all 
                     frequency windows under that key

        freq_center  [dictionary] dictionary with two keys 'cc' and 'sim' to 
                     specify frequency centers (in Hz) of the selected frequency 
                     windows for subband delay transform of CLEANed and 
                     simulated visibilities respectively. The values under these 
                     keys can be a scalar, list or numpy array and are 
                     independent of each other. If a scalar is provided, the
                     same will be applied to all frequency windows. Default=None
                     uses the center frequency from the class attribute named 
                     channels for both keys 'cc' and 'sim'

        shape        [dictionary] dictionary with two keys 'cc' and 'sim' to 
                     specify frequency window shape for subband delay transform 
                     of CLEANed and simulated visibilities respectively. Values 
                     held by the keys must be a string. Accepted values for the
                     string are 'rect' or 'RECT' (for rectangular), 'bnw' and 
                     'BNW' (for Blackman-Nuttall), and 'bhw' or 'BHW' (for 
                     Blackman-Harris). Default=None sets it to 'rect' 
                     (rectangular window) for both keys

        pad          [dictionary] dictionary with two keys 'cc' and 'sim' to 
                     specify padding fraction relative to the number of frequency 
                     channels for CLEANed and simualted visibilities respectively. 
                     Values held by the keys must be a non-negative scalar. For 
                     e.g., a pad of 1.0 pads the frequency axis with zeros of 
                     the same width as the number of channels. After the delay 
                     transform, the transformed visibilities are downsampled by a 
                     factor of 1+pad. If a negative value is specified, delay 
                     transform will be performed with no padding. Default=None 
                     sets to padding factor to 1.0 under both keys.

        bpcorrect    [boolean] Only applicable on delay CLEANed visibilities. 
                     If True, correct for frequency weights that were applied 
                     during the original delay transform using which the delay 
                     CLEAN was done. This would flatten the bandpass after delay
                     CLEAN. If False (default), do not apply the correction, 
                     namely, inverse of bandpass weights

        action       [string or None] If set to 'return' it returns the output 
                     dictionary and updates its attribute 
                     subband_delay_spectra else just updates the attribute.
                     Default=None (just updates the attribute)

        verbose      [boolean] If set to True (default), print diagnostic and 
                     progress messages. If set to False, no such messages are
                     printed.

        Output: 

        If keyword input action is set to None (default), the output
        is internally stored in the class attribute 
        subband_delay_spectra. If action is set to 'return', this 
        output is returned. The output is a dictionary that contains two top
        level keys, namely, 'cc' and 'sim' denoting information about CLEAN
        and simulated visibilities respectively. Under each of these keys is
        information about delay spectra of different frequency sub-bands (n_win 
        in number) in the form of a dictionary under the following keys:
        'freq_center' 
                    [numpy array] contains the center frequencies 
                    (in Hz) of the frequency subbands of the subband
                    delay spectra. It is of size n_win. It is roughly 
                    equivalent to redshift(s)
        'freq_wts'  [numpy array] Contains frequency weights applied 
                    on each frequency sub-band during the subband delay 
                    transform. It is of size n_win x nchan. 
        'bw_eff'    [numpy array] contains the effective bandwidths 
                    (in Hz) of the subbands being delay transformed. It
                    is of size n_win. It is roughly equivalent to width 
                    in redshift or along line-of-sight
        'shape'     [string] shape of the window function applied. 
                    Accepted values are 'rect' (rectangular), 'bhw'
                    (Blackman-Harris), 'bnw' (Blackman-Nuttall). 
        'bpcorrect' [boolean] If True (default), correct for frequency
                    weights that were applied during the original 
                    delay transform using which the delay CLEAN was 
                    done. This would flatten the bandpass after delay
                    CLEAN. If False, do not apply the correction, 
                    namely, inverse of bandpass weights. This applies only 
                    CLEAned visibilities under the 'cc' key and hence is
                    present only if the top level key is 'cc' and absent
                    for key 'sim'
        'npad'      [scalar] Numbber of zero-padded channels before
                    performing the subband delay transform. 
        'lags'      [numpy array] lags of the subband delay spectra 
                    after padding in frequency during the transform. It
                    is of size nchan+npad where npad is the number of 
                    frequency channels padded specified under the key 
                    'npad'
        'lag_kernel'
                    [numpy array] delay transform of the frequency 
                    weights under the key 'freq_wts'. It is of size
                    n_bl x n_win x (nchan+npad) x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the total bandwidth 
                    (nchan x df) simulated.
        'skyvis_lag'
                    [numpy array] subband delay spectra of simulated 
                    or CLEANed noiseless visibilities, depending on whether
                    the top level key is 'cc' or 'sim' respectively,
                    after applying the frequency weights under the key 
                    'freq_wts'. It is of size 
                    n_bl x n_win x (nchan+npad) x n_t. 
        'vis_lag'   [numpy array] subband delay spectra of simulated 
                    or CLEANed noisy visibilities, depending on whether
                    the top level key is 'cc' or 'sim' respectively,
                    after applying the frequency weights under the key 
                    'freq_wts'. It is of size 
                    n_bl x n_win x (nchan+npad) x n_t. 
        'vis_noise_lag'   
                    [numpy array] subband delay spectra of simulated 
                    noise after applying the frequency weights under 
                    the key 'freq_wts'. Only present if top level key is 'sim'
                    and absent for 'cc'. It is of size 
                    n_bl x n_win x (nchan+npad) x n_t. 
        'skyvis_res_lag'
                    [numpy array] subband delay spectra of residuals
                    after delay CLEAN of simualted noiseless 
                    visibilities obtained after applying frequency 
                    weights specified under key 'freq_wts'. Only present for
                    top level key 'cc' and absent for 'sim'. It is of
                    size n_bl x n_win x (nchan+npad) x n_t
        'vis_res_lag'
                    [numpy array] subband delay spectra of residuals
                    after delay CLEAN of simualted noisy 
                    visibilities obtained after applying frequency 
                    weights specified under key 'freq_wts'. Only present for
                    top level key 'cc' and absent for 'sim'. It is of
                    size n_bl x n_win x (nchan+npad) x n_t
        ------------------------------------------------------------------------
        """

        try:
            bw_eff
        except NameError:
            raise NameError('Effective bandwidth must be specified')
        else:
            if not isinstance(bw_eff, dict):
                raise TypeError('Effective bandiwdth must be specified as a dictionary')
            for key in ['cc','sim']:
                if key in bw_eff:
                    if not isinstance(bw_eff[key], (int, float, list, NP.ndarray)):
                        raise TypeError('Value of effective bandwidth must be a scalar, list or numpy array')
                    bw_eff[key] = NP.asarray(bw_eff[key]).reshape(-1)
                    if NP.any(bw_eff[key] <= 0.0):
                        raise ValueError('All values in effective bandwidth must be strictly positive')

        if freq_center is None:
            freq_center = {key: NP.asarray(self.f[self.f.size/2]).reshape(-1) for key in ['cc', 'sim']}
            # freq_center = NP.asarray(self.f[self.f.size/2]).reshape(-1)
        elif isinstance(freq_center, dict):
            for key in ['cc', 'sim']:
                if isinstance(freq_center[key], (int, float, list, NP.ndarray)):
                    freq_center[key] = NP.asarray(freq_center[key]).reshape(-1)
                    if NP.any((freq_center[key] <= self.f.min()) | (freq_center[key] >= self.f.max())):
                        raise ValueError('Value(s) of frequency center(s) must lie strictly inside the observing band')

                else:
                    raise TypeError('Values(s) of frequency center must be scalar, list or numpy array')
        else:
            raise TypeError('Input frequency center must be specified as a dictionary')

        for key in ['cc', 'sim']:
            if (bw_eff[key].size == 1) and (freq_center[key].size > 1):
                bw_eff[key] = NP.repeat(bw_eff[key], freq_center[key].size)
            elif (bw_eff[key].size > 1) and (freq_center[key].size == 1):
                freq_center[key] = NP.repeat(freq_center[key], bw_eff[key].size)
            elif bw_eff[key].size != freq_center[key].size:
                raise ValueError('Effective bandwidth(s) and frequency center(s) must have same number of elements')
            
        if shape is not None:
            if not isinstance(shape, dict):
                raise TypeError('Window shape must be specified as a dictionary')
            for key in ['cc', 'sim']:
                if not isinstance(shape[key], str):
                    raise TypeError('Window shape must be a string')
                if shape[key] not in ['rect', 'bhw', 'bnw', 'RECT', 'BHW', 'BNW']:
                    raise ValueError('Invalid value for window shape specified.')
        else:
            shape = {key: 'rect' for key in ['cc', 'sim']}
            # shape = 'rect'

        if pad is None:
            pad = {key: 1.0 for key in ['cc', 'sim']}
        else:
            if not isinstance(pad, dict):
                raise TypeError('Padding for delay transform must be specified as a dictionary')
            for key in ['cc', 'sim']:
                if not isinstance(pad[key], (int, float)):
                    raise TypeError('pad fraction must be a scalar value.')
                if pad[key] < 0.0:
                    pad[key] = 0.0
                    if verbose:
                        print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        if not isinstance(bpcorrect, bool):
            raise TypeError('Input keyword bpcorrect must be of boolean type')

        vis_noise_freq = NP.copy(self.ia.vis_noise_freq)
        result = {}
        for key in ['cc', 'sim']:
            if (key == 'sim') or ((key == 'cc') and (self.cc_lags is not None)):
                freq_wts = NP.empty((bw_eff[key].size, self.f.size))
                frac_width = DSP.window_N2width(n_window=None, shape=shape[key])
                window_loss_factor = 1 / frac_width
                n_window = NP.round(window_loss_factor * bw_eff[key] / self.df).astype(NP.int)
                ind_freq_center, ind_channels, dfrequency = LKP.find_1NN(self.f.reshape(-1,1), freq_center[key].reshape(-1,1), distance_ULIM=0.5*self.df, remove_oob=True)
                sortind = NP.argsort(ind_channels)
                ind_freq_center = ind_freq_center[sortind]
                ind_channels = ind_channels[sortind]
                dfrequency = dfrequency[sortind]
                n_window = n_window[sortind]
    
                for i,ind_chan in enumerate(ind_channels):
                    window = DSP.windowing(n_window[i], shape=shape[key], centering=True)
                    window_chans = self.f[ind_chan] + self.df * (NP.arange(n_window[i]) - int(n_window[i]/2))
                    ind_window_chans, ind_chans, dfreq = LKP.find_1NN(self.f.reshape(-1,1), window_chans.reshape(-1,1), distance_ULIM=0.5*self.df, remove_oob=True)
                    sind = NP.argsort(ind_window_chans)
                    ind_window_chans = ind_window_chans[sind]
                    ind_chans = ind_chans[sind]
                    dfreq = dfreq[sind]
                    window = window[ind_window_chans]
                    window = NP.pad(window, ((ind_chans.min(), self.f.size-1-ind_chans.max())), mode='constant', constant_values=((0.0,0.0)))
                    freq_wts[i,:] = window
    
                bpcorrection_factor = 1.0
                npad = int(self.f.size * pad[key])
                lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=True)
    
                if key == 'cc':
                    skyvis_freq = self.cc_skyvis_freq[:,:self.f.size,:]
                    vis_freq = self.cc_vis_freq[:,:self.f.size,:]
                    skyvis_res_freq = self.cc_skyvis_res_freq[:,:self.f.size,:]
                    vis_res_freq = self.cc_vis_res_freq[:,:self.f.size,:]
                    if bpcorrect:
                        bpcorrection_factor = NP.where(NP.abs(self.bp_wts)>0.0, 1/self.bp_wts, 0.0)
                        bpcorrection_factor = bpcorrection_factor[:,NP.newaxis,:,:]
                else:
                    skyvis_freq = NP.copy(self.ia.skyvis_freq)
                    vis_freq = NP.copy(self.ia.vis_freq)
    
                skyvis_lag = DSP.FT1D(NP.pad(skyvis_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                vis_lag = DSP.FT1D(NP.pad(vis_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df        
                vis_noise_lag = DSP.FT1D(NP.pad(vis_noise_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                lag_kernel = DSP.FT1D(NP.pad(self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                result[key] = {'freq_center': freq_center[key], 'shape': shape[key], 'freq_wts': freq_wts, 'bw_eff': bw_eff[key], 'npad': npad, 'lags': lags, 'skyvis_lag': skyvis_lag, 'vis_lag': vis_lag, 'lag_kernel': lag_kernel, 'lag_corr_length': self.f.size / NP.sum(freq_wts, axis=1)}
                if key == 'cc':
                    skyvis_res_lag = DSP.FT1D(NP.pad(skyvis_res_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                    vis_res_lag = DSP.FT1D(NP.pad(vis_res_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                    result[key]['vis_res_lag'] = vis_res_lag
                    result[key]['skyvis_res_lag'] = skyvis_res_lag
                    result[key]['bpcorrect'] = bpcorrect
                else:
                    result[key]['vis_noise_lag'] = vis_noise_lag
        if verbose:
            print '\tSub-band(s) delay transform computed'

        self.subband_delay_spectra = result
        if action == 'return':
            return result

    #############################################################################

    def get_horizon_delay_limits(self, phase_center=None,
                                 phase_center_coords=None):

        """
        -------------------------------------------------------------------------
        Estimates the delay envelope determined by the sky horizon for the 
        baseline(s) for the phase centers 
    
        Inputs:
    
        phase_center
                [numpy array] Phase center of the observation as 2-column or
                3-column numpy array. Two columns are used when it is specified
                in 'hadec' or 'altaz' coordinates as indicated by the input 
                phase_center_coords or by three columns when 'dircos' coordinates 
                are used. This is where the telescopes will be phased up to as 
                reference. Coordinate system for the phase_center is specified 
                by another input phase_center_coords. Default=None implies the 
                corresponding attribute from the DelaySpectrum instance is used.
                This is a Nx2 or Nx3 array

        phase_center_coords
                [string] Coordinate system for array phase center. Accepted 
                values are 'hadec' (HA-Dec), 'altaz' (Altitude-Azimuth) or
                'dircos' (direction cosines). Default=None implies the 
                corresponding attribute from the DelaySpectrum instance is used.

        Outputs:
        
        horizon_envelope: 
             NxMx2 matrix where M is the number of baselines and N is the number 
             of phase centers. horizon_envelope[:,:,0] contains the minimum delay 
             after accounting for (any) non-zenith phase center.
             horizon_envelope[:,:,1] contains the maximum delay after accounting 
             for (any) non-zenith phase center(s).
        -------------------------------------------------------------------------
        """

        if phase_center is None:
            phase_center = self.ia.phase_center
            phase_center_coords = self.ia.phase_center_coords

        if phase_center_coords not in ['hadec', 'altaz', 'dircos']:
            raise ValueError('Phase center coordinates must be "altaz", "hadec" or "dircos"')
        
        if phase_center_coords == 'hadec':
            pc_altaz = GEOM.hadec2altaz(phase_center, self.ia.latitude, units='degrees')
            pc_dircos = GEOM.altaz2dircos(pc_altaz, units='degrees')
        elif phase_center_coords == 'altaz':
            pc_dircos = GEOM.altaz2dircos(phase_center, units='degrees')
        elif phase_center_coords == 'dircos':
            pc_dircos = phase_center

        horizon_envelope = DLY.horizon_delay_limits(self.ia.baselines, pc_dircos, units='mks')
        return horizon_envelope
        
    #############################################################################
        
    def set_horizon_delay_limits(self):

        """
        -------------------------------------------------------------------------
        Estimates the delay envelope determined by the sky horizon for the 
        baseline(s) for the phase centers of the DelaySpectrum instance. No 
        output is returned. Uses the member function get_horizon_delay_limits()
        -------------------------------------------------------------------------
        """

        self.horizon_delay_limits = self.get_horizon_delay_limits()
        
    #############################################################################
        
    def save(self, outfile, tabtype='BinTabelHDU', overwrite=False,
             verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the interferometer array delay spectrum information to disk. 

        Inputs:

        outfile      [string] Filename with full path to be saved to. Will be
                     appended with '.fits' extension for the interferometer array
                     data and '.cc.fits' for delay spectrum data

        Keyword Input(s):

        tabtype      [string] indicates table type for one of the extensions in 
                     the FITS file. Allowed values are 'BinTableHDU' and 
                     'TableHDU' for binary and ascii tables respectively. Default 
                     is 'BinTableHDU'.
                     
        overwrite    [boolean] True indicates overwrite even if a file already 
                     exists. Default = False (does not overwrite)
                     
        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('No filename provided. Aborting DelaySpectrum.save()...')

        if verbose:
            print '\nSaving information about interferometer array...'

        self.ia.save(outfile, tabtype=tabtype, overwrite=overwrite,
                     verbose=verbose)

        if verbose:
            print '\nSaving information about delay spectra...'

        hdulist = []
        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['EXTNAME'] = 'PRIMARY'
        hdulist[0].header['NCHAN'] = (self.f.size, 'Number of frequency channels')
        hdulist[0].header['NLAGS'] = (self.lags.size, 'Number of lags')
        hdulist[0].header['freq_resolution'] = (self.df, 'Frequency resolution (Hz)')
        hdulist[0].header['N_ACC'] = (self.n_acc, 'Number of accumulations')
        hdulist[0].header['PAD'] = (self.pad, 'Padding factor')
        hdulist[0].header['DBUFFER'] = (self.clean_window_buffer, 'CLEAN window buffer (1/bandwidth)')
        hdulist[0].header['IARRAY'] = (outfile+'.fits', 'Location of InterferometerArray simulated visibilities')

        if verbose:
            print '\tCreated a primary HDU.'

        # cols = []
        # cols += [fits.Column(name='frequency', format='D', array=self.f)]
        # cols += [fits.Column(name='lag', format='D', array=self.lags)]
        # columns = _astropy_columns(cols, tabtype=tabtype)
        # tbhdu = fits.new_table(columns)
        # tbhdu.header.set('EXTNAME', 'SPECTRAL INFO')
        # hdulist += [tbhdu]
        # if verbose:
        #     print '\tCreated an extension for spectral information.'

        hdulist += [fits.ImageHDU(self.f, name='FREQUENCIES')]
        hdulist += [fits.ImageHDU(self.lags, name='LAGS')]
        if verbose:
            print '\tCreated an extension for spectral information.'

        hdulist += [fits.ImageHDU(self.horizon_delay_limits, name='HORIZON LIMITS')]
        if verbose:
            print '\tCreated an extension for horizon delay limits of size {0[0]} x {0[1]} x {0[2]} as a function of snapshot instance, baseline, and (min,max) limits'.format(self.horizon_delay_limits.shape)

        hdulist += [fits.ImageHDU(self.bp_wts, name='BANDPASS')]
        if verbose:
            print '\tCreated an extension for bandpass functions of size {0[0]} x {0[1]} x {0[2]} as a function of baseline,  frequency, and snapshot instance'.format(self.bp.shape)

        hdulist += [fits.ImageHDU(self.bp_wts, name='BANDPASS WEIGHTS')]
        if verbose:
            print '\tCreated an extension for bandpass weights of size {0[0]} x {0[1]} x {0[2]} as a function of baseline,  frequency, and snapshot instance'.format(self.bp_wts.shape)

        if self.lag_kernel is not None:
            hdulist += [fits.ImageHDU(self.lag_kernel.real, name='LAG KERNEL REAL')]
            hdulist += [fits.ImageHDU(self.lag_kernel.imag, name='LAG KERNEL IMAG')]
            if verbose:
                print '\tCreated an extension for convolving lag kernel of size {0[0]} x {0[1]} x {0[2]} as a function of baseline, lags, and snapshot instance'.format(self.lag_kernel.shape)
        
        if self.skyvis_lag is not None:
            hdulist += [fits.ImageHDU(self.skyvis_lag.real, name='NOISELESS DELAY SPECTRA REAL')]
            hdulist += [fits.ImageHDU(self.skyvis_lag.imag, name='NOISELESS DELAY SPECTRA IMAG')]
        if self.vis_lag is not None:
            hdulist += [fits.ImageHDU(self.vis_lag.real, name='NOISY DELAY SPECTRA REAL')]
            hdulist += [fits.ImageHDU(self.vis_lag.imag, name='NOISY DELAY SPECTRA IMAG')]
        if self.vis_noise_lag is not None:
            hdulist += [fits.ImageHDU(self.vis_noise_lag.real, name='DELAY SPECTRA NOISE REAL')]
            hdulist += [fits.ImageHDU(self.vis_noise_lag.imag, name='DELAY SPECTRA NOISE IMAG')]
            
        if self.cc_freq is not None:
            hdulist += [fits.ImageHDU(self.cc_freq, name='CLEAN FREQUENCIES')]
        if self.cc_lags is not None:
            hdulist += [fits.ImageHDU(self.cc_lags, name='CLEAN LAGS')]
        if verbose:
            print '\tCreated an extension for spectral axes of clean components'

        if self.cc_lag_kernel is not None:
            hdulist += [fits.ImageHDU(self.cc_lag_kernel.real, name='CLEAN LAG KERNEL REAL')]
            hdulist += [fits.ImageHDU(self.cc_lag_kernel.imag, name='CLEAN LAG KERNEL IMAG')]
            if verbose:
                print '\tCreated an extension for deconvolving lag kernel of size {0[0]} x {0[1]} x {0[2]} as a function of baseline, lags, and snapshot instance'.format(self.cc_lag_kernel.shape)

        if self.cc_skyvis_lag is not None:
            hdulist += [fits.ImageHDU(self.cc_skyvis_lag.real, name='CLEAN NOISELESS DELAY SPECTRA REAL')]
            hdulist += [fits.ImageHDU(self.cc_skyvis_lag.imag, name='CLEAN NOISELESS DELAY SPECTRA IMAG')]

        if self.cc_skyvis_res_lag is not None:
            hdulist += [fits.ImageHDU(self.cc_skyvis_res_lag.real, name='CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL')]
            hdulist += [fits.ImageHDU(self.cc_skyvis_res_lag.imag, name='CLEAN NOISELESS DELAY SPECTRA RESIDUALS IMAG')]

        if self.cc_skyvis_freq is not None:
            hdulist += [fits.ImageHDU(self.cc_skyvis_freq.real, name='CLEAN NOISELESS VISIBILITIES REAL')]
            hdulist += [fits.ImageHDU(self.cc_skyvis_freq.imag, name='CLEAN NOISELESS VISIBILITIES IMAG')]

        if self.cc_skyvis_res_freq is not None:
            hdulist += [fits.ImageHDU(self.cc_skyvis_res_freq.real, name='CLEAN NOISELESS VISIBILITIES RESIDUALS REAL')]
            hdulist += [fits.ImageHDU(self.cc_skyvis_res_freq.imag, name='CLEAN NOISELESS VISIBILITIES RESIDUALS IMAG')]

        if self.cc_vis_lag is not None:
            hdulist += [fits.ImageHDU(self.cc_vis_lag.real, name='CLEAN NOISY DELAY SPECTRA REAL')]
            hdulist += [fits.ImageHDU(self.cc_vis_lag.imag, name='CLEAN NOISY DELAY SPECTRA IMAG')]

        if self.cc_vis_res_lag is not None:
            hdulist += [fits.ImageHDU(self.cc_vis_res_lag.real, name='CLEAN NOISY DELAY SPECTRA RESIDUALS REAL')]
            hdulist += [fits.ImageHDU(self.cc_vis_res_lag.imag, name='CLEAN NOISY DELAY SPECTRA RESIDUALS IMAG')]

        if self.cc_vis_freq is not None:
            hdulist += [fits.ImageHDU(self.cc_vis_freq.real, name='CLEAN NOISY VISIBILITIES REAL')]
            hdulist += [fits.ImageHDU(self.cc_vis_freq.imag, name='CLEAN NOISY VISIBILITIES IMAG')]

        if self.cc_vis_res_freq is not None:
            hdulist += [fits.ImageHDU(self.cc_vis_res_freq.real, name='CLEAN NOISY VISIBILITIES RESIDUALS REAL')]
            hdulist += [fits.ImageHDU(self.cc_vis_res_freq.imag, name='CLEAN NOISY VISIBILITIES RESIDUALS IMAG')]
        
        if verbose:
            print '\tCreated extensions for clean components of noiseless, noisy and residuals of visibilities in frequency and delay coordinates of size {0[0]} x {0[1]} x {0[2]} as a function of baselines, lags/frequency and snapshot instance'.format(self.lag_kernel.shape)

        if self.subband_delay_spectra:
            hdulist[0].header['SBDS'] = (1, 'Presence of Subband Delay Spectra')
            hdulist[0].header['SBDS-WSHAPE'] = (self.subband_delay_spectra['shape'], 'Shape of subband frequency weights')
            hdulist[0].header['SBDS-BPCORR'] = (int(self.subband_delay_spectra['bpcorrect']), 'Truth value for clean component subband delay spectrum bandpass windows weights correction')
            hdulist[0].header['SBDS-NPAD'] = (self.subband_delay_spectra['npad'], 'Number of zero-padded channels for subband delay spectra')
            hdulist[0].header['SBDS-DPOOL'] = (self.subband_delay_spectra['datapool'], 'Data pool for subband delay spectra')
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['freq_center'], name='SBDS-F0')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['freq_wts'], name='SBDS-FWTS')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['bw_eff'], name='SBDS-BWEFF')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['lags'], name='SBDS-LAGS')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['lag_kernel'].real, name='SBDS-LAGKERN-REAL')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['lag_kernel'].imag, name='SBDS-LAGKERN-IMAG')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['lag_corr_length'], name='SBDS-LAGCORR')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['skyvis_lag'].real, name='SBDS-SKYVISLAG-REAL')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['skyvis_lag'].imag, name='SBDS-SKYVISLAG-IMAG')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['vis_lag'].real, name='SBDS-VISLAG-REAL')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['vis_lag'].imag, name='SBDS-VISLAG-IMAG')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['vis_noise_lag'].real, name='SBDS-NOISELAG-REAL')]
            hdulist += [fits.ImageHDU(self.subband_delay_spectra['vis_noise_lag'].imag, name='SBDS-NOISELAG-IMAG')]
            if self.subband_delay_spectra['datapool'] == 'ccvis':
                hdulist += [fits.ImageHDU(self.subband_delay_spectra['skyvis_res_lag'].real, name='SBDS-CCSKYVISRESLAG-REAL')]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra['skyvis_res_lag'].imag, name='SBDS-CCSKYVISRESLAG-IMAG')]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra['vis_res_lag'].real, name='SBDS-CCVISRESLAG-REAL')]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra['vis_res_lag'].imag, name='SBDS-CCVISRESLAG-IMAG')]

            if verbose:
                print '\tCreated extensions for information on subband delay spectra for simulated and clean components of visibilities as a function of baselines, lags/frequency and snapshot instance'

        hdu = fits.HDUList(hdulist)
        hdu.writeto(outfile+'.cc.fits', clobber=overwrite)

################################################################################

class DelayPowerSpectrum(object):

    """
    ----------------------------------------------------------------------------
    Class to manage delay power spectrum from visibility measurements of a 
    multi-element interferometer array. 

    Attributes:

    cosmo       [instance of cosmology class from astropy] An instance of class
                FLRW or default_cosmology of astropy cosmology module. 

    ds          [instance of class DelaySpectrum] An instance of class 
                DelaySpectrum that contains the information on delay spectra of
                simulated visibilities

    f           [list or numpy vector] frequency channels in Hz

    lags        [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as channels. This is 
                computed in member function delay_transform().

    cc_lags     [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as cc_freq. This is computed in 
                member function delayClean().

    df          [scalar] Frequency resolution (in Hz)

    bl          [M x 3 Numpy array] The baseline vectors associated with the
                M interferometers in SI units 

    bl_length   [M-element numpy array] Lengths of the baseline in SI units
    
    f0          [scalar] Central frequency (in Hz)

    wl0         [scalar] Central wavelength (in m)

    z           [scalar] redshift

    bw          [scalar] (effective) bandwidth (in Hz)

    kprll       [numpy array] line-of-sight wavenumbers (in h/Mpc) corresponding
                to delays in the delay spectrum

    kperp       [numpy array] transverse wavenumbers (in h/Mpc) corresponding
                to baseline lengths

    drz_los     [scalar] comoving line-of-sight depth (Mpc/h) corresponding to 
                specified redshift and bandwidth for redshifted 21 cm line

    rz_transverse
                [scalar] comoving transverse distance (Mpc/h) corresponding to 
                specified redshift for redshifted 21 cm line

    rz_los      [scalar] comoving line-of-sight distance (Mpc/h) corresponding 
                to specified redshift for redshifted 21 cm line

    jacobian1   [scalar] first jacobian in conversion of delay spectrum to 
                power spectrum. It is equal to A_eff / wl**2 / bw

    jacobian2   [scalar] second jacobian in conversion of delay spectrum to 
                power spectrum. It is equal to rz_transverse**2 * drz_los / bw

    Jy2K        [scalar] factor to convert Jy/Sr to K. It is equal to 
                wl**2 * Jy / (2k)

    K2Jy        [scalar] factor to convert K to Jy/Sr. It is equal to 1/Jy2K

    dps         [dictionary of numpy arrays] contains numpy arrays containing
                delay power spectrum in units of K^2 (Mpc/h)^3 under the 
                following keys:
                'skyvis'    [numpy array] delay power spectrum of noiseless 
                            delay spectra
                'vis'       [numpy array] delay power spectrum of noisy delay
                            spectra
                'noise'     [numpy array] delay power spectrum of thermal noise
                            delay spectra
                'cc_skyvis' [numpy array] delay power spectrum of clean 
                            components of noiseless delay spectra
                'cc_vis'    [numpy array] delay power spectrum of clean 
                            components of noisy delay spectra
                'cc_skyvis_res' 
                            [numpy array] delay power spectrum of residuals 
                            after delay cleaning of noiseless delay spectra
                'cc_vis_res'
                            [numpy array] delay power spectrum of residuals  
                            after delay cleaning of noisy delay spectra

    Member functions:

    __init__()  Initialize an instance of class DelayPowerSpectrum

    comoving_los_depth() 
                Compute comoving line-of-sight depth (Mpc/h) corresponding to 
                specified redshift and bandwidth for redshifted 21 cm line

    comoving_transverse_distance() 
                Compute comoving transverse distance (Mpc/h) corresponding to 
                specified redshift for redshifted 21 cm line

    comoving_los_distance()
                Compute comoving line-of-sight distance (Mpc/h) corresponding 
                to specified redshift for redshifted 21 cm line

    k_parallel()
                Compute line-of-sight wavenumbers (h/Mpc) corresponding to 
                specified delays and redshift for redshifted 21 cm line

    k_perp()    Compute transverse wavenumbers (h/Mpc) corresponding to 
                specified baseline lengths and redshift for redshifted 21 cm 
                line assuming a mean wavelength (in m) for the relationship 
                between baseline lengths and spatial frequencies (u and v)

    compute_power_spectrum()
                Compute delay power spectrum in units of K^2 (Mpc/h)^3 from the 
                delay spectrum in units of Jy Hz
    ----------------------------------------------------------------------------
    """

    def __init__(self, dspec, cosmo=cosmo100):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class DelayPowerSpectrum. Attributes 
        initialized are: ds, cosmo, f, df, f0, z, bw, drz_los, rz_transverse,
        rz_los, kprll, kperp, jacobian1, jacobian2

        Inputs:

        dspec    [instance of class DelaySpectrum] An instance of class 
                 DelaySpectrum that contains the information on delay spectra of
                 simulated visibilities

        cosmo    [instance of a cosmology class in Astropy] An instance of class
                 FLRW or default_cosmology of astropy cosmology module. Default
                 value is set using concurrent cosmology but keep 
                 H0=100 km/s/Mpc
        ------------------------------------------------------------------------
        """
        
        try:
            dspec
        except NameError:
            raise NameError('No delay spectrum instance supplied for initialization')

        if not isinstance(dspec, DelaySpectrum):
            raise TypeError('Input dspec must be an instance of class DelaySpectrum')

        if not isinstance(cosmo, (CP.FLRW, CP.default_cosmology)):
            raise TypeError('Input cosmology must be a cosmology class defined in Astropy')

        self.cosmo = cosmo
        self.ds = dspec
        self.f = self.ds.f
        self.lags = self.ds.lags
        self.cc_lags = self.ds.cc_lags
        self.bl = self.ds.ia.baselines
        self.bl_length = self.ds.ia.baseline_lengths
        self.df = self.ds.df
        self.f0 = self.f[self.f.size/2]
        self.wl0 = FCNST.c / self.f0
        self.z = CNST.rest_freq_HI / self.f0 - 1
        self.bw = self.df * self.f.size
        self.kprll = self.k_parallel(self.lags, redshift=self.z, action='return')   # in h/Mpc
        self.kperp = self.k_perp(self.bl_length, redshift=self.z, action='return')   # in h/Mpc        

        self.drz_los = self.comoving_los_depth(self.bw, self.z, action='return')   # in Mpc/h
        self.rz_transverse = self.comoving_transverse_distance(self.z, action='return')   # in Mpc/h
        self.rz_los = self.comoving_los_distance(self.z, action='return')   # in Mpc/h

        self.jacobian1 = NP.mean(self.ds.ia.A_eff) / self.wl0**2 / self.bw
        self.jacobian2 = self.rz_transverse**2 * self.drz_los / self.bw
        self.Jy2K = self.wl0**2 * CNST.Jy / (2*FCNST.k)
        self.K2Jy = 1 / self.Jy2K

        self.dps = {}
        self.dps['skyvis'] = None
        self.dps['vis'] = None
        self.dps['noise'] = None
        self.dps['cc_skyvis'] = None
        self.dps['cc_vis'] = None
        self.dps['cc_skyvis_res'] = None
        self.dps['cc_vis_res'] = None        

    ############################################################################

    def comoving_los_depth(self, bw, redshift, action=None):

        """
        ------------------------------------------------------------------------
        Compute comoving line-of-sight depth (Mpc/h) corresponding to specified 
        redshift and bandwidth for redshifted 21 cm line

        Inputs:

        bw        [scalar] bandwidth in Hz

        redshift  [scalar] redshift

        action    [string] If set to None (default), the comoving depth 
                  along the line of sight (Mpc/h) and specified reshift are 
                  stored internally as attributes of the instance of class
                  DelayPowerSpectrum. If set to 'return', the comoving depth
                  along line of sight (Mpc/h) computed is returned

        Outputs:

        If keyword input action is set to 'return', the comoving depth along 
        line of sight (Mpc/h) computed is returned
        ------------------------------------------------------------------------
        """

        drz_los = (FCNST.c/1e3) * bw * (1+redshift)**2 / CNST.rest_freq_HI / self.cosmo.H0.value / self.cosmo.efunc(redshift)   # in Mpc/h
        if action is None:
            self.z = redshift
            self.drz_los = drz_los
            return
        else:
            return drz_los

    ############################################################################

    def comoving_transverse_distance(self, redshift, action=None):

        """
        ------------------------------------------------------------------------
        Compute comoving transverse distance (Mpc/h) corresponding to specified 
        redshift for redshifted 21 cm line

        Inputs:

        redshift  [scalar] redshift

        action    [string] If set to None (default), the comoving 
                  transverse distance (Mpc/h) and specified reshift are stored 
                  internally as attributes of the instance of class
                  DelayPowerSpectrum. If set to 'return', the comoving 
                  transverse distance (Mpc/h) computed is returned

        Outputs:

        If keyword input action is set to 'return', the comoving transverse 
        distance (Mpc/h) computed is returned
        ------------------------------------------------------------------------
        """

        rz_transverse = self.cosmo.comoving_transverse_distance(redshift).value   # in Mpc/h
        if action is None:
            self.z = redshift
            self.rz_transverse = rz_transverse
            return
        else:
            return rz_transverse

    ############################################################################

    def comoving_los_distance(self, redshift, action=None):

        """
        ------------------------------------------------------------------------
        Compute comoving line-of-sight distance (Mpc/h) corresponding to 
        specified redshift for redshifted 21 cm line

        Inputs:

        redshift  [scalar] redshift

        action    [string] If set to None (default), the comoving 
                  line-of-sight distance (Mpc/h) and specified reshift are 
                  stored internally as attributes of the instance of class
                  DelayPowerSpectrum. If set to 'return', the comoving 
                  line-of-sight distance (Mpc/h) computed is returned

        Outputs:

        If keyword input action is set to 'return', the comoving line-of-sight 
        distance (Mpc/h) computed is returned
        ------------------------------------------------------------------------
        """

        rz_los = self.cosmo.comoving_distance(redshift).value   # in Mpc/h
        if action is None:
            self.z = redshift
            self.rz_los = rz_los
            return
        else:
            return rz_los
        
    ############################################################################

    def k_parallel(self, lags, redshift, action=None):

        """
        ------------------------------------------------------------------------
        Compute line-of-sight wavenumbers (h/Mpc) corresponding to specified 
        delays and redshift for redshifted 21 cm line

        Inputs:

        lags      [numpy array] geometric delays (in seconds) obtained as 
                  Fourier conjugate variable of frequencies in the bandpass

        redshift  [scalar] redshift

        action    [string] If set to None (default), the line-of-sight 
                  wavenumbers (h/Mpc) and specified reshift are 
                  stored internally as attributes of the instance of class
                  DelayPowerSpectrum. If set to 'return', the line-of-sight 
                  wavenumbers (h/Mpc) computed is returned

        Outputs:

        If keyword input action is set to 'return', the line-of-sight 
        wavenumbers (h/Mpc) computed is returned
        ------------------------------------------------------------------------
        """

        kprll = 2 * NP.pi * lags * self.cosmo.H0.value * CNST.rest_freq_HI * self.cosmo.efunc(redshift) / FCNST.c / (1+redshift)**2 * 1e3
        if action is None:
            self.z = redshift
            self.kprll = kprll
            return
        else:
            return kprll

    ############################################################################

    def k_perp(self, baseline_length, redshift, action=None):

        """
        ------------------------------------------------------------------------
        Compute transverse wavenumbers (h/Mpc) corresponding to specified 
        baseline lengths and redshift for redshifted 21 cm line assuming a
        mean wavelength (in m) for the relationship between baseline lengths and
        spatial frequencies (u and v)

        Inputs:

        baseline_length      
                  [numpy array] baseline lengths (in m) 

        redshift  [scalar] redshift

        action    [string] If set to None (default), the transverse 
                  wavenumbers (h/Mpc) and specified reshift are stored 
                  internally as attributes of the instance of class
                  DelayPowerSpectrum. If set to 'return', the transverse 
                  wavenumbers (h/Mpc) computed is returned

        Outputs:

        If keyword input action is set to 'return', the transverse 
        wavenumbers (h/Mpc) computed is returned
        ------------------------------------------------------------------------
        """

        kperp = 2 * NP.pi * (baseline_length/self.wl0) / self.comoving_transverse_distance(redshift, action='return')
        if action is None:
            self.z = redshift
            self.kperp = kperp
            return
        else:
            return kperp
        
    ############################################################################

    def compute_power_spectrum(self):

        """
        ------------------------------------------------------------------------
        Compute delay power spectrum in units of K^2 (Mpc/h)^3 from the delay
        spectrum in units of Jy Hz. 
        ------------------------------------------------------------------------
        """

        self.dps = {}
        factor = self.jacobian1 * self.jacobian2 * self.Jy2K**2
        if self.ds.skyvis_lag is not None: self.dps['skyvis'] = self.ds.skyvis_lag**2 * factor
        if self.ds.vis_lag is not None: self.dps['vis'] = self.ds.vis_lag**2 * factor
        if self.ds.vis_noise_lag is not None: self.dps['noise'] = self.ds.vis_noise_lag**2 * factor
        if self.ds.cc_skyvis_lag is not None: self.dps['cc_skyvis'] = self.ds.cc_skyvis_lag**2 * factor
        if self.ds.cc_vis_lag is not None: self.dps['cc_vis'] = self.ds.cc_vis_lag**2 * factor
        if self.ds.cc_skyvis_res_lag is not None: self.dps['cc_skyvis_res'] = self.ds.cc_skyvis_res_lag**2 * factor
        if self.ds.cc_vis_res_lag is not None: self.dps['cc_vis_res'] = self.ds.cc_vis_res_lag**2 * factor

    ############################################################################

