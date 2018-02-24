from __future__ import division
import numpy as NP
import multiprocessing as MP
import itertools as IT
import progressbar as PGB
# import aipy as AP
import astropy

from astropy.io import fits
import astropy.cosmology as CP
import scipy.constants as FCNST
import healpy as HP
from distutils.version import LooseVersion
import yaml
from astroutils import writer_module as WM
from astroutils import constants as CNST
from astroutils import DSP_modules as DSP
from astroutils import mathops as OPS
from astroutils import geometry as GEOM
from astroutils import lookup_operations as LKP
import prisim
from prisim import primary_beams as PB
from prisim import interferometry as RI
from prisim import baseline_delay_horizon as DLY

prisim_path = prisim.__path__[0]+'/'

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

# def _gentle_clean(dd, _w, tol=1e-1, area=None, stop_if_div=True, maxiter=100,
#                   verbose=False, autoscale=True):

#     if verbose:
#         print "Performing gentle clean..."

#     scale_factor = 1.0
#     if autoscale:
#         scale_factor = NP.nanmax(NP.abs(_w))
#     dd /= scale_factor
#     _w /= scale_factor

#     cc, info = AP.deconv.clean(dd, _w, tol=tol, area=area, stop_if_div=False,
#                                maxiter=maxiter, verbose=verbose)
#     #dd = info['res']

#     cc = NP.zeros_like(dd)
#     inside_res = NP.std(dd[area!=0])
#     outside_res = NP.std(dd[area==0])
#     initial_res = inside_res
#     #print inside_res,'->',
#     ncycle=0
#     if verbose:
#         print "inside_res outside_res"
#         print inside_res, outside_res
#     inside_res = 2*outside_res #just artifically bump up the inside res so the loop runs at least once
#     while(inside_res>outside_res and maxiter>0):
#         if verbose: print '.',
#         _d_cl, info = AP.deconv.clean(dd, _w, tol=tol, area=area, stop_if_div=stop_if_div, maxiter=maxiter, verbose=verbose, pos_def=True)
#         res = info['res']
#         inside_res = NP.std(res[area!=0])
#         outside_res = NP.std(res[area==0])
#         dd = info['res']
#         cc += _d_cl
#         ncycle += 1
#         if verbose: print inside_res*scale_factor, outside_res*scale_factor
#         if ncycle>1000: break

#     info['ncycle'] = ncycle-1

#     dd *= scale_factor
#     _w *= scale_factor
#     cc *= scale_factor
#     info['initial_residual'] = initial_res * scale_factor
#     info['final_residual'] = inside_res * scale_factor
    
#     return cc, info

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

################################################################################

def dkprll_deta(redshift, cosmo=cosmo100):

    """
    ----------------------------------------------------------------------------
    Compute jacobian to transform delays (eta or tau) to line-of-sight 
    wavenumbers (h/Mpc) corresponding to specified redshift(s) and cosmology
    corresponding to the HI 21 cm line

    Inputs:

    redshift  [scalar, list or numpy array] redshift(s). Must be a 
              scalar, list or numpy array

    cosmo     [instance of cosmology class from astropy] An instance of class
              FLRW or default_cosmology of astropy cosmology module. Default
              uses Flat lambda CDM cosmology with Omega_m=0.27, 
              H0=100 km/s/Mpc

    Outputs:

    Jacobian to convert eta (lags) to k_parallel. Same size as redshift
    ----------------------------------------------------------------------------
    """

    if not isinstance(redshift, (int, float, list, NP.ndarray)):
        raise TypeError('redshift must be a scalar, list or numpy array')
    redshift = NP.asarray(redshift)
    if NP.any(redshift < 0.0):
        raise ValueError('redshift(s) must be non-negative')

    if not isinstance(cosmo, (CP.FLRW, CP.default_cosmology)):
        raise TypeError('Input cosmology must be a cosmology class defined in Astropy')
    
    jacobian = 2 * NP.pi * cosmo.H0.value * CNST.rest_freq_HI * cosmo.efunc(redshift) / FCNST.c / (1+redshift)**2 * 1e3

    return jacobian

################################################################################

def beam3Dvol(beam, freqs, freq_wts=None, hemisphere=True):

    """
    ----------------------------------------------------------------------------
    Compute 3D volume relevant for power spectrum given an antenna power 
    pattern. It is estimated by summing square of the beam in angular and 
    frequency coordinates and in units of "Sr Hz".

    Inputs:

    beam        [numpy array] Antenna power pattern with peak normalized to 
                unity. It can be of shape (npix x nchan) or (npix x 1) or 
                (npix,). npix must be a HEALPix compatible value. nchan is the
                number of frequency channels, same as the size of input freqs.
                If it is of shape (npix x 1) or (npix,), the beam will be 
                assumed to be identical for all frequency channels.

    freqs       [list or numpy array] Frequency channels (in Hz) of size nchan

    freq_wts    [numpy array] Frequency weights to be applied to the
                beam. Must be of shape (nchan,) or (nwin, nchan)

    Keyword Inputs:

    hemisphere  [boolean] If set to True (default), the 3D volume will be 
                estimated using the upper hemisphere. If False, the full sphere
                is used.

    Output:

    The product Omega x bandwdith (in Sr Hz) computed using the integral of 
    squared power pattern. It is of shape (nwin,)
    ----------------------------------------------------------------------------
    """

    try:
        beam, freqs
    except NameError:
        raise NameError('Both inputs beam and freqs must be specified')

    if not isinstance(beam, NP.ndarray):
        raise TypeError('Input beam must be a numpy array')

    if not isinstance(freqs, (list, NP.ndarray)):
        raise TypeError('Input freqs must be a list or numpy array')
    freqs = NP.asarray(freqs).astype(NP.float).reshape(-1)
    if freqs.size < 2:
        raise ValueError('Input freqs does not have enough elements to determine frequency resolution')

    if beam.ndim > 2:
        raise ValueError('Invalid dimensions for beam')
    elif beam.ndim == 2:
        if beam.shape[1] != 1:
            if beam.shape[1] != freqs.size:
                raise ValueError('Dimensions of beam do not match the number of frequency channels')
    elif beam.ndim == 1:
        beam = beam.reshape(-1,1)
    else:
        raise ValueError('Invalid dimensions for beam')

    if freq_wts is not None:
        if not isinstance(freq_wts, NP.ndarray):
            raise TypeError('Input freq_wts must be a numpy array')
        if freq_wts.ndim == 1:
            freq_wts = freq_wts.reshape(1,-1)
        elif freq_wts.ndim > 2:
            raise ValueError('Input freq_wts must be of shape nwin x nchan')

        freq_wts = NP.asarray(freq_wts).astype(NP.float).reshape(-1,freqs.size)
        if freq_wts.shape[1] != freqs.size:
            raise ValueError('Input freq_wts does not have shape compatible with freqs')
    else:
        freq_wts = NP.ones(freqs.size, dtype=NP.float).reshape(1,-1)

    eps = 1e-10
    if beam.max() > 1.0+eps:
        raise ValueError('Input beam maximum exceeds unity. Input beam should be normalized to peak of unity')

    nside = HP.npix2nside(beam.shape[0])
    domega = HP.nside2pixarea(nside, degrees=False)
    df = freqs[1] - freqs[0]
    bw = df * freqs.size
    weighted_beam = beam[:,NP.newaxis,:] * freq_wts[NP.newaxis,:,:]

    theta, phi = HP.pix2ang(nside, NP.arange(beam.shape[0]))
    if hemisphere:
        ind, = NP.where(theta <= NP.pi/2)  # Select upper hemisphere
    else:
        ind = NP.arange(beam.shape[0])

    omega_bw = domega * df * NP.nansum(weighted_beam[ind,:,:]**2, axis=(0,2))
    if NP.any(omega_bw > 4*NP.pi*bw):
        raise ValueError('3D volume estimated from beam exceeds the upper limit. Check normalization of the input beam')

    return omega_bw

################################################################################

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
                            'npad'. It roughly corresponds to k_parallel.
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
                            after delay CLEAN of simulated noiseless 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x (nchan+npad) x n_t
                'vis_res_lag'
                            [numpy array] subband delay spectra of residuals
                            after delay CLEAN of simulated noisy 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x (nchan+npad) x n_t
                'skyvis_net_lag'
                            [numpy array] subband delay spectra of sum of 
                            residuals and clean components
                            after delay CLEAN of simulated noiseless 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x (nchan+npad) x n_t
                'vis_res_lag'
                            [numpy array] subband delay spectra of sum of 
                            residuals and clean components
                            after delay CLEAN of simulated noisy 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x (nchan+npad) x n_t

    subband_delay_spectra_resampled
                [dictionary] Very similar to the attribute 
                subband_delay_spectra except now it has been resampled along 
                delay axis to contain usually only independent delay bins. It 
                contains two top level keys, namely, 'cc' and 'sim' 
                denoting information about CLEAN and simulated visibilities 
                respectively. Under each of these keys is information about delay 
                spectra of different frequency sub-bands (n_win in number) after 
                resampling to independent number of delay bins in the 
                form of a dictionary under the following keys:
                'freq_center' 
                            [numpy array] contains the center frequencies 
                            (in Hz) of the frequency subbands of the subband
                            delay spectra. It is of size n_win. It is roughly 
                            equivalent to redshift(s)
                'bw_eff'    [numpy array] contains the effective bandwidths 
                            (in Hz) of the subbands being delay transformed. It
                            is of size n_win. It is roughly equivalent to width 
                            in redshift or along line-of-sight
                'lags'      [numpy array] lags of the subband delay spectra 
                            after padding in frequency during the transform. It
                            is of size nlags where nlags is the number of 
                            independent delay bins. It roughly corresponds to 
                            k_parallel.
                'lag_kernel'
                            [numpy array] delay transform of the frequency 
                            weights under the key 'freq_wts'. It is of size
                            n_bl x n_win x nlags x n_t.
                'lag_corr_length' 
                            [numpy array] It is the correlation timescale (in 
                            pixels) of the resampled subband delay spectra. It is 
                            proportional to inverse of effective bandwidth. It
                            is of size n_win. The unit size of a pixel is 
                            determined by the difference between adjacent pixels 
                            in lags under key 'lags' which in turn is 
                            usually approximately inverse of the effective
                            bandwidth of the subband
                'skyvis_lag'
                            [numpy array] subband delay spectra of simulated 
                            or CLEANed noiseless visibilities, depending on 
                            whether the top level key is 'cc' or 'sim' 
                            respectively, after applying the frequency weights 
                            under the key 'freq_wts'. It is of size 
                            n_bl x n_win x nlags x n_t. 
                'vis_lag'   [numpy array] subband delay spectra of simulated 
                            or CLEANed noisy visibilities, depending on whether
                            the top level key is 'cc' or 'sim' respectively,
                            after applying the frequency weights under the key 
                            'freq_wts'. It is of size 
                            n_bl x n_win x nlags x n_t. 
                'vis_noise_lag'   
                            [numpy array] subband delay spectra of simulated 
                            noise after applying the frequency weights under 
                            the key 'freq_wts'. Only present if top level key is 
                            'sim' and absent for 'cc'. It is of size 
                            n_bl x n_win x nlags x n_t. 
                'skyvis_res_lag'
                            [numpy array] subband delay spectra of residuals
                            after delay CLEAN of simulated noiseless 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x nlags x n_t
                'vis_res_lag'
                            [numpy array] subband delay spectra of residuals
                            after delay CLEAN of simulated noisy 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x nlags x n_t
                'skyvis_net_lag'
                            [numpy array] subband delay spectra of sum of 
                            residuals and clean components
                            after delay CLEAN of simulated noiseless 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x nlags x n_t
                'vis_res_lag'
                            [numpy array] subband delay spectra of sum of 
                            residuals and clean components
                            after delay CLEAN of simulated noisy 
                            visibilities obtained after applying frequency 
                            weights specified under key 'freq_wts'. Only present 
                            for top level key 'cc' and absent for 'sim'. It is of
                            size n_bl x n_win x nlags x n_t

    Member functions:

    __init__()  Initializes an instance of class DelaySpectrum
                        
    delay_transform()  
                Transforms the visibilities from frequency axis onto 
                delay (time) axis using an IFFT. This is performed for 
                noiseless sky visibilities, thermal noise in visibilities, 
                and observed visibilities. 

    delay_transform_allruns()        
                Transforms the visibilities of multiple runs from frequency 
                axis onto delay (time) axis using an IFFT. 

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

    subband_delay_transform_allruns()
                Computes delay transform on multiple frequency sub-bands with 
                specified weights for multiple realizations of visibilities

    subband_delay_transform_closure_phase()
                Computes delay transform of closure phases on antenna triplets 
                on multiple frequency sub-bands with specified weights

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
        cc_lag_kernel, subband_delay_spectra, subband_delay_spectra_resampled

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
                raise KeyError('Extension "FREQUENCIES" not found in header')

            self.lags = None
            if 'LAGS' in extnames:
                self.lags = hdulist['LAGS'].data

            self.cc_lags = None
            if 'CLEAN LAGS' in extnames:
                self.cc_lags = hdulist['CLEAN LAGS'].data

            self.cc_freq = None
            if 'CLEAN FREQUENCIES' in extnames:
                self.cc_freq = hdulist['CLEAN FREQUENCIES'].data
                
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
            self.subband_delay_spectra_resampled = {}
            if 'SBDS' in hdulist[0].header:
                for key in ['cc', 'sim']:
                    if '{0}-SBDS'.format(key) in hdulist[0].header:
                        self.subband_delay_spectra[key] = {}
                        self.subband_delay_spectra[key]['shape'] = hdulist[0].header['{0}-SBDS-WSHAPE'.format(key)]
                        if key == 'cc':
                            self.subband_delay_spectra[key]['bpcorrect'] = bool(hdulist[0].header['{0}-SBDS-BPCORR'.format(key)])
                        self.subband_delay_spectra[key]['npad'] = hdulist[0].header['{0}-SBDS-NPAD'.format(key)]
                        self.subband_delay_spectra[key]['freq_center'] = hdulist['{0}-SBDS-F0'.format(key)].data
                        self.subband_delay_spectra[key]['freq_wts'] = hdulist['{0}-SBDS-FWTS'.format(key)].data
                        self.subband_delay_spectra[key]['bw_eff'] = hdulist['{0}-SBDS-BWEFF'.format(key)].data
                        self.subband_delay_spectra[key]['lags'] = hdulist['{0}-SBDS-LAGS'.format(key)].data
                        self.subband_delay_spectra[key]['lag_kernel'] = hdulist['{0}-SBDS-LAGKERN-REAL'.format(key)].data + 1j * hdulist['{0}-SBDS-LAGKERN-IMAG'.format(key)].data
                        self.subband_delay_spectra[key]['lag_corr_length'] = hdulist['{0}-SBDS-LAGCORR'.format(key)].data
                        self.subband_delay_spectra[key]['skyvis_lag'] = hdulist['{0}-SBDS-SKYVISLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDS-SKYVISLAG-IMAG'.format(key)].data
                        self.subband_delay_spectra[key]['vis_lag'] = hdulist['{0}-SBDS-VISLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDS-VISLAG-IMAG'.format(key)].data
                        if key == 'sim':
                            self.subband_delay_spectra[key]['vis_noise_lag'] = hdulist['{0}-SBDS-NOISELAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDS-NOISELAG-IMAG'.format(key)].data
                        if key == 'cc':
                            self.subband_delay_spectra[key]['skyvis_res_lag'] = hdulist['{0}-SBDS-SKYVISRESLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDS-SKYVISRESLAG-IMAG'.format(key)].data
                            self.subband_delay_spectra[key]['vis_res_lag'] = hdulist['{0}-SBDS-VISRESLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDS-VISRESLAG-IMAG'.format(key)].data
                            self.subband_delay_spectra[key]['skyvis_net_lag'] = self.subband_delay_spectra[key]['skyvis_lag'] + self.subband_delay_spectra[key]['skyvis_res_lag']
                            self.subband_delay_spectra[key]['vis_net_lag'] = self.subband_delay_spectra[key]['vis_lag'] + self.subband_delay_spectra[key]['vis_res_lag']

            if 'SBDS-RS' in hdulist[0].header:
                for key in ['cc', 'sim']:
                    if '{0}-SBDS-RS'.format(key) in hdulist[0].header:
                        self.subband_delay_spectra_resampled[key] = {}
                        self.subband_delay_spectra_resampled[key]['freq_center'] = hdulist['{0}-SBDSRS-F0'.format(key)].data
                        self.subband_delay_spectra_resampled[key]['bw_eff'] = hdulist['{0}-SBDSRS-BWEFF'.format(key)].data
                        self.subband_delay_spectra_resampled[key]['lags'] = hdulist['{0}-SBDSRS-LAGS'.format(key)].data
                        self.subband_delay_spectra_resampled[key]['lag_kernel'] = hdulist['{0}-SBDSRS-LAGKERN-REAL'.format(key)].data + 1j * hdulist['{0}-SBDSRS-LAGKERN-IMAG'.format(key)].data
                        self.subband_delay_spectra_resampled[key]['lag_corr_length'] = hdulist['{0}-SBDSRS-LAGCORR'.format(key)].data
                        self.subband_delay_spectra_resampled[key]['skyvis_lag'] = hdulist['{0}-SBDSRS-SKYVISLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDSRS-SKYVISLAG-IMAG'.format(key)].data
                        self.subband_delay_spectra_resampled[key]['vis_lag'] = hdulist['{0}-SBDSRS-VISLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDSRS-VISLAG-IMAG'.format(key)].data
                        if key == 'sim':
                            self.subband_delay_spectra_resampled[key]['vis_noise_lag'] = hdulist['{0}-SBDSRS-NOISELAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDSRS-NOISELAG-IMAG'.format(key)].data
                        if key == 'cc':
                            self.subband_delay_spectra_resampled[key]['skyvis_res_lag'] = hdulist['{0}-SBDSRS-SKYVISRESLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDSRS-SKYVISRESLAG-IMAG'.format(key)].data
                            self.subband_delay_spectra_resampled[key]['vis_res_lag'] = hdulist['{0}-SBDSRS-VISRESLAG-REAL'.format(key)].data + 1j * hdulist['{0}-SBDSRS-VISRESLAG-IMAG'.format(key)].data
                            self.subband_delay_spectra_resampled[key]['skyvis_net_lag'] = self.subband_delay_spectra_resampled[key]['skyvis_lag'] + self.subband_delay_spectra_resampled[key]['skyvis_res_lag']
                            self.subband_delay_spectra_resampled[key]['vis_net_lag'] = self.subband_delay_spectra_resampled[key]['vis_lag'] + self.subband_delay_spectra_resampled[key]['vis_res_lag']
                            
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
        self.subband_delay_spectra_resampled = {}

    #############################################################################

    def delay_transform(self, pad=1.0, freq_wts=None, downsample=True,
                        action=None, verbose=True):

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

        action      [boolean] If set to None (default), just return the delay-
                    transformed quantities. If set to 'store', these quantities
                    will be stored as internal attributes

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
        else:
            freq_wts = self.bp_wts
        if verbose:
            print '\tFrequency window weights assigned.'

        if not isinstance(downsample, bool):
            raise TypeError('Input downsample must be of boolean type')

        if verbose:
            print '\tInput parameters have been verified to be compatible.\n\tProceeding to compute delay transform.'
            
        result = {}
        result['freq_wts'] = freq_wts
        result['pad'] = pad
        result['lags'] = DSP.spectral_axis(int(self.f.size*(1+pad)), delx=self.df, use_real=False, shift=True)
        if pad == 0.0:
            result['vis_lag'] = DSP.FT1D(self.ia.vis_freq * self.bp * freq_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            result['skyvis_lag'] = DSP.FT1D(self.ia.skyvis_freq * self.bp * freq_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            result['vis_noise_lag'] = DSP.FT1D(self.ia.vis_noise_freq * self.bp * freq_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            result['lag_kernel'] = DSP.FT1D(self.bp * freq_wts, ax=1, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            if verbose:
                print '\tDelay transform computed without padding.'
        else:
            npad = int(self.f.size * pad)
            result['vis_lag'] = DSP.FT1D(NP.pad(self.ia.vis_freq * self.bp * freq_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            result['skyvis_lag'] = DSP.FT1D(NP.pad(self.ia.skyvis_freq * self.bp * freq_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            result['vis_noise_lag'] = DSP.FT1D(NP.pad(self.ia.vis_noise_freq * self.bp * freq_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            result['lag_kernel'] = DSP.FT1D(NP.pad(self.bp * freq_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            if verbose:
                print '\tDelay transform computed with padding fraction {0:.1f}'.format(pad)

        if downsample:
            result['vis_lag'] = DSP.downsampler(result['vis_lag'], 1+pad, axis=1)
            result['skyvis_lag'] = DSP.downsampler(result['skyvis_lag'], 1+pad, axis=1)
            result['vis_noise_lag'] = DSP.downsampler(result['vis_noise_lag'], 1+pad, axis=1)
            result['lag_kernel'] = DSP.downsampler(result['lag_kernel'], 1+pad, axis=1)
            result['lags'] = DSP.downsampler(result['lags'], 1+pad)
            result['lags'] = result['lags'].flatten()
            if verbose:
                print '\tDelay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'delay_transform() completed successfully.'

        if action == 'store':
            self.pad = pad
            self.lags = result['lags']
            self.bp_wts = freq_wts
            self.vis_lag = result['vis_lag']
            self.skyvis_lag = result['skyvis_lag']
            self.vis_noise_lag = result['vis_noise_lag']
            self.lag_kernel = result['lag_kernel']

        return result

    #############################################################################
        
    # def clean(self, pad=1.0, freq_wts=None, clean_window_buffer=1.0,
    #           verbose=True):

    #     """
    #     ------------------------------------------------------------------------
    #     TO BE DEPRECATED!!! USE MEMBER FUNCTION delayClean()

    #     Transforms the visibilities from frequency axis onto delay (time) axis
    #     using an IFFT and deconvolves the delay transform quantities along the 
    #     delay axis. This is performed for noiseless sky visibilities, thermal
    #     noise in visibilities, and observed visibilities. 

    #     Inputs:

    #     pad         [scalar] Non-negative scalar indicating padding fraction 
    #                 relative to the number of frequency channels. For e.g., a 
    #                 pad of 1.0 pads the frequency axis with zeros of the same 
    #                 width as the number of channels. If a negative value is 
    #                 specified, delay transform will be performed with no padding

    #     freq_wts    [numpy vector or array] window shaping to be applied before
    #                 computing delay transform. It can either be a vector or size
    #                 equal to the number of channels (which will be applied to all
    #                 time instances for all baselines), or a nchan x n_snapshots 
    #                 numpy array which will be applied to all baselines, or a 
    #                 n_baselines x nchan numpy array which will be applied to all 
    #                 timestamps, or a n_baselines x nchan x n_snapshots numpy 
    #                 array. Default (None) will not apply windowing and only the
    #                 inherent bandpass will be used.

    #     verbose     [boolean] If set to True (default), print diagnostic and 
    #                 progress messages. If set to False, no such messages are
    #                 printed.
    #     ------------------------------------------------------------------------
    #     """

    #     if not isinstance(pad, (int, float)):
    #         raise TypeError('pad fraction must be a scalar value.')
    #     if pad < 0.0:
    #         pad = 0.0
    #         if verbose:
    #             print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'
    
    #     if freq_wts is not None:
    #         if freq_wts.size == self.f.size:
    #             freq_wts = NP.repeat(NP.expand_dims(NP.repeat(freq_wts.reshape(1,-1), self.ia.baselines.shape[0], axis=0), axis=2), self.n_acc, axis=2)
    #         elif freq_wts.size == self.f.size * self.n_acc:
    #             freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(self.f.size, -1), axis=0), self.ia.baselines.shape[0], axis=0)
    #         elif freq_wts.size == self.f.size * self.ia.baselines.shape[0]:
    #             freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(-1, self.f.size), axis=2), self.n_acc, axis=2)
    #         elif freq_wts.size == self.f.size * self.ia.baselines.shape[0] * self.n_acc:
    #             freq_wts = freq_wts.reshape(self.ia.baselines.shape[0], self.f.size, self.n_acc)
    #         else:
    #             raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
    #         self.bp_wts = freq_wts
    #         if verbose:
    #             print '\tFrequency window weights assigned.'

    #     bw = self.df * self.f.size
    #     pc = self.ia.phase_center
    #     pc_coords = self.ia.phase_center_coords
    #     if pc_coords == 'hadec':
    #         pc_altaz = GEOM.hadec2altaz(pc, self.ia.latitude, units='degrees')
    #         pc_dircos = GEOM.altaz2dircos(pc_altaz, units='degrees')
    #     elif pc_coords == 'altaz':
    #         pc_dircos = GEOM.altaz2dircos(pc, units='degrees')
        
    #     npad = int(self.f.size * pad)
    #     lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=False)
    #     dlag = lags[1] - lags[0]
    
    #     clean_area = NP.zeros(self.f.size + npad, dtype=int)
    #     skyvis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.skyvis_freq*self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
    #     vis_lag = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.ia.vis_freq*self.bp*self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
    #     lag_kernel = (npad + self.f.size) * self.df * DSP.FT1D(NP.pad(self.bp, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=False)
        
    #     ccomponents_noiseless = NP.zeros_like(skyvis_lag)
    #     ccres_noiseless = NP.zeros_like(skyvis_lag)
    
    #     ccomponents_noisy = NP.zeros_like(vis_lag)
    #     ccres_noisy = NP.zeros_like(vis_lag)
        
    #     for snap_iter in xrange(self.n_acc):
    #         progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Baselines '.format(self.ia.baselines.shape[0]), PGB.ETA()], maxval=self.ia.baselines.shape[0]).start()
    #         for bl_iter in xrange(self.ia.baselines.shape[0]):
    #             clean_area[NP.logical_and(lags <= self.horizon_delay_limits[snap_iter,bl_iter,1]+clean_window_buffer/bw, lags >= self.horizon_delay_limits[snap_iter,bl_iter,0]-clean_window_buffer/bw)] = 1
    
    #             cc_noiseless, info_noiseless = _gentle_clean(skyvis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=False, autoscale=True)
    #             ccomponents_noiseless[bl_iter,:,snap_iter] = cc_noiseless
    #             ccres_noiseless[bl_iter,:,snap_iter] = info_noiseless['res']
    
    #             cc_noisy, info_noisy = _gentle_clean(vis_lag[bl_iter,:,snap_iter], lag_kernel[bl_iter,:,snap_iter], area=clean_area, stop_if_div=False, verbose=False, autoscale=True)
    #             ccomponents_noisy[bl_iter,:,snap_iter] = cc_noisy
    #             ccres_noisy[bl_iter,:,snap_iter] = info_noisy['res']
    
    #             progress.update(bl_iter+1)
    #         progress.finish()
    
    #     deta = lags[1] - lags[0]
    #     cc_skyvis = NP.fft.fft(ccomponents_noiseless, axis=1) * deta
    #     cc_skyvis_res = NP.fft.fft(ccres_noiseless, axis=1) * deta
    
    #     cc_vis = NP.fft.fft(ccomponents_noisy, axis=1) * deta
    #     cc_vis_res = NP.fft.fft(ccres_noisy, axis=1) * deta
    
    #     self.skyvis_lag = NP.fft.fftshift(skyvis_lag, axes=1)
    #     self.vis_lag = NP.fft.fftshift(vis_lag, axes=1)
    #     self.lag_kernel = NP.fft.fftshift(lag_kernel, axes=1)
    #     self.cc_skyvis_lag = NP.fft.fftshift(ccomponents_noiseless, axes=1)
    #     self.cc_skyvis_res_lag = NP.fft.fftshift(ccres_noiseless, axes=1)
    #     self.cc_vis_lag = NP.fft.fftshift(ccomponents_noisy, axes=1)
    #     self.cc_vis_res_lag = NP.fft.fftshift(ccres_noisy, axes=1)

    #     self.cc_skyvis_net_lag = self.cc_skyvis_lag + self.cc_skyvis_res_lag
    #     self.cc_vis_net_lag = self.cc_vis_lag + self.cc_vis_res_lag
    #     self.lags = NP.fft.fftshift(lags)

    #     self.cc_skyvis_freq = cc_skyvis
    #     self.cc_skyvis_res_freq = cc_skyvis_res
    #     self.cc_vis_freq = cc_vis
    #     self.cc_vis_res_freq = cc_vis_res

    #     self.cc_skyvis_net_freq = cc_skyvis + cc_skyvis_res
    #     self.cc_vis_net_freq = cc_vis + cc_vis_res

    #     self.clean_window_buffer = clean_window_buffer
        
    #############################################################################
        
    def delay_transform_allruns(self, vis, pad=1.0, freq_wts=None, 
                                downsample=True, verbose=True):

        """
        ------------------------------------------------------------------------
        Transforms the visibilities of multiple runs from frequency axis onto 
        delay (time) axis using an IFFT. 

        Inputs:

        vis         [numpy array] Visibilities which will be delay transformed.
                    It must be of shape (...,nbl,nchan,ntimes)

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
                    array or have shape identical to input vis. Default (None) 
                    will not apply windowing and only the inherent bandpass will 
                    be used.

        downsample  [boolean] If set to True (default), the delay transform
                    quantities will be downsampled by exactly the same factor
                    that was used in padding. For instance, if pad is set to 
                    1.0, the downsampling will be by a factor of 2. If set to 
                    False, no downsampling will be done even if the original 
                    quantities were padded 

        verbose     [boolean] If set to True (default), print diagnostic and 
                    progress messages. If set to False, no such messages are
                    printed.

        Output:

        Dictionary containing delay spectrum information. It contains the 
        following keys and values:
        'lags'      [numpy array] lags of the subband delay spectra with or
                    without resampling. If not resampled it is of size 
                    nlags=nchan+npad where npad is the number of frequency 
                    channels padded specified under the key 'npad'. If 
                    resampled, it is of shape nlags where nlags is the number 
                    of independent delay bins
        'lag_kernel'
                    [numpy array] The delay kernel which is the result of the
                    bandpass shape and the spectral window used in determining
                    the delay spectrum. It is of shape 
                    n_bl x n_win x nlags x n_t.
        'vis_lag'   [numpy array] delay spectra of visibilities, after 
                    applying the frequency weights under the key 'freq_wts'. It 
                    is of size n_win x (n1xn2x... n_runs dims) x n_bl x nlags x
                    x n_t. 
        ------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing to compute delay transform...\n\tChecking input parameters for compatibility...'

        try:
            vis
        except NameError:
            raise NameError('Input vis must be provided')

        if not isinstance(vis, NP.ndarray):
            raise TypeError('Input vis must be a numpy array')
        elif vis.ndim < 3:
            raise ValueError('Input vis must be at least 3-dimensional')
        elif vis.shape[-3:] == (self.ia.baselines.shape[0],self.f.size,self.n_acc):
            if vis.ndim == 3:
                shp = (1,) + vis.shape
            else:
                shp = vis.shape
            vis = vis.reshape(shp)
        else:
            raise ValueError('Input vis does not have compatible shape')

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        if freq_wts is not None:
            if freq_wts.shape == self.f.shape:
                freq_wts = freq_wts.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+(1,-1,1))
            elif freq_wts.shape == (self.f.size, self.n_acc):
                freq_wts = freq_wts.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+(1,self.f.size,self.n_acc))
            elif freq_wts.shape == (self.ia.baselines.shape[0], self.f.size):
                freq_wts = freq_wts.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+(self.ia.baselines.shape[0],self.f.size,1))
            elif freq_wts.shape == (self.ia.baselines.shape[0], self.f.size, self.n_acc):
                freq_wts = freq_wts.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+(self.ia.baselines.shape[0],self.f.size,self.n_acc))
            elif not freq_wts.shape != vis.shape:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
        else:
            freq_wts = self.bp_wts.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+self.bp_wts.shape)
        bp = self.bp.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+self.bp.shape)
        if verbose:
            print '\tFrequency window weights assigned.'

        if not isinstance(downsample, bool):
            raise TypeError('Input downsample must be of boolean type')

        if verbose:
            print '\tInput parameters have been verified to be compatible.\n\tProceeding to compute delay transform.'
            
        result = {}
        result['freq_wts'] = freq_wts
        result['pad'] = pad
        result['lags'] = DSP.spectral_axis(int(self.f.size*(1+pad)), delx=self.df, use_real=False, shift=True)
        if pad == 0.0:
            result['vis_lag'] = DSP.FT1D(vis * bp * freq_wts, ax=-2, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            result['lag_kernel'] = DSP.FT1D(bp * freq_wts, ax=-2, inverse=True, use_real=False, shift=True) * self.f.size * self.df
            if verbose:
                print '\tDelay transform computed without padding.'
        else:
            npad = int(self.f.size * pad)
            pad_shape = NP.zeros((len(vis.shape[:-3]),2), dtype=NP.int).tolist()
            pad_shape += [[0,0], [0,npad], [0,0]]
            result['vis_lag'] = DSP.FT1D(NP.pad(vis * bp * freq_wts, pad_shape, mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            result['lag_kernel'] = DSP.FT1D(NP.pad(bp * freq_wts, pad_shape, mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
            if verbose:
                print '\tDelay transform computed with padding fraction {0:.1f}'.format(pad)

        if downsample:
            result['vis_lag'] = DSP.downsampler(result['vis_lag'], 1+pad, axis=-2)
            result['lag_kernel'] = DSP.downsampler(result['lag_kernel'], 1+pad, axis=-2)
            result['lags'] = DSP.downsampler(result['lags'], 1+pad)
            result['lags'] = result['lags'].flatten()
            if verbose:
                print '\tDelay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'delay_transform() completed successfully.'

        return result

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
        pad_factor = (1.0 + 1.0*npad/self.f.size) # to make sure visibilities after CLEANing are at the same amplitude level as before CLEANing
        cc_skyvis = NP.fft.fft(ccomponents_noiseless, axis=1) * deta * pad_factor
        cc_skyvis_res = NP.fft.fft(ccres_noiseless, axis=1) * deta * pad_factor
    
        cc_vis = NP.fft.fft(ccomponents_noisy, axis=1) * deta * pad_factor
        cc_vis_res = NP.fft.fft(ccres_noisy, axis=1) * deta * pad_factor
    
        self.lags = lags
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
                                fftpow=None, pad=None, bpcorrect=False, action=None,
                                verbose=True):

        """
        ------------------------------------------------------------------------
        Computes delay transform on multiple frequency sub-bands with specified
        weights

        Inputs:

        bw_eff       [dictionary] dictionary with two keys 'cc' and 'sim' to
                     specify effective bandwidths (in Hz) on the selected 
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

        fftpow       [dictionary] dictionary with two keys 'cc' and 'sim' to 
                     specify the power to which the FFT of the window will be 
                     raised. The values under these keys must be a positive 
                     scalar. Default = 1.0 for each key

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

        action       [string or None] If set to None (default) just updates the 
                     attribute. If set to 'return_oversampled' it returns the 
                     output dictionary corresponding to oversampled delay space
                     quantities and updates its attribute 
                     subband_delay_spectra with full resolution in delay space. 
                     If set to 'return_resampled' it returns the output 
                     dictionary corresponding to resampled/downsampled delay
                     space quantities and updates the attribute.

        verbose      [boolean] If set to True (default), print diagnostic and 
                     progress messages. If set to False, no such messages are
                     printed.

        Output: 

        If keyword input action is set to None (default), the output
        is internally stored in the class attributes
        subband_delay_spectra and subband_delay_spectra_resampled. If action is 
        set to 'return_oversampled', the following  
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

        If action is set to 'return_resampled', the following  
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
        'bw_eff'    [numpy array] contains the effective bandwidths 
                    (in Hz) of the subbands being delay transformed. It
                    is of size n_win. It is roughly equivalent to width 
                    in redshift or along line-of-sight
        'lags'      [numpy array] lags of the resampled subband delay spectra 
                    after padding in frequency during the transform. It
                    is of size nlags where nlags is the number of 
                    independent delay bins
        'lag_kernel'
                    [numpy array] delay transform of the frequency 
                    weights under the key 'freq_wts'. It is of size
                    n_bl x n_win x nlags x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the resampled subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the effective bandwidth 
        'skyvis_lag'
                    [numpy array] resampled subband delay spectra of simulated 
                    or CLEANed noiseless visibilities, depending on whether
                    the top level key is 'cc' or 'sim' respectively,
                    after applying the frequency weights under the key 
                    'freq_wts'. It is of size 
                    n_bl x n_win x nlags x n_t. 
        'vis_lag'   [numpy array] resampled subband delay spectra of simulated 
                    or CLEANed noisy visibilities, depending on whether
                    the top level key is 'cc' or 'sim' respectively,
                    after applying the frequency weights under the key 
                    'freq_wts'. It is of size 
                    n_bl x n_win x nlags x n_t. 
        'vis_noise_lag'   
                    [numpy array] resampled subband delay spectra of simulated 
                    noise after applying the frequency weights under 
                    the key 'freq_wts'. Only present if top level key is 'sim'
                    and absent for 'cc'. It is of size 
                    n_bl x n_win x nlags x n_t. 
        'skyvis_res_lag'
                    [numpy array] resampled subband delay spectra of residuals
                    after delay CLEAN of simualted noiseless 
                    visibilities obtained after applying frequency 
                    weights specified under key 'freq_wts'. Only present for
                    top level key 'cc' and absent for 'sim'. It is of
                    size n_bl x n_win x nlags x n_t
        'vis_res_lag'
                    [numpy array] resampled subband delay spectra of residuals
                    after delay CLEAN of simualted noisy 
                    visibilities obtained after applying frequency 
                    weights specified under key 'freq_wts'. Only present for
                    top level key 'cc' and absent for 'sim'. It is of
                    size n_bl x n_win x nlags x n_t
        ------------------------------------------------------------------------
        """

        try:
            bw_eff
        except NameError:
            raise NameError('Effective bandwidth must be specified')
        else:
            if not isinstance(bw_eff, dict):
                raise TypeError('Effective bandwidth must be specified as a dictionary')
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

        if fftpow is None:
            fftpow = {key: 1.0 for key in ['cc', 'sim']}
        else:
            if not isinstance(fftpow, dict):
                raise TypeError('Power to raise FFT of window by must be specified as a dictionary')
            for key in ['cc', 'sim']:
                if not isinstance(fftpow[key], (int, float)):
                    raise TypeError('Power to raise window FFT by must be a scalar value.')
                if fftpow[key] < 0.0:
                    raise ValueError('Power for raising FFT of window by must be positive.')

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
                freq_wts = NP.empty((bw_eff[key].size, self.f.size), dtype=NP.float_)
                frac_width = DSP.window_N2width(n_window=None, shape=shape[key], fftpow=fftpow[key], area_normalize=False, power_normalize=True)
                window_loss_factor = 1 / frac_width
                n_window = NP.round(window_loss_factor * bw_eff[key] / self.df).astype(NP.int)
                ind_freq_center, ind_channels, dfrequency = LKP.find_1NN(self.f.reshape(-1,1), freq_center[key].reshape(-1,1), distance_ULIM=0.5*self.df, remove_oob=True)
                sortind = NP.argsort(ind_channels)
                ind_freq_center = ind_freq_center[sortind]
                ind_channels = ind_channels[sortind]
                dfrequency = dfrequency[sortind]
                n_window = n_window[sortind]
    
                for i,ind_chan in enumerate(ind_channels):
                    window = NP.sqrt(frac_width * n_window[i]) * DSP.window_fftpow(n_window[i], shape=shape[key], fftpow=fftpow[key], centering=True, peak=None, area_normalize=False, power_normalize=True)
                    # window = NP.sqrt(frac_width * n_window[i]) * DSP.windowing(n_window[i], shape=shape[key], centering=True, peak=None, area_normalize=False, power_normalize=True)
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
                    skyvis_net_freq = self.cc_skyvis_net_freq[:,:self.f.size,:]
                    vis_net_freq = self.cc_vis_net_freq[:,:self.f.size,:]
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
                    skyvis_net_lag = DSP.FT1D(NP.pad(skyvis_net_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                    vis_net_lag = DSP.FT1D(NP.pad(vis_net_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                    result[key]['vis_res_lag'] = vis_res_lag
                    result[key]['skyvis_res_lag'] = skyvis_res_lag
                    result[key]['vis_net_lag'] = vis_net_lag
                    result[key]['skyvis_net_lag'] = skyvis_net_lag
                    result[key]['bpcorrect'] = bpcorrect
                else:
                    result[key]['vis_noise_lag'] = vis_noise_lag
        if verbose:
            print '\tSub-band(s) delay transform computed'

        self.subband_delay_spectra = result

        result_resampled = {}
        for key in ['cc', 'sim']:
            if key in result:
                result_resampled[key] = {}
                result_resampled[key]['freq_center'] = result[key]['freq_center']
                result_resampled[key]['bw_eff'] = result[key]['bw_eff']
    
                downsample_factor = NP.min((self.f.size + npad) * self.df / result_resampled[key]['bw_eff'])
                result_resampled[key]['lags'] = DSP.downsampler(result[key]['lags'], downsample_factor, axis=-1, method='interp', kind='linear')
                result_resampled[key]['lag_kernel'] = DSP.downsampler(result[key]['lag_kernel'], downsample_factor, axis=2, method='interp', kind='linear')
                result_resampled[key]['skyvis_lag'] = DSP.downsampler(result[key]['skyvis_lag'], downsample_factor, axis=2, method='FFT')
                result_resampled[key]['vis_lag'] = DSP.downsampler(result[key]['vis_lag'], downsample_factor, axis=2, method='FFT')
                dlag = result_resampled[key]['lags'][1] - result_resampled[key]['lags'][0]
                result_resampled[key]['lag_corr_length'] = (1/result[key]['bw_eff']) / dlag
                if key == 'cc': 
                    result_resampled[key]['skyvis_res_lag'] = DSP.downsampler(result[key]['skyvis_res_lag'], downsample_factor, axis=2, method='FFT')
                    result_resampled[key]['vis_res_lag'] = DSP.downsampler(result[key]['vis_res_lag'], downsample_factor, axis=2, method='FFT')
                    result_resampled[key]['skyvis_net_lag'] = DSP.downsampler(result[key]['skyvis_net_lag'], downsample_factor, axis=2, method='FFT')
                    result_resampled[key]['vis_net_lag'] = DSP.downsampler(result[key]['vis_net_lag'], downsample_factor, axis=2, method='FFT')
                else:
                    result_resampled[key]['vis_noise_lag'] = DSP.downsampler(result[key]['vis_noise_lag'], downsample_factor, axis=2, method='FFT')
        if verbose:
            print '\tDownsampled Sub-band(s) delay transform computed'

        self.subband_delay_spectra_resampled = result_resampled

        if action is not None:
            if action == 'return_oversampled':
                return result
            if action == 'return_resampled':
                return result_resampled

    #############################################################################

    def subband_delay_transform_allruns(self, vis, bw_eff, freq_center=None, 
                                        shape=None, fftpow=None, pad=None, 
                                        bpcorrect=False, action=None,
                                        verbose=True):

        """
        ------------------------------------------------------------------------
        Computes delay transform on multiple frequency sub-bands with specified
        weights for multiple realizations of visibilities

        Inputs:

        vis          [numpy array] Visibilities which will be delay transformed.
                     It must be of shape (...,nbl,nchan,ntimes)

        bw_eff       [scalar, list or numpy array] effective bandwidths (in Hz) 
                     on the selected frequency windows for subband delay 
                     transform of visibilities. The values can be a scalar, list 
                     or numpy array. If a scalar value is provided, the same 
                     will be applied to all frequency windows.

        freq_center  [scalar, list or numpy array] frequency centers (in Hz) of 
                     the selected frequency windows for subband delay transform 
                     of visibilities. The values can be a scalar, list or numpy 
                     array. If a scalar is provided, the same will be applied 
                     to all frequency windows. Default=None uses the center 
                     frequency from the class attribute

        shape        [string] frequency window shape for subband delay transform 
                     of visibilities. It must be a string. Accepted values for the
                     string are 'rect' or 'RECT' (for rectangular), 'bnw' and 
                     'BNW' (for Blackman-Nuttall), and 'bhw' or 'BHW' (for 
                     Blackman-Harris). Default=None sets it to 'rect' 
                     (rectangular window)

        fftpow       [scalar] the power to which the FFT of the window will be 
                     raised. The value must be a positive scalar. Default = 1.0 

        pad          [scalar] padding fraction relative to the number of 
                     frequency channels. Value must be a non-negative scalar. 
                     For e.g., a pad of 1.0 pads the frequency axis with zeros 
                     of the same width as the number of channels. After the 
                     delay transform, the transformed visibilities are 
                     downsampled by a factor of 1+pad. If a negative value is 
                     specified, delay transform will be performed with no 
                     padding. Default=None sets to padding factor to 1.0

        action       [string or None] If set to 'return_oversampled' it returns 
                     the output dictionary corresponding to oversampled delay 
                     space quantities with full resolution in delay space. If 
                     set to None (default) or 'return_resampled' it returns the 
                     output dictionary corresponding to resampled/downsampled 
                     delay space quantities.

        verbose      [boolean] If set to True (default), print diagnostic and 
                     progress messages. If set to False, no such messages are
                     printed.

        Output: 

        The output is a dictionary that contains information about delay spectra 
        of different frequency sub-bands (n_win in number). If action is set to
        'return_resampled', it contains the following keys and values:
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
                    n_win x (1 x 1 x ... nruns times) x n_bl x
                    (nchan+npad) x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the total bandwidth 
                    (nchan x df) simulated. It is of size n_win
        'vis_lag'   [numpy array] subband delay spectra of visibilities, 
                    after applying the frequency weights under the key 
                    'freq_wts'. It is of size 
                    n_win x (n1xn2x... n_runs dims) x n_bl x (nchan+npad) x 
                    x n_t. 

        If action is set to 'return_resampled', the following  
        output is returned. The output is a dictionary that contains  
        information about delay spectra of different frequency sub-bands 
        (n_win in number) with the following keys and values:
        'freq_center' 
                    [numpy array] contains the center frequencies 
                    (in Hz) of the frequency subbands of the subband
                    delay spectra. It is of size n_win. It is roughly 
                    equivalent to redshift(s)
        'bw_eff'    [numpy array] contains the effective bandwidths 
                    (in Hz) of the subbands being delay transformed. It
                    is of size n_win. It is roughly equivalent to width 
                    in redshift or along line-of-sight
        'lags'      [numpy array] lags of the resampled subband delay spectra 
                    after padding in frequency during the transform. It
                    is of size nlags where nlags is the number of 
                    independent delay bins
        'lag_kernel'
                    [numpy array] delay transform of the frequency 
                    weights under the key 'freq_wts'. It is of size
                    n_win x (1 x 1 x ... nruns times) x n_bl x nlags x n_t
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the total bandwidth 
                    (nchan x df) simulated. It is of size n_win
        'vis_lag'   [numpy array] subband delay spectra of visibilities, 
                    after applying the frequency weights under the key 
                    'freq_wts'. It is of size 
                    n_win x (n1xn2x... n_runs dims) x n_bl x nlags x n_t
        ------------------------------------------------------------------------
        """

        try:
            vis, bw_eff
        except NameError:
            raise NameError('Input visibilities and effective bandwidth must be specified')
        else:
            if not isinstance(vis, NP.ndarray):
                raise TypeError('Input vis must be a numpy array')
            elif vis.ndim < 3:
                raise ValueError('Input vis must be at least 3-dimensional')
            elif vis.shape[-3:] == (self.ia.baselines.shape[0],self.f.size,self.n_acc):
                if vis.ndim == 3:
                    shp = (1,) + vis.shape
                else:
                    shp = vis.shape
                vis = vis.reshape(shp)
            else:
                raise ValueError('Input vis does not have compatible shape')
            
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
            if shape.lower() not in ['rect', 'bhw', 'bnw']:
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

        result = {}
        freq_wts = NP.empty((bw_eff.size, self.f.size), dtype=NP.float_)
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

        freq_wts = freq_wts.reshape((bw_eff.size,)+tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+(1,self.f.size,1))
        bp = self.bp.reshape(tuple(NP.ones(len(vis.shape[:-3]),dtype=NP.int))+self.bp.shape)
        npad = int(self.f.size * pad)
        lags = DSP.spectral_axis(self.f.size + npad, delx=self.df, use_real=False, shift=True)

        pad_shape = [[0,0]] + NP.zeros((len(vis.shape[:-3]),2), dtype=NP.int).tolist()
        pad_shape += [[0,0], [0,npad], [0,0]]
        vis_lag = DSP.FT1D(NP.pad(vis[NP.newaxis,...] * bp[NP.newaxis,...] * freq_wts, pad_shape, mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
        lag_kernel = DSP.FT1D(NP.pad(bp[NP.newaxis,...] * freq_wts, pad_shape, mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
        result = {'freq_center': freq_center, 'shape': shape, 'freq_wts': freq_wts, 'bw_eff': bw_eff, 'npad': npad, 'lags': lags, 'vis_lag': vis_lag, 'lag_kernel': lag_kernel, 'lag_corr_length': self.f.size / NP.squeeze(NP.sum(freq_wts, axis=-2))}

        if verbose:
            print '\tSub-band(s) delay transform computed'

        if action is not None:
            action = 'return_resampled'
        if action == 'return_oversampled':
            return result
        elif action == 'return_resampled':
            downsample_factor = NP.min((self.f.size + npad) * self.df / result['bw_eff'])
            result['lags'] = DSP.downsampler(result['lags'], downsample_factor, axis=-1, method='interp', kind='linear')
            result['lag_kernel'] = DSP.downsampler(result['lag_kernel'], downsample_factor, axis=-2, method='interp', kind='linear')
            result['vis_lag'] = DSP.downsampler(result['vis_lag'], downsample_factor, axis=-2, method='FFT')
            dlag = result['lags'][1] - result['lags'][0]
            result['lag_corr_length'] = (1/result['bw_eff']) / dlag
            return result
        else:
            raise ValueError('Invalid value specified for keyword input action')

        if verbose:
            print '\tDownsampled Sub-band(s) delay transform computed'

    #############################################################################

    def subband_delay_transform_closure_phase(self, bw_eff, cpinfo=None,
                                              antenna_triplets=None,
                                              specsmooth_info=None,
                                              delay_filter_info=None,
                                              spectral_window_info=None,
                                              freq_center=None, shape=None, 
                                              fftpow=None, pad=None, action=None,
                                              verbose=True):

        """
        ------------------------------------------------------------------------
        Computes delay transform of closure phases on antenna triplets on 
        multiple frequency sub-bands with specified weights. It will have units 
        of Hz

        Inputs:

        bw_eff      [scalar or numpy array] effective bandwidths (in Hz) on the 
                    selected frequency windows for subband delay transform of 
                    closure phases. If a scalar value is provided, the same 
                    will be applied to all frequency windows

        cpinfo      [dictionary] If set to None, it will be determined based on
                    other inputs. Otherwise, it will be used directly. The 
                    dictionary will contain the following keys and values:
                    'closure_phase_skyvis'  [numpy array] [optional] Closure 
                                            phases (in radians) for the given 
                                            antenna triplets from the noiseless 
                                            visibilities. It is of shape
                                            ntriplets x ... x nchan x ntimes
                    'closure_phase_vis'     [numpy array] [optional] Closure 
                                            phases (in radians) for the given 
                                            antenna triplets for noisy 
                                            visibilities. It is of shape 
                                            ntriplets x ... x nchan x ntimes
                    'closure_phase_noise'   [numpy array] [optional] Closure 
                                            phases (in radians) for the given 
                                            antenna triplets for thermal noise 
                                            in visibilities. It is of shape 
                                            ntriplets x ... x nchan x ntimes
                    'antenna_triplets'      [list of tuples] List of 
                                            three-element tuples of antenna IDs 
                                            for which the closure phases are 
                                            calculated.
                    'baseline_triplets'     [numpy array] List of 3x3 numpy 
                                            arrays. Each 3x3 unit in the list 
                                            represents triplets of baseline 
                                            vectors where the three rows denote 
                                            the three baselines in the triplet 
                                            and the three columns define the x-, 
                                            y- and z-components of the triplet. 
                                            The number of 3x3 unit elements in 
                                            the list will equal the number of 
                                            elements in the list under key 
                                            'antenna_triplets'. 

        antenna_triplets
                    [list of tuples] List of antenna ID triplets where each 
                    triplet is given as a tuple. If set to None (default), all
                    the unique triplets based on the antenna layout attribute
                    in class InterferometerArray

        specsmooth_info         
                    [NoneType or dictionary] Spectral smoothing window to be 
                    applied prior to the delay transform. If set to None, no 
                    smoothing is done. This is usually set if spectral 
                    smoothing is to be done such as in the case of RFI. The 
                    smoothing window parameters are specified using the
                    following keys and values:
                    'op_type'     [string] Smoothing operation type. 
                                  Default='median' (currently accepts only 
                                  'median' or 'interp'). 
                    'window_size' [integer] Size of smoothing window (in 
                                  pixels) along frequency axis. Applies only
                                  if op_type is set to 'median'
                    'maskchans'   [NoneType or numpy array] Numpy boolean array
                                  of size nchan. False entries imply those
                                  channels are not masked and will be used in 
                                  in interpolation while True implies they are
                                  masked and will not be used in determining the
                                  interpolation function. If set to None, all
                                  channels are assumed to be unmasked (False).
                    'evalchans'   [NoneType or numpy array] Channel numbers at 
                                  which visibilities are to be evaluated. Will 
                                  be useful for filling in RFI flagged channels.
                                  If set to None, all channels will be evaluated
                    'noiseRMS'    [NoneType or scalar or numpy array] If set to 
                                  None (default), the rest of the parameters are 
                                  used in determining the RMS of thermal noise. 
                                  If specified as scalar, all other parameters 
                                  will be ignored in estimating noiseRMS and 
                                  this value will be used instead. If specified 
                                  as a numpy array, it must be of shape 
                                  broadcastable to (nbl,nchan,ntimes). So 
                                  accpeted shapes can be (1,1,1), (1,1,ntimes), 
                                  (1,nchan,1), (nbl,1,1), (1,nchan,ntimes), 
                                  (nbl,nchan,1), (nbl,1,ntimes), or 
                                  (nbl,nchan,ntimes). 

        delay_filter_info
                    [NoneType or dictionary] Info containing delay filter 
                    parameters. If set to None (default), no delay filtering is
                    performed. Otherwise, delay filter is applied on each of the
                    visibilities in the triplet before computing the closure
                    phases. The delay filter parameters are specified in a 
                    dictionary as follows:
                    'type'      [string] 'horizon' (default) or 'regular'. If
                                set to 'horizon', the horizon delay limits are
                                estimated from the respective baseline lengths
                                in the triplet. If set to 'regular', the extent
                                of the filter is determined by the 'min' and
                                'width' keys (see below). 
                    'min'       [scalar] Non-negative number (in seconds) that
                                specifies the minimum delay in the filter span.
                                If not specified, it is assumed to be 0. If 
                                'type' is set to 'horizon', the 'min' is ignored 
                                and set to 0. 
                    'width'     [scalar] Non-negative number (in numbers of 
                                inverse bandwidths). If 'type' is set to 
                                'horizon', the width represents the delay 
                                buffer beyond the horizon. If 'type' is set to
                                'regular', this number has to be positive and
                                determines the span of the filter starting from
                                the minimum delay in key 'min'. 
                    'mode'      [string] 'discard' (default) or 'retain'. If set
                                to 'discard', the span defining the filter is
                                discarded and the rest retained. If set to 
                                'retain', the span defining the filter is 
                                retained and the rest discarded. For example, 
                                if 'type' is set to 'horizon' and 'mode' is set
                                to 'discard', the horizon-to-horizon is 
                                filtered out (discarded).

        spectral_window_info    
                    [NoneType or dictionary] Spectral window parameters to 
                    determine the spectral weights and apply to the visibilities 
                    in the frequency domain before filtering in the delay domain. 
                    THESE PARAMETERS ARE APPLIED ON THE INDIVIDUAL VISIBILITIES 
                    THAT GO INTO THE CLOSURE PHASE. THESE ARE NOT TO BE CONFUSED 
                    WITH THE PARAMETERS THAT WILL BE USED IN THE ACTUAL DELAY 
                    TRANSFORM OF CLOSURE PHASE SPECTRA WHICH ARE SPECIFIED
                    SEPARATELY FURTHER BELOW. 
                    If set to None (default), unity spectral weights are applied. 
                    If spectral weights are to be applied, it must be a provided 
                    as a dictionary with the following keys and values:
                    bw_eff       [scalar] effective bandwidths (in Hz) for the 
                                 spectral window
                    freq_center  [scalar] frequency center (in Hz) for the 
                                 spectral window
                    shape        [string] frequency window shape for the 
                                 spectral window. Accepted values are 'rect' or 
                                 'RECT' (for rectangular), 'bnw' and 'BNW' (for 
                                 Blackman-Nuttall), and 'bhw' or 'BHW' (for 
                                 Blackman-Harris). Default=None sets it to 'rect' 
                    fftpow       [scalar] power to which the FFT of the window 
                                 will be raised. The value must be a positive 
                                 scalar. 

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

        action      [string or None] If set to None (default) just updates the 
                    attribute. If set to 'return_oversampled' it returns the 
                    output dictionary corresponding to oversampled delay space
                    quantities with full resolution in delay space. If set to 
                    None (default) or 'return_resampled', it returns the output 
                    dictionary corresponding to resampled or downsampled delay 
                    space quantities.

        verbose     [boolean] If set to True (default), print diagnostic and 
                    progress messages. If set to False, no such messages are
                    printed.

        Output: 

        If keyword input action is set to 'return_oversampled', the following  
        output is returned. The output is a dictionary that contains information 
        about delay spectra of different frequency sub-bands (n_win in number) 
        under the following keys:
        'antenna_triplets'
                    [list of tuples] List of antenna ID triplets where each 
                    triplet is given as a tuple. Closure phase delay spectra in
                    subbands is computed for each of these antenna triplets
        'baseline_triplets'     
                    [numpy array] List of 3x3 numpy arrays. Each 3x3
                    unit in the list represents triplets of baseline
                    vectors where the three rows denote the three 
                    baselines in the triplet and the three columns 
                    define the x-, y- and z-components of the 
                    triplet. The number of 3x3 unit elements in the 
                    list will equal the number of elements in the 
                    list under key 'antenna_triplets'. Closure phase delay 
                    spectra in subbands is computed for each of these baseline
                    triplets which correspond to the antenna triplets
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
                    n_triplets x ... x n_win x (nchan+npad) x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the total bandwidth 
                    (nchan x df) simulated.
        'closure_phase_skyvis'
                    [numpy array] subband delay spectra of closure phases
                    of noiseless sky visiblities from the specified 
                    antenna triplets. It is of size n_triplets x ... n_win x 
                    nlags x n_t. It is in units of Hz
        'closure_phase_vis'
                    [numpy array] subband delay spectra of closure phases
                    of noisy sky visiblities from the specified antenna 
                    triplets. It is of size n_triplets x ... x n_win x 
                    nlags x n_t. It is in units of Hz
        'closure_phase_noise'
                    [numpy array] subband delay spectra of closure phases
                    of noise visiblities from the specified antenna triplets.
                    It is of size n_triplets x ... x n_win x nlags x n_t. It 
                    is in units of Hz

        If action is set to 'return_resampled', the following  
        output is returned. The output is a dictionary that contains 
        information about closure phases. Under each of these keys is
        information about delay spectra of different frequency sub-bands 
        (n_win in number) under the following keys:
        'antenna_triplets'
                    [list of tuples] List of antenna ID triplets where each 
                    triplet is given as a tuple. Closure phase delay spectra in
                    subbands is computed for each of these antenna triplets
        'baseline_triplets'     
                    [numpy array] List of 3x3 numpy arrays. Each 3x3
                    unit in the list represents triplets of baseline
                    vectors where the three rows denote the three 
                    baselines in the triplet and the three columns 
                    define the x-, y- and z-components of the 
                    triplet. The number of 3x3 unit elements in the 
                    list will equal the number of elements in the 
                    list under key 'antenna_triplets'. Closure phase delay 
                    spectra in subbands is computed for each of these baseline
                    triplets which correspond to the antenna triplets
        'freq_center' 
                    [numpy array] contains the center frequencies 
                    (in Hz) of the frequency subbands of the subband
                    delay spectra. It is of size n_win. It is roughly 
                    equivalent to redshift(s)
        'bw_eff'    [numpy array] contains the effective bandwidths 
                    (in Hz) of the subbands being delay transformed. It
                    is of size n_win. It is roughly equivalent to width 
                    in redshift or along line-of-sight
        'lags'      [numpy array] lags of the resampled subband delay spectra 
                    after padding in frequency during the transform. It
                    is of size nlags where nlags is the number of 
                    independent delay bins
        'lag_kernel'
                    [numpy array] delay transform of the frequency 
                    weights under the key 'freq_wts'. It is of size
                    n_triplets x ... x n_win x nlags x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the resampled subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the effective bandwidth 
        'closure_phase_skyvis'
                    [numpy array] subband delay spectra of closure phases
                    of noiseless sky visiblities from the specified 
                    antenna triplets. It is of size n_triplets x ... x n_win x 
                    nlags x n_t. It is in units of Hz
        'closure_phase_vis'
                    [numpy array] subband delay spectra of closure phases
                    of noisy sky visiblities from the specified antenna 
                    triplets. It is of size n_triplets x ... x n_win x 
                    nlags x n_t. It is in units of Hz
        'closure_phase_noise'
                    [numpy array] subband delay spectra of closure phases
                    of noise visiblities from the specified antenna triplets.
                    It is of size n_triplets x ... x n_win x nlags x n_t. It is 
                    in units of Hz
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

        if cpinfo is not None:
            if not isinstance(cpinfo, dict):
                raise TypeError('Input cpinfo must be a dictionary')
        else:
            cpinfo = self.ia.getClosurePhase(antenna_triplets=antenna_triplets, specsmooth_info=specsmooth_info, delay_filter_info=delay_filter_info, spectral_window_info=spectral_window_info)
        result = {'antenna_triplets': cpinfo['antenna_triplets'], 'baseline_triplets': cpinfo['baseline_triplets']}

        freq_wts = NP.empty((bw_eff.size, self.f.size), dtype=NP.float_)
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
    
        # lag_kernel = DSP.FT1D(NP.pad(self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
        # lag_kernel = DSP.FT1D(NP.pad(freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
        result = {'freq_center': freq_center, 'shape': shape, 'freq_wts': freq_wts, 'bw_eff': bw_eff, 'npad': npad, 'lags': lags, 'lag_corr_length': self.f.size / NP.sum(freq_wts, axis=-1)}

        for key in cpinfo:
            if key in ['closure_phase_skyvis', 'closure_phase_vis', 'closure_phase_noise']:
                available_CP_key = key
                ndim_padtuple = [(0,0) for i in range(1+len(cpinfo[key].shape[:-2]))] + [(0,npad), (0,0)]
                result[key] = DSP.FT1D(NP.pad(NP.exp(-1j*cpinfo[key].reshape(cpinfo[key].shape[:-2]+(1,)+cpinfo[key].shape[-2:])) * freq_wts.reshape(tuple(NP.ones(len(cpinfo[key].shape[:-2])).astype(int))+freq_wts.shape+(1,)), ndim_padtuple, mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
                # result[key] = DSP.FT1D(NP.pad(NP.exp(-1j*cpinfo[key][:,NP.newaxis,:,:]) * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
        lag_kernel = DSP.FT1D(NP.pad(freq_wts.reshape(tuple(NP.ones(len(cpinfo[available_CP_key].shape[:-2])).astype(int))+freq_wts.shape+(1,)), ndim_padtuple, mode='constant'), ax=-2, inverse=True, use_real=False, shift=True) * (npad + self.f.size) * self.df
        result['lag_kernel'] = lag_kernel
        if verbose:
            print '\tSub-band(s) delay transform computed'

        result_resampled = {'antenna_triplets': cpinfo['antenna_triplets'], 'baseline_triplets': cpinfo['baseline_triplets']}
        result_resampled['freq_center'] = result['freq_center']
        result_resampled['bw_eff'] = result['bw_eff']
        result_resampled['freq_wts'] = result['freq_wts']
    
        downsample_factor = NP.min((self.f.size + npad) * self.df / result_resampled['bw_eff'])
        result_resampled['lags'] = DSP.downsampler(result['lags'], downsample_factor, axis=-1, method='interp', kind='linear')
        result_resampled['lag_kernel'] = DSP.downsampler(result['lag_kernel'], downsample_factor, axis=-2, method='interp', kind='linear')
        dlag = result_resampled['lags'][1] - result_resampled['lags'][0]
        result_resampled['lag_corr_length'] = (1/result['bw_eff']) / dlag
        for key in ['closure_phase_skyvis', 'closure_phase_vis', 'closure_phase_noise']:
            if key in result:
                result_resampled[key] = DSP.downsampler(result[key], downsample_factor, axis=-2, method='FFT')

        if verbose:
            print '\tDownsampled Sub-band(s) delay transform computed'

        if (action is None) or (action.lower() == 'return_resampled'):
            return result_resampled
        elif action.lower() == 'return_oversampled':
            return result
        else:
            raise ValueError('Invalid action specified')

################################################################################

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
        
    def save(self, ds_outfile, ia_outfile, tabtype='BinTabelHDU', overwrite=False,
             verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the interferometer array delay spectrum information to disk. 

        Inputs:

        outfile      [string] Filename with full path for  for delay spectrum 
                     data to be saved to. Will be appended with '.ds.fits'

        ia_outfile   [string] Filename with full path for interferometer array
                     data to be saved to. Will be appended with '.fits' 
                     extension 

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
            ds_outfile, ia_outfile
        except NameError:
            raise NameError('Both delay spectrum and interferometer array output filenames must be specified. Aborting DelaySpectrum.save()...')

        if verbose:
            print '\nSaving information about interferometer array...'

        self.ia.save(ia_outfile, tabtype=tabtype, overwrite=overwrite,
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
        hdulist[0].header['IARRAY'] = (ia_outfile+'.fits', 'Location of InterferometerArray simulated visibilities')

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

        hdulist += [fits.ImageHDU(self.bp, name='BANDPASS')]
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
            for key in self.subband_delay_spectra:
                hdulist[0].header['{0}-SBDS'.format(key)] = (1, 'Presence of {0} Subband Delay Spectra'.format(key))
                hdulist[0].header['{0}-SBDS-WSHAPE'.format(key)] = (self.subband_delay_spectra[key]['shape'], 'Shape of {0} subband frequency weights'.format(key))
                if key == 'cc':
                    hdulist[0].header['{0}-SBDS-BPCORR'.format(key)] = (int(self.subband_delay_spectra[key]['bpcorrect']), 'Truth value for {0} subband delay spectrum bandpass windows weights correction'.format(key))
                hdulist[0].header['{0}-SBDS-NPAD'.format(key)] = (self.subband_delay_spectra[key]['npad'], 'Number of zero-padded channels for subband delay spectra'.format(key))
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['freq_center'], name='{0}-SBDS-F0'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['freq_wts'], name='{0}-SBDS-FWTS'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['bw_eff'], name='{0}-SBDS-BWEFF'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['lags'], name='{0}-SBDS-LAGS'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['lag_kernel'].real, name='{0}-SBDS-LAGKERN-REAL'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['lag_kernel'].imag, name='{0}-SBDS-LAGKERN-IMAG'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['lag_corr_length'], name='{0}-SBDS-LAGCORR'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['skyvis_lag'].real, name='{0}-SBDS-SKYVISLAG-REAL'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['skyvis_lag'].imag, name='{0}-SBDS-SKYVISLAG-IMAG'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['vis_lag'].real, name='{0}-SBDS-VISLAG-REAL'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['vis_lag'].imag, name='{0}-SBDS-VISLAG-IMAG'.format(key))]
                if key == 'sim':
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['vis_noise_lag'].real, name='{0}-SBDS-NOISELAG-REAL'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['vis_noise_lag'].imag, name='{0}-SBDS-NOISELAG-IMAG'.format(key))]
                if key == 'cc':
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['skyvis_res_lag'].real, name='{0}-SBDS-SKYVISRESLAG-REAL'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['skyvis_res_lag'].imag, name='{0}-SBDS-SKYVISRESLAG-IMAG'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['vis_res_lag'].real, name='{0}-SBDS-VISRESLAG-REAL'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra[key]['vis_res_lag'].imag, name='{0}-SBDS-VISRESLAG-IMAG'.format(key))]

            if verbose:
                print '\tCreated extensions for information on subband delay spectra for simulated and clean components of visibilities as a function of baselines, lags/frequency and snapshot instance'

        if self.subband_delay_spectra_resampled:
            hdulist[0].header['SBDS-RS'] = (1, 'Presence of Resampled Subband Delay Spectra')
            for key in self.subband_delay_spectra_resampled:
                hdulist[0].header['{0}-SBDS-RS'.format(key)] = (1, 'Presence of {0} Reampled Subband Delay Spectra'.format(key))
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['freq_center'], name='{0}-SBDSRS-F0'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['bw_eff'], name='{0}-SBDSRS-BWEFF'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['lags'], name='{0}-SBDSRS-LAGS'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['lag_kernel'].real, name='{0}-SBDSRS-LAGKERN-REAL'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['lag_kernel'].imag, name='{0}-SBDSRS-LAGKERN-IMAG'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['lag_corr_length'], name='{0}-SBDSRS-LAGCORR'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['skyvis_lag'].real, name='{0}-SBDSRS-SKYVISLAG-REAL'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['skyvis_lag'].imag, name='{0}-SBDSRS-SKYVISLAG-IMAG'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['vis_lag'].real, name='{0}-SBDSRS-VISLAG-REAL'.format(key))]
                hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['vis_lag'].imag, name='{0}-SBDSRS-VISLAG-IMAG'.format(key))]
                if key == 'sim':
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['vis_noise_lag'].real, name='{0}-SBDSRS-NOISELAG-REAL'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['vis_noise_lag'].imag, name='{0}-SBDSRS-NOISELAG-IMAG'.format(key))]
                if key == 'cc':
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['skyvis_res_lag'].real, name='{0}-SBDSRS-SKYVISRESLAG-REAL'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['skyvis_res_lag'].imag, name='{0}-SBDSRS-SKYVISRESLAG-IMAG'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['vis_res_lag'].real, name='{0}-SBDSRS-VISRESLAG-REAL'.format(key))]
                    hdulist += [fits.ImageHDU(self.subband_delay_spectra_resampled[key]['vis_res_lag'].imag, name='{0}-SBDSRS-VISRESLAG-IMAG'.format(key))]

            if verbose:
                print '\tCreated extensions for information on resampled subband delay spectra for simulated and clean components of visibilities as a function of baselines, lags/frequency and snapshot instance'
                
        hdu = fits.HDUList(hdulist)
        hdu.writeto(ds_outfile+'.ds.fits', clobber=overwrite)

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

    horizon_kprll_limits
                [numpy array] limits on k_parallel corresponding to limits on
                horizon delays. It is of size NxMx2 denoting the neagtive and 
                positive horizon delay limits where N is the number of 
                timestamps, M is the number of baselines. The 0 index in the 
                third dimenstion denotes the negative horizon limit while 
                the 1 index denotes the positive horizon limit

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
                'cc_skyvis_net' 
                            [numpy array] delay power spectrum of sum of 
                            residuals and clean components
                            after delay cleaning of noiseless delay spectra
                'cc_vis_net'
                            [numpy array] delay power spectrum of sum of 
                            residuals and clean components  
                            after delay cleaning of noisy delay spectra

    subband_delay_power_spectra
                [dictionary] contains two top level keys, namely, 'cc' and 'sim' 
                denoting information about CLEAN and simulated visibilities 
                respectively. Essentially this is the power spectrum equivalent 
                of the attribute suuband_delay_spectra under class DelaySpectrum. 
                Under each of these keys is information about delay power spectra 
                of different frequency sub-bands (n_win in number) in the form of 
                a dictionary under the following keys: 
                'z'         [numpy array] contains the redshifts corresponding to
                            center frequencies (in Hz) of the frequency subbands 
                            of the subband delay spectra. It is of size n_win. 
                'dz'        [numpy array] contains the width in redshifts 
                            corresponding to the effective bandwidths (in Hz) of 
                            the subbands being delay transformed. It is of size 
                            n_win. 
                'kprll'     [numpy array] line-of-sight k-modes (in h/Mpc) 
                            corresponding to lags of the subband delay spectra. 
                            It is of size n_win x (nchan+npad)
                'kperp'     [numpy array] transverse k-modes (in h/Mpc) 
                            corresponding to the baseline lengths and the 
                            center frequencies. It is of size 
                            n_win x n_bl
                horizon_kprll_limits
                            [numpy array] limits on k_parallel corresponding to 
                            limits on horizon delays for each subband. It is of 
                            size N x n_win x M x 2 denoting the neagtive and 
                            positive horizon delay limits where N is the number 
                            of timestamps, n_win is the number of subbands, M is 
                            the number of baselines. The 0 index in the fourth 
                            dimenstion denotes the negative horizon limit while 
                            the 1 index denotes the positive horizon limit
                'rz_transverse'
                            [numpy array] transverse comoving distance 
                            (in Mpc/h) corresponding to the different redshifts
                            under key 'z'. It is of size n_win
                'drz_los'   [numpy array] line-of-sight comoving depth (in 
                            Mpc/h) corresponding to the redshift widths under 
                            key 'dz' and redshifts under key 'z'. It is of size
                            n_win
                'jacobian1' [numpy array] first jacobian in conversion of delay 
                            spectrum to power spectrum. It is equal to 
                            A_eff / wl**2 / bw. It is of size n_win
                'jacobian2' [numpy array] second jacobian in conversion of delay 
                            spectrum to power spectrum. It is equal to 
                            rz_transverse**2 * drz_los / bw. It is of size n_win
                'Jy2K'      [numpy array] factor to convert Jy/Sr to K. It is 
                            equal to wl**2 * Jy / (2k). It is of size n_win
                'factor'    [numpy array] conversion factor to convert delay
                            spectrum (in Jy Hz) to delay power spectrum (in 
                            K^2 (Mpc/h)^3). It is equal to 
                            jacobian1 * jacobian2 * Jy2K**2. It is of size n_win
                'skyvis_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to noiseless simulated (under top level 
                            key 'sim') or CLEANed (under top level key 'cc') 
                            delay spectrum under key 'skyvis_lag' in attribute 
                            subband_delay_spectra under instance of class 
                            DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t
                'vis_lag'   [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to noisy simulated (under top level 
                            key 'sim') or CLEANed (under top level key 'cc') 
                            delay spectrum under key 'vis_lag' in attribute 
                            subband_delay_spectra under instance of class 
                            DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t
                'vis_noise_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to thermal noise simulated (under top 
                            level key 'sim') delay spectrum under key 
                            'vis_noise_lag' in attribute subband_delay_spectra 
                            under instance of class DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t
                'skyvis_res_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to CLEAN residuals (under top level key 
                            'cc') from noiseless simulated delay spectrum under 
                            key 'skyvis_res_lag' in attribute 
                            subband_delay_spectra under instance of class 
                            DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t
                'vis_res_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to CLEAN residuals (under top level key 
                            'cc') from noisy delay spectrum under key 
                            'vis_res_lag' in attribute subband_delay_spectra 
                            under instance of class DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t
                'skyvis_net_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to sum of CLEAN components and 
                            residuals (under top level key 
                            'cc') from noiseless simulated delay spectrum under 
                            key 'skyvis_net_lag' in attribute 
                            subband_delay_spectra under instance of class 
                            DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t
                'vis_net_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to sum of CLEAN components and 
                            residuals (under top level key 
                            'cc') from noisy delay spectrum under key 
                            'vis_net_lag' in attribute subband_delay_spectra 
                            under instance of class DelaySpectrum. It is of size 
                            n_bl x n_win x (nchan+npad) x n_t

    subband_delay_power_spectra_resampled
                [dictionary] contains two top level keys, namely, 'cc' and 'sim' 
                denoting information about CLEAN and simulated visibilities 
                respectively. Essentially this is the power spectrum equivalent 
                of the attribute suuband_delay_spectra_resampled under class 
                DelaySpectrum. Under each of these keys is information about 
                delay power spectra of different frequency sub-bands (n_win in 
                number) in the form of a dictionary under the following keys: 
                'kprll'     [numpy array] line-of-sight k-modes (in h/Mpc) 
                            corresponding to lags of the subband delay spectra. 
                            It is of size n_win x nlags, where nlags is the 
                            resampeld number of delay bins
                'kperp'     [numpy array] transverse k-modes (in h/Mpc) 
                            corresponding to the baseline lengths and the 
                            center frequencies. It is of size 
                            n_win x n_bl
                'horizon_kprll_limits'
                            [numpy array] limits on k_parallel corresponding to 
                            limits on horizon delays for each subband. It is of 
                            size N x n_win x M x 2 denoting the negative and 
                            positive horizon delay limits where N is the number 
                            of timestamps, n_win is the number of subbands, M is 
                            the number of baselines. The 0 index in the fourth 
                            dimenstion denotes the negative horizon limit while 
                            the 1 index denotes the positive horizon limit
                'skyvis_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to noiseless simulated (under top level 
                            key 'sim') or CLEANed (under top level key 'cc') 
                            delay spectrum under key 'skyvis_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t
                'vis_lag'   [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to noisy simulated (under top level 
                            key 'sim') or CLEANed (under top level key 'cc') 
                            delay spectrum under key 'vis_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t
                'vis_noise_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to thermal noise simulated (under top 
                            level key 'sim') delay spectrum under key 
                            'vis_noise_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t
                'skyvis_res_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to CLEAN residuals (under top level key 
                            'cc') from noiseless simulated delay spectrum under 
                            key 'skyvis_res_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t
                'vis_res_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to CLEAN residuals (under top level key 
                            'cc') from noisy delay spectrum under key 
                            'vis_res_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t
                'skyvis_net_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to sum of CLEAN components and 
                            residuals (under top level key 
                            'cc') from noiseless simulated delay spectrum under 
                            key 'skyvis_net_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t
                'vis_net_lag'
                            [numpy array] delay power spectrum (in K^2 (Mpc/h)^3) 
                            corresponding to sum of CLEAN components and 
                            residuals (under top level key 
                            'cc') from noisy delay spectrum under key 
                            'vis_net_lag' in attribute 
                            subband_delay_spectra_resampled under instance of 
                            class DelaySpectrum. It is of size 
                            n_bl x n_win x nlags x n_t

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

    compute_power_spectrum_allruns()
                Compute delay power spectrum in units of K^2 (Mpc/h)^3 from the 
                delay spectrum in units of Jy Hz from multiple runs of 
                visibilities

    compute_individual_closure_phase_power_spectrum()
                Compute delay power spectrum of closure phase in units of 
                K^2 (Mpc/h)^3 from the delay spectrum in units of Jy Hz where 
                the original visibility amplitudes of closure phase complex 
                exponents are assumed to be 1 Jy across the band

    compute_averaged_closure_phase_power_spectrum()
                Compute delay power spectrum of closure phase in units of 
                K^2 (Mpc/h)^3 from the delay spectrum in units of Jy Hz and 
                average over 'auto' and 'cross' modes, where the original 
                visibility amplitudes of closure phase complex exponents are 
                assumed to be 1 Jy across the band
    ----------------------------------------------------------------------------
    """

    def __init__(self, dspec, cosmo=cosmo100):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class DelayPowerSpectrum. Attributes 
        initialized are: ds, cosmo, f, df, f0, z, bw, drz_los, rz_transverse,
        rz_los, kprll, kperp, jacobian1, jacobian2, subband_delay_power_spectra,
        subband_delay_power_spectra_resampled

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
        self.f0 = self.f[int(self.f.size/2)]
        self.wl0 = FCNST.c / self.f0
        self.z = CNST.rest_freq_HI / self.f0 - 1
        self.bw = self.df * self.f.size
        self.kprll = self.k_parallel(self.lags, redshift=self.z, action='return')   # in h/Mpc
        self.kperp = self.k_perp(self.bl_length, redshift=self.z, action='return')   # in h/Mpc        
        self.horizon_kprll_limits = self.k_parallel(self.ds.horizon_delay_limits, redshift=self.z, action='return')    # in h/Mpc

        self.drz_los = self.comoving_los_depth(self.bw, self.z, action='return')   # in Mpc/h
        self.rz_transverse = self.comoving_transverse_distance(self.z, action='return')   # in Mpc/h
        self.rz_los = self.comoving_los_distance(self.z, action='return')   # in Mpc/h

        # self.jacobian1 = NP.mean(self.ds.ia.A_eff) / self.wl0**2 / self.bw
        omega_bw = self.beam3Dvol(freq_wts=self.ds.bp_wts[0,:,0])
        self.jacobian1 = 1 / omega_bw
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
        self.dps['cc_skyvis_net'] = None
        self.dps['cc_vis_net'] = None

        self.subband_delay_power_spectra = {}
        self.subband_delay_power_spectra_resampled = {}

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
        wavenumbers (h/Mpc) computed is returned. It is of same size as input
        lags
        ------------------------------------------------------------------------
        """

        eta2kprll = dkprll_deta(redshift, cosmo=self.cosmo)
        kprll = eta2kprll * lags
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

    def beam3Dvol(self, freq_wts=None, nside=32):

        if self.ds.ia.simparms_file is not None:
            parms_file = open(self.ds.ia.simparms_file, 'r')
            parms = yaml.safe_load(parms_file)
            parms_file.close()
            # sky_nside = parms['fgparm']['nside']
            beam_info = parms['beam']
            use_external_beam = beam_info['use_external']
            beam_chromaticity = beam_info['chromatic']
            if use_external_beam:
                beam_file = beam_info['file']
                if beam_info['filefmt'].lower() in ['hdf5', 'fits']:
                    beam_filefmt = beam_info['filefmt']
                else:
                    raise ValueError('Invalid beam file format specified')
                if beam_info['filepathtype'] == 'default':
                    beam_file = prisim_path+'data/beams/' + beam_file
                beam_pol = beam_info['pol']
                beam_id = beam_info['identifier']
                select_beam_freq = beam_info['select_freq']
                if select_beam_freq is None:
                    select_beam_freq = self.f0
                pbeam_spec_interp_method = beam_info['spec_interp']
                if beam_filefmt.lower() == 'fits':
                    extbeam = fits.getdata(beam_file, extname='BEAM_{0}'.format(beam_pol))
                    beam_freqs = fits.getdata(beam_file, extname='FREQS_{0}'.format(beam_pol))
                else:
                    raise ValueError('The external beam file format is currently not supported.')
                extbeam = extbeam.reshape(-1,beam_freqs.size)
                beam_nside = HP.npix2nside(extbeam.shape[0])
                if beam_nside < nside:
                    nside = beam_nside
                theta, phi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
                theta_phi = NP.hstack((theta.reshape(-1,1), phi.reshape(-1,1)))
                if beam_chromaticity:
                    if pbeam_spec_interp_method == 'fft':
                        extbeam = extbeam[:,:-1]
                        beam_freqs = beam_freqs[:-1]
                
                    interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(extbeam), theta_phi=theta_phi, inloc_axis=beam_freqs, outloc_axis=self.f, axis=1, kind=pbeam_spec_interp_method, assume_sorted=True)
                else:
                    nearest_freq_ind = NP.argmin(NP.abs(beam_freqs - select_beam_freq))
                    interp_logbeam = OPS.healpix_interp_along_axis(NP.log10(NP.repeat(extbeam[:,nearest_freq_ind].reshape(-1,1), self.f.size, axis=1)), theta_phi=theta_phi, inloc_axis=self.f, outloc_axis=self.f, axis=1, assume_sorted=True)
                interp_logbeam_max = NP.nanmax(interp_logbeam, axis=0)
                interp_logbeam_max[interp_logbeam_max <= 0.0] = 0.0
                interp_logbeam_max = interp_logbeam_max.reshape(1,-1)
                interp_logbeam = interp_logbeam - interp_logbeam_max
                beam = 10**interp_logbeam
            else:
                theta, phi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
                alt = 90.0 - NP.degrees(theta)
                az = NP.degrees(phi)
                altaz = NP.hstack((alt.reshape(-1,1), az.reshape(-1,1)))
                beam = PB.primary_beam_generator(altaz, self.f, self.ds.ia.telescope, freq_scale='Hz', skyunits='altaz', east2ax1=0.0, pointing_info=None, pointing_center=None)
        else:
            theta, phi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
            alt = 90.0 - NP.degrees(theta)
            az = NP.degrees(phi)
            altaz = NP.hstack((alt.reshape(-1,1), az.reshape(-1,1)))
            beam = PB.primary_beam_generator(altaz, self.f, self.ds.ia.telescope, freq_scale='Hz', skyunits='altaz', east2ax1=0.0, pointing_info=None, pointing_center=None)
            # omega_bw =  self.wl0**2 / NP.mean(self.ds.ia.A_eff) * self.bw

        omega_bw = beam3Dvol(beam, self.f, freq_wts=freq_wts, hemisphere=True)
        return omega_bw

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
        if self.ds.skyvis_lag is not None: self.dps['skyvis'] = NP.abs(self.ds.skyvis_lag)**2 * factor
        if self.ds.vis_lag is not None: self.dps['vis'] = NP.abs(self.ds.vis_lag)**2 * factor
        if self.ds.vis_noise_lag is not None: self.dps['noise'] = NP.abs(self.ds.vis_noise_lag)**2 * factor
        if self.ds.cc_lags is not None:
            if self.ds.cc_skyvis_lag is not None: self.dps['cc_skyvis'] = NP.abs(self.ds.cc_skyvis_lag)**2 * factor
            if self.ds.cc_vis_lag is not None: self.dps['cc_vis'] = NP.abs(self.ds.cc_vis_lag)**2 * factor
            if self.ds.cc_skyvis_res_lag is not None: self.dps['cc_skyvis_res'] = NP.abs(self.ds.cc_skyvis_res_lag)**2 * factor
            if self.ds.cc_vis_res_lag is not None: self.dps['cc_vis_res'] = NP.abs(self.ds.cc_vis_res_lag)**2 * factor
            if self.ds.cc_skyvis_net_lag is not None: self.dps['cc_skyvis_net'] = NP.abs(self.ds.cc_skyvis_net_lag)**2 * factor
            if self.ds.cc_vis_net_lag is not None: self.dps['cc_vis_net'] = NP.abs(self.ds.cc_vis_net_lag)**2 * factor

        if self.ds.subband_delay_spectra:
            for key in self.ds.subband_delay_spectra:
                self.subband_delay_power_spectra[key] = {}
                wl = FCNST.c / self.ds.subband_delay_spectra[key]['freq_center']
                self.subband_delay_power_spectra[key]['z'] = CNST.rest_freq_HI / self.ds.subband_delay_spectra[key]['freq_center'] - 1
                self.subband_delay_power_spectra[key]['dz'] = CNST.rest_freq_HI / self.ds.subband_delay_spectra[key]['freq_center']**2 * self.ds.subband_delay_spectra[key]['bw_eff']
                kprll = NP.empty((self.ds.subband_delay_spectra[key]['freq_center'].size, self.ds.subband_delay_spectra[key]['lags'].size))
                kperp = NP.empty((self.ds.subband_delay_spectra[key]['freq_center'].size, self.bl_length.size))
                horizon_kprll_limits = NP.empty((self.ds.n_acc, self.ds.subband_delay_spectra[key]['freq_center'].size, self.bl_length.size, 2))
                for zind,z in enumerate(self.subband_delay_power_spectra[key]['z']):
                    kprll[zind,:] = self.k_parallel(self.ds.subband_delay_spectra[key]['lags'], z, action='return')
                    kperp[zind,:] = self.k_perp(self.bl_length, z, action='return')
                    horizon_kprll_limits[:,zind,:,:] = self.k_parallel(self.ds.horizon_delay_limits, z, action='return')
                self.subband_delay_power_spectra[key]['kprll'] = kprll
                self.subband_delay_power_spectra[key]['kperp'] = kperp
                self.subband_delay_power_spectra[key]['horizon_kprll_limits'] = horizon_kprll_limits
                self.subband_delay_power_spectra[key]['rz_transverse'] = self.comoving_transverse_distance(self.subband_delay_power_spectra[key]['z'], action='return')
                self.subband_delay_power_spectra[key]['drz_los'] = self.comoving_los_depth(self.ds.subband_delay_spectra[key]['bw_eff'], self.subband_delay_power_spectra[key]['z'], action='return')
                # self.subband_delay_power_spectra[key]['jacobian1'] = NP.mean(self.ds.ia.A_eff) / wl**2 / self.ds.subband_delay_spectra[key]['bw_eff']
                omega_bw = self.beam3Dvol(freq_wts=self.ds.subband_delay_spectra[key]['freq_wts'])
                self.subband_delay_power_spectra[key]['jacobian1'] = 1 / omega_bw
                self.subband_delay_power_spectra[key]['jacobian2'] = self.subband_delay_power_spectra[key]['rz_transverse']**2 * self.subband_delay_power_spectra[key]['drz_los'] / self.ds.subband_delay_spectra[key]['bw_eff']
                self.subband_delay_power_spectra[key]['Jy2K'] = wl**2 * CNST.Jy / (2*FCNST.k)
                self.subband_delay_power_spectra[key]['factor'] = self.subband_delay_power_spectra[key]['jacobian1'] * self.subband_delay_power_spectra[key]['jacobian2'] * self.subband_delay_power_spectra[key]['Jy2K']**2
                conversion_factor = self.subband_delay_power_spectra[key]['factor'].reshape(1,-1,1,1)
                self.subband_delay_power_spectra[key]['skyvis_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['skyvis_lag'])**2 * conversion_factor
                self.subband_delay_power_spectra[key]['vis_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['vis_lag'])**2 * conversion_factor
                if key == 'cc':
                    self.subband_delay_power_spectra[key]['skyvis_res_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['skyvis_res_lag'])**2 * conversion_factor
                    self.subband_delay_power_spectra[key]['vis_res_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['vis_res_lag'])**2 * conversion_factor
                    self.subband_delay_power_spectra[key]['skyvis_net_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['skyvis_net_lag'])**2 * conversion_factor
                    self.subband_delay_power_spectra[key]['vis_net_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['vis_net_lag'])**2 * conversion_factor
                else:
                    self.subband_delay_power_spectra[key]['vis_noise_lag'] = NP.abs(self.ds.subband_delay_spectra[key]['vis_noise_lag'])**2 * conversion_factor

        if self.ds.subband_delay_spectra_resampled:
            for key in self.ds.subband_delay_spectra_resampled:
                self.subband_delay_power_spectra_resampled[key] = {}
                kprll = NP.empty((self.ds.subband_delay_spectra_resampled[key]['freq_center'].size, self.ds.subband_delay_spectra_resampled[key]['lags'].size))
                kperp = NP.empty((self.ds.subband_delay_spectra_resampled[key]['freq_center'].size, self.bl_length.size))
                horizon_kprll_limits = NP.empty((self.ds.n_acc, self.ds.subband_delay_spectra_resampled[key]['freq_center'].size, self.bl_length.size, 2))
                for zind,z in enumerate(self.subband_delay_power_spectra[key]['z']):
                    kprll[zind,:] = self.k_parallel(self.ds.subband_delay_spectra_resampled[key]['lags'], z, action='return')
                    kperp[zind,:] = self.k_perp(self.bl_length, z, action='return')
                    horizon_kprll_limits[:,zind,:,:] = self.k_parallel(self.ds.horizon_delay_limits, z, action='return')
                self.subband_delay_power_spectra_resampled[key]['kprll'] = kprll
                self.subband_delay_power_spectra_resampled[key]['kperp'] = kperp
                self.subband_delay_power_spectra_resampled[key]['horizon_kprll_limits'] = horizon_kprll_limits
                conversion_factor = self.subband_delay_power_spectra[key]['factor'].reshape(1,-1,1,1)
                self.subband_delay_power_spectra_resampled[key]['skyvis_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['skyvis_lag'])**2 * conversion_factor
                self.subband_delay_power_spectra_resampled[key]['vis_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['vis_lag'])**2 * conversion_factor
                if key == 'cc':
                    self.subband_delay_power_spectra_resampled[key]['skyvis_res_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['skyvis_res_lag'])**2 * conversion_factor
                    self.subband_delay_power_spectra_resampled[key]['vis_res_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['vis_res_lag'])**2 * conversion_factor
                    self.subband_delay_power_spectra_resampled[key]['skyvis_net_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['skyvis_net_lag'])**2 * conversion_factor
                    self.subband_delay_power_spectra_resampled[key]['vis_net_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['vis_net_lag'])**2 * conversion_factor
                else:
                    self.subband_delay_power_spectra_resampled[key]['vis_noise_lag'] = NP.abs(self.ds.subband_delay_spectra_resampled[key]['vis_noise_lag'])**2 * conversion_factor

    ############################################################################

    def compute_power_spectrum_allruns(self, dspec, subband=False):

        """
        ------------------------------------------------------------------------
        Compute delay power spectrum in units of K^2 (Mpc/h)^3 from the delay
        spectrum in units of Jy Hz from multiple runs of visibilities

        Inputs:

        dspec   [dictionary] Delay spectrum information. If subband is set to
                False, it contains the keys 'vislag1' and maybe 'vislag2' 
                (optional). If subband is set to True, it must contain these
                keys as well - 'lags', 'freq_center', 'bw_eff', 'freq_wts' as
                well. The value under these keys are described below:
                'vislag1'   [numpy array] subband delay spectra of first set of
                            visibilities. It is of size 
                            n_win x (n1xn2x... n_runs dims) x n_bl x nlags x n_t
                            if subband is set to True or of shape 
                            (n1xn2x... n_runs dims) x n_bl x nlags x n_t if 
                            subband is set to False
                            It must be specified independent of subband value
                'vislag2'   [numpy array] subband delay spectra of second set of
                            visibilities (optional). If not specified, value 
                            under key 'vislag1' is copied under this key and 
                            auto-delay spectrum is computed. If explicitly 
                            specified, it must be of same shape as value under 
                            'vislag1' and cross-delay spectrum will be computed. 
                            It is of size 
                            n_win x (n1xn2x... n_runs dims) x n_bl x nlags x n_t
                            if subband is set to True or of shape 
                            (n1xn2x... n_runs dims) x n_bl x nlags x n_t if 
                            subband is set to False. It is applicable 
                            independent of value of input subband
                'lags'      [numpy array] Contains the lags in the delay 
                            spectrum. Applicable only if subband is set to True.
                            It is of size nlags
                'freq_center'
                            [numpy array] frequency centers (in Hz) of the 
                            selected frequency windows for subband delay 
                            transform of visibilities. The values can be a 
                            scalar, list or numpy array. Applicable only if
                            subband is set to True. It is of size n_win
                'bw_eff'    [scalar, list or numpy array] effective bandwidths 
                            (in Hz) on the selected frequency windows for 
                            subband delay transform of visibilities. The values 
                            can be a scalar, list or numpy array. Applicable 
                            only if subband is set to True. It is of size n_win
                'freq_wts'  [numpy array] Contains frequency weights applied 
                            on each frequency sub-band during the subband delay 
                            transform. It is of size n_win x nchan. Applicable
                            only if subband is set to True.

        subband [boolean] If set to False (default), the entire band is used in
                          determining the delay power spectrum and only value
                          under key 'vislag1' and optional key 'vislag2' in 
                          input dspec is required. If set to True, delay pwoer
                          spectrum in specified subbands is determined. In 
                          addition to key 'vislag1' and optional key 'vislag2', 
                          following keys are also required in input dictionary 
                          dspec, namely, 'lags', 'freq_center', 'bw_eff', 
                          'freq_wts'

        Output:

        Dictionary containing delay power spectrum (in units of K^2 (Mpc/h)^3) 
        of shape (n1xn2x... n_runs dims) x n_bl x nlags x n_t under key 
        'fullband' if subband is set to False or of shape 
        n_win x (n1xn2x... n_runs dims) x n_bl x nlags x n_t under key 'subband'
        if subband is set to True. 
        ------------------------------------------------------------------------
        """

        try:
            dspec
        except NameError:
            raise NameError('Input dspec must be specified')

        if not isinstance(dspec, dict):
            raise TypeError('Input dspec must be a dictionary')
        else:
            mode = 'auto'
            if 'vislag1' not in dspec:
                raise KeyError('Key "vislag1" not found in input dspec')
            if not isinstance(dspec['vislag1'], NP.ndarray):
                raise TypeError('Value under key "vislag1" must be a numpy array')
            if 'vislag2' not in dspec:
                dspec['vislag2'] = dspec['vislag1']
            else:
                mode = 'cross'
            if not isinstance(dspec['vislag2'], NP.ndarray):
                raise TypeError('Value under key "vislag2" must be a numpy array')
            if dspec['vislag1'].shape != dspec['vislag2'].shape:
                raise ValueError('Value under keys "vislag1" and "vislag2" must have same shape')

        if not isinstance(subband, bool):
            raise TypeError('Input subband must be boolean')

        dps = {}
        if not subband:
            factor = self.jacobian1 * self.jacobian2 * self.Jy2K**2 # scalar
            factor = factor.reshape(tuple(NP.ones(dspec['vislag1'].ndim, dtype=NP.int)))
            key = 'fullband'
        else:
            dspec['freq_center'] = NP.asarray(dspec['freq_center']).ravel() # n_win
            dspec['bw_eff'] = NP.asarray(dspec['bw_eff']).ravel() # n_win
            wl = FCNST.c / dspec['freq_center'] # n_win
            redshift = CNST.rest_freq_HI / dspec['freq_center'] - 1 # n_win
            dz = CNST.rest_freq_HI / dspec['freq_center']**2 * dspec['bw_eff'] # n_win
            kprll = NP.empty((dspec['freq_center'].size, dspec['lags'].size)) # n_win x nlags
            kperp = NP.empty((dspec['freq_center'].size, self.bl_length.size)) # n_win x nbl
            for zind,z in enumerate(redshift):
                kprll[zind,:] = self.k_parallel(dspec['lags'], z, action='return')
                kperp[zind,:] = self.k_perp(self.bl_length, z, action='return')
            rz_transverse = self.comoving_transverse_distance(redshift, action='return') # n_win
            drz_los = self.comoving_los_depth(dspec['bw_eff'], redshift, action='return') # n_win
            omega_bw = self.beam3Dvol(freq_wts=NP.squeeze(dspec['freq_wts'])) 
            jacobian1 = 1 / omega_bw # n_win
            jacobian2 = rz_transverse**2 * drz_los / dspec['bw_eff'] # n_win
            Jy2K = wl**2 * CNST.Jy / (2*FCNST.k) # n_win
            factor = jacobian1 * jacobian2 * Jy2K**2 # n_win
            factor = factor.reshape((-1,)+tuple(NP.ones(dspec['vislag1'].ndim-1, dtype=NP.int)))
            key = 'subband'
        dps[key] = dspec['vislag1'] * dspec['vislag2'].conj() * factor
        dps[key] = dps[key].real
        if mode == 'cross':
            dps[key] *= 2
        return dps

    ############################################################################

    def compute_individual_closure_phase_power_spectrum(self, closure_phase_delay_spectra):

        """
        ------------------------------------------------------------------------
        Compute delay power spectrum of closure phase in units of K^2 (Mpc/h)^3 
        from the delay spectrum in units of Jy Hz where the original visibility
        amplitudes of closure phase complex exponents are assumed to be 1 Jy 
        across the band

        Inputs:

        closure_phase_delay_spectra
        [dictionary] contains information about closure phase delay spectra of 
        different frequency sub-bands (n_win in number) under the following 
        keys:
        'antenna_triplets'
                    [list of tuples] List of antenna ID triplets where each 
                    triplet is given as a tuple. Closure phase delay spectra in
                    subbands is computed for each of these antenna triplets
        'baseline_triplets'     
                    [numpy array] List of 3x3 numpy arrays. Each 3x3
                    unit in the list represents triplets of baseline
                    vectors where the three rows denote the three 
                    baselines in the triplet and the three columns 
                    define the x-, y- and z-components of the 
                    triplet. The number of 3x3 unit elements in the 
                    list will equal the number of elements in the 
                    list under key 'antenna_triplets'. Closure phase delay 
                    spectra in subbands is computed for each of these baseline
                    triplets which correspond to the antenna triplets
        'freq_center' 
                    [numpy array] contains the center frequencies 
                    (in Hz) of the frequency subbands of the subband
                    delay spectra. It is of size n_win. It is roughly 
                    equivalent to redshift(s)
        'bw_eff'    [numpy array] contains the effective bandwidths 
                    (in Hz) of the subbands being delay transformed. It
                    is of size n_win. It is roughly equivalent to width 
                    in redshift or along line-of-sight
        'lags'      [numpy array] lags of the resampled subband delay spectra 
                    after padding in frequency during the transform. It
                    is of size nlags where nlags is the number of 
                    independent delay bins
        'lag_kernel'
                    [numpy array] delay transform of the frequency 
                    weights under the key 'freq_wts'. It is of size
                    n_bl x n_win x nlags x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the resampled subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the effective bandwidth 
        'closure_phase_skyvis' (optional)
                    [numpy array] subband delay spectra of closure phases
                    of noiseless sky visiblities from the specified 
                    antenna triplets. It is of size n_triplets x n_win x 
                    nlags x n_t. It must be in units of Jy Hz.
        'closure_phase_vis' (optional)
                    [numpy array] subband delay spectra of closure phases
                    of noisy sky visiblities from the specified antenna 
                    triplets. It is of size n_triplets x n_win x nlags x n_t.
                    It must be in units of Jy Hz.
        'closure_phase_noise' (optional)
                    [numpy array] subband delay spectra of closure phases
                    of noise visiblities from the specified antenna triplets.
                    It is of size n_triplets x n_win x nlags x n_t. It must be 
                    in units of Jy Hz.
        
        Output:

        Dictionary with closure phase delay power spectra containing the 
        following keys and values:
        'z'         [numpy array] Redshifts corresponding to the centers of the
                    frequency subbands. Same size as number of values under key
                    'freq_center' which is n_win
        'kprll'     [numpy array] k_parallel (h/Mpc) for different subbands and
                    various delays. It is of size n_win x nlags
        'kperp'     [numpy array] k_perp (h/Mpc) for different subbands and the
                    antenna/baseline triplets. It is of size n_win x n_triplets
                    x 3 x 3 where the 3 x 3 refers to 3 different baselines and 
                    3 components of the baseline vector respectively
        'horizon_kprll_limits' 
                    [numpy array] limits on k_parallel corresponding to limits 
                    on horizon delays for each of the baseline triplets and 
                    subbands. It is of shape n_t x n_win x n_triplets x 3 x 2, 
                    where 3 is for the three baselines involved in the triplet, 
                    2 limits (upper and lower). It has units of h/Mpc
        'closure_phase_skyvis'
                    [numpy array] subband delay power spectra of closure phases
                    of noiseless sky visiblities from the specified 
                    antenna triplets. It is of size n_triplets x n_win x 
                    nlags x n_t. It is in units of K^2 (Mpc/h)^3. This is 
                    returned if this key is present in the input 
                    closure_phase_delay_spectra
        'closure_phase_vis'
                    [numpy array] subband delay power spectra of closure phases
                    of noisy sky visiblities from the specified antenna 
                    triplets. It is of size n_triplets x n_win x nlags x n_t.
                    It is in units of K^2 (Mpc/h)^3. This is returned if this 
                    key is present in the input closure_phase_delay_spectra
        'closure_phase_noise'
                    [numpy array] subband delay power spectra of closure phases
                    of noise visiblities from the specified antenna triplets.
                    It is of size n_triplets x n_win x nlags x n_t. It is in 
                    units of K^2 (Mpc/h)^3. This is returned if this key is 
                    present in the input closure_phase_delay_spectra
        ------------------------------------------------------------------------
        """

        try:
            closure_phase_delay_spectra
        except NameError:
            raise NameError('Input closure_phase_delay_spectra must be provided')

        closure_phase_delay_power_spectra = {}
        wl = FCNST.c / closure_phase_delay_spectra['freq_center']
        z = CNST.rest_freq_HI / closure_phase_delay_spectra['freq_center'] - 1
        dz = CNST.rest_freq_HI / closure_phase_delay_spectra['freq_center']**2 * closure_phase_delay_spectra['bw_eff']
        kprll = NP.empty((closure_phase_delay_spectra['freq_center'].size, closure_phase_delay_spectra['lags'].size))
        kperp = NP.empty((closure_phase_delay_spectra['freq_center'].size, len(closure_phase_delay_spectra['antenna_triplets']), 3)) # n_win x n_triplets x 3, where 3 is for the three baselines involved
        horizon_kprll_limits = NP.empty((self.ds.n_acc, closure_phase_delay_spectra['freq_center'].size, len(closure_phase_delay_spectra['antenna_triplets']), 3, 2)) # n_t x n_win x n_triplets x 3 x 2, where 3 is for the three baselines involved

        for zind,redshift in enumerate(z):
            kprll[zind,:] = self.k_parallel(closure_phase_delay_spectra['lags'], redshift, action='return')
            for triplet_ind, ant_triplet in enumerate(closure_phase_delay_spectra['antenna_triplets']):
                bl_lengths = NP.sqrt(NP.sum(closure_phase_delay_spectra['baseline_triplets'][triplet_ind]**2, axis=1))
                kperp[zind,triplet_ind,:] = self.k_perp(bl_lengths, redshift, action='return')
                horizon_delay_limits = bl_lengths.reshape(1,-1,1) / FCNST.c # 1x3x1, where 1 phase center, 3 is for the three baselines involved in the triplet, 1 upper limit
                horizon_delay_limits = NP.concatenate((horizon_delay_limits, -horizon_delay_limits), axis=2) # 1x3x2, where 1 phase center, 3 is for the three baselines involved in the triplet, 2 limits (upper and lower)
                horizon_kprll_limits[:,zind,triplet_ind,:,:] = self.k_parallel(horizon_delay_limits, redshift, action='return') # 1 x n_win x n_triplets x 3 x 2, where 1 phase center, 3 is for the three baselines involved in the triplet, 2 limits (upper and lower)
        
        closure_phase_delay_power_spectra['z'] = z
        closure_phase_delay_power_spectra['kprll'] = kprll
        closure_phase_delay_power_spectra['kperp'] = kperp
        closure_phase_delay_power_spectra['horizon_kprll_limits'] = horizon_kprll_limits
        rz_transverse = self.comoving_transverse_distance(closure_phase_delay_power_spectra['z'], action='return')
        drz_los = self.comoving_los_depth(closure_phase_delay_spectra['bw_eff'], closure_phase_delay_power_spectra['z'], action='return')
        omega_bw = self.beam3Dvol(freq_wts=closure_phase_delay_spectra['freq_wts'])
        jacobian1 = 1 / omega_bw
        jacobian2 = rz_transverse**2 * drz_los / closure_phase_delay_spectra['bw_eff']
        Jy2K = wl**2 * CNST.Jy / (2*FCNST.k)
        factor = jacobian1 * jacobian2 * Jy2K**2
        conversion_factor = factor.reshape(1,-1,1,1)
        for key in ['closure_phase_skyvis', 'closure_phase_vis', 'closure_phase_noise']:
            if key in closure_phase_delay_spectra:
                closure_phase_delay_power_spectra[key] = NP.abs(closure_phase_delay_spectra[key])**2 * conversion_factor

        return closure_phase_delay_power_spectra

    ############################################################################

    def compute_averaged_closure_phase_power_spectrum(self, closure_phase_delay_spectra):

        """
        ------------------------------------------------------------------------
        Compute delay power spectrum of closure phase in units of K^2 (Mpc/h)^3 
        from the delay spectrum in units of Jy Hz and average over 'auto' and 
        'cross' modes, where the original visibility amplitudes of closure phase 
        complex exponents are assumed to be 1 Jy across the band

        Inputs:

        closure_phase_delay_spectra
        [dictionary] contains information about closure phase delay spectra of 
        different frequency sub-bands (n_win in number) under the following 
        keys:
        'antenna_triplets'
                    [list of tuples] List of antenna ID triplets where each 
                    triplet is given as a tuple. Closure phase delay spectra in
                    subbands is computed for each of these antenna triplets
        'baseline_triplets'     
                    [numpy array] List of 3x3 numpy arrays. Each 3x3
                    unit in the list represents triplets of baseline
                    vectors where the three rows denote the three 
                    baselines in the triplet and the three columns 
                    define the x-, y- and z-components of the 
                    triplet. The number of 3x3 unit elements in the 
                    list will equal the number of elements in the 
                    list under key 'antenna_triplets'. Closure phase delay 
                    spectra in subbands is computed for each of these baseline
                    triplets which correspond to the antenna triplets
        'freq_center' 
                    [numpy array] contains the center frequencies 
                    (in Hz) of the frequency subbands of the subband
                    delay spectra. It is of size n_win. It is roughly 
                    equivalent to redshift(s)
        'bw_eff'    [numpy array] contains the effective bandwidths 
                    (in Hz) of the subbands being delay transformed. It
                    is of size n_win. It is roughly equivalent to width 
                    in redshift or along line-of-sight
        'lags'      [numpy array] lags of the resampled subband delay spectra 
                    after padding in frequency during the transform. It
                    is of size nlags where nlags is the number of 
                    independent delay bins
        'lag_kernel'
                    [numpy array] delay transform of the frequency 
                    weights under the key 'freq_wts'. It is of size
                    n_bl x n_win x nlags x n_t.
        'lag_corr_length' 
                    [numpy array] It is the correlation timescale (in 
                    pixels) of the resampled subband delay spectra. It is 
                    proportional to inverse of effective bandwidth. It
                    is of size n_win. The unit size of a pixel is 
                    determined by the difference between adjacent pixels 
                    in lags under key 'lags' which in turn is 
                    effectively inverse of the effective bandwidth 
        'closure_phase_skyvis' (optional)
                    [numpy array] subband delay spectra of closure phases
                    of noiseless sky visiblities from the specified 
                    antenna triplets. It is of size n_triplets x n_win x 
                    nlags x n_t. It must be in units of Jy Hz.
        'closure_phase_vis' (optional)
                    [numpy array] subband delay spectra of closure phases
                    of noisy sky visiblities from the specified antenna 
                    triplets. It is of size n_triplets x n_win x nlags x n_t.
                    It must be in units of Jy Hz.
        'closure_phase_noise' (optional)
                    [numpy array] subband delay spectra of closure phases
                    of noise visiblities from the specified antenna triplets.
                    It is of size n_triplets x n_win x nlags x n_t. It must be 
                    in units of Jy Hz.
        
        Output:

        Dictionary with closure phase delay power spectra containing the 
        following keys and values:
        'z'         [numpy array] Redshifts corresponding to the centers of the
                    frequency subbands. Same size as number of values under key
                    'freq_center' which is n_win
        'kprll'     [numpy array] k_parallel (h/Mpc) for different subbands and
                    various delays. It is of size n_win x nlags
        'kperp'     [numpy array] k_perp (h/Mpc) for different subbands and the
                    antenna/baseline triplets. It is of size n_win x n_triplets
                    x 3 x 3 where the 3 x 3 refers to 3 different baselines and 
                    3 components of the baseline vector respectively
        'horizon_kprll_limits' 
                    [numpy array] limits on k_parallel corresponding to limits 
                    on horizon delays for each of the baseline triplets and 
                    subbands. It is of shape n_t x n_win x n_triplets x 3 x 2, 
                    where 3 is for the three baselines involved in the triplet, 
                    2 limits (upper and lower). It has units of h/Mpc
        'auto'      [dictionary] average of diagonal terms in the power spectrum
                    matrix with possibly the following keys and values:
                    'closure_phase_skyvis'
                          [numpy array] subband delay power spectra of closure 
                          phases of noiseless sky visiblities from the specified 
                          antenna triplets. It is of size n_triplets x n_win x 
                          nlags x n_t. It is in units of K^2 (Mpc/h)^3. This is 
                          returned if this key is present in the input 
                          closure_phase_delay_spectra
                    'closure_phase_vis'
                          [numpy array] subband delay power spectra of closure 
                          phases of noisy sky visiblities from the specified 
                          antenna triplets. It is of size 
                          1 x n_win x nlags x n_t. It is in units of 
                          K^2 (Mpc/h)^3. This is returned if this key is present 
                          in the input closure_phase_delay_spectra
                    'closure_phase_noise'
                          [numpy array] subband delay power spectra of closure 
                          phases of noise visiblities from the specified antenna 
                          triplets. It is of size 1 x n_win x nlags x n_t. It is 
                          in units of K^2 (Mpc/h)^3. This is returned if this 
                          key is present in the input 
                          closure_phase_delay_spectra
        'cross'     [dictionary] average of off-diagonal terms in the power 
                    spectrum matrix with possibly the following keys and values:
                    'closure_phase_skyvis'
                          [numpy array] subband delay power spectra of closure 
                          phases of noiseless sky visiblities from the specified 
                          antenna triplets. It is of size n_triplets x n_win x 
                          nlags x n_t. It is in units of K^2 (Mpc/h)^3. This is 
                          returned if this key is present in the input 
                          closure_phase_delay_spectra
                    'closure_phase_vis'
                          [numpy array] subband delay power spectra of closure 
                          phases of noisy sky visiblities from the specified 
                          antenna triplets. It is of size 
                          1 x n_win x nlags x n_t. It is in units of 
                          K^2 (Mpc/h)^3. This is returned if this key is present 
                          in the input closure_phase_delay_spectra
                    'closure_phase_noise'
                          [numpy array] subband delay power spectra of closure 
                          phases of noise visiblities from the specified antenna 
                          triplets. It is of size 1 x n_win x nlags x n_t. It is 
                          in units of K^2 (Mpc/h)^3. This is returned if this 
                          key is present in the input 
                          closure_phase_delay_spectra
        ------------------------------------------------------------------------
        """

        try:
            closure_phase_delay_spectra
        except NameError:
            raise NameError('Input closure_phase_delay_spectra must be provided')

        closure_phase_delay_power_spectra = {}
        wl = FCNST.c / closure_phase_delay_spectra['freq_center']
        z = CNST.rest_freq_HI / closure_phase_delay_spectra['freq_center'] - 1
        dz = CNST.rest_freq_HI / closure_phase_delay_spectra['freq_center']**2 * closure_phase_delay_spectra['bw_eff']
        kprll = NP.empty((closure_phase_delay_spectra['freq_center'].size, closure_phase_delay_spectra['lags'].size))
        kperp = NP.empty((closure_phase_delay_spectra['freq_center'].size, len(closure_phase_delay_spectra['antenna_triplets']), 3)) # n_win x n_triplets x 3, where 3 is for the three baselines involved
        horizon_kprll_limits = NP.empty((self.ds.n_acc, closure_phase_delay_spectra['freq_center'].size, len(closure_phase_delay_spectra['antenna_triplets']), 3, 2)) # n_t x n_win x n_triplets x 3 x 2, where 3 is for the three baselines involved

        for zind,redshift in enumerate(z):
            kprll[zind,:] = self.k_parallel(closure_phase_delay_spectra['lags'], redshift, action='return')
            for triplet_ind, ant_triplet in enumerate(closure_phase_delay_spectra['antenna_triplets']):
                bl_lengths = NP.sqrt(NP.sum(closure_phase_delay_spectra['baseline_triplets'][triplet_ind]**2, axis=1))
                kperp[zind,triplet_ind,:] = self.k_perp(bl_lengths, redshift, action='return')
                horizon_delay_limits = bl_lengths.reshape(1,-1,1) / FCNST.c # 1x3x1, where 1 phase center, 3 is for the three baselines involved in the triplet, 1 upper limit
                horizon_delay_limits = NP.concatenate((horizon_delay_limits, -horizon_delay_limits), axis=2) # 1x3x2, where 1 phase center, 3 is for the three baselines involved in the triplet, 2 limits (upper and lower)
                horizon_kprll_limits[:,zind,triplet_ind,:,:] = self.k_parallel(horizon_delay_limits, redshift, action='return') # 1 x n_win x n_triplets x 3 x 2, where 1 phase center, 3 is for the three baselines involved in the triplet, 2 limits (upper and lower)
        
        closure_phase_delay_power_spectra['z'] = z
        closure_phase_delay_power_spectra['kprll'] = kprll
        closure_phase_delay_power_spectra['kperp'] = kperp
        closure_phase_delay_power_spectra['horizon_kprll_limits'] = horizon_kprll_limits
        rz_transverse = self.comoving_transverse_distance(closure_phase_delay_power_spectra['z'], action='return')
        drz_los = self.comoving_los_depth(closure_phase_delay_spectra['bw_eff'], closure_phase_delay_power_spectra['z'], action='return')
        omega_bw = self.beam3Dvol(freq_wts=closure_phase_delay_spectra['freq_wts'])
        jacobian1 = 1 / omega_bw
        jacobian2 = rz_transverse**2 * drz_los / closure_phase_delay_spectra['bw_eff']
        Jy2K = wl**2 * CNST.Jy / (2*FCNST.k)
        factor = jacobian1 * jacobian2 * Jy2K**2
        for key in ['closure_phase_skyvis', 'closure_phase_vis', 'closure_phase_noise']:
            if key in closure_phase_delay_spectra:
                ndim_shape = NP.ones(closure_phase_delay_spectra[key].ndim, dtype=int)
                ndim_shape[-3] = -1
                ndim_shape = tuple(ndim_shape)
                conversion_factor = factor.reshape(ndim_shape)

        for mode in ['auto', 'cross']:
            closure_phase_delay_power_spectra[mode] = {}
            for key in ['closure_phase_skyvis', 'closure_phase_vis', 'closure_phase_noise']:
                if key in closure_phase_delay_spectra:
                    nruns = closure_phase_delay_spectra[key].shape[0]
                    if mode == 'auto':
                        closure_phase_delay_power_spectra[mode][key] = NP.mean(NP.abs(closure_phase_delay_spectra[key])**2, axis=0, keepdims=True) * conversion_factor
                    else:
                        closure_phase_delay_power_spectra[mode][key] = 1.0 / (nruns*(nruns-1)) * (NP.abs(NP.sum(closure_phase_delay_spectra[key], axis=0, keepdims=True))**2 - nruns * closure_phase_delay_power_spectra['auto'][key]) * conversion_factor

        return closure_phase_delay_power_spectra

    ############################################################################

