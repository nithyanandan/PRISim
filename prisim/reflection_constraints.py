import numpy as NP
import yaml, argparse, h5py, warnings
from astropy.io import fits
import astropy.cosmology as CP
import scipy.constants as FCNST
from scipy import interpolate
from astroutils import constants as CNST
from prisim import interferometry as RI
from prisim import delay_spectrum as DS

cosmoPlanck15 = CP.Planck15 # Planck 2015 cosmology
cosmo100 = cosmoPlanck15.clone(name='Modified Planck 2015 cosmology with h=1.0', H0=100.0) # Modified Planck 2015 cosmology with h=1.0, H= 100 km/s/Mpc

################################################################################

def parsefile(parmsfile):
    with open(parmsfile, 'r') as fileobj:
        parms = yaml.safe_load(fileobj)
    return parms

################################################################################

def estimate(parms, action='return'):
    projectdir = parms['dirstruct']['projectdir']
    subdir = projectdir + parms['dirstruct']['subdir']
    simdir = subdir + parms['dirstruct']['simdir'] + 'simdata/'
    simfile = simdir + 'simvis'
    analysisdir = projectdir + parms['dirstruct']['analysisdir']
    outfile = analysisdir + parms['dirstruct']['outfile']
    figdir = analysisdir + parms['dirstruct']['figdir']

    processinfo = parms['processing']
    subbandinfo = processinfo['subband']
    freq_window_centers = {key: NP.asarray(subbandinfo['freq_center']) for key in ['cc', 'sim']}
    freq_window_bw = {key: NP.asarray(subbandinfo['bw_eff']) for key in ['cc', 'sim']}
    freq_window_shape = {key: subbandinfo['shape'] for key in ['cc', 'sim']}
    freq_window_fftpow = {key: subbandinfo['fftpow'] for key in ['cc', 'sim']}
    pad = {key: 1.0 for key in ['cc', 'sim']}
    redshifts = NP.asarray(processinfo['redshifts'])
    kprll_cut = NP.asarray(processinfo['kprll_cut'])
    bll = processinfo['bllength']
    max_delay = processinfo['max_delay']
    if max_delay is not None:
        if not isinstance(max_delay, (int,float)):
            raise TypeError('Input max_delay must be a scalar')
        max_delay *= 1e-6 # in seconds from microseconds
        if max_delay < 0.0:
            max_delay = None
        
    lst_range = processinfo['lst_range']
    if lst_range is not None:
        lst_range = NP.asarray(lst_range) # in hours
        lst_range = lst_range.reshape(-1,2)

    lidz_model = parms['eorparm']['lidz_model']
    model_21cmfast = parms['eorparm']['21cmfast_model']
    if lidz_model:
        lidz_modelfile = parms['eorparm']['lidz_modelfile']
    if model_21cmfast:
        modelfile_21cmfast = parms['eorparm']['21cmfast_modelfile']

    if lidz_model:
        hdulist = fits.open(lidz_modelfile)
        eor_model_freqs = hdulist['FREQUENCY'].data
        lidz_eor_model_redshifts = hdulist['REDSHIFT'].data
        eor_modelinfo = [hdulist['{0}'.format(i)].data for i in range(eor_model_freqs.size)]
        lidz_eor_model_k = NP.asarray([modelinfo[:,0] for modelinfo in eor_modelinfo])
        lidz_eor_model_Pk = [modelinfo[:,1] for modelinfo in eor_modelinfo]
        lidz_eor_model_Pk = NP.asarray(lidz_eor_model_Pk)
    
    if model_21cmfast:
        hdulist = fits.open(modelfile_21cmfast)
        eor_model_freqs = hdulist['FREQUENCY'].data
        eor_21cmfast_model_redshifts = hdulist['REDSHIFT'].data
        eor_modelinfo = [hdulist['{0}'.format(i)].data for i in range(eor_model_freqs.size)]
        eor_21cmfast_model_k = NP.asarray([modelinfo[:,0] for modelinfo in eor_modelinfo])
        eor_21cmfast_model_Pk = [modelinfo[:,1] for modelinfo in eor_modelinfo]
        eor_21cmfast_model_Pk = NP.asarray(eor_21cmfast_model_Pk)

    sim = RI.InterferometerArray(None, None, None, init_file=simfile)
    dsobj = DS.DelaySpectrum(interferometer_array=sim)
    dspec = dsobj.delay_transform(action='store')
    subband_DSpec = dsobj.subband_delay_transform(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow , pad=pad, bpcorrect=False, action='return_oversampled')
    dpsobj = DS.DelayPowerSpectrum(dsobj)
    dpsobj.compute_power_spectrum()

    attninfo = {'lidz': {}, '21cmfast': {}}
    for key in attninfo:
        if key == 'lidz':
            eormodel = {'z': lidz_eor_model_redshifts, 'k': lidz_eor_model_k, 'PS': lidz_eor_model_Pk}
        elif key == '21cmfast':
            eormodel = {'z': eor_21cmfast_model_redshifts, 'k': eor_21cmfast_model_k, 'PS': eor_21cmfast_model_Pk}
        else:
            raise ValueError('Specified EoR model not currently supported')
        attninfo[key] = reflection_attenuation(dpsobj, eormodel, redshifts, kprll_cut, bll, max_delay=max_delay, lst_range=lst_range)

    if action == 'save':
        with h5py.File(outfile, 'w') as fobj:
            for key in attninfo:
                for subkey in ['z', 'kprll_cut', 'delays', 'attenuation']:
                    dset = fobj.create_dataset('{0}/{1}'.format(key,subkey), data=attninfo[key][subkey])
                    if subkey == 'kprll_cut':
                        dset.attrs['units'] = 'h/Mpc'
                    elif subkey == 'delays':
                        dset.attrs['units'] = 's'
                    elif subkey == 'attenuation':
                        dset.attrs['units'] = 'dB'
    else:
        return attninfo

################################################################################

def reflection_attenuation(dps, modelPSinfo, redshifts, kprll_cut, bll,
                           max_delay=None, lst_range=None, cosmo=cosmo100):

    """
    ----------------------------------------------------------------------------
    Produces antenna reflection-attenuation performance as a function of 
    redshift, kprll_thresholds, and delays if provided with an EoR model and
    foreground delay power spectrum

    Inputs:

    dps         [instance of class DelayPowerSpectrum] Instance of class
                DelayPowerSpectrum containing foreground power spectrum

    modelPSinfo [dictionary] Consists of information on the EoR model. It has
                the following keys and values:
                'z'     [numpy array] redshifts, shape=(n_z_model,)
                'k'     [numpy array] cosmological comoving wavenumber (h/Mpc).
                        Shape=(n_z_model,n_k_model)
                'PS'    [numpy array] EoR model power spectrum (in units of 
                        K^2 (Mpc/h)^3). Shape=(n_z_model,n_k_model), same as 
                        value under key 'k'

    redshifts   [numpy array] Redshifts at which the EoR model and reflection 
                constraints are to be estimated. Shape=(nz,)

    kprll_cut   [numpy array] Thresholds in k_prll (in h/Mpc units) above which
                the EoR power spectrum must be detectable despite the 
                reflections. Shape=(n_kpl_cut,)

    bll         [scalar (int or float)] Baseline length (in m) to select for 
                estimating reflection constraints. Foreground power spectra
                on baseline vectors with this length will be averaged.

    max_delay   [scalar (int or float)] Maximum delay (in s) for which 
                reflection constraints are to be evaluated. If set to None
                (default), it is set to maximum delay in the EoR model minus
                the maximum delay in the foreground model

    lst_range   [numpy array] LST range (in hours) to average the foreground
                power spectra before estimating reflection constraints.

    cosmo       [instance of cosmology class from astropy] An instance of class
                FLRW or default_cosmology of astropy cosmology module. Default
                uses Planck 2015 cosmology with H0=100 h km/s/Mpc

    Output:

    Dictionary containing information in reflection-attenuation constraints. It
    has the following keys and values:
    'z'         [numpy array] Redshifts at which reflection-attentuation 
                constraints have been evaluated. Its values may not be identical
                to the input redshifts because of the nearest-neighbor 
                determination based on the redshifts provided in the EoR model.
                Shape=(nz,)
    'kprll_cut' [numpy array] k_parallel thresholds above which the reflection
                constraints have been evaluated for in order to detect the EoR
                power spectrum in all kprll > kprll_cut despite the reflections.
                It is identical to input kprll_cut. Shape=(n_kpl_cut,)
    'delays'    [numpy array] Delays (in s) at which reflection attenuation 
                constraints have been estimated. It ranges from 0 to max_delay.
                Shape=(n_delays,).
    'attenuation' 
                [numpy array] Reflection attenuation constraints (in dB). It has
                shape=(nz,n_kpl_cut,n_delays)
    ----------------------------------------------------------------------------
    """

    if not isinstance(dps, DS.DelayPowerSpectrum):
        raise TypeError('Input dps must be an instance of class DelayPowerSpectrum')

    if not isinstance(modelPSinfo, dict):
        raise TypeError('Input modelPSinfo must be a dictionary')

    if not isinstance(redshifts, NP.ndarray):
        raise TypeError('Input redshifts must be a numpy array')
    redshifts = redshifts.ravel()
    if NP.any(redshifts < 0.0):
        raise ValueError('Input redshifts cannot be negative')

    if not isinstance(kprll_cut, NP.ndarray):
        raise TypeError('Input kprll_cut must be a numpy array')
    kprll_cut = kprll_cut.ravel()

    if not isinstance(bll, (int,float)):
        raise TypeError('Input bll (baseline length) must be a scalar')
    bll_ind = NP.where(NP.abs(dps.bl_length - bll) <= 1e-10)[0]

    if lst_range is not None:
        if not isinstance(lst_range, NP.ndarray):
            raise TypeError('Input lst_range must be a numpy array')
        if lst_range.size == 2:
            lst_range = lst_range.reshape(1,-1)
        elif lst_range.ndim != 2:
            raise ValueError('Input lst_range must be a 2D numpy array')
        else:
            if lst_range.shape[1] != 2:
                raise ValueError('Input lst_range must be a nlst x 2 array')
        if NP.any(lst_range[:,1] < lst_range[:,0]):
            raise ValueError('Input lst_range must be in the order (min,max)')
        lst_range = 15.0 * lst_range
        lst_ind = []
        for ind in range(lst_range.shape[0]):
            lst_ind += NP.where(NP.logical_and(dps.ds.ia.lst >= lst_range[ind,0], dps.ds.ia.lst <= lst_range[ind,1]))[0].tolist()
        lst_ind = NP.asarray(lst_ind)
    else:
        lst_ind = NP.arange(dps.ds.ia.lst.size)

    fgmdl_power_shape = dps.subband_delay_power_spectra['sim']['skyvis_lag'].shape # (nbl,nspw,nlags,nlst)
    if fgmdl_power_shape[1] > 1:
        warnings.warn('Multiple spectral windows found. Only the first will be used as reference for foreground power spectra')
    zind_FG = 0
    fgmdl_power = NP.mean(dps.subband_delay_power_spectra['sim']['skyvis_lag'][NP.ix_(bll_ind,NP.arange(fgmdl_power_shape[1]),NP.arange(fgmdl_power_shape[2]),lst_ind)], axis=(0,3)) # shape=(nspw,nlags)
    fgmdl_power_upper = fgmdl_power[:,fgmdl_power.shape[1]//2 + 1:] # shape=(nspw,nlags_upper)
    fgmdl_power_lower = fgmdl_power[:,:fgmdl_power.shape[1]//2 + 1] # shape=(nspw,nlags_lower)
    fgpow = NP.copy(fgmdl_power_lower) # shape=(nspw,nlags_lower)
    fgpow[:,fgpow.shape[1]-1-fgmdl_power_upper.shape[1]:fgpow.shape[1]-1] = 0.5 * (fgpow[:,fgpow.shape[1]-1-fgmdl_power_upper.shape[1]:fgpow.shape[1]-1] + fgmdl_power_upper[:,::-1])
    fgpow = fgpow[:,::-1]
    tau_FG = -1.0 * dps.ds.subband_delay_spectra['sim']['lags'][0:fgpow.shape[1]][::-1] # shape=(nlags_lower,)

    modelkeys = ['z', 'k', 'PS']
    for modelkey in modelkeys:
        if modelkey not in modelPSinfo:
            raise KeyError('Input modelPSinfo does not contain the key {0}'.format(modelkey))
        if not isinstance(modelPSinfo[modelkey], NP.ndarray):
            raise TypeError('Value under key {0} of input modelPSinfo must be a numpy array'.format(key))
    if modelPSinfo['k'].shape != modelPSinfo['PS'].shape:
        raise ValueError('k and PS arrays in modelPSinfo do not match in size')

    conversion_factor_kperp_from_bll = DS.dkperp_dbll(redshifts, cosmo=cosmo)
    conversion_factor_kprll_from_delay = DS.dkprll_deta(redshifts, cosmo=cosmo)
    min_of_max_k = NP.amax(modelPSinfo['k'], axis=1).min()
    max_delay_model = NP.amin(min_of_max_k / conversion_factor_kprll_from_delay)
    if max_delay is not None:
        if not isinstance(max_delay, (int,float)):
            raise TypeError('Input max_delay must be a scalar')
        if max_delay < 0.0:
            max_delay = None
    if max_delay is None:
        max_delay = NP.copy(max_delay_model)
    max_delay = min(max_delay, max_delay - tau_FG.max())
    
    eps_tau = 1e-14

    dtau = tau_FG[1] - tau_FG[0]
    tau = NP.arange(0.0, max_delay+eps_tau, dtau) # Delays at which reflection constraints will be evaluated, shape=(ntau,)

    delay_model = NP.arange(0.0, max_delay_model+eps_tau, dtau) # Delays at which EoR model has to be evaluated, shape=(ndelays,)
    kprll_model = delay_model.reshape(1,-1) * conversion_factor_kprll_from_delay.reshape(-1,1) # nz x ndelays

    tau_cut = kprll_cut.reshape(1,-1) / conversion_factor_kprll_from_delay.reshape(-1,1) # nz x ntaucut

    attenuation = NP.ones((redshifts.size, kprll_cut.size, tau.size)) # nz x ndelaycut x ntau
    z_out = []
    for zind,z in enumerate(redshifts):
        nearest_z_ind = NP.argmin(NP.abs(modelPSinfo['z'] - z))
        z_out += [modelPSinfo['z'][nearest_z_ind]]
        wl = FCNST.c / CNST.rest_freq_HI * (1+modelPSinfo['z'][nearest_z_ind]) 
        kperp_model = (NP.mean(dps.bl_length[bll_ind])/wl) * conversion_factor_kperp_from_bll
        interpfunc_eorPSmodel = interpolate.interp1d(NP.log10(modelPSinfo['k'][nearest_z_ind,:]), NP.log10(modelPSinfo['PS'][nearest_z_ind,:]), kind='cubic', bounds_error=False)

        model_k = NP.sqrt(kprll_model[zind,:]**2 + kperp_model[zind]**2) # shape=(nkprll,)
        model_PS = 10 ** interpfunc_eorPSmodel(NP.log10(model_k)) # shape=(nkprll,)
        for taucutind,taucut in enumerate(tau_cut[zind,:]):
            for tauind, tau_refl in enumerate(tau):
                fg_ind = NP.where(tau_FG>=max(taucut-tau_refl, 0.0))[0]
                eor_ind = NP.where(NP.logical_and(delay_model>=max(taucut,tau_refl), delay_model<=tau_FG.max()+tau_refl+eps_tau))[0]

                if fg_ind.size != eor_ind.size:
                    raise ValueError('Mismatch in number of elements to compare')
                attenuation[zind,taucutind,tauind] = NP.nanmax(NP.sqrt(fgpow[zind_FG,fg_ind] / model_PS[eor_ind]))
    attenuation = 10.0 * NP.log10(attenuation)
    attenuation = NP.clip(attenuation, 0.0, attenuation.max())
    return {'z': z_out, 'kprll_cut': kprll_cut, 'delays': tau, 'attenuation': attenuation}

################################################################################

    
