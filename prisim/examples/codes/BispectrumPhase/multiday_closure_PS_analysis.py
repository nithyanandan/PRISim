from __future__ import print_function, division
from builtins import map, range
import copy, glob
import progressbar as PGB
import numpy as NP
import numpy.ma as MA
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.ticker as PLTick
import yaml, argparse, warnings
from astropy.io import ascii
import astropy.units as U
import astropy.constants as FCNST
import astropy.cosmology as cosmology
from astroutils import DSP_modules as DSP
from astroutils import constants as CNST
from astroutils import mathops as OPS
from astroutils import nonmathops as NMO
from astroutils import lookup_operations as LKP
import astroutils
import prisim
from prisim import interferometry as RI
from prisim import bispectrum_phase as BSP
from prisim import delay_spectrum as DS
import ipdb as PDB

PLT.switch_backend("TkAgg")

cosmoPlanck15 = cosmology.Planck15 # Planck 2015 cosmology
cosmo100 = cosmoPlanck15.clone(name='Modified Planck 2015 cosmology with h=1.0', H0=100.0) # Modified Planck 2015 cosmology with h=1.0, H= 100 km/s/Mpc

print('AstroUtils git # {0}\nPrisim git # {1}'.format(astroutils.__githash__, prisim.__githash__))

if __name__ == '__main__':
    
    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to analyze closure phases from multiple days')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default='/data3/t_nithyanandan/codes/mine/python/projects/closure/multiday_closure_PS_analysis_parms.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    projectdir = parms['dirStruct']['projectdir']
    datadir = projectdir + parms['dirStruct']['datadir']
    figdir = datadir + parms['dirStruct']['figdir']
    modelsdir = parms['dirStruct']['modeldir']
    infiles = parms['dirStruct']['infiles']
    visfile = parms['dirStruct']['visfile']
    visfiletype = parms['dirStruct']['visfiletype']
    hdf5_infile = parms['dirStruct']['hdf5_infile']
    model_hdf5files = parms['dirStruct']['model_hdf5files']
    model_labels = parms['dirStruct']['model_labels']

    telescope_parms = parms['telescope']
    site_latitude = telescope_parms['latitude']
    site_longitude = telescope_parms['longitude']

    preprocessinfo = parms['preProcessing']
    preprocess = preprocessinfo['action']
    flagchans = preprocessinfo['flagchans']
    if flagchans is not None:
        flagchans = NP.asarray(preprocessinfo['flagchans']).reshape(-1)
    flagants = preprocessinfo['flagants']
    if flagants is not None:
        flagants = NP.asarray(preprocessinfo['flagants']).reshape(-1)
    daybinsize = preprocessinfo['daybinsize']
    ndaybins = preprocessinfo['ndaybins']
    lstbinsize = preprocessinfo['lstbinsize']
    band_center = preprocessinfo['band_center']
    freq_resolution = preprocessinfo['freq_resolution']
    mdl_ndaybins = preprocessinfo['mdl_ndaybins']

    dspecinfo = parms['delaySpectrum']
    subbandinfo = dspecinfo['subband']
    freq_window_centers = NP.asarray(subbandinfo['freq_center'])
    freq_window_bw = NP.asarray(subbandinfo['bw_eff'])
    freq_window_shape = subbandinfo['shape']
    freq_window_fftpow = subbandinfo['fftpow']
    pad = dspecinfo['pad']

    apply_flags = dspecinfo['applyflags']
    if apply_flags:
        applyflags_str = 'Y'
    else:
        applyflags_str = 'N'

    bl = NP.asarray(dspecinfo['bl'])
    if bl.shape[0] != 3:
        raise ValueError('Input bl must be made of three vectors forming the triad')
    bltol = dspecinfo['bltol']

    infile = infiles[0]
    infile_no_ext = hdf5_infile.split('.hdf5')[0]

    # visdata = NP.load(visfile)
    if visfile is None:
        visinfo = None
    else:
        if visfiletype == 'hdf5':
            visinfo = NMO.load_dict_from_hdf5(visfile+'.hdf5')
            blind, blrefind, dbl = LKP.find_1NN(visinfo['baseline']['blvect'], bl, distance_ULIM=bltol, remove_oob=True)
            if blrefind.size != 3:
                blind_missing = NP.setdiff1d(NP.arange(3), blind, assume_unique=True)
                blind_next, blrefind_next, dbl_next = LKP.find_1NN(visinfo['baseline']['blvect'], -1*bl[blind_missing,:], distance_ULIM=bltol, remove_oob=True)
                if blind_next.size + blind.size != 3:
                    raise ValueError('Exactly three baselines were not found in the reference baselines')
                else:
                    blind = NP.append(blind, blind_missing[blind_next])
                    blrefind = NP.append(blrefind, blrefind_next)
            else:
                blind_missing = []

            vistriad = MA.array(visinfo['vis_real'][blrefind,:,:] + 1j * visinfo['vis_imag'][blrefind,:,:], mask=visinfo['mask'][blrefind,:,:])
            if len(blind_missing) > 0:
                vistriad[-blrefind_next.size:,:,:] = vistriad[-blrefind_next.size:,:,:].conj()
        else:
            visinfo = RI.InterferometerArray(None, None, None, init_file=visfile)

    tmpnpzdata = NP.load(datadir+infile)
    nchan = tmpnpzdata['flags'].shape[-1]
    freqs = band_center + freq_resolution * (NP.arange(nchan) - int(0.5*nchan)) 
    # cpinfo2 = BSP.loadnpz(datadir+infile)
    cpObj = BSP.ClosurePhase(datadir+hdf5_infile, freqs, infmt='hdf5')

    cpObj.smooth_in_tbins(daybinsize=daybinsize, ndaybins=ndaybins, lstbinsize=lstbinsize)
    cpObj.subtract(NP.zeros(1024))
    cpObj.subsample_differencing(daybinsize=None, ndaybins=4, lstbinsize=lstbinsize)

    cpDSobj = BSP.ClosurePhaseDelaySpectrum(cpObj)

    if visinfo is not None:
        if visfiletype == 'hdf5':
            visscaleinfo = {'vis': vistriad, 'lst': visinfo['header']['LST'], 'smoothinfo': {'op_type': 'interp1d', 'interp_kind': 'linear'}}
        else:
            visscaleinfo = {'vis': visinfo, 'bltriplet': bl, 'smoothinfo': {'op_type': 'interp1d', 'interp_kind': 'linear'}}
    else:
        visscaleinfo = None
    cpds = cpDSobj.FT(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow, pad=pad, datapool='prelim', visscaleinfo=visscaleinfo, method='fft', resample=True, apply_flags=apply_flags)

    model_cpObjs = []
    if model_hdf5files is not None:
        for i in range(len(model_hdf5files)):
            mdl_infile_no_ext = model_hdf5files[i].split('.hdf5')[0]
            model_cpObj = BSP.ClosurePhase(modelsdir+model_hdf5files[i], freqs, infmt='hdf5')
            model_cpObj.smooth_in_tbins(daybinsize=daybinsize, ndaybins=mdl_ndaybins[i], lstbinsize=lstbinsize)
            model_cpObj.subsample_differencing(daybinsize=None, ndaybins=4, lstbinsize=lstbinsize)
            model_cpObj.subtract(NP.zeros(1024))
            model_cpObjs += [copy.deepcopy(model_cpObj)]

    plot_info = parms['plot']
    plots = [key for key in plot_info if plot_info[key]['action']]

    PLT.ion()
    if ('1' in plots) or ('1a' in plots) or ('1b' in plots) or ('1c' in plots) or ('1d' in plots):
        triads = list(map(tuple, cpDSobj.cPhase.cpinfo['raw']['triads']))
        ntriads = len(triads)
        lst = cpDSobj.cPhase.cpinfo['raw']['lst']
        ntimes = lst.size
        tbins = cpDSobj.cPhase.cpinfo['processed']['prelim']['lstbins']
        ntbins = tbins.size
        dlst = lst[1] - lst[0]
        dtbins = cpDSobj.cPhase.cpinfo['processed']['prelim']['dlstbins']
        flags = cpDSobj.cPhase.cpinfo['raw']['flags']
        wts_raw = cpDSobj.cPhase.cpinfo['processed']['native']['wts'].data
        wts_proc = cpDSobj.cPhase.cpinfo['processed']['prelim']['wts'].data
        freq_wts = cpds['freq_wts']

        if '1a' in plots:
            triad = tuple(plot_info['1a']['triad'])
            triad_ind = triads.index(triad)
            
            fig = PLT.figure(figsize=(4,2.8))
            ax = fig.add_subplot(111)
            ax.imshow(wts_raw[triad_ind,0,:,:].T, origin='lower', extent=[1e-6*freqs.min(), 1e-6*freqs.max(), lst.min(), lst.max()+NP.mean(dlst)], vmin=wts_raw.min(), vmax=wts_raw.max(), interpolation='none', cmap='gray')
            ax.text(0.5, 0.97, '({0[0]:0d}, {0[1]:0d}, {0[2]:0d})'.format(triad), transform=ax.transAxes, fontsize=12, weight='semibold', ha='center', va='top', color='red')
            ax.set_xlim(1e-6*freqs.min(), 1e-6*freqs.max())
            ax.set_ylim(lst.min(), lst.max()+NP.mean(dlst))
            ax.set_aspect('auto')
            ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium')
            ax.set_ylabel('LST [hours]', fontsize=12, weight='medium')
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(left=0.2)
            fig.subplots_adjust(bottom=0.2)
            fig.subplots_adjust(right=0.98)
            
            PLT.savefig(figdir + '{0}_time_frequency_flags_triad_{1[0]:0d}_{1[1]:0d}_{1[2]:0d}.png'.format(infile_no_ext, triad), bbox_inches=0)
            PLT.savefig(figdir + '{0}_time_frequency_flags_triad_{1[0]:0d}_{1[1]:0d}_{1[2]:0d}.eps'.format(infile_no_ext, triad), bbox_inches=0)

            fig = PLT.figure(figsize=(4,2.8))
            ax = fig.add_subplot(111)
            wtsimg = ax.imshow(wts_proc[:,0,triad_ind,:], origin='lower', extent=[1e-6*freqs.min(), 1e-6*freqs.max(), tbins.min(), tbins.max()+NP.mean(dtbins)], vmin=wts_proc.min(), vmax=wts_proc.max(), interpolation='none', cmap='gray')
            ax.text(0.5, 0.97, '({0[0]:0d}, {0[1]:0d}, {0[2]:0d})'.format(triad), transform=ax.transAxes, fontsize=12, weight='semibold', ha='center', va='top', color='red')
            ax.set_xlim(1e-6*freqs.min(), 1e-6*freqs.max())
            ax.set_ylim(tbins.min(), tbins.max()+NP.mean(dtbins))
            ax.set_aspect('auto')
            ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium')
            ax.set_ylabel('LST [hours]', fontsize=12, weight='medium')
            cbax = fig.add_axes([0.86, 0.2, 0.02, 0.75])
            cbar = fig.colorbar(wtsimg, cax=cbax, orientation='vertical')
            cbax.yaxis.tick_right()
            # cbax.yaxis.set_label_position('right')
            
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(left=0.2)
            fig.subplots_adjust(bottom=0.2)
            fig.subplots_adjust(right=0.85)
            PLT.savefig(figdir + '{0}_time_frequency_wts_triad_{1[0]:0d}_{1[1]:0d}_{1[2]:0d}.png'.format(infile_no_ext, triad), bbox_inches=0)
            PLT.savefig(figdir + '{0}_time_frequency_wts_triad_{1[0]:0d}_{1[1]:0d}_{1[2]:0d}.eps'.format(infile_no_ext, triad), bbox_inches=0)
            
        if '1b' in plots:
            triad = tuple(plot_info['1b']['triad'])
            triad_ind = triads.index(triad)
            net_wts_raw = wts_raw[:,0,triad_ind,:][NP.newaxis,:,:] * freq_wts[:,NP.newaxis,:] # nspw x nlst x nchan
            net_wts_proc = wts_proc[:,0,triad_ind,:][NP.newaxis,:,:] * freq_wts[:,NP.newaxis,:] # nspw x nlst x nchan

            # net_wts_raw = wts_raw[triad_ind,0,:,:][NP.newaxis,:,:] * freq_wts[:,:,NP.newaxis]
            # net_wts_proc = wts_proc[triad_ind,0,:,:][NP.newaxis,:,:] * freq_wts[:,:,NP.newaxis]
            
            nrow = freq_wts.shape[0]
            fig, axs = PLT.subplots(nrows=nrow, sharex=True, sharey=True, figsize=(3.5,6))
            for axind in range(len(axs)):
                wtsimg = axs[axind].imshow(net_wts_proc[axind,:,:], origin='lower', extent=[1e-6*freqs.min(), 1e-6*freqs.max(), tbins.min(), tbins.max()+NP.mean(dtbins)], norm=PLTC.LogNorm(vmin=1e-6, vmax=net_wts_proc.max()), interpolation='none', cmap='binary')
                if axind == 0:
                    axs[axind].text(0.97, 0.97, '({0[0]:0d}, {0[1]:0d}, {0[2]:0d})'.format(triad), transform=axs[axind].transAxes, fontsize=12, weight='semibold', ha='right', va='top', color='red')
                axs[axind].set_xlim(1e-6*freqs.min(), 1e-6*freqs.max())
                axs[axind].set_ylim(tbins.min(), tbins.max()+NP.mean(dtbins))
                axs[axind].set_aspect('auto')
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(left=0.2)
            fig.subplots_adjust(bottom=0.12)
            fig.subplots_adjust(right=0.85)
            cbax = fig.add_axes([0.86, 0.12, 0.02, 0.3])
            cbar = fig.colorbar(wtsimg, cax=cbax, orientation='vertical')
            cbax.yaxis.tick_right()
            big_ax = fig.add_subplot(111)
            # big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
            big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
            big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium', labelpad=20)
            big_ax.set_ylabel('LST [seconds]', fontsize=12, weight='medium', labelpad=35)
            PLT.savefig(figdir + '{0}_time_frequency_netwts_triad_{1[0]:0d}_{1[1]:0d}_{1[2]:0d}.png'.format(infile_no_ext, triad), bbox_inches=0)
            PLT.savefig(figdir + '{0}_time_frequency_netwts_triad_{1[0]:0d}_{1[1]:0d}_{1[2]:0d}.eps'.format(infile_no_ext, triad), bbox_inches=0)

        if '1c' in plots:
            ncol = 5
            nrow = min(6, int(NP.ceil(1.0*ntriads/ncol)))
            npages = int(NP.ceil(1.0 * ntriads / (nrow*ncol)))
            for pagei in range(npages):
                if pagei > 0:
                    ntriads_remain = ntriads - pagei * nrow * ncol
                    nrow = min(6, int(NP.ceil(1.0*ntriads_remain/ncol)))
                fig, axs = PLT.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(8,6.4))
                for i in range(nrow):
                    for j in range(ncol):
                        if i*ncol+j < ntriads:
                            axs[i,j].imshow(wts_raw[i*ncol+j,0,:,:].T, origin='lower', extent=[1e-6*freqs.min(), 1e-6*freqs.max(), lst.min(), lst.max()+NP.mean(dlst)], vmin=0, vmax=1, interpolation='none', cmap='gray')
                            axs[i,j].text(0.5, 0.97, '({0[0]:0d}, {0[1]:0d}, {0[2]:0d})'.format(triads[i*ncol+j,:]), transform=axs[i,j].transAxes, fontsize=10, weight='medium', ha='center', va='top', color='red')
                        else:
                            axs[i,j].axis('off')
                        axs[i,j].set_xlim(1e-6*freqs.min(), 1e-6*freqs.max())
                        axs[i,j].set_ylim(lst.min(), lst.max()+NP.mean(dlst))
                        axs[i,j].set_aspect('auto')
                fig.subplots_adjust(hspace=0, wspace=0)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(left=0.1)
                fig.subplots_adjust(bottom=0.15)
                fig.subplots_adjust(right=0.98)
                big_ax = fig.add_subplot(111)
                # big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium', labelpad=20)
                big_ax.set_ylabel('LST [seconds]', fontsize=12, weight='medium', labelpad=35)
                PLT.savefig(figdir + '{0}_time_frequency_flags_page_{1:03d}_of_{2:0d}.png'.format(infile_no_ext, pagei+1, npages), bbox_inches=0)
                PLT.savefig(figdir + '{0}_time_frequency_flags_page_{1:03d}_of_{2:0d}.eps'.format(infile_no_ext, pagei+1, npages), bbox_inches=0)
            
        if '1d' in plots:
            datastage = plot_info['1d']['datastage']
            if datastage.lower() not in ['native', 'prelim']:
                raise ValueError('Input datastage value invalid')
            elif datastage.lower() == 'native':
                cphase = cpObj.cpinfo['processed'][datastage]['cphase']
                datastr = '{0}'.format(datastage)
            else:
                statistic = plot_info['1d']['statistic']
                cphase = cpObj.cpinfo['processed'][datastage]['cphase'][statistic]
                datastr = '{0}_{1}'.format(datastage, statistic)
            mask = cphase.mask 
    
            timetriad_selection = plot_info['1d']['selection']
            if timetriad_selection is not None:
                dayind = timetriad_selection['dayind']
            else:
                dayind = 0
            for key in timetriad_selection:
                if timetriad_selection[key] is not None:
                    if key == 'triads':
                        triads = list(map(tuple, timetriad_selection[key]))
                    elif key == 'lstrange':
                        lstrange = timetriad_selection[key]
                        if datastage.lower() == 'native':
                            lstbins = cpObj.cpinfo['raw']['lst'][:,dayind]
                        else:
                            lstbins = cpObj.cpinfo['processed']['prelim']['lstbins']
                        if lstrange is None:
                            lstinds = NP.arange(lstbins.size)
                        else:
                            lstrange = NP.asarray(lstrange)
                            lstinds = NP.where(NP.logical_and(lstbins >= lstrange.min(), lstbins <= lstrange.max()))[0]
                else:
                    if key == 'triads':
                        triads = list(map(tuple, cpDSobj.cPhase.cpinfo['raw']['triads']))
                    elif key == 'lstrange':
                        if datastage.lower() == 'native':
                            lstbins = cpObj.cpinfo['raw']['lst'][:,dayind]
                        else:
                            lstbins = cpObj.cpinfo['processed']['prelim']['lstbins']
                        lstinds = NP.arange(lstbins.size)
            sparseness = plot_info['1d']['sparseness']
            if sparseness < 1.0:
                sparseness = 1.0
            sparsestr = '{0:.1f}'.format(sparseness)
            sparsenum = NP.ceil(freqs.size / sparseness).astype(NP.int)
            if sparsenum == freqs.size:
                indchan = NP.arange(freqs.size)
            applyflags = plot_info['1d']['applyflags']
            if applyflags:
                flags_str = 'flags'
            else:
                flags_str = 'noflags'

            ncol = 3
            nrow = min(4, int(NP.ceil(old_div(1.0*lstinds.size,ncol))))
            npages = int(NP.ceil(old_div(1.0 * lstinds.size, (nrow*ncol))))
            nlst_remain = lstinds.size
            for pagei in range(npages):
                if pagei > 0:
                    nlst_remain = lstinds.size - pagei * nrow * ncol
                    nrow = min(4, int(NP.ceil(old_div(1.0*nlst_remain,ncol))))
                fig, axs = PLT.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, figsize=(8,6.4))
                for i in range(nrow):
                    for j in range(ncol):
                        lstind = (lstinds.size - nlst_remain) + i*ncol+j
                        lind = lstinds[lstind]
                        if lstind < lstinds.size:
                            for triad in triads:
                                triad_ind = triads.index(triad)
                                if sparsenum < freqs.size:
                                    indchan = NP.sort(NP.random.randint(freqs.size, size=sparsenum))
                                axs[i,j].plot(1e-6*freqs[indchan], cphase[lind,dayind,triad_ind,indchan], marker='.', ms=2, ls='none')
                                if applyflags:
                                    flagind = mask[lind,dayind,triad_ind,:]
                                    axs[i,j].plot(1e-6*freqs[flagind], cphase[lind,dayind,triad_ind,flagind].data, marker='.', ms=1, color='black', ls='none')
                                axs[i,j].text(0.5, 0.97, '{0:.2f} hrs'.format(lstbins[lind]), transform=axs[i,j].transAxes, fontsize=10, weight='medium', ha='center', va='top', color='black')
                        else:
                            axs[i,j].axis('off')
                        axs[i,j].set_xlim(1e-6*freqs.min(), 1e-6*freqs.max())
                        axs[i,j].set_ylim(-3.5,3.5)

                fig.subplots_adjust(hspace=0, wspace=0)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(left=0.1)
                fig.subplots_adjust(bottom=0.15)
                fig.subplots_adjust(right=0.98)

                big_ax = fig.add_subplot(111)
                # big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium', labelpad=20)
                big_ax.set_ylabel(r'$\phi_\nabla$ [radians]', fontsize=12, weight='medium', labelpad=35)

            PLT.savefig(figdir + '{0}_cp_spectra_{1}_{2}_{3}_triads_day_{4}_{5:.1f}x_sparse_page_{6:03d}_of_{7:0d}.png'.format(infile_no_ext, flags_str, datastr, len(triads), dayind, sparseness, pagei+1, npages), bbox_inches=0)
            PLT.savefig(figdir + '{0}_cp_spectra_{1}_{2}_{3}_triads_day_{4}_{5:.1f}x_sparse_page_{6:03d}_of_{7:0d}.eps'.format(infile_no_ext, flags_str, datastr, len(triads), dayind, sparseness, pagei+1, npages), bbox_inches=0)

            # fig = PLT.figure(figsize=(3.75,3))
            # ax = fig.add_subplot(111)
            # for lstind in lstinds:
            #     for triad in triads:
            #         triad_ind = triads.index(triad)
            #         if sparsenum < freqs.size:
            #             indchan = NP.sort(NP.random.randint(freqs.size, size=sparsenum))
            #         ax.plot(1e-6*freqs[indchan], cphase[lstind,dayind,triad_ind,indchan], marker='.', ms=2, ls='none')
            #         if applyflags:
            #             flagind = mask[lstind,dayind,triad_ind,:]
            #             ax.plot(1e-6*freqs[flagind], cphase[lstind,dayind,triad_ind,flagind].data, marker='.', ms=1, color='black', ls='none')
            # ax.set_xlim(1e-6*freqs.min(), 1e-6*freqs.max())
            # ax.set_ylim(-3.5,3.5)
            # ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium')
            # ax.set_ylabel(r'$\phi_\nabla$ [radians]', fontsize=12, weight='medium')
            # fig.subplots_adjust(top=0.95)
            # fig.subplots_adjust(left=0.16)
            # fig.subplots_adjust(bottom=0.18)
            # fig.subplots_adjust(right=0.98)
                
            # PLT.savefig(figdir + '{0}_cp_spectra_{1}_{2}_{3}_triads_{4}_times_{5:.1f}x_sparse.png'.format(infile_no_ext, flags_str, datastr, len(triads), lstinds.size, sparseness), bbox_inches=0)
            # PLT.savefig(figdir + '{0}_cp_spectra_{1}_{2}_{3}_triads_{4}_times_{5:.1f}x_sparse.eps'.format(infile_no_ext, flags_str, datastr, len(triads), lstinds.size, sparseness), bbox_inches=0)

    if ('2' in plots) or ('2a' in plots) or ('2b' in plots) or ('2c' in plots) or ('2d' in plots):
        dir_PS = plot_info['2']['PS_dir']
        infile_pfx_a = plot_info['2']['infile_pfx_a']
        outfile_pfx_a = plot_info['2']['outfile_pfx_a']
        infile_pfx_b = plot_info['2']['infile_pfx_b']
        outfile_pfx_b = plot_info['2']['outfile_pfx_b']
        sampling = plot_info['2']['sampling']
        statistic = plot_info['2']['statistic']
        cohax = plot_info['2']['cohax']
        incohax = plot_info['2']['incohax']
        collapseax_a = plot_info['2']['collapseax_a']
        collapseax_b = plot_info['2']['collapseax_b']
        datapool = plot_info['2']['datapool']
        pspec_unit_type = plot_info['2']['units']
        ps_errtype = plot_info['2']['errtype']
        errshade = {}
        for errtype in ps_errtype:
            if errtype.lower() == 'ssdiff':
                errshade[errtype] = '0.8'
            elif errtype.lower() == 'psdiff':
                errshade[errtype] = '0.6'
        nsigma = plot_info['2']['nsigma']
        beaminfo = plot_info['2']['beaminfo']
        xlim = plot_info['2']['xlim']
        if infile_pfx_a is not None:
            ps_infile_a = datadir + dir_PS + infile_pfx_a + '_' + infile_no_ext + '.hdf5'
            pserr_infile_a = datadir + dir_PS + infile_pfx_a + '_' + infile_no_ext + '_errinfo.hdf5'
        if outfile_pfx_a is not None:
            ps_outfile_a = datadir + dir_PS + outfile_pfx_a + '_' + infile_no_ext + '.hdf5'
            pserr_outfile_a = datadir + dir_PS + outfile_pfx_a + '_' + infile_no_ext + '_errinfo.hdf5'

        if infile_pfx_b is not None:
            ps_infile_b = datadir + dir_PS + infile_pfx_b + '_' + infile_no_ext + '.hdf5'
            pserr_infile_b = datadir + dir_PS + infile_pfx_b + '_' + infile_no_ext + '_errinfo.hdf5'
        if outfile_pfx_b is not None:
            ps_outfile_b = datadir + dir_PS + outfile_pfx_b + '_' + infile_no_ext + '.hdf5'
            pserr_outfile_b = datadir + dir_PS + outfile_pfx_b + '_' + infile_no_ext + '_errinfo.hdf5'

        timetriad_selection = plot_info['2']['selection']
        if timetriad_selection is not None:
            dayind = timetriad_selection['days']
        for key in timetriad_selection:
            if timetriad_selection[key] is not None:
                if key == 'triads':
                    triads = list(map(tuple, timetriad_selection[key]))
                elif key == 'lstrange':
                    lstrange = timetriad_selection[key]
                    lstbins = cpObj.cpinfo['processed']['prelim']['lstbins']
                    if lstrange is None:
                        lstinds = NP.arange(lstbins.size)
                    else:
                        lstrange = NP.asarray(lstrange)
                        lstinds = NP.where(NP.logical_and(lstbins >= lstrange.min(), lstbins <= lstrange.max()))[0]
                        if lstinds.size == 0:
                            raise ValueError('No data found in the specified LST range.')
            else:
                if key == 'triads':
                    triads = list(map(tuple, cpDSobj.cPhase.cpinfo['raw']['triads']))
                elif key == 'lstrange':
                    lstbins = cpObj.cpinfo['processed']['prelim']['lstbins']
                    lstinds = NP.arange(lstbins.size)
        selection = {'triads': triads, 'lst': lstinds, 'days': dayind}
        autoinfo = {'axes': cohax}
        xinfo_a = {'axes': incohax, 'avgcov': False, 'collapse_axes': collapseax_a, 'dlst_range': timetriad_selection['dlst_range']}
        xinfo_b = {'axes': incohax, 'avgcov': False, 'collapse_axes': collapseax_b, 'dlst_range': timetriad_selection['dlst_range']}

        if pspec_unit_type == 'K':
            pspec_unit = 'mK2 Mpc3'
        else:
            pspec_unit = 'Jy2 Mpc'

        subselection = plot_info['2']['subselection']
        mdl_day = plot_info['2']['modelinfo']['mdl_day']
        mdl_cohax = plot_info['2']['modelinfo']['mdl_cohax']
        mdl_incohax = plot_info['2']['modelinfo']['mdl_incohax']
        mdl_collapseax_a = plot_info['2']['modelinfo']['mdl_collapax_a']
        mdl_collapseax_b = plot_info['2']['modelinfo']['mdl_collapax_b']
        mdl_dir_PS = plot_info['2']['modelinfo']['PS_dir']
        mdl_infile_pfx_a = plot_info['2']['modelinfo']['infile_pfx_a']
        mdl_outfile_pfx_a = plot_info['2']['modelinfo']['outfile_pfx_a']
        mdl_infile_pfx_b = plot_info['2']['modelinfo']['infile_pfx_b']
        mdl_outfile_pfx_b = plot_info['2']['modelinfo']['outfile_pfx_b']
        
        if model_hdf5files is not None:
            mdl_autoinfo = [{'axes': mdl_cohax[i]} for i in range(len(model_hdf5files))]
            mdl_xinfo_a = [{'axes': mdl_incohax[i], 'avgcov': False, 'collapse_axes': mdl_collapseax_a[i], 'dlst_range': timetriad_selection['dlst_range']} for i in range(len(model_hdf5files))]
            mdl_xinfo_b = [{'axes': mdl_incohax[i], 'avgcov': False, 'collapse_axes': mdl_collapseax_b[i], 'dlst_range': timetriad_selection['dlst_range']} for i in range(len(model_hdf5files))]

        if statistic is None:
            statistic = ['mean', 'median']
        else:
            statistic = [statistic]
            
        if infile_pfx_a is not None:
            xcpdps2_a = BSP.read_CPhase_cross_power_spectrum(ps_infile_a)
            xcpdps2_a_errinfo = BSP.read_CPhase_cross_power_spectrum(pserr_infile_a)
        else:
            xcpdps2_a = cpDSobj.compute_power_spectrum(selection=selection, autoinfo=autoinfo, xinfo=xinfo_a, units=pspec_unit_type, beamparms=beaminfo)
            xcpdps2_a_errinfo = cpDSobj.compute_power_spectrum_uncertainty(selection=selection, autoinfo=autoinfo, xinfo=xinfo_a, units=pspec_unit_type, beamparms=beaminfo)
        if outfile_pfx_a is not None:
            BSP.save_CPhase_cross_power_spectrum(xcpdps2_a, ps_outfile_a)
            BSP.save_CPhase_cross_power_spectrum(xcpdps2_a_errinfo, pserr_outfile_a)

        if infile_pfx_b is not None:
            xcpdps2_b = BSP.read_CPhase_cross_power_spectrum(ps_infile_b)
            xcpdps2_b_errinfo = BSP.read_CPhase_cross_power_spectrum(pserr_infile_b)
        else:
            xcpdps2_b = cpDSobj.compute_power_spectrum(selection=selection, autoinfo=autoinfo, xinfo=xinfo_b, units=pspec_unit_type, beamparms=beaminfo)
            xcpdps2_b_errinfo = cpDSobj.compute_power_spectrum_uncertainty(selection=selection, autoinfo=autoinfo, xinfo=xinfo_b, units=pspec_unit_type, beamparms=beaminfo)
        if outfile_pfx_b is not None:
            BSP.save_CPhase_cross_power_spectrum(xcpdps2_b, ps_outfile_b)
            BSP.save_CPhase_cross_power_spectrum(xcpdps2_b_errinfo, pserr_outfile_b)

        nsamples_incoh = xcpdps2_a[sampling]['whole']['nsamples_incoh']
        nsamples_coh = xcpdps2_a[sampling]['whole']['nsamples_coh']

        model_cpDSobjs = []
        cpds_models = []
        xcpdps2_a_models = []
        xcpdps2_a_errinfo_models = []
        xcpdps2_b_models = []
        xcpdps2_b_errinfo_models = []

        if model_hdf5files is not None:
            if mdl_infile_pfx_a is not None:
                if isinstance(mdl_infile_pfx_a, list):
                    if (len(mdl_infile_pfx_a) > 0):
                        if not isinstance(mdl_dir_PS, list):
                            if isinstance(mdl_dir_PS, str):
                                mdl_dir_PS = [mdl_dir_PS] * len(model_hdf5files)
                            else:
                                raise TypeError('PS directory for models must be a list of strings')
                        else:
                            if len(mdl_dir_PS) != len(model_hdf5files):
                                raise ValueError('Input model PS directories must match the number of models being analyzed.')
                else:
                    raise TypeError('Input model PS infile_a prefixes must be specified as a list of strings')

            if mdl_infile_pfx_b is not None:
                if isinstance(mdl_infile_pfx_b, list):
                    if len(mdl_infile_pfx_b) != len(mdl_infile_pfx_b):
                        raise ValueError('Length of input model PS infile_b prefixes must match the length of input model PS infile_a prefixes')
                else:
                    raise TypeError('Input model PS infile_b prefixes must be specified as a list of strings')
                
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Models '.format(len(model_hdf5files)), PGB.ETA()], maxval=len(model_hdf5files)).start()

            for i in range(len(model_hdf5files)):
                mdl_infile_no_ext = model_hdf5files[i].split('.hdf5')[0]
                mdl_ps_infile_a_provided = False
                mdl_pserr_infile_a_provided = False
                mdl_ps_infile_b_provided = False
                mdl_pserr_infile_b_provided = False
                if mdl_infile_pfx_a is not None:
                    if len(mdl_infile_pfx_a) > 0:
                        if mdl_infile_pfx_a[i] is not None:
                            if not isinstance(mdl_infile_pfx_a[i], str):
                                raise TypeError('Input {0}-th model cross PS file must be a string'.format(i+1))
                            else:
                                try:
                                    model_xcpdps2_a = BSP.read_CPhase_cross_power_spectrum(mdl_dir_PS[i]+mdl_infile_pfx_a[i]+'_'+mdl_infile_no_ext+'.hdf5')
                                except IOError as xcption:
                                    mdl_ps_infile_a_provided = False
                                    warnings.warn('Provided model cross-power spectrum infile_a "{0}" could not be opened. Will proceed with computing of model cross power spectrum based on parameters specified.'.format(mdl_dir_PS[i]+mdl_infile_pfx_a[i]+'.hdf5'))
                                else:
                                    mdl_ps_infile_a_provided = True
                                    xcpdps2_a_models += [copy.deepcopy(model_xcpdps2_a)]

                                try:
                                    model_xcpdps2_a_errinfo = BSP.read_CPhase_cross_power_spectrum(mdl_dir_PS[i]+mdl_infile_pfx_a[i]+'_'+mdl_infile_no_ext+'_errinfo.hdf5')
                                except IOError as xcption:
                                    mdl_pserr_infile_a_provided = False
                                    warnings.warn('Provided model cross-power spectrum infile_a "{0}" could not be opened. Will proceed with computing of model cross power spectrum based on parameters specified.'.format(mdl_dir_PS[i]+mdl_infile_pfx_a[i]+'_errinfo.hdf5'))
                                else:
                                    mdl_pserr_infile_a_provided = True
                                    xcpdps2_a_errinfo_models += [copy.deepcopy(model_xcpdps2_a_errinfo)]

                if mdl_infile_pfx_b is not None:
                    if len(mdl_infile_pfx_b) > 0:
                        if mdl_infile_pfx_b[i] is not None:
                            if not isinstance(mdl_infile_pfx_b[i], str):
                                raise TypeError('Input {0}-th model cross PS file must be a string'.format(i+1))
                            else:
                                try:
                                    model_xcpdps2_b = BSP.read_CPhase_cross_power_spectrum(mdl_dir_PS[i]+mdl_infile_pfx_b[i]+'_'+mdl_infile_no_ext+'.hdf5')
                                except IOError as xcption:
                                    mdl_ps_infile_b_provided = False
                                    warnings.warn('Provided model cross-power spectrum infile_b "{0}" could not be opened. Will proceed with computing of model cross power spectrum based on parameters specified.'.format(mdl_dir_PS[i]+mdl_infile_pfx_b[i]+'.hdf5'))
                                else:
                                    mdl_ps_infile_b_provided = True
                                    xcpdps2_b_models += [copy.deepcopy(model_xcpdps2_b)]

                                try:
                                    model_xcpdps2_b_errinfo = BSP.read_CPhase_cross_power_spectrum(mdl_dir_PS[i]+mdl_infile_pfx_b[i]+'_'+mdl_infile_no_ext+'_errinfo.hdf5')
                                except IOError as xcption:
                                    mdl_pserr_infile_b_provided = False
                                    warnings.warn('Provided model cross-power spectrum infile_b "{0}" could not be opened. Will proceed with computing of model cross power spectrum based on parameters specified.'.format(mdl_dir_PS[i]+mdl_infile_pfx_b[i]+'_errinfo.hdf5'))
                                else:
                                    mdl_pserr_infile_b_provided = True
                                    xcpdps2_b_errinfo_models += [copy.deepcopy(model_xcpdps2_b_errinfo)]
                                    
                if (not mdl_ps_infile_a_provided) or (not mdl_pserr_infile_a_provided) or (not mdl_ps_infile_b_provided) or (not mdl_pserr_infile_b_provided):
                    # model_cpObj = BSP.ClosurePhase(modelsdir+model_hdf5files[i], freqs, infmt='hdf5')
                    # model_cpObj.smooth_in_tbins(daybinsize=daybinsize, ndaybins=mdl_ndaybins[i], lstbinsize=lstbinsize)
                    # model_cpObj.subsample_differencing(daybinsize=None, ndaybins=4, lstbinsize=lstbinsize)
                    # model_cpObj.subtract(NP.zeros(1024))
                    # model_cpObjs += [copy.deepcopy(model_cpObj)]
                    model_cpDSobjs += [BSP.ClosurePhaseDelaySpectrum(model_cpObjs[i])]
                    cpds_models += [model_cpDSobjs[i].FT(freq_window_bw, freq_center=freq_window_centers, shape=freq_window_shape, fftpow=freq_window_fftpow, pad=pad, datapool='prelim', visscaleinfo=visscaleinfo, method='fft', resample=True, apply_flags=apply_flags)]

                    if not mdl_ps_infile_a_provided:
                        xcpdps2_a_models += [model_cpDSobjs[i].compute_power_spectrum(selection=selection, autoinfo=mdl_autoinfo[i], xinfo=mdl_xinfo_a[i], units=pspec_unit_type, beamparms=beaminfo)]
                    if not mdl_pserr_infile_a_provided:
                        xcpdps2_a_errinfo_models += [model_cpDSobjs[i].compute_power_spectrum_uncertainty(selection=selection, autoinfo=autoinfo, xinfo=xinfo_a, units=pspec_unit_type, beamparms=beaminfo)]

                    if not mdl_ps_infile_b_provided:
                        xcpdps2_b_models += [model_cpDSobjs[i].compute_power_spectrum(selection=selection, autoinfo=mdl_autoinfo[i], xinfo=mdl_xinfo_b[i], units=pspec_unit_type, beamparms=beaminfo)]
                    if not mdl_pserr_infile_b_provided:
                        xcpdps2_b_errinfo_models += [model_cpDSobjs[i].compute_power_spectrum_uncertainty(selection=selection, autoinfo=autoinfo, xinfo=xinfo_b, units=pspec_unit_type, beamparms=beaminfo)]

                else:
                    model_cpObjs += [None]
                    model_cpDSobjs += [None]
                    cpds_models += [None]
                    
                if mdl_outfile_pfx_a is not None:
                    if isinstance(mdl_outfile_pfx_a, str):
                        mdl_outfile_pfx_a = [mdl_outfile_pfx_a] * len(model_hdf5files)
                    if not isinstance(mdl_outfile_pfx_a, list):
                        raise TypeError('The model cross-power spectrum outfile prefixes must be specified as a list with item for each model.')
                    if len(mdl_outfile_pfx_a) != len(mdl_dir_PS):
                        raise ValueError('Invalid number of model cross-power output files specified')
                    mdl_ps_outfile_a = mdl_dir_PS[i] + mdl_outfile_pfx_a[i] + '_' + mdl_infile_no_ext + '.hdf5'
                    mdl_pserr_outfile_a = mdl_dir_PS[i] + mdl_outfile_pfx_a[i] + '_' + mdl_infile_no_ext + '_errinfo.hdf5'
                    BSP.save_CPhase_cross_power_spectrum(xcpdps2_a_models[-1], mdl_ps_outfile_a)
                    BSP.save_CPhase_cross_power_spectrum(xcpdps2_a_errinfo_models[-1], mdl_pserr_outfile_a)

                if mdl_outfile_pfx_b is not None:
                    if isinstance(mdl_outfile_pfx_b, str):
                        mdl_outfile_pfx_b = [mdl_outfile_pfx_b] * len(model_hdf5files)
                    if not isinstance(mdl_outfile_pfx_b, list):
                        raise TypeError('The model cross-power spectrum outfile prefixes must be specified as a list with item for each model.')
                    if len(mdl_outfile_pfx_b) != len(mdl_dir_PS):
                        raise ValueError('Invalid number of model cross-power output files specified')
                    mdl_ps_outfile_b = mdl_dir_PS[i] + mdl_outfile_pfx_b[i] + '_' + mdl_infile_no_ext + '.hdf5'
                    mdl_pserr_outfile_b = mdl_dir_PS[i] + mdl_outfile_pfx_b[i] + '_' + mdl_infile_no_ext + '_errinfo.hdf5'
                    BSP.save_CPhase_cross_power_spectrum(xcpdps2_b_models[-1], mdl_ps_outfile_b)
                    BSP.save_CPhase_cross_power_spectrum(xcpdps2_b_errinfo_models[-1], mdl_pserr_outfile_b)
                    
                progress.update(i+1)
            progress.finish()

        spw = subselection['spw']
        if spw is None:
            spwind = NP.arange(xcpdps2_a[sampling]['z'].size)
        else:
            spwind = NP.asarray(spw)
        lstind = NMO.find_list_in_list(xcpdps2_a[sampling][datapool[0]]['diagoffsets'][1], NP.asarray(subselection['lstdiag']))
        dayind = NP.asarray(subselection['day'])
        dayind_models = NP.asarray(mdl_day)
        triadind = NMO.find_list_in_list(xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3], NP.asarray(subselection['triaddiag']))

        mdl_colrs = ['red', 'green', 'blue', 'cyan', 'gray', 'orange']

        if '2a' in plots:
            for stat in statistic:
                for zind in spwind:
                    for lind in lstind:
                        for di,dind in enumerate(dayind):
                            maxabsvals = []
                            minabsvals = []
                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                            if len(datapool) == 1:
                                axs = [axs]
                            for dpoolind,dpool in enumerate(datapool):
                                for trno,trind in enumerate([triadind[0]]):
                                    if model_hdf5files is not None:
                                        for mdlind, mdl in enumerate(model_labels):
                                            if dpool in xcpdps2_a_models[mdlind][sampling]:
                                                psval = (1/3.0) * xcpdps2_a_models[mdlind][sampling][dpool][stat][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to(pspec_unit).value
                                                negind = psval.real < 0.0
                                                posind = NP.logical_not(negind)
                                                maxabsvals += [NP.abs(psval.real).max()]
                                                minabsvals += [NP.abs(psval.real).min()]
                                                if sampling == 'oversampled':
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,posind], psval.real[posind], ls='none', marker='.', ms=1, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='|', ms=1, color=mdl_colrs[mdlind])
                                                else:
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,:], NP.abs(psval.real), ls='-', lw=1, marker='.', ms=1, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='o', ms=2, color=mdl_colrs[mdlind])

                                    if dpool in xcpdps2_a[sampling]:
                                        psval = (1/3.0) * xcpdps2_a[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                        negind = psval.real < 0.0
                                        posind = NP.logical_not(negind)
                                        maxabsvals += [NP.abs(psval.real).max()]
                                        minabsvals += [NP.abs(psval.real).min()]
                                    
                                        if sampling == 'oversampled':
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,posind], psval.real[posind], ls='none', marker='.', ms=1, color='black', label='FG+N')
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='|', ms=1, color='black')
                                        else:
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], NP.abs(psval.real), ls='-', lw=1, marker='.', ms=1, color='black', label='FG+N')
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='o', ms=2, color='black')
                                            
                                    legend = axs[dpoolind].legend(loc='upper right', shadow=False, fontsize=8)
                                    if trno == 0:
                                        axs[dpoolind].set_yscale('log')
                                        axs[dpoolind].text(0.05, 0.97, 'Real', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.87, r'$z=$'+' {0:.1f}'.format(xcpdps2_a[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.77, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.67, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

                                    axt = axs[dpoolind].twiny()
                                    axt.set_xlim(1e6*xcpdps2_a[sampling]['lags'].min(), 1e6*xcpdps2_a[sampling]['lags'].max())
                                    # axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')

                                if xlim is None:
                                    axs[dpoolind].set_xlim(0.99*xcpdps2_a[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a[sampling]['kprll'][zind,:].max())
                                else:
                                    axs[dpoolind].set_xlim(xlim)
                                axs[dpoolind].set_ylim(0.5*min(minabsvals), 2*max(maxabsvals))
                            fig.subplots_adjust(top=0.85)
                            fig.subplots_adjust(bottom=0.16)
                            fig.subplots_adjust(left=0.22)
                            fig.subplots_adjust(right=0.98)
                                                
                            big_ax = fig.add_subplot(111)
                            big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                            # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                            big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                            big_ax.set_xticks([])
                            big_ax.set_yticks([])
                            big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                            if pspec_unit_type == 'K':
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=30)
                            else:
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=30)
        
                            big_axt = big_ax.twiny()
                            big_axt.set_xticks([])
                            big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
        
                            PLT.savefig(figdir + '{0}_log_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.png'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_log_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.pdf'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_log_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.eps'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)

                            maxabsvals = []
                            minabsvals = []
                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                            if len(datapool) == 1:
                                axs = [axs]
                            for dpoolind,dpool in enumerate(datapool):
                                for trno,trind in enumerate([triadind[0]]):
                                    if model_hdf5files is not None:
                                        for mdlind, mdl in enumerate(model_labels):
                                            if dpool in xcpdps2_a_models[mdlind][sampling]:
                                                psval = (1/3.0) * xcpdps2_a_models[mdlind][sampling][dpool][stat][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to(pspec_unit).value
                                                negind = psval.imag < 0.0
                                                posind = NP.logical_not(negind)
                                                maxabsvals += [NP.abs(psval.imag).max()]
                                                minabsvals += [NP.abs(psval.imag).min()]
                                                if sampling == 'oversampled':
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,posind], psval.imag[posind], ls='none', marker='.', ms=1, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='|', ms=1, color=mdl_colrs[mdlind])
                                                else:
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,:], NP.abs(psval.imag), ls='-', lw=1, marker='.', ms=1, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                                    axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='o', ms=2, color=mdl_colrs[mdlind])

                                    if dpool in xcpdps2_a[sampling]:
                                        psval = (1/3.0) * xcpdps2_a[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                        negind = psval.imag < 0.0
                                        posind = NP.logical_not(negind)
                                        maxabsvals += [NP.abs(psval.imag).max()]
                                        minabsvals += [NP.abs(psval.imag).min()]
                                    
                                        if sampling == 'oversampled':
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,posind], psval.imag[posind], ls='none', marker='.', ms=1, color='black', label='FG+N')
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='|', ms=1, color='black')
                                        else:
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], NP.abs(psval.imag), ls='-', lw=1, marker='.', ms=1, color='black', label='FG+N')
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='o', ms=2, color='black')

                                    legend = axs[dpoolind].legend(loc='upper right', shadow=False, fontsize=8)
                                    if trno == 0:
                                        axs[dpoolind].set_yscale('log')
                                        axs[dpoolind].set_xlim(0.99*xcpdps2_a[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a[sampling]['kprll'][zind,:].max())
                                        axs[dpoolind].text(0.05, 0.97, 'Imag', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.87, r'$z=$'+' {0:.1f}'.format(xcpdps2_a[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.77, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.67, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

                                    axt = axs[dpoolind].twiny()
                                    axt.set_xlim(1e6*xcpdps2_a[sampling]['lags'].min(), 1e6*xcpdps2_a[sampling]['lags'].max())
                                    # axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')
                                    
                                axs[dpoolind].set_ylim(0.5*min(minabsvals), 2*max(maxabsvals))
                            fig.subplots_adjust(top=0.85)
                            fig.subplots_adjust(bottom=0.16)
                            fig.subplots_adjust(left=0.22)
                            fig.subplots_adjust(right=0.98)
                                                
                            big_ax = fig.add_subplot(111)
                            big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                            # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                            big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                            big_ax.set_xticks([])
                            big_ax.set_yticks([])
                            big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                            if pspec_unit_type == 'K':
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=30)
                            else:
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=30)
        
                            big_axt = big_ax.twiny()
                            big_axt.set_xticks([])
                            big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
        
                            PLT.savefig(figdir + '{0}_log_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.png'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_log_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.pdf'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_log_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.eps'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)

        if '2b' in plots:
            for stat in statistic:
                for zind in spwind:
                    for lind in lstind:
                        for di,dind in enumerate(dayind):
                            maxabsvals = []
                            minabsvals = []
                            maxvals = []
                            minvals = []
                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                            if len(datapool) == 1:
                                axs = [axs]
                            for dpoolind,dpool in enumerate(datapool):
                                for trno,trind in enumerate([triadind[0]]):
                                    if model_hdf5files is not None:
                                        for mdlind, mdl in enumerate(model_labels):
                                            if dpool in xcpdps2_a_models[mdlind][sampling]:
                                                psval = (1/3.0) * xcpdps2_a_models[mdlind][sampling][dpool][stat][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to(pspec_unit).value

                                                # negind = psval.real < 0.0
                                                # posind = NP.logical_not(negind)
                                                maxabsvals += [NP.abs(psval.real).max()]
                                                minabsvals += [NP.abs(psval.real).min()]
                                                maxvals += [psval.real.max()]
                                                minvals += [psval.real.min()]
                                                axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,:], psval.real, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))

                                    if dpool in xcpdps2_a[sampling]:
                                        psval = (1/3.0) * xcpdps2_a[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                        psrms = (1/3.0) * NP.nanstd(xcpdps2_a_errinfo[sampling]['errinfo'][stat][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                        
                                        maxabsvals += [NP.abs(psval.real + psrms).max()]
                                        minabsvals += [NP.abs(psval.real).min()]
                                        maxvals += [(psval.real + psrms).max()]
                                        minvals += [(psval.real - psrms).min()]
                                    
                                        # axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], psval.real, ls='none', marker='.', ms=1, color='black', label='FG+N')
                                        axs[dpoolind].errorbar(xcpdps2_a[sampling]['kprll'][zind,:], psval.real, yerr=psrms, xerr=None, ecolor='0.8', ls='none', marker='.', ms=4, color='black', label='FG+N')
                                            
                                    legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
                                    if trno == 0:
                                        axs[dpoolind].text(0.05, 0.97, 'Real', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps2_a[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.92, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.87, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

                                    axt = axs[dpoolind].twiny()
                                    axt.set_xlim(1e6*xcpdps2_a[sampling]['lags'].min(), 1e6*xcpdps2_a[sampling]['lags'].max())

                                minvals = NP.asarray(minvals)
                                maxvals = NP.asarray(maxvals)
                                minabsvals = NP.asarray(minabsvals)
                                maxabsvals = NP.asarray(maxabsvals)
                                if xlim is None:
                                    axs[dpoolind].set_xlim(0.99*xcpdps2_a[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a[sampling]['kprll'][zind,:].max())
                                else:
                                    axs[dpoolind].set_xlim(xlim)
                                if NP.min(minvals) < 0.0:
                                    axs[dpoolind].set_ylim(1.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                else:
                                    axs[dpoolind].set_ylim(0.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
                                tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
                                axs[dpoolind].yaxis.set_major_locator(tickloc)
                                axs[dpoolind].grid(color='0.8', which='both', linestyle=':', lw=1)
                                
                            fig.subplots_adjust(top=0.85)
                            fig.subplots_adjust(bottom=0.16)
                            fig.subplots_adjust(left=0.22)
                            fig.subplots_adjust(right=0.98)
                                                
                            big_ax = fig.add_subplot(111)
                            big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                            # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                            big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                            big_ax.set_xticks([])
                            big_ax.set_yticks([])
                            big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                            if pspec_unit_type == 'K':
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
                            else:
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=40)
        
                            big_axt = big_ax.twiny()
                            big_axt.set_xticks([])
                            big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
        
                            # PLT.savefig(figdir + '{0}_symlog_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.png'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_symlog_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.pdf'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            # PLT.savefig(figdir + '{0}_symlog_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.eps'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)

                            maxabsvals = []
                            minabsvals = []
                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                            if len(datapool) == 1:
                                axs = [axs]
                            for dpoolind,dpool in enumerate(datapool):
                                for trno,trind in enumerate([triadind[0]]):
                                    if model_hdf5files is not None:
                                        for mdlind, mdl in enumerate(model_labels):
                                            if dpool in xcpdps2_a_models[mdlind][sampling]:
                                                psval = (1/3.0) * xcpdps2_a_models[mdlind][sampling][dpool][stat][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to(pspec_unit).value
                                                # negind = psval.imag < 0.0
                                                # posind = NP.logical_not(negind)
                                                maxabsvals += [NP.abs(psval.imag).max()]
                                                minabsvals += [NP.abs(psval.imag).min()]
                                                maxvals += [psval.imag.max()]
                                                minvals += [psval.imag.min()]
                                                axs[dpoolind].plot(xcpdps2_a_models[mdlind][sampling]['kprll'][zind,:], psval.imag, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))

                                    if dpool in xcpdps2_a[sampling]:
                                        psval = (1/3.0) * xcpdps2_a[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                        psrms = (1/3.0) * NP.nanstd(xcpdps2_a_errinfo[sampling]['errinfo'][stat][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                        maxabsvals += [NP.abs(psval.imag + psrms).max()]
                                        minabsvals += [NP.abs(psval.imag).min()]
                                        maxvals += [(psval.imag + psrms).max()]
                                        minvals += [(psval.imag - psrms).min()]
                                    
                                        # axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], psval.imag, ls='none', marker='.', ms=1, color='black', label='FG+N')
                                        axs[dpoolind].errorbar(xcpdps2_a[sampling]['kprll'][zind,:], psval.imag, yerr=psrms, xerr=None, ecolor='0.8', ls='none', marker='.', ms=4, color='black', label='FG+N')

                                    legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
                                    if trno == 0:
                                        axs[dpoolind].set_xlim(0.99*xcpdps2_a[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a[sampling]['kprll'][zind,:].max())
                                        axs[dpoolind].text(0.05, 0.97, 'Imag', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps2_a[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.92, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.87, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

                                    axt = axs[dpoolind].twiny()
                                    axt.set_xlim(1e6*xcpdps2_a[sampling]['lags'].min(), 1e6*xcpdps2_a[sampling]['lags'].max())
                                    
                                minvals = NP.asarray(minvals)
                                maxvals = NP.asarray(maxvals)
                                minabsvals = NP.asarray(minabsvals)
                                maxabsvals = NP.asarray(maxabsvals)
                                if min(minvals) < 0.0:
                                    axs[dpoolind].set_ylim(1.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                else:
                                    axs[dpoolind].set_ylim(0.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
                                tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
                                axs[dpoolind].yaxis.set_major_locator(tickloc)
                                axs[dpoolind].grid(color='0.8', which='both', linestyle=':', lw=1)
                            fig.subplots_adjust(top=0.85)
                            fig.subplots_adjust(bottom=0.16)
                            fig.subplots_adjust(left=0.22)
                            fig.subplots_adjust(right=0.98)
                                                
                            big_ax = fig.add_subplot(111)
                            big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                            # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                            big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                            big_ax.set_xticks([])
                            big_ax.set_yticks([])
                            big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                            if pspec_unit_type == 'K':
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
                            else:
                                big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=40)
        
                            big_axt = big_ax.twiny()
                            big_axt.set_xticks([])
                            big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
        
                            # PLT.savefig(figdir + '{0}_symlog_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.png'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_symlog_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.pdf'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
                            # PLT.savefig(figdir + '{0}_symlog_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}.eps'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str), bbox_inches=0)
            
        if ('2c' in plots) or ('2d' in plots):
            avg_incohax_a = plot_info['2c']['incohax_a']
            diagoffsets_incohax_a = plot_info['2c']['diagoffsets_a']
            diagoffsets_a = []
            avg_incohax_b = plot_info['2c']['incohax_b']
            diagoffsets_incohax_b = plot_info['2c']['diagoffsets_b']
            diagoffsets_b = []
            for combi,incax_comb in enumerate(avg_incohax_a):
                diagoffsets_a += [{}]
                for incaxind,incax in enumerate(incax_comb):
                    diagoffsets_a[-1][incax] = NP.asarray(diagoffsets_incohax_a[combi][incaxind])
            xcpdps2_a_avg, excpdps2_a_avg = BSP.incoherent_cross_power_spectrum_average(xcpdps2_a, excpdps=xcpdps2_a_errinfo, diagoffsets=diagoffsets_a)

            avg_xcpdps2_a_models = []
            avg_excpdps2_a_models = []

            for combi,incax_comb in enumerate(avg_incohax_b):
                diagoffsets_b += [{}]
                for incaxind,incax in enumerate(incax_comb):
                    diagoffsets_b[-1][incax] = NP.asarray(diagoffsets_incohax_b[combi][incaxind])
            
            # xcpdps2_b_avg, excpdps2_b_avg = BSP.incoherent_cross_power_spectrum_average(xcpdps2_b, excpdps=None, diagoffsets=diagoffsets_b)
            xcpdps2_b_avg, excpdps2_b_avg = BSP.incoherent_cross_power_spectrum_average(xcpdps2_b, excpdps=xcpdps2_b_errinfo, diagoffsets=diagoffsets_b)
            avg_xcpdps2_b_models = []
            avg_excpdps2_b_models = []

            if model_hdf5files is not None:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Models '.format(len(model_hdf5files)), PGB.ETA()], maxval=len(model_hdf5files)).start()
    
                for i in range(len(model_hdf5files)):
                    avg_xcpdps2_a_model, avg_excpdps2_a_model = BSP.incoherent_cross_power_spectrum_average(xcpdps2_a_models[i], excpdps=xcpdps2_a_errinfo_models[i], diagoffsets=diagoffsets_a)
                    avg_xcpdps2_a_models += [copy.deepcopy(avg_xcpdps2_a_model)]
                    avg_excpdps2_a_models += [copy.deepcopy(avg_excpdps2_a_model)]

                    # avg_xcpdps2_b_model, avg_excpdps2_b_model = BSP.incoherent_cross_power_spectrum_average(xcpdps2_b_models[i], excpdps=None, diagoffsets=diagoffsets_b)
                    avg_xcpdps2_b_model, avg_excpdps2_b_model = BSP.incoherent_cross_power_spectrum_average(xcpdps2_b_models[i], excpdps=xcpdps2_b_errinfo_models[i], diagoffsets=diagoffsets_b)
                    avg_xcpdps2_b_models += [copy.deepcopy(avg_xcpdps2_b_model)]
                    avg_excpdps2_b_models += [copy.deepcopy(avg_excpdps2_b_model)]

                    progress.update(i+1)
                progress.finish()

            # Save incoherent cross power average of the main dataset and its uncertainties
            xps_avg_outfile_b = datadir + dir_PS + outfile_pfx_b + '_' + infile_no_ext + '.npz'
            xpserr_avg_outfile_b = datadir + dir_PS + outfile_pfx_b + '_' + infile_no_ext + '_errinfo.npz'
            

            # if '2c' in plots:
            #     lstind = [0]
            #     triadind = [0]
            #     for stat in statistic:
            #         for zind in spwind:
            #             for lind in lstind:
            #                 for di,dind in enumerate(dayind):
            #                     for combi in range(len(diagoffsets)):
            #                         maxabsvals = []
            #                         minabsvals = []
            #                         maxvals = []
            #                         minvals = []
            #                         fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
            #                         if len(datapool) == 1:
            #                             axs = [axs]
            #                         for dpoolind,dpool in enumerate(datapool):
            #                             for trno,trind in enumerate(triadind):
            #                                 if model_hdf5files is not None:
            #                                     for mdlind, mdl in enumerate(model_labels):
            #                                         if dpool in avg_xcpdps2_a_models[mdlind][sampling]:
            #                                             psval = (1/3.0) * avg_xcpdps2_a_models[mdlind][sampling][dpool][stat][combi][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to(pspec_unit).value
            #                                             maxabsvals += [NP.abs(psval.real).max()]
            #                                             minabsvals += [NP.abs(psval.real).min()]
            #                                             maxvals += [psval.real.max()]
            #                                             minvals += [psval.real.min()]
            #                                             axs[dpoolind].plot(avg_xcpdps2_a_models[mdlind][sampling]['kprll'][zind,:], psval.real, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                    
            #                                 if dpool in xcpdps2_a_avg[sampling]:
            #                                     psval = (1/3.0) * xcpdps2_a_avg[sampling][dpool][stat][combi][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
            #                                     psrms = (1/3.0) * NP.nanstd(excpdps2_a_avg[sampling]['errinfo'][stat][combi][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                                
            #                                     maxabsvals += [NP.abs(psval.real + psrms).max()]
            #                                     minabsvals += [NP.abs(psval.real).min()]
            #                                     maxvals += [(psval.real + psrms).max()]
            #                                     minvals += [(psval.real - psrms).min()]
                                            
            #                                     axs[dpoolind].errorbar(xcpdps2_a_avg[sampling]['kprll'][zind,:], psval.real, yerr=psrms, xerr=None, ecolor='0.8', ls='none', marker='.', ms=4, color='black', label='FG+N')
                                                    
            #                                 legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
            #                                 if trno == 0:
            #                                     axs[dpoolind].text(0.05, 0.97, 'Real', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
            #                                     axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps2_a_avg[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
            #                                     axs[dpoolind].text(0.05, 0.92, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a_avg['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
            #                                     axs[dpoolind].text(0.05, 0.87, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                    
            #                                 axt = axs[dpoolind].twiny()
            #                                 axt.set_xlim(1e6*xcpdps2_a_avg[sampling]['lags'].min(), 1e6*xcpdps2_a_avg[sampling]['lags'].max())
                                    
            #                             minvals = NP.asarray(minvals)
            #                             maxvals = NP.asarray(maxvals)
            #                             minabsvals = NP.asarray(minabsvals)
            #                             maxabsvals = NP.asarray(maxabsvals)
            #                             if xlim is None:
            #                                 axs[dpoolind].set_xlim(0.99*xcpdps2_a_avg[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a_avg[sampling]['kprll'][zind,:].max())
            #                             else:
            #                                 axs[dpoolind].set_xlim(xlim)
            #                             if NP.min(minvals) < 0.0:
            #                                 axs[dpoolind].set_ylim(1.5*NP.min(minvals), 2*NP.max(maxabsvals))
            #                             else:
            #                                 axs[dpoolind].set_ylim(0.5*NP.min(minvals), 2*NP.max(maxabsvals))
            #                             axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
            #                             tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
            #                             axs[dpoolind].yaxis.set_major_locator(tickloc)
            #                             axs[dpoolind].grid(color='0.8', which='both', linestyle=':', lw=1)
                                        
            #                         fig.subplots_adjust(top=0.85)
            #                         fig.subplots_adjust(bottom=0.16)
            #                         fig.subplots_adjust(left=0.22)
            #                         fig.subplots_adjust(right=0.98)
                                                        
            #                         big_ax = fig.add_subplot(111)
            #                         big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
            #                         # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
            #                         big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            #                         big_ax.set_xticks([])
            #                         big_ax.set_yticks([])
            #                         big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
            #                         if pspec_unit_type == 'K':
            #                             big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
            #                         else:
            #                             big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=40)
                                    
            #                         big_axt = big_ax.twiny()
            #                         big_axt.set_xticks([])
            #                         big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
                                    
            #                         PLT.savefig(figdir + '{0}_symlog_incoh_avg_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}_comb_{9:0d}.pdf'.format(infile_no_ext, xcpdps2_a_avg[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a_avg['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a_avg[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str, combi), bbox_inches=0)
            #                         # PLT.savefig(figdir + '{0}_symlog_incoh_avg_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}_comb_{9:0d}.eps'.format(infile_no_ext, xcpdps2_a_avg[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a_avg['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a_avg[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str, combi), bbox_inches=0)

            if '2c' in plots:
                lstind = [0]
                triadind = [0]
                dayind = [0]
                dayind_models = NP.zeros(len(model_labels), dtype=int).reshape(1,-1)
                    
                for stat in statistic:
                    for zind in spwind:
                        for lind in lstind:
                            for di,dind in enumerate(dayind):
                                for combi in range(len(diagoffsets_b)):
                                    maxabsvals = []
                                    minabsvals = []
                                    maxvals = []
                                    minvals = []
                                    fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                                    if len(datapool) == 1:
                                        axs = [axs]
                                    for dpoolind,dpool in enumerate(datapool):
                                        for trno,trind in enumerate(triadind):
                                            if model_hdf5files is not None:
                                                for mdlind, mdl in enumerate(model_labels):
                                                    if dpool in avg_xcpdps2_b_models[mdlind][sampling]:
                                                        psval = (2/3.0) * avg_xcpdps2_b_models[mdlind][sampling][dpool][stat][combi][zind,lind,dayind_models[di][mdlind],trind,:].to(pspec_unit).value
                                                        maxabsvals += [NP.abs(psval.real).max()]
                                                        minabsvals += [NP.abs(psval.real).min()]
                                                        maxvals += [psval.real.max()]
                                                        minvals += [psval.real.min()]
                                                        axs[dpoolind].plot(avg_xcpdps2_b_models[mdlind][sampling]['kprll'][zind,:], psval.real, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                    
                                            if dpool in xcpdps2_b_avg[sampling]:
                                                psval = (2/3.0) * xcpdps2_b_avg[sampling][dpool][stat][combi][zind,lind,dind,trind,:].to(pspec_unit).value
                                                psrms_ssdiff = (2/3.0) * NP.nanstd(excpdps2_a_avg[sampling]['errinfo'][stat][combi][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                                if 2 in avg_incohax_b[combi]:
                                                    ind_dayax_in_incohax = avg_incohax_b[combi].index(2)
                                                    if 0 in diagoffsets_incohax_b[combi][ind_dayax_in_incohax]:
                                                        rms_inflation_factor = 2.0 * NP.sqrt(2.0)
                                                    else:
                                                        rms_inflation_factor = NP.sqrt(2.0)
                                                else:
                                                    rms_inflation_factor = NP.sqrt(2.0)
                                                psrms_psdiff = (2/3.0) * (xcpdps2_a_avg[sampling][dpool][stat][combi][zind,lind,1,1,trind,:] - xcpdps2_a_avg[sampling][dpool][stat][combi][zind,lind,0,0,trind,:]).to(pspec_unit).value
                                                psrms_psdiff = NP.abs(psrms_psdiff.real) / rms_inflation_factor

                                                psrms_max = NP.amax(NP.vstack((psrms_ssdiff, psrms_psdiff)), axis=0)
                                                
                                                maxabsvals += [NP.abs(psval.real + nsigma*psrms_max).max()]
                                                minabsvals += [NP.abs(psval.real).min()]
                                                maxvals += [(psval.real + nsigma*psrms_max).max()]
                                                minvals += [(psval.real - nsigma*psrms_max).min()]

                                                for errtype in ps_errtype:
                                                    if errtype.lower() == 'ssdiff':
                                                        axs[dpoolind].errorbar(xcpdps2_b_avg[sampling]['kprll'][zind,:], psval.real, yerr=nsigma*psrms_ssdiff, xerr=None, ecolor=errshade[errtype.lower()], ls='none', marker='.', ms=4, color='black')
                                                    elif errtype.lower() == 'psdiff':
                                                        axs[dpoolind].errorbar(xcpdps2_b_avg[sampling]['kprll'][zind,:], psval.real, yerr=nsigma*psrms_psdiff, xerr=None, ecolor=errshade[errtype.lower()], ls='none', marker='.', ms=4, color='black', label='FG+N')
                                                # axs[dpoolind].errorbar(xcpdps2_b_avg[sampling]['kprll'][zind,:], psval.real, yerr=psrms, xerr=None, ecolor='0.8', ls='none', marker='.', ms=4, color='black', label='FG+N')
                                                    
                                            legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
                                            if trno == 0:
                                                # axs[dpoolind].text(0.05, 0.97, 'Real', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                                axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps2_b_avg[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
                                                # axs[dpoolind].text(0.05, 0.92, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a_avg['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                                # axs[dpoolind].text(0.05, 0.87, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                    
                                            axt = axs[dpoolind].twiny()
                                            axt.set_xlim(1e6*xcpdps2_b_avg[sampling]['lags'].min(), 1e6*xcpdps2_b_avg[sampling]['lags'].max())
                                    
                                        axs[dpoolind].axhline(y=0, xmin=0, xmax=1, ls='-', lw=1, color='black')

                                        minvals = NP.asarray(minvals)
                                        maxvals = NP.asarray(maxvals)
                                        minabsvals = NP.asarray(minabsvals)
                                        maxabsvals = NP.asarray(maxabsvals)
                                        if xlim is None:
                                            axs[dpoolind].set_xlim(0.99*xcpdps2_b_avg[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_b_avg[sampling]['kprll'][zind,:].max())
                                        else:
                                            axs[dpoolind].set_xlim(xlim)
                                        if NP.min(minvals) < 0.0:
                                            axs[dpoolind].set_ylim(1.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                        else:
                                            axs[dpoolind].set_ylim(0.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                        axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
                                        tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
                                        axs[dpoolind].yaxis.set_major_locator(tickloc)
                                        axs[dpoolind].grid(color='0.8', which='both', linestyle=':', lw=1)
                                        
                                    fig.subplots_adjust(top=0.85)
                                    fig.subplots_adjust(bottom=0.16)
                                    fig.subplots_adjust(left=0.22)
                                    fig.subplots_adjust(right=0.98)
                                                        
                                    big_ax = fig.add_subplot(111)
                                    big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                                    # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                                    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                                    big_ax.set_xticks([])
                                    big_ax.set_yticks([])
                                    big_ax.set_xlabel(r'$\kappa_\parallel$'+' [pseudo '+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                                    if pspec_unit_type == 'K':
                                        big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(\kappa_\parallel)$ [pseudo mK$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
                                    else:
                                        big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(\kappa_\parallel)$ [pseudo Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=40)
                                    
                                    big_axt = big_ax.twiny()
                                    big_axt.set_xticks([])
                                    big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
                                    
                                    PLT.savefig(figdir + '{0}_symlog_incoh_avg_real_cpdps_err_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_flags_{5}_comb_{6:0d}.pdf'.format(infile_no_ext, xcpdps2_b_avg[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_b_avg['dlst'][0], applyflags_str, combi), bbox_inches=0)
                                    
            if '2d' in plots:
                kbin_min = plot_info['2d']['kbin_min']
                kbin_max = plot_info['2d']['kbin_max']
                num_kbins = plot_info['2d']['num_kbins']
                kbintype = plot_info['2d']['kbintype']
                if (kbin_min is None) or (kbin_max is None):
                    kbins = None
                else:
                    if num_kbins is None:
                        raise ValueError('Input num_kbins must be set if kbin range is provided')
                    if kbintype == 'linear':
                        kbins = NP.linspace(kbin_min, kbin_max, num=num_kbins, endpoint=True)
                    elif kbintype == 'log':
                        if kbin_min > 0.0:
                            kbins = NP.geomspace(kbin_min, kbin_max, num=num_kbins, endpoint=True)
                        elif kbin_min == 0.0:
                            eps_k = 1e-3
                            kbins = NP.geomspace(kbin_min+eps_k, kbin_max, num=num_kbins, endpoint=True)
                        else:
                            eps_k = 1e-3
                            kbins_pos = NP.geomspace(eps_k, kbin_max, num=num_kbins, endpoint=True)
                            ind_kbin_thresh = NP.argmin(kbins_pos[kbins_pos >= NP.abs(kbin_min)])
                            kbins_neg = -1 * kbins_pos[:ind_kbin_thresh+1][::-1]
                            kbins = NP.hstack((kbins_neg, kbins_pos))
                    else:
                        raise ValueError('Input kbintype must be set to "linear" or "log"')
                xcpdps2_a_avg_kbin = BSP.incoherent_kbin_averaging(xcpdps2_a_avg, kbins=kbins, kbintype=kbintype)
                excpdps2_a_avg_kbin = BSP.incoherent_kbin_averaging(excpdps2_a_avg, kbins=kbins, kbintype=kbintype)
                xcpdps2_a_avg_kbin_models = []
                excpdps2_a_avg_kbin_models = []

                xcpdps2_b_avg_kbin = BSP.incoherent_kbin_averaging(xcpdps2_b_avg, kbins=kbins, kbintype=kbintype)
                excpdps2_b_avg_kbin = BSP.incoherent_kbin_averaging(excpdps2_b_avg, kbins=kbins, kbintype=kbintype)
                xcpdps2_b_avg_kbin_models = []
                excpdps2_b_avg_kbin_models = []

                if model_hdf5files is not None:
                    for i in range(len(model_hdf5files)):
                        xcpdps2_a_avg_kbin_models += [BSP.incoherent_kbin_averaging(avg_xcpdps2_a_models[i], kbins=kbins, kbintype=kbintype)]
                        excpdps2_a_avg_kbin_models += [BSP.incoherent_kbin_averaging(avg_excpdps2_a_models[i], kbins=kbins, kbintype=kbintype)]
                        xcpdps2_b_avg_kbin_models += [BSP.incoherent_kbin_averaging(avg_xcpdps2_b_models[i], kbins=kbins, kbintype=kbintype)]
                        excpdps2_b_avg_kbin_models += [BSP.incoherent_kbin_averaging(avg_excpdps2_b_models[i], kbins=kbins, kbintype=kbintype)]
                    
                lstind = [0]
                triadind = [0]
                dayind = [0]
                dayind_models = NP.zeros(len(model_labels), dtype=int).reshape(1,-1)
    
                for stat in statistic:
                    for zind in spwind:
                        for lind in lstind:
                            for di,dind in enumerate(dayind):
                                for pstype in ['PS', 'Del2']:
                                    for combi in range(len(diagoffsets_b)):
                                        maxabsvals = []
                                        minabsvals = []
                                        maxvals = []
                                        minvals = []
                                        fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                                        if len(datapool) == 1:
                                            axs = [axs]
                                        for dpoolind,dpool in enumerate(datapool):
                                            for trno,trind in enumerate(triadind):
                                                if model_hdf5files is not None:
                                                    for mdlind, mdl in enumerate(model_labels):
                                                        if dpool in xcpdps2_b_avg_kbin_models[mdlind][sampling]:
                                                            if pstype == 'PS':
                                                                psval = (2/3.0) * xcpdps2_b_avg_kbin_models[mdlind][sampling][dpool][stat][pstype][combi][zind,lind,dayind_models[di][mdlind],trind,:].to(pspec_unit).value
                                                                # psval = (2/3.0) * xcpdps2_a_avg_kbin_models[mdlind][sampling][dpool][stat][pstype][combi][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to(pspec_unit).value
                                                            else:
                                                                psval = (2/3.0) * xcpdps2_b_avg_kbin_models[mdlind][sampling][dpool][stat][pstype][combi][zind,lind,dayind_models[di][mdlind],trind,:].to('mK2').value
                                                                # psval = (2/3.0) * xcpdps2_a_avg_kbin_models[mdlind][sampling][dpool][stat][pstype][combi][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to('K2').value
                                                            kval = xcpdps2_b_avg_kbin_models[mdlind][sampling]['kbininfo'][dpool][stat][combi][zind,lind,dayind_models[di][mdlind],trind,:].to('Mpc-1').value
                                                            # kval = xcpdps2_a_avg_kbin_models[mdlind][sampling]['kbininfo'][dpool][stat][combi][zind,lind,dayind_models[di][mdlind][0],dayind_models[di][mdlind][1],trind,:].to('Mpc-1').value
                                                            maxabsvals += [NP.nanmin(NP.abs(psval.real))]
                                                            minabsvals += [NP.nanmin(NP.abs(psval.real))]
                                                            maxvals += [NP.nanmax(psval.real)]
                                                            minvals += [NP.nanmin(psval.real)]
                                                            axs[dpoolind].plot(kval, psval.real, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                
                                                if dpool in xcpdps2_b_avg_kbin[sampling]:
                                                    if pstype == 'PS':
                                                        psval = (2/3.0) * xcpdps2_b_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,dind,trind,:].to(pspec_unit).value
                                                        psrms_ssdiff = (2/3.0) * NP.nanstd(excpdps2_b_avg_kbin[sampling]['errinfo'][stat][pstype][combi][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                                        psrms_psdiff = (2/3.0) * (xcpdps2_a_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,1,1,trind,:] - xcpdps2_a_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,0,0,trind,:]).to(pspec_unit).value
                                                        # psval = (2/3.0) * xcpdps2_a_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                                        # psrms = (2/3.0) * NP.nanstd(excpdps2_a_avg_kbin[sampling]['errinfo'][stat][pstype][combi][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                                    else:
                                                        psval = (2/3.0) * xcpdps2_b_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,dind,trind,:].to('mK2').value
                                                        psrms_ssdiff = (2/3.0) * NP.nanstd(excpdps2_b_avg_kbin[sampling]['errinfo'][stat][pstype][combi][zind,lind,:,trind,:], axis=0).to('mK2').value
                                                        psrms_psdiff = (1/3.0) * (xcpdps2_a_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,1,1,trind,:] - xcpdps2_a_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,0,0,trind,:]).to('K2').value

                                                        # psval = (2/3.0) * xcpdps2_a_avg_kbin[sampling][dpool][stat][pstype][combi][zind,lind,dind[0],dind[1],trind,:].to('mK2').value
                                                        # psrms = (2/3.0) * NP.nanstd(excpdps2_a_avg_kbin[sampling]['errinfo'][stat][pstype][combi][zind,lind,:,trind,:], axis=0).to('mK2').value
                                                    if 2 in avg_incohax_b[combi]:
                                                        ind_dayax_in_incohax = avg_incohax_b[combi].index(2)
                                                        if 0 in diagoffsets_incohax_b[combi][ind_dayax_in_incohax]:
                                                            rms_inflation_factor = 2.0 * NP.sqrt(2.0)
                                                        else:
                                                            rms_inflation_factor = NP.sqrt(2.0)
                                                    else:
                                                        rms_inflation_factor = NP.sqrt(2.0)
                                                    psrms_psdiff = NP.abs(psrms_psdiff.real) / rms_inflation_factor
                                                    psrms_max = NP.amax(NP.vstack((psrms_ssdiff, psrms_psdiff)), axis=0)

                                                    kval = xcpdps2_b_avg_kbin[sampling]['kbininfo'][dpool][stat][combi][zind,lind,dind,trind,:].to('Mpc-1').value
                                                    # kval = xcpdps2_a_avg_kbin[sampling]['kbininfo'][dpool][stat][combi][zind,lind,dind[0],dind[1],trind,:].to('Mpc-1').value
                                                    
                                                    maxabsvals += [NP.nanmax(NP.abs(psval.real + nsigma*psrms_max.real))]
                                                    minabsvals += [NP.nanmin(NP.abs(psval.real))]
                                                    maxvals += [NP.nanmax(psval.real + nsigma*psrms_max.real)]
                                                    minvals += [NP.nanmin(psval.real - nsigma*psrms_max.real)]
                                                    for errtype in ps_errtype:
                                                        if errtype.lower() == 'ssdiff':
                                                            axs[dpoolind].errorbar(kval, psval.real, yerr=nsigma*psrms_ssdiff, xerr=None, ecolor=errshade[errtype.lower()], ls='none', marker='.', ms=4, color='black')
                                                        elif errtype.lower() in 'psdiff':
                                                            axs[dpoolind].errorbar(kval, psval.real, yerr=nsigma*psrms_psdiff, xerr=None, ecolor=errshade[errtype.lower()], ls='none', marker='.', ms=4, color='black', label='FG+N')
                                                    # axs[dpoolind].errorbar(kval, psval.real, yerr=psrms, xerr=None, ecolor='0.8', ls='none', marker='.', ms=4, color='black', label='FG+N')
                                                legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
                                                if trno == 0:
                                                    # axs[dpoolind].text(0.05, 0.97, 'Real', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                                    axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps2_b_avg_kbin['resampled']['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
                                                    # axs[dpoolind].text(0.05, 0.92, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a_avg_kbin['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                                    # axs[dpoolind].text(0.05, 0.87, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                            
                                            axs[dpoolind].axhline(y=0, xmin=0, xmax=1, ls='-', lw=1, color='black')

                                            minvals = NP.asarray(minvals)
                                            maxvals = NP.asarray(maxvals)
                                            minabsvals = NP.asarray(minabsvals)
                                            maxabsvals = NP.asarray(maxabsvals)
                                            axs[dpoolind].set_xlim(0.99*NP.nanmin(xcpdps2_b_avg_kbin['resampled']['kbininfo']['kbin_edges'][zind].to('Mpc-1').value), 1.01*NP.nanmax(xcpdps2_b_avg_kbin['resampled']['kbininfo']['kbin_edges'][zind].to('Mpc-1').value))
                                            if NP.min(minvals) < 0.0:
                                                axs[dpoolind].set_ylim(1.5*NP.nanmin(minvals), 2*NP.nanmax(maxabsvals))
                                            else:
                                                axs[dpoolind].set_ylim(0.5*NP.nanmin(minvals), 2*NP.nanmax(maxabsvals))
                                            axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
                                            tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
                                            axs[dpoolind].yaxis.set_major_locator(tickloc)
                                            axs[dpoolind].grid(color='0.8', which='both', linestyle=':', lw=1)
    
                                        fig.subplots_adjust(top=0.85)
                                        fig.subplots_adjust(bottom=0.16)
                                        fig.subplots_adjust(left=0.22)
                                        fig.subplots_adjust(right=0.98)
                                                            
                                        big_ax = fig.add_subplot(111)
                                        big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                                        # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                                        big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                                        big_ax.set_xticks([])
                                        big_ax.set_yticks([])
                                        big_ax.set_xlabel(r'$\kappa_\parallel$'+' [pseudo '+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                                        if pstype == 'PS':
                                            big_ax.set_ylabel(r'$\frac{1}{3}\, P_\nabla(\kappa_\parallel)$ [pseudo mK$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
                                        else:
                                            big_ax.set_ylabel(r'$\frac{1}{3}\, \Delta_\nabla^2(\kappa_\parallel)$ [pseudo mK$^2$]', fontsize=12, weight='medium', labelpad=40)
                                        
                                        big_axt = big_ax.twiny()
                                        big_axt.set_xticks([])
                                        big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
                                        if pstype == 'PS':
                                            PLT.savefig(figdir + '{0}_symlog_incoh_kbin_avg_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_flags_{5}_comb_{6:0d}.pdf'.format(infile_no_ext, xcpdps2_a_avg_kbin[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_b_avg_kbin['dlst'][0], applyflags_str, combi), bbox_inches=0)
                                            # PLT.savefig(figdir + '{0}_symlog_incoh_kbin_avg_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}_comb_{9:0d}.pdf'.format(infile_no_ext, xcpdps2_a_avg_kbin[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a_avg_kbin['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a_avg_kbin[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str, combi), bbox_inches=0)
                                        else:
                                            PLT.savefig(figdir + '{0}_symlog_incoh_kbin_avg_real_cpDel2_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_flags_{5}_comb_{6:0d}.pdf'.format(infile_no_ext, xcpdps2_a_avg_kbin[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_b_avg_kbin['dlst'][0], applyflags_str, combi), bbox_inches=0)
                                            # PLT.savefig(figdir + '{0}_symlog_incoh_kbin_avg_real_cpDel2_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_{7:0d}_flags_{8}_comb_{9:0d}.pdf'.format(infile_no_ext, xcpdps2_a_avg_kbin[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a_avg_kbin['dlst'][0], subselection['lstdiag'][lind], dind, xcpdps2_a_avg_kbin[sampling][datapool[0]]['diagoffsets'][3][trind], applyflags_str, combi), bbox_inches=0)

        if '2e' in plots:
            subselection = plot_info['2e']['subselection']
            autoinfo = {'axes': cohax}
            xinfo = {'axes': incohax, 'avgcov': False, 'collapse_axes': collapseax, 'dlst_range': timetriad_selection['dlst_range']}

            if statistic is None:
                statistic = ['mean', 'median']
            else:
                statistic = [statistic]
                
            spw = subselection['spw']
            if spw is None:
                spwind = NP.arange(xcpdps2_a[sampling]['z'].size)
            else:
                spwind = NP.asarray(spw)
            lstind = NMO.find_list_in_list(xcpdps2_a[sampling][datapool[0]]['diagoffsets'][1], NP.asarray(subselection['lstdiag']))
            dayind = NP.asarray(subselection['day'])
            triadind = NMO.find_list_in_list(xcpdps2_a[sampling][datapool[0]]['diagoffsets'][3], NP.asarray(subselection['triaddiag']))

            colrs = ['red', 'green', 'blue', 'cyan', 'gray', 'orange']
            for stat in statistic:
                for zind in spwind:
                    for lind in lstind:
                        for dind in dayind:
                            maxabsvals = []
                            minabsvals = []
                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                            if len(datapool) == 1:
                                axs = [axs]
                            for dpoolind,dpool in enumerate(datapool):
                                for trno,trind in enumerate(triadind):
                                    if dpool in xcpdps2_a[sampling]:
                                        psval = xcpdps2_a[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                        negind = psval.real < 0.0
                                        posind = NP.logical_not(negind)
                                        maxabsvals += [NP.abs(psval.real).max()]
                                        minabsvals += [NP.abs(psval.real).min()]
                                    
                                        if sampling == 'oversampled':
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,posind], psval.real[posind], ls='none', marker='.', ms=1, color=colrs[trno], label=r'$\Delta$Tr={0:0d}'.format(subselection['triaddiag'][trno]))
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='|', ms=1, color=colrs[trno])
                                        else:
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], NP.abs(psval.real), ls='-', lw=1, marker='.', ms=1, color=colrs[trno], label=r'$\Delta$Tr={0:0d}'.format(subselection['triaddiag'][trno]))
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='o', ms=2, color=colrs[trno])
                                            
                                    legend = axs[dpoolind].legend(loc='upper right', shadow=False, fontsize=8)
                                    if trno == 0:
                                        axs[dpoolind].set_yscale('log')
                                        axs[dpoolind].set_xlim(0.99*xcpdps2_a[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a[sampling]['kprll'][zind,:].max())
                                        axs[dpoolind].set_ylim(1e-3, 1e8)

                                        axs[dpoolind].text(0.05, 0.97, 'Real', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.87, r'$z=$'+' {0:.1f}'.format(xcpdps2_a[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.77, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.67, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

                                    axt = axs[dpoolind].twiny()
                                    axt.set_xlim(1e6*xcpdps2_a[sampling]['lags'].min(), 1e6*xcpdps2_a[sampling]['lags'].max())
                                    # axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')
                                    
                            fig.subplots_adjust(top=0.85)
                            fig.subplots_adjust(bottom=0.16)
                            fig.subplots_adjust(left=0.24)
                            fig.subplots_adjust(right=0.98)
                                                
                            big_ax = fig.add_subplot(111)
                            big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                            # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                            big_ax.set_xticks([])
                            big_ax.set_yticks([])
                            big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                            if pspec_unit_type == 'K':
                                big_ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=30)
                            else:
                                big_ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=30)
        
                            big_axt = big_ax.twiny()
                            big_axt.set_xticks([])
                            big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
        
                            # PLT.savefig(figdir + '{0}_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_flags_{7}.png'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_flags_{7}.pdf'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, applyflags_str), bbox_inches=0)
                            # PLT.savefig(figdir + '{0}_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_flags_{7}.eps'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, applyflags_str), bbox_inches=0)

                            maxabsvals = []
                            minabsvals = []
                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                            if len(datapool) == 1:
                                axs = [axs]
                            for dpoolind,dpool in enumerate(datapool):
                                for trno,trind in enumerate(triadind):
                                    if dpool in xcpdps2_a[sampling]:
                                        psval = xcpdps2_a[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
                                        negind = psval.imag < 0.0
                                        posind = NP.logical_not(negind)
                                        maxabsvals += [NP.abs(psval.imag).max()]
                                        minabsvals += [NP.abs(psval.imag).min()]
                                    
                                        if sampling == 'oversampled':
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,posind], psval.imag[posind], ls='none', marker='.', ms=1, color=colrs[trno], label=r'$\Delta$Tr={0:0d}'.format(subselection['triaddiag'][trno]))
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='|', ms=1, color=colrs[trno])
                                        else:
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], NP.abs(psval.imag), ls='-', lw=1, marker='.', ms=1, color=colrs[trno], label=r'$\Delta$Tr={0:0d}'.format(subselection['triaddiag'][trno]))
                                            axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='o', ms=2, color=colrs[trno])

                                        axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,posind], psval.imag[posind], ls='none', marker='.', ms=1, color=colrs[trno], label=r'$\Delta$Tr={0:0d}'.format(subselection['triaddiag'][trno]))
                                        axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,negind], NP.abs(psval.imag[negind]), ls='none', marker='|', ms=1, color=colrs[trno])
                                        # axs[dpoolind].plot(xcpdps2_a[sampling]['kprll'][zind,:], NP.abs(psval), ls='-', lw=0.5, color=colrs[trno])
                                        
                                    legend = axs[dpoolind].legend(loc='upper right', shadow=False, fontsize=8)
                                    if trno == 0:
                                        axs[dpoolind].set_yscale('log')
                                        axs[dpoolind].set_xlim(0.99*xcpdps2_a[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2_a[sampling]['kprll'][zind,:].max())
                                        axs[dpoolind].set_ylim(1e-3, 1e8)

                                        axs[dpoolind].text(0.05, 0.97, 'Imag', transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.87, r'$z=$'+' {0:.1f}'.format(xcpdps2_a[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.77, r'$\Delta$'+'LST = {0:.1f} s'.format(lind*3.6e3*xcpdps2_a['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                        axs[dpoolind].text(0.05, 0.67, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

                                    axt = axs[dpoolind].twiny()
                                    axt.set_xlim(1e6*xcpdps2_a[sampling]['lags'].min(), 1e6*xcpdps2_a[sampling]['lags'].max())
                                    # axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')
                                    
                            fig.subplots_adjust(top=0.85)
                            fig.subplots_adjust(bottom=0.16)
                            fig.subplots_adjust(left=0.24)
                            fig.subplots_adjust(right=0.98)
                                                
                            big_ax = fig.add_subplot(111)
                            big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                            # big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
                            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
                            big_ax.set_xticks([])
                            big_ax.set_yticks([])
                            big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
                            if pspec_unit_type == 'K':
                                big_ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [K$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=30)
                            else:
                                big_ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=30)
        
                            big_axt = big_ax.twiny()
                            big_axt.set_xticks([])
                            big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
        
                            # PLT.savefig(figdir + '{0}_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_flags_{7}.png'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, applyflags_str), bbox_inches=0)
                            PLT.savefig(figdir + '{0}_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_flags_{7}.pdf'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, applyflags_str), bbox_inches=0)
                            # PLT.savefig(figdir + '{0}_imag_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_lstdiag_{5:0d}_day_{6[0]:0d}_{6[1]:0d}_triaddiags_flags_{7}.eps'.format(infile_no_ext, xcpdps2_a[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps2_a['dlst'][0], subselection['lstdiag'][lind], dind, applyflags_str), bbox_inches=0)
                            


                            
                    # PLT.savefig(figdir + '{0}_closure_phase_delay_power_spectra_{1}_{2}_triads_{3}x{4:.1f}sx{5:.1f}d_{6}_statistic_nsamples_incoh_{7}_flags_{8}.png'.format(infile_no_ext, sampling, xcpdps2_a['triads_ind'].size, xcpdps2_a['lst'].size, 3.6e3*xcpdps2_a['dlst'][0], xcpdps2_a['dday'][0], stat, nsamples_incoh, applyflags_str), bbox_inches=0)
                    # PLT.savefig(figdir + '{0}_closure_phase_delay_power_spectra_{1}_{2}_triads_{3}x{4:.1f}sx{5:.1f}d_{6}_statistic_nsamples_incoh_{7}_flags_{8}.eps'.format(infile_no_ext, sampling, xcpdps2_a['triads_ind'].size, xcpdps2_a['lst'].size, 3.6e3*xcpdps2_a['dlst'][0], xcpdps2_a['dday'][0], stat, nsamples_incoh, applyflags_str), bbox_inches=0)

        # if '2f' in plots:
        #     antloc_file = plot_info['2f']['antloc_file']
        #     anttable = ascii.read(antloc_file)
        #     ant_E = anttable['East']
        #     ant_N = anttable['North']
        #     ant_U = anttable['Up']
        #     antlocs = NP.concatenate((ant_E.reshape(-1,1), ant_N.reshape(-1,1), ant_U.reshape(-1,1)))
        #     antnums = NP.arange(len(anttable))

        #     selection = plot_info['2f']['selection']
        #     for key in selection:
        #         if selection[key] is not None:
        #             if key == 'triads':
        #                 selection[key] = map(tuple,selection[key])
        #             else:
        #                 selection[key] = NP.asarray(selection[key])

        #     subselection = plot_info['2f']['subselection']
        #     statistic = plot_info['2f']['statistic']
        #     datapool = plot_info['2f']['datapool']

        #     cohax = plot_info['2f']['cohax']
        #     incohax = plot_info['2f']['incohax']
        #     collapseax = plot_info['2f']['collapseax']
        #     autoinfo = {'axes': cohax}
        #     xinfo = {'axes': incohax, 'avgcov': False, 'collapse_axes': collapseax, 'dlst_range': selection['dlst_range']}

        #     xcpdps2f = cpDSobj.compute_power_spectrum_new(selection=selection, autoinfo=autoinfo, xinfo=xinfo)
        #     nsamples_incoh = xcpdps2f[sampling]['whole']['nsamples_incoh']
        #     nsamples_coh = xcpdps2f[sampling]['whole']['nsamples_coh']

        #     if statistic is None:
        #         statistic = 'mean'
                
        #     spw = subselection['spw']
        #     if spw is None:
        #         spwind = NP.arange(xcpdps[sampling]['z'])
        #     else:
        #         spwind = NP.asarray(spw)
        #     lstind = NMO.find_list_in_list(xcpdps2f[sampling][datapool[0]]['diagoffsets'][1], NP.asarray(subselection['lstdiag']))
        #     dayind = NP.asarray(subselection['day'])

        #     tau_ind = NP.where(NP.logical_and(NP.abs(1e6*xcpdps2f[sampling]['lags']) >= 0.6, NP.abs(1e6*xcpdps2f[sampling]['lags']) <= 1.5))[0]

        #     colrs = ['red', 'green', 'blue', 'cyan', 'orange', 'gray']
    
        #     for stat in statistic:
        #         for zind in spwind:
        #             for lind in lstind:
        #                 for dind in dayind:
        #                     fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(2.4*len(datapool), 3.6))
        #                     if len(datapool) == 1:
        #                         axs = [axs]
        #                     for dpoolind,dpool in enumerate(datapool):
        #                         peak12_ratio = NP.max(NP.abs(xcpdps2f[sampling][dpool][stat][zind,lind,:,:,:]), axis=-1) / NP.max(NP.abs(xcpdps2f[sampling][dpool][stat][zind,lind,:,:,tau_ind]), axis=-1)
        #                         for trno1 in NP.arange(xcpdps2f['triads'].size):
        #                             for trno2 in NP.range(trno1, xcpdps2f['triads'].size):
        #                                 tr1_antinds = NMO.find_list_in_list(antnums, xcpdps2f['triads'][trind])
        #                                 tr1_antinds = NMO.find_list_in_list(antnums, xcpdps2f['triads'][trind])



        #                             if dpool in xcpdps2f[sampling]:
        #                                 psval = xcpdps2f[sampling][dpool][stat][zind,lind,dind[0],dind[1],trind,:].to(pspec_unit).value
        #                                 negind = psval.real < 0.0
        #                                 posind = NP.logical_not(negind)
                                    
        #                                 axs[dpoolind].plot(xcpdps2f[sampling]['kprll'][zind,posind], psval.real[posind], ls='none', marker='.', ms=1, color=colrs[trno], label=r'$\Delta$Tr={0:0d}'.format(subselection['triaddiag'][trno]))
        #                                 axs[dpoolind].plot(xcpdps2f[sampling]['kprll'][zind,negind], NP.abs(psval.real[negind]), ls='none', marker='|', ms=1, color=colrs[trno])
        #                                 axs[dpoolind].plot(xcpdps2f[sampling]['kprll'][zind,:], NP.abs(psval), ls='-', lw=0.5, color=colrs[trno])
                                        
        #                             axs[dpoolind].set_yscale('log')
        #                             axs[dpoolind].set_xlim(0.99*xcpdps2f[sampling]['kprll'][zind,:].min(), 1.01*xcpdps2f[sampling]['kprll'][zind,:].max())
        #                             axs[dpoolind].set_ylim(1e-3, 1e8)
        #                             legend = axs[dpoolind].legend(loc='upper right', shadow=False, fontsize=8)
        #                             axs[dpoolind].text(0.05, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps2f[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
        #                             axs[dpoolind].text(0.05, 0.87, r'$\Delta$'+'LST = {0:.1f} s'.format(3.6e3*xcpdps2f['dlst'][0]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
        #                             axs[dpoolind].text(0.05, 0.77, 'G{0[0]:0d}{0[1]:0d}'.format(dind), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')

        #                             axt = axs[dpoolind].twiny()
        #                             axt.set_xlim(1e6*xcpdps2f[sampling]['lags'].min(), 1e6*xcpdps2f[sampling]['lags'].max())
        #                             # axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')
                                    
        #                     fig.subplots_adjust(top=0.85)
        #                     fig.subplots_adjust(bottom=0.16)
        #                     fig.subplots_adjust(left=0.24)
        #                     fig.subplots_adjust(right=0.98)
                                                
        #                     big_ax = fig.add_subplot(111)
        #                     # big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
        #                     big_ax.set_axis_bgcolor('none') # matplotlib.__version__ < 2.0.0
        #                     big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        #                     big_ax.set_xticks([])
        #                     big_ax.set_yticks([])
        #                     big_ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium', labelpad=20)
        #                     big_ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=35)
        
        #                     big_axt = big_ax.twiny()
        #                     big_axt.set_xticks([])
        #                     big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)






        #     colrs = ['red', 'green', 'blue']
    
        #     for stat in statistic:
        #         for dpool in ['whole', 'submodel', 'residual']:
        #             if dpool in xcpdps[sampling]:
        #                 psval = NP.mean(xcpdps[sampling][dpool][stat], axis=tuple(axes_to_avg))
        #                 fig = PLT.figure(figsize=(3.5,3.5))
        #                 ax = fig.add_subplot(111)
        #                 for zind,z in enumerate(xcpdps[sampling]['z']):
        #                     negind = psval[zind,:] < 0.0
        #                     posind = NP.logical_not(negind)
                        
        #                     ax.plot(xcpdps[sampling]['kprll'][zind,posind], psval[zind,posind], ls='none', marker='.', ms=4, color=colrs[zind], label=r'$z$={0:.1f}'.format(z))
        #                     ax.plot(xcpdps[sampling]['kprll'][zind,negind], NP.abs(psval[zind,negind]), ls='none', marker='|', ms=4, color=colrs[zind])
        #                 ax.set_yscale('log')
        #                 ax.set_xlim(0.99*xcpdps[sampling]['kprll'][zind,:].min(), 1.01*xcpdps[sampling]['kprll'][zind,:].max())
        #                 ax.set_ylim(1e-3, 1e8)
        #                 ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium')
        #                 ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=0)
        #                 legend = ax.legend(loc='upper right', shadow=False, fontsize=10)
        #                 axt = ax.twiny()
        #                 axt.set_xlim(1e6*xcpdps[sampling]['lags'].min(), 1e6*xcpdps[sampling]['lags'].max())
        #                 axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')
                        
        #                 fig.subplots_adjust(top=0.85)
        #                 fig.subplots_adjust(bottom=0.16)
        #                 fig.subplots_adjust(left=0.2)
        #                 fig.subplots_adjust(right=0.98)
                                    
        #                 PLT.savefig(figdir + '{0}_closure_phase_delay_power_spectra_{1}_{2}_triads_{3}x{4:.1f}sx{5:.1f}d_{6}_statistic_nsamples_incoh_{7}_flags_{8}.png'.format(infile_no_ext, sampling, xcpdps['triads_ind'].size, xcpdps['lst'].size, 3.6e3*xcpdps['dlst'][0], xcpdps['dday'][0], stat, nsamples_incoh, applyflags_str), bbox_inches=0)
        #                 PLT.savefig(figdir + '{0}_closure_phase_delay_power_spectra_{1}_{2}_triads_{3}x{4:.1f}sx{5:.1f}d_{6}_statistic_nsamples_incoh_{7}_flags_{8}.eps'.format(infile_no_ext, sampling, xcpdps['triads_ind'].size, xcpdps['lst'].size, 3.6e3*xcpdps['dlst'][0], xcpdps['dday'][0], stat, nsamples_incoh, applyflags_str), bbox_inches=0)

        # # for stat in statistic:
        # #     fig = PLT.figure(figsize=(3.5,3.5))
        # #     ax = fig.add_subplot(111)
        # #     for zind,z in enumerate(xcpdps[sampling]['z']):
        # #         if len(avgax) > 0:
        # #             psval = NP.mean(xcpdps[sampling][stat], axis=tuple(avgax), keepdims=True)
        # #         else:
        # #             psval = NP.copy(xcpdps[sampling][stat])
        # #         negind = psval[zind,lstind,dayind,triadind,:] < 0.0
        # #         posind = NP.logical_not(negind)

        # #         ax.plot(xcpdps[sampling]['kprll'][zind,posind], psval[zind,lstind,dayind,triadind,posind], ls='none', marker='.', ms=4, color=colrs[zind], label=r'$z$={0:.1f}'.format(z))
        # #         ax.plot(xcpdps[sampling]['kprll'][zind,negind], NP.abs(psval[zind,lstind,dayind,triadind,negind]), ls='none', marker='|', ms=4, color=colrs[zind])
        # #     ax.set_yscale('log')
        # #     ax.set_xlim(0.99*xcpdps[sampling]['kprll'][zind,:].min(), 1.01*xcpdps[sampling]['kprll'][zind,:].max())
        # #     ax.set_ylim(1e-8, 1e2)
        # #     ax.set_xlabel(r'$k_\parallel$'+' ['+r'$h$'+' Mpc'+r'$^{-1}$'+']', fontsize=12, weight='medium')
        # #     ax.set_ylabel(r'$P_\nabla(k_\parallel)$ [$h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=0)
        # #     legend = ax.legend(loc='upper right', shadow=False, fontsize=10)
        # #     axt = ax.twiny()
        # #     axt.set_xlim(1e6*xcpdps[sampling]['lags'].min(), 1e6*xcpdps[sampling]['lags'].max())
        # #     axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium')
        
        # #     fig.subplots_adjust(top=0.85)
        # #     fig.subplots_adjust(bottom=0.16)
        # #     fig.subplots_adjust(left=0.2)
        # #     fig.subplots_adjust(right=0.98)
                        
        # #     PLT.savefig(figdir + '{0}_closure_phase_delay_power_spectra_{1}_{2}_triads_{3}x{4:.1f}sx{5:.1f}d_{6}_statistic_nsamples_incoh_{7}_flags_{8}.png'.format(infile_no_ext, sampling, xcpdps['triads_ind'].size, xcpdps['lst'].size, 3.6e3*xcpdps['dlst'][0], xcpdps['dday'][0], stat, nsamples_incoh, applyflags_str), bbox_inches=0)
        # #     PLT.savefig(figdir + '{0}_closure_phase_delay_power_spectra_{1}_{2}_triads_{3}x{4:.1f}sx{5:.1f}d_{6}_statistic_nsamples_incoh_{7}_flags_{8}.eps'.format(infile_no_ext, sampling, xcpdps['triads_ind'].size, xcpdps['lst'].size, 3.6e3*xcpdps['dlst'][0], xcpdps['dday'][0], stat, nsamples_incoh, applyflags_str), bbox_inches=0)
            
    if ('3' in plots) or ('3a' in plots) or ('3b' in plots) or ('3c' in plots):

        HI_PS_dir = plot_info['3']['21cm_PS_dir']
        sim_rootdir = plot_info['3']['sim_rootdir']
        visdirs = plot_info['3']['visdirs']
        simvisdirs = [sim_rootdir+visdir for visdir in visdirs]
        simlabels = plot_info['3']['simlabels']
        visfile_prefix = plot_info['3']['visfile_prfx']

        theory_HI_PS_files = glob.glob(HI_PS_dir+'ps_*')
        z_theory_HI_PS_files = NP.asarray([fname.split('/')[-1].split('_')[3].split('z')[1] for fname in theory_HI_PS_files], dtype=NP.float)
        h_Planck15 = DS.cosmoPlanck15.h

        z_freq_window_centers = CNST.rest_freq_HI / freq_window_centers - 1.0
        psfile_inds = [NP.argmin(NP.abs(z_theory_HI_PS_files - z_freq_window_center)) for z_freq_window_center in z_freq_window_centers]

        simvis_objs = [RI.InterferometerArray(None, None, None, init_file=simvisdir+visfile_prefix) for simvisdir in simvisdirs]

        select_lst = plot_info['3']['lst']
        simlst = (simvis_objs[0].lst / 15.0) # in hours
        if select_lst is None:
            lstind = NP.asarray(NP.floor(simlst.size/2.0).astype(int)).reshape(-1)
        elif isinstance(select_lst, (int,float)):
            lstind = NP.asarray(NP.argmin(NP.abs(simlst - select_lst))).reshape(-1)
        elif isinstance(select_lst, list):
            lstind = NP.asarray([NP.argmin(NP.abs(simlst - select_lst[i])) for i in range(len(select_lst))])
        else:
            raise TypeError('Invalid format for selecting LST')

        sysT = plot_info['3']['Tsys']

        if '3a' in plots:
            spw = plot_info['3a']['spw']
            if spw is not None:
                spwind = NP.asarray(spw).reshape(-1)

            blvects = NP.asarray(plot_info['3a']['bl'])
            bll = NP.sqrt(NP.sum(blvects**2, axis=1))
            blo = NP.degrees(NP.arctan2(blvects[:,1], blvects[:,0]))
            bltol = plot_info['3a']['bltol']
            blinds, blrefinds, dbl = LKP.find_1NN(simvis_objs[0].baselines, blvects, distance_ULIM=bltol, remove_oob=True)

            blcolrs = ['black', 'red', 'cyan']
            for lind in lstind:
                fig, axs = PLT.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw={'height_ratios': [2, 1]}, figsize=(3.6, 3), constrained_layout=False)
                for simind,simlbl in enumerate(simlabels):
                    if spw is not None:
                        for zind in spwind:
                            axs[simind].axvspan((freq_window_centers[zind]-0.5*freq_window_bw[zind])/1e6, (freq_window_centers[zind]+0.5*freq_window_bw[zind])/1e6, facecolor='0.8')
                    for blno, blrefind in enumerate(blrefinds):
                        if simind == 0:
                            axs[simind].plot(simvis_objs[simind].channels/1e6, NP.abs(simvis_objs[simind].skyvis_freq[blrefind,:,lind]), ls='-', color=blcolrs[blno], label='{0:.1f} m, {1:.1f}'.format(bll[blno], blo[blno])+r'$^\circ$')
                            if blno == blinds.size-1:
                                axs[simind].plot(simvis_objs[simind].channels/1e6, simvis_objs[0].vis_rms_freq[blrefind,:,lind], ls='--', color='black', label='Noise RMS')
                                axs[simind].text(0.05, 0.95, 'FG', transform=axs[simind].transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                                axs[simind].set_ylabel(r'$|V|$ [Jy]', fontsize=12, weight='medium')
                                legend = axs[simind].legend(loc='upper right', shadow=False, fontsize=7)
                        else:
                            axs[simind].plot(simvis_objs[simind].channels/1e6, NP.abs(simvis_objs[0].skyvis_freq[blrefind,:,lind] + simvis_objs[simind].skyvis_freq[blrefind,:,lind]) - NP.abs(simvis_objs[0].skyvis_freq[blrefind,:,lind]), ls='-', color=blcolrs[blno], alpha=0.5)
                            if blno == blinds.size-1:
                                axs[simind].set_ylim(-5e-3, 4e-3)
                                axs[simind].text(0.95, 0.05, 'H I', transform=axs[simind].transAxes, fontsize=8, weight='medium', ha='right', va='bottom', color='black')
                                axs[simind].set_ylabel(r'$\delta |V|$ [Jy]', fontsize=12, weight='medium')
                fig.subplots_adjust(hspace=0, wspace=0)
                fig.subplots_adjust(top=0.95)
                fig.subplots_adjust(bottom=0.15)
                fig.subplots_adjust(left=0.25)
                fig.subplots_adjust(right=0.98)

                big_ax = fig.add_subplot(111)
                big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium', labelpad=20)
                PLT.savefig(figdir+'model_visibility_spectrum_{0:.1f}m_lst_{1:.3f}hr.pdf'.format(bll[blno], simlst[lind]), bbox_inches=0)
                        
        if '3b' in plots:
            spw = plot_info['3b']['spw']
            if spw is not None:
                spwind = NP.asarray(spw).reshape(-1)
            for lind in lstind:
                fig, axs = PLT.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw={'height_ratios': [2, 1]}, figsize=(3.6, 3), constrained_layout=False)
                for simind,simlbl in enumerate(simlabels):
                    if spw is not None:
                        for zind in spwind:
                            axs[simind].axvspan((freq_window_centers[zind]-0.5*freq_window_bw[zind])/1e6, (freq_window_centers[zind]+0.5*freq_window_bw[zind])/1e6, facecolor='0.8')
                    if simind == 0:
                        axs[simind].plot(model_cpObjs[simind].f/1e6, model_cpObjs[simind].cpinfo['processed']['native']['cphase'][lind,0,0,:], ls='-', color='black')
                        axs[simind].set_ylim(-NP.pi, NP.pi)
                        axs[simind].set_ylabel(r'$\phi_\nabla^\mathrm{F}(f)$ [rad]', fontsize=12, weight='medium')
                    elif simind == 1:
                        axs[simind].plot(model_cpObjs[simind].f/1e6, model_cpObjs[simind].cpinfo['processed']['native']['cphase'][lind,0,0,:] - model_cpObjs[0].cpinfo['processed']['native']['cphase'][lind,0,0,:], ls='-', color='black')
                        axs[simind].set_ylim(-2e-4, 2e-4)
                        axs[simind].set_ylabel(r'$\delta\phi_\nabla^\mathrm{HI}(f)$ [rad]', fontsize=12, weight='medium')

                fig.subplots_adjust(hspace=0, wspace=0)
                fig.subplots_adjust(top=0.95, bottom=0.15, left=0.25, right=0.98)

                big_ax = fig.add_subplot(111)
                big_ax.set_facecolor('none') # matplotlib.__version__ >= 2.0.0
                big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                big_ax.set_xticks([])
                big_ax.set_yticks([])
                big_ax.set_xlabel(r'$f$ [MHz]', fontsize=12, weight='medium', labelpad=20)
                PLT.savefig(figdir+'model_cPhase_spectrum_EQ28_lst_{0:.3f}hr.pdf'.format(simlst[lind]), bbox_inches=0)                

        PDB.set_trace()
        if '3c' in plots:
            n_days = plot_info['3c']['n_days']
            n_batches = plot_info['3c']['n_batches']
            t_field = plot_info['3c']['t_field'] * U.min
            t_int = plot_info['3c']['t_int'] * U.s
            n_pairs_of_batches = n_batches * (n_batches - 1) / 2.0 # Number of pairs of batches going into the cross-product
            n_int_per_field = t_field * 60.0 / t_int # Number of coherent integrations on a field
            npol = plot_info['3c']['npol']
    
            sampling = plot_info['3c']['sampling']
            spw = plot_info['3c']['spw']
            if spw is None:
                spwind = NP.arange(simDPS_objs[0].subband_delay_power_spectra['sim']['z'].size)
            else:
                spwind = NP.asarray(spw)
            
            eff_A = plot_info['3c']['A_eff']
            if isinstance(eff_A, (int,float)):
                eff_A = eff_A + NP.zeros_like(freq_window_centers)
            elif isinstance(eff_A, list):
                eff_A = NP.asarray(eff_A) + NP.zeros_like(freq_window_centers)
            else:
                raise TypeError('Effective area must be a scalar or list')
            eff_A = eff_A * U.m**2
            
            blvects = NP.asarray(plot_info['3c']['bl'])
            bll = NP.sqrt(NP.sum(blvects**2, axis=1))
            blo = NP.degrees(NP.arctan2(blvects[:,1], blvects[:,0]))
            bltol = plot_info['3c']['bltol']
            blinds, blrefinds, dbl = LKP.find_1NN(simvis_objs[0].baselines, blvects, distance_ULIM=bltol, remove_oob=True)

            bl_same_bin = plot_info['3c']['bl_same_bin']
            blvctinds = []
            blvctrefinds = []
            blhists = []
            blwts_coherent = []
            blwts_incoherent = []
            for blgrpind in range(len(bl_same_bin)):
                blvctgrp = NP.asarray(bl_same_bin[blgrpind])
                indNN_list, blind_ngbrof, blind_ngbrin = LKP.find_NN(simvis_objs[0].baselines, blvctgrp, distance_ULIM=bltol, flatten=True)
                blvctinds += [blind_ngbrin]
                blvctrefinds += [blind_ngbrof]
                blhist, blind_type, bl_binnum, ri = OPS.binned_statistic(blind_ngbrin, values=None, statistic='count', bins=list(range(blind_ngbrin.max()+2)), range=None)
                blhists += [blhist]
                blwts_coherent += [NP.sum(blhist**2)]
                blwts_incoherent += [NP.sum(blhist)]

            if sysT is None:
                sysT = simvis_objs[0].Tsys
            elif isinstance(sysT, (int,float)):
                sysT = sysT + NP.zeros_like(simvis_objs[0].shape)
            else:
                raise TypeError('Input system temperature in invalid format')
            sysT = sysT * U.K
    
            freqinds = NP.asarray([NP.argmin(NP.abs(simvis_objs[0].channels - fwin)) for fwin in freq_window_centers])
            nearest_Tsys = sysT[NP.ix_(blrefinds,freqinds,lstind)]

            df = simvis_objs[0].freq_resolution * U.Hz
            sysT_per_unit_visibility = nearest_Tsys / NP.sqrt(df * t_int * n_days) # Noise RMS temperature (in K) per batch. Of this, 1/sqrt(2) each in real and imaginary parts
            sysT_per_unit_visibility_real = sysT_per_unit_visibility / NP.sqrt(2.0) # in K
            sysT_per_unit_visibility_imag = sysT_per_unit_visibility / NP.sqrt(2.0) # in K
            rms_noise_K_dspec_bin = sysT_per_unit_visibility * NP.sqrt(freq_window_bw.reshape(1,-1,1)*U.Hz / df) * df # in K.Hz, of which 1/sqrt(2) each in real and imaginary parts
            rms_noise_K_dspec_bin_real = rms_noise_K_dspec_bin / NP.sqrt(2.0) # in K.Hz
            rms_noise_K_dspec_bin_imag = rms_noise_K_dspec_bin / NP.sqrt(2.0) # in K.Hz

            # Product of two independent Gaussian random variables is a modified Bessel function of the second kind with RMS as below:
            
            rms_noise_K_crosssprod_bin_real = NP.sqrt(rms_noise_K_dspec_bin_real**2 * rms_noise_K_dspec_bin_real**2 + rms_noise_K_dspec_bin_imag**2 * rms_noise_K_dspec_bin_imag**2) / NP.sqrt(npol * n_pairs_of_batches * n_int_per_field) # in K^2 Hz^2, per baseline
            rms_noise_K_crosssprod_bin_imag = NP.sqrt(rms_noise_K_dspec_bin_real**2 * rms_noise_K_dspec_bin_imag**2 + rms_noise_K_dspec_bin_real**2 * rms_noise_K_dspec_bin_imag**2) / NP.sqrt(npol * n_pairs_of_batches * n_int_per_field) # in K^2 Hz^2, per baseline

            rest_freq_HI = CNST.rest_freq_HI * U.Hz
            center_redshifts = rest_freq_HI / (freq_window_centers * U.Hz) - 1.0
            redshifts_ulim = rest_freq_HI / ((freq_window_centers - 0.5 * freq_window_bw) * U.Hz) - 1
            redshifts_llim = rest_freq_HI / ((freq_window_centers + 0.5 * freq_window_bw) * U.Hz) - 1

            center_redshifts = center_redshifts.to_value()
            redshifts_ulim = redshifts_ulim.to_value()
            redshifts_llim = redshifts_llim.to_value()

            wl = FCNST.c / (freq_window_centers * U.Hz)
            rz = cosmo100.comoving_distance(center_redshifts)
            drz = cosmo100.comoving_distance(redshifts_ulim) - cosmo100.comoving_distance(redshifts_llim)

            conv_factor1 = (wl**2 / eff_A)
            conv_factor2 = rz**2 * drz / (freq_window_bw * U.Hz)**2
            conv_factor = conv_factor1 * conv_factor2

            noise_xpspec_rms_real = rms_noise_K_crosssprod_bin_real * conv_factor.reshape(1,-1,1)

            noise_xpspec_rms_real_blgroups = []
            for blgrpind in range(len(bl_same_bin)):
                noise_xpspec_rms_real_blgroups += [{'coh_bl': noise_xpspec_rms_real[blgrpind].to('K2 Mpc3') / NP.sqrt(blwts_coherent[blgrpind]), 'incoh_bl': noise_xpspec_rms_real[blgrpind].to('K2 Mpc3') / NP.sqrt(blwts_incoherent[blgrpind])}]

            simDS_objs = [DS.DelaySpectrum(interferometer_array=simvis_obj) for simvis_obj in simvis_objs]
    
            simDPS_objs = []
            for simind,simlbl in enumerate(simlabels):
                dspec = simDS_objs[simind].delay_transform(action='store')
                subband_dspec = simDS_objs[simind].subband_delay_transform({key: freq_window_bw for key in ['cc', 'sim']}, freq_center={key: freq_window_centers for key in ['cc', 'sim']}, shape={key: freq_window_shape for key in ['cc', 'sim']}, fftpow={key: freq_window_fftpow for key in ['cc', 'sim']}, pad={key: pad for key in ['cc', 'sim']}, bpcorrect=False, action='return_resampled')
            simDPS_objs = []
            for simind,simlbl in enumerate(simlabels):
                simDPS_objs += [DS.DelayPowerSpectrum(simDS_objs[simind])]
                simDPS_objs[simind].compute_power_spectrum()

            colrs_sim = ['black', 'black']
            colrs_ref = ['gray', 'gray']
            # colrs_sim = ['red', 'blue']
            # colrs_ref = ['orange', 'cyan']
            lstyles = [':', '-']
            for blno, blrefind in enumerate(blrefinds):
                for lstno,lind in enumerate(lstind):
                    for zind in spwind:
                        pstable = ascii.read(theory_HI_PS_files[psfile_inds[zind]])
                        k = pstable['col1'] # in 1/Mpc
                        delta2 = 1e-6 * pstable['col2'] # in K^2
                        pk = 2 * NP.pi**2 / k**3 * delta2 # in K^2 Mpc^3
                        k_h = k / h_Planck15 # in h/Mpc
                        pk_h = pk * h_Planck15**3 # in K^2 (Mpc/h)^3
                        
                        kprll_sim = simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['kprll'][zind,:]
                        kperp_sim = simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['kperp'][zind,blrefind]
                        k_sim = NP.sqrt(kperp_sim**2 + kprll_sim**2)
                        
                        log10_ps_interped = OPS.interpolate_array(NP.log10(pk_h), NP.log10(k_h), NP.log10(k_sim), axis=-1, kind='linear')
                        ps_interped = 10**log10_ps_interped
                        
                        fig = PLT.figure(figsize=(4.0, 3.6))
                        ax = fig.add_subplot(111)

                        for simind,simlbl in enumerate(simlabels):
                            if simind == 0:
                                ax.plot(simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['kprll'][zind,:], 1e6*simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['skyvis_lag'][blrefind,zind,:,lind], ls=lstyles[simind], color=colrs_sim[zind], label=r'$P_\mathrm{F}$'+' ({0:.1f} MHz)'.format(freq_window_centers[zind]/1e6))
                            else:
                                ax.plot(simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['kprll'][zind,:], 1e6*simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['skyvis_lag'][blrefind,zind,:,lind], ls=lstyles[simind], color=colrs_sim[zind], label=r'$P_\mathrm{HI}$'+' (sim), '+r'$z=$'+'{0:.1f}'.format(simDPS_objs[simind].subband_delay_power_spectra['sim']['z'][zind]))
                                ax.plot(simDPS_objs[simind].subband_delay_power_spectra_resampled['sim']['kprll'][zind,:], 1e6*ps_interped, ls='-', color=colrs_ref[zind], label=r'$P_\mathrm{HI}$'+' (ref), '+r'$z=$'+'{0:.1f}'.format(simDPS_objs[simind].subband_delay_power_spectra['sim']['z'][zind]))
                        ax.axhline(y=noise_xpspec_rms_real_blgroups[blno]['coh_bl'][zind,lstno].to('mK2 Mpc3').value, ls='--', color='gray', label=r'$P_\mathrm{N}$'+' (red.)')
                        ax.axhline(y=noise_xpspec_rms_real_blgroups[blno]['incoh_bl'][zind,lstno].to('mK2 Mpc3').value, ls='--', color='black', label=r'$P_\mathrm{N}$'+' (non-red.)')
                        ax.set_yscale('log')
                        ax.legend(loc='upper right', shadow=False, fontsize=7.5)
                        ax.text(0.1, 0.9, '{0:.1f} m'.format(bll[blno]), transform=ax.transAxes, fontsize=8, weight='medium', ha='left', va='top', color='black')
                        ax.set_xlabel(r'$k_\parallel$ [$h$ Mpc$^{-1}$]')
                        ax.set_ylabel(r'$P_b(k_\parallel)$ [mK$^2$ $h^{-3}$ Mpc$^3$]')
                        axt = ax.twiny()
                        axt.set_xlim(1e6*simDS_objs[simind].subband_delay_spectra_resampled['sim']['lags'].min(), 1e6*simDS_objs[simind].subband_delay_spectra_resampled['sim']['lags'].max())
                        axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]')
                        fig.subplots_adjust(bottom=0.15, left=0.18, right=0.98)

                        # PLT.savefig(figdir+'delay_PS_{0:.1f}m_z_{1:.1f}_lst_{2:.3f}hr.pdf'.format(bll[blno], simDPS_objs[simind].subband_delay_power_spectra['sim']['z'][zind], simlst[lind]), bbox_inches=0)
                    
        PDB.set_trace()
    

