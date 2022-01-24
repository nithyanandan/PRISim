import copy
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.ticker as PLTick
import yaml, argparse, warnings
import progressbar as PGB
from prisim import bispectrum_phase as BSP
import ipdb as PDB

PLT.switch_backend("TkAgg")

if __name__ == '__main__':
    
    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to analyze closure phases from multiple days from multiple sources such as polarizations')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default='/data3/t_nithyanandan/codes/mine/python/projects/closure/combine_pol_multiday_EQ28_data_RA_1.6_closure_PS_analysis_parms.yaml', type=str, required=False, help='File specifying input parameters')

    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    datadirs = parms['dirStruct']['datadirs']
    infiles_a = parms['dirStruct']['infiles_a']
    infiles_a_errinfo = parms['dirStruct']['err_infiles_a']
    infiles_b = parms['dirStruct']['infiles_b']
    infiles_b_errinfo = parms['dirStruct']['err_infiles_b']
    model_labels = parms['dirStruct']['modelinfo']['model_labels']
    mdldirs = parms['dirStruct']['modelinfo']['mdldirs']
    mdl_infiles_a = parms['dirStruct']['modelinfo']['infiles_a']
    mdl_infiles_a_errinfo = parms['dirStruct']['modelinfo']['err_infiles_a']
    mdl_infiles_b = parms['dirStruct']['modelinfo']['infiles_b']
    mdl_infiles_b_errinfo = parms['dirStruct']['modelinfo']['err_infiles_b']
    outdir = parms['dirStruct']['outdir']
    figdir = outdir + parms['dirStruct']['figdir']
    plotfile_pfx = parms['dirStruct']['plotfile_pfx']

    xcpdps_a = []
    excpdps_a = []
    xcpdps_b = []
    excpdps_b = []
    for fileind,indir in enumerate(datadirs):
        infile_a = indir + infiles_a[fileind]
        infile_a_errinfo = indir + infiles_a_errinfo[fileind]
        infile_b = indir + infiles_b[fileind]
        infile_b_errinfo = indir + infiles_b_errinfo[fileind]

        xcpdps_a += [BSP.read_CPhase_cross_power_spectrum(infile_a)]
        excpdps_a += [BSP.read_CPhase_cross_power_spectrum(infile_a_errinfo)]
        xcpdps_b += [BSP.read_CPhase_cross_power_spectrum(infile_b)]
        excpdps_b += [BSP.read_CPhase_cross_power_spectrum(infile_b_errinfo)]

    xcpdps_a_avg_pol, excpdps_a_avg_pol = BSP.incoherent_cross_power_spectrum_average(xcpdps_a, excpdps=excpdps_a, diagoffsets=None)
    xcpdps_b_avg_pol, excpdps_b_avg_pol = BSP.incoherent_cross_power_spectrum_average(xcpdps_b, excpdps=excpdps_b, diagoffsets=None)

    models_xcpdps_a_avg_pol = []
    models_excpdps_a_avg_pol = []
    models_xcpdps_b_avg_pol = []
    models_excpdps_b_avg_pol = []
    for mdlind, model in enumerate(model_labels):
        mdl_xcpdps_a = []
        mdl_excpdps_a = []
        mdl_xcpdps_b = []
        mdl_excpdps_b = []
        for fileind,mdldir in enumerate(mdldirs[mdlind]):
            mdl_infile_a = mdldir + mdl_infiles_a[mdlind][fileind]
            mdl_infile_a_errinfo = mdldir + mdl_infiles_a_errinfo[mdlind][fileind]
            mdl_infile_b = mdldir + mdl_infiles_b[mdlind][fileind]
            mdl_infile_b_errinfo = mdldir + mdl_infiles_b_errinfo[mdlind][fileind]

            mdl_xcpdps_a += [BSP.read_CPhase_cross_power_spectrum(mdl_infile_a)]
            mdl_excpdps_a += [BSP.read_CPhase_cross_power_spectrum(mdl_infile_a_errinfo)]
            mdl_xcpdps_b += [BSP.read_CPhase_cross_power_spectrum(mdl_infile_b)]
            mdl_excpdps_b += [BSP.read_CPhase_cross_power_spectrum(mdl_infile_b_errinfo)]
        mdl_xcpdps_a_avg_pol, mdl_excpdps_a_avg_pol = BSP.incoherent_cross_power_spectrum_average(mdl_xcpdps_a, excpdps=mdl_excpdps_a, diagoffsets=None)
        models_xcpdps_a_avg_pol += [mdl_xcpdps_a_avg_pol]
        models_excpdps_a_avg_pol += [mdl_excpdps_a_avg_pol]
        mdl_xcpdps_b_avg_pol, mdl_excpdps_b_avg_pol = BSP.incoherent_cross_power_spectrum_average(mdl_xcpdps_b, excpdps=mdl_excpdps_b, diagoffsets=None)
        models_xcpdps_b_avg_pol += [mdl_xcpdps_b_avg_pol]
        models_excpdps_b_avg_pol += [mdl_excpdps_b_avg_pol]

    plot_info = parms['plot']
    plots = [key for key in plot_info if plot_info[key]['action']]

    PLT.ion()
    if ('2' in plots) or ('2a' in plots) or ('2b' in plots) or ('2c' in plots) or ('2d' in plots):

        sampling = plot_info['2']['sampling']
        statistic = plot_info['2']['statistic']
        datapool = plot_info['2']['datapool']
        pspec_unit_type = plot_info['2']['units']

        if pspec_unit_type == 'K':
            pspec_unit = 'mK2 Mpc3'
        else:
            pspec_unit = 'Jy2 Mpc'

        spw = plot_info['2']['spw']
        if spw is None:
            spwind = NP.arange(xcpdps2_a[sampling]['z'].size)
        else:
            spwind = NP.asarray(spw)
            
        if statistic is None:
            statistic = ['mean', 'median']
        else:
            statistic = [statistic]
            
        ps_errtype = plot_info['2']['errtype']
        errshade = {}
        for errtype in ps_errtype:
            if errtype.lower() == 'ssdiff':
                errshade[errtype] = '0.8'
            elif errtype.lower() == 'psdiff':
                errshade[errtype] = '0.6'

        nsigma = plot_info['2']['nsigma']
        
        mdl_colrs = ['red', 'green', 'blue', 'cyan', 'gray', 'orange']

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
            xcpdps_a_avg_pol_diag, excpdps_a_avg_pol_diag = BSP.incoherent_cross_power_spectrum_average(xcpdps_a_avg_pol, excpdps=excpdps_a_avg_pol, diagoffsets=diagoffsets_a)

            models_xcpdps_a_avg_pol_diag = []
            models_excpdps_a_avg_pol_diag = []

            for combi,incax_comb in enumerate(avg_incohax_b):
                diagoffsets_b += [{}]
                for incaxind,incax in enumerate(incax_comb):
                    diagoffsets_b[-1][incax] = NP.asarray(diagoffsets_incohax_b[combi][incaxind])
            
            xcpdps_b_avg_pol_diag, excpdps_b_avg_pol_diag = BSP.incoherent_cross_power_spectrum_average(xcpdps_b_avg_pol, excpdps=excpdps_b_avg_pol, diagoffsets=diagoffsets_b)
            models_xcpdps_b_avg_pol_diag = []
            models_excpdps_b_avg_pol_diag = []

            if len(model_labels) > 0:
                progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Models '.format(len(model_labels)), PGB.ETA()], maxval=len(model_labels)).start()
    
                for i in range(len(model_labels)):
                    model_xcpdps_a_avg_pol_diag, model_excpdps_a_avg_pol_diag = BSP.incoherent_cross_power_spectrum_average(models_xcpdps_a_avg_pol[i], excpdps=models_excpdps_a_avg_pol[i], diagoffsets=diagoffsets_a)
                    models_xcpdps_a_avg_pol_diag += [copy.deepcopy(model_xcpdps_a_avg_pol_diag)]
                    models_excpdps_a_avg_pol_diag += [copy.deepcopy(model_excpdps_a_avg_pol_diag)]

                    model_xcpdps_b_avg_pol_diag, model_excpdps_b_avg_pol_diag = BSP.incoherent_cross_power_spectrum_average(models_xcpdps_b_avg_pol[i], excpdps=models_excpdps_b_avg_pol[i], diagoffsets=diagoffsets_b)
                    models_xcpdps_b_avg_pol_diag += [copy.deepcopy(model_xcpdps_b_avg_pol_diag)]
                    models_excpdps_b_avg_pol_diag += [copy.deepcopy(model_excpdps_b_avg_pol_diag)]

                    progress.update(i+1)
                progress.finish()

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
                                            # if len(model_labels) > 0:
                                            #     for mdlind, mdl in enumerate(model_labels):
                                            #         if dpool in models_xcpdps_b_avg_pol_diag[mdlind][sampling]:
                                            #             psval = (1/3.0) * models_xcpdps_b_avg_pol_diag[mdlind][sampling][dpool][stat][combi][zind,lind,dayind_models[di][mdlind],trind,:].to(pspec_unit).value
                                            #             maxabsvals += [NP.abs(psval.real).max()]
                                            #             minabsvals += [NP.abs(psval.real).min()]
                                            #             maxvals += [psval.real.max()]
                                            #             minvals += [psval.real.min()]
                                            #             axs[dpoolind].plot(models_xcpdps_b_avg_pol_diag[mdlind][sampling]['kprll'][zind,:], psval.real, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                                    
                                            if dpool in xcpdps_b_avg_pol_diag[sampling]:
                                                psval = (2/3.0) * xcpdps_b_avg_pol_diag[sampling][dpool][stat][combi][zind,lind,dind,trind,:].to(pspec_unit).value
                                                psrms_ssdiff = (2/3.0) * NP.nanstd(excpdps_a_avg_pol_diag[sampling]['errinfo'][stat][combi][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                                if 2 in avg_incohax_b[combi]:
                                                    ind_dayax_in_incohax = avg_incohax_b[combi].index(2)
                                                    if 0 in diagoffsets_incohax_b[combi][ind_dayax_in_incohax]:
                                                        rms_inflation_factor = 2.0 * NP.sqrt(2.0)
                                                    else:
                                                        rms_inflation_factor = NP.sqrt(2.0)
                                                else:
                                                    rms_inflation_factor = NP.sqrt(2.0)
                                                psrms_psdiff = (2/3.0) * (xcpdps_a_avg_pol_diag[sampling][dpool][stat][combi][zind,lind,1,1,trind,:] - xcpdps_a_avg_pol_diag[sampling][dpool][stat][combi][zind,lind,0,0,trind,:]).to(pspec_unit).value
                                                psrms_psdiff = NP.abs(psrms_psdiff.real) / rms_inflation_factor

                                                psrms_max = NP.amax(NP.vstack((psrms_ssdiff, psrms_psdiff)), axis=0)
                                                
                                                maxabsvals += [NP.abs(psval.real + nsigma*psrms_max).max()]
                                                minabsvals += [NP.abs(psval.real).min()]
                                                maxvals += [(psval.real + nsigma*psrms_max).max()]
                                                minvals += [(psval.real - nsigma*psrms_max).min()]

                                                for errtype in ps_errtype:
                                                    if errtype.lower() == 'ssdiff':
                                                        axs[dpoolind].errorbar(xcpdps_b_avg_pol_diag[sampling]['kprll'][zind,:], psval.real, yerr=nsigma*psrms_ssdiff, xerr=None, ecolor=errshade[errtype], ls='none', marker='.', ms=4, color='black')
                                                    elif errtype.lower() == 'psdiff':
                                                        axs[dpoolind].errorbar(xcpdps_b_avg_pol_diag[sampling]['kprll'][zind,:], psval.real, yerr=nsigma*psrms_psdiff, xerr=None, ecolor=errshade[errtype], ls='none', marker='.', ms=4, color='black', label='FG+N')
                                                    
                                            # legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
                                            if trno == 0:
                                                axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps_b_avg_pol_diag[sampling]['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
                                    
                                            axt = axs[dpoolind].twiny()
                                            axt.set_xlim(1e6*xcpdps_b_avg_pol_diag[sampling]['lags'].min(), 1e6*xcpdps_b_avg_pol_diag[sampling]['lags'].max())
                                    
                                        axs[dpoolind].axhline(y=0, xmin=0, xmax=1, ls='-', lw=1, color='black')

                                        minvals = NP.asarray(minvals)
                                        maxvals = NP.asarray(maxvals)
                                        minabsvals = NP.asarray(minabsvals)
                                        maxabsvals = NP.asarray(maxabsvals)
                                        axs[dpoolind].set_xlim(0.99*xcpdps_b_avg_pol_diag[sampling]['kprll'][zind,:].min(), 1.01*xcpdps_b_avg_pol_diag[sampling]['kprll'][zind,:].max())
                                        if NP.min(minvals) < 0.0:
                                            axs[dpoolind].set_ylim(1.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                        else:
                                            axs[dpoolind].set_ylim(0.5*NP.min(minvals), 2*NP.max(maxabsvals))
                                        axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
                                        tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
                                        axs[dpoolind].yaxis.set_major_locator(tickloc)
                                        axs[dpoolind].grid(color='0.9', which='both', linestyle=':', lw=1)
                                        
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
                                        big_ax.set_ylabel(r'$\frac{2}{3}\, P_\nabla(\kappa_\parallel)$ [pseudo mK$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
                                    else:
                                        big_ax.set_ylabel(r'$\frac{2}{3}\, P_\nabla(\kappa_\parallel)$ [pseudo Jy$^2h^{-1}$ Mpc]', fontsize=12, weight='medium', labelpad=40)
                                    
                                    big_axt = big_ax.twiny()
                                    big_axt.set_xticks([])
                                    big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
                                    
                                    PLT.savefig(figdir + '{0}_symlog_incoh_avg_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_comb_{5:0d}.pdf'.format(plotfile_pfx, xcpdps_b_avg_pol_diag[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps_b_avg_pol_diag['dlst'][0], combi), bbox_inches=0)

            PDB.set_trace()
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
                xcpdps_a_avg_pol_diag_kbin = BSP.incoherent_kbin_averaging(xcpdps_a_avg_pol_diag, kbins=kbins, kbintype=kbintype)
                excpdps_a_avg_pol_diag_kbin = BSP.incoherent_kbin_averaging(excpdps_a_avg_pol_diag, kbins=kbins, kbintype=kbintype)
                models_xcpdps_a_avg_pol_diag_kbin = []
                models_excpdps_a_avg_pol_diag_kbin = []

                xcpdps_b_avg_pol_diag_kbin = BSP.incoherent_kbin_averaging(xcpdps_b_avg_pol_diag, kbins=kbins, kbintype=kbintype)
                excpdps_b_avg_pol_diag_kbin = BSP.incoherent_kbin_averaging(excpdps_b_avg_pol_diag, kbins=kbins, kbintype=kbintype)
                models_xcpdps_b_avg_pol_diag_kbin = []
                models_excpdps_b_avg_pol_diag_kbin = []

                if len(model_labels) > 0:
                    for i in range(len(model_labels)):
                        models_xcpdps_a_avg_pol_diag_kbin += [BSP.incoherent_kbin_averaging(models_xcpdps_a_avg_pol_diag[i], kbins=kbins, kbintype=kbintype)]
                        models_excpdps_a_avg_pol_diag_kbin += [BSP.incoherent_kbin_averaging(models_excpdps_a_avg_pol_diag[i], kbins=kbins, kbintype=kbintype)]
                        models_xcpdps_b_avg_pol_diag_kbin += [BSP.incoherent_kbin_averaging(models_xcpdps_b_avg_pol_diag[i], kbins=kbins, kbintype=kbintype)]
                        models_excpdps_b_avg_pol_diag_kbin += [BSP.incoherent_kbin_averaging(models_excpdps_b_avg_pol_diag[i], kbins=kbins, kbintype=kbintype)]
                    
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
                                        if pstype == 'Del2':
                                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 6.0))
                                        else:
                                            fig, axs = PLT.subplots(nrows=1, ncols=len(datapool), sharex=True, sharey=True, figsize=(4.0*len(datapool), 3.6))
                                        if len(datapool) == 1:
                                            axs = [axs]
                                        for dpoolind,dpool in enumerate(datapool):
                                            for trno,trind in enumerate(triadind):
                                                if pstype == 'Del2':
                                                    if len(model_labels) > 0:
                                                        for mdlind, mdl in enumerate(model_labels):
                                                            if dpool in models_xcpdps_b_avg_pol_diag_kbin[mdlind][sampling]:
                                                                if pstype == 'PS':
                                                                    psval = (2/3.0) * models_xcpdps_b_avg_pol_diag_kbin[mdlind][sampling][dpool][stat][pstype][combi][zind,lind,dayind_models[di][mdlind],trind,:].to(pspec_unit).value
                                                                else:
                                                                    psval = (2/3.0) * models_xcpdps_b_avg_pol_diag_kbin[mdlind][sampling][dpool][stat][pstype][combi][zind,lind,dayind_models[di][mdlind],trind,:].to('mK2').value
                                                                kval = models_xcpdps_b_avg_pol_diag_kbin[mdlind][sampling]['kbininfo'][dpool][stat][combi][zind,lind,dayind_models[di][mdlind],trind,:].to('Mpc-1').value
                                                                maxabsvals += [NP.nanmin(NP.abs(psval.real))]
                                                                minabsvals += [NP.nanmin(NP.abs(psval.real))]
                                                                maxvals += [NP.nanmax(psval.real)]
                                                                minvals += [NP.nanmin(psval.real)]
                                                                axs[dpoolind].plot(kval, psval.real, ls='none', marker='.', ms=3, color=mdl_colrs[mdlind], label='{0}'.format(mdl))
                
                                                if dpool in xcpdps_b_avg_pol_diag_kbin[sampling]:
                                                    if pstype == 'PS':
                                                        psval = (2/3.0) * xcpdps_b_avg_pol_diag_kbin[sampling][dpool][stat][pstype][combi][zind,lind,dind,trind,:].to(pspec_unit).value
                                                        psrms_ssdiff = (2/3.0) * NP.nanstd(excpdps_b_avg_pol_diag_kbin[sampling]['errinfo'][stat][pstype][combi][zind,lind,:,trind,:], axis=0).to(pspec_unit).value
                                                        psrms_psdiff = (2/3.0) * (xcpdps_a_avg_pol_diag_kbin[sampling][dpool][stat][pstype][combi][zind,lind,1,1,trind,:] - xcpdps_a_avg_pol_diag_kbin[sampling][dpool][stat][pstype][combi][zind,lind,0,0,trind,:]).to(pspec_unit).value
                                                    else:
                                                        psval = (2/3.0) * xcpdps_b_avg_pol_diag_kbin[sampling][dpool][stat][pstype][combi][zind,lind,dind,trind,:].to('mK2').value
                                                        psrms_ssdiff = (2/3.0) * NP.nanstd(excpdps_b_avg_pol_diag_kbin[sampling]['errinfo'][stat][pstype][combi][zind,lind,:,trind,:], axis=0).to('mK2').value
                                                        psrms_psdiff = (2/3.0) * (xcpdps_a_avg_pol_diag_kbin[sampling][dpool][stat][pstype][combi][zind,lind,1,1,trind,:] - xcpdps_a_avg_pol_diag_kbin[sampling][dpool][stat][pstype][combi][zind,lind,0,0,trind,:]).to('mK2').value

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

                                                    kval = xcpdps_b_avg_pol_diag_kbin[sampling]['kbininfo'][dpool][stat][combi][zind,lind,dind,trind,:].to('Mpc-1').value
                                                    
                                                    maxabsvals += [NP.nanmax(NP.abs(psval.real + nsigma*psrms_max.real))]
                                                    minabsvals += [NP.nanmin(NP.abs(psval.real))]
                                                    maxvals += [NP.nanmax(psval.real + nsigma*psrms_max.real)]
                                                    minvals += [NP.nanmin(psval.real - nsigma*psrms_max.real)]
                                                    for errtype in ps_errtype:
                                                        if errtype.lower() == 'ssdiff':
                                                            axs[dpoolind].errorbar(kval, psval.real, yerr=nsigma*psrms_ssdiff, xerr=None, ecolor=errshade[errtype.lower()], ls='none', marker='.', ms=4, color='black')
                                                        elif errtype.lower() == 'psdiff':
                                                            axs[dpoolind].errorbar(kval, psval.real, yerr=nsigma*psrms_psdiff, xerr=None, ecolor=errshade[errtype.lower()], ls='none', marker='.', ms=4, color='black', label='Data')

                                                if pstype == 'Del2':
                                                    legend = axs[dpoolind].legend(loc='center', bbox_to_anchor=(0.5,0.3), shadow=False, fontsize=8)
                                                if trno == 0:
                                                    axs[dpoolind].text(0.95, 0.97, r'$z=$'+' {0:.1f}'.format(xcpdps_b_avg_pol_diag_kbin['resampled']['z'][zind]), transform=axs[dpoolind].transAxes, fontsize=8, weight='medium', ha='right', va='top', color='black')
                                            
                                            axs[dpoolind].axhline(y=0, xmin=0, xmax=1, ls='-', lw=1, color='black')

                                            minvals = NP.asarray(minvals)
                                            maxvals = NP.asarray(maxvals)
                                            minabsvals = NP.asarray(minabsvals)
                                            maxabsvals = NP.asarray(maxabsvals)
                                            axs[dpoolind].set_xlim(0.99*NP.nanmin(xcpdps_b_avg_pol_diag_kbin['resampled']['kbininfo']['kbin_edges'][zind].to('Mpc-1').value), 1.01*NP.nanmax(xcpdps_b_avg_pol_diag_kbin['resampled']['kbininfo']['kbin_edges'][zind].to('Mpc-1').value))
                                            if NP.min(minvals) < 0.0:
                                                axs[dpoolind].set_ylim(1.5*NP.nanmin(minvals), 2*NP.nanmax(maxabsvals))
                                            else:
                                                axs[dpoolind].set_ylim(0.5*NP.nanmin(minvals), 2*NP.nanmax(maxabsvals))
                                            axs[dpoolind].set_yscale('symlog', linthreshy=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))))
                                            tickloc = PLTick.SymmetricalLogLocator(linthresh=10**NP.floor(NP.log10(NP.min(minabsvals[minabsvals > 0.0]))), base=100.0)
                                            axs[dpoolind].yaxis.set_major_locator(tickloc)
                                            axs[dpoolind].grid(color='0.8', which='both', linestyle=':', lw=1)
    
                                        fig.subplots_adjust(top=0.95)
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
                                            big_ax.set_ylabel(r'$\frac{2}{3}\, P_\nabla(\kappa_\parallel)$ [pseudo mK$^2h^{-3}$ Mpc$^3$]', fontsize=12, weight='medium', labelpad=40)
                                        else:
                                            big_ax.set_ylabel(r'$\frac{2}{3}\, \Delta_\nabla^2(\kappa_\parallel)$ [pseudo mK$^2$]', fontsize=12, weight='medium', labelpad=40)
                                        
                                        # big_axt = big_ax.twiny()
                                        # big_axt.set_xticks([])
                                        # big_axt.set_xlabel(r'$\tau$'+' ['+r'$\mu$'+'s]', fontsize=12, weight='medium', labelpad=20)
                                        if pstype == 'PS':
                                            PLT.savefig(figdir + '{0}_symlog_incoh_kbin_avg_real_cpdps_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_comb_{5:0d}.pdf'.format(plotfile_pfx, xcpdps_a_avg_pol_diag_kbin[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps_b_avg_pol_diag_kbin['dlst'][0], combi), bbox_inches=0)
                                        else:
                                            PLT.savefig(figdir + '{0}_symlog_incoh_kbin_avg_real_cpDel2_z_{1:.1f}_{2}_{3}_dlst_{4:.1f}s_comb_{5:0d}.pdf'.format(plotfile_pfx, xcpdps_a_avg_pol_diag_kbin[sampling]['z'][zind], stat, sampling, 3.6e3*xcpdps_b_avg_pol_diag_kbin['dlst'][0], combi), bbox_inches=0)

    PDB.set_trace()
    
