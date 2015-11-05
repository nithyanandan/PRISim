import multiprocessing as MP
import numpy as NP
import healpy as HP
import scipy.constants as FCNST
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
import my_operations as OPS
import ipdb as PDB
import objgraph as OG

def accumulate_fringes(fringes):
    if len(fringes.shape) == 1:
        fringes = fringes.reshape(1,-1)
    accumulated_fringes = NP.cumsum(fringes, axis=1)
    return accumulated_fringes

def accumulated_fringe_differencing(acc_fringes, sinza, fringelen, sinza_begin=0.0, sinza_end=1.0):
    PDB.set_trace()
    if not isinstance(sinza_begin, float):
        raise ValueError('sinza_begin must be a scalar')
    if not isinstance(sinza_end, float):
        raise ValueError('sinza_end must be a scalar')
    if (sinza_begin < 0.0) or (sinza_end > 1.0) or (sinza_begin >= sinza_end):
        raise ValueError('Invalid value(s) specified for sinza_begin and/or sinza_end')
    if len(acc_fringes.shape) == 1:
        acc_fringes = acc_fringes.reshape(1,-1)
    sortind = NP.argsort(sinza)
    if NP.sum(NP.abs(sortind - NP.arange(sinza.size))) > 0:
        raise IndexError('sin(ZA) must be sorted')
    fringelen = NP.asarray(fringelen)
    if len(fringelen.shape) != acc_fringes.shape[0]:
        if len(fringelen.shape) != 1:
            raise ValueError('Shapes of accumulated fringe matrix and fringe bin size are incompatible')
    if fringelen.size == 1:
        sinza_bin_edges = NP.arange(sinza_begin, sinza_end, fringelen)
        sinza_bin_edges = NP.append(sinza_bin_edges, sinza_end)
        bin_edges_ind = NP.searchsorted(sinza, sinza_bin_edges)
        fringe_cycle_coherence = [acc_fringes[:,bin_edge_ind[i+1]] - acc_fringes[:,bin_edges_ind[i]] for i in range(bin_edges_ind.size - 1)]
    else:
        fringe_cycle_coherence = []
        for chan in range(fringelen.size):
            sinza_bin_edges = NP.arange(sinza_begin, sinza_end, fringelen[chan])
            sinza_bin_edges = NP.append(sinza_bin_edges, sinza_end)
            bin_edges_ind = NP.searchsorted(sinza, sinza_bin_edges)
            fringe_cycle_coherence += [acc_fringes[chan,bin_edge_ind[i+1]] - acc_fringes[chan,bin_edges_ind[i]] for i in range(bin_edges_ind.size - 1)]

def fringe_coherence(u, sinza):
    fringes = NP.exp(-1j * 2*NP.pi * u * sinza)
    fringe_cycle_coherence = NP.zeros(NP.ceil(u).astype(NP.int), dtype=NP.complex64)
    dsinza = 1/u
    eps = 1e-10
    sinza_intervals = NP.arange(0, 1, dsinza[0])
    sinza_intervals = NP.append(sinza_intervals, 1.0) - eps
    bincount, n_bin_edges, binnum, ri = OPS.binned_statistic(sinza, statistic='count', bins=sinza_intervals)
    for i in range(bincount.size):
        ind = ri[ri[i]:ri[i+1]]
        fringe_cycle_coherence[i] = NP.sum(fringes[ui,ind])
    return fringe_cycle_coherence

def unwrap_fringe_coherence(arg):
    return fringe_coherence(*arg)

# @profile
def main():

    # 01) Fringe pattern behaviour
    
    # 02) Fringe cycle coherence
    
    # 03) Fringe cycle coherence from 2D sky

    # 04) Delay bin coherence from 2D sky
    
    plot_01 = False
    plot_02 = False
    plot_03 = False
    plot_04 = True
    
    def za2sinza(za):
        sinza = NP.sin(NP.radians(za))
        return ['{0:.2f}'.format(l) for l in sinza]
    
    def sinza2za(sinza):
        za = NP.degrees(NP.arcsin(sinza))
        return ['{0:.1f}'.format(theta) for theta in za]
    
    if plot_01:
    
        # 01) Fringe pattern behaviour
        
        bll = 10.0 # Baseline length in m
        wl = 2.0  # Wavelength in m
        # freq = 150e6  # Frequency in Hz
        # wl = FCNST.c / freq
        freq = FCNST.c / wl
        n_theta = 10001
        theta = NP.pi * (NP.arange(n_theta) - n_theta/2)/n_theta
        y_offset = 10.0 * NP.cos(theta)
        
        l = NP.sin(theta) # Direction cosine
        
        fringe = NP.exp(-1j * 2 * NP.pi * bll * l / wl)
        fringe_l_interval = wl / bll
        l_intervals = NP.arange(-1, 1, fringe_l_interval)
        theta_intervals = NP.degrees(NP.arcsin(l_intervals))
        theta_intervals = NP.append(theta_intervals, 90.0)
        
        bincount, theta_bin_edges, binnum, ri = OPS.binned_statistic(NP.degrees(theta), statistic='count', bins=theta_intervals)
        fringe_sum = NP.zeros(bincount.size, dtype=NP.complex64)
        for i in range(bincount.size):
            ind = ri[ri[i]:ri[i+1]]
            fringe_sum[i] = NP.sum(fringe[ind])
        
        fig = PLT.figure(figsize=(6,6))
        axth = fig.add_subplot(111)
        fringe_theta = axth.plot(NP.degrees(theta), fringe.real, 'k.', ms=2)
        
        for theta_interval in theta_intervals:
            axth.axvline(x=theta_interval, ymin=0, ymax=0.75, ls=':', lw=0.5, color='black')
        axth.set_xlim(-90, 90)
        axth.set_xlabel(r'$\theta$'+' [degree]')
        axth.set_ylabel('Fringe [Arbitrary units]', fontsize=12)
        
        axdc = axth.twiny()
        fringe_dircos = axdc.plot(l, y_offset.max()+fringe.real, 'r.', ms=2)
        fringe_dircos_on_sky = axdc.plot(l, y_offset+fringe.real, color='orange', marker='.', ms=2)
        sky_curve = axdc.plot(l, y_offset, color='orange', ls='--', lw=1)
        axdc.set_xlim(-1.0, 1.0)
        axdc.set_xlabel(r'$\sin\,\theta$', fontsize=12, color='red')
        
        axr = axth.twinx()
        # axr.bar(theta_intervals[:-1], bincount/theta.size, width=theta_intervals[1:]-theta_intervals[:-1], color='white', fill=False)
        axr.errorbar(0.5*(theta_intervals[1:]+theta_intervals[:-1]), NP.abs(fringe_sum)/bincount, yerr=1/NP.sqrt(bincount), color='b', ecolor='b', fmt='o-', ms=10, lw=2)
        axr.set_xlim(-90,90)
        axr.set_ylim(-0.2, 0.4)
        axr.set_ylabel('Coherence Amplitude in Fringe Cycle', fontsize=12, color='blue')
        
        fig.subplots_adjust(right=0.85)
        
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/wide_field_effects_demo.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/wide_field_effects_demo.eps', bbox_inches=0)
        
        PLT.show()
        
    if plot_02:
    
        # 02) Fringe cycle coherence
        
        u = 2.0 ** NP.arange(13)
        u_max = u.max()
        dza = NP.degrees(1/(512*u_max))
        za = NP.arange(0.0, 90.0, dza)
        sinza = NP.sin(NP.radians(za))
    
        fringes = NP.exp(-1j*2*NP.pi*u.reshape(-1,1)*sinza.reshape(1,-1))
        fringe_cycle_coherence = NP.zeros((u.size, u_max), dtype=NP.complex64)
        sample_counts = NP.zeros((u.size, u_max))
        independent_sample_counts = NP.zeros((u.size, u_max))
    
        dsinza = 1/u
        last_fringe_cycle_boundary_sinza = 1.0-dsinza
        last_fringe_cycle_boundary = NP.degrees(NP.arcsin(last_fringe_cycle_boundary_sinza))

        eps = 1e-10
        sinza_intervals = NP.arange(0, 1, dsinza[-1])
        sinza_intervals = NP.append(sinza_intervals, 1.0) - eps
        za_intervals = NP.degrees(NP.arcsin(sinza_intervals))
        # za_intervals = NP.append(za_intervals, 90.0)
        bincount, za_bin_edges, binnum, ri = OPS.binned_statistic(sinza, statistic='count', bins=sinza_intervals)
        for ui in range(u.size):
            for i in range(u[ui].astype(NP.int)):
                begini = i*u_max/u[ui]
                endi = (i+1)*u_max/u[ui]
                begini = begini.astype(NP.int)
                endi = endi.astype(NP.int)                
                fine_fringe_cycle_ind = NP.arange(begini, endi)
                ind = ri[ri[begini]:ri[endi]]
                fringe_cycle_coherence[ui,fine_fringe_cycle_ind] = NP.sum(fringes[ui,ind])
                sample_counts[ui,fine_fringe_cycle_ind] = float(ind.size)
                independent_sample_counts[ui,fine_fringe_cycle_ind] = 2.0 / NP.sqrt(1-NP.mean(sinza[ind])**2)                

                # ind = ri[ri[i]:ri[i+1]]
                # fringe_cycle_coherence[ui,ind] = NP.sum(fringes[ui,ind])
                # sample_counts[ui,ind] = bincount[i]
                # independent_sample_counts[ui,ind] = 2.0 / NP.sqrt(1-NP.mean(sinza[ind])**2)
        norm_fringe_cycle_coherence = fringe_cycle_coherence / sample_counts
        fringe_cycle_efficiency = fringe_cycle_coherence / za.size
        norm_fringe_cycle_coherence_SNR = norm_fringe_cycle_coherence * NP.sqrt(independent_sample_counts)
    
        fig = PLT.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        fringe_matrix = ax.pcolormesh(sinza_intervals[:-1], u, NP.abs(norm_fringe_cycle_coherence), norm=PLTC.LogNorm(vmin=1e-6, vmax=NP.abs(norm_fringe_cycle_coherence).max()))
        # fringe_matrix = ax.imshow(NP.abs(norm_fringe_cycle_coherence), origin='lower', extent=[za.min(), 90.0, u.min(), u.max()], norm=PLTC.LogNorm(vmin=1e-6, vmax=NP.abs(norm_fringe_cycle_coherence).max()))
        last_fringe_cycle_edge = ax.plot(last_fringe_cycle_boundary_sinza, u, ls='--', lw=2, color='white')
        ax.set_xlim(sinza.min(), 1.0)
        ax.set_ylim(u.min(), u.max())
        ax.set_yscale('log')
        ax.set_xlabel(r'$|\,\sin\,\theta\,|$ [degrees]', fontsize=14)
        ax.set_ylabel(r'$b/\lambda$', fontsize=14)    
        ax.set_aspect('auto')
    
        axt = ax.twiny()
        axt.set_xticks(ax.get_xticks())
        axt.set_xbound(ax.get_xbound())
        axt.set_xticklabels(sinza2za(ax.get_xticks()))
        axt.set_xlabel(r'$|\,\theta\,|$', fontsize=14)
    
        fig.subplots_adjust(right=0.88, left=0.15, top=0.9)
    
        cbax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(fringe_matrix, cax=cbax, orientation='vertical')
    
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_coherence.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_coherence.eps', bbox_inches=0)    

        fig = PLT.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        fringe_matrix = ax.pcolormesh(sinza_intervals[:-1], u, NP.abs(fringe_cycle_efficiency), norm=PLTC.LogNorm(vmin=1e-8, vmax=NP.abs(fringe_cycle_efficiency).max()))
        # fringe_matrix = ax.imshow(NP.abs(norm_fringe_cycle_efficiency), origin='lower', extent=[za.min(), 90.0, u.min(), u.max()], norm=PLTC.LogNorm(vmin=1e-6, vmax=NP.abs(norm_fringe_cycle_efficiency).max()))
        last_fringe_cycle_edge = ax.plot(last_fringe_cycle_boundary_sinza, u, ls='--', lw=2, color='white')
        ax.set_xlim(sinza.min(), 1.0)
        ax.set_ylim(u.min(), u.max())
        ax.set_yscale('log')
        ax.set_xlabel(r'$|\,\sin\,\theta\,|$ [degrees]', fontsize=14)
        ax.set_ylabel(r'$b/\lambda$', fontsize=14)    
        ax.set_aspect('auto')
    
        axt = ax.twiny()
        axt.set_xticks(ax.get_xticks())
        axt.set_xbound(ax.get_xbound())
        axt.set_xticklabels(sinza2za(ax.get_xticks()))
        axt.set_xlabel(r'$|\,\theta\,|$', fontsize=14)
    
        fig.subplots_adjust(right=0.88, left=0.15, top=0.9)
    
        cbax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(fringe_matrix, cax=cbax, orientation='vertical')
    
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_efficiency.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_efficiency.eps', bbox_inches=0)    
        
        # fig = PLT.figure(figsize=(6,6))
        # ax = fig.add_subplot(111)
        # fringe_matrix_SNR = ax.imshow(NP.abs(norm_fringe_cycle_coherence_SNR), origin='lower', extent=[za.min(), 90.0, u.min(), u.max()], norm=PLTC.LogNorm(vmin=NP.abs(norm_fringe_cycle_coherence_SNR).min(), vmax=NP.abs(norm_fringe_cycle_coherence_SNR).max()))
        # ax.set_xlim(za.min(), 90.0)
        # ax.set_ylim(u.min(), u.max())
        # ax.set_xlabel(r'$|\,\theta\,|$ [degrees]', fontsize=14)
        # ax.set_ylabel(r'$b/\lambda$', fontsize=14)    
        # ax.set_aspect('auto')
    
        # axt = ax.twiny()
        # axt.set_xticks(ax.get_xticks())
        # axt.set_xbound(ax.get_xbound())
        # axt.set_xticklabels(za2sinza(ax.get_xticks()))
        # axt.set_xlabel(r'$|\,\sin\,\theta\,|$', fontsize=14)
    
        # fig.subplots_adjust(right=0.88, left=0.15, top=0.9)
    
        # cbax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        # cbar = fig.colorbar(fringe_matrix_SNR, cax=cbax, orientation='vertical')
        
        PLT.show()
        
    if plot_03:
    
        # 03) Fringe cycle coherence from 2D sky
        
        u = 2.0 ** NP.arange(10)
        u_max = u.max()
    
        dsa = 1/(u_max ** 2)
        npix_orig = 4 * NP.pi / (dsa/8)
    
        nside = HP.pixelfunc.get_min_valid_nside(npix_orig)
        npix = HP.nside2npix(nside)
        pixarea = HP.nside2pixarea(nside)
        nring = int(4*nside - 1)
        isotheta = NP.pi/(nring+1) * (1 + NP.arange(nring))
        ison = NP.sin(isotheta)
    
        theta, az = HP.pix2ang(nside, NP.arange(npix))
        n = NP.cos(theta)
    
        qrtr_sky_ind, = NP.where((theta <= NP.pi/2) & (az <= NP.pi))
        theta = theta[qrtr_sky_ind]
        az = az[qrtr_sky_ind]
        n = n[qrtr_sky_ind]
        
        fringes = NP.exp(-1j * 2*NP.pi * u.reshape(-1,1) * n.reshape(1,-1))
        fringe_cycle_coherence = NP.zeros((u.size, u_max), dtype=NP.complex64)
        sample_counts = NP.zeros((u.size, u_max))
        
        dn = 1/u
        last_fringe_cycle_boundary_n = 1.0-dn
        last_fringe_cycle_boundary_za = NP.degrees(NP.arcsin(last_fringe_cycle_boundary_n))
    
        eps = 1e-10
        n_intervals = NP.arange(0, 1, dn[-1])
        n_intervals = NP.append(n_intervals, 1.0) - eps
        ang_intervals = NP.degrees(NP.arcsin(n_intervals))
        bincount, n_bin_edges, binnum, ri = OPS.binned_statistic(n, statistic='count', bins=n_intervals)
        for ui in range(u.size):
            for i in range(u[ui].astype(NP.int)):
                begini = i*u_max/u[ui]
                endi = (i+1)*u_max/u[ui]
                begini = begini.astype(NP.int)
                endi = endi.astype(NP.int)                
                fine_fringe_cycle_ind = NP.arange(begini, endi)
                ind = ri[ri[begini]:ri[endi]]
                fringe_cycle_coherence[ui,fine_fringe_cycle_ind] = NP.sum(fringes[ui,ind])
                sample_counts[ui,fine_fringe_cycle_ind] = float(ind.size)
    
        norm_fringe_cycle_coherence = fringe_cycle_coherence / sample_counts
        fringe_cycle_efficiency = fringe_cycle_coherence / (npix/4)
        fringe_cycle_solid_angle = sample_counts * pixarea
    
        fig = PLT.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        fringe_matrix = ax.pcolormesh(n_intervals[:-1], u, NP.abs(fringe_cycle_efficiency), norm=PLTC.LogNorm(vmin=1e-8, vmax=NP.abs(fringe_cycle_efficiency).max()))
        # fringe_matrix = ax.pcolormesh(n_intervals[:-1], u, NP.abs(norm_fringe_cycle_coherence), norm=PLTC.LogNorm(vmin=1e-6, vmax=NP.abs(norm_fringe_cycle_coherence).max()))
        last_fringe_cycle_edge = ax.plot(last_fringe_cycle_boundary_n, 1.5*u, ls='--', lw=2, color='black')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(u.min(), u.max())
        ax.set_yscale('log')
        ax.set_xlabel(r'$|\,\sin\,\theta\,|$', fontsize=14)
        ax.set_ylabel(r'$b/\lambda$', fontsize=14)    
        ax.set_aspect('auto')
    
        axt = ax.twiny()
        axt.set_xticks(ax.get_xticks())
        axt.set_xbound(ax.get_xbound())
        axt.set_xticklabels(sinza2za(ax.get_xticks()))
        axt.set_xlabel(r'$|\,\theta\,|$ [degrees]', fontsize=14)
    
        fig.subplots_adjust(right=0.88, left=0.15, top=0.9)
    
        cbax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(fringe_matrix, cax=cbax, orientation='vertical')
    
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_efficiency_2D.png', bbox_inches=0)
        PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_efficiency_2D.eps', bbox_inches=0)    

        # fig = PLT.figure(figsize=(6,6))
        # ax = fig.add_subplot(111)
        # solid_angle_matrix = ax.pcolormesh(n_intervals[:-1], u, NP.abs(fringe_cycle_solid_angle), norm=PLTC.LogNorm(vmin=NP.abs(fringe_cycle_solid_angle).min(), vmax=NP.abs(fringe_cycle_solid_angle).max()))
        # last_fringe_cycle_edge = ax.plot(last_fringe_cycle_boundary_n, u, ls='--', lw=2, color='black')
        # ax.set_xlim(0.0, 1.0)
        # ax.set_ylim(u.min(), u.max())
        # ax.set_yscale('log')
        # ax.set_xlabel(r'$|\,\sin\,\theta\,|$', fontsize=14)
        # ax.set_ylabel(r'$b/\lambda$', fontsize=14)    
        # ax.set_aspect('auto')
    
        # axt = ax.twiny()
        # axt.set_xticks(ax.get_xticks())
        # axt.set_xbound(ax.get_xbound())
        # axt.set_xticklabels(sinza2za(ax.get_xticks()))
        # axt.set_xlabel(r'$|\,\theta\,|$ [degrees]', fontsize=14)
    
        # fig.subplots_adjust(right=0.88, left=0.15, top=0.9)
    
        # cbax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        # cbar = fig.colorbar(solid_angle_matrix, cax=cbax, orientation='vertical')
        # cbax.set_xlabel(r'$\Omega$ [Sr]', fontsize=12, labelpad=10)
        # cbax.xaxis.set_label_position('top')
    
        # PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_solid_angle_2D.png', bbox_inches=0)
        # PLT.savefig('/data3/t_nithyanandan/project_global_EoR/figures/fringe_cycle_solid_angle_2D.eps', bbox_inches=0)    
        
    if plot_04:
    
        # 04) Delay bin coherence from 2D sky
        
        # f0 = 30e9    # Center frequency in Hz
        # bw = 1e9   # Bandwidth in Hz
        # nchan = 64  # number of frequency channels
        # l_delay_bins = 32
        # bl = 0.5 * l_delay_bins / bw * FCNST.c  # baseline length in m
        # wl0 = FCNST.c / f0
        # freq_resolution = bw / nchan
        # chans = (NP.arange(nchan) - nchan/2) * freq_resolution + f0
        # wl = FCNST.c / chans
        # l_binsize = 1.0 / (l_delay_bins/2)

        # u = bl / wl
        # u_max = u.max()

        # dl_coarse = 1.0 / u_max           # resolution in direction cosine
        # eps = 1e-10
        # # nl = NP.ceil(1.0 / dl_coarse).astype(NP.int)
        # # dl_coarse = 1.0 / nl
        # dl = dl_coarse / 4
        # dm = dl_coarse
        # lv = NP.arange(0.0, 1.0, dl)
        # # lv = NP.append(lv, 1.0-eps)
        # # mv = NP.arange(-1.0+eps, 1.0-eps, dm)
        # mv = NP.append(-lv[-1:0:-1], lv)
        # # mv = NP.append(mv, 1.0-eps) 

        # lgrid, mgrid = NP.meshgrid(lv, mv)
        # lmrad = NP.sqrt(lgrid**2 + mgrid**2)
        # ngrid = NP.empty_like(lgrid)
        # sagrid = NP.empty_like(lgrid)
        # ngrid.fill(NP.nan)
        # sagrid.fill(NP.nan)
        # PDB.set_trace()
        # valid_ind = lmrad <= 1.0
        # ngrid[valid_ind] = NP.sqrt(1.0 - lmrad[valid_ind]**2)
        # sagrid[valid_ind] = dl * dm / ngrid[valid_ind]

        # lvect = lgrid[valid_ind]
        # mvect = mgrid[valid_ind]
        # nvect = ngrid[valid_ind]
        # savect = sagrid[valid_ind]
        
        # # fringes = NP.exp(-1j * 2*NP.pi * u.reshape(-1,1) * lvect.reshape(1,-1))
        # fringes = NP.empty((lgrid.shape[0], lgrid.shape[1], u.shape[0]), dtype=NP.complex64)
        # fringes.fill(NP.nan)
        # fringes[valid_ind] = NP.exp(-1j * 2*NP.pi * u.reshape(1,1,-1) * lgrid[:,:,NP.newaxis])
        # delay_bin_coherence = NP.zeros((u.size, l_delay_bins/2), dtype=NP.complex64)
    
        # l_intervals = NP.arange(0.0, 1.0, l_binsize)
        # l_intervals = NP.append(l_intervals, 1.0-eps)
        # delay_intervals = bl * l_intervals / FCNST.c
        # bincount, l_bin_edges, binnum, ri = OPS.binned_statistic(lvect, statistic='count', bins=l_intervals)
        # PDB.set_trace()

        # for di in range(l_delay_bins/2):
        #     ind = ri[ri[di]:ri[di+1]]
        #     delay_bin_coherence[:,di] = NP.sum(fringes[:,ind] * savect[ind], axis=1)

        # lgrid, mgrid = NP.meshgrid(lv, mv)
        # lgrid = lgrid.ravel()
        # mgrid = mgrid.ravel()
        # lmrad = NP.sqrt(lgrid**2 + mgrid**2)
        # ngrid = NP.empty_like(lgrid)
        # sagrid = NP.empty_like(lgrid)
        # ngrid.fill(NP.nan)
        # sagrid.fill(NP.nan)
        # valid_ind = lmrad <= 1.0
        # ngrid[valid_ind] = NP.sqrt(1.0 - lmrad[valid_ind]**2)
        # sagrid[valid_ind] = dl * dm / ngrid[valid_ind]

        # lvect = lgrid[valid_ind]
        # mvect = mgrid[valid_ind]
        # nvect = ngrid[valid_ind]
        # savect = sagrid[valid_ind]
        
        # fringes = NP.exp(-1j * 2*NP.pi * u.reshape(-1,1) * lvect.reshape(1,-1))
        # delay_bin_coherence = NP.zeros((u.size, l_delay_bins/2), dtype=NP.complex64)
    
        # l_intervals = NP.arange(0.0, 1.0, l_binsize)
        # l_intervals = NP.append(l_intervals, 1.0-eps)
        # delay_intervals = bl * l_intervals / FCNST.c
        # bincount, l_bin_edges, binnum, ri = OPS.binned_statistic(lvect, statistic='count', bins=l_intervals)
        # PDB.set_trace()

        # for di in range(l_delay_bins/2):
        #     ind = ri[ri[di]:ri[di+1]]
        #     delay_bin_coherence[:,di] = NP.sum(fringes[:,ind] * savect[ind], axis=1)

            
        # f0 = 30e9    # Center frequency in Hz
        # bw = 1e9   # Bandwidth in Hz
        # nchan = 64  # number of frequency channels
        # n_delay_bins = 32
        # bl = 0.5 * n_delay_bins / bw * FCNST.c  # baseline length in m
        # wl0 = FCNST.c / f0
        # freq_resolution = bw / nchan
        # chans = (NP.arange(nchan) - nchan/2) * freq_resolution + f0
        # wl = FCNST.c / chans
        # n_binsize = 1.0 / (n_delay_bins/2)

        # u = bl / wl
        # u_max = u.max()

        # dsa = 1/(u_max ** 2)   # solid angle resolution
        # npix_orig = 4 * NP.pi / (dsa/4)
    
        # nside = HP.pixelfunc.get_min_valid_nside(npix_orig)
        # npix = HP.nside2npix(nside)
        # pixarea = HP.nside2pixarea(nside)
        # # nring = int(4*nside - 1)
        # # isotheta = NP.pi/(nring+1) * (1 + NP.arange(nring))
        # # ison = NP.sin(isotheta)
    
        # theta, az = HP.pix2ang(nside, NP.arange(npix))
        # n = NP.cos(theta)

        # qrtr_sky_ind, = NP.where((theta <= NP.pi/2) & (az <= NP.pi))
        # theta = theta[qrtr_sky_ind]
        # az = az[qrtr_sky_ind]
        # n = n[qrtr_sky_ind]
        
        # fringes = NP.exp(-1j * 2*NP.pi * u.reshape(-1,1) * n.reshape(1,-1))
        # delay_bin_coherence = NP.zeros((u.size, n_delay_bins/2), dtype=NP.complex64)

        # PDB.set_trace()
        # eps = 1e-10
        # n_intervals = NP.arange(0.0, 1.0, n_binsize)
        # n_intervals = NP.append(n_intervals, 1.0) - eps
        # delay_intervals = bl * n_intervals / FCNST.c
        # bincount, n_bin_edges, binnum, ri = OPS.binned_statistic(n, statistic='count', bins=n_intervals)
        # for di in range(n_delay_bins/2):
        #     ind = ri[ri[di]:ri[di+1]]
        #     delay_bin_coherence[:,di] = NP.sum(fringes[:,ind], axis=1) * pixarea
        # #     # for ui in range(u.size):
        # #     #     delay_bin_coherence[ui,di] = NP.sum(fringes[ui,ind]) * pixarea

        # delay_spectrum_coherence = NP.fft.ifftshift(NP.fft.ifft2(delay_bin_coherence, axes=(0,)), axes=0) * bw

        # fig = PLT.figure(figsize=(6,6))
        # ax = fig.add_subplot(111)
        # dspec = ax.imshow(NP.abs(delay_bin_coherence), origin='lower', extent=[delay_intervals.min(), delay_intervals.max(), chans.min(), chans.max()])
        # ax.set_aspect('auto')

        # fig = PLT.figure(figsize=(6,6))
        # ax = fig.add_subplot(111)
        # dspec = ax.imshow(NP.abs(delay_spectrum_coherence), origin='lower', extent=[delay_intervals.min(), delay_intervals.max(), -0.5/freq_resolution, 0.5/freq_resolution])
        # ax.set_aspect('auto')
        # PLT.show()

        f0 = 30e9    # Center frequency in Hz
        bw = 1e9   # Bandwidth in Hz
        nchan = 40  # number of frequency channels
        n_delay_bins = 32
        bl = 0.5 * n_delay_bins / bw * FCNST.c  # baseline length in m
        wl0 = FCNST.c / f0
        freq_resolution = bw / nchan
        chans = (NP.arange(nchan) - nchan/2) * freq_resolution + f0
        wl = FCNST.c / chans
        n_binsize = 1.0 / (n_delay_bins/2)

        u = bl / wl
        u_max = u.max()
                
        dsa = 1/(u_max ** 2)
        npix_orig = 4 * NP.pi / (dsa/8)
    
        nside = HP.pixelfunc.get_min_valid_nside(npix_orig)
        npix = HP.nside2npix(nside)
        pixarea = HP.nside2pixarea(nside)
        nring = int(4*nside - 1)
        isotheta = NP.pi/(nring+1) * (1 + NP.arange(nring))
        ison = NP.sin(isotheta)
    
        theta, az = HP.pix2ang(nside, NP.arange(npix))
        n = NP.cos(theta)
    
        qrtr_sky_ind, = NP.where((theta <= NP.pi/2) & (az <= NP.pi))
        theta = theta[qrtr_sky_ind]
        az = az[qrtr_sky_ind]
        n = n[qrtr_sky_ind]
        sortind = NP.argsort(n)
        nsorted = n[sortind]

        fringes = NP.exp(-1j * 2*NP.pi * u.reshape(-1,1) * nsorted.reshape(1,-1))
        accumulated_fringes = accumulate_fringes(fringes) * pixarea
        fringe_cycle_coherence = accumulated_fringe_differencing(accumulated_fringes, nsorted, 1/u_max)
        
        PDB.set_trace()
        nproc = max(MP.cpu_count()/2-1, 1)
        # chunksize = int(ceil(u.size/float(nproc)))
        pool = MP.Pool(processes=nproc)
        fringe_cycle_coherences = pool.map(unwrap_fringe_coherence, IT.izip(u.tolist(), IT.repeat(nsorted)))
        PDB.set_trace()

        # fringes = NP.exp(-1j * 2*NP.pi * u.reshape(-1,1) * n.reshape(1,-1))
        # fringe_cycle_coherence = NP.zeros((chans.size, NP.ceil(u[0]).astype(NP.int)), dtype=NP.complex64)
        
        # dn = 1/u
        # last_fringe_cycle_boundary_n = 1.0-dn
        # last_fringe_cycle_boundary_za = NP.degrees(NP.arcsin(last_fringe_cycle_boundary_n))

        # eps = 1e-10
        # n_intervals = NP.arange(0, 1, dn[0])
        # n_intervals = NP.append(n_intervals, 1.0) - eps
        # bincount, n_bin_edges, binnum, ri = OPS.binned_statistic(n, statistic='count', bins=n_intervals)
        # PDB.set_trace()
        # for ui in range(u.size):
        #     for i in range(bincount.size):
        #         ind = ri[ri[i]:ri[i+1]]
        #         fringe_cycle_coherence[ui,i] = NP.sum(fringes[ui,ind]) * pixarea
        #     if (ui == 0) or (ui == chans.size-1): PDB.set_trace()
        

if __name__ == '__main__':
    main()
