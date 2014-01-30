import sys as SYS
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.animation as MOV
import baseline_delay_horizon as DLY
import geometry as GEOM
import interferometry as RI

# antenna_file = 'c:/Users/Regular/My Documents/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
antenna_file = '/Users/t_nithyanandan/Downloads/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

ant_locs = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(1,2,3))
bl = RI.baseline_generator(ant_locs, auto=False, conjugate=False)
bl_lengths = NP.sqrt(NP.sum(bl**2,axis=1))

decl = -27.0 # in degrees
lat = -27.0 # in degrees
ha_range = 4.0 # in hrs
ha_step = 1.0/60. # in hrs
ha = 15.0*NP.arange(-ha_range/2, ha_range/2, ha_step)
ha = NP.asarray(ha).reshape(-1,1)
dec = decl + NP.zeros((len(ha),1))
dec.reshape(-1,1)

hadec = NP.hstack((ha, dec))

altaz = GEOM.hadec2altaz(hadec, lat, units='degrees')

dircos = GEOM.altaz2dircos(altaz, units='degrees')

dm = DLY.delay_envelope(bl, dircos, units='mks')

min_delays = dm[:,:,1]-dm[:,:,0]
max_delays = NP.sum(dm,axis=2)

fig = PLT.figure(figsize=(14,8))
ax = PLT.axes([0.12,0.1,0.8,0.8])
PLT.xlim(NP.min(bl_lengths)-1.0,NP.max(bl_lengths)+100.0)
PLT.ylim(1e9*(NP.min(min_delays)-1.e-8),1e9*(NP.max(max_delays)+1.e-8))
PLT.xlabel('Baseline Length [m]', fontsize=18, weight='semibold')
PLT.ylabel('Delay [ns]', fontsize=18, weight='semibold')
PLT.title('Delay Envelope', fontsize=18, weight='semibold')

hadec_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=18)

l, = PLT.plot([], [], 'k.', markersize=2)

# def init():
#     l.set_xdata([])
#     l.set_ydata([])
#     hadec_text.set_text('')
#     return l, hadec_text

def update(i, bl, dmat, hd, line, hd_text):
    line.set_xdata(NP.asarray([bl_lengths, bl_lengths]))
    line.set_ydata(NP.asarray([dmat[i,:,1]+dmat[i,:,0], dmat[i,:,1]-dmat[i,:,0]]))
    # label_str = 'HA = {0:+.3f} hrs, DEC = {1:+.2f} deg'.format(hd[i,0]/15., hd[i,1])

    sign_ha = '+' if hd[i,0] >= 0.0 else '-'
    hh = NP.int(NP.abs(hd[i,0])/15.0)
    mm = NP.int(60.0*(NP.abs(hd[i,0])/15.0 - hh))
    ss = 3600.0*(NP.abs(hd[i,0])/15.0 - hh - mm/60.0)

    sign_dec = '+' if hd[i,1] >= 0.0 else '-'
    dd = NP.int(NP.abs(hd[i,1]))
    amin = NP.int(60.0*(NP.abs(hd[i,1]) - dd))
    asec = 3600.0*(NP.abs(hd[i,1]) - dd - amin/60.0)

    label_str = ' HA: {0}{1:02d} h {2:02d} m {3:06.3f} s\n Dec: {4}{5:02d} d {6:02d} \' {7:05.2f} "'.format(sign_ha, hh, mm, ss, sign_dec, dd, amin, asec)
    hd_text.set_text(label_str)
    return line, hd_text

anim = MOV.FuncAnimation(fig, update, fargs=(bl_lengths, 1.e9*dm, hadec, l, hadec_text), frames=dm.shape[0], interval=100, blit=False)
PLT.show()

anim.save('/Users/t_nithyanandan/Downloads/MWA_delay_horizon.gif', writer='imagemagick', fps=10)
# anim.save('/Users/t_nithyanandan/Downloads/MWA_delay_horizon.mp4', writer='ffmpeg', fps=10)


#########

# l = PLT.plot([], [], 'b+', [], [], 'r+', markersize=10)

# def init():
#     l[0].set_xdata([])
#     l[0].set_ydata([])
#     l[1].set_xdata([])
#     l[1].set_ydata([])
#     hadec_text.set_text('')
#     return l, hadec_text

# def update(i, bl, dmat, hd, line, hd_text):
#     line[0].set_xdata(NP.asarray(bl_lengths))
#     line[0].set_ydata(NP.asarray(dmat[i,:,1]+dmat[i,:,0]))
#     line[1].set_xdata(NP.asarray(bl_lengths))
#     line[1].set_ydata(NP.asarray(dmat[i,:,1]-dmat[i,:,0]))

#     label_str = 'HA = {0:+.3f} hrs, DEC = {1:+.2f} deg'.format(hd[i,0]/15., hd[i,1])
#     hd_text.set_text(label_str)
#     return line, hd_text

# anim = MOV.FuncAnimation(fig, update, fargs=(bl_lengths, 1.e9*dm, hadec, l, hadec_text), frames=dm.shape[0], interval=100, blit=True, init_func=init)
# PLT.show()
