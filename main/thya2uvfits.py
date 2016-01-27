from interferometry import InterferometerArray
from astropy.io import fits
from astropy.constants import c
import optparse,sys,os
import numpy as n
from astropy.time import Time
import ipdb as PDB

def i2a(i):
    #convert from unit index to MWA ant index
    # assumes zero indexed antenna numbers 0-127
    tens = int(i/8)+1
    ones = i%8+1
    return tens*10+ones
def a2i(a):
    #convert from MWA to unit ant index
    #returns a zero indexed number 0-127
    eights = int(a/10)-1
    ones = a%10
    return eights*8+ones-1
def jds2midnight(t):
    #input: an array of jds
    #output: jd of midnight of the day on which the observation started
    return n.floor(n.min(t)-0.5)+0.5 
i2a = n.vectorize(i2a)
a2i = n.vectorize(a2i)

o = optparse.OptionParser()
o.set_usage('thya2uvfits.py [options]')
o.set_description(__doc__)
#o.add_option('--drift',action='store_true',
#    help='limit pointings to zenith')
opts,args = o.parse_args(sys.argv[1:])

for visfile in args:
    #labels,baselines,channels
    print visfile,"-->"
    outfile = '.'.join(os.path.basename(visfile).split('.')[:-1])+'.uvfits'
    print "     ",outfile
    vis = InterferometerArray(None,None,None,init_file=visfile)
    time = map(float,vis.timestamp)#nithya uses strings of JD time for timestamps
    time = Time(time,scale='utc',format='gps').jd #convert to float JD
    #date_0 = Time(time[0],scale='utc',format='jd').isot
    #phase to some point 
    # use the median lst for RA
    if vis.pointing_coords=='hadec':
        phase_RA = vis.lst[0] - vis.pointing_center[0,0] #use the HA of the first time
    elif vis.pointing_coords=='radec':
        phase_RA = vis.pointing_center[0,0]
    else:
        print "ERROR: Danny is too lazy to convert your silly altaz coordinates."
        print "TODO: see general/modules/geometry.hadec2altaz"
        sys.exit()
    phase_DEC = vis.latitude
    #changes the uvws and data
    print "phasing to:"
    print "ra =",phase_RA
    print "dec =", phase_DEC
    zenith_baselines = vis.baselines.copy()
    vis.phase_centering(n.array([[phase_RA,phase_DEC]]),phase_center_coords='radec',
        do_delay_transform=False)
    vis.project_baselines(
            ref_point={'location':n.array([[phase_RA,phase_DEC]]),
            'coords':'radec'})
    baselines=vis.baselines.copy()
    #print vis.baselines.shape,vis.projected_baselines.shape
    times = vis.timestamp  #string of jds
    times = map(float,times)
    #convert from gps seconds to jd
    #times = Time(times,scale='utc',format='jd').jd
    ant1 = n.array(map(int,[l[0] for l in vis.labels]))
    ant2 = n.array(map(int,[l[1] for l in vis.labels]))
    print "found {n} baselines".format(n=len(ant1))
    print "found {n} antennas".format(n=len(set(ant1)))
    print "min(ant1),max(ant1)",n.min(ant1),n.max(ant1)
    print "min(ant2),max(ant2)",n.min(ant2),n.max(ant2)
    freqs = vis.channels #frequencies in Hz
    
    visibilities = vis.skyvis_freq # dimensions nbls,nchan,ntimes 
    #XXX where is the polarization?  Looking at the nithya code, I think sims are single pol
    # Assume we're only looking at XX for now
    
    
    telescope_name = vis.telescope.get('id','MYSTERY')
    n_freq = visibilities.shape[1]
    n_time = visibilities.shape[2]
    n_bl = visibilities.shape[0]
    n_blt = n_bl*n_time
    n_pol =1
    print "n_freq = ",n_freq
    print "n_time = ",n_time
    print "n_bl = ",n_bl
    print "n_blt = ",n_blt
    

     
    
    ###### The uvfits half of the script
    #at the interface visibilities need to be in dimensions Nblt,Nchan
    print "Converting antennas to index notation"
    ant1 = a2i(ant1)+1
    ant2 = a2i(ant2)+1
    print "ant1 min max = ",n.min(ant1),n.max(ant1)
    print "ant2 min max = ",n.min(ant2),n.max(ant2)
    #Most file formats like i<j  go through the ants and correct this
    i2j = n.argwhere(ant1>ant2).squeeze()
    visibilities[i2j,:,:] = n.conj(visibilities[i2j,:,:])
    to_1 = ant2[i2j].copy()
    ant2[i2j] = ant1[i2j].copy()
    ant1[i2j] = to_1
    vis.projected_baselines[i2j] *= -1


    baselines = n.tile(ant1 + ant2*2**8,n_time)
    print "found ",len(baselines),"baselines"
    #b = ant1 + ant2*(2**8)
    #print ant1[b<0],ant2[b<0]
    print "baseline range ",n.min(baselines),n.max(baselines) 
    uu = n.reshape(vis.projected_baselines[:,0,:],(n_blt),order='F')/c
    vv = n.reshape(vis.projected_baselines[:,1,:],(n_blt),order='F')/c
    ww = n.reshape(vis.projected_baselines[:,2,:],(n_blt),order='F')/c
    
#    time_ref = n.median(times)  
    time_ref = jds2midnight(times) 
    times -= time_ref

    #reshape from n_times to n_blts
    #times.shape = (n_time,1)
    times = n.repeat(times,n_bl)
    #times.shape = (n_blt,)
    
    #use the antenna positions relative to the first antenna in the array
    ref_ant =  ant1[n.argmax([n.sum(ant1==a) for a in set(ant1)])] #the antenna with the most baselines to other
    # antennas, not necessarily guaranteed to give you a full antenna map 
    positions = zenith_baselines[ant1==ref_ant,:]


    #PDB.set_trace()
    #visibilites = n.reshape(visibilities,(n_blt,n_freq))
    #visibilities.shape = (n_blt,n_freq)  #XXX This might need to be n.reshape or similar
    #the initial axes ordering is bl,chan,time
    visibilities = n.swapaxes(visibilities,1,2) #change to bl,time,chan 
    visibilities = n.reshape(visibilities,(n_blt,n_freq),order='F')

    # HANDOFF TO UVFITS WRITING

    v_container = n.zeros((n_blt,1,1,n_freq,n_pol,3))
    v_slice = n.zeros((1,1,n_freq,n_pol,3))    
    # Load using loops. Is there a better way?
    for i in range(n_blt):
        v_slice[0,0,:,0,0] = n.real(visibilities[i,:n_freq])
        v_slice[0,0,:,0,1] = n.imag(visibilities[i,:n_freq])
        v_slice[0,0,:,0,2] = 1.0
        #TODO: add the YYs some day or maybe even XYs?!?
        v_container[i] = v_slice
    uvparnames = ['UU','VV','WW','BASELINE','DATE']
    print "times[0] = ",times[0]
    print "baselines[0] =", baselines[0]
    print "len(baselines)",len(baselines)
    print "len(times)",len(times)
    print "time_ref = ",time_ref
    parvals = [uu,vv,ww,baselines,times]    
    
    hdu10 = fits.GroupData(v_container,parnames=uvparnames,pardata=parvals,bitpix=-32)
    hdu10 = fits.GroupsHDU(hdu10)
    
    hdu10.header['PTYPE1'] = 'UU      '
    hdu10.header['PSCAL1'] = 1.0
    hdu10.header['PZERO1'] = 0.0

    hdu10.header['PTYPE2'] = 'VV      '
    hdu10.header['PSCAL2'] = 1.0
    hdu10.header['PZERO2'] = 0.0

    hdu10.header['PTYPE3'] = 'WW      '
    hdu10.header['PSCAL3'] = 1.0
    hdu10.header['PZERO3'] = 0.0

    hdu10.header['PTYPE4'] = 'BASELINE'
    hdu10.header['PSCAL4'] = 1.0
    hdu10.header['PZERO4'] = 0.0

    hdu10.header['PTYPE5'] = 'DATE    '
    hdu10.header['PSCAL5'] = 1.0
    hdu10.header['PZERO5'] = time_ref #jd midnight

    hdu10.header['DATE-OBS'] = Time(time_ref,scale='utc',format='jd').iso


    hdu10.header['CTYPE2'] = 'COMPLEX '
    hdu10.header['CRVAL2'] = 1.0
    hdu10.header['CRPIX2'] = 1.0
    hdu10.header['CDELT2'] = 1.0
    
    hdu10.header['CTYPE3'] = 'STOKES '
    hdu10.header['CRVAL3'] = -5.0
    hdu10.header['CRPIX3'] = 1.0
    hdu10.header['CDELT3'] = -1.0
    
    #frequency bins need to be checked
    hdu10.header['CTYPE4'] = 'FREQ'
    hdu10.header['CRVAL4'] = freqs[0]#  + 4.0e4
    hdu10.header['CRPIX4'] = 1 # 
    hdu10.header['CDELT4'] = freqs[1]-freqs[0]
    print "freq res = ",freqs[1]-freqs[0]
    
    hdu10.header['CTYPE5'] = 'RA'
    hdu10.header['CRVAL5'] = phase_RA
    hdu10.header['OBJECT'] = 'EoR'
    hdu10.header['TELESCOP'] = telescope_name
    #hdu10.header['OBSRA'] = phase_RA
    #hdu10.header['OBSDEC'] = phase_DEC
    hdu10.header['CTYPE6'] = 'DEC'
    hdu10.header['CRVAL6'] = phase_DEC
    hdu10.header['EPOCH'] =  2000
    hdu10.header['BUNIT'] =  'Jy'
    hdu10.header['BSCALE'] =  1.0
    hdu10.header['BZERO'] =  0.0
    hdu10.header['INSTRUME'] = 'NITHYA'
    
    #
    # Create an hdu for the antenna table
    #
    ####
    # load the antenna positions (stopgap until they are supplied by an update
    positions = n.loadtxt('MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt')
    #4 columns, ant#, E, N, Z      
    n_tiles = len(positions)
    
    annames = n.arange(1,len(positions)+1)#positions[:,0].astype(str)
    print "Tile Names:",annames
    nosta = n.arange(n_tiles) + 1
    mntsta = [0] * n_tiles
    staxof = [0] * n_tiles
    poltya = ['X'] * n_tiles
    polaa = [90.0] * n_tiles
    #polcala = [[0.0, 0.0, 0.0]] * n_tiles
    polcala = positions[:,1:]
    poltyb = ['Y'] * n_tiles
    polab = [0.0] * n_tiles
    polcalb = positions[:,1:]
    
    #stabxyz = [[0.0, 0.0, 0.0]] * n_tiles
    stabxyz = positions[:,1:]
    
    col1 = fits.Column(name='ANNAME', format='8A', array=annames)
    col2 = fits.Column(name='STABXYZ', format='3D', array=stabxyz) 
    col3 = fits.Column(name='NOSTA', format='1J', array=nosta)
    col4 = fits.Column(name='MNTSTA', format='1J', array=mntsta) 
    col5 = fits.Column(name='STAXOF', format='1E', array=staxof) 
    col6 = fits.Column(name='POLTYA', format='1A', array=poltya)
    col7 = fits.Column(name='POLAA', format='1E', array=polaa) 
    col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
    col9 = fits.Column(name='POLTYB', format='1A', array=poltyb) 
    col10 = fits.Column(name='POLAB', format='1E', array=polab)
    col11 = fits.Column(name='POLCALB', format='3E', array=polcalb) 
    
    cols = fits.ColDefs([col1, col2,col3, col4,col5, col6,col7, col8,col9, col10,col11])
    # This only works for astropy 0.4 which is not available from pip
    ant_hdu = fits.BinTableHDU.from_columns(cols)
    
    
    #ant_hdu = fits.new_table(cols)
    ant_hdu.header['EXTNAME'] = 'AIPS AN'
    ant_hdu.header['FREQ'] = freqs[0]
    # Some spoofed antenna table headers, these may need correcting
    ant_hdu.header['ARRAYX'] = -2557572.345962
    ant_hdu.header['ARRAYY'] = 5091627.14195476
    ant_hdu.header['ARRAYZ'] = -2856535.56228611
    ant_hdu.header['GSTIAO'] = 331.448628115495
    ant_hdu.header['DEGPDY'] = 360.985
    ant_hdu.header['DATE'] = times[0]
    ant_hdu.header['POLARX'] = 0.0
    ant_hdu.header['POLARY'] = 0.0
    ant_hdu.header['UT1UTC'] = 0.0
    ant_hdu.header['DATUTC'] = 0.0
    ant_hdu.header['TIMSYS'] = 'UTC  '
    ant_hdu.header['ARRNAM'] = telescope_name
    ant_hdu.header['NUMORB'] = 0
    ant_hdu.header['NOPCAL'] = 3
    ant_hdu.header['FREQID'] = -1
    ant_hdu.header['IATUTC'] = 35.
    
    # Create hdulist and write out file
    
    hdulist = fits.HDUList(hdus=[hdu10,ant_hdu])
    hdulist.writeto(outfile,clobber=True)
    PDB.set_trace()
