from __future__ import division
import numpy as NP
import numpy.ma as MA
import progressbar as PGB
import h5py
import warnings
from astroutils import DSP_modules as DSP
from astroutils import nonmathops as NMO
from astroutils import mathops as OPS
import prisim

prisim_path = prisim.__path__[0]+'/'    

################################################################################

def npz2hdf5(npzfile, hdf5file):

    """
    ----------------------------------------------------------------------------
    Read an input NPZ file containing closure phase data output from CASA and
    save it to HDF5 format

    Inputs:

    npzfile     [string] Input NPZ file including full path containing closure 
                phase data. It must have the following files/keys inside:
                'phase'     [numpy array] Closure phase (radians). It is of 
                            shape ntriads x npol x nchan x ntimes
                'tr'        [numpy array] Array of triad tuples, of shape 
                            ntriads x 3
                'flags'     [numpy array] Array of flags (boolean), of shape
                            ntriads x npol x nchan x ntimes
                'lst'       [numpy array] Array of LST, of size ntimes

    hdf5file    [string] Output HDF5 file including full path.
    ----------------------------------------------------------------------------
    """

    npzdata = NP.load(npzfile)
    cpdata = npzdata['phase']
    triadsdata = npzdata['tr']
    flagsdata = npzdata['flags']
    lstdata = npzdata['LAST']

    cp = NP.asarray(cpdata).astype(NP.float64)
    triads = NP.asarray(triadsdata)
    flags = NP.asarray(flagsdata).astype(NP.bool)
    lst = NP.asarray(lstdata).astype(NP.float64)

    with h5py.File(hdf5file, 'w') as fobj:
        datapool = ['raw']
        for dpool in datapool:
            if dpool == 'raw':
                qtys = ['cphase', 'triads', 'flags', 'lst']
            for qty in qtys:
                if qty == 'cphase':
                    data = NP.copy(cp)
                elif qty == 'triads':
                    data = NP.copy(triads)
                elif qty == 'flags':
                    data = NP.copy(flags)
                elif qty == 'lst':
                    data = NP.copy(lst)
                dset = fobj.create_dataset('{0}/{1}'.format(dpool, qty), data=data, compression='gzip', compression_opts=9)
            
################################################################################
        
class ClosurePhase(object):

    """
    ----------------------------------------------------------------------------
    Class to hold and operate on Closure Phase information. 

    It has the following attributes and member functions.

    Attributes:

    Member functions:

    __init__()      Initialize an instance of class ClosurePhase

    expicp()        Compute and return complex exponential of the closure phase 
                    as a masked array

    smooth_in_tbins()
                    Smooth the complex exponentials of closure phases in time 
                    bins. Both mean and median smoothing is produced.

    save()          Save contents of attribute cpinfo in external HDF5 file
    ----------------------------------------------------------------------------
    """
    
    def __init__(self, infile, infmt='npz'):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class ClosurePhase

        Inputs:

        infile      [string] Input file including full path. It could be a NPZ
                    with raw data, or a HDF5 file that could contain raw or 
                    processed data. The input file format is specified in the 
                    input infmt. If it is a NPZ file, it must contain the 
                    following keys/files:
                    'phase'     [numpy array] Closure phase (radians). It is of 
                                shape ntriads x npol x nchan x ntimes
                    'tr'        [numpy array] Array of triad tuples, of shape 
                                ntriads x 3
                    'flags'     [numpy array] Array of flags (boolean), of shape
                                ntriads x npol x nchan x ntimes
                    'lst'       [numpy array] Array of LST, of size ntimes

        infmt       [string] Input file format. Accepted values are 'npz' 
                    (default) and 'hdf5'.
        ------------------------------------------------------------------------
        """

        if not isinstance(infile, str):
            raise TypeError('Input infile must be a string')

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

    def smooth_in_tbins(self, tbinsize=None):

        """
        ------------------------------------------------------------------------
        Smooth the complex exponentials of closure phases in time bins. Both
        mean and median smoothing is produced.

        Inputs:

        tbinsize    [NoneType or scalar] Time-bin size (in seconds) over which
                    mean and median are estimated
        ------------------------------------------------------------------------
        """

        if tbinsize is not None:
            if not isinstance(tbinsize, (int,float)):
                raise TypeError('Input tbinsize must be a scalar')
            tbinsize = tbinsize / 24 / 3.6e3 # in days
            tres = self.cpinfo['raw']['lst'][1] - self.cpinfo['raw']['lst'][0] # in days
            textent = tres * self.cpinfo['raw']['lst'].size # in seconds
            if tbinsize > tres:
                tbinsize = NP.clip(tbinsize, tres, textent)
                eps = 1e-10
                tbins = NP.arange(self.cpinfo['raw']['lst'].min(), self.cpinfo['raw']['lst'].max() + tres + eps, tbinsize)
                tbinintervals = tbins[1:] - tbins[:-1]
                tbincenters = tbins[:-1] + 0.5 * tbinintervals
                counts, tbin_edges, tbinnum, ri = OPS.binned_statistic(self.cpinfo['raw']['lst'].ravel(), statistic='count', bins=tbins)
                counts = counts.astype(NP.int)

                if 'prelim' not in self.cpinfo['processed']:
                    self.cpinfo['processed']['prelim'] = {}
                self.cpinfo['processed']['prelim']['eicp'] = {}
                self.cpinfo['processed']['prelim']['cphase'] = {}

                wts_tbins = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], self.cpinfo['processed']['native']['eicp'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], counts.size))
                eicp_tmean = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], self.cpinfo['processed']['native']['eicp'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], counts.size), dtype=NP.complex128)
                eicp_tmedian = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], self.cpinfo['processed']['native']['eicp'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], counts.size), dtype=NP.complex128)
                cp_trms = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], self.cpinfo['processed']['native']['eicp'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], counts.size))
                cp_tmad = NP.zeros((self.cpinfo['processed']['native']['eicp'].shape[0], self.cpinfo['processed']['native']['eicp'].shape[1], self.cpinfo['processed']['native']['eicp'].shape[2], counts.size))

                for binnum in xrange(counts.size):
                    ind_tbin = ri[ri[binnum]:ri[binnum+1]]
                    wts_tbins[:,:,:,binnum] = NP.sum(self.cpinfo['processed']['native']['wts'][:,:,:,ind_tbin], axis=3)
                    eicp_tmean[:,:,:,binnum] = NP.exp(1j*NP.angle(MA.mean(self.cpinfo['processed']['native']['eicp'][:,:,:,ind_tbin], axis=3)))
                    eicp_tmedian[:,:,:,binnum] = NP.exp(1j*NP.angle(MA.median(self.cpinfo['processed']['native']['eicp'][:,:,:,ind_tbin].real, axis=3) + 1j * MA.median(self.cpinfo['processed']['native']['eicp'][:,:,:,ind_tbin].imag, axis=3)))
                    cp_trms[:,:,:,binnum] = MA.std(self.cpinfo['processed']['native']['cphase'][:,:,:,ind_tbin], axis=-1).data
                    cp_tmad[:,:,:,binnum] = MA.median(NP.abs(self.cpinfo['processed']['native']['cphase'][:,:,:,ind_tbin] - NP.angle(eicp_tmedian[:,:,:,binnum][:,:,:,NP.newaxis])), axis=-1).data
                mask = wts_tbins <= 0.0
                self.cpinfo['processed']['prelim']['wts'] = MA.array(wts_tbins, mask=mask)
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
        
    ############################################################################
