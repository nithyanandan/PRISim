#!python

import numpy as NP
import healpy as HP
import yaml, h5py
import argparse
import ipdb as PDB

def read_SKA_low_healpix_beam(infile):
    with h5py.File(infile, 'r') as fileobj:
        try:
            array_pattern = fileobj['array_pattern'].value
        except KeyError:
            array_pattern = 1.0
        epattern = {}
        try:
            element_pattern_grp = fileobj['element_pattern']
        except KeyError:
            epattern = {p: 1.0 for p in ['P1', 'P2']}
        else:
            for pol in element_pattern_grp:
                epattern[pol] = element_pattern_grp[pol].value
        net_voltage_pattern = {p: array_pattern * epattern[p] for p in epattern}
        try:
            frequencies = fileobj['frequency'].value
        except KeyError:
            raise KeyError('Key "frequency" not found in input file')
        try: 
            nsides = {p: HP.npix2nside(net_voltage_pattern[p].shape[1]) for p in epattern}
        except ValueError:
            raise ValueError('Invalid nside for data in input file')
        
    return (frequencies, nsides, net_voltage_pattern)

def write_HEALPIX(beaminfo, outfile, outfmt='HDF5'):

    try:
        outfile, beaminfo
    except NameError:
        raise NameError('Inputs outfile and beaminfo must be specified')

    if not isinstance(outfile, str):
        raise TypeError('Output filename must be a string')

    if not isinstance(beaminfo, dict):
        raise TypeError('Input beaminfo must be a dictionary')

    if 'gains' not in beaminfo:
        raise KeyError('Input beaminfo missing "gains" key')
    if 'freqs' not in beaminfo:
        raise KeyError('Input beaminfo missing "freqs" key')
    
    if not isinstance(outfmt, str):
        raise TypeError('Output format must be specified in a string')
    if outfmt.lower() not in ['fits', 'hdf5']:
        raise ValueError('Output file format invalid')

    outfilename = outfile + '.' + outfmt.lower()
    if outfmt.lower() == 'hdf5':
        with h5py.File(outfilename, 'w') as fileobj:
            hdr_grp = fileobj.create_group('header')
            hdr_grp['npol'] = len(beaminfo['gains'].keys())
            hdr_grp['source'] = beaminfo['source']
            hdr_grp['nchan'] = beaminfo['freqs'].size
            hdr_grp['nside'] = beaminfo['nside']
            hdr_grp['gainunit'] = beaminfo['gainunit']
            spec_grp = fileobj.create_group('spectral_info')
            spec_grp['freqs'] = beaminfo['freqs']
            spec_grp['freqs'].attrs['units'] = 'Hz'
            gain_grp = fileobj.create_group('gain_info')
            for key in beaminfo['gains']: # Different polarizations
                dset = gain_grp.create_dataset(key, data=beaminfo['gains'][key], chunks=(1,beaminfo['gains'][key].shape[1]), compression='gzip', compression_opts=9)
    else:
        hdulist = []
        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['EXTNAME'] = 'PRIMARY'
        hdulist[0].header['NPOL'] = (beaminfo['npol'], 'Number of polarizations')
        hdulist[0].header['SOURCE'] = (beaminfo['source'], 'Source of data')
        hdulist[0].header['GAINUNIT'] = (beaminfo['gainunit'], 'Units of gain')
        # hdulist[0].header['NSIDE'] = (beaminfo['nside'], 'NSIDE parameter of HEALPIX')
        # hdulist[0].header['NCHAN'] = (beaminfo['freqs'].size, 'Number of frequency channels')
        for pi,pol in enumerate(pols):
            hdu = fits.ImageHDU(beaminfo['gains'][pol].T, name='BEAM_{0}'.format(pol))
            hdu.header['PIXTYPE'] = ('HEALPIX', 'Type of pixelization')
            hdu.header['ORDERING'] = ('RING', 'Pixel ordering scheme, either RING or NESTED')
            hdu.header['NSIDE'] = (beaminfo['nside'], 'NSIDE parameter of HEALPIX')
            npix = HP.nside2npix(beaminfo['nside'])
            hdu.header['NPIX'] = (npix, 'Number of HEALPIX pixels')
            hdu.header['FIRSTPIX'] = (0, 'First pixel # (0 based)')
            hdu.header['LASTPIX'] = (npix-1, 'Last pixel # (0 based)')
            hdulist += [hdu]
            hdulist += [fits.ImageHDU(beaminfo['freqs'], name='FREQS_{0}'.format(pol))]
        outhdu = fits.HDUList(hdulist)
        outhdu.writeto(outfilename, clobber=True)
            
if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to convert SKA simulated beams into PRISim compatible HDF5 format')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=None, type=file, required=True, help='File specifying input parameters. Example file in prisim/examples/pbparms/SKA_low_healpix_beam_to_HDF5.yaml')
    
    args = vars(parser.parse_args())
    
    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    ioparms = parms['io']
    indir = ioparms['indir']
    infile = indir + ioparms['infile']
    outdir = ioparms['outdir']
    outfmt = ioparms['outfmt']
    outfile = outdir + ioparms['outfile']
    nside_out = parms['processing']['nside_out']
    if nside_out is not None:
        if not isinstance(nside_out, int):
            raise TypeError('nside_out must be an integer')
        if not HP.isnsideok(nside_out):
            raise ValueError('Invalid nside value specified in nside_out')
    gainunit_out = parms['processing']['gainunit_out']
    if gainunit_out is None:
        gainunit_out = 'regular'
    elif not isinstance(gainunit_out, str):
        raise TypeError('gainunit_out must be a string')
    else:
        if gainunit_out not in ['regular', 'dB']:
            raise ValueError('gainunit_out must be set to "dB" or "regular"')
        
    wait_after_run = parms['processing']['wait']
    beam_src = parms['misc']['source']
    pols = ['P1', 'P2']
    
    gains = {}

    freqs, nsides, net_voltage_pattern = read_SKA_low_healpix_beam(infile)
    sortind = NP.argsort(freqs)
    freqs = freqs[sortind]
    for pi,pol in enumerate(pols):
        net_voltage_pattern[pol] = net_voltage_pattern[pol][sortind,:] # nchan x npix
        net_voltage_pattern[pol] = net_voltage_pattern[pol] / NP.amax(NP.abs(net_voltage_pattern[pol]), axis=1, keepdims=True)
        gains[pol] = NP.abs(net_voltage_pattern[pol])**2 # nchan x npix
        current_gainunit = 'regular'
        if nside_out is not None:
            if nside_out != nsides[pol]:
                gains[pol] = 10.0 * NP.log10(gains[pol])
                current_gainunit = 'dB'
                gains[pol] = NP.asarray(HP.ud_grade(list(gains[pol]), nside_out))
                gains[pol] = gains[pol] - NP.amax(gains[pol], axis=0, keepdims=True)
        else:
            nside_out = nsides[pol] # assuming nside for both pols are identical
        if gainunit_out == 'dB':
            if current_gainunit == 'regular':
                gains[pol] = 10.0 * NP.log10(gains[pol])
        else:
            if current_gainunit == 'dB':
                gains[pol] = 10**(gains[pol]/10.0)
                
    beaminfo = {'npol': len(pols), 'nside': nside_out, 'source': beam_src, 'freqs': freqs, 'gains': gains, 'gainunit': gainunit_out}

    write_HEALPIX(beaminfo, outfile, outfmt=outfmt)

    if wait_after_run:
        PDB.set_trace()
    
