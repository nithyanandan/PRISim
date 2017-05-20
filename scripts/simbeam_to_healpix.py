#!python

import ast
import numpy as NP
import healpy as HP
import yaml, h5py
import argparse
from scipy import interpolate
import ipdb as PDB

def read_FEKO(infile):
    freqs = []
    theta_list = []
    phi_list = []
    gaindB = []
    ntheta = None
    nphi = None
    theta_range = [0.0, 0.0]
    phi_range = [0.0, 0.0]
    with open(infile, 'r') as fileobj:
        for linenum,line in enumerate(fileobj.readlines()):
            words = line.split()
            if 'Frequency' in line:
                freqs += [ast.literal_eval(words[1])]
                gaindB += [[]]
            if ntheta is None:
                if 'Theta Samples' in line:
                    ntheta = ast.literal_eval(words[-1])
            if nphi is None:
                if 'Phi Samples' in line:
                    nphi = ast.literal_eval(words[-1])
            if (line[0] != '#') and (line[0] != '*') and (len(words) > 0):
                gaindB[-1] += [ast.literal_eval(words[-1])]
                if len(gaindB) <= 1:
                    theta_list += [ast.literal_eval(words[0])]
                    phi_list += [ast.literal_eval(words[1])]
    if len(gaindB) != len(freqs):
        raise IndexError('Number of frequencies do not match number of channels in gains. Requires debugging.')
    freqs = NP.asarray(freqs)
    theta_list = NP.asarray(theta_list)
    phi_list = NP.asarray(phi_list)
    gaindB = NP.asarray(gaindB)
    theta = NP.linspace(theta_list.min(), theta_list.max(), ntheta)
    phi = NP.linspace(phi_list.min(), phi_list.max(), nphi)
    return (freqs, theta_list, phi_list, theta, phi, gaindB)

def convert_to_healpix(theta, phi, gains, nside=32, interp_method='spline', gainunits='dB', angunits='radians'):
    try:
        theta, phi, gains
    except NameError:
        raise NameError('Inputs theta, phi and gains must be specified')
    if not HP.isnsideok(nside):
        raise ValueError('Specified nside invalid')
    if not isinstance(interp_method, str):
        raise TypeError('Input interp_method must be a string')
    if interp_method not in ['spline', 'nearest', 'healpix']:
        raise valueError('Input interp_method value specified is invalid')
    if gains.shape == (theta.size, phi.size):
        gridded = True
    elif (gains.size == theta.size) and (gains.size == phi.size):
        gridded = False
    else:
        raise ValueError('Inputs theta, phi and gains have incompatible dimensions')
    
    if angunits.lower() == 'degrees':
        theta = NP.radians(theta)
        phi = NP.radians(phi)

    hmap = NP.empty(HP.nside2npix(nside))
    wtsmap = NP.empty(HP.nside2npix(nside))
    hmap.fill(NP.nan)
    wtsmap.fill(NP.nan)
    
    if interp_method == 'spline':
        if gainunits.lower() != 'db':
            gains = 10.0 * NP.log10(gains)
        # gains -= NP.amax(gains)
        hpxtheta, hpxphi = HP.pix2ang(nside, NP.arange(HP.nside2npix(nside)))
        if gridded:
            interp_func = interpolate.RectBivariateSpline(theta, phi, gains)
            hmap = interp_func.ev(hpxtheta, hpxphi)
        else:
            # interp_func = interpolate.interp2d(theta, phi, gains, kind='cubic')
            # hmap = interp_func(hpxtheta, hpxphi)
            hmap = interpolate.griddata(NP.hstack((theta.reshape(-1,1),phi.reshape(-1,1))), gains, NP.hstack((hpxtheta.reshape(-1,1),hpxphi.reshape(-1,1))), method='cubic')
        hmap -= NP.nanmax(hmap)
        if gainunits.lower() != 'db':
            hmap = 10**(hmap/10)
    else:
        if gainunits.lower() == 'db':
            gains = 10**(gains/10.0)
        # gains /= NP.amax(gains)
        if gridded:
            phi_flattened, theta_flattened = NP.meshgrid(phi, theta)
            theta_flattened = theta_flattened.flatten()
            phi_flattened = phi_flattened.flatten()
            gains = gains.flatten()
        else:
            theta_flattened = theta
            phi_flattened = phi
        if interp_method == 'healpix':
            ngbrs, wts = HP.get_interp_weights(nside, theta_flattened, phi=phi_flattened)
            hmap[NP.unique(ngbrs)] = 0.0
            wtsmap[NP.unique(ngbrs)] = 0.0
            for i in xrange(theta_flattened.size):
                hmap[ngbrs[:,i]] += wts[:,i] * gains[i]
                wtsmap[ngbrs[:,i]] += wts[:,i]
        else: # nearest neighbour
            nnpix = HP.ang2pix(nside, theta_flattened, phi_flattened)
            hmap[NP.unique(nnpix)] = 0.0
            wtsmap[NP.unique(nnpix)] = 0.0
            for i in xrange(theta_flattened.size):
                hmap[nnpix[i]] += gains[i]
                wtsmap[nnpix[i]] += 1
        ind_nan = NP.isnan(wtsmap)
        other_nanind = wtsmap < 1e-12
        ind_nan = ind_nan | other_nanind
        wtsmap[ind_nan] = NP.nan
        hmap /= wtsmap
        hmap /= NP.nanmax(hmap)
        if gainunits.lower() == 'db':
            hmap = 10.0 * NP.log10(hmap)
    ind_nan = NP.isnan(hmap)
    hmap[ind_nan] = HP.UNSEEN

    return hmap

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to convert simulated beams into healpix format')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=None, type=file, required=True, help='File specifying input parameters')
    
    args = vars(parser.parse_args())
    
    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    ioparms = parms['io']
    indir = ioparms['indir']
    infmt = ioparms['infmt']
    infile = indir + ioparms['infile']
    outdir = ioparms['outdir']
    outfmt = ioparms['outfmt']
    outfile = outdir + ioparms['outfile'] + outfmt.lower()
    nside = parms['processing']['nside']
    gainunits = parms['processing']['gainunits']
    interp_method = parms['processing']['interp']
    
    if infmt.lower() == 'feko':
        freqs, theta_list, phi_list, theta, phi, gaindB = read_FEKO(infile)
    
    hmap1 = convert_to_healpix(theta_list, phi_list, gaindB[0], interp_method='spline', gainunits=gainunits, angunits='degrees')
    hmap2 = convert_to_healpix(theta, phi, gaindB[0].reshape(phi.size,theta.size).T, interp_method='spline', gainunits=gainunits, angunits='degrees')
    hmap3 = convert_to_healpix(theta, phi, gaindB[0].reshape(phi.size,theta.size).T, interp_method=interp_method, gainunits=gainunits, angunits='degrees')
    hmap4 = convert_to_healpix(theta_list, phi_list, gaindB[0], interp_method='healpix', gainunits=gainunits, angunits='degrees')
    hmap5 = convert_to_healpix(theta_list, phi_list, gaindB[0], interp_method='nearest', gainunits=gainunits, angunits='degrees')

    PDB.set_trace()
