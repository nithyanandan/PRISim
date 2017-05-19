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
        gaindB_channel = None
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
    
    if infmt.lower() == 'feko':
        freqs, theta_list, phit_list, theta, phi, gaindB = read_FEKO(infile)
    
    
