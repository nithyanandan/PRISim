import numpy as NP
import argparse
import yaml
from astropy.io import ascii
import matplotlib.pyplot as PLT
import interferometry as RI
import ipdb as PDB

parser = argparse.ArgumentParser(description='Program to generate antennas in a circular ring.')
parser.add_argument('-i','--input', dest='parms', help='Parameters input file name', type=file, required=True)

args = vars(parser.parse_args())

with args['parms'] as parmsfile:
    cfg = yaml.safe_load(parmsfile)

if cfg['project'] not in ['project_MWA', 'project_global_EoR', 'project_HERA', 'project_drift_scan', 'project_beams', 'project_LSTbin']:
    raise ValueError('Project does not exist')

project_dir = cfg['project']

if cfg['array']['layout'] != 'CIRC':
    raise ValueError('Antenna layout specified is not a circular ring')

if cfg['array']['minR'] is None:
    raise ValueError('Minimum radius of circular ring not specified')

minR = cfg['array']['minR']
maxR = cfg['array']['maxR']

if cfg['antenna']['size'] is None:
    raise ValueError('Antenna size not specified')

antsize = cfg['antenna']['size']

PDB.set_trace()
aalayout = RI.circular_antenna_array(antsize, minR, maxR=maxR)
bl, blid = RI.baseline_generator(aalayout, auto=False, conjugate=False)
ubl, ublind, ublcount = RI.uniq_baselines(bl)
bll = NP.sqrt(NP.sum(ubl**2, axis=1))
ubll, ubllind, ubllcount = RI.uniq_baselines(NP.hstack((bll.reshape(-1,1),NP.zeros((bll.size,1)),NP.zeros((bll.size,1)))))
ubll = bll[ubllind]
    
PDB.set_trace()




