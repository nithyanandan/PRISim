import numpy as NP 
from astropy.io import fits
from astropy.io import ascii
import scipy.constants as FCNST
from scipy import interpolate
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.animation as MOV
from scipy.interpolate import griddata
import datetime as DT
import time 
import progressbar as PGB
import healpy as HP
import geometry as GEOM
import interferometry as RI
import catalog as CTLG
import constants as CNST
import my_DSP_modules as DSP 
import my_operations as OPS
import primary_beams as PB
import baseline_delay_horizon as DLY
import ipdb as PDB

HERA_beam_model_file = '/data3/t_nithyanandan/project_HERA/parameters/gain_full_pattern_screen_cone.csv'
beamdata = ascii.read(HERA_beam_model_file, format='csv', data_start=1)
colnames = beamdata.colnames
theta_model = beamdata[beamdata.colnames[0]]
phi_model = NP.arange(0,180,5)
freq_model = NP.arange(0.08, 0.22, 0.01)

beam_model = []
for i in xrange(len(beamdata.colnames)-1):
    if i == 0:
        beam_model = beamdata[colnames[i+1]].reshape(-1,1)
    else:
        beam_model = NP.hstack((beam_model, beamdata[colnames[i+1]].reshape(-1,1)))

beam_model = beam_model.reshape(theta_model.size, freq_model.size, phi_model.size)
beam_model = 10**(beam_model/10)  # Convert from dB to linear units

max_beam_model = NP.amax(NP.amax(beam_model, axis=0, keepdims=True), axis=2, keepdims=True)
beam_model /= max_beam_model



