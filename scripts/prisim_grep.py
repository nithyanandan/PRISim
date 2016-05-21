#!python

import os, glob, sys
import yaml
import argparse
import numpy as NP
import astroutils.nonmathops as NMO
import prisim

prisim_path = prisim.__path__[0]+'/'

def findType(refval):
    valtype = ''
    if isinstance(refval, bool):
        valtype = 'bool'
    elif isinstance(refval, str):
        valtype = 'str'
    elif isinstance(refval, (int, float)):
        valtype = 'num'
    elif isinstance(refval, list):
        if isinstance(refval[0], str):
            valtype = 'str'
        elif isinstance(refval[0], (int,float)):
            valtype = 'num'
        else:
            raise TypeError('refval must be a list containing strings or scalar numbers')
    else:
        raise TypeError('refval must be a boolean, string, scalar, list of strings or list of numbers')
    return valtype

def grepBoolean(vals, refval):
    select_ind = NP.equal(vals, refval)
    return select_ind

def grepString(vals, refval):
    select_ind = NP.asarray([val in refval for val in vals], dtype=NP.bool)
    return select_ind

def grepScalarRange(vals, refval):
    select_ind = NP.logical_and(vals >= refval[0], vals <= refval[1])
    return select_ind

def grepValue(vals, refval):
    valtype = findType(refval)
    if valtype == 'bool':
        vals = NP.asarray(vals, dtype=NP.bool)
        select_ind = grepBoolean(vals, refval)
    elif valtype == 'str':
        vals = NP.asarray(vals)
        select_ind = grepString(vals, refval)
    elif valtype == 'num':
        vals = NP.asarray(vals, dtype=NP.float)
        vals[NP.equal(vals, None)] = NP.nan
        select_ind = grepScalarRange(vals, refval)
    else:
        raise TypeError('Unknown type found. Requires debugging')
    return select_ind

def grepPRISim(parms, verbose=True):
    rootdir = parms['dirstruct']['rootdir']
    project = parms['dirstruct']['project']
    if project is None:
        project_dir = ''
    elif isinstance(project, str):
        project_dir = project
    
    if not os.path.isdir(rootdir):
        raise OSError('Specified root directory does not exist')
    
    if not os.path.isdir(rootdir+project_dir):
        raise OSError('Specified project directory does not exist')
    
    if project is None:
        projects = os.listdir(rootdir)
    else:
        projects = [project_dir]
    
    simparms_list = []
    metadata_list = []
    for proj in projects:
        for simrun in os.listdir(rootdir+proj):
            try:
                with open(rootdir+proj+'/'+simrun+'/metainfo/simparms.yaml', 'r') as parmsfile:
                    simparms_list += [{rootdir+proj+'/'+simrun+'/': yaml.safe_load(parmsfile)}]
                with open(rootdir+proj+'/'+simrun+'/metainfo/meta.yaml', 'r') as metafile:
                    metadata_list += [{rootdir+proj+'/'+simrun+'/': yaml.safe_load(metafile)}]
            except IOError:
                pass
    
    parms_list = []
    for simind, parm in enumerate(simparms_list):
        simrunkey = parm.keys()[0]
        parm[simrunkey].update(metadata_list[simind][simrunkey])
        parms_list += [parm[simrunkey]]
            
    reduced_parms = NMO.recursive_find_notNone_in_dict(parms)
    select_ind = NP.asarray([True] * len(parms_list), dtype=NP.bool)
    if verbose:
        print '\nThe following parameters are searched for:'
    for ikey, ival in reduced_parms.iteritems():
        if verbose:
            print '\t'+ikey
        for subkey in ival.iterkeys():
            vals = [parm[ikey][subkey] for parm in parms_list]
            refval = reduced_parms[ikey][subkey]
            select_ind = NP.logical_and(select_ind, grepValue(vals, refval))
            if verbose:
                print '\t\t'+subkey

    select_ind, = NP.where(select_ind)
    outkeys = [metadata_list[ind].keys()[0] for ind in select_ind]
    return outkeys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to search metadata of PRISim simulations')

    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/dbparms/defaultdbparms.yaml', type=file, required=False, help='File specifying input database search parameters')
    
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    args = vars(parser.parse_args())
    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    selectsims = grepPRISim(parms, verbose=args['verbose'])
    print '\nThe following simulation runs were found to contain the searched parameters:\n'
    for simrun in selectsims:
        print '\t'+simrun
    print '\n'
