#!python

import os, glob, sys
import yaml
import argparse
import numpy as NP
import astroutils.nonmathops as NMO
import prisim

prisim_path = prisim.__path__[0]+'/'

def searchPRISimDB(parms):

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
            with open(rootdir+proj+'/'+simrun+'/metainfo/simparms.yaml', 'r') as parmsfile:
                simparms_list += [{rootdir+proj+'/'+simrun+'/': yaml.safe_load(parmsfile)}]
            with open(rootdir+proj+'/'+simrun+'/metainfo/meta.yaml', 'r') as metafile:
                metadata_list += [{rootdir+proj+'/'+simrun+'/': yaml.safe_load(metafile)}]
    
    parms_list = []
    for simind, simrun in enumerate(os.listdir(rootdir+proj)):
        parm = simparms_list[simind].copy()
        simrunkey = rootdir+proj+'/'+simrun+'/'
        parm[simrunkey].update(metadata_list[simind][simrunkey])
        parms_list += [parm[simrunkey]]

    reduced_parms = NMO.recursive_find_notNone_in_dict(parms)
    select_ind = NP.asarray([True] * len(parms_list))
    for ikey, ival in reduced_parms.iteritems():
        if ikey == 'telescope':
            if 'id' in ival:
                telescope_ids = NP.asarray([parm['telescope']['id'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.asarray([tscope in reduced_parms['telescope']['id'] for tscope in telescope_ids]))
            if 'latitude' in ival:
                latitudes = NP.asarray([parm['telescope']['latitude'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.logical_and(latitudes >= reduced_parms['telescope']['latitude'][0], latitudes <= reduced_parms['telescope']['latitude'][1]))
            if 'longitude' in ival:
                longitudes = NP.asarray([parm['telescope']['longitude'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.logical_and(longitudes >= reduced_parms['telescope']['longitude'][0], longitudes <= reduced_parms['telescope']['longitude'][1]))
            if 'A_eff' in ival:
                effective_areas = NP.asarray([parm['telescope']['A_eff'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.logical_and(effective_areas >= reduced_parms['telescope']['A_eff'][0], effective_areas <= reduced_parms['telescope']['A_eff'][1]))
            if 'Tsys' in ival:
                system_temperatures = NP.asarray([parm['telescope']['Tsys'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.logical_and(system_temperatures >= reduced_parms['telescope']['Tsys'][0], system_temperatures <= reduced_parms['telescope']['Tsys'][1]))

    select_ind, = NP.where(select_ind)
    outkeys = [metadata_list[ind].keys()[0] for ind in select_ind]
    for okey in outkeys:
        print okey

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to search metadata of PRISim simulations')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/dbparms/defaultdbparms.yaml', type=file, required=False, help='File specifying input database search parameters')
    
    args = vars(parser.parse_args())
    
    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    searchPRISimDB(parms)



