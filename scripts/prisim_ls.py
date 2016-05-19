#!python

import os, glob, sys
import yaml
import argparse
import numpy as NP
import numpy.ma as ma
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
    select_ind = NP.asarray([True] * len(parms_list))
    for ikey, ival in reduced_parms.iteritems():
        if ikey == 'telescope':
            if 'id' in ival:
                telescope_ids = NP.asarray([parm['telescope']['id'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.asarray([tscope in reduced_parms['telescope']['id'] for tscope in telescope_ids]))
            if 'latitude' in ival:
                latitudes = NP.asarray([parm['telescope']['latitude'] for parm in parms_list], dtype=NP.float)
                latitudes[NP.equal(latitudes, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(latitudes >= reduced_parms['telescope']['latitude'][0], latitudes <= reduced_parms['telescope']['latitude'][1]))
            if 'longitude' in ival:
                longitudes = NP.asarray([parm['telescope']['longitude'] for parm in parms_list], dtype=NP.float)
                longitudes[NP.equal(longitudes, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(longitudes >= reduced_parms['telescope']['longitude'][0], longitudes <= reduced_parms['telescope']['longitude'][1]))
            if 'A_eff' in ival:
                effective_areas = NP.asarray([parm['telescope']['A_eff'] for parm in parms_list], dtype=NP.float)
                effective_areas[NP.equal(effective_areas, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(effective_areas >= reduced_parms['telescope']['A_eff'][0], effective_areas <= reduced_parms['telescope']['A_eff'][1]))
            if 'Tsys' in ival:
                system_temperatures = NP.asarray([parm['telescope']['Tsys'] for parm in parms_list], dtype=NP.float)
                system_temperatures[NP.equal(system_temperatures, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(system_temperatures >= reduced_parms['telescope']['Tsys'][0], system_temperatures <= reduced_parms['telescope']['Tsys'][1]))

        if ikey == 'array':
            if 'file' in ival:
                layout_files = NP.asarray([parm['array']['file'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.asarray([arrfile in reduced_parms['array']['file'] for arrfile in layout_files]))
            if 'layout' in ival:
                layouts = NP.asarray([parm['array']['layout'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.asarray([arrlayout in reduced_parms['array']['layout'] for arrlayout in layouts]))
            if 'minR' in ival:
                minRs = NP.asarray([parm['array']['minR'] for parm in parms_list], dtype=NP.float)
                minRs[NP.equal(minRs, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(minRs >= reduced_parms['array']['minR'][0], latitudes <= reduced_parms['array']['minR'][1]))
            if 'maxR' in ival:
                maxRs = NP.asarray([parm['array']['maxR'] for parm in parms_list], dtype=NP.float)
                maxRs[NP.equal(maxRs, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(maxRs >= reduced_parms['array']['maxR'][0], maxRs <= reduced_parms['array']['maxR'][1]))

        if ikey == 'baseline':
            if 'direction' in ival:
                directions = NP.asarray([parm['baseline']['direction'] for parm in parms_list])
                select_ind = NP.logical_and(select_ind, NP.asarray([direction in reduced_parms['baseline']['direction'] for direction in directions]))
            if 'min' in ival:
                mins = NP.asarray([parm['baseline']['min'] for parm in parms_list], dtype=NP.float)
                mins[NP.equal(mins, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(mins >= reduced_parms['baseline']['min'][0], mins <= reduced_parms['baseline']['min'][1]))
            if 'max' in ival:
                maxs = NP.asarray([parm['baseline']['max'] for parm in parms_list], dtype=NP.float)
                maxs[NP.equal(maxs, None)] = NP.nan
                select_ind = NP.logical_and(select_ind, NP.logical_and(maxs >= reduced_parms['baseline']['max'][0], maxs <= reduced_parms['baseline']['max'][1]))
                
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



