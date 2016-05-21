#!python

import os, glob, sys
import yaml
import argparse
import numpy as NP
import numpy.ma as ma
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
            
# def searchPRISimDB(parms):

#     rootdir = parms['dirstruct']['rootdir']
#     project = parms['dirstruct']['project']
#     if project is None:
#         project_dir = ''
#     elif isinstance(project, str):
#         project_dir = project
    
#     if not os.path.isdir(rootdir):
#         raise OSError('Specified root directory does not exist')
    
#     if not os.path.isdir(rootdir+project_dir):
#         raise OSError('Specified project directory does not exist')
    
#     if project is None:
#         projects = os.listdir(rootdir)
#     else:
#         projects = [project_dir]
    
#     simparms_list = []
#     metadata_list = []
#     for proj in projects:
#         for simrun in os.listdir(rootdir+proj):
#             try:
#                 with open(rootdir+proj+'/'+simrun+'/metainfo/simparms.yaml', 'r') as parmsfile:
#                     simparms_list += [{rootdir+proj+'/'+simrun+'/': yaml.safe_load(parmsfile)}]
#                 with open(rootdir+proj+'/'+simrun+'/metainfo/meta.yaml', 'r') as metafile:
#                     metadata_list += [{rootdir+proj+'/'+simrun+'/': yaml.safe_load(metafile)}]
#             except IOError:
#                 pass
    
#     parms_list = []
#     for simind, parm in enumerate(simparms_list):
#         simrunkey = parm.keys()[0]
#         parm[simrunkey].update(metadata_list[simind][simrunkey])
#         parms_list += [parm[simrunkey]]
            
#     reduced_parms = NMO.recursive_find_notNone_in_dict(parms)
#     select_ind = NP.asarray([True] * len(parms_list))
#     for ikey, ival in reduced_parms.iteritems():
#         if ikey == 'telescope':
#             if 'id' in ival:
#                 telescope_ids = NP.asarray([parm[ikey]['id'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([tscope in reduced_parms[ikey]['id'] for tscope in telescope_ids], dtype=NP.bool))
#             if 'latitude' in ival:
#                 latitudes = NP.asarray([parm[ikey]['latitude'] for parm in parms_list], dtype=NP.float)
#                 latitudes[NP.equal(latitudes, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(latitudes >= reduced_parms[ikey]['latitude'][0], latitudes <= reduced_parms[ikey]['latitude'][1]))
#             if 'longitude' in ival:
#                 longitudes = NP.asarray([parm[ikey]['longitude'] for parm in parms_list], dtype=NP.float)
#                 longitudes[NP.equal(longitudes, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(longitudes >= reduced_parms[ikey]['longitude'][0], longitudes <= reduced_parms[ikey]['longitude'][1]))
#             if 'A_eff' in ival:
#                 effective_areas = NP.asarray([parm[ikey]['A_eff'] for parm in parms_list], dtype=NP.float)
#                 effective_areas[NP.equal(effective_areas, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(effective_areas >= reduced_parms[ikey]['A_eff'][0], effective_areas <= reduced_parms[ikey]['A_eff'][1]))
#             if 'Tsys' in ival:
#                 system_temperatures = NP.asarray([parm[ikey]['Tsys'] for parm in parms_list], dtype=NP.float)
#                 system_temperatures[NP.equal(system_temperatures, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(system_temperatures >= reduced_parms[ikey]['Tsys'][0], system_temperatures <= reduced_parms[ikey]['Tsys'][1]))
#             if 'Trx' in ival:
#                 rcvr_temperatures = NP.asarray([parm[ikey]['Trx'] for parm in parms_list], dtype=NP.float)
#                 rcvr_temperatures[NP.equal(rcvr_temperatures, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(rcvr_temperatures >= reduced_parms[ikey]['Trx'][0], rcvr_temperatures <= reduced_parms[ikey]['Trx'][1]))
#             if 'Tant_ref' in ival:
#                 ant_temperatures = NP.asarray([parm[ikey]['Tant_ref'] for parm in parms_list], dtype=NP.float)
#                 ant_temperatures[NP.equal(ant_temperatures, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(ant_temperatures >= reduced_parms[ikey]['Tant_ref'][0], ant_temperatures <= reduced_parms[ikey]['Tant_ref'][1]))
#             if 'Tant_freqref' in ival:
#                 ant_freqref = NP.asarray([parm[ikey]['Tant_freqref'] for parm in parms_list], dtype=NP.float)
#                 ant_freqref[NP.equal(ant_freqref, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(ant_freqref >= reduced_parms[ikey]['Tant_freqref'][0], ant_freqref <= reduced_parms[ikey]['Tant_freqref'][1]))

#         if ikey == 'array':
#             if 'file' in ival:
#                 layout_files = NP.asarray([parm[ikey]['file'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([arrfile in reduced_parms[ikey]['file'] for arrfile in layout_files], dtype=NP.bool))
#             if 'layout' in ival:
#                 layouts = NP.asarray([parm[ikey]['layout'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([arrlayout in reduced_parms[ikey]['layout'] for arrlayout in layouts], dtype=NP.bool))
#             if 'minR' in ival:
#                 minRs = NP.asarray([parm[ikey]['minR'] for parm in parms_list], dtype=NP.float)
#                 minRs[NP.equal(minRs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(minRs >= reduced_parms[ikey]['minR'][0], latitudes <= reduced_parms[ikey]['minR'][1]))
#             if 'maxR' in ival:
#                 maxRs = NP.asarray([parm[ikey]['maxR'] for parm in parms_list], dtype=NP.float)
#                 maxRs[NP.equal(maxRs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(maxRs >= reduced_parms[ikey]['maxR'][0], maxRs <= reduced_parms[ikey]['maxR'][1]))

#         if ikey == 'baseline':
#             if 'direction' in ival:
#                 directions = NP.asarray([parm[ikey]['direction'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([direction in reduced_parms[ikey]['direction'] for direction in directions], dtype=NP.bool))
#             if 'min' in ival:
#                 mins = NP.asarray([parm[ikey]['min'] for parm in parms_list], dtype=NP.float)
#                 mins[NP.equal(mins, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(mins >= reduced_parms[ikey]['min'][0], mins <= reduced_parms[ikey]['min'][1]))
#             if 'max' in ival:
#                 maxs = NP.asarray([parm[ikey]['max'] for parm in parms_list], dtype=NP.float)
#                 maxs[NP.equal(maxs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(maxs >= reduced_parms[ikey]['max'][0], maxs <= reduced_parms[ikey]['max'][1]))

#         if ikey == 'antenna':
#             if 'shape' in ival:
#                 ant_shapes = NP.asarray([parm[ikey]['shape'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([antshape in reduced_parms[ikey]['shape'] for antshape in ant_shapes], dtype=NP.bool))
#             if 'size' in ival:
#                 ant_sizes = NP.asarray([parm[ikey]['size'] for parm in parms_list], dtype=NP.float)
#                 ant_sizes[NP.equal(ant_sizes, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(ant_sizes >= reduced_parms[ikey]['size'][0], ant_sizes <= reduced_parms[ikey]['size'][1]))
#             if 'ocoords' in ival:
#                 ant_ocoords = NP.asarray([parm[ikey]['ocoords'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([ant_ocoord in reduced_parms[ikey]['ocoords'] for ant_ocoord in ant_ocoords], dtype=NP.bool))
#             if 'phased_array' in ival:
#                 is_phased_arrays = NP.asarray([parm[ikey]['phased_array'] for parm in parms_list], dtype=NP.bool)
#                 select_ind = NP.logical_and(select_ind, NP.equal(is_phased_arrays, reduced_parms[ikey]['phased_array']))
#             if 'ground_plane' in ival:
#                 ground_planes = NP.asarray([parm[ikey]['ground_plane'] for parm in parms_list], dtype=NP.float)
#                 ground_planes[NP.equal(ground_planes, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(ground_planes >= reduced_parms[ikey]['ground_plane'][0], ground_planes <= reduced_parms[ikey]['ground_plane'][1]))

#         if ikey == 'phasedarray':
#             if 'file' in ival:
#                 phsdarray_files = NP.asarray([parm[ikey]['file'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([phsdarray_file in reduced_parms[ikey]['file'] for phsdarray_file in phsdarray_files], dtype=NP.bool))
#             if 'delayerr' in ival:
#                 delayerrs = NP.asarray([parm[ikey]['delayerr'] for parm in parms_list], dtype=NP.float)
#                 delayerrs[NP.equal(delayerrs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(delayerrs >= reduced_parms[ikey]['delayerr'][0], delayerrs <= reduced_parms[ikey]['delayerr'][1]))
#             if 'gainerr' in ival:
#                 gainerrs = NP.asarray([parm[ikey]['gainerr'] for parm in parms_list], dtype=NP.float)
#                 gainerrs[NP.equal(gainerrs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(gainerrs >= reduced_parms[ikey]['gainerr'][0], gainerrs <= reduced_parms[ikey]['gainerr'][1]))
#             if 'nrand' in ival:
#                 nrands = NP.asarray([parm[ikey]['nrand'] for parm in parms_list], dtype=NP.float)
#                 nrands[NP.equal(nrands, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(nrands >= reduced_parms[ikey]['nrand'][0], nrands <= reduced_parms[ikey]['nrand'][1]))
                
#         if ikey == 'beam':
#             if 'use_external' in ival:
#                 use_external_beams = NP.asarray([parm[ikey]['use_external'] for parm in parms_list], dtype=NP.bool)
#                 select_ind = NP.logical_and(select_ind, NP.equal(use_external_beams, reduced_parms[ikey]['use_external']))
#             if 'file' in ival:
#                 extbeam_files = NP.asarray([parm[ikey]['file'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([extbeam_file in reduced_parms[ikey]['file'] for extbeam_file in extbeam_files], dtype=NP.bool))
#             if 'identifier' in ival:
#                 beam_ids = NP.asarray([parm[ikey]['identifier'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([beamid in reduced_parms[ikey]['identifier'] for beamid in beam_ids], dtype=NP.bool))
#             if 'pol' in ival:
#                 beam_pols = NP.asarray([parm[ikey]['pol'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([beampol in reduced_parms[ikey]['pol'] for beampol in beam_pols], dtype=NP.bool))
#             if 'chromatic' in ival:
#                 chrmbeams = NP.asarray([parm[ikey]['chromatic'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.equal(chrmbeams, reduced_parms[ikey]['chromatic']))
#             if 'select_freq' in ival:
#                 achrmfreqs = NP.asarray([parm[ikey]['select_freq'] for parm in parms_list], dtype=NP.float)
#                 achrmfreqs[NP.equal(achrmfreqs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(achrmfreqs >= reduced_parms[ikey]['select_freq'][0], achrmfreqs <= reduced_parms[ikey]['select_freq'][1]))
#             if 'spec_interp' in ival:
#                 beam_spec_interps = NP.asarray([parm[ikey]['spec_interp'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([bmspecinterp in reduced_parms[ikey]['spec_interp'] for bmspecinterp in beam_spec_interps], dtype=NP.bool))

#         if ikey == 'bandpass':
#             if 'freq' in ival:
#                 freqs = NP.asarray([parm[ikey]['freq'] for parm in parms_list], dtype=NP.float)
#                 freqs[NP.equal(freqs, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(freqs >= reduced_parms[ikey]['freq'][0], freqs <= reduced_parms[ikey]['freq'][1]))
#             if 'freq_resolution' in ival:
#                 freq_resolutions = NP.asarray([parm[ikey]['freq_resolution'] for parm in parms_list], dtype=NP.float)
#                 freq_resolutions[NP.equal(freq_resolutions, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(freq_resolutions >= reduced_parms[ikey]['freq_resolution'][0], freq_resolutions <= reduced_parms[ikey]['freq_resolution'][1]))
#             if 'nchan' in ival:
#                 nchans = NP.asarray([parm[ikey]['nchan'] for parm in parms_list], dtype=NP.float)
#                 nchans[NP.equal(nchans, None)] = NP.nan
#                 select_ind = NP.logical_and(select_ind, NP.logical_and(nchans >= reduced_parms[ikey]['nchan'][0], nchans <= reduced_parms[ikey]['nchan'][1]))
#             if 'pfb_method' in ival:
#                 pfbmethods = NP.asarray([parm[ikey]['pfb_method'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([pfbmethod in reduced_parms[ikey]['pfb_method'] for pfbmethod in pfbmethods], dtype=NP.bool))
#             if 'pfb_file' in ival:
#                 pfb_files = NP.asarray([parm[ikey]['pfb_file'] for parm in parms_list])
#                 select_ind = NP.logical_and(select_ind, NP.asarray([pfb_file in reduced_parms[ikey]['pfb_file'] for pfb_file in pfb_files], dtype=NP.bool))
                
#     select_ind, = NP.where(select_ind)
#     outkeys = [metadata_list[ind].keys()[0] for ind in select_ind]
#     for okey in outkeys:
#         print okey

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to search metadata of PRISim simulations')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=prisim_path+'examples/dbparms/defaultdbparms.yaml', type=file, required=False, help='File specifying input database search parameters')
    
    args = vars(parser.parse_args())
    
    with args['infile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    # searchPRISimDB(parms)
    selectsims = grepPRISim(parms)
    print '\nThe following simulation runs were found to contain the searched parameters:\n'
    for simrun in selectsims:
        print '\t'+simrun
    print '\n'
