#!python

from __future__ import print_function
import glob
import itertools
import yaml
import argparse
import numpy as NP
import prisim

prisim_path = prisim.__path__[0]+'/'

def lsPRISim(args):
    project_dir = args['project']
    simid = args['simid']
    folder_separator = ''
    if not project_dir.endswith('/'):
        folder_separator = '/'
    simdir_pattern = project_dir + folder_separator + simid
    temp_simdirs = glob.glob(simdir_pattern)
    simdirs = [temp_simdir for temp_simdir in temp_simdirs if not temp_simdir.endswith(('.', '..'))]
    
    simparms_list = []
    for simdir in simdirs:
        try:
            with open(simdir+'/metainfo/simparms.yaml', 'r') as parmsfile:
                simparms_list += [{simdir+'/': yaml.safe_load(parmsfile)}]
        except IOError:
            pass

    parmsDB = {}
    for parmind, parm in enumerate(simparms_list):
        for ikey, ival in parm.values()[0].iteritems():
            if isinstance(ival, dict):
                for subkey in ival.iterkeys():
                    key = (ikey, subkey)
                    if key in parmsDB:
                        parmsDB[key] += [parm.values()[0][ikey][subkey]]
                    else:
                        parmsDB[key] = [parm.values()[0][ikey][subkey]]
                    
    parmsDBselect = {}
    nuniqDBselect = {}
    for key in parmsDB:
        vals = sorted(parmsDB[key])
        uniqvals = [val for val,_ in itertools.groupby(vals)]
        if len(uniqvals) > 1:
            parmsDBselect[key] = parmsDB[key]
            nuniqDBselect[key] = len(uniqvals)

    linestr = '\n'
    if args['format'] == 'csv':
        delimiter = ','
    else:
        delimiter = '\t'
    if args['change']:
        if parmsDBselect:
            keys = sorted(parmsDBselect.keys())
            linestr = 'PRISim-ID'
            for key in keys:
                linestr += delimiter+key[0]+':'+key[1]
            linestr += '\n'
            for parmind, parm in enumerate(simparms_list):
                linestr += '\n'+parm.keys()[0]
                for key in parmsDBselect:
                    linestr += delimiter+str(parm.values()[0][key[0]][key[1]])
            linestr += '\n\nNumber of unique values'
            for key in parmsDBselect:
                linestr += delimiter+'{0:0d}/{1:0d}'.format(nuniqDBselect[key], len(simparms_list))
    else:
        if parmsDB:
            keys = sorted(parmsDB.keys())
            linestr = 'PRISim-ID'
            for key in keys:
                linesstr += delimiter+key[0]+':'+key[1]
            linestr += '\n'
            for parmind, parm in enumerate(simparms_list):
                linestr += '\n'+parm.keys()[0]
                for key in parmsDB:
                    linestr += delimiter+str(parm.values()[0][key[0]][key[1]])
    return linestr+'\n'
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to list metadata of PRISim simulations')

    dir_group = parser.add_argument_group('Search targets', 'Target data directories for search')
    dir_group.add_argument('-p', '--project', dest='project', required=True, type=str, help='Project directory to search simulation parameters in')
    dir_group.add_argument('-s', '--simid', dest='simid', required=False, type=str, default='*', help='Simulation ID filter')

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('-a', '--all', dest='all', default=True, action='store_true')
    filter_group.add_argument('-c', '--change', dest='change', default=False, action='store_true')

    output_group = parser.add_argument_group('Output specifications', 'Output specifications')
    output_group.add_argument('-f', '--format', dest='format', default='tsv', choices=['csv', 'tsv'], type=str, required=False, help='Output format (tab/comma separated)')
    output_group.add_argument('-o', '--output', dest='output', type=str, required=False, help='Output file path')

    args = vars(parser.parse_args())
    linestr = lsPRISim(args)

    if args['output'] is not None:
        try:
            with open(args['output'], 'w+') as outfile:
                outfile.write(linestr)
        except IOError:
            print(linestr)
            raise IOError('Specified output file/folder invalid')
    else:
        print(linestr)
