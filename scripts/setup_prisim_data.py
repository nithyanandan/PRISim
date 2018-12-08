#!python

import os
import argparse
import yaml
import gdown
import tarfile
import prisim

prisim_path = prisim.__path__[0]+'/'
tarfilename = 'prisim_data.tar.gz'
# url_default = 'https://www.dropbox.com/s/7y9go1bzjfa0rkv/prisim_data.tar.gz?dl=1'

def download(url=None, outfile=None, verbose=True):
    if url is not None:
        if not isinstance(url, str):
            raise TypeError('Input url must be a string')
        if outfile is None:
            outfile = prisim_path+tarfilename
        elif not isinstance(outfile, str):
            raise TypeError('outfile must be a string')
        gdown.download(url, outfile, quiet=(not verbose))

# def download_old(url=None, outfile=None, verbose=True):
#     if url is None:
#         url = url_default
#     elif not isinstance(url, str):
#         raise TypeError('Input url must be a string')
#     if outfile is None:
#         outfile = prisim_path+tarfilename
#     elif not isinstance(outfile, str):
#         raise TypeError('outfile must be a string')

#     if verbose:
#         print('Downloading PRISim package data from {0} ...'.format(url))

#     r = requests.get(url)
#     with open(outfile, 'wb') as fhandle:
#         fhandle.write(r.content)

#     if verbose:
#         print('Downloaded PRISim package data into {0}'.format(outfile))

def extract(infile=None, outdir=None, verbose=True):
    if infile is None:
        infile = prisim_path + tarfilename
    elif not isinstance(infile, str):
        raise TypeError('infile must be a string')
    if outdir is None:
        outdir = prisim_path
    elif not isinstance(outdir, str):
        raise TypeError('outdir must be a string')
    
    if verbose:
        print('Extracting PRISim package data from {0} ...'.format(infile))

    with tarfile.open(infile, 'r:gz') as tar:
        tar.extractall(outdir)

    if verbose:
        print('Extracted PRISim package data into {0}'.format(outdir))

def cleanup(infile=None, verbose=True):
    if infile is None:
        infile = prisim_path + tarfilename
    elif not isinstance(infile, str):
        raise TypeError('infile must be a string')

    if verbose:
        print('Cleaning up intermediate file {0} of PRISim package data ...'.format(infile))

    if os.path.isfile(infile):
        os.remove(infile)
        
    if verbose:
        print('Cleaned up PRISim package data.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to write PRISim output visibilities in UVFITS format')

    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', type=file, required=False, default=prisim_path+'examples/ioparms/data_setup_parms.yaml', help='File specifying PRISim data setup parameters')

    args = vars(parser.parse_args())
    with args['parmsfile'] as parms_file:
        parms = yaml.safe_load(parms_file)

    action_types = ['download', 'extract', 'cleanup']
    for action_type in action_types:
        if action_type not in parms:
            parms[action_type] = {}
            parms[action_type]['action'] = False

    for action_type in action_types:
        if parms[action_type]['action']:
            if action_type == 'download':
                keys = ['url', 'fid', 'fname']
            elif action_type == 'extract':
                keys = ['fname', 'dir']
            else:
                keys = ['fname']
            for key in keys:
                if key not in parms[action_type]:
                    parms[action_type][key] = None

            if action_type == 'download':
                download(url=parms[action_type]['url']+parms[action_type]['fid'], outfile=parms[action_type]['fname'], verbose=parms['verbose'])
                # gdown.download()
            elif action_type == 'extract':
                extract(infile=parms[action_type]['fname'], outdir=parms[action_type]['dir'], verbose=parms['verbose'])
            else:
                cleanup(infile=parms[action_type]['fname'], verbose=parms['verbose'])
    if parms['verbose']:
        print('PRISim package data successfully set up.')
