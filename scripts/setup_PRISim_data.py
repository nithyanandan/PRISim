#!python

import os
import argparse
import requests
import prisim

prisim_path = prisim.__path__[0]+'/'
tarfilename = 'data.tar.gz'
url_default = 'https://www.dropbox.com/s/rhdtp3opsio50fa/prisim_data.tar.gz?dl=1'

def download(url=None, outfile=None):
    if url is None:
        url = url_default
    elif not isinstance(url, str):
        raise TypeError('Input url must be a string')
    if outfile is None:
        outfile = tarfile
    elif not isinstance(outfile, str):
        raise TypeError('outfile must be a string')

    r = requests.get(url)
    with open(outfile, 'wb') as fhandle:
        fhandle.write(r.content)

def extract(infile=None, outdir=None):
    if infile is None:
        infile = prisim_path + tarfilename
    elif not isinstance(infile, str):
        raise TypeError('infile must be a string')
    if outdir is None:
        outdir = prisim_path
    elif not isinstance(outdir, str):
        raise TypeError('outdir must be a string')
    
    with tarfile.open(infile, 'r:gz') as tar:
        tar.extractall(outdir)

def cleanup(infile=None):
    if infile is None:
        infile = prisim_path + tarfilename
    elif not isinstance(infile, str):
        raise TypeError('infile must be a string')
    if os.path.isfile(infile):
        os.remove(infile)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to write PRISim output visibilities in UVFITS format')

    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', type=file, required=True, default=prisim_path+'examples/ioparms/data_setup_parms.yaml', help='File specifying PRISim data setup parameters')

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
                keys = ['url', 'fname']
            elif action_type == 'extract':
                keys = ['fname', 'dir']
            else:
                keys = ['fname']
            for key in keys:
                if key not in parms[action_type]:
                    parms[action_type][key] = None

            if action_type == 'download':
                download(url=parms[action_type]['url'], outfile=parms[action_type]['fname'])
            elif action_type == 'extract':
                extract(infile=parms[action_type]['fname'], outdir=parms[action_type]['dir'])
            else:
                cleanup(infile=parms[action_type]['fname'])
