#!python

import yaml, argparse, ast, warnings
import numpy as NP
from astropy.io import ascii
from astropy.time import Time
import prisim

prisim_path = prisim.__path__[0]+'/'

def simparms_from_pyuvsim_to_prisim(pyuvsim_parms, prisim_parms):
    if not isinstance(pyuvsim_parms, dict):
        raise TypeError('Input pyuvsim_parms must be a dictionary')
    if not isinstance(prisim_parms, dict):
        raise TypeError('Input prisim_parms must be a dictionary')

    #I/O and directory structure

    pyuvsim_outpath = pyuvsim_parms['filing']['outdir']
    pyuvsim_outpath_hierarchy = pyuvsim_outpath.split('/')
    pyuvsim_outpath_hierarchy = [item for item in pyuvsim_outpath_hierarchy if item != '']
    prisim_parms['dirstruct']['rootdir'] = '/' + '/'.join(pyuvsim_outpath_hierarchy[:-1]) + '/'
    prisim_parms['dirstruct']['project'] = '/'.join(pyuvsim_outpath_hierarchy[-1:])
    prisim_parms['dirstruct']['simid'] = pyuvsim_parms['filing']['outfile_name']

    # Telescope parameters

    pyuvsim_telescope_parms = pyuvsim_parms['telescope']
    with open(pyuvsim_telescope_parms['telescope_config_name'], 'r') as pyuvsim_telescope_config_file:
        pyuvsim_telescope_config = yaml.safe_load(pyuvsim_telescope_config_file)
    telescope_location = ast.literal_eval(pyuvsim_telescope_config['telescope_location'])
    prisim_parms['telescope']['latitude'] = telescope_location[0]
    prisim_parms['telescope']['longitude'] = telescope_location[1]
    prisim_parms['telescope']['altitude'] = telescope_location[2]

    # Array parameters

    prisim_parms['array']['redundant'] = True
    prisim_parms['array']['layout'] = None
    prisim_parms['array']['file'] = pyuvsim_telescope_parms['array_layout']
    prisim_parms['array']['filepathtype'] = 'custom'
    prisim_parms['array']['parser']['data_start'] = 1
    prisim_parms['array']['parser']['label'] = 'Name'
    prisim_parms['array']['parser']['east'] = 'E'
    prisim_parms['array']['parser']['north'] = 'N'
    prisim_parms['array']['parser']['up'] = 'U'

    # Antenna power pattern parameters

    if pyuvsim_telescope_config['beam_paths'][0].lower() == 'uniform':
        prisim_parms['antenna']['shape'] = 'delta'
    if pyuvsim_telescope_config['beam_paths'][0].lower() == 'gaussian':
        prisim_parms['antenna']['shape'] = 'gaussian'
        prisim_parms['antenna']['size'] = pyuvsim_telescope_config['diameter']
    if pyuvsim_telescope_config['beam_paths'][0].lower() == 'airy':
        prisim_parms['antenna']['shape'] = 'dish'
        prisim_parms['antenna']['size'] = pyuvsim_telescope_config['diameter']

    if pyuvsim_telescope_config['beam_paths'][0].lower() in ['uniform', 'airy', 'gaussian']:
        prisim_parms['beam']['use_external'] = False
        prisim_parms['beam']['file'] = None
    else:
        prisim_parms['beam']['use_external'] = True
        prisim_parms['beam']['file'] = pyuvsim_telescope_config['beam_paths'][0]
        prisim_parms['beam']['filepathtype'] = 'custom'
        prisim_parms['beam']['filefmt'] = 'UVBeam'

    # Bandpass parameters

    prisim_parms['bandpass']['freq_resolution'] = pyuvsim_parms['freq']['channel_width']
    prisim_parms['bandpass']['nchan'] = pyuvsim_parms['freq']['Nfreqs']
    if prisim_parms['bandpass']['nchan'] == 1:
        warnings.warn('Single channel simulation is not supported currently in PRISim. Request at least two frequency channels.')
    pyuvsim_start_freq = pyuvsim_parms['freq']['start_freq']
    pyuvsim_freqs = pyuvsim_start_freq + prisim_parms['bandpass']['freq_resolution'] * NP.arange(prisim_parms['bandpass']['nchan'])
    prisim_parms['bandpass']['freq'] = pyuvsim_start_freq + 0.5 * prisim_parms['bandpass']['nchan'] * prisim_parms['bandpass']['freq_resolution']

    # Observing parameters

    prisim_parms['obsparm']['n_acc'] = pyuvsim_parms['time']['Ntimes']
    prisim_parms['obsparm']['t_acc'] = pyuvsim_parms['time']['integration_time']
    prisim_parms['obsparm']['obs_mode'] = 'drift'

    prisim_parms['pointing']['jd_init'] = pyuvsim_parms['time']['start_time']
    prisim_parms['obsparm']['obs_date'] = Time(prisim_parms['pointing']['jd_init'], scale='utc', format='jd').iso.split(' ')[0].replace('-', '/')

    prisim_parms['pointing']['lst_init'] = None
    prisim_parms['pointing']['drift_init']['alt'] = 90.0
    prisim_parms['pointing']['drift_init']['az'] = 270.0
    prisim_parms['pointing']['drift_init']['ha'] = None
    prisim_parms['pointing']['drift_init']['dec'] = None

    # Sky model

    prisim_parms['skyparm']['model'] = 'custom'
    prisim_parms['catalog']['filepathtype'] = 'custom'
    prisim_parms['catalog']['custom_file'] = pyuvsim_parms['sources']['catalog'].split('.txt')[0] + '_prisim.txt'
    pyuvsim_catalog = ascii.read(pyuvsim_parms['sources']['catalog'], comment='#', header_start=0, data_start=1)
    ra_colname = ''
    dec_colname = ''
    epoch = ''
    for colname in pyuvsim_catalog.colnames:
        if 'RA' in colname:
            ra_colname = colname
            ra_deg = pyuvsim_catalog[colname].data
            epoch = ra_colname.split('_')[1].split()[0][1:]
        if 'Dec' in colname:
            dec_colname = colname
            dec_deg = pyuvsim_catalog[colname].data
        if 'Flux' in colname:
            fint = pyuvsim_catalog[colname].data.astype(NP.float)
        if 'Frequency' in colname:
            ref_freq = pyuvsim_catalog[colname].data.astype(NP.float)
    
    spindex = NP.zeros(fint.size, dtype=NP.float)
    majax = NP.zeros(fint.size, dtype=NP.float)
    minax = NP.zeros(fint.size, dtype=NP.float)
    pa = NP.zeros(fint.size, dtype=NP.float)
    prisim_parms['skyparm']['epoch'] = epoch
    prisim_parms['skyparm']['flux_unit'] = 'Jy'
    prisim_parms['skyparm']['flux_min'] = None
    prisim_parms['skyparm']['flux_max'] = None
    prisim_parms['skyparm']['custom_reffreq'] = float(ref_freq[0]) / 1e9

    ascii.write([ra_deg, dec_deg, fint, spindex, majax, minax, pa], prisim_parms['catalog']['custom_file'], names=['RA', 'DEC', 'F_INT', 'SPINDEX', 'MAJAX', 'MINAX', 'PA'], delimiter='    ', format='fixed_width', formats={'RA': '%11.7f', 'DEC': '%12.7f', 'F_INT': '%10.4f', 'SPINDEX': '%8.5f', 'MAJAX': '%8.5f', 'MINAX': '%8.5f', 'PA': '%8.5f'}, bookend=False, overwrite=True)

    # Save format parameters

    prisim_parms['save_formats']['npz'] = False
    prisim_parms['save_formats']['uvfits'] = False
    prisim_parms['save_formats']['uvh5'] = True
   
    return prisim_parms

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to convert simulation parameter configurations from one simulator to another')

    ## Parse input arguments

    io_group = parser.add_argument_group('Input/Output parameters', 'Input/output specifications')
    io_group.add_argument('-i', '--infile', dest='infile', default=None, type=str, required=False, help='Full path to file specifying input parameters')
    io_group.add_argument('-o', '--outfile', dest='outfile', default=None, type=str, required=True, help='Full path to file specifying output parameters')
    io_group.add_argument('--from', dest='from', default=None, type=str, required=True, help='String specifying origin simulation configuration. Accepts "prisim", "pyuvsim"')
    io_group.add_argument('--to', dest='to', default=None, type=str, required=True, help='String specifying destination simulation configuration. Accepts "prisim", "pyuvsim"')

    args = vars(parser.parse_args())
    if args['from'].lower() not in ['prisim', 'pyuvsim']:
        raise ValueError('Originating simulation must be set to "prisim" or "pyuvsim"')
    if args['to'].lower() not in ['prisim', 'pyuvsim']:
        raise ValueError('Destination simulation must be set to "prisim" or "pyuvsim"')
    if args['from'].lower() == args['to'].lower():
        raise ValueError('Origin and destination simulation types must not be equal')

    if args['to'].lower() == 'prisim':
        prisim_template_file = prisim_path+'examples/simparms/defaultparms.yaml'
        with open(prisim_template_file, 'r') as prisim_parms_file:
            prisim_parms = yaml.safe_load(prisim_parms_file)
        with open(args['infile'], 'r') as pyuvsim_parms_file:
            pyuvsim_parms = yaml.safe_load(pyuvsim_parms_file)
        outparms = simparms_from_pyuvsim_to_prisim(pyuvsim_parms, prisim_parms)
    elif args['from'].lower() == 'prisim':
        with open(args['infile'], 'r') as prisim_parms_file:
            prisim_parms = yaml.safe_load(prisim_template_file)
        outparms = simparms_from_pyuvsim_to_prisim(prisim_parms)

    with open(args['outfile'], 'w') as outfile:
        yaml.dump(outparms, outfile, default_flow_style=False)

