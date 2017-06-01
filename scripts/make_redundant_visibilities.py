#!python

import yaml
import argparse
import numpy as NP
from prisim import interferometry as RI
import ipdb as PDB

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to duplicate redundant baseline measurements')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-s', '--simfile', dest='simfile', type=str, required=True, help='HDF5 file from PRISim simulation')
    input_group.add_argument('-p', '--parmsfile', dest='parmsfile', default=None, type=str, required=False, help='File specifying simulation parameters')

    output_group = parser.add_argument_group('Output parameters', 'Output specifications')
    output_group.add_argument('-o', '--outfile', dest='outfile', default=None, type=str, required=True, help='Output File with redundant measurements')
    output_group.add_argument('--outfmt', dest='outfmt', default=['hdf5'], type=str, required=True, nargs='*', choices=['HDF5', 'hdf5', 'UVFITS', 'uvfits'], help='Output file format')

    misc_group = parser.add_argument_group('Misc parameters', 'Misc specifications')
    misc_group.add_argument('-w', '--wait', dest='wait', action='store_true', help='Wait after run')
    
    args = vars(parser.parse_args())

    simobj = RI.InterferometerArray(None, None, None, init_file=args['simfile'])

    if args['parmsfile'] is not None:
        parmsfile = args['parmsfile']
    else:
        parmsfile = simvis.simparms_file
        
    with open(parmsfile, 'r') as pfile:
        parms = yaml.safe_load(pfile)

    outfile = args['outfile']
    wait_after_run = args['wait']

    blinfo = RI.getBaselineInfo(parms)
    bl = blinfo['bl']
    blgroups = blinfo['groups']
    bl_length = NP.sqrt(NP.sum(bl**2, axis=1))

    # array_is_redundant = parms['array']['redundant']
    # if not array_is_redundant:
    #     raise ValueError('Simulations assumed array was non-redundant to begin with.')

    # fg_str = parms['fgparm']['model']
    # use_HI_monopole = False
    # if fg_str == 'HI_monopole':
    #     use_HI_monopole = True
    # antenna_file = parms['array']['file']
    # array_layout = parms['array']['layout']
    # minR = parms['array']['minR']
    # maxR = parms['array']['maxR']
    # antpos_rms_tgtplane = parms['array']['rms_tgtplane']
    # antpos_rms_elevation = parms['array']['rms_elevation']
    # antpos_rms_seed = parms['array']['seed']
    # if antpos_rms_seed is None:
    #     antpos_rms_seed = NP.random.randint(1, high=100000)
    # elif isinstance(antpos_rms_seed, (int,float)):
    #     antpos_rms_seed = int(NP.abs(antpos_rms_seed))
    # else:
    #     raise ValueError('Random number seed must be a positive integer')
    # minbl = parms['baseline']['min']
    # maxbl = parms['baseline']['max']
    # bldirection = parms['baseline']['direction']

    # if (antenna_file is None) and (array_layout is None):
    #     raise ValueError('One of antenna array file or layout must be specified')
    # if (antenna_file is not None) and (array_layout is not None):
    #     raise ValueError('Only one of antenna array file or layout must be specified')
    
    # if antenna_file is not None:
    #     if not isinstance(antenna_file, str):
    #         raise TypeError('Filename containing antenna array elements must be a string')
    #     if parms['array']['filepathtype'] == 'default':
    #         antenna_file = prisim_path+'data/array_layouts/'+antenna_file
        
    #     antfile_parser = parms['array']['parser']
    #     if 'comment' in antfile_parser:
    #         comment = antfile_parser['comment']
    #         if comment is None:
    #             comment = '#'
    #         elif not isinstance(comment, str):
    #             raise TypeError('Comment expression must be a string')
    #     else:
    #         comment = '#'
    #     if 'delimiter' in antfile_parser:
    #         delimiter = antfile_parser['delimiter']
    #         if delimiter is not None:
    #             if not isinstance(delimiter, str):
    #                 raise TypeError('Delimiter expression must be a string')
    #         else:
    #             delimiter = ' '
    #     else:
    #         delimiter = ' '
    
    #     if 'data_start' in antfile_parser:
    #         data_start = antfile_parser['data_start']
    #         if not isinstance(data_start, int):
    #             raise TypeError('data_start parameter must be an integer')
    #     else:
    #         raise KeyError('data_start parameter not provided')
    #     if 'data_end' in antfile_parser:
    #         data_end = antfile_parser['data_end']
    #         if data_end is not None:
    #             if not isinstance(data_end, int):
    #                 raise TypeError('data_end parameter must be an integer')
    #     else:
    #         data_end = None
    #     if 'header_start' in antfile_parser:
    #         header_start = antfile_parser['header_start']
    #         if not isinstance(header_start, int):
    #             raise TypeError('header_start parameter must be an integer')
    #     else:
    #         raise KeyError('header_start parameter not provided')
    
    #     if 'label' not in antfile_parser:
    #         antfile_parser['label'] = None
    #     elif antfile_parser['label'] is not None:
    #         antfile_parser['label'] = str(antfile_parser['label'])
    
    #     if 'east' not in antfile_parser:
    #         raise KeyError('Keyword for "east" coordinates not provided')
    #     else:
    #         if not isinstance(antfile_parser['east'], str):
    #             raise TypeError('Keyword for "east" coordinates must be a string')
    #     if 'north' not in antfile_parser:
    #         raise KeyError('Keyword for "north" coordinates not provided')
    #     else:
    #         if not isinstance(antfile_parser['north'], str):
    #             raise TypeError('Keyword for "north" coordinates must be a string')
    #     if 'up' not in antfile_parser:
    #         raise KeyError('Keyword for "up" coordinates not provided')
    #     else:
    #         if not isinstance(antfile_parser['up'], str):
    #             raise TypeError('Keyword for "up" coordinates must be a string')
    
    #     try:
    #         ant_info = ascii.read(antenna_file, comment=comment, delimiter=delimiter, header_start=header_start, data_start=data_start, data_end=data_end, guess=False)
    #     except IOError:
    #         raise IOError('Could not open file containing antenna locations.')
    
    #     if (antfile_parser['east'] not in ant_info.colnames) or (antfile_parser['north'] not in ant_info.colnames) or (antfile_parser['up'] not in ant_info.colnames):
    #         raise KeyError('One of east, north, up coordinates incompatible with the table in antenna_file')
    
    #     if antfile_parser['label'] is not None:
    #         ant_label = ant_info[antfile_parser['label']].data.astype('str')
    #     else:
    #         ant_label = NP.arange(len(ant_info)).astype('str')
    
    #     east = ant_info[antfile_parser['east']].data
    #     north = ant_info[antfile_parser['north']].data
    #     elev = ant_info[antfile_parser['up']].data
    
    #     if (east.dtype != NP.float) or (north.dtype != NP.float) or (elev.dtype != NP.float):
    #         raise TypeError('Antenna locations must be of floating point type')
    
    #     ant_locs = NP.hstack((east.reshape(-1,1), north.reshape(-1,1), elev.reshape(-1,1)))
    # else:
    #     if array_layout not in ['MWA-128T', 'HERA-7', 'HERA-19', 'HERA-37', 'HERA-61', 'HERA-91', 'HERA-127', 'HERA-169', 'HERA-217', 'HERA-271', 'HERA-331', 'PAPER-64', 'PAPER-112', 'HIRAX-1024', 'CIRC']:
    #         raise ValueError('Invalid array layout specified')
    
    #     if array_layout == 'MWA-128T':
    #         ant_info = NP.loadtxt(prisim_path+'data/array_layouts/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
    #         ant_label = ant_info[:,0].astype(int).astype(str)
    #         ant_locs = ant_info[:,1:]
    #     elif array_layout == 'HERA-7':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=7)
    #     elif array_layout == 'HERA-19':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=19)
    #     elif array_layout == 'HERA-37':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=37)
    #     elif array_layout == 'HERA-61':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=61)
    #     elif array_layout == 'HERA-91':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=91)
    #     elif array_layout == 'HERA-127':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=127)
    #     elif array_layout == 'HERA-169':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=169)
    #     elif array_layout == 'HERA-217':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=217)
    #     elif array_layout == 'HERA-271':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=271)
    #     elif array_layout == 'HERA-331':
    #         ant_locs, ant_label = RI.hexagon_generator(14.6, n_total=331)
    #     elif array_layout == 'PAPER-64':
    #         ant_locs, ant_label = RI.rectangle_generator([30.0, 4.0], [8, 8])
    #     elif array_layout == 'PAPER-112':
    #         ant_locs, ant_label = RI.rectangle_generator([15.0, 4.0], [16, 7])
    #     elif array_layout == 'HIRAX-1024':
    #         ant_locs, ant_label = RI.rectangle_generator(7.0, n_side=32)
    #     elif array_layout == 'CIRC':
    #         ant_locs, ant_label = RI.circular_antenna_array(element_size, minR, maxR=maxR)
    #     ant_label = NP.asarray(ant_label)
    # if ant_locs.shape[1] == 2:
    #     ant_locs = NP.hstack((ant_locs, NP.zeros(ant_label.size).reshape(-1,1)))
    # antpos_rstate = NP.random.RandomState(antpos_rms_seed)
    # deast = antpos_rms_tgtplane/NP.sqrt(2.0) * antpos_rstate.randn(ant_label.size)
    # dnorth = antpos_rms_tgtplane/NP.sqrt(2.0) * antpos_rstate.randn(ant_label.size)
    # dup = antpos_rms_elevation * antpos_rstate.randn(ant_label.size)
    # denu = NP.hstack((deast.reshape(-1,1), dnorth.reshape(-1,1), dup.reshape(-1,1)))
    # ant_locs = ant_locs + denu
    # ant_locs_orig = NP.copy(ant_locs)
    # ant_label_orig = NP.copy(ant_label)
    # ant_id = NP.arange(ant_label.size, dtype=int)
    # ant_id_orig = NP.copy(ant_id)
    # layout_info = {'positions': ant_locs_orig, 'labels': ant_label_orig, 'ids': ant_id_orig, 'coords': 'ENU'}

    # bl_orig, bl_label_orig, bl_id_orig = RI.baseline_generator(ant_locs_orig, ant_label=ant_label_orig, ant_id=ant_id_orig, auto=False, conjugate=False)
    
    # blo = NP.angle(bl_orig[:,0] + 1j * bl_orig[:,1], deg=True)
    # neg_blo_ind = (blo < -67.5) | (blo > 112.5)
    # # neg_blo_ind = NP.logical_or(blo < -0.5*180.0/n_bins_baseline_orientation, blo > 180.0 - 0.5*180.0/n_bins_baseline_orientation)
    # bl_orig[neg_blo_ind,:] = -1.0 * bl_orig[neg_blo_ind,:]
    # blo = NP.angle(bl_orig[:,0] + 1j * bl_orig[:,1], deg=True)
    # maxlen = max(max(len(albl[0]), len(albl[1])) for albl in bl_label_orig)
    # bl_label_orig = [tuple(reversed(bl_label_orig[i])) if neg_blo_ind[i] else bl_label_orig[i] for i in xrange(bl_label_orig.size)]
    # bl_label_orig = NP.asarray(bl_label_orig, dtype=[('A2', '|S{0:0d}'.format(maxlen)), ('A1', '|S{0:0d}'.format(maxlen))])
    # bl_id_orig = [tuple(reversed(bl_id_orig[i])) if neg_blo_ind[i] else bl_id_orig[i] for i in xrange(bl_id_orig.size)]
    # bl_id_orig = NP.asarray(bl_id_orig, dtype=[('A2', int), ('A1', int)])
    # bl_length_orig = NP.sqrt(NP.sum(bl_orig**2, axis=1))
    # sortind_orig = NP.argsort(bl_length_orig, kind='mergesort')
    # bl_orig = bl_orig[sortind_orig,:]
    # blo = blo[sortind_orig]
    # bl_label_orig = bl_label_orig[sortind_orig]
    # bl_id_orig = bl_id_orig[sortind_orig]
    # bl_length_orig = bl_length_orig[sortind_orig]
    
    # bl = NP.copy(bl_orig)
    # bl_label = NP.copy(bl_label_orig)
    # bl_id = NP.copy(bl_id_orig)
    # bl_orientation = NP.copy(blo)
    # if array_is_redundant:
    #     bl, select_bl_ind, bl_count, allinds = RI.uniq_baselines(bl)
    #     bl_label = bl_label[select_bl_ind]
    #     bl_id = bl_id[select_bl_ind]
    #     bl_orientation = bl_orientation[select_bl_ind]
    # bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
    # # bl_orientation = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
    # sortind = NP.argsort(bl_length, kind='mergesort')
    # bl = bl[sortind,:]
    # bl_label = bl_label[sortind]
    # bl_id = bl_id[sortind]
    # bl_length = bl_length[sortind]
    # bl_orientation = bl_orientation[sortind]
    # if array_is_redundant:
    #     bl_count = bl_count[sortind]
    #     select_bl_ind = select_bl_ind[sortind]
    #     allinds = [allinds[i] for i in sortind]
    
    # if minbl is None:
    #     minbl = 0.0
    # elif not isinstance(minbl, (int,float)):
    #     raise TypeError('Minimum baseline length must be a scalar')
    # elif minbl < 0.0:
    #     minbl = 0.0
    
    # if maxbl is None:
    #     maxbl = bl_length.max()
    # elif not isinstance(maxbl, (int,float)):
    #     raise TypeError('Maximum baseline length must be a scalar')
    # elif maxbl < minbl:
    #     maxbl = bl_length.max()
    
    # min_blo = -67.5
    # max_blo = 112.5
    # subselect_bl_ind = NP.zeros(bl_length.size, dtype=NP.bool)
    
    # if bldirection is not None:
    #     if isinstance(bldirection, str):
    #         if bldirection not in ['SE', 'E', 'NE', 'N']:
    #             raise ValueError('Invalid baseline direction criterion specified')
    #         else:
    #             bldirection = [bldirection]
    #     if isinstance(bldirection, list):
    #         for direction in bldirection:
    #             if direction in ['SE', 'E', 'NE', 'N']:
    #                 if direction == 'SE':
    #                     oind = (bl_orientation >= -67.5) & (bl_orientation < -22.5)
    #                     subselect_bl_ind[oind] = True
    #                 elif direction == 'E':
    #                     oind = (bl_orientation >= -22.5) & (bl_orientation < 22.5)
    #                     subselect_bl_ind[oind] = True
    #                 elif direction == 'NE':
    #                     oind = (bl_orientation >= 22.5) & (bl_orientation < 67.5)
    #                     subselect_bl_ind[oind] = True
    #                 else:
    #                     oind = (bl_orientation >= 67.5) & (bl_orientation < 112.5)
    #                     subselect_bl_ind[oind] = True
    #     else:
    #         raise TypeError('Baseline direction criterion must specified as string or list of strings')
    # else:
    #     subselect_bl_ind = NP.ones(bl_length.size, dtype=NP.bool)
    
    # subselect_bl_ind = subselect_bl_ind & (bl_length >= minbl) & (bl_length <= maxbl)
    # bl_label = bl_label[subselect_bl_ind]
    # bl_id = bl_id[subselect_bl_ind]
    # bl = bl[subselect_bl_ind,:]
    # bl_length = bl_length[subselect_bl_ind]
    # bl_orientation = bl_orientation[subselect_bl_ind]
    # if array_is_redundant:
    #     bl_count = bl_count[subselect_bl_ind]
    #     select_bl_ind = select_bl_ind[subselect_bl_ind]
    #     allinds = [allinds[i] for i in range(subselect_bl_ind.size) if subselect_bl_ind[i]]
    
    # if use_HI_monopole:
    #     bllstr = map(str, bl_length)
    #     uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
    #     count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
    #     count_uniq_bll = NP.asarray(count_uniq_bll)
    
    #     bl = bl[ind_uniq_bll,:]
    #     bl_label = bl_label[ind_uniq_bll]
    #     bl_id = bl_id[ind_uniq_bll]
    #     bl_orientation = bl_orientation[ind_uniq_bll]
    #     bl_length = bl_length[ind_uniq_bll]
    #     if array_is_redundant:
    #         bl_count = bl_count[ind_uniq_bll]
    #         select_bl_ind = select_bl_ind[ind_uniq_bll]
    #         allinds = [allinds[i] for i in ind_uniq_bll]
    
    #     sortind = NP.argsort(bl_length, kind='mergesort')
    #     bl = bl[sortind,:]
    #     bl_label = bl_label[sortind]
    #     bl_id = bl_id[sortind]
    #     bl_length = bl_length[sortind]
    #     bl_orientation = bl_orientation[sortind]
    #     count_uniq_bll = count_uniq_bll[sortind]
    #     if array_is_redundant:
    #         bl_count = bl_count[sortind]
    #         select_bl_ind = select_bl_ind[sortind]
    #         allinds = [allinds[i] for i in sortind]
    
    # total_baselines = bl_length.size
    # if array_is_redundant:
    #     blgroups = {}
    #     blgroups_reversemap = {}
    #     for labelind, label in enumerate(bl_label_orig[select_bl_ind]):
    #         if bl_count[labelind] > 0:
    #             blgroups[tuple(label)] = bl_label_orig[NP.asarray(allinds[labelind])]
    #             for lbl in bl_label_orig[NP.asarray(allinds[labelind])]:
    #                 blgroups_reversemap[tuple(lbl)] = tuple(label)
    
    # try:
    #     labels = bl_label.tolist()
    # except NameError:
    #     labels = []
    #     labels += [label_prefix+'{0:0d}'.format(i+1) for i in xrange(bl.shape[0])]
    
    # try:
    #     ids = bl_id.tolist()
    # except NameError:
    #     ids = range(bl.shape[0])

    # if bl_label_orig.size == bl_label.size:
    #     raise ValueError('No redundant baselines found.')

    simbl = simobj.baselines
    if simbl.shape[0] == bl.shape[0]:
        simbll = NP.sqrt(NP.sum(simbl**2, axis=1))
        simblo = NP.angle(simbl[:,0] + 1j * simbl[:,1], deg=True)
        simblza = NP.degrees(NP.arccos(simbl[:,2] / simbll))
        
        simblstr = ['{0[0]:.2f}_{0[1]:.3f}_{0[2]:.3f}'.format(lo) for lo in zip(simbll,simblza,simblo)]
    
        inp_blo = NP.angle(bl[:,0] + 1j * bl[:,1], deg=True)
        inp_blza = NP.degrees(NP.arccos(bl[:,2] / bl_length))
        inp_blstr = ['{0[0]:.2f}_{0[1]:.3f}_{0[2]:.3f}'.format(lo) for lo in zip(bl_length,inp_blza,inp_blo)]

        uniq_inp_blstr, inp_ind, inp_invind = NP.unique(inp_blstr, return_index=True, return_inverse=True)  ## if numpy.__version__ < 1.9.0
        uniq_sim_blstr, sim_ind, sim_invind = NP.unique(simblstr, return_index=True, return_inverse=True)  ## if numpy.__version__ < 1.9.0
        # uniq_inp_blstr, inp_ind, inp_invind, inp_frequency = NP.unique(inp_blstr, return_index=True, return_inverse=True, return_counts=True)  ## if numpy.__version__ >= 1.9.0
        # uniq_sim_blstr, sim_ind, sim_invind, sim_frequency = NP.unique(simblstr, return_index=True, return_inverse=True, return_counts=True)  ## if numpy.__version__ >= 1.9.0

        if simbl.shape[0] != uniq_sim_blstr.size:
            raise ValueError('Non-redundant baselines already found in the simulations')
        
        if not NP.array_equal(uniq_inp_blstr, uniq_sim_blstr):
            if args['parmsfile'] is None:
                raise IOError('Layout from simulations do not match simulated data.')
            else:
                raise IOError('Layout from input simulation parameters file do not match simulated data.')

        simobj.duplicate_measurements(blgroups)

        for outfmt in args['outfmt']:
            if outfmt.lower() == 'hdf5':
                simobj.save(outfile, fmt=outfmt, verbose=True, tabtype='BinTableHDU', npz=False, overwrite=True, uvfits_parms=None)
            else:
                uvfits_parms = None
                if parms['save_formats']['phase_center'] is None:
                    phase_center = simobj.pointing_center[0,:].reshape(1,-1)
                    phase_center_coords = simobj.pointing_coords
                    if phase_center_coords == 'dircos':
                        phase_center = GEOM.dircos2altaz(phase_center, units='degrees')
                        phase_center_coords = 'altaz'
                    if phase_center_coords == 'altaz':
                        phase_center = GEOM.altaz2hadec(phase_center, simobj.latitude, units='degrees')
                        phase_center_coords = 'hadec'
                    if phase_center_coords == 'hadec':
                        phase_center = NP.hstack((simobj.lst[0]-phase_center[0,0], phase_center[0,1]))
                        phase_center_coords = 'radec'
                    if phase_center_coords != 'radec':
                        raise ValueError('Invalid phase center coordinate system')
                        
                    uvfits_ref_point = {'location': phase_center.reshape(1,-1), 'coords': 'radec'}
                else:
                    uvfits_ref_point = {'location': NP.asarray(parms['save_formats']['phase_center']).reshape(1,-1), 'coords': 'radec'}
                uvfits_parms = {'ref_point': uvfits_ref_point, 'method': parms['save_formats']['uvfits_method']}
                
                simobj.write_uvfits(outfile, uvfits_parms=uvfits_parms, overwrite=True)
    if wait_after_run:
        PDB.set_trace()
