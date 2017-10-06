from __future__ import division
import numpy as NP
import scipy.constants as FCNST
from scipy import interpolate
import datetime as DT
import progressbar as PGB
import os, ast
import copy
import astropy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time
import warnings
import h5py
from distutils.version import LooseVersion
import psutil
from astroutils import geometry as GEOM
from astroutils import gridding_modules as GRD
from astroutils import constants as CNST
from astroutils import DSP_modules as DSP
from astroutils import catalog as SM
from astroutils import lookup_operations as LKP
from astroutils import nonmathops as NMO
import prisim
import baseline_delay_horizon as DLY
import primary_beams as PB
try:
    from pyuvdata import UVData
except ImportError:
    uvdata_module_found = False
else:
    uvdata_module_found = True
try:
    from mwapy.pb import primary_beam as MWAPB
except ImportError:
    mwa_tools_found = False
else:
    mwa_tools_found = True

prisim_path = prisim.__path__[0]+'/'    

################################################################################

def _astropy_columns(cols, tabtype='BinTableHDU'):

    """
    ----------------------------------------------------------------------------
    !!! FOR INTERNAL USE ONLY !!!
    This internal routine checks for Astropy version and produces the FITS
    columns based on the version

    Inputs:

    cols    [list of Astropy FITS columns] These are a list of Astropy FITS
            columns

    tabtype [string] specifies table type - 'BinTableHDU' (default) for binary
            tables and 'TableHDU' for ASCII tables

    Outputs:

    columns [Astropy FITS column data]
    ----------------------------------------------------------------------------
    """

    try:
        cols
    except NameError:
        raise NameError('Input cols not specified')

    if tabtype not in ['BinTableHDU', 'TableHDU']:
        raise ValueError('tabtype specified is invalid.')

    use_ascii = False
    if tabtype == 'TableHDU':
        use_ascii = True
    if astropy.__version__ == '0.4':
        columns = fits.ColDefs(cols, tbtype=tabtype)
    elif LooseVersion(astropy.__version__)>=LooseVersion('0.4.2'):
        columns = fits.ColDefs(cols, ascii=use_ascii)
    return columns

################################################################################

def read_gaintable(gainsfile, axes_order=None):

    """
    ---------------------------------------------------------------------------
    Read gain table from file and return

    Input:

    gainsfile   [string] Filename including the full path that contains the
                instrument gains. It must be in HDF5 format. It must contain
                the following structure:
                'antenna-based'     [dictionary] Contains antenna-based
                                    instrument gain information. It has the
                                    following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency'. Must be
                                                specified (no defaults)
                                    'gains'     [scalar or numpy array]
                                                Complex antenna-based
                                                instrument gains. Must be
                                                of shape (nax1, nax2, nax3)
                                                where ax1, ax2 and ax3 are
                                                specified by the axes
                                                ordering under key 'ordering'.
                                                If there is no variations in
                                                gains along an axis, then the
                                                corresponding nax may be set
                                                to 1 and the gains will be
                                                replicated along that axis
                                                using numpy array broadcasting.
                                                For example, shapes (nax1,1,1),
                                                (1,1,1), (1,nax2,nax3) are
                                                acceptable. If specified as a
                                                scalar, it will be replicated
                                                along all three axes, namely,
                                                'label', 'frequency' and 'time'
                                    'label'     [None or list or numpy array]
                                                List of antenna labels that
                                                correspond to the nax along
                                                the 'label' axis. If the
                                                nax=1 along the 'label' axis,
                                                this may be set to None, else
                                                it must be specified and must
                                                match the nax.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nax=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nax.
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                nax=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nax. It must be
                                                a float and can be in seconds,
                                                hours, days, etc.
                'baseline-based'    [dictionary] Contains baseline-based
                                    instrument gain information. It has the
                                    following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency'. Must be
                                                specified (no defaults)
                                    'gains'     [scalar or numpy array]
                                                Complex baseline-based
                                                instrument gains. Must be
                                                of shape (nax1, nax2, nax3)
                                                where ax1, ax2 and ax3 are
                                                specified by the axes
                                                ordering under key 'ordering'.
                                                If there is no variations in
                                                gains along an axis, then the
                                                corresponding nax may be set
                                                to 1 and the gains will be
                                                replicated along that axis
                                                using numpy array broadcasting.
                                                For example, shapes (nax1,1,1),
                                                (1,1,1), (1,nax2,nax3) are
                                                acceptable. If specified as a
                                                scalar, it will be replicated
                                                along all three axes, namely,
                                                'label', 'frequency' and 'time'
                                    'label'     [None or list or numpy array]
                                                List of baseline labels that
                                                correspond to the nax along
                                                the 'label' axis. If the
                                                nax=1 along the 'label' axis
                                                this may be set to None, else
                                                it must be specified and must
                                                match the nax.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nax=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nax.
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                nax=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nax. It must be
                                                a float and can be in seconds,
                                                hours, days, etc.

    axes_order  [None or list or numpy array] The gaintable which is read is
                stored in this axes ordering. If set to None, it will store in
                this order ['label', 'frequency', 'time']

    Output:

    gaintable   [None or dictionary] If set to None, all antenna- and baseline-
                based gains must be set to unity. If returned as dictionary, it
                contains the loaded gains. It contains the following keys and
                values:
                'antenna-based'     [None or dictionary] Contains antenna-based
                                    instrument gain information. If set to None,
                                    all antenna-based gains are set to unity.
                                    If returned as dictionary, it has the
                                    following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency' as specified
                                                in input axes_order
                                    'gains'     [scalar or numpy array]
                                                Complex antenna-based
                                                instrument gains. Must be
                                                of shape (nant, nchan, nts)
                                                If there is no variations in
                                                gains along an axis, then the
                                                corresponding nax may be set
                                                to 1 and the gains will be
                                                replicated along that axis
                                                using numpy array broadcasting.
                                                For example, shapes (nant,1,1),
                                                (1,1,1), (1,nchan,nts) are
                                                acceptable. If specified as a
                                                scalar, it will be replicated
                                                along all three axes, namely,
                                                'label', 'frequency' and 'time'
                                    'label'     [None or list or numpy array]
                                                List of antenna labels that
                                                correspond to nant along
                                                the 'label' axis. If nant=1,
                                                this may be set to None, else
                                                it will be specified and will
                                                match the nant.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nchan=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nchan.
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                ntimes=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the ntimes. It will
                                                be a float and in same units as
                                                given in input
                'baseline-based'    [None or dictionary] Contains baseline-based
                                    instrument gain information. If set to None,
                                    all baseline-based gains are set to unity.
                                    If returned as dictionary, it has the
                                    following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency' as specified
                                                in input axes_order
                                    'gains'     [scalar or numpy array]
                                                Complex baseline-based
                                                instrument gains. Must be
                                                of shape (nbl, nchan, nts)
                                                If there is no variations in
                                                gains along an axis, then the
                                                corresponding nax may be set
                                                to 1 and the gains will be
                                                replicated along that axis
                                                using numpy array broadcasting.
                                                For example, shapes (nbl,1,1),
                                                (1,1,1), (1,nchan,nts) are
                                                acceptable. If specified as a
                                                scalar, it will be replicated
                                                along all three axes, namely,
                                                'label', 'frequency' and 'time'
                                    'label'     [None or list or numpy array]
                                                List of baseline labels that
                                                correspond to nbl along the
                                                'label' axis. If nbl=1 along
                                                the 'label' axis this may be
                                                set to None, else it will be
                                                specified and will match nbl.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nchan=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nchan.
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                ntimes=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the ntimes. It will
                                                be a float and in same units as
                                                given in input
    ---------------------------------------------------------------------------
    """

    if axes_order is None:
        axes_order = ['label', 'frequency', 'time']
    elif not isinstance(axes_order, (list, NP.ndarray)):
        raise TypeError('axes_order must be a list')
    else:
        if len(axes_order) != 3:
            raise ValueError('axes_order must be a three element list')
        for orderkey in ['label', 'frequency', 'time']:
            if orderkey not in axes_order:
                raise ValueError('axes_order does not contain key "{0}"'.format(orderkey))

    gaintable = {}
    try:
        with h5py.File(gainsfile, 'r') as fileobj:
            for gainkey in fileobj:
                try:
                    gaintable[gainkey] = {}
                    grp = fileobj[gainkey]
                    if isinstance(grp['gains'].value, (NP.float32, NP.float64, NP.complex64, NP.complex128)):
                        gaintable[gainkey]['gains'] = NP.asarray(grp['gains'].value).reshape(1,1,1)
                    elif isinstance(grp['gains'].value, NP.ndarray):
                        if 'ordering' in grp:
                            ordering = list(grp['ordering'].value)
                        else:
                            raise KeyError('Axes ordering for gains not specified')

                        if len(ordering) != 3:
                            raise ValueError('Ordering must contain three elements')
                        elif ('time' not in ordering) or ('label' not in ordering) or ('frequency' not in ordering):
                            raise ValueError('Required elements not found in ordering of instrument gains')
                        else:
                            if grp['gains'].value.ndim == 3:
                                transpose_order = NMO.find_list_in_list(ordering, axes_order)
                                gaintable[gainkey]['gains'] = NP.transpose(grp['gains'].value, axes=transpose_order)
                                for subkey in ['time', 'label', 'frequency']:
                                    gaintable[gainkey][subkey] = None
                                    if isinstance(grp[subkey].value, NP.ndarray):
                                        if gaintable[gainkey]['gains'].shape[axes_order.index(subkey)] > 1:
                                            if subkey not in grp:
                                                raise KeyError('Key "{0}" not specified'.format(subkey))
                                            else:
                                                if not isinstance(grp[subkey].value, (list, NP.ndarray)):
                                                    raise TypeError('"{0} key must be specified as a list or numpy array'.format(subkey))
                                                gaintable[gainkey][subkey] = NP.asarray(grp[subkey].value).ravel()
                                                if gaintable[gainkey][subkey].size != gaintable[gainkey]['gains'].shape[axes_order.index(subkey)]:
                                                    raise ValueError('List of labels and the gains do not match in dimensions')
                                    else:
                                        raise TypeError('Value of key "{0}" in {1} gains must be a numpy array'.format(subkey, gainkey))
                            else:
                                raise ValueError('Gains array must be three-dimensional. Use fake dimension if there is no variation along any particular axis.')
                    else:
                        warnings.warn('Invalid data type specified for {0} instrument gains. Proceeding with defaults (unity gains)'.format(gainkey))
                    gaintable[gainkey]['ordering'] = axes_order
                except KeyError:
                    warnings.warn('No info found on {0} instrument gains. Proceeding with defaults (unity gains)'.format(gainkey))
    except IOError:
        warnings.warn('Invalid file specified for instrument gains. Proceeding with defaults (unity gains)')
        gaintable = None
    if not gaintable:
        gaintable = None

    return gaintable

################################################################################

def extract_gains(gaintable, bl_labels, freq_index=None, time_index=None,
                  axes_order=None):

    """
    ---------------------------------------------------------------------------
    Extract complex instrument gains for given baselines from the gain table.

    Inputs:

    gaintable   [None or dictionary] If set to None, all antenna- and baseline-
                based gains must be set to unity. If returned as dictionary, it
                contains the loaded gains. It contains the following keys and
                values:
                'antenna-based'     [None or dictionary] Contains antenna-based
                                    instrument gain information. If set to None,
                                    all antenna-based gains are set to unity.
                                    If returned as dictionary, it has the
                                    following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency'. Must be
                                                specified (no defaults)
                                    'gains'     [scalar or numpy array]
                                                Complex antenna-based
                                                instrument gains. Must be
                                                of shape (nant, nchan, nts)
                                                If there is no variations in
                                                gains along an axis, then the
                                                corresponding nax may be set
                                                to 1 and the gains will be
                                                replicated along that axis
                                                using numpy array broadcasting.
                                                For example, shapes (nant,1,1),
                                                (1,1,1), (1,nchan,nts) are
                                                acceptable. If specified as a
                                                scalar, it will be replicated
                                                along all three axes, namely,
                                                'label', 'frequency' and
                                                'time'.
                                    'label'     [None or list or numpy array]
                                                List or antenna labels that
                                                correspond to nant along
                                                the 'label' axis. If nant=1,
                                                this may be set to None, else
                                                it will be specified and will
                                                match the nant.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nchan=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nchan
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                ntimes=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the ntimes. It must
                                                be a float and can be in
                                                seconds, hours, days, etc.
                'baseline-based'    [None or dictionary] Contains baseline-based
                                    instrument gain information. If set to None,
                                    all baseline-based gains are set to unity.
                                    If returned as dictionary, it has the
                                    following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency'. Must be
                                                specified (no defaults)
                                    'gains'     [scalar or numpy array]
                                                Complex baseline-based
                                                instrument gains. Must be
                                                of shape (nbl, nchan, nts)
                                                If there is no variations in
                                                gains along an axis, then the
                                                corresponding nax may be set
                                                to 1 and the gains will be
                                                replicated along that axis
                                                using numpy array broadcasting.
                                                For example, shapes (nant,1,1),
                                                (1,1,1), (1,nchan,nts) are
                                                acceptable. If specified as a
                                                scalar, it will be replicated
                                                along all three axes, namely,
                                                'label', 'frequency' and
                                                'time'.
                                    'label'     [None or list or numpy array]
                                                List or baseline labels that
                                                correspond to nbl along
                                                the 'label' axis. If nbl=1
                                                along the 'label' axis
                                                this may be set to None, else
                                                it will be specified and will
                                                match nbl.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nchan=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nchan
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                ntimes=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the ntimes. It must
                                                be a float and can be in
                                                seconds, hours, days, etc.

    bl_labels   [Numpy structured array tuples] Labels of antennas in the pair
                used to produce the baseline vector under fields 'A2' and 'A1'
                for second and first antenna respectively. The baseline vector
                is obtained by position of antennas under 'A2' minus position
                of antennas under 'A1'

    freq_index  [None, int, list or numpy array] Index (scalar) or indices
                (list or numpy array) along the frequency axis at which gains
                are to be extracted. If set to None, gains at all frequencies
                in the gain table will be extracted.

    time_index  [None, int, list or numpy array] Index (scalar) or indices
                (list or numpy array) along the time axis at which gains
                are to be extracted. If set to None, gains at all timesin the
                gain table will be extracted.

    axes_order  [None or list or numpy array] Axes ordering for extracted
                gains. It must contain the three elements 'label',
                'frequency', and 'time'. If set to None, it will be returned
                in the same order as in the input gaintable.

    Outputs:

    [numpy array] Complex gains of shape nbl x nchan x nts for the specified
    baselines, frequencies and times.
    ---------------------------------------------------------------------------
    """

    try:
        gaintable, bl_labels
    except NameError:
        raise NameError('Inputs gaintable and bl_labels must be specified')

    blgains = NP.asarray(1.0).reshape(1,1,1)

    if gaintable is not None:
        a1_labels = bl_labels['A1']
        a2_labels = bl_labels['A2']
        for gainkey in ['antenna-based', 'baseline-based']:
            if gainkey in gaintable:
                temp_axes_order = ['label', 'frequency', 'time']
                inp_order = gaintable[gainkey]['ordering']
                temp_transpose_order = NMO.find_list_in_list(inp_order, temp_axes_order)
                if NP.all(inp_order == temp_axes_order):
                    gains = NP.copy(gaintable[gainkey]['gains'])
                else:
                    gains = NP.transpose(NP.copy(gaintable[gainkey]['gains']), axes=temp_transpose_order)

                if freq_index is None:
                    freq_index = NP.arange(gains.shape[1])
                elif isinstance(freq_index, (int,list,NP.ndarray)):
                    freq_index = NP.asarray(freq_index).ravel()
                if NP.any(freq_index >= gains.shape[1]):
                    raise IndexError('Input freq_index cannot exceed the frequency dimensions in the gain table')
                if time_index is None:
                    time_index = NP.arange(gains.shape[2])
                elif isinstance(time_index, (int,list,NP.ndarray)):
                    time_index = NP.asarray(time_index).ravel()
                if NP.any(time_index >= gains.shape[2]):
                    raise IndexError('Input time_index cannot exceed the time dimensions in the gain table')

                if gains.shape[0] == 1:
                    blgains = blgains * gains[:,freq_index,time_index].reshape(1,freq_index.size,time_index.size)
                else:
                    labels = gaintable[gainkey]['label']
                    if gainkey == 'antenna-based':
                        ind1 = NMO.find_list_in_list(labels, a1_labels)
                        ind2 = NMO.find_list_in_list(labels, a2_labels)
                        if NP.sum(ind1.mask) > 0:
                            raise IndexError('Some antenna gains could not be found')
                        if NP.sum(ind2.mask) > 0:
                            raise IndexError('Some antenna gains could not be found')
                        blgains = blgains * gains[NP.ix_(ind2,freq_index,time_index)].reshape(ind2.size,freq_index.size,time_index.size) * gains[NP.ix_(ind1,freq_index,time_index)].conj().reshape(ind1.size,freq_index.size,time_index.size)
                    else:
                        labels_conj = [tuple(reversed(label)) for label in labels]
                        labels_conj = NP.asarray(labels_conj, dtype=labels.dtype)
                        labels_conj_appended = NP.concatenate((labels, labels_conj), axis=0)
                        gains_conj_appended = NP.concatenate((gains, gains.conj()), axis=0)
                        ind = NMO.find_list_in_list(labels_conj_appended, bl_labels)
                        selected_gains = gains_conj_appended[NP.ix_(ind.compressed(),freq_index,time_index)]
                        if ind.compressed().size == 1:
                            selected_gains = selected_gains.reshape(NP.sum(~ind.mask),freq_index.size,time_index.size)
                        blgains[~ind.mask, ...] = blgains[~ind.mask, ...] * selected_gains
                if axes_order is None:
                    axes_order = inp_order
                elif not isinstance(axes_order, (list, NP.ndarray)):
                    raise TypeError('axes_order must be a list')
                else:
                    if len(axes_order) != 3:
                        raise ValueError('axes_order must be a three element list')
                    for orderkey in ['label', 'frequency', 'time']:
                        if orderkey not in axes_order:
                            raise ValueError('axes_order does not contain key "{0}"'.format(orderkey))

                transpose_order = NMO.find_list_in_list(inp_order, axes_order)
                blgains = NP.transpose(blgains, axes=transpose_order)

    return blgains

################################################################################

def hexagon_generator(spacing, n_total=None, n_side=None, orientation=None,
                      center=None):

    """
    ------------------------------------------------------------------------
    Generate a grid of baseline locations filling a regular hexagon.
    Primarily intended for HERA experiment.

    Inputs:

    spacing      [scalar] positive scalar specifying the spacing between
                 antennas. Must be specified, no default.

    n_total      [scalar] positive integer specifying the total number of
                 antennas to be placed in the hexagonal array. This value
                 will be checked if it valid for a regular hexagon. If
                 n_total is specified, n_side must not be specified.
                 Default = None.

    n_side       [scalar] positive integer specifying the number of antennas
                 on the side of the hexagonal array. If n_side is specified,
                 n_total should not be specified. Default = None

    orientation  [scalar] counter-clockwise angle (in degrees) by which the
                 principal axis of the hexagonal array is to be rotated.
                 Default = None (means 0 degrees)

    center       [2-element list or numpy array] specifies the center of the
                 array. Must be in the same units as spacing. The hexagonal
                 array will be centered on this position.

    Outputs:

    Two element tuple with these elements in the following order:

    xy           [2-column array] x- and y-locations. x is in the first
                 column, y is in the second column. Number of xy-locations
                 is equal to the number of rows which is equal to n_total

    id           [numpy array of string] unique antenna identifier. Numbers
                 from 0 to n_antennas-1 in string format.

    Notes:

    If n_side is the number of antennas on the side of the hexagon, then
    n_total = 3*n_side**2 - 3*n_side + 1
    ------------------------------------------------------------------------
    """

    try:
        spacing
    except NameError:
        raise NameError('No spacing provided.')

    if not isinstance(spacing, (int, float)):
        raise TypeError('spacing must be scalar value')

    if spacing <= 0:
        raise ValueError('spacing must be positive')

    if orientation is not None:
        if not isinstance(orientation, (int,float)):
            raise TypeError('orientation must be a scalar')

    if center is not None:
        if not isinstance(center, (list, NP.ndarray)):
            raise TypeError('center must be a list or numpy array')
        center = NP.asarray(center)
        if center.size != 2:
            raise ValueError('center should be a 2-element vector')
        center = center.reshape(1,-1)

    if (n_total is None) and (n_side is None):
        raise NameError('n_total or n_side must be provided')
    elif (n_total is not None) and (n_side is not None):
        raise ValueError('Only one of n_total or n_side must be specified.')
    elif n_total is not None:
        if not isinstance(n_total, int):
            raise TypeError('n_total must be an integer')
        if n_total <= 0:
            raise ValueError('n_total must be positive')
    else:
        if not isinstance(n_side, int):
            raise TypeError('n_side must be an integer')
        if n_side <= 0:
            raise ValueError('n_side must be positive')

    if n_total is not None:
        sqroots = NP.roots([3.0, -3.0, 1.0-n_total])
        valid_ind = NP.logical_and(sqroots.real >= 1, sqroots.imag == 0.0)
        if NP.any(valid_ind):
            sqroot = sqroots[valid_ind]
        else:
            raise ValueError('No valid root found for the quadratic equation with the specified n_total')

        n_side = NP.round(sqroot).astype(NP.int)
        if (3*n_side**2 - 3*n_side + 1 != n_total):
            raise ValueError('n_total is not a valid number for a hexagonal array')
    else:
        n_total = 3*n_side**2 - 3*n_side + 1

    xref = NP.arange(2*n_side-1, dtype=NP.float)
    xloc, yloc = [], []
    for i in range(1,n_side):
        x = xref[:-i] + i * NP.cos(NP.pi/3)   # Select one less antenna each time and displace
        y = i*NP.sin(NP.pi/3) * NP.ones(2*n_side-1-i)
        xloc += x.tolist() * 2   # Two lists, one for the top and the other for the bottom
        yloc += y.tolist()   # y-locations of the top list
        yloc += (-y).tolist()   # y-locations of the bottom list

    xloc += xref.tolist()   # Add the x-locations of central line of antennas
    yloc += [0.0] * int(2*n_side-1)   # Add the y-locations of central line of antennas

    if len(xloc) != len(yloc):
        raise ValueError('Sizes of x- and y-locations do not agree')

    xy = zip(xloc, yloc)
    if len(xy) != n_total:
        raise ValueError('Sizes of x- and y-locations do not agree with n_total')

    xy = NP.asarray(xy)
    xy = xy - NP.mean(xy, axis=0, keepdims=True)    # Shift the center to origin
    if orientation is not None:   # Perform any rotation
        angle = NP.radians(orientation)
        rot_matrix = NP.asarray([[NP.cos(angle), -NP.sin(angle)],
                                 [NP.sin(angle), NP.cos(angle)]])
        xy = NP.dot(xy, rot_matrix.T)

    xy *= spacing    # Scale by the spacing
    if center is not None:   # Shift the center
        xy += center

    return (NP.asarray(xy), map(str, range(n_total)))

################################################################################

def rectangle_generator(spacing, n_side, orientation=None, center=None):

    """
    ------------------------------------------------------------------------
    Generate a grid of baseline locations filling a rectangular array.
    Primarily intended for HIRAX, CHIME and PAPER experiments

    Inputs:

    spacing      [2-element list or numpy array] positive integers specifying
    		 the spacing between antennas. Must be specified, no default.

    n_side       [2-element list or numpy array] positive integers specifying
    		 the number of antennas on each side of the rectangular array.
    		 Atleast one value should be specified, no default.

    orientation  [scalar] counter-clockwise angle (in degrees) by which the
                 principal axis of the rectangular array is to be rotated.
                 Default = None (means 0 degrees)

    center       [2-element list or numpy array] specifies the center of the
                 array. Must be in the same units as spacing. The rectangular
                 array will be centered on this position.

    Outputs:

    Two element tuple with these elements in the following order:

    xy           [2-column array] x- and y-locations. x is in the first
                 column, y is in the second column. Number of xy-locations
                 is equal to the number of rows which is equal to n_total

    id           [numpy array of string] unique antenna identifier. Numbers
                 from 0 to n_antennas-1 in string format.

    Notes:

    ------------------------------------------------------------------------
    """
    try:
        spacing
    except NameError:
        raise NameError('No spacing provided.')

    if spacing is not None:
        if not isinstance(spacing, (int, float, list, NP.ndarray)):
            raise TypeError('spacing must be a scalar or list/numpy array')
        spacing = NP.asarray(spacing)
        if spacing.size < 2:
            spacing = NP.resize(spacing,(1,2))
        if NP.all(NP.less_equal(spacing,NP.zeros((1,2)))):
            raise ValueError('spacing must be positive')

    if orientation is not None:
        if not isinstance(orientation, (int,float)):
            raise TypeError('orientation must be a scalar')

    if center is not None:
        if not isinstance(center, (list, NP.ndarray)):
            raise TypeError('center must be a list or numpy array')
        center = NP.asarray(center)
        if center.size != 2:
            raise ValueError('center should be a 2-element vector')
        center = center.reshape(1,-1)

    if n_side is None:
        raise NameError('Atleast one value of n_side must be provided')
    else:
        if not isinstance(n_side,  (int, float, list, NP.ndarray)):
            raise TypeError('n_side must be a scalar or list/numpy array')
        n_side = NP.asarray(n_side)
        if n_side.size < 2:
            n_side = NP.resize(n_side,(1,2))
        if NP.all(NP.less_equal(n_side,NP.zeros((1,2)))):
            raise ValueError('n_side must be positive')

	n_total = NP.prod(n_side, dtype=NP.uint8)
	xn,yn = NP.hsplit(n_side,2)
	xn = NP.asscalar(xn)
	yn = NP.asscalar(yn)

	xs,ys = NP.hsplit(spacing,2)
	xs = NP.asscalar(xs)
	ys = NP.asscalar(ys)

	n_total = xn*yn

	x = NP.linspace(0, xn-1, xn)
	x = x - NP.mean(x)
	x = x*xs

	y = NP.linspace(0, yn-1, yn)
	y = y - NP.mean(y)
	y = y*ys

	xv, yv = NP.meshgrid(x,y)

	xy = NP.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))

    if len(xy) != n_total:
        raise ValueError('Sizes of x- and y-locations do not agree with n_total')

    if orientation is not None:   # Perform any rotation
 	angle = NP.radians(orientation)
 	rot_matrix = NP.asarray([[NP.cos(angle), -NP.sin(angle)], [NP.sin(angle), NP.cos(angle)]])
 	xy = NP.dot(xy, rot_matrix.T)

    if center is not None:   # Shift the center
        xy += center

    return (NP.asarray(xy), map(str, range(n_total)))

################################################################################

def circular_antenna_array(antsize, minR, maxR=None):

    """
    ---------------------------------------------------------------------------
    Create antenna layout in a circular ring of minimum and maximum radius with
    antennas of a given size

    Inputs:

    antsize   [scalar] Antenna size. Critical to determining number of antenna
              elements that can be placed on a circle. No default.

    minR      [scalar] Minimum radius of the circular ring. Must be in same
              units as antsize. No default. Must be greater than 0.5*antsize.

    maxR      [scalar] Maximum radius of circular ring. Must be >= minR.
              Default=None means maxR is set equal to minR.

    Outputs:

    xy        [2-column numpy array] Antenna locations in the same units as
              antsize returned as a 2-column numpy array where the number of
              rows equals the number of antenna locations generated and x,
              and y locations make the two columns.
    ---------------------------------------------------------------------------
    """

    try:
        antsize, minR
    except NameError:
        raise NameError('antsize, and minR must be specified')

    if (antsize is None) or (minR is None):
        raise ValueError('antsize and minR cannot be NoneType')

    if not isinstance(antsize, (int, float)):
        raise TypeError('antsize must be a scalar')
    if antsize <= 0.0:
        raise ValueError('antsize must be positive')

    if not isinstance(minR, (int, float)):
        raise TypeError('minR must be a scalar')
    if minR <= 0.0:
        raise ValueError('minR must be positive')

    if minR < 0.5*antsize:
        minR = 0.5*antsize

    if maxR is None:
        maxR = minR

    if not isinstance(maxR, (int, float)):
        raise TypeError('maxR must be a scalar')
    elif maxR < minR:
        maxR = minR

    if maxR - minR < antsize:
        radii = minR + NP.zeros(1)
    else:
        radii = minR + antsize * NP.arange((maxR-minR)/antsize)
    nants = 2 * NP.pi * radii / antsize
    nants = nants.astype(NP.int)

    x = [(radii[i] * NP.cos(2*NP.pi*NP.arange(nants[i])/nants[i])).tolist() for i in range(radii.size)]
    y = [(radii[i] * NP.sin(2*NP.pi*NP.arange(nants[i])/nants[i])).tolist() for i in range(radii.size)]

    xpos = [xi for sublist in x for xi in sublist]
    ypos = [yi for sublist in y for yi in sublist]
    x = NP.asarray(xpos)
    y = NP.asarray(ypos)

    xy = NP.hstack((x.reshape(-1,1), y.reshape(-1,1)))

    return (xy, map(str, range(NP.sum(nants))))

################################################################################

def baseline_generator(antenna_locations, ant_label=None, ant_id=None,
                       auto=False, conjugate=False):

    """
    ---------------------------------------------------------------------------
    Generate baseline from antenna locations.

    Inputs:

    antenna_locations: List of tuples containing antenna coordinates,
                       or list of instances of class Point containing
                       antenna coordinates, or Numpy array (Nx3) array
                       with each row specifying an antenna location.

    Input keywords:

    ant_label          [list of strings] Unique string identifier for each
                       antenna. Default = None. If None provided,
                       antennas will be indexed by an integer starting
                       from 0 to N(ants)-1

    ant_id             [list of integers] Unique integer identifier for each
                       antenna. Default = None. If None provided,
                       antennas will be indexed by an integer starting
                       from 0 to N(ants)-1

    auto:              [Default=False] If True, compute zero spacings of
                       antennas with themselves.

    conjugate:         [Default=False] If True, compute conjugate
                       baselines.

    Output:

    baseline_locations: Baseline locations in the same data type as
                        antenna locations (list of tuples, list of
                        instances of class Point or Numpy array of size
                        Nb x 3 with each row specifying one baseline
                        vector)

    antpair_labels      [Numpy structured array tuples] Labels of
                        antennas in the pair used to produce the
                        baseline vector under fields 'A2' and 'A1' for
                        second and first antenna respectively. The
                        baseline vector is obtained by position of
                        antennas under 'A2' minus position of antennas
                        under 'A1'

    antpair_ids         [Numpy structured array tuples] IDs of antennas
                        in the pair used to produce the baseline vector
                        under fields 'A2' and 'A1' for second and first
                        antenna respectively. The baseline vector is
                        obtained by position of antennas under 'A2'
                        minus position of antennas under 'A1'
    -------------------------------------------------------------------
    """

    try:
        antenna_locations
    except NameError:
        print 'No antenna locations supplied. Returning from baseline_generator()'
        return None

    inp_type = 'tbd'

    if not isinstance(antenna_locations, NP.ndarray):
        if isinstance(antenna_locations, list):
            if isinstance(antenna_locations[0], GEOM.Point):
                inp_type = 'loo' # list of objects
            elif isinstance(antenna_locations[0], tuple):
                inp_type = 'lot' # list of tuples
                antenna_locations = [(tuple(loc) if len(loc) == 3 else (tuple([loc[0],0.0,0.0]) if len(loc) == 1 else (tuple([loc[0],loc[1],0.0]) if len(loc) == 2 else (tuple([loc[0],loc[1],loc[2]]))))) for loc in antenna_locations if len(loc) != 0] # Remove empty tuples and validate the data range and data type for antenna locations. Force it to have three components for every antenna location.
        elif isinstance(antenna_locations, GEOM.Point):
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return GEOM.Point()
        elif isinstance(antenna_locations, tuple):
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return (0.0,0.0,0.0)
        else:
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return (0.0,0.0,0.0)
    else:
        inp_type = 'npa' # A numpy array
        if antenna_locations.shape[0] == 1:
            if not auto:
                print 'No non-zero spacings found since auto=False.'
                return None
            else:
                return NP.zeros(1,3)
        else:
            if antenna_locations.shape[1] > 3:
                antenna_locations = antenna_locations[:,:3]
            elif antenna_locations.shape[1] < 3:
                antenna_locations = NP.hstack((antenna_locations, NP.zeros((antenna_locations.shape[0],3-antenna_locations.shape[1]))))

    if isinstance(antenna_locations, list):
        num_ants = len(antenna_locations)
    else:
        num_ants = antenna_locations.shape[0]

    if ant_label is not None:
        if isinstance(ant_label, list):
            if len(ant_label) != num_ants:
                raise ValueError('Dimensions of ant_label and antenna_locations do not match.')
        elif isinstance(ant_label, NP.ndarray):
            if ant_label.size != num_ants:
                raise ValueError('Dimensions of ant_label and antenna_locations do not match.')
            ant_label = ant_label.tolist()
    else:
        ant_label = ['{0:0d}'.format(i) for i in xrange(num_ants)]

    if ant_id is not None:
        if isinstance(ant_id, list):
            if len(ant_id) != num_ants:
                raise ValueError('Dimensions of ant_id and antenna_locations do not match.')
        elif isinstance(ant_id, NP.ndarray):
            if ant_id.size != num_ants:
                raise ValueError('Dimensions of ant_id and antenna_locations do not match.')
            ant_id = ant_id.tolist()
    else:
        ant_id = range(num_ants)

    if inp_type == 'loo':
        if auto:
            baseline_locations = [antenna_locations[j]-antenna_locations[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            # antpair_labels = [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            antpair_labels = [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            antpair_ids = [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
        else:
            baseline_locations = [antenna_locations[j]-antenna_locations[i] for i in range(0,num_ants) for j in range(0,num_ants) if j > i]
            # antpair_labels = [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            antpair_labels = [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            antpair_ids = [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
        if conjugate:
            baseline_locations += [antenna_locations[j]-antenna_locations[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            # antpair_labels += [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            antpair_labels += [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            antpair_ids += [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
    elif inp_type == 'lot':
        if auto:
            baseline_locations = [tuple((antenna_locations[j][0]-antenna_locations[i][0], antenna_locations[j][1]-antenna_locations[i][1], antenna_locations[j][2]-antenna_locations[i][2])) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            # antpair_labels = [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            antpair_labels = [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            antpair_ids = [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
        else:
            baseline_locations = [tuple((antenna_locations[j][0]-antenna_locations[i][0], antenna_locations[j][1]-antenna_locations[i][1], antenna_locations[j][2]-antenna_locations[i][2])) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            # antpair_labels = [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            antpair_labels = [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            antpair_ids = [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
        if conjugate:
            baseline_locations += [tuple((antenna_locations[j][0]-antenna_locations[i][0], antenna_locations[j][1]-antenna_locations[i][1], antenna_locations[j][2]-antenna_locations[i][2])) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            # antpair_labels += [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            antpair_labels += [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            antpair_ids += [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
    elif inp_type == 'npa':
        if auto:
            baseline_locations = [antenna_locations[j,:]-antenna_locations[i,:] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            # antpair_labels = [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            antpair_labels = [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
            antpair_ids = [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j >= i]
        else:
            baseline_locations = [antenna_locations[j,:]-antenna_locations[i,:] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            # antpair_labels = [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            antpair_labels = [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
            antpair_ids = [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j > i]
        if conjugate:
            baseline_locations += [antenna_locations[j,:]-antenna_locations[i,:] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            # antpair_labels += [ant_label[j]+'-'+ant_label[i] for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            antpair_labels += [(ant_label[j], ant_label[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]
            antpair_ids += [(ant_id[j], ant_id[i]) for i in xrange(0,num_ants) for j in xrange(0,num_ants) if j < i]

        baseline_locations = NP.asarray(baseline_locations)
        maxlen = max(len(albl) for albl in ant_label)
        antpair_labels = NP.asarray(antpair_labels, dtype=[('A2', '|S{0:0d}'.format(maxlen)), ('A1', '|S{0:0d}'.format(maxlen))])
        antpair_ids = NP.asarray(antpair_ids, dtype=[('A2', int), ('A1', int)])

    return baseline_locations, antpair_labels, antpair_ids

#################################################################################

def uniq_baselines(baseline_locations, redundant=None):

    """
    ---------------------------------------------------------------------------
    Identify unique, redundant or non-redundant baselines from a given set of
    baseline locations.

    Inputs:

    baseline_locations [2- or 3-column numpy array] Each row of the array
                       specifies a baseline vector from which the required
                       set of baselines have to be identified

    redundant          [None or boolean] If set to None (default), all the
                       unique baselines including redundant and non-redundant
                       baselines are returned. If set to True, only redundant
                       baselines that occur more than once are returned. If set
                       to False, only non-redundant baselines that occur
                       exactly once are returned.

    Output:

    4-element tuple with the selected baselines, their unique indices in the 
    input, their count and the indices of all occurences of each unique 
    baseline. The first element of this tuple is a 3-column numpy array 
    which is a subset of baseline_locations containing the requested type of 
    baselines. The second element of the tuple contains the selected indices 
    of the input array from which the first element in the tuple is determined
    relative to the input array. The third element of the tuple contains the 
    count of these selected baselines. In case of redundant and unique 
    baselines, the order of repeated baselines does not matter and any one of 
    those baselines could be returned without preserving the order. The fourth
    element in the tuple contains a list of lists where each element in the
    top level list corresponds to a unique baseline and consists of indices
    of all occurrences of input baselines redundant with this unique baseline 
    ---------------------------------------------------------------------------
    """

    try:
        baseline_locations
    except NameError:
        raise NameError('baseline_locations not provided')

    if not isinstance(baseline_locations, NP.ndarray):
        raise TypeError('baseline_locations must be a numpy array')

    if redundant is not None:
        if not isinstance(redundant, bool):
            raise TypeError('keyword "redundant" must be set to None or a boolean value')

    blshape = baseline_locations.shape
    if blshape[1] > 3:
        baseline_locations = baseline_locations[:,:3]
    elif blshape[1] < 3:
        baseline_locations = NP.hstack((baseline_locations, NP.zeros((blshape[0],3-blshape[1]))))

    blo = NP.angle(baseline_locations[:,0] + 1j * baseline_locations[:,1], deg=True)
    blo[blo >= 180.0] -= 180.0
    blo[blo < 0.0] += 180.0

    bll = NP.sqrt(NP.sum(baseline_locations**2, axis=1))
    blza = NP.degrees(NP.arccos(baseline_locations[:,2] / bll))

    blstr = ['{0[0]:.2f}_{0[1]:.3f}_{0[2]:.3f}'.format(lo) for lo in zip(bll,3.6e3*blza,3.6e3*blo)]

    uniq_blstr, ind, invind = NP.unique(blstr, return_index=True, return_inverse=True)  ## if numpy.__version__ < 1.9.0

    # uniq_blstr, ind, invind, frequency = NP.unique(blstr, return_index=True, return_inverse=True, return_counts=True)  ## if numpy.__version__ >= 1.9.0

    count_blstr = [(ubstr,blstr.count(ubstr)) for ubstr in uniq_blstr]  ## if numpy.__version__ < 1.9.0
    if redundant is None:
        retind = NP.copy(ind)
        counts = [tup[1] for tup in count_blstr]
        counts = NP.asarray(counts)
    else:
        if not redundant:
            ## if numpy.__version__ < 1.9.0
            non_redn_ind = [i for i,tup in enumerate(count_blstr) if tup[1] == 1]
            retind = ind[NP.asarray(non_redn_ind)]
            counts = NP.ones(retind.size)
        else:
            ## if numpy.__version__ < 1.9.0
            redn_ind_counts = [(i,tup[1]) for i,tup in enumerate(count_blstr) if tup[1] > 1]
            redn_ind, counts = zip(*redn_ind_counts)
            retind = ind[NP.asarray(redn_ind)]
            counts = NP.asarray(counts)
            
    allinds_where_found = NMO.find_all_occurrences_list1_in_list2(invind[retind], invind)
    return (baseline_locations[retind,:], retind, counts, allinds_where_found)

#################################################################################

def getBaselineInfo(inpdict):

    """
    ---------------------------------------------------------------------------
    Generate full baseline info from a given layout and return information 
    about redundancy and the mapping between unique and redundant baselines

    Input:

    inpdict [dictionary] It contains the following keys and values:
            'array'     [dictionary] It contains the following keys and values:
                        'redundant' [boolean] If this key is present, it says
                                    whether the array could be redundant (true)
                                    or not (false). If key is absent, this 
                                    value is assumed to be true. When it is set
                                    to true, it basically checks for redundancy
                                    otherwise not. It is not meant to say if 
                                    the array is actually redundant or not but
                                    only used for redundancy check to happen or
                                    not
                        'layout'    [string] Preset array layouts mutually 
                                    exclusive to antenna file. Only one of 
                                    these must be specified. Accepted 
                                    values are 'MWA-128T', 'HERA-7', 'HERA-19', 
                                    'HERA-37', 'HERA-61', 'HERA-91', 
                                    'HERA-127', 'HERA-169', 'HERA-217', 
                                    'HERA-271', 'HERA-331', 'PAPER-64', 
                                    'PAPER-112', 'HIRAX-1024', 'CHIME', 'CIRC', 
                                    or None (if layout file is specified). 
                        'file'      [string] File containing antenna locations
                                    parsed according to info in parser (see 
                                    below). If preset layout is specified, this
                                    must be set to None.
                        'filepathtype'
                                    [string] Accepted values are 'default' (if
                                    layout file can be found in prisim path, 
                                    namely, prisim/data/array_layouts folder)
                                    and 'custom'. If set to 'default', only 
                                    filename should be specified in file and it
                                    will be searched in the default 
                                    array_layouts folder
                                    prisim/data/array_layouts. 
                                    If set to 'custom' then the full path
                                    to the file must be specified.
                        'parser'    [dictionary] Will be used for parsing the 
                                    file if file is specified for array layout.
                                    It contains the following keys and values:
                                    'comment'   [string] Character used to 
                                                denote commented lines to be
                                                ignored. Default=None ('#')
                                    'delimiter' [string] Delimiter string. 
                                                Accepted values are whitespace 
                                                (default or None), ',' and '|'
                                    'data_strart'
                                                [integer] Line index for the 
                                                start of data not counting 
                                                comment or blank lines. A line 
                                                with only whitespace is 
                                                considered blank. It is 
                                                required. No defaults. 
                                                Indexing starts from 0
                                    'data_end'  [integer] Line index for the end 
                                                of data not counting comment or 
                                                blank lines. This value can be 
                                                negative to count from the end. 
                                                Default is None (all the way to 
                                                end of file). Indexing starts 
                                                from 0.
                                    'header_start'
                                                [integer] Line index for the 
                                                header line not counting comment 
                                                or blank lines. A line with only 
                                                whitespace is considered blank. 
                                                Must be provided. No defaults
                                    'label'     [string] String in the header 
                                                containing antenna labels. If 
                                                set to None (default), antenna 
                                                labels will be automatically 
                                                assigned. e.g. of some accepted 
                                                values are None, 'label', 'id', 
                                                'antid', etc. This must be found 
                                                in the header
                                    'east'      [string] String specifying East 
                                                coordinates in the header and 
                                                data. Must be provided. No 
                                                defaults.
                                    'north'     [string] String specifying North
                                                coordinates in the header and 
                                                data. Must be provided. No 
                                                defaults.
                                    'up'        [string] String specifying 
                                                elevation coordinates in the 
                                                header and data. Must be 
                                                provided. No defaults.
                        'minR'      [string] Minimum radius of circular ring.
                                    Applies only when layout = 'CIRC'

                        'maxR'      [string] Maximum radius of circular ring.
                                    Applies only when layout = 'CIRC'
                        'rms_tgtplane'
                                    [float] Perturbation of antenna positions
                                    (in m) in tangent plane. Default=0.0
                        'rms_elevation'
                                    [float] Perturbation of antenna positions
                                    (in m) in perpendicular to tangent plane.
                                    Default=0.0
                        'seed'      [integer] Random number seed for antenna
                                    position perturbations. Default=None means 
                                    no fixed seed
            'baseline'  [dictionary] Parameters specifying baseline
                        selection criteria. It consists of the following keys
                        and values:
                        'min'       [float] Minimum baseline in distance
                                    units (m). Default=None (0.0)
                        'max'       [float] Maximum baseline in distance
                                    units (m). Default=None (max baseline)
                        'direction' [string] Baseline vector directions to
                                    select. Default=None (all directions).
                                    Other accepted values are 'E' (east)
                                    'SE' (south-east), 'NE' (north-east),
                                    and 'N' (north). Multiple values from
                                    this accepted list can be specified
                                    as a list of strings. e.g., ['N', 'E'], 
                                    ['NE', 'SE', 'E'], ['SE', 'E', 'NE', 'N'] 
                                    which is equivalent to None, etc.
            'fgparm'    [dictionary] Sky model specification. It contains the
                        following keys and values:
                        'model'     [string] Sky model. Accepted values
                                    are 'csm' (NVSS+SUMSS point sources), 
                                    'dsm' (diffuse emission), 'asm' (both
                                    point sources and diffuse emission),
                                    'sumss' (SUMSS catalog), nvss (NVSS
                                    catalog), 'mss' (Molonglo Sky Survey), 
                                    'gleam' (GLEAM catalog), 'custom' 
                                    (user-defined catalog), 'usm' (uniform 
                                    sky model), 'mwacs' (MWACS catalog), 
                                    'HI_monopole' (global EoR), HI_cube (HI 
                                    cube from external simulations), and 
                                    'HI_fluctuations' (HI fluctuations with 
                                    the global mean signal removed). If set
                                    'HI_monopole' or 'monopole' the orientation
                                    of the baseline vector does not matter 
                                    and only unique baseline lengths will be 
                                    selected if value under 'redundant' key is
                                    set to True. 

    Output:

    Dictionary containing the following keys and values. 
    'bl'    [numpy array] Baseline vectors (unique ones or all depending on
            value in key 'redundant'). It is of shape nbl x 3 and will 
            consist of unique baselines if value under key 'redundant' was
            set to True. Otherwise, redundancy will not be checked and all
            baselines will be returned.
    'label' [numpy recarray] A unique label of each of the baselines. 
            Shape is nbl where each element is a recarray under fields 'A1'
            (first antenna label) and 'A2' (second antenna label)
    'id'    [numpy recarray] A unique identifier of each of the baselines. 
            Shape is nbl where each element is a recarray under fields 'A1'
            (first antenna id) and 'A2' (second antenna id)
    'redundancy' 
            [boolean] If the array was originally found to be made of unique 
            baselines (False) or redundant baselines were found (True). Even
            if set to False, the baselines may still be redundant because
            redundancy may never have been checked if value under key
            'redundant' was set to False
    'groups'
            [dictionary] Contains the grouping of unique baselines and the
            redundant baselines as numpy recarray under each unique baseline 
            category/flavor. It contains as keys the labels (tuple of A1, A2) 
            of unique baselines and the value under each of these keys is a 
            list of baseline labels that are redundant under that category
    'reversemap'
            [dictionary] Contains the baseline category for each baseline. 
            The keys are baseline labels as tuple and the value under each 
            key is the label of the unique baseline category that it falls 
            under. 
    'layout_info'
            [dictionary] Contains the antenna layout information with the 
            following keys and values:
            'positions' [numpy array] Antenna locations with shape nant x 3
            'labels'    [numpy array of strings] Antenna labels of size nant
            'ids'       [numpy array of strings] Antenna IDs of size nant
            'coords'    [string] Coordinate system in which antenna locations
                        are specified. Currently only returns 'ENU' for East-
                        North-Up coordinate system
    ---------------------------------------------------------------------------
    """

    try:
        inpdict
    except NameError:
        raise NameError('Input inpdict must be specified')

    if not isinstance(inpdict, dict):
        raise TypeError('Input inpdict must be a dictionary')

    if 'array' in inpdict:
        if 'redundant' in inpdict['array']:
            array_is_redundant = inpdict['array']['redundant']
        else:
            array_is_redundant = True
    else:
        raise KeyError('Key "array" not found in input inpdict')

    fg_str = inpdict['fgparm']['model']
    use_HI_monopole = False
    if fg_str == 'HI_monopole':
        use_HI_monopole = True
    antenna_file = inpdict['array']['file']
    array_layout = inpdict['array']['layout']
    minR = inpdict['array']['minR']
    maxR = inpdict['array']['maxR']
    antpos_rms_tgtplane = inpdict['array']['rms_tgtplane']
    antpos_rms_elevation = inpdict['array']['rms_elevation']
    antpos_rms_seed = inpdict['array']['seed']
    if antpos_rms_seed is None:
        antpos_rms_seed = NP.random.randint(1, high=100000)
    elif isinstance(antpos_rms_seed, (int,float)):
        antpos_rms_seed = int(NP.abs(antpos_rms_seed))
    else:
        raise ValueError('Random number seed must be a positive integer')
    minbl = inpdict['baseline']['min']
    maxbl = inpdict['baseline']['max']
    bldirection = inpdict['baseline']['direction']

    if (antenna_file is None) and (array_layout is None):
        raise ValueError('One of antenna array file or layout must be specified')
    if (antenna_file is not None) and (array_layout is not None):
        raise ValueError('Only one of antenna array file or layout must be specified')
    
    if antenna_file is not None:
        if not isinstance(antenna_file, str):
            raise TypeError('Filename containing antenna array elements must be a string')
        if inpdict['array']['filepathtype'] == 'default':
            antenna_file = prisim_path+'data/array_layouts/'+antenna_file
        
        antfile_parser = inpdict['array']['parser']
        if 'comment' in antfile_parser:
            comment = antfile_parser['comment']
            if comment is None:
                comment = '#'
            elif not isinstance(comment, str):
                raise TypeError('Comment expression must be a string')
        else:
            comment = '#'
        if 'delimiter' in antfile_parser:
            delimiter = antfile_parser['delimiter']
            if delimiter is not None:
                if not isinstance(delimiter, str):
                    raise TypeError('Delimiter expression must be a string')
            else:
                delimiter = ' '
        else:
            delimiter = ' '
    
        if 'data_start' in antfile_parser:
            data_start = antfile_parser['data_start']
            if not isinstance(data_start, int):
                raise TypeError('data_start parameter must be an integer')
        else:
            raise KeyError('data_start parameter not provided')
        if 'data_end' in antfile_parser:
            data_end = antfile_parser['data_end']
            if data_end is not None:
                if not isinstance(data_end, int):
                    raise TypeError('data_end parameter must be an integer')
        else:
            data_end = None
        if 'header_start' in antfile_parser:
            header_start = antfile_parser['header_start']
            if not isinstance(header_start, int):
                raise TypeError('header_start parameter must be an integer')
        else:
            raise KeyError('header_start parameter not provided')
    
        if 'label' not in antfile_parser:
            antfile_parser['label'] = None
        elif antfile_parser['label'] is not None:
            antfile_parser['label'] = str(antfile_parser['label'])
    
        if 'east' not in antfile_parser:
            raise KeyError('Keyword for "east" coordinates not provided')
        else:
            if not isinstance(antfile_parser['east'], str):
                raise TypeError('Keyword for "east" coordinates must be a string')
        if 'north' not in antfile_parser:
            raise KeyError('Keyword for "north" coordinates not provided')
        else:
            if not isinstance(antfile_parser['north'], str):
                raise TypeError('Keyword for "north" coordinates must be a string')
        if 'up' not in antfile_parser:
            raise KeyError('Keyword for "up" coordinates not provided')
        else:
            if not isinstance(antfile_parser['up'], str):
                raise TypeError('Keyword for "up" coordinates must be a string')
    
        try:
            ant_info = ascii.read(antenna_file, comment=comment, delimiter=delimiter, header_start=header_start, data_start=data_start, data_end=data_end, guess=False)
        except IOError:
            raise IOError('Could not open file containing antenna locations.')
    
        if (antfile_parser['east'] not in ant_info.colnames) or (antfile_parser['north'] not in ant_info.colnames) or (antfile_parser['up'] not in ant_info.colnames):
            raise KeyError('One of east, north, up coordinates incompatible with the table in antenna_file')
    
        if antfile_parser['label'] is not None:
            ant_label = ant_info[antfile_parser['label']].data.astype('str')
        else:
            ant_label = NP.arange(len(ant_info)).astype('str')
    
        east = ant_info[antfile_parser['east']].data
        north = ant_info[antfile_parser['north']].data
        elev = ant_info[antfile_parser['up']].data
    
        if (east.dtype != NP.float) or (north.dtype != NP.float) or (elev.dtype != NP.float):
            raise TypeError('Antenna locations must be of floating point type')
    
        ant_locs = NP.hstack((east.reshape(-1,1), north.reshape(-1,1), elev.reshape(-1,1)))
    else:
        if array_layout not in ['MWA-128T', 'HERA-7', 'HERA-19', 'HERA-37', 'HERA-61', 'HERA-91', 'HERA-127', 'HERA-169', 'HERA-217', 'HERA-271', 'HERA-331', 'PAPER-64', 'PAPER-112', 'HIRAX-1024', 'CHIME', 'CIRC']:
            raise ValueError('Invalid array layout specified')
    
        if array_layout == 'MWA-128T':
            ant_info = NP.loadtxt(prisim_path+'data/array_layouts/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt', skiprows=6, comments='#', usecols=(0,1,2,3))
            ant_label = ant_info[:,0].astype(int).astype(str)
            ant_locs = ant_info[:,1:]
        elif array_layout == 'HERA-7':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=7)
        elif array_layout == 'HERA-19':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=19)
        elif array_layout == 'HERA-37':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=37)
        elif array_layout == 'HERA-61':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=61)
        elif array_layout == 'HERA-91':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=91)
        elif array_layout == 'HERA-127':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=127)
        elif array_layout == 'HERA-169':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=169)
        elif array_layout == 'HERA-217':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=217)
        elif array_layout == 'HERA-271':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=271)
        elif array_layout == 'HERA-331':
            ant_locs, ant_label = hexagon_generator(14.6, n_total=331)
        elif array_layout == 'PAPER-64':
            ant_locs, ant_label = rectangle_generator([30.0, 4.0], [8, 8])
        elif array_layout == 'PAPER-112':
            ant_locs, ant_label = rectangle_generator([15.0, 4.0], [16, 7])
        elif array_layout == 'HIRAX-1024':
            ant_locs, ant_label = rectangle_generator(7.0, n_side=32)
        elif array_layout == 'CIRC':
            ant_locs, ant_label = circular_antenna_array(element_size, minR, maxR=maxR)
        ant_label = NP.asarray(ant_label)
    if ant_locs.shape[1] == 2:
        ant_locs = NP.hstack((ant_locs, NP.zeros(ant_label.size).reshape(-1,1)))
    antpos_rstate = NP.random.RandomState(antpos_rms_seed)
    deast = antpos_rms_tgtplane/NP.sqrt(2.0) * antpos_rstate.randn(ant_label.size)
    dnorth = antpos_rms_tgtplane/NP.sqrt(2.0) * antpos_rstate.randn(ant_label.size)
    dup = antpos_rms_elevation * antpos_rstate.randn(ant_label.size)
    denu = NP.hstack((deast.reshape(-1,1), dnorth.reshape(-1,1), dup.reshape(-1,1)))
    ant_locs = ant_locs + denu
    ant_locs_orig = NP.copy(ant_locs)
    ant_label_orig = NP.copy(ant_label)
    ant_id = NP.arange(ant_label.size, dtype=int)
    ant_id_orig = NP.copy(ant_id)
    layout_info = {'positions': ant_locs_orig, 'labels': ant_label_orig, 'ids': ant_id_orig, 'coords': 'ENU'}

    bl_orig, bl_label_orig, bl_id_orig = baseline_generator(ant_locs_orig, ant_label=ant_label_orig, ant_id=ant_id_orig, auto=False, conjugate=False)
    
    blo = NP.angle(bl_orig[:,0] + 1j * bl_orig[:,1], deg=True)
    neg_blo_ind = (blo < -67.5) | (blo > 112.5)
    bl_orig[neg_blo_ind,:] = -1.0 * bl_orig[neg_blo_ind,:]
    blo = NP.angle(bl_orig[:,0] + 1j * bl_orig[:,1], deg=True)
    maxlen = max(max(len(albl[0]), len(albl[1])) for albl in bl_label_orig)
    bl_label_orig = [tuple(reversed(bl_label_orig[i])) if neg_blo_ind[i] else bl_label_orig[i] for i in xrange(bl_label_orig.size)]
    bl_label_orig = NP.asarray(bl_label_orig, dtype=[('A2', '|S{0:0d}'.format(maxlen)), ('A1', '|S{0:0d}'.format(maxlen))])
    bl_id_orig = [tuple(reversed(bl_id_orig[i])) if neg_blo_ind[i] else bl_id_orig[i] for i in xrange(bl_id_orig.size)]
    bl_id_orig = NP.asarray(bl_id_orig, dtype=[('A2', int), ('A1', int)])
    bl_length_orig = NP.sqrt(NP.sum(bl_orig**2, axis=1))
    sortind_orig = NP.argsort(bl_length_orig, kind='mergesort')
    bl_orig = bl_orig[sortind_orig,:]
    blo = blo[sortind_orig]
    bl_label_orig = bl_label_orig[sortind_orig]
    bl_id_orig = bl_id_orig[sortind_orig]
    bl_length_orig = bl_length_orig[sortind_orig]
    
    bl = NP.copy(bl_orig)
    bl_label = NP.copy(bl_label_orig)
    bl_id = NP.copy(bl_id_orig)
    bl_orientation = NP.copy(blo)

    if array_is_redundant:
        bl, select_bl_ind, bl_count, allinds = uniq_baselines(bl)
    else:
        select_bl_ind = NP.arange(bl.shape[0])
        bl_count = NP.ones(bl.shape[0], dtype=int)
        allinds = select_bl_ind.reshape(-1,1).tolist()
    bl_label = bl_label[select_bl_ind]
    bl_id = bl_id[select_bl_ind]
    bl_orientation = bl_orientation[select_bl_ind]
    if NP.any(bl_count > 1):
        redundancy = True
    else:
        redundancy = False

    bl_length = NP.sqrt(NP.sum(bl**2, axis=1))
    sortind = NP.argsort(bl_length, kind='mergesort')
    bl = bl[sortind,:]
    bl_label = bl_label[sortind]
    bl_id = bl_id[sortind]
    bl_length = bl_length[sortind]
    bl_orientation = bl_orientation[sortind]

    bl_count = bl_count[sortind]
    select_bl_ind = select_bl_ind[sortind]
    allinds = [allinds[i] for i in sortind]
    
    if minbl is None:
        minbl = 0.0
    elif not isinstance(minbl, (int,float)):
        raise TypeError('Minimum baseline length must be a scalar')
    elif minbl < 0.0:
        minbl = 0.0
    
    if maxbl is None:
        maxbl = bl_length.max()
    elif not isinstance(maxbl, (int,float)):
        raise TypeError('Maximum baseline length must be a scalar')
    elif maxbl < minbl:
        maxbl = bl_length.max()
    
    min_blo = -67.5
    max_blo = 112.5
    subselect_bl_ind = NP.zeros(bl_length.size, dtype=NP.bool)
    
    if bldirection is not None:
        if isinstance(bldirection, str):
            if bldirection not in ['SE', 'E', 'NE', 'N']:
                raise ValueError('Invalid baseline direction criterion specified')
            else:
                bldirection = [bldirection]
        if isinstance(bldirection, list):
            for direction in bldirection:
                if direction in ['SE', 'E', 'NE', 'N']:
                    if direction == 'SE':
                        oind = (bl_orientation >= -67.5) & (bl_orientation < -22.5)
                        subselect_bl_ind[oind] = True
                    elif direction == 'E':
                        oind = (bl_orientation >= -22.5) & (bl_orientation < 22.5)
                        subselect_bl_ind[oind] = True
                    elif direction == 'NE':
                        oind = (bl_orientation >= 22.5) & (bl_orientation < 67.5)
                        subselect_bl_ind[oind] = True
                    else:
                        oind = (bl_orientation >= 67.5) & (bl_orientation < 112.5)
                        subselect_bl_ind[oind] = True
        else:
            raise TypeError('Baseline direction criterion must specified as string or list of strings')
    else:
        subselect_bl_ind = NP.ones(bl_length.size, dtype=NP.bool)
    
    subselect_bl_ind = subselect_bl_ind & (bl_length >= minbl) & (bl_length <= maxbl)
    bl_label = bl_label[subselect_bl_ind]
    bl_id = bl_id[subselect_bl_ind]
    bl = bl[subselect_bl_ind,:]
    bl_length = bl_length[subselect_bl_ind]
    bl_orientation = bl_orientation[subselect_bl_ind]

    bl_count = bl_count[subselect_bl_ind]
    select_bl_ind = select_bl_ind[subselect_bl_ind]
    allinds = [allinds[i] for i in range(subselect_bl_ind.size) if subselect_bl_ind[i]]
    
    if use_HI_monopole:
        bllstr = map(str, bl_length)
        uniq_bllstr, ind_uniq_bll = NP.unique(bllstr, return_index=True)
        count_uniq_bll = [bllstr.count(ubll) for ubll in uniq_bllstr]
        count_uniq_bll = NP.asarray(count_uniq_bll)
    
        bl = bl[ind_uniq_bll,:]
        bl_label = bl_label[ind_uniq_bll]
        bl_id = bl_id[ind_uniq_bll]
        bl_orientation = bl_orientation[ind_uniq_bll]
        bl_length = bl_length[ind_uniq_bll]

        bl_count = bl_count[ind_uniq_bll]
        select_bl_ind = select_bl_ind[ind_uniq_bll]
        allinds = [allinds[i] for i in ind_uniq_bll]
    
        sortind = NP.argsort(bl_length, kind='mergesort')
        bl = bl[sortind,:]
        bl_label = bl_label[sortind]
        bl_id = bl_id[sortind]
        bl_length = bl_length[sortind]
        bl_orientation = bl_orientation[sortind]
        count_uniq_bll = count_uniq_bll[sortind]

        bl_count = bl_count[sortind]
        select_bl_ind = select_bl_ind[sortind]
        allinds = [allinds[i] for i in sortind]
    
    blgroups = {}
    blgroups_reversemap = {}
    for labelind, label in enumerate(bl_label_orig[select_bl_ind]):
        if bl_count[labelind] > 0:
            blgroups[tuple(label)] = bl_label_orig[NP.asarray(allinds[labelind])]
            for lbl in bl_label_orig[NP.asarray(allinds[labelind])]:
                # blgroups_reversemap[tuple(lbl)] = tuple(label)
                blgroups_reversemap[tuple(lbl)] = NP.asarray([label], dtype=bl_label.dtype)
    
    if array_is_redundant:
        if bl_label_orig.size == bl_label.size:
            warnings.warn('No redundant baselines found. Proceeding...')

    outdict = {'bl': bl, 'id': bl_id, 'label': bl_label, 'groups': blgroups, 'reversemap': blgroups_reversemap, 'redundancy': redundancy, 'layout_info': layout_info}
    return outdict

#################################################################################

def antenna_power(skymodel, telescope_info, pointing_info, freq_scale=None):

    """
    ---------------------------------------------------------------------------
    Generate antenna power received from sky when a sky model, telescope and
    pointing parameters are provided.

    Inputs:

    skymodel  [instance of class SkyModel] Sky model specified as an instance
              of class SkyModel

    telescope_info
              [dictionary] dictionary that specifies the type of element,
              element size and orientation. It consists of the following keys
              and values:
              'latitude'    [float] latitude of the telescope site (in degrees).
                            If this key is not present, the latitude of MWA
                            (-26.701 degrees) will be assumed.
              'id'          [string] If set, will ignore the other keys and use
                            telescope details for known telescopes. Accepted
                            values are 'mwa', 'vla', 'gmrt', and 'hera',
                            'paper', 'hirax' and 'chime'
              'shape'       [string] Shape of antenna element. Accepted values
                            are 'dipole', 'delta', and 'dish'. Will be ignored
                            if key 'id' is set. 'delta' denotes a delta
                            function for the antenna element which has an
                            isotropic radiation pattern. 'delta' is the default
                            when keys 'id' and 'shape' are not set.
              'size'        [scalar] Diameter of the telescope dish (in meters)
                            if the key 'shape' is set to 'dish' or length of
                            the dipole if key 'shape' is set to 'dipole'. Will
                            be ignored if key 'shape' is set to 'delta'. Will
                            be ignored if key 'id' is set and a preset value
                            used for the diameter or dipole.
              'orientation' [list or numpy array] If key 'shape' is set to
                            dipole, it refers to the orientation of the dipole
                            element unit vector whose magnitude is specified by
                            length. If key 'shape' is set to 'dish', it refers
                            to the position on the sky to which the dish is
                            pointed. For a dipole, this unit vector must be
                            provided in the local ENU coordinate system aligned
                            with the direction cosines coordinate system or in
                            the Alt-Az coordinate system. This will be
                            used only when key 'shape' is set to 'dipole'.
                            This could be a 2-element vector (transverse
                            direction cosines) where the third (line-of-sight)
                            component is determined, or a 3-element vector
                            specifying all three direction cosines or a two-
                            element coordinate in Alt-Az system. If not provided
                            it defaults to an eastward pointing dipole. If key
                            'shape' is set to 'dish', the orientation refers
                            to the pointing center of the dish on the sky. It
                            can be provided in Alt-Az system as a two-element
                            vector or in the direction cosine coordinate
                            system as a two- or three-element vector. If not
                            set in the case of a dish element, it defaults to
                            zenith. This is not to be confused with the key
                            'pointing_center' in dictionary 'pointing_info'
                            which refers to the beamformed pointing center of
                            the array. The coordinate system is specified by
                            the key 'ocoords'
              'ocoords'     [scalar string] specifies the coordinate system
                            for key 'orientation'. Accepted values are 'altaz'
                            and 'dircos'.
              'element_locs'
                            [2- or 3-column array] Element locations that
                            constitute the tile. Each row specifies
                            location of one element in the tile. The
                            locations must be specified in local ENU
                            coordinate system. First column specifies along
                            local east, second along local north and the
                            third along local up. If only two columns are
                            specified, the third column is assumed to be
                            zeros. If 'elements_locs' is not provided, it
                            assumed to be a one-element system and not a
                            phased array as far as determination of primary
                            beam is concerned.
              'groundplane' [scalar] height of telescope element above the
                            ground plane (in meteres). Default = None will
                            denote no ground plane effects.
              'ground_modify'
                            [dictionary] contains specifications to modify
                            the analytically computed ground plane pattern. If
                            absent, the ground plane computed will not be
                            modified. If set, it may contain the following
                            keys:
                            'scale' [scalar] positive value to scale the
                                    modifying factor with. If not set, the
                                    scale factor to the modification is unity.
                            'max'   [scalar] positive value to clip the
                                    modified and scaled values to. If not set,
                                    there is no upper limit

    pointing_info
              [dictionary] Contains information about the pointing. It carries
              the following keys and values:

              'lst'    [numpy array] LST values (in degrees) for each pointing

              'pointing_coords'
                       [string scalar] Coordinate system in which the
                       pointing_center is specified. Accepted values are
                       'radec', 'hadec', 'altaz' or 'dircos'. Must be specified
                       if pointing_center is specified

              'pointing_center'
                       [numpy array] coordinates of pointing center (in the
                       coordinate system specified under key 'pointing_coords').
                       Mx2 array when value under key 'pointing_coords' is set
                       to 'radec', 'hadec' or 'altaz', or Mx3 array when the
                       value in 'pointing_coords' is set to 'dircos'. Number of
                       rows M should be equal to number of pointings and LST.
                       If only one row (M=1) is provided the same pointing
                       center in the given coordinate system will apply to all
                       pointings.

    freq_scale
              [string scalar] Units of frequency. Accepted values are 'Hz',
              'kHz', 'MHz' or 'GHz'. If None provided, default is set to 'GHz'

    Output:

    2-dimensional numpy array containing the antenna power. The rows denote
    the different pointings and columns denote the frequency spectrum obtained
    from the frequencies specified in the sky model.

    Notes:

    For each pointing the visible sky spectrum is multiplied with the power
    pattern and summed over all sky locations to obtain the received antenna
    power as a function of pointings and frequency.
    ---------------------------------------------------------------------------
    """

    try:
        skymodel, telescope_info, pointing_info
    except NameError:
        raise NameError('Sky model, telescope and pointing information must be provided')

    if not isinstance(skymodel, SM.SkyModel):
        raise TypeError('Input parameter skymodel must be an instance of class SkyModel')

    if not isinstance(telescope_info, dict):
        raise TypeError('Input parameter telescope_info must be a dictionary')

    if not isinstance(pointing_info, dict):
        raise TypeError('Input parameter pointing_info must be a dictionary')

    if 'latitude' in telescope_info:
        latitude = telescope_info['latitude']
    else:
        latitude = -26.701

    n_src = skymodel.location.shape[0]
    nchan = skymodel.frequency.size

    if 'lst' not in pointing_info:
        raise KeyError('Key "lst" not provided in input parameter pointing_info')
    else:
        lst = NP.asarray(pointing_info['lst'])
        n_lst = lst.size

    if 'pointing_center' not in pointing_info:
        pointing_center = NP.repeat(NP.asarray([90.0, 270.0]).reshape(1,-1), n_lst, axis=0)
        pointing_coords = 'altaz'
    else:
        if 'pointing_coords' not in pointing_info:
            raise KeyError('key "pointing_info" not found in input parameter pointing_info')
        pointing_coords = pointing_info['pointing_coords']

        if not isinstance(pointing_info['pointing_center'], NP.ndarray):
            raise TypeError('Value in key "pointing_center" in input parameter pointing_info must be a numpy array')
        pointing_center = pointing_info['pointing_center']
        if len(pointing_center.shape) > 2:
            raise ValueError('Value under key "pointing_center" in input parameter pointing_info cannot exceed two dimensions')
        if len(pointing_center.shape) < 2:
            pointing_center = pointing_center.reshape(1,-1)

        if (pointing_coords == 'dircos') and (pointing_center.shape[1] != 3):
            raise ValueError('Value under key "pointing_center" in input parameter pointing_info must be a 3-column array for direction cosine coordinate system')
        elif pointing_center.shape[1] != 2:
            raise ValueError('Value under key "pointing_center" in input parameter pointing_info must be a 2-column array for RA-Dec, HA-Dec and Alt-Az coordinate systems')

        n_pointings = pointing_center.shape[0]

        if (n_pointings != n_lst) and (n_pointings != 1):
            raise ValueError('Number of pointing centers and number of LST must match')
        if n_pointings < n_lst:
            pointing_center = NP.repeat(pointing_center, n_lst, axis=0)

    n_snaps = lst.size

    if pointing_coords == 'dircos':
        pointings_altaz = GEOM.dircos2altaz(pointing_center, units='degrees')
    elif pointing_coords == 'hadec':
        pointings_altaz = GEOM.hadec2altaz(pointing_center, latitude, units='degrees')
    elif pointing_coords == 'radec':
        pointings_altaz = GEOM.hadec2altaz(NP.hstack(((lst-pointing_center[:,0]).reshape(-1,1), pointing_center[:,1].reshape(-1,1))), latitude, units='degrees')
    else:
        pointings_altaz = NP.copy(pointing_center)

    if skymodel.coords == 'radec':
        lst_temp = NP.hstack((lst.reshape(-1,1),NP.zeros(n_snaps).reshape(-1,1)))  # Prepare fake LST for numpy broadcasting
        lst_temp = lst_temp.T
        lst_temp = lst_temp[NP.newaxis,:,:]
        sky_hadec = lst_temp - skymodel.location[:,:,NP.newaxis]  # Reverses sign of declination
        sky_hadec[:,1,:] *= -1   # Correct for the reversal of sign in the declination
        sky_hadec = NP.concatenate(NP.split(sky_hadec, n_snaps, axis=2), axis=0)
        sky_hadec = NP.squeeze(sky_hadec, axis=2)
        sky_altaz = GEOM.hadec2altaz(sky_hadec, latitude, units='degrees')
    elif skymodel.coords == 'hadec':
        sky_altaz = GEOM.hadec2altaz(skymodel.location, latitude, units='degrees')
    elif skymodel.coords == 'dircos':
        sky_altaz = GEOM.dircos2altaz(skymodel.location, units='degrees')
    else:
        sky_altaz = NP.copy(skymodel.location)

    sky_altaz = NP.split(sky_altaz, range(0,sky_altaz.shape[0],n_src)[1:], axis=0)  # Split sky_altaz into a list of arrays
    retval = []

    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=len(sky_altaz)).start()
    for i in xrange(len(sky_altaz)):
        pinfo = {}
        pinfo['pointing_center'] = pointings_altaz[i,:]
        pinfo['pointing_coords'] = 'altaz'
        # if 'element_locs' in telescope_info:
        #     pinfo['element_locs'] = telescope_info['element_locs']

        upper_hemisphere_ind = sky_altaz[i][:,0] >= 0.0
        upper_skymodel = skymodel.subset(indices=NP.where(upper_hemisphere_ind)[0])
        pb = PB.primary_beam_generator(sky_altaz[i][upper_hemisphere_ind,:], skymodel.frequency, telescope_info, freq_scale=freq_scale, skyunits='altaz', pointing_info=pinfo)
        spectrum = upper_skymodel.generate_spectrum()

        retval += [NP.sum(pb*spectrum, axis=0) / NP.sum(pb, axis=0)]

        progress.update(i+1)
    progress.finish()

    return NP.asarray(retval)

#################################################################################

class GainInfo(object):

    """
    ----------------------------------------------------------------------------
    Class to manage instrument gains

    Attributes:

    gaintable   [None or dictionary] If set to None, all antenna- and
                baseline-based gains will be set to unity. If returned as
                dictionary, it contains the loaded gains. It contains the
                following keys and values:
                'antenna-based'     [None or dictionary] Contains antenna-
                                    based instrument gain information. If
                                    set to None, all antenna-based gains are
                                    set to unity. If returned as dictionary,
                                    it has the following keys and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency' as specified
                                                in input axes_order
                                    'gains'     [scalar or numpy array]
                                                Complex antenna-based
                                                instrument gains. Must be
                                                of shape (nant, nchan, nts)
                                                If there is no variations in
                                                gains along an axis, then
                                                the corresponding nax may be
                                                set to 1 and the gains will
                                                be replicated along that
                                                axis using numpy array
                                                broadcasting. For example,
                                                shapes (nant,1,1), (1,1,1),
                                                (1,nchan,nts) are
                                                acceptable. If specified as
                                                a scalar, it will be
                                                replicated along all three
                                                axes, namely, 'label',
                                                'frequency' and 'time'.
                                    'label'     [None or list or numpy
                                                array] List or antenna
                                                labels that correspond to
                                                nant along the 'label' axis.
                                                If nant=1, this may be set
                                                to None, else it will be
                                                specified and will match the
                                                nant.
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nchan=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nchan
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                ntimes=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the ntimes. It must
                                                be a float and can be in
                                                seconds, hours, days, etc.
                'baseline-based'    [None or dictionary] Contains baseline-
                                    based instrument gain information. If
                                    set to None, all baseline-based gains
                                    are set to unity. If returned as
                                    dictionary, it has the following keys
                                    and values:
                                    'ordering'  [list or numpy array] Three
                                                element list of strings
                                                indicating the ordering of
                                                axes - 'time', 'label',
                                                and 'frequency' as
                                                specified in input
                                                axes_order
                                    'gains'     [scalar or numpy array]
                                                Complex baseline-based
                                                instrument gains. Must be
                                                of shape (nbl, nchan, nts)
                                                If there is no variations in
                                                gains along an axis, then
                                                the corresponding nax may be
                                                set to 1 and the gains will
                                                be replicated along that
                                                axis using numpy array
                                                broadcasting. For example,
                                                shapes (nant,1,1), (1,1,1),
                                                (1,nchan,nts) are
                                                acceptable. If specified as
                                                a scalar, it will be
                                                replicated along all three
                                                axes, namely, 'label',
                                                'frequency' and 'time'.
                                    'label'     [None or list or numpy
                                                array] List or baseline
                                                labels that correspond to
                                                nbl along the 'label' axis.
                                                If nbl=1 along the 'label'
                                                axis this may be set to
                                                None, else it will be
                                                specified and will match nbl
                                    'frequency' [None or list or numpy array]
                                                Frequency channels that
                                                correspond to the nax along
                                                the 'frequency' axis. If the
                                                nchan=1 along the 'frequency'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the nchan
                                    'time'      [None or list or numpy array]
                                                Observation times that
                                                correspond to the nax along
                                                the 'time' axis. If the
                                                ntimes=1 along the 'time'
                                                axis, this may be set to None,
                                                else it must be specified and
                                                must match the ntimes. It must
                                                be a float and can be in
                                                seconds, hours, days, etc.

    interpfuncs [dictionary] Determined in member function interpolator().
                Contains interpolation information under two keys, namely,
                'antenna-based' and 'baseline-based'. Under each of these keys
                is another dictionary with the following keys and values:
                'dims'      [numpy array of strings] Contains the axes labels
                            of the interpolated axes for antenna or baseline
                            labels. It could contain a single element ['time'],
                            of ['frequency'] indicating 1D splines along that
                            axis or contain two elements 'time' and 'frequency'
                            indicating 2D splines. 1D splines will have been
                            obtained with scipy.interpolate.interp1d while
                            2D splines obtained with scipy.interpolate.interp2d
                'interp'    [numpy recArray] Holds the interpolation functions
                            (instances of scipy.interpolate.interp1d or
                            scipy.interpolate.interp2d depending on the value
                            under 'dims' key) for each antenna or baseline
                            label. It is of size nbl. Each entry in this
                            numpy recArray has two fields, 'real' for
                            interpolation of real part and 'imag' for the
                            imaginary part. If it is a one element recarray,
                            then it applies to all antennas and baselines
                Member function interpolate_gains() uses this attribute to
                return interpolated gains

    splinefuncs [dictionary] Determined in member function splinator().
                Contains spline information under two keys, namely,
                'antenna-based' and 'baseline-based'. Under each of these keys
                is another dictionary with the following keys and values:
                'dims'      [numpy array of strings] Contains the axes labels
                            of the interpolated axes for antenna or baseline
                            labels. It could contain a single element ['time'],
                            of ['frequency'] indicating 1D splines along that
                            axis or contain two elements 'time' and 'frequency'
                            indicating 2D splines. 1D splines will have been
                            obtained with scipy.interpolate.UnivariateSpline
                            while 2D splines obtained with
                            scipy.interpolate.RectBivariateSpline
                'interp'    [numpy recArray] Holds the spline functions
                            (instances of scipy.interpolate.UnivariateSpline or
                            scipy.interpolate.RectBivariateSpline depending on
                            the value under 'dims' key) for each antenna or
                            baseline label. It is of size nbl. Each entry in
                            this numpy recArray has two fields, 'real' for
                            interpolation of real part and 'imag' for the
                            imaginary part. If it is a one element recarray,
                            then it applies to all antennas and baselines.
                Member function spline_gains() uses this attribute to return
                spline-interpolated gains

    Member functions:

    __init__()  Initialize an instance of class GainInfo from a file

    read_gaintable()
                Read gain table from file in HDF5 format and return and/or
                store as attribute

    eval_gains()
                Extract complex instrument gains for given baselines from the
                gain table

    interpolator()
                Sets up interpolation functions and stores them in the
                attribute interpfuncs. Better alternative is to use splinator()

    splinator() Sets up spline functions and stores them in the attribute
                splinefuncs. Better alternative to interpolator()

    interpolate_gains()
                Interpolate at the specified baselines for the given
                frequencies and times using attribute interpfuncs. Better
                alternative is to use spline_gains()

    spline_gains()
                Evaluate spline at the specified baselines for the given
                frequencies and times using attribute splinefuncs. Better
                alternative to interpolate_gains()

    nearest_gains()
                Extract complex instrument gains for given baselines from the
                gain table determined by nearest neighbor logic

    write_gaintable()
                Write gain table with specified axes ordering to external file
                in HDF5 format
    -----------------------------------------------------------------------------
    """

    def __init__(self, init_file=None, axes_order=None):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class GainInfo from a file

        Attributes initialized are:
        gaintable, interpfuncs, splinefuncs

        Read docstring of class GainInfo for details on these attributes

        Keyword Inputs:

        gainsfile   [string] Filename including the full path that contains the
                    instrument gains. It must be in HDF5 format. It must contain
                    the following structure:
                    'antenna-based'     [dictionary] Contains antenna-based
                                        instrument gain information. It has the
                                        following keys and values:
                                        'ordering'  [list or numpy array] Three
                                                    element list of strings
                                                    indicating the ordering of
                                                    axes - 'time', 'label',
                                                    and 'frequency'. Must be
                                                    specified (no defaults)
                                        'gains'     [scalar or numpy array]
                                                    Complex antenna-based
                                                    instrument gains. Must be
                                                    of shape (nax1, nax2, nax3)
                                                    where ax1, ax2 and ax3 are
                                                    specified by the axes
                                                    ordering under key
                                                    'ordering'. If there is no
                                                    variations in gains along an
                                                    axis, then the corresponding
                                                    nax may be set to 1 and the
                                                    gains will be replicated
                                                    along that axis using numpy
                                                    array broadcasting. For
                                                    example, shapes (nax1,1,1),
                                                    (1,1,1), (1,nax2,nax3) are
                                                    acceptable. If specified as
                                                    a scalar, it will be
                                                    replicated along all three
                                                    axes, namely, 'label',
                                                    'frequency' and 'time'.
                                        'label'     [None or list or numpy array]
                                                    List or antenna labels that
                                                    correspond to the nax along
                                                    the 'label' axis. If the
                                                    nax=1 along the 'label' axis,
                                                    this may be set to None, else
                                                    it must be specified and must
                                                    match the nax.
                                        'frequency' [None or list or numpy
                                                    array] Frequency channels
                                                    that correspond to the
                                                    nax along the 'frequency'
                                                    axis. If the nax=1 along the
                                                    'frequency' axis, this may
                                                    be set to None, else it must
                                                    be specified and must match
                                                    the nax.
                                        'time'      [None or list or numpy
                                                    array] Observation times
                                                    that correspond to the nax
                                                    along the 'time' axis. If
                                                    the nax=1 along the 'time'
                                                    axis, this may be set to
                                                    None, else it must be
                                                    specified and must match the
                                                    nax. It must be a float and
                                                    can be in seconds, hours,
                                                    days, etc.
                    'baseline-based'    [dictionary] Contains baseline-based
                                        instrument gain information. It has the
                                        following keys and values:
                                        'ordering'  [list or numpy array] Three
                                                    element list of strings
                                                    indicating the ordering of
                                                    axes - 'time', 'label',
                                                    and 'frequency'. Must be
                                                    specified (no defaults)
                                        'gains'     [scalar or numpy array]
                                                    Complex baseline-based
                                                    instrument gains. Must be
                                                    of shape (nax1, nax2, nax3)
                                                    where ax1, ax2 and ax3 are
                                                    specified by the axes
                                                    ordering under key
                                                    'ordering'. If there is no
                                                    variations in gains along an
                                                    axis, then the corresponding
                                                    nax may be set to 1 and the
                                                    gains will be replicated
                                                    along that axis using numpy
                                                    array broadcasting. For
                                                    example, shapes (nax1,1,1),
                                                    (1,1,1), (1,nax2,nax3) are
                                                    acceptable. If specified as
                                                    a scalar, it will be
                                                    replicated along all three
                                                    axes, namely, 'label',
                                                    'frequency' and 'time'.
                                        'label'     [None or list or numpy
                                                    array] List of baseline
                                                    labels that correspond to
                                                    the nax along the 'label'
                                                    axis. If the nax=1 along the
                                                    'label' axis this may be set
                                                    to None, else it must be
                                                    specified and must match the
                                                    nax.
                                        'frequency' [None or list or numpy
                                                    array] Frequency channels
                                                    that correspond to the
                                                    nax along the 'frequency'
                                                    axis. If the nax=1 along the
                                                    'frequency' axis, this may
                                                    be set to None, else it must
                                                    be specified and must match
                                                    the nax.
                                        'time'      [None or list or numpy
                                                    array] Observation times
                                                    that correspond to the nax
                                                    along the 'time' axis. If
                                                    the nax=1 along the 'time'
                                                    axis, this may be set to
                                                    None, else it must be
                                                    specified and must match the
                                                    nax. It must be a float and
                                                    can be in seconds, hours,
                                                    days, etc.

        axes_order  [None or list or numpy array] The gaintable which is read is
                    stored in this axes ordering. If set to None, it will store
                    in this order ['label', 'frequency', 'time']
        ------------------------------------------------------------------------
        """

        self.gaintable = None
        self.interpfuncs = {key: None for key in ['antenna-based', 'baseline-based']}
        self.splinefuncs = {key: None for key in ['antenna-based', 'baseline-based']}
        if init_file is not None:
            self.gaintable = self.read_gaintable(init_file, axes_order=axes_order, action='return')
        self.interpolator()
        self.splinator(smoothness=None)

    #############################################################################

    def read_gaintable(self, gainsfile, axes_order=None, action='return'):

        """
        ------------------------------------------------------------------------
        Read gain table from file in HDF5 format and return and/or store as
        attribute

        Input:

        gainsfile   [string] Filename including the full path that contains the
                    instrument gains. It must be in HDF5 format. It must contain
                    the following structure:
                    'antenna-based'     [dictionary] Contains antenna-based
                                        instrument gain information. It has the
                                        following keys and values:
                                        'ordering'  [list or numpy array] Three
                                                    element list of strings
                                                    indicating the ordering of
                                                    axes - 'time', 'label',
                                                    and 'frequency'. Must be
                                                    specified (no defaults)
                                        'gains'     [scalar or numpy array]
                                                    Complex antenna-based
                                                    instrument gains. Must be
                                                    of shape (nax1, nax2, nax3)
                                                    where ax1, ax2 and ax3 are
                                                    specified by the axes
                                                    ordering under key
                                                    'ordering'. If there is no
                                                    variations in gains along an
                                                    axis, then the corresponding
                                                    nax may be set to 1 and the
                                                    gains will be replicated
                                                    along that axis using numpy
                                                    array broadcasting. For
                                                    example, shapes (nax1,1,1),
                                                    (1,1,1), (1,nax2,nax3) are
                                                    acceptable. If specified as
                                                    a scalar, it will be
                                                    replicated along all three
                                                    axes, namely, 'label',
                                                    'frequency' and 'time'.
                                        'label'     [None or list or numpy
                                                    array] List or antenna
                                                    labels that correspond to
                                                    the nax along the 'label'
                                                    axis. If the nax=1 along the
                                                    'label' axis, this may be
                                                    set to None, else it must be
                                                    specified and must match the
                                                    nax.
                                        'frequency' [None or list or numpy
                                                    array] Frequency channels
                                                    that correspond to the nax
                                                    along the 'frequency' axis.
                                                    If the nax=1 along the
                                                    'frequency' axis, this may
                                                    be set to None, else it must
                                                    be specified and must match
                                                    the nax.
                                        'time'      [None or list or numpy
                                                    array] Observation times
                                                    that correspond to the nax
                                                    along the 'time' axis. If
                                                    the nax=1 along the 'time'
                                                    axis, this may be set to
                                                    None, else it must be
                                                    specified and must match the
                                                    nax. It must be a float and
                                                    can be in seconds, hours,
                                                    days, etc.
                    'baseline-based'    [dictionary] Contains baseline-based
                                        instrument gain information. It has the
                                        following keys and values:
                                        'ordering'  [list or numpy array] Three
                                                    element list of strings
                                                    indicating the ordering of
                                                    axes - 'time', 'label',
                                                    and 'frequency'. Must be
                                                    specified (no defaults)
                                        'gains'     [scalar or numpy array]
                                                    Complex baseline-based
                                                    instrument gains. Must be
                                                    of shape (nax1, nax2, nax3)
                                                    where ax1, ax2 and ax3 are
                                                    specified by the axes
                                                    ordering under key
                                                    'ordering'. If there is no
                                                    variations in gains along an
                                                    axis, then the corresponding
                                                    nax may be set to 1 and the
                                                    gains will be replicated
                                                    along that axis using numpy
                                                    array broadcasting. For
                                                    example, shapes (nax1,1,1),
                                                    (1,1,1), (1,nax2,nax3) are
                                                    acceptable. If specified as
                                                    a scalar, it will be
                                                    replicated along all three
                                                    axes, namely, 'label',
                                                    'frequency' and 'time'.
                                        'label'     [None or list or numpy
                                                    array] List of baseline
                                                    labels that correspond to
                                                    the nax along the 'label'
                                                    axis. If the nax=1 along the
                                                    'label' axis this may be set
                                                    to None, else it must be
                                                    specified and must match the
                                                    nax.
                                        'frequency' [None or list or numpy
                                                    array] Frequency channels
                                                    that correspond to the nax
                                                    along the 'frequency' axis.
                                                    If the nax=1 along the
                                                    'frequency' axis, this may
                                                    be set to None, else it must
                                                    be specified and must match
                                                    the nax.
                                        'time'      [None or list or numpy
                                                    array] Observation times
                                                    that correspond to the nax
                                                    along the 'time' axis. If
                                                    the nax=1 along the 'time'
                                                    axis, this may be set to
                                                    None, else it must be
                                                    specified and must match the
                                                    nax. It must be a float and
                                                    can be in seconds, hours,
                                                    days, etc.

        axes_order  [None or list or numpy array] The gaintable which is read is
                    stored in this axes ordering. If set to None, it will store
                    in this order ['label', 'frequency', 'time']

        action      [string] If set to 'store' (default), the gain table will
                    be stored as attribute in addition to being returned. If set
                    to 'return' the gain table will be returned.

        Output:

        gaintable   [None or dictionary] If set to None, all antenna- and
                    baseline-based gains will be set to unity. If returned as
                    dictionary, it contains the loaded gains. It contains the
                    following keys and values:
                    'antenna-based'     [None or dictionary] Contains antenna-
                                        based instrument gain information. If
                                        set to None, all antenna-based gains are
                                        set to unity. If returned as dictionary,
                                        it has the following keys and values:
                                        'ordering'  [list or numpy array] Three
                                                    element list of strings
                                                    indicating the ordering of
                                                    axes - 'time', 'label',
                                                    and 'frequency' as specified
                                                    in input axes_order
                                        'gains'     [scalar or numpy array]
                                                    Complex antenna-based
                                                    instrument gains. Must be
                                                    of shape (nant, nchan, nts)
                                                    If there is no variations in
                                                    gains along an axis, then
                                                    the corresponding nax may be
                                                    set to 1 and the gains will
                                                    be replicated along that
                                                    axis using numpy array
                                                    broadcasting. For example,
                                                    shapes (nant,1,1), (1,1,1),
                                                    (1,nchan,nts) are
                                                    acceptable. If specified as
                                                    a scalar, it will be
                                                    replicated along all three
                                                    axes, namely, 'label',
                                                    'frequency' and 'time'.
                                        'label'     [None or list or numpy
                                                    array] List or antenna
                                                    labels that correspond to
                                                    nant along the 'label' axis.
                                                    If nant=1, this may be set
                                                    to None, else it will be
                                                    specified and will match the
                                                    nant.
                                        'frequency' [None or list or numpy
                                                    array] Frequency channels
                                                    that correspond to the nax
                                                    along the 'frequency' axis.
                                                    If the nchan=1 along the
                                                    'frequency' axis, this may
                                                    be set to None, else it must
                                                    be specified and must match
                                                    the nchan.
                                        'time'      [None or list or numpy
                                                    array] Observation times
                                                    that correspond to the nax
                                                    along the 'time' axis. If
                                                    the ntimes=1 along the
                                                    'time' axis, this may be set
                                                    to None, else it must be
                                                    specified and must match the
                                                    ntimes. It will be a float
                                                    and in same units as given
                                                    in input
                    'baseline-based'    [None or dictionary] Contains baseline-
                                        based instrument gain information. If
                                        set to None, all baseline-based gains
                                        are set to unity. If returned as
                                        dictionary, it has the following keys
                                        and values:
                                        'ordering'  [list or numpy array] Three
                                                    element list of strings
                                                    indicating the ordering of
                                                    axes - 'time', 'label',
                                                    and 'frequency' as
                                                    specified in input
                                                    axes_order
                                        'gains'     [scalar or numpy array]
                                                    Complex baseline-based
                                                    instrument gains. Must be
                                                    of shape (nbl, nchan, nts)
                                                    If there is no variations in
                                                    gains along an axis, then
                                                    the corresponding nax may be
                                                    set to 1 and the gains will
                                                    be replicated along that
                                                    axis using numpy array
                                                    broadcasting. For example,
                                                    shapes (nant,1,1), (1,1,1),
                                                    (1,nchan,nts) are
                                                    acceptable. If specified as
                                                    a scalar, it will be
                                                    replicated along all three
                                                    axes, namely, 'label',
                                                    'frequency' and 'time'.
                                        'label'     [None or list or numpy
                                                    array] List or baseline
                                                    labels that correspond to
                                                    nbl along the 'label' axis.
                                                    If nbl=1 along the 'label'
                                                    axis this may be set to
                                                    None, else it will be
                                                    specified and will match nbl
                                        'frequency' [None or list or numpy
                                                    array] Frequency channels
                                                    that correspond to the nax
                                                    along the 'frequency' axis.
                                                    If the nchan=1 along the
                                                    'frequency' axis, this may
                                                    be set to None, else it must
                                                    be specified and must match
                                                    the nchan.
                                        'time'      [None or list or numpy
                                                    array] Observation times
                                                    that correspond to the nax
                                                    along the 'time' axis. If
                                                    the ntimes=1 along the
                                                    'time' axis, this may be set
                                                    to None, else it must be
                                                    specified and must match the
                                                    ntimes. It will be a float
                                                    and in same units as given
                                                    in input
        ------------------------------------------------------------------------
        """

        if not isinstance(action, str):
            return TypeError('Input parameter action must be a string')
        action = action.lower()
        if action not in ['store', 'return']:
            raise ValueError('Invalid value specified for input parameter action')

        gaintable = read_gaintable(gainsfile, axes_order=axes_order)
        if action == 'store':
            self.gaintable = gaintable
        return gaintable

    #############################################################################

    def interpolator(self, kind='linear'):

        """
        ------------------------------------------------------------------------
        Sets up interpolation functions and stores them in the attribute
        interpfuncs. Better alternative is to use splinator()

        Inputs:

        kind    [string] Type of interpolation. Accepted values are
                'linear' (default), 'cubic' or 'quintic'. See documentation
                of scipy.interpolate.interp1d and scipy.interpolate.interp2d
                for details
        ------------------------------------------------------------------------
        """

        kind = kind.lower()
        if kind not in ['linear', 'cubic', 'quintic']:
            raise ValueError('Specified kind of interpolation invalid')

        if self.gaintable is not None:
            for gainkey in self.gaintable:
                if self.gaintable[gainkey] is not None:
                    self.interpfuncs[gainkey] = None
                    if self.gaintable[gainkey]['gains'] is not None:
                        if isinstance(self.gaintable[gainkey]['gains'], NP.ndarray):
                            if self.gaintable[gainkey]['gains'].ndim != 3:
                                raise ValueError('Gains must be a 3D numpy array')
                            # if self.gaintable[gainkey]['gains'].size > 1:
                            if (self.gaintable[gainkey]['gains'].shape[self.gaintable[gainkey]['ordering'].index('frequency')] > 1) or (self.gaintable[gainkey]['gains'].shape[self.gaintable[gainkey]['ordering'].index('time')] > 1):
                                temp_axes_order = ['label', 'frequency', 'time']
                                inp_order = self.gaintable[gainkey]['ordering']
                                temp_transpose_order = NMO.find_list_in_list(inp_order, temp_axes_order)
                                if NP.all(inp_order == temp_axes_order):
                                    gains = NP.copy(self.gaintable[gainkey]['gains'])
                                else:
                                    gains = NP.transpose(NP.copy(self.gaintable[gainkey]['gains']), axes=temp_transpose_order)
                                dims = []
                                for ax in NP.arange(1,3):
                                    if gains.shape[ax] > 1:
                                        dims += [temp_axes_order[ax]]
                                dims = NP.asarray(dims)
                                interpf = []
                                for labelind in xrange(gains.shape[0]):
                                    if dims.size == 1:
                                        interpf_real = interpolate.interp1d(self.gaintable[gainkey][dims[0]], gains[labelind,:,:].real.ravel(), kind=kind, bounds_error=True)
                                        interpf_imag = interpolate.interp1d(self.gaintable[gainkey][dims[0]], gains[labelind,:,:].imag.ravel(), kind=kind, bounds_error=True)
                                    else:
                                        interpf_real = interpolate.interp2d(self.gaintable[gainkey]['time'], self.gaintable[gainkey]['frequency'], gains[labelind,:,:].real, kind=kind, bounds_error=True)
                                        interpf_imag = interpolate.interp2d(self.gaintable[gainkey]['time'], self.gaintable[gainkey]['frequency'], gains[labelind,:,:].imag, kind=kind, bounds_error=True)
                                    interpf += [(copy.copy(interpf_real), copy.copy(interpf_imag))]
                                self.interpfuncs[gainkey] = {'interp': NP.asarray(interpf, dtype=[('real', NP.object), ('imag', NP.object)]), 'dims': dims}

    ############################################################################

    def splinator(self, smoothness=None):

        """
        -----------------------------------------------------------------------
        Sets up spline functions and stores them in the attribute splinefuncs.
        Better alternative to interpolator()

        Inputs:

        smoothness  [integer or float] Smoothness of spline interpolation. Must
                    be positive. If set to None (default), it will set equal to
                    the number of samples using which the spline functions are
                    estimated. Read documentation of
                    scipy.interpolate.UnivariateSpline and
                    scipy.interpolate.RectBivariateSpline for more details
        -----------------------------------------------------------------------
        """
        if smoothness is not None:
            if not isinstance(smoothness, (int,float)):
                raise TypeError('Input smoothness must be a scalar')
            if smoothness <= 0.0:
                raise ValueError('Input smoothness must be a positive number')

        if self.gaintable is not None:
            for gainkey in self.gaintable:
                if self.gaintable[gainkey] is not None:
                    self.splinefuncs[gainkey] = None
                    if self.gaintable[gainkey]['gains'] is not None:
                        if isinstance(self.gaintable[gainkey]['gains'], NP.ndarray):
                            if self.gaintable[gainkey]['gains'].ndim != 3:
                                raise ValueError('Gains must be a 3D numpy array')
                            # if self.gaintable[gainkey]['gains'].size > 1:
                            if (self.gaintable[gainkey]['gains'].shape[self.gaintable[gainkey]['ordering'].index('frequency')] > 1) or (self.gaintable[gainkey]['gains'].shape[self.gaintable[gainkey]['ordering'].index('time')] > 1):
                                temp_axes_order = ['label', 'frequency', 'time']
                                inp_order = self.gaintable[gainkey]['ordering']
                                temp_transpose_order = NMO.find_list_in_list(inp_order, temp_axes_order)
                                if NP.all(inp_order == temp_axes_order):
                                    gains = NP.copy(self.gaintable[gainkey]['gains'])
                                else:
                                    gains = NP.transpose(NP.copy(self.gaintable[gainkey]['gains']), axes=temp_transpose_order)
                                dims = []
                                for ax in NP.arange(1,3):
                                    if gains.shape[ax] > 1:
                                        dims += [temp_axes_order[ax]]
                                dims = NP.asarray(dims)
                                interpf = []
                                for labelind in xrange(gains.shape[0]):
                                    if dims.size == 1:
                                        if smoothness is None:
                                            smoothness = self.gaintable[gainkey][dims[0]].size
                                        interpf_real = interpolate.UnivariateSpline(self.gaintable[gainkey][dims[0]], gains[labelind,:,:].real.ravel(), s=smoothness, ext='raise')
                                        interpf_imag = interpolate.UnivariateSpline(self.gaintable[gainkey][dims[0]], gains[labelind,:,:].imag.ravel(), s=smoothness, ext='raise')
                                    else:
                                        if smoothness is None:
                                            smoothness = gains.shape[1]*gains.shape[2]
                                        interpf_real = interpolate.RectBivariateSpline(self.gaintable[gainkey]['time'], self.gaintable[gainkey]['frequency'], gains[labelind,:,:].real.T, bbox=[self.gaintable[gainkey]['time'].min(), self.gaintable[gainkey]['time'].max(), self.gaintable[gainkey]['frequency'].min(), self.gaintable[gainkey]['frequency'].max()], s=smoothness)
                                        interpf_imag = interpolate.RectBivariateSpline(self.gaintable[gainkey]['time'], self.gaintable[gainkey]['frequency'], gains[labelind,:,:].imag.T, bbox=[self.gaintable[gainkey]['time'].min(), self.gaintable[gainkey]['time'].max(), self.gaintable[gainkey]['frequency'].min(), self.gaintable[gainkey]['frequency'].max()], s=smoothness)
                                    interpf += [(copy.copy(interpf_real), copy.copy(interpf_imag))]
                                self.splinefuncs[gainkey] = {'interp': NP.asarray(interpf, dtype=[('real', NP.object), ('imag', NP.object)]), 'dims': dims}

    #############################################################################

    def interpolate_gains(self, bl_labels, freqs=None, times=None,
                          axes_order=None):

        """
        ------------------------------------------------------------------------
        Interpolate at the specified baselines for the given frequencies and
        times using attribute interpfuncs. Better alternative is to use
        spline_gains()

        Inputs:

        bl_labels   [Numpy structured array tuples] Labels of antennas in the
                    pair used to produce the baseline vector under fields 'A2'
                    and 'A1' for second and first antenna respectively. The
                    baseline vector is obtained by position of antennas under
                    'A2' minus position of antennas under 'A1'. The array is of
                    size nbl

        freqs       [None or numpy array] Array of frequencies at which the
                    gains are to be interpolated using the attribute
                    interpfuncs. If set to None (default), all frequencies in
                    the gaintable are assumed. The specified frequencies must
                    always lie within the range which was used in creating the
                    interpolation functions, otherwise an exception will be
                    raised. The array is of size nchan

        times       [None or numpy array] Array of times at which the gains
                    are to be interpolated using the attribute interpfuncs. If
                    set to None (default), all times in the gaintable are
                    assumed. The specified times must always lie within the
                    range which was used in creating the interpolation
                    functions, otherwise an exception will be raised. The
                    array is of size nts

        axes_order  [None or list or numpy array] Axes ordering for extracted
                    gains. It must contain the three elements 'label',
                    'frequency', and 'time'. If set to None, it will be
                    returned in the same order as in the input gaintable.

        Outputs:

        [numpy array] Complex gains of shape nbl x nchan x nts for the
        specified baselines, frequencies and times.
        ------------------------------------------------------------------------
        """
        try:
            bl_labels
        except NameError:
            raise NameError('Input bl_labels must be specified')

        blgains = NP.asarray(1.0).reshape(1,1,1)
        if self.gaintable is not None:
            a1_labels = bl_labels['A1']
            a2_labels = bl_labels['A2']
            for key in ['antenna-based', 'baseline-based']:
                if self.interpfuncs[key] is not None:
                    labels = self.gaintable[key]['label']
                    if freqs is None:
                        if self.gaintable[key]['frequency'] is not None:
                            freqs = self.gaintable[key]['frequency']
                    elif isinstance(freqs, (int,list,NP.ndarray)):
                        freqs = NP.asarray(freqs).ravel()
                    else:
                        raise TypeError('Input freqs must be a scalar, list or numpy array')

                    if times is None:
                        if self.gaintable[key]['time'] is not None:
                            times = self.gaintable[key]['time']
                    elif isinstance(times, (int,list,NP.ndarray)):
                        times = NP.asarray(times).ravel()
                    else:
                        raise TypeError('Input times must be a scalar, list or numpy array')

                    if self.gaintable[key]['frequency'] is not None:
                        ib_freq_index = NP.logical_and(freqs <= NP.amax(self.gaintable[key]['frequency']), freqs >= NP.amin(self.gaintable[key]['frequency']))
                        oobl_freq_index = freqs < NP.amin(self.gaintable[key]['frequency'])
                        oobr_freq_index = freqs > NP.amax(self.gaintable[key]['frequency'])
                        oob_freq_index = NP.logical_not(ib_freq_index)
                        if NP.any(oob_freq_index):
                            raise ValueError('One or more of the frequencies outside interpolation range')
                    else:
                        if freqs is not None:
                            ib_freq_index = NP.ones(freqs.size, dtype=NP.bool)
                            oob_freq_index = NP.zeros(freqs.size, dtype=NP.bool)
                            oobl_freq_index = NP.zeros(freqs.size, dtype=NP.bool)
                            oobr_freq_index = NP.zeros(freqs.size, dtype=NP.bool)
                        else:
                            ib_freq_index = None
                            oob_freq_index = None

                    if self.gaintable[key]['time'] is not None:
                        ib_time_index = NP.logical_and(times <= NP.amax(self.gaintable[key]['time']), times >= NP.amin(self.gaintable[key]['time']))
                        oobl_time_index = times < NP.amin(self.gaintable[key]['time'])
                        oobr_time_index = times > NP.amax(self.gaintable[key]['time'])
                        oob_time_index = NP.logical_not(ib_time_index)
                        if NP.any(oob_time_index):
                            raise ValueError('One or more of the times outside interpolation range')
                    else:
                        if times is not None:
                            ib_time_index = NP.ones(times.size, dtype=NP.bool)
                            oob_time_index = NP.zeros(times.size, dtype=NP.bool)
                            oobl_time_index = NP.zeros(times.size, dtype=NP.bool)
                            oobr_time_index = NP.zeros(times.size, dtype=NP.bool)
                        else:
                            ib_time_index = None
                            oob_time_index = None

                    if isinstance(self.interpfuncs[key], dict):
                        if 'dims' not in self.interpfuncs[key]:
                            raise KeyError('Key "dims" not found in attribute interpfuncs[{0}]'.format(key))
                        if not isinstance(self.interpfuncs[key]['dims'], NP.ndarray):
                            raise TypeError('Key "dims" in attribute interpfuncs[{0}] must contain a numpy array'.format(key))

                        if self.interpfuncs[key]['dims'].size == 1:
                            if self.interpfuncs[key]['dims'][0] == 'time':
                                ntimes = ib_time_index.size
                                if freqs is None:
                                    nchan = 1
                                else:
                                    nchan = ib_freq_index.size
                                inp = times[ib_time_index]
                            else:
                                nchan = ib_freq_index.size
                                if times is None:
                                    ntimes = 1
                                else:
                                    ntimes = ib_time_index.size
                                inp = freqs[ib_freq_index]
                        else:
                            inp_times = times[ib_time_index]
                            inp_freqs = freqs[ib_freq_index]
                            ntimes = ib_time_index.size
                            nchan = ib_freq_index.size

                        if key == 'antenna-based':
                            ind1 = NMO.find_list_in_list(labels, a1_labels)
                            ind2 = NMO.find_list_in_list(labels, a2_labels)
                            if NP.sum(ind1.mask) > 0:
                                raise IndexError('Some antenna gains could not be found')
                            if NP.sum(ind2.mask) > 0:
                                raise IndexError('Some antenna gains could not be found')
                            g1_conj = None
                            g2 = None
                            for i in xrange(ind1.size):
                                if self.interpfuncs[key]['dims'].size == 1:
                                    if g1_conj is None:
                                        g1_conj = (self.interpfuncs[key]['interp']['real'][ind1[i]](inp) - 1j * self.interpfuncs[key]['interp']['imag'][ind1[i]](inp)).reshape(1,nchan,ntimes)
                                        g2 = (self.interpfuncs[key]['interp']['real'][ind2[i]](inp) + 1j * self.interpfuncs[key]['interp']['imag'][ind2[i]](inp)).reshape(1,nchan,ntimes)
                                    else:
                                        g1_conj = NP.concatenate((g1_conj, (self.interpfuncs[key]['interp']['real'][ind1[i]](inp) - 1j * self.interpfuncs[key]['interp']['imag'][ind1[i]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                        g2 = NP.concatenate((g2, (self.interpfuncs[key]['interp']['real'][ind2[i]](inp) + 1j * self.interpfuncs[key]['interp']['imag'][ind2[i]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                else:
                                    if g1_conj is None:
                                        g1_conj = (self.interpfuncs[key]['interp']['real'][ind1[i]](inp_times,inp_freqs) - 1j * self.interpfuncs[key]['interp']['imag'][ind1[i]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)
                                        g2 = (self.interpfuncs[key]['interp']['real'][ind2[i]](inp_times,inp_freqs) + 1j * self.interpfuncs[key]['interp']['imag'][ind2[i]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)
                                    else:
                                        g1_conj = NP.concatenate((g1_conj, (self.interpfuncs[key]['interp']['real'][ind1[i]](inp_times,inp_freqs) - 1j * self.interpfuncs[key]['interp']['imag'][ind1[i]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)), axis=0)
                                        g2 = NP.concatenate((g2, (self.interpfuncs[key]['interp']['real'][ind2[i]](inp_times,inp_freqs) + 1j * self.interpfuncs[key]['interp']['imag'][ind2[i]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)), axis=0)

                            blgains = blgains * g1_conj * g2 * NP.ones((1,nchan,ntimes), dtype=NP.complex)
                        else:
                            g12 = None
                            for labelind,label in enumerate(bl_labels):
                                if label in labels:
                                    ind = NP.where(self.gaintable[key]['label'] == label)[0]
                                    if self.interpfuncs[key]['dims'].size == 1:
                                        if g12 is None:
                                            g12 = (self.interpfuncs[key]['interp']['real'][ind[0]](inp) + 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.interpfuncs[key]['interp']['real'][ind[0]](inp) + 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                    else:
                                        if g12 is None:
                                            g12 = (self.interpfuncs[key]['interp']['real'][ind[0]](inp_times,inp_freqs) + 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.interpfuncs[key]['interp']['real'][ind[0]](inp_times,inp_freqs) + 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)), axis=0)
                                elif NP.asarray([tuple(reversed(label))], dtype=bl_labels.dtype)[0] in labels:
                                    ind = NP.where(labels == NP.asarray([tuple(reversed(label))], dtype=bl_labels.dtype)[0])[0]
                                    if self.interpfuncs[key]['dims'].size == 1:
                                        if g12 is None:
                                            g12 = (self.interpfuncs[key]['interp']['real'][ind[0]](inp) - 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.interpfuncs[key]['interp']['real'][ind[0]](inp) - 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                    else:
                                        if g12 is None:
                                            g12 = (self.interpfuncs[key]['interp']['real'][ind[0]](inp_times,inp_freqs) - 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.interpfuncs[key]['interp']['real'][ind[0]](inp_times,inp_freqs) - 1j * self.interpfuncs[key]['interp']['imag'][ind[0]](inp_times,inp_freqs)).reshape(1,nchan,ntimes)), axis=0)
                                else:
                                    if g12 is None:
                                        g12 = NP.ones((1,nchan,ntimes), dtype=NP.complex)
                                    else:
                                        g12 = NP.concatenate((g12, NP.ones((1,nchan,ntimes), dtype=NP.complex)), axis=0)
                            blgains = blgains * g12 * NP.ones((1,nchan,ntimes), dtype=NP.complex)

        interp_axes_order = ['label', 'frequency', 'time']
        if axes_order is None:
            axes_order = self.gaintable['antenna-based']['ordering']
        elif not isinstance(axes_order, (list, NP.ndarray)):
            raise TypeError('axes_order must be a list')
        else:
            if len(axes_order) != 3:
                raise ValueError('axes_order must be a three element list')
            for orderkey in ['label', 'frequency', 'time']:
                if orderkey not in axes_order:
                    raise ValueError('axes_order does not contain key "{0}"'.format(orderkey))

        transpose_order = NMO.find_list_in_list(interp_axes_order, axes_order)
        blgains = NP.transpose(blgains, axes=transpose_order)

        return blgains

    #############################################################################

    def spline_gains(self, bl_labels, freqs=None, times=None, axes_order=None):

        """
        ------------------------------------------------------------------------
        Evaluate spline at the specified baselines for the given frequencies and
        times using attribute splinefuncs. Better alternative to
        interpolate_gains()

        Inputs:

        bl_labels   [Numpy structured array tuples] Labels of antennas in the
                    pair used to produce the baseline vector under fields 'A2'
                    and 'A1' for second and first antenna respectively. The
                    baseline vector is obtained by position of antennas under
                    'A2' minus position of antennas under 'A1'. The array is of
                    size nbl

        freqs       [None or numpy array] Array of frequencies at which the
                    gains are to be interpolated using the attribute
                    splinefuncs. If set to None (default), all frequencies in
                    the gaintable are assumed. The specified frequencies must
                    always lie within the range which was used in creating the
                    interpolation functions, otherwise an exception will be
                    raised. The array is of size nchan

        times       [None or numpy array] Array of times at which the gains
                    are to be interpolated using the attribute splinefuncs. If
                    set to None (default), all times in the gaintable are
                    assumed. The specified times must always lie within the
                    range which was used in creating the interpolation
                    functions, otherwise an exception will be raised. The array
                    is of size nts

        axes_order  [None or list or numpy array] Axes ordering for extracted
                    gains. It must contain the three elements 'label',
                    'frequency', and 'time'. If set to None, it will be
                    returned in the same order as in the input gaintable.

        Outputs:

        [numpy array] Complex gains of shape nbl x nchan x nts for the specified
        baselines, frequencies and times.
        ---------------------------------------------------------------------------
        """

        try:
            bl_labels
        except NameError:
            raise NameError('Input bl_labels must be specified')

        blgains = NP.asarray(1.0).reshape(1,1,1)
        if self.gaintable is not None:
            a1_labels = bl_labels['A1']
            a2_labels = bl_labels['A2']
            for key in ['antenna-based', 'baseline-based']:
                if self.splinefuncs[key] is not None:
                    labels = self.gaintable[key]['label']
                    if freqs is None:
                        if self.gaintable[key]['frequency'] is not None:
                            freqs = self.gaintable[key]['frequency']
                    elif isinstance(freqs, (int,list,NP.ndarray)):
                        freqs = NP.asarray(freqs).ravel()
                    else:
                        raise TypeError('Input freqs must be a scalar, list or numpy array')

                    if times is None:
                        if self.gaintable[key]['time'] is not None:
                            times = self.gaintable[key]['time']
                    elif isinstance(times, (int,list,NP.ndarray)):
                        times = NP.asarray(times).ravel()
                    else:
                        raise TypeError('Input times must be a scalar, list or numpy array')

                    if self.gaintable[key]['frequency'] is not None:
                        ib_freq_index = NP.logical_and(freqs <= NP.amax(self.gaintable[key]['frequency']), freqs >= NP.amin(self.gaintable[key]['frequency']))
                        oobl_freq_index = freqs < NP.amin(self.gaintable[key]['frequency'])
                        oobr_freq_index = freqs > NP.amax(self.gaintable[key]['frequency'])
                        oob_freq_index = NP.logical_not(ib_freq_index)
                        if NP.any(oob_freq_index):
                            raise IndexError('One or more of the frequencies outside interpolation range')
                    else:
                        if freqs is not None:
                            ib_freq_index = NP.ones(freqs.size, dtype=NP.bool)
                            oob_freq_index = NP.zeros(freqs.size, dtype=NP.bool)
                            oobl_freq_index = NP.zeros(freqs.size, dtype=NP.bool)
                            oobr_freq_index = NP.zeros(freqs.size, dtype=NP.bool)
                        else:
                            ib_freq_index = None
                            oob_freq_index = None

                    if self.gaintable[key]['time'] is not None:
                        ib_time_index = NP.logical_and(times <= NP.amax(self.gaintable[key]['time']), times >= NP.amin(self.gaintable[key]['time']))
                        oobl_time_index = times < NP.amin(self.gaintable[key]['time'])
                        oobr_time_index = times > NP.amax(self.gaintable[key]['time'])
                        oob_time_index = NP.logical_not(ib_time_index)
                        if NP.any(oob_time_index):
                            raise IndexError('One or more of the times outside interpolation range')
                    else:
                        if times is not None:
                            ib_time_index = NP.ones(times.size, dtype=NP.bool)
                            oob_time_index = NP.zeros(times.size, dtype=NP.bool)
                            oobl_time_index = NP.zeros(times.size, dtype=NP.bool)
                            oobr_time_index = NP.zeros(times.size, dtype=NP.bool)
                        else:
                            ib_time_index = None
                            oob_time_index = None

                    if isinstance(self.splinefuncs[key], dict):
                        if 'dims' not in self.splinefuncs[key]:
                            raise KeyError('Key "dims" not found in attribute splinefuncs[{0}]'.format(key))
                        if not isinstance(self.splinefuncs[key]['dims'], NP.ndarray):
                            raise TypeError('Key "dims" in attribute splinefuncs[{0}] must contain a numpy array'.format(key))

                        if self.splinefuncs[key]['dims'].size == 1:
                            if self.splinefuncs[key]['dims'][0] == 'time':
                                ntimes = ib_time_index.size
                                if freqs is None:
                                    nchan = 1
                                else:
                                    nchan = ib_freq_index.size
                                inp = times[ib_time_index]
                            else:
                                nchan = ib_freq_index.size
                                if times is None:
                                    ntimes = 1
                                else:
                                    ntimes = ib_time_index.size
                                inp = freqs[ib_freq_index]
                        else:
                            inp_times = times[ib_time_index]
                            inp_freqs = freqs[ib_freq_index]
                            ntimes = ib_time_index.size
                            nchan = ib_freq_index.size

                        tgrid, fgrid = NP.meshgrid(inp_times, inp_freqs)
                        tvec = tgrid.ravel()
                        fvec = fgrid.ravel()

                        if key == 'antenna-based':
                            ind1 = NMO.find_list_in_list(labels, a1_labels)
                            ind2 = NMO.find_list_in_list(labels, a2_labels)
                            if NP.sum(ind1.mask) > 0:
                                raise IndexError('Some antenna gains could not be found')
                            if NP.sum(ind2.mask) > 0:
                                raise IndexError('Some antenna gains could not be found')
                            g1_conj = None
                            g2 = None
                            for i in xrange(ind1.size):
                                if self.splinefuncs[key]['dims'].size == 1:
                                    if g1_conj is None:
                                        g1_conj = (self.splinefuncs[key]['interp']['real'][ind1[i]](inp) - 1j * self.splinefuncs[key]['interp']['imag'][ind1[i]](inp)).reshape(1,nchan,ntimes)
                                        g2 = (self.splinefuncs[key]['interp']['real'][ind2[i]](inp) + 1j * self.splinefuncs[key]['interp']['imag'][ind2[i]](inp)).reshape(1,nchan,ntimes)
                                    else:
                                        g1_conj = NP.concatenate((g1_conj, (self.splinefuncs[key]['interp']['real'][ind1[i]](inp) - 1j * self.splinefuncs[key]['interp']['imag'][ind1[i]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                        g2 = NP.concatenate((g2, (self.splinefuncs[key]['interp']['real'][ind2[i]](inp) + 1j * self.splinefuncs[key]['interp']['imag'][ind2[i]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                else:
                                    if g1_conj is None:
                                        g1_conj = (self.splinefuncs[key]['interp']['real'][ind1[i]].ev(tvec,fvec) - 1j * self.splinefuncs[key]['interp']['imag'][ind1[i]].ev(tvec,fvec)).reshape(1,nchan,ntimes)
                                        g2 = (self.splinefuncs[key]['interp']['real'][ind2[i]].ev(tvec,fvec) + 1j * self.splinefuncs[key]['interp']['imag'][ind2[i]].ev(tvec,fvec)).reshape(1,nchan,ntimes)
                                    else:
                                        g1_conj = NP.concatenate((g1_conj, (self.splinefuncs[key]['interp']['real'][ind1[i]].ev(tvec,fvec) - 1j * self.splinefuncs[key]['interp']['imag'][ind1[i]].ev(tvec,fvec)).reshape(1,nchan,ntimes)), axis=0)
                                        g2 = NP.concatenate((g2, (self.splinefuncs[key]['interp']['real'][ind2[i]].ev(tvec,fvec) + 1j * self.splinefuncs[key]['interp']['imag'][ind2[i]].ev(tvec,fvec)).reshape(1,nchan,ntimes)), axis=0)

                            blgains = blgains * g1_conj * g2 * NP.ones((1,nchan,ntimes), dtype=NP.complex)
                        else:
                            g12 = None
                            for labelind,label in enumerate(bl_labels):
                                if label in labels:
                                    ind = NP.where(self.gaintable[key]['label'] == label)[0]
                                    if self.splinefuncs[key]['dims'].size == 1:
                                        if g12 is None:
                                            g12 = (self.splinefuncs[key]['interp']['real'][ind[0]](inp) + 1j * self.splinefuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.splinefuncs[key]['interp']['real'][ind[0]](inp) + 1j * self.splinefuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                    else:
                                        if g12 is None:
                                            g12 = (self.splinefuncs[key]['interp']['real'][ind[0]].ev(tvec,fvec) + 1j * self.splinefuncs[key]['interp']['imag'][ind[0]].ev(tvec,fvec)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.splinefuncs[key]['interp']['real'][ind[0]].ev(tvec,fvec) + 1j * self.splinefuncs[key]['interp']['imag'][ind[0]].ev(tvec,fvec)).reshape(1,nchan,ntimes)), axis=0)
                                elif NP.asarray([tuple(reversed(label))], dtype=bl_labels.dtype)[0] in labels:
                                    ind = NP.where(labels == NP.asarray([tuple(reversed(label))], dtype=bl_labels.dtype)[0])[0]
                                    if self.splinefuncs[key]['dims'].size == 1:
                                        if g12 is None:
                                            g12 = (self.splinefuncs[key]['interp']['real'][ind[0]](inp) - 1j * self.splinefuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.splinefuncs[key]['interp']['real'][ind[0]](inp) - 1j * self.splinefuncs[key]['interp']['imag'][ind[0]](inp)).reshape(1,nchan,ntimes)), axis=0)
                                    else:
                                        if g12 is None:
                                            g12 = (self.splinefuncs[key]['interp']['real'][ind[0]].ev(tvec,fvec) - 1j * self.splinefuncs[key]['interp']['imag'][ind[0]].ev(tvec,fvec)).reshape(1,nchan,ntimes)
                                        else:
                                            g12 = NP.concatenate((g12, (self.splinefuncs[key]['interp']['real'][ind[0]].ev(tvec,fvec) - 1j * self.splinefuncs[key]['interp']['imag'][ind[0]].ev(tvec,fvec)).reshape(1,nchan,ntimes)), axis=0)
                                else:
                                    if g12 is None:
                                        g12 = NP.ones((1,nchan,ntimes), dtype=NP.complex)
                                    else:
                                        g12 = NP.concatenate((g12, NP.ones((1,nchan,ntimes), dtype=NP.complex)), axis=0)
                            blgains = blgains * g12 * NP.ones((1,nchan,ntimes), dtype=NP.complex)

        interp_axes_order = ['label', 'frequency', 'time']
        if axes_order is None:
            axes_order = self.gaintable['antenna-based']['ordering']
        elif not isinstance(axes_order, (list, NP.ndarray)):
            raise TypeError('axes_order must be a list')
        else:
            if len(axes_order) != 3:
                raise ValueError('axes_order must be a three element list')
            for orderkey in ['label', 'frequency', 'time']:
                if orderkey not in axes_order:
                    raise ValueError('axes_order does not contain key "{0}"'.format(orderkey))

        transpose_order = NMO.find_list_in_list(interp_axes_order, axes_order)
        blgains = NP.transpose(blgains, axes=transpose_order)

        return blgains

    #############################################################################

    def nearest_gains(self, bl_labels, freqs=None, times=None, axes_order=None):

        """
        ------------------------------------------------------------------------
        Extract complex instrument gains for given baselines from the gain table
        determined by nearest neighbor logic

        Inputs:

        bl_labels   [Numpy structured array tuples] Labels of antennas in the
                    pair used to produce the baseline vector under fields 'A2'
                    and 'A1' for second and first antenna respectively. The
                    baseline vector is obtained by position of antennas under
                    'A2' minus position of antennas under 'A1'

        freqs       [None or numpy array] Array of frequencies at which the
                    gains are to be interpolated using the attribute
                    splinefuncs. If set to None (default), all frequencies in
                    the gaintable are assumed. The specified frequencies must
                    always lie within the range which was used in creating the
                    interpolation functions, otherwise an exception will be
                    raised. The array is of size nchan

        times       [None or numpy array] Array of times at which the gains
                    are to be interpolated using the attribute splinefuncs. If
                    set to None (default), all times in the gaintable are
                    assumed. The specified times must always lie within the
                    range which was used in creating the interpolation
                    functions, otherwise an exception will be raised. The array
                    is of size nts

        axes_order  [None or list or numpy array] Axes ordering for extracted
                    gains. It must contain the three elements 'label',
                    'frequency', and 'time'. If set to None, it will be
                    returned in the same order as in the input gaintable.

        Outputs:

        [numpy array] Complex gains of shape nbl x nchan x nts for the specified
        baselines, frequencies and times.
        ------------------------------------------------------------------------
        """

        try:
            bl_labels
        except NameError:
            raise NameError('Input bl_labels must be specified')

        blgains = NP.asarray(1.0).reshape(1,1,1)
        if self.gaintable is not None:
            a1_labels = bl_labels['A1']
            a2_labels = bl_labels['A2']
            for gainkey in ['antenna-based', 'baseline-based']:
                if gainkey in self.gaintable:
                    temp_axes_order = ['label', 'frequency', 'time']
                    inp_order = self.gaintable[gainkey]['ordering']
                    temp_transpose_order = NMO.find_list_in_list(inp_order, temp_axes_order)
                    if NP.all(inp_order == temp_axes_order):
                        gains = NP.copy(self.gaintable[gainkey]['gains'])
                    else:
                        gains = NP.transpose(NP.copy(self.gaintable[gainkey]['gains']), axes=temp_transpose_order)

                    freqs_to_search = copy.copy(freqs)
                    if freqs_to_search is None:
                        freqs_to_search = copy.copy(self.gaintable[gainkey]['frequency'])

                    if freqs_to_search is not None:
                        if self.gaintable[gainkey]['frequency'] is not None:
                            inpind, refind_freqs, distNN= LKP.find_1NN(self.gaintable[gainkey]['frequency'].reshape(-1,1), freqs_to_search.reshape(-1,1), remove_oob=True)
                        else:
                            refind_freqs = None
                    if refind_freqs is None:
                        refind_freqs = NP.arange(gains.shape[1])

                    times_to_search = copy.copy(times)
                    if times_to_search is None:
                        times_to_search = copy.copy(self.gaintable[gainkey]['time'])

                    if times_to_search is not None:
                        if self.gaintable[gainkey]['time'] is not None:
                            inpind, refind_times, distNN = LKP.find_1NN(self.gaintable[gainkey]['time'].reshape(-1,1), times_to_search.reshape(-1,1), remove_oob=True)
                        else:
                            refind_times = None
                    if refind_times is None:
                        refind_times = NP.arange(gains.shape[2])

                    if gains.shape[0] == 1:
                        blgains = blgains * gains[:,refind_freqs,refind_times].reshape(1,refind_freqs.size,refind_times.size)
                    else:
                        labels = self.gaintable[gainkey]['label']
                        if gainkey == 'antenna-based':
                            ind1 = NMO.find_list_in_list(labels, a1_labels)
                            ind2 = NMO.find_list_in_list(labels, a2_labels)
                            if NP.sum(ind1.mask) > 0:
                                raise IndexError('Some antenna gains could not be found')
                            if NP.sum(ind2.mask) > 0:
                                raise IndexError('Some antenna gains could not be found')
                            blgains = blgains * gains[NP.ix_(ind2,refind_freqs,refind_times)].reshape(ind2.size,refind_freqs.size,refind_times.size) * gains[NP.ix_(ind1,refind_freqs,refind_times)].conj().reshape(ind1.size,refind_freqs.size,refind_times.size)
                        else:
                            labels_conj = [tuple(reversed(label)) for label in labels]
                            labels_conj = NP.asarray(labels_conj, dtype=labels.dtype)
                            labels_conj_appended = NP.concatenate((labels, labels_conj), axis=0)
                            gains_conj_appended = NP.concatenate((gains, gains.conj()), axis=0)
                            ind = NMO.find_list_in_list(labels_conj_appended, bl_labels)
                            selected_gains = gains_conj_appended[NP.ix_(ind.compressed(),refind_freqs,refind_times)]
                            if ind.compressed().size == 1:
                                selected_gains = selected_gains.reshape(NP.sum(~ind.mask),refind_freqs.size,refind_times.size)
                            blgains[~ind.mask, ...] = blgains[~ind.mask, ...] * selected_gains
                    if axes_order is None:
                        axes_order = inp_order
                    elif not isinstance(axes_order, (list, NP.ndarray)):
                        raise TypeError('axes_order must be a list')
                    else:
                        if len(axes_order) != 3:
                            raise ValueError('axes_order must be a three element list')
                        for orderkey in ['label', 'frequency', 'time']:
                            if orderkey not in axes_order:
                                raise ValueError('axes_order does not contain key "{0}"'.format(orderkey))

                    transpose_order = NMO.find_list_in_list(inp_order, axes_order)
                    blgains = NP.transpose(blgains, axes=transpose_order)

        return blgains

    #############################################################################

    def eval_gains(self, bl_labels, freq_index=None, time_index=None,
                   axes_order=None):

        """
        ------------------------------------------------------------------------
        Extract complex instrument gains for given baselines from the gain table

        Inputs:

        bl_labels   [Numpy structured array tuples] Labels of antennas in the
                    pair used to produce the baseline vector under fields 'A2'
                    and 'A1' for second and first antenna respectively. The
                    baseline vector is obtained by position of antennas under
                    'A2' minus position of antennas under 'A1'

        freq_index  [None, int, list or numpy array] Index (scalar) or indices
                    (list or numpy array) along the frequency axis at which
                    gains are to be extracted. If set to None, gains at all
                    frequencies in the gain table will be extracted.

        time_index  [None, int, list or numpy array] Index (scalar) or indices
                    (list or numpy array) along the time axis at which gains
                    are to be extracted. If set to None, gains at all timesin
                    the gain table will be extracted.

        axes_order  [None or list or numpy array] Axes ordering for extracted
                    gains. It must contain the three elements 'label',
                    'frequency', and 'time'. If set to None, it will be
                    returned in the same order as in the input gaintable.

        Outputs:

        [numpy array] Complex gains of shape nbl x nchan x nts for the specified
        baselines, frequencies and times.
        ------------------------------------------------------------------------
        """

        return extract_gains(self.gaintable, bl_labels, freq_index=None,
                             time_index=None, axes_order=None)

    #############################################################################

    def write_gaintable(self, outfile, axes_order=None, compress=True,
                        compress_fmt='gzip', compress_opts=9):

        """
        ------------------------------------------------------------------------
        Write gain table with specified axes ordering to external file in HDF5
        format

        Inputs:

        outfile     [string] Filename including full path into which the gain
                    table will be written

        axes_order  [None or list or numpy array] The axes ordering of gain
                    table that will be written to external file specified in
                    outfile. If set to None, it will store in the same order
                    as in the attribute gaintable

        compress    [boolean] Specifies if the gain table is written in
                    compressed format. The compression format and compression
                    parameters are specified in compress_fmt and compress_opts
                    respectively

        compress_fmt
                    [string] Accepted values are 'gzip' (default) or 'lzf'. See
                    h5py module documentation for comparison of these
                    compression formats

        compress_opts
                    [integer] Applies only if compress_fmt is set to 'gzip'. It
                    must be an integer in the range 0 to 9. Default=9 implies
                    maximum compression
        ------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('outfile not specified')

        if axes_order is not None:
            if not isinstance(axes_order, (list, NP.ndarray)):
                raise TypeError('axes_order must be a list')
            else:
                if len(axes_order) != 3:
                    raise ValueError('axes_order must be a three element list')
                for orderkey in ['label', 'frequency', 'time']:
                    if orderkey not in axes_order:
                        raise ValueError('axes_order does not contain key "{0}"'.format(orderkey))

        if not isinstance(compress, bool):
            raise TypeError('Input parameter compress must be boolean')

        if compress:
            if not isinstance(compress_fmt, str):
                raise TypeError('Input parameter compress_fmt must be a string')
            compress_fmt = compress_fmt.lower()
            if compress_fmt not in ['gzip', 'lzf']:
                raise ValueError('Input parameter compress_fmt invalid')
            if compress_fmt == 'gzip':
                if not isinstance(compress_opts, int):
                    raise TypeError('Input parameter compress_opts must be an integer')
                compress_opts = NP.clip(compress_opts, 0, 9)

        with h5py.File(outfile, 'w') as fileobj:
            for gainkey in self.gaintable:
                if self.gaintable[gainkey] is not None:
                    if axes_order is not None:
                        transpose_order = NMO.find_list_in_list(self.gaintable[gainkey]['ordering'], axes_order)
                    else:
                        axes_order = self.gaintable[gainkey]['ordering']
                    if NP.all(self.gaintable[gainkey]['ordering'] == axes_order):
                        gains = NP.copy(self.gaintable[gainkey]['gains'])
                    else:
                        gains = NP.transpose(NP.copy(self.gaintable[gainkey]['gains']), axes=transpose_order)

                    grp = fileobj.create_group(gainkey)
                    for subkey in self.gaintable[gainkey]:
                        if subkey == 'gains':
                            if compress:
                                chunkshape = []
                                for ind,axis in enumerate(axes_order):
                                    if axis == 'frequency':
                                        chunkshape += [gains.shape[ind]]
                                    else:
                                        chunkshape += [1]
                                chunkshape = tuple(chunkshape)
                                if compress_fmt == 'gzip':
                                    dset = grp.create_dataset(subkey, data=gains, chunks=chunkshape, compression=compress_fmt, compression_opts=compress_opts)
                                else:
                                    dset = grp.create_dataset(subkey, data=gains, chunks=chunkshape, compression=compress_fmt)
                            else:
                                grp.create_dataset(subkey, data=gains, chunks=chunkshape)
                        elif subkey == 'ordering':
                            dset = grp.create_dataset(subkey, data=axes_order)
                        else:
                            if isinstance(self.gaintable[gainkey][subkey], NP.ndarray):
                                dset = grp.create_dataset(subkey, data=self.gaintable[gainkey][subkey])

#################################################################################

class ROI_parameters(object):

    """
    ----------------------------------------------------------------------------
    Class to manage information on the regions of interest for different
    snapshots in an observation.

    Attributes:

    skymodel    [instance of class SkyModel] The common sky model for all the
                observing instances from which the ROI is determined based on
                a subset corresponding to each snapshot observation.

    freq        [numpy vector] Frequency channels (with units specified by the
                attribute freq_scale)

    freq_scale  [string] string specifying the units of frequency. Accepted
                values are 'GHz', 'MHz' and 'Hz'. Default = 'GHz'

    telescope   [dictionary] Contains information about the telescope parameters
                using which the primary beams in the regions of interest are
                determined. It specifies the type of element, element size and
                orientation. It consists of the following keys and information:
                'id'          [string] If set, will ignore the other keys and use
                              telescope details for known telescopes. Accepted
                              values are 'mwa', 'vla', 'gmrt', 'hera', 'paper',
                              'hirax', 'chime' and 'mwa_tools'. If using
                              'mwa_tools', the MWA_Tools and mwapb modules must
                              be installed and imported.
                'shape'       [string] Shape of antenna element. Accepted values
                              are 'dipole', 'delta', and 'dish'. Will be ignored
                              if key 'id' is set. 'delta' denotes a delta
                              function for the antenna element which has an
                              isotropic radiation pattern. 'delta' is the default
                              when keys 'id' and 'shape' are not set.
                'size'        [scalar] Diameter of the telescope dish (in meters)
                              if the key 'shape' is set to 'dish' or length of
                              the dipole if key 'shape' is set to 'dipole'. Will
                              be ignored if key 'shape' is set to 'delta'. Will
                              be ignored if key 'id' is set and a preset value
                              used for the diameter or dipole.
                'orientation' [list or numpy array] If key 'shape' is set to
                              dipole, it refers to the orientation of the dipole
                              element unit vector whose magnitude is specified by
                              length. If key 'shape' is set to 'dish', it refers
                              to the position on the sky to which the dish is
                              pointed. For a dipole, this unit vector must be
                              provided in the local ENU coordinate system aligned
                              with the direction cosines coordinate system or in
                              the Alt-Az coordinate system. This will be
                              used only when key 'shape' is set to 'dipole'.
                              This could be a 2-element vector (transverse
                              direction cosines) where the third (line-of-sight)
                              component is determined, or a 3-element vector
                              specifying all three direction cosines or a two-
                              element coordinate in Alt-Az system. If not provided
                              it defaults to an eastward pointing dipole. If key
                              'shape' is set to 'dish', the orientation refers
                              to the pointing center of the dish on the sky. It
                              can be provided in Alt-Az system as a two-element
                              vector or in the direction cosine coordinate
                              system as a two- or three-element vector. If not
                              set in the case of a dish element, it defaults to
                              zenith. This is not to be confused with the key
                              'pointing_center' in dictionary 'pointing_info'
                              which refers to the beamformed pointing center of
                              the array. The coordinate system is specified by
                              the key 'ocoords'
                'ocoords'     [scalar string] specifies the coordinate system
                              for key 'orientation'. Accepted values are 'altaz'
                              and 'dircos'.
                'element_locs'
                              [2- or 3-column array] Element locations that
                              constitute the tile. Each row specifies
                              location of one element in the tile. The
                              locations must be specified in local ENU
                              coordinate system. First column specifies along
                              local east, second along local north and the
                              third along local up. If only two columns are
                              specified, the third column is assumed to be
                              zeros. If 'elements_locs' is not provided, it
                              assumed to be a one-element system and not a
                              phased array as far as determination of primary
                              beam is concerned.
                'groundplane' [scalar] height of telescope element above the
                              ground plane (in meteres). Default = None will
                              denote no ground plane effects.
                'ground_modify'
                              [dictionary] contains specifications to modify
                              the analytically computed ground plane pattern. If
                              absent, the ground plane computed will not be
                              modified. If set, it may contain the following
                              keys:
                              'scale' [scalar] positive value to scale the
                                      modifying factor with. If not set, the
                                      scale factor to the modification is unity.
                              'max'   [scalar] positive value to clip the
                                      modified and scaled values to. If not set,
                                      there is no upper limit
                'latitude'    [scalar] specifies latitude of the telescope site
                              (in degrees). Default = None (advisable to specify
                              a real value)
                'longitude'   [scalar] specifies latitude of the telescope site
                              (in degrees). Default = 0 (GMT)
                'altitude'    [scalar] Specifies altitude of the telescope site
                              (in m) above the surface of the Earth. Default=0m
                'pol'         [string] specifies polarization when using
                              MWA_Tools for primary beam computation. Value of
                              key 'id' in attribute dictionary telescope must be
                              set to 'mwa_tools'. 'X' or 'x' denotes
                              X-polarization. Y-polarization is specified by 'Y'
                              or 'y'. If polarization is not specified when 'id'
                              of telescope is set to 'mwa_tools', it defaults
                              to X-polarization.

    info        [dictionary] contains information about the region of interest.
                It consists of the following keys and information:
                'radius'  [list of scalars] list of angular radii (in degrees),
                          one entry for each snapshot observation which defines
                          the region of interest.
                'center'  [list of numpy vectors] list of centers of regions of
                          interest. For each snapshot, there is one element in
                          the list each of which is a center of corresponding
                          region of interest. Each numpy vector could be made of
                          two elements (Alt-Az) or three elements (direction
                          cosines).
                'ind'     [list of numpy vectors] list of vectors of indices
                          that define the region of interest as a subset of the
                          sky model. Each element of the list is a numpy vector
                          of indices indexing into the sky model corresponding
                          to each snapshot.
                'pbeam'   [list of numpy arrays] list of array of primary beam
                          values in the region of interest. The size of each
                          element in the list corresponding to each snapshot is
                          n_roi x nchan where n_roi is the number of pixels in
                          region of interest.

    pinfo       [list of dictionaries] Each dictionary element in the list
                corresponds to a specific snapshot. It contains information
                relating to the pointing center. The pointing center can be
                specified either via element delay compensation or by directly
                specifying the pointing center in a certain coordinate system.
                Default = None (pointing centered at zenith). Each dictionary
                element may consist of the following keys and information:
                'gains'           [numpy array] Complex element gains. Must be of
                                  size equal to the number of elements as
                                  specified by the number of rows in
                                  'element_locs'. If set to None (default), all
                                  element gains are assumed to be unity.
                'delays'          [numpy array] Delays (in seconds) to be applied
                                  to the tile elements. Size should be equal to
                                  number of tile elements (number of rows in
                                  antpos). Default = None will set all element
                                  delays to zero phasing them to zenith.
                'pointing_center' [numpy array] This will apply in the absence of
                                  key 'delays'. This can be specified as a row
                                  vector. Should have two-columns if using Alt-Az
                                  coordinates, or two or three columns if using
                                  direction cosines. There is no default. The
                                  coordinate system must be specified in
                                  'pointing_coords' if 'pointing_center' is to be
                                  used.
                'pointing_coords' [string scalar] Coordinate system in which the
                                  pointing_center is specified. Accepted values
                                  are 'altaz' or 'dircos'. Must be provided if
                                  'pointing_center' is to be used. No default.
                'delayerr'        [int, float] RMS jitter in delays used in the
                                  beamformer. Random jitters are drawn from a
                                  normal distribution with this rms. Must be
                                  a non-negative scalar. If not provided, it
                                  defaults to 0 (no jitter).

    Member functions:

    __init__()  Initializes an instance of class ROI_parameters using default
                values or using a specified initialization file

    append_settings()
                Determines and appends ROI (regions of interest) parameter
                information for each snapshot observation using the input
                parameters provided. Optionally also computes the primary beam
                values in the region of interest using the telescope parameters.

    save()      Saves the information about the regions of interest to a FITS
                file on disk

    -----------------------------------------------------------------------------
    """

    def __init__(self, init_file=None):

        """
        -------------------------------------------------------------------------
        Initializes an instance of class ROI_parameters using default values or
        using a specified initialization file

        Class attribute initialized are:
        skymodel, freq, freq_scale, telescope, info, and pinfo

        Read docstring of class ROI_parameters for details on these attributes.

        Keyword input(s):

        init_file    [string] Location of the initialization file from which an
                     instance of class ROI_parameters will be created. File
                     format must be compatible with the one saved to disk by
                     member function save()
        -------------------------------------------------------------------------
        """

        argument_init = False
        init_file_success = False
        if init_file is not None:
            try:
                hdulist = fits.open(init_file)
            except IOError:
                argument_init = True
                print '\tinit_file provided but could not open the initialization file. Attempting to initialize with input parameters...'
            if not argument_init:
                n_obs = hdulist[0].header['n_obs']
                extnames = [hdulist[i].header['EXTNAME'] for i in xrange(1,len(hdulist))]

                self.info = {}
                self.info['radius'] = []
                self.info['center'] = []
                self.info['ind'] = []
                self.info['pbeam'] = []
                self.telescope = {}
                if 'id' in hdulist[0].header:
                    self.telescope['id'] = hdulist[0].header['telescope']

                if 'latitude' in hdulist[0].header:
                    self.telescope['latitude'] = hdulist[0].header['latitude']
                else:
                    self.telescope['latitude'] = None

                if 'longitude' in hdulist[0].header:
                    self.telescope['longitude'] = hdulist[0].header['longitude']
                else:
                    self.telescope['longitude'] = 0.0

                if 'altitude' in hdulist[0].header:
                    self.telescope['altitude'] = hdulist[0].header['altitude']
                else:
                    self.telescope['altitude'] = 0.0

                try:
                    self.telescope['shape'] = hdulist[0].header['element_shape']
                except KeyError:
                    raise KeyError('Antenna element shape not found in the init_file header')

                try:
                    self.telescope['size'] = hdulist[0].header['element_size']
                except KeyError:
                    raise KeyError('Antenna element size not found in the init_file header')

                try:
                    self.telescope['ocoords'] = hdulist[0].header['element_ocoords']
                except KeyError:
                    raise KeyError('Antenna element orientation coordinate system not found in the init_file header')

                if 'ANTENNA ELEMENT ORIENTATION' in extnames:
                    self.telescope['orientation'] = hdulist['ANTENNA ELEMENT ORIENTATION'].data.reshape(1,-1)
                else:
                    raise KeyError('Extension named "orientation" not found in init_file.')

                if 'ANTENNA ELEMENT LOCATIONS' in extnames:
                    self.telescope['element_locs'] = hdulist['ANTENNA ELEMENT LOCATIONS'].data

                if 'ground_plane' in hdulist[0].header:
                    self.telescope['groundplane'] = hdulist[0].header['ground_plane']
                    if 'ground_modify_scale' in hdulist[0].header:
                        if 'ground_modify' not in self.telescope:
                            self.telescope['ground_modify'] = {}
                        self.telescope['ground_modify']['scale'] = hdulist[0].header['ground_modify_scale']
                    if 'ground_modify_max' in hdulist[0].header:
                        if 'ground_modify' not in self.telescope:
                            self.telescope['ground_modify'] = {}
                        self.telescope['ground_modify']['max'] = hdulist[0].header['ground_modify_max']
                else:
                    self.telescope['groundplane'] = None

                if 'FREQ' in extnames:
                    self.freq = hdulist['FREQ'].data
                else:
                    raise KeyError('Extension named "FREQ" not found in init_file.')

                self.info['ind'] = [hdulist['IND_{0:0d}'.format(i)].data for i in range(n_obs)]
                self.info['pbeam'] = [hdulist['PB_{0:0d}'.format(i)].data for i in range(n_obs)]

                self.pinfo = []
                if 'ANTENNA ELEMENT LOCATIONS' in extnames:
                    for i in range(n_obs):
                        self.pinfo += [{}]

                        # try:
                        #     self.pinfo[-1]['delays'] = hdulist['DELAYS_{0:0d}'.format(i)].data
                        # except KeyError:
                        #     raise KeyError('Extension DELAYS_{0:0d} for phased array beamforming not found in init_file'.format(i))

                        if 'DELAYS_{0:0d}'.format(i) in extnames:
                            self.pinfo[-1]['delays'] = hdulist['DELAYS_{0:0d}'.format(i)].data

                        if 'DELAYERR' in hdulist['DELAYS_{0:0d}'.format(i)].header:
                            delayerr = hdulist['DELAYS_{0:0d}'.format(i)].header['delayerr']
                            if delayerr <= 0.0:
                                self.pinfo[-1]['delayerr'] = None
                            else:
                                self.pinfo[-1]['delayerr'] = delayerr

                len_pinfo = len(self.pinfo)
                if len_pinfo > 0:
                    if len_pinfo != n_obs:
                        raise ValueError('Inconsistency in number of pointings in header and number of phased array delay settings')

                for i in range(n_obs):
                    if 'POINTING_CENTER_{0:0d}'.format(i) in extnames:
                        if len_pinfo == 0:
                            self.pinfo += [{}]
                        self.pinfo[i]['pointing_center'] = hdulist['POINTING_CENTER_{0:0d}'.format(i)].data
                        try:
                            self.pinfo[i]['pointing_coords'] = hdulist['POINTING_CENTER_{0:0d}'.format(i)].header['pointing_coords']
                        except KeyError:
                            raise KeyError('Header of extension POINTING_CENTER_{0:0d} not found to contain key "pointing_coords" in init_file'.format(i))

                len_pinfo = len(self.pinfo)
                if len_pinfo > 0:
                    if len_pinfo != n_obs:
                        raise ValueError('Inconsistency in number of pointings in header and number of pointing centers')

                hdulist.close()
                init_file_success = True
                return
        else:
            argument_init = True

        if (not argument_init) and (not init_file_success):
            raise ValueError('Initialization failed with the use of init_file.')

        self.skymodel = None
        self.telescope = None
        self.info = {}
        self.info['radius'] = []
        self.info['ind'] = []
        self.info['pbeam'] = []
        self.info['center'] = []
        self.info['center_coords'] = None

        self.pinfo = []
        self.freq = None

    #############################################################################

    def append_settings(self, skymodel, freq, pinfo=None, lst=None,
                        roi_info=None, telescope=None, freq_scale='GHz'):

        """
        ------------------------------------------------------------------------
        Determines and appends ROI (regions of interest) parameter information
        for each snapshot observation using the input parameters provided.
        Optionally also computes the primary beam values in the region of
        interest using the telescope parameters.

        Inputs:

        skymodel [instance of class SkyModel] The common sky model for all the
                 observing instances from which the ROI is determined based on
                 a subset corresponding to each snapshot observation.

        freq     [numpy vector] Frequency channels (with units specified by the
                 attribute freq_scale)

        pinfo    [list of dictionaries] Each dictionary element in the list
                 corresponds to a specific snapshot. It contains information
                 relating to the pointing center. The pointing center can be
                 specified either via element delay compensation or by directly
                 specifying the pointing center in a certain coordinate system.
                 Default = None (pointing centered at zenith). Each dictionary
                 element may consist of the following keys and information:
                 'gains'           [numpy array] Complex element gains. Must be
                                   of size equal to the number of elements as
                                   specified by the number of rows in
                                   'element_locs'. If set to None (default), all
                                   element gains are assumed to be unity.
                 'delays'          [numpy array] Delays (in seconds) to be
                                   applied to the tile elements. Size should be
                                   equal to number of tile elements (number of
                                   rows in antpos). Default = None will set all
                                   element delays to zero phasing them to zenith
                 'pointing_center' [numpy array] This will apply in the absence
                                   of key 'delays'. This can be specified as a
                                   row vector. Should have two-columns if using
                                   Alt-Az coordinates, or two or three columns
                                   if using direction cosines. There is no
                                   default. The coordinate system must be
                                   specified in 'pointing_coords' if
                                   'pointing_center' is to be used.
                 'pointing_coords' [string scalar] Coordinate system in which
                                   the pointing_center is specified. Accepted
                                   values are 'altaz' or 'dircos'. Must be
                                   provided if 'pointing_center' is to be used.
                                   No default.
                 'delayerr'        [int, float] RMS jitter in delays used in
                                   the beamformer. Random jitters are drawn
                                   from a normal distribution with this rms.
                                   Must be a non-negative scalar. If not
                                   provided, it defaults to 0 (no jitter).

    telescope   [dictionary] Contains information about the telescope parameters
                using which the primary beams in the regions of interest are
                determined. It specifies the type of element, element size and
                orientation. It consists of the following keys and information:
                'id'          [string] If set, will ignore the other keys and
                              use telescope details for known telescopes.
                              Accepted values are 'mwa', 'vla', 'gmrt', 'hera',
                              'paper', 'hirax', 'chime' and 'mwa_tools'. If
                              using 'mwa_tools', the MWA_Tools and mwapb modules
                              must be installed and imported.
                'shape'       [string] Shape of antenna element. Accepted values
                              are 'dipole', 'delta', and 'dish'. Will be ignored
                              if key 'id' is set. 'delta' denotes a delta
                              function for the antenna element which has an
                              isotropic radiation pattern. 'delta' is the
                              default when keys 'id' and 'shape' are not set.
                'size'        [scalar] Diameter of the telescope dish (in
                              meters) if the key 'shape' is set to 'dish' or
                              length of the dipole if key 'shape' is set to
                              'dipole'. Will be ignored if key 'shape' is set to
                              'delta'. Will be ignored if key 'id' is set and a
                              preset value used for the diameter or dipole.
                'orientation' [list or numpy array] If key 'shape' is set to
                              dipole, it refers to the orientation of the dipole
                              element unit vector whose magnitude is specified
                              by length. If key 'shape' is set to 'dish', it
                              refers to the position on the sky to which the
                              dish is pointed. For a dipole, this unit vector
                              must be provided in the local ENU coordinate
                              system aligned with the direction cosines
                              coordinate system or in the Alt-Az coordinate
                              system. This will be used only when key 'shape'
                              is set to 'dipole'. This could be a 2-element
                              vector (transverse direction cosines) where the
                              third (line-of-sight) component is determined,
                              or a 3-element vector specifying all three
                              direction cosines or a two-element coordinate in
                              Alt-Az system. If not provided it defaults to an
                              eastward pointing dipole. If key
                              'shape' is set to 'dish', the orientation refers
                              to the pointing center of the dish on the sky. It
                              can be provided in Alt-Az system as a two-element
                              vector or in the direction cosine coordinate
                              system as a two- or three-element vector. If not
                              set in the case of a dish element, it defaults to
                              zenith. This is not to be confused with the key
                              'pointing_center' in dictionary 'pointing_info'
                              which refers to the beamformed pointing center of
                              the array. The coordinate system is specified by
                              the key 'ocoords'
                'ocoords'     [scalar string] specifies the coordinate system
                              for key 'orientation'. Accepted values are 'altaz'
                              and 'dircos'.
                'element_locs'
                              [2- or 3-column array] Element locations that
                              constitute the tile. Each row specifies
                              location of one element in the tile. The
                              locations must be specified in local ENU
                              coordinate system. First column specifies along
                              local east, second along local north and the
                              third along local up. If only two columns are
                              specified, the third column is assumed to be
                              zeros. If 'elements_locs' is not provided, it
                              assumed to be a one-element system and not a
                              phased array as far as determination of primary
                              beam is concerned.
                'groundplane' [scalar] height of telescope element above the
                              ground plane (in meteres). Default = None will
                              denote no ground plane effects.
                'ground_modify'
                              [dictionary] contains specifications to modify
                              the analytically computed ground plane pattern. If
                              absent, the ground plane computed will not be
                              modified. If set, it may contain the following
                              keys:
                              'scale' [scalar] positive value to scale the
                                      modifying factor with. If not set, the
                                      scale factor to the modification is unity.
                              'max'   [scalar] positive value to clip the
                                      modified and scaled values to. If not set,
                                      there is no upper limit
                'latitude'    [scalar] specifies latitude of the telescope site
                              (in degrees). Default = None, otherwise should
                              equal the value specified during initialization
                              of the instance
                'longitude'   [scalar] specifies longitude of the telescope site
                              (in degrees). Default = None, otherwise should
                              equal the value specified during initialization
                              of the instance
                'altitude'    [scalar] specifies altitude of the telescope site
                              (in m). Default = None, otherwise should
                              equal the value specified during initialization
                              of the instance
                'pol'         [string] specifies polarization when using
                              MWA_Tools for primary beam computation. Value of
                              key 'id' in attribute dictionary telescope must be
                              set to 'mwa_tools'. 'X' or 'x' denotes
                              X-polarization. Y-polarization is specified by 'Y'
                              or 'y'. If polarization is not specified when 'id'
                              of telescope is set to 'mwa_tools', it defaults
                              to X-polarization.

        ------------------------------------------------------------------------
        """

        try:
            skymodel, freq, pinfo
        except NameError:
            raise NameError('skymodel, freq, and pinfo must be specified.')

        if not isinstance(skymodel, SM.SkyModel):
            raise TypeError('skymodel should be an instance of class SkyModel.')
        elif skymodel is not None:
            self.skymodel = skymodel

        if freq is None:
            raise ValueError('freq must be specified using a numpy array')
        elif not isinstance(freq, NP.ndarray):
            raise TypeError('freq must be specified using a numpy array')
        self.freq = freq.ravel()

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            self.freq = NP.asarray(freq)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            self.freq = NP.asarray(freq) * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            self.freq = NP.asarray(freq) * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            self.freq = NP.asarray(freq) * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')
        self.freq_scale = 'Hz'

        if self.telescope is None:
            if isinstance(telescope, dict):
                self.telescope = telescope
            else:
                raise TypeError('Input telescope must be a dictionary.')

        if roi_info is None:
            raise ValueError('roi_info dictionary must be set.')

        pbeam_input = False
        if 'ind' in roi_info:
            if roi_info['ind'] is not None:
                self.info['ind'] += [roi_info['ind']]
                if 'pbeam' in roi_info:
                    if roi_info['pbeam'] is not None:
                        try:
                            pb = roi_info['pbeam'].reshape(-1,self.freq.size)
                        except ValueError:
                            raise ValueError('Number of columns of primary beam in key "pbeam" of dictionary roi_info must be equal to number of frequency channels.')

                        if NP.asarray(roi_info['ind']).size == pb.shape[0]:
                            self.info['pbeam'] += [roi_info['pbeam'].astype(NP.float32)]
                        else:
                            raise ValueError('Number of elements in values in key "ind" and number of rows of values in key "pbeam" must be identical.')
                        pbeam_input = True

                if not pbeam_input: # Will require sky positions in Alt-Az coordinates
                    if skymodel.coords == 'radec':
                        if self.telescope['latitude'] is None:
                            raise ValueError('Latitude of the observatory must be provided.')
                        if lst is None:
                            raise ValueError('LST must be provided.')
                        skypos_altaz = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.location[:,0]).reshape(-1,1), skymodel.location[:,1].reshape(-1,1))), self.telescope['latitude'], units='degrees')
                    elif skymodel.coords == 'hadec':
                        if self.telescope['latitude'] is None:
                            raise ValueError('Latitude of the observatory must be provided.')
                        skypos_altaz = GEOM.hadec2altaz(skymodel.location, self.telescope['latitude'], units='degrees')

                    elif skymodel.coords == 'dircos':
                        skypos_altaz = GEOM.dircos2altaz(skymodel.location, units='degrees')
                    elif skymodel.coords == 'altaz':
                        skypos_altaz = skymodel.location
                    else:
                        raise KeyError('skycoords invalid or unspecified in skymodel')
            if 'radius' in roi_info:
                self.info['radius'] += [roi_info['radius']]
            if 'center' in roi_info:
                self.info['center'] += [roi_info['center']]
        else:
            if roi_info['radius'] is None:
                roi_info['radius'] = 90.0
            else:
                roi_info['radius'] = max(0.0, min(roi_info['radius'], 90.0))
            self.info['radius'] += [roi_info['radius']]

            if roi_info['center'] is None:
                self.info['center'] += [NP.asarray([90.0, 270.0]).reshape(1,-1)]
            else:
                roi_info['center'] = NP.asarray(roi_info['center']).reshape(1,-1)
                if roi_info['center_coords'] == 'dircos':
                    self.info['center'] += [GEOM.dircos2altaz(roi_info['center'], units='degrees')]
                elif roi_info['center_coords'] == 'altaz':
                    self.info['center'] += [roi_info['center']]
                elif roi_info['center_coords'] == 'hadec':
                    self.info['center'] += [GEOM.hadec2altaz(roi_info['center'], self.telescope['latitude'], units='degrees')]
                elif roi_info['center_coords'] == 'radec':
                    if lst is None:
                        raise KeyError('LST not provided for coordinate conversion')
                    hadec = NP.asarray([lst-roi_info['center'][0,0], roi_info['center'][0,1]]).reshape(1,-1)
                    self.info['center'] += [GEOM.hadec2altaz(hadec, self.telescope['latitude'], units='degrees')]
                elif roi_info['center_coords'] == 'dircos':
                    self.info['center'] += [GEOM.dircos2altaz(roi_info['center'], units='degrees')]
                else:
                    raise ValueError('Invalid coordinate system specified for center')

            if skymodel.coords == 'radec':
                if self.telescope['latitude'] is None:
                    raise ValueError('Latitude of the observatory must be provided.')

                if lst is None:
                    raise ValueError('LST must be provided.')
                skypos_altaz = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.location[:,0]).reshape(-1,1), skymodel.location[:,1].reshape(-1,1))), self.telescope['latitude'], units='degrees')
            elif skymodel.coords == 'hadec':
                if self.telescope['latitude'] is None:
                    raise ValueError('Latitude of the observatory must be provided.')
                skypos_altaz = GEOM.hadec2altaz(skymodel.location, self.telescope['latitude'], units='degrees')
            elif skymodel.coords == 'dircos':
                skypos_altaz = GEOM.dircos2altaz(skymodel.location, units='degrees')
            elif skymodel.coords == 'altaz':
                skypos_altaz = skymodel.location
            else:
                raise KeyError('skycoords invalid or unspecified in skymodel')

            dtheta = GEOM.sphdist(self.info['center'][-1][0,1], self.info['center'][-1][0,0], 270.0, 90.0)
            if dtheta > 1e-2: # ROI center is not zenith
                m1, m2, d12 = GEOM.spherematch(self.info['center'][-1][0,0], self.info['center'][-1][0,1], skypos_altaz[:,0], skypos_altaz[:,1], roi_info['radius'], maxmatches=0)
            else:
                m2, = NP.where(skypos_altaz[:,0] >= 90.0-roi_info['radius']) # select sources whose altitude (angle above horizon) is 90-radius
            self.info['ind'] += [m2]

        if self.info['center_coords'] is None:
            if 'center_coords' in roi_info:
                if (roi_info['center_coords'] == 'altaz') or (roi_info['center_coords'] == 'dircos') or (roi_info['center_coords'] == 'hadec') or (roi_info['center_coords'] == 'radec'):
                    self.info['center_coords'] = roi_info['center_coords']

        if not pbeam_input:
            if pinfo is None:
                raise ValueError('Pointing info dictionary pinfo must be specified.')
            self.pinfo += [pinfo]

            if 'pointing_coords' in pinfo: # Convert pointing coordinate to Alt-Az
                if (pinfo['pointing_coords'] != 'dircos') and (pinfo['pointing_coords'] != 'altaz'):
                    if self.telescope['latitude'] is None:
                        raise ValueError('Latitude of the observatory must be provided.')
                    if pinfo['pointing_coords'] == 'radec':
                        if lst is None:
                            raise ValueError('LST must be provided.')
                        self.pinfo[-1]['pointing_center'] = NP.asarray([lst-pinfo['pointing_center'][0,0], pinfo['pointing_center'][0,1]]).reshape(1,-1)
                        self.pinfo[-1]['pointing_center'] = GEOM.hadec2altaz(self.pinfo[-1]['pointing_center'], self.telescope['latitude'], units='degrees')
                    elif pinfo[-1]['pointing_coords'] == 'hadec':
                        self.pinfo[-1]['pointing_center'] = GEOM.hadec2altaz(pinfo[-1]['pointing_center'], self.telescope['latitude'], units='degrees')
                    else:
                        raise ValueError('pointing_coords in dictionary pinfo must be "dircos", "altaz", "hadec" or "radec".')
                    self.pinfo[-1]['pointing_coords'] = 'altaz'

            ind = self.info['ind'][-1]
            if 'id' in self.telescope:
                if self.telescope['id'] == 'mwa_tools':
                    if not mwa_tools_found:
                        raise ImportError('MWA_Tools could not be imported which is required for power pattern computation.')

                    pbeam = NP.empty((ind.size, self.freq.size))
                    for i in xrange(self.freq.size):
                        pbx_MWA, pby_MWA = MWAPB.MWA_Tile_advanced(NP.radians(90.0-skypos_altaz[ind,0]).reshape(-1,1), NP.radians(skypos_altaz[ind,1]).reshape(-1,1), freq=self.freq[i], delays=self.pinfo[-1]['delays']/435e-12)
                        if 'pol' in self.telescope:
                            if (self.telescope['pol'] == 'X') or (self.telescope['pol'] == 'x'):
                                pbeam[:,i] = pbx_MWA.ravel()
                            elif (self.telescope['pol'] == 'Y') or (self.telescope['pol'] == 'y'):
                                pbeam[:,i] = pby_MWA.ravel()
                            else:
                                raise ValueError('Key "pol" in attribute dictionary telescope is invalid.')
                        else:
                            self.telescope['pol'] = 'X'
                            pbeam[:,i] = pbx_MWA.ravel()
                else:
                    pbeam = PB.primary_beam_generator(skypos_altaz[ind,:], self.freq, self.telescope, freq_scale=self.freq_scale, skyunits='altaz', pointing_info=self.pinfo[-1])
            else:
                pbeam = PB.primary_beam_generator(skypos_altaz[ind,:], self.freq, self.telescope, freq_scale=self.freq_scale, skyunits='altaz', pointing_info=self.pinfo[-1])

            self.info['pbeam'] += [pbeam.astype(NP.float32)]

    #############################################################################

    def save(self, infile, tabtype='BinTableHDU', overwrite=False, verbose=True):

        """
        ------------------------------------------------------------------------
        Saves the information about the regions of interest to a FITS file on
        disk

        Inputs:

        infile       [string] Filename with full path to be saved to. Will be
                     appended with '.fits' extension

        Keyword Input(s):

        tabtype      [string] indicates table type for one of the extensions in
                     the FITS file. Allowed values are 'BinTableHDU' and
                     'TableHDU' for binary ascii tables respectively. Default is
                     'BinTableHDU'.

        overwrite    [boolean] True indicates overwrite even if a file already
                     exists. Default = False (does not overwrite)

        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        ----------------------------------------------------------------------------
        """

        try:
            infile
        except NameError:
            raise NameError('No filename provided. Aborting ROI_parameters.save()...')

        filename = infile + '.fits'

        if verbose:
            print '\nSaving information about regions of interest...'

        hdulist = []

        hdulist += [fits.PrimaryHDU()]
        hdulist[0].header['EXTNAME'] = 'PRIMARY'
        hdulist[0].header['n_obs'] = (len(self.info['ind']), 'Number of observations')
        if 'id' in self.telescope:
            hdulist[0].header['telescope'] = (self.telescope['id'], 'Telescope Name')
        hdulist[0].header['element_shape'] = (self.telescope['shape'], 'Antenna element shape')
        hdulist[0].header['element_size'] = (self.telescope['size'], 'Antenna element size [m]')
        hdulist[0].header['element_ocoords'] = (self.telescope['ocoords'], 'Antenna element orientation coordinates')
        if self.telescope['latitude'] is not None:
            hdulist[0].header['latitude'] = (self.telescope['latitude'], 'Latitude (in degrees)')
        hdulist[0].header['longitude'] = (self.telescope['longitude'], 'Longitude (in degrees)')
        if self.telescope['altitude'] is not None:
            hdulist[0].header['altitude'] = (self.telescope['altitude'], 'Altitude (in m)')
        if self.telescope['groundplane'] is not None:
            hdulist[0].header['ground_plane'] = (self.telescope['groundplane'], 'Antenna element height above ground plane [m]')
            if 'ground_modify' in self.telescope:
                if 'scale' in self.telescope['ground_modify']:
                    hdulist[0].header['ground_modify_scale'] = (self.telescope['ground_modify']['scale'], 'Ground plane modification scale factor')
                if 'max' in self.telescope['ground_modify']:
                    hdulist[0].header['ground_modify_max'] = (self.telescope['ground_modify']['max'], 'Maximum ground plane modification')

        hdulist += [fits.ImageHDU(self.telescope['orientation'], name='Antenna element orientation')]
        if verbose:
            print '\tCreated an extension for antenna element orientation.'

        if 'element_locs' in self.telescope:
            hdulist += [fits.ImageHDU(self.telescope['element_locs'], name='Antenna element locations')]

        hdulist += [fits.ImageHDU(self.freq, name='FREQ')]
        if verbose:
            print '\t\tCreated an extension HDU of {0:0d} frequency channels'.format(self.freq.size)

        for i in range(len(self.info['ind'])):
            hdulist += [fits.ImageHDU(self.info['ind'][i], name='IND_{0:0d}'.format(i))]
            hdulist += [fits.ImageHDU(self.info['pbeam'][i], name='PB_{0:0d}'.format(i))]
            if self.pinfo: # if self.pinfo is not empty
                if 'delays' in self.pinfo[i]:
                    hdulist += [fits.ImageHDU(self.pinfo[i]['delays'], name='DELAYS_{0:0d}'.format(i))]
                    if 'delayerr' in self.pinfo[i]:
                        if self.pinfo[i]['delayerr'] is not None:
                            hdulist[-1].header['delayerr'] = (self.pinfo[i]['delayerr'], 'Jitter in delays [s]')
                        else:
                            hdulist[-1].header['delayerr'] = (0.0, 'Jitter in delays [s]')

                if 'pointing_center' in self.pinfo[i]:
                    hdulist += [fits.ImageHDU(self.pinfo[i]['pointing_center'], name='POINTING_CENTER_{0:0d}'.format(i))]
                    if 'pointing_coords' in self.pinfo[i]:
                        hdulist[-1].header['pointing_coords'] = (self.pinfo[i]['pointing_coords'], 'Pointing coordinate system')
                    else:
                        raise KeyError('Key "pointing_coords" not found in attribute pinfo.')

        if verbose:
            print '\t\tCreated HDU extensions for {0:0d} observations containing ROI indices and primary beams'.format(len(self.info['ind']))

        if verbose:
            print '\tNow writing FITS file to disk...'

        hdu = fits.HDUList(hdulist)
        hdu.writeto(filename, clobber=overwrite)

        if verbose:
            print '\tRegions of interest information written successfully to FITS file on disk:\n\t\t{0}\n'.format(filename)

#################################################################################

class InterferometerArray(object):

    """
    ----------------------------------------------------------------------------
    Class to manage information on a multi-element interferometer array.

    Attributes:

    A_eff       [scalar, list or numpy vector] Effective area of the
                interferometers (in m^2). If a scalar is provided, it is assumed
                to be identical for all interferometers. Otherwise, one value
                must be specified for each interferometer. Default is
                pi * (25/2)^2, appropriate for a 25 m VLA dish.

    baselines:  [M x 3 Numpy array] The baseline vectors associated with the
                M interferometers in SI units. The coordinate system of these
                vectors is specified by another attribute baseline_coords.

    baseline_coords
                [string] Coordinate system for the baseline vectors. Default is
                'localenu'. Other accepted values are 'equatorial'

    baseline_lengths
                [M-element numpy array] Lengths of the baseline in SI units

    projected_baselines
                [M x 3 x n_snaps Numpy array] The projected baseline vectors
                associated with the M interferometers and number of snapshots in
                SI units. The coordinate system of these vectors is specified by
                either pointing_center, phase_center or as specified in input to
                member function project_baselines().

    bp          [numpy array] Bandpass weights of size n_baselines x nchan x
                n_acc, where n_acc is the number of accumulations in the
                observation, nchan is the number of frequency channels, and
                n_baselines is the number of baselines

    bp_wts      [numpy array] Additional weighting to be applied to the bandpass
                shapes during the application of the member function
                delay_transform(). Same size as attribute bp.

    channels    [list or numpy vector] frequency channels in Hz

    eff_Q       [scalar, list or numpy vector] Efficiency of the interferometers,
                one value for each interferometer. Default = 0.89, appropriate
                for the VLA. Has to be between 0 and 1. If only a scalar value
                provided, it will be assumed to be identical for all the
                interferometers. Otherwise, one value must be provided for each
                of the interferometers.

    freq_resolution
                [scalar] Frequency resolution (in Hz)

    labels      [list of 2-element tuples] A unique identifier (tuple of
                strings) for each of the interferometers.

    lags        [numpy vector] Time axis obtained when the frequency axis is
                inverted using a FFT. Same size as channels. This is
                computed in member function delay_transform().

    lag_kernel  [numpy array] Inverse Fourier Transform of the frequency
                bandpass shape. In other words, it is the impulse response
                corresponding to frequency bandpass. Same size as attributes
                bp and bp_wts. It is initialized in __init__() member function
                but effectively computed in member function delay_transform()

    latitude    [Scalar] Latitude of the interferometer's location. Default
                is 34.0790 degrees North corresponding to that of the VLA.

    altitude    [Scalar] Altitude of the interferometer's location. Default
                is 0 m.

    lst         [list] List of LST (in degrees) for each timestamp

    n_acc       [scalar] Number of accumulations

    groups      [dictionary] Contains the grouping of unique baselines and the
                redundant baselines as numpy recarray under each unique baseline 
                category/flavor. It contains as keys the labels (tuple of A1, A2) 
                of unique baselines and the value under each of these keys is a 
                list of baseline labels that are redundant under that category

    bl_reversemap
                [dictionary] Contains the baseline category for each baseline. 
                The keys are baseline labels as tuple and the value under each 
                key is the label of the unique baseline category that it falls 
                under. 

    gaininfo    [None or instance of class GainInfo] Instance of class
                Gaininfo. If set to None, default gains assumed to be unity.

    gradient_mode
                [string] If set to None, visibilities will be simulated as
                usual. If set to string, both visibilities and visibility
                gradients with respect to the quantity specified in the
                string will be simulated. Currently accepted value is
                'baseline'. Plan to incorporate gradients with respect to
                'skypos' and 'frequency' as well in the future.

    gradient    [dictionary] If gradient_mode is set to None, it is an empty
                dictionary. If gradient_mode is not None, this quantity holds
                the gradient under the key specified by gradient_mode.
                Currently, supports 'baseline' key. Other gradients will be
                supported in future. It contains the following keys and values.
                If gradient_mode == 'baseline':
                'baseline'  [numpy array] Visibility gradients with respect to
                            baseline vector. Complex numpy array of shape
                            3 x nbl x nchan x nts

    obs_catalog_indices
                [list of lists] Each element in the top list corresponds to a
                timestamp. Inside each top list is a list of indices of sources
                from the catalog which are observed inside the region of
                interest. This is computed inside member function observe().

    pointing_center
                [2-column numpy array] Pointing center (latitude and
                longitude) of the observation at a given timestamp. This is
                where the telescopes will be phased up to as reference.
                Coordinate system for the pointing_center is specified by another
                attribute pointing_coords.

    phase_center
                [2-column numpy array] Phase center (latitude and
                longitude) of the observation at a given timestamp. This is
                where the telescopes will be phased up to as reference.
                Coordinate system for the phase_center is specified by another
                attribute phase_center_coords.

    pointing_coords
                [string] Coordinate system for telescope pointing. Accepted
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz'
                (Altitude-Azimuth). Default = 'hadec'.

    phase_center_coords
                [string] Coordinate system for array phase center. Accepted
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz'
                (Altitude-Azimuth). Default = 'hadec'.

    skycoords   [string] Coordinate system for the sky positions of sources.
                Accepted values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or
                'altaz' (Altitude-Azimuth). Default = 'radec'.

    skyvis_freq [numpy array] Complex visibility due to sky emission (in Jy or K)
                along frequency axis for each interferometer estimated from the
                specified external catalog. Same size as vis_freq. Used in the
                member function observe(). Read its docstring for more details.
                Has dimensions n_baselines x nchan x n_snaps.

    skyvis_lag  [numpy array] Complex visibility due to sky emission (in Jy Hz or
                K Hz) along the delay axis for each interferometer obtained by
                FFT of skyvis_freq along frequency axis. Same size as vis_freq.
                Created in the member function delay_transform(). Read its
                docstring for more details. Same dimensions as skyvis_freq

    telescope   [dictionary] dictionary that specifies the type of element,
                element size and orientation. It consists of the following keys
                and values:
                'id'          [string] If set, will ignore the other keys and use
                              telescope details for known telescopes. Accepted
                              values are 'mwa', 'vla', 'gmrt', 'hera', 'paper',
                              'hirax', 'chime'and other custom values.
                              Default = 'mwa'
                'shape'       [string] Shape of antenna element. Accepted values
                              are 'dipole', 'delta', and 'dish'. Will be ignored
                              if key 'id' is set. 'delta' denotes a delta
                              function for the antenna element which has an
                              isotropic radiation pattern. 'dish' is the default
                              when keys 'id' and 'shape' are not set.
                'size'        [scalar] Diameter of the telescope dish (in meters)
                              if the key 'shape' is set to 'dish' or length of
                              the dipole if key 'shape' is set to 'dipole'. Will
                              be ignored if key 'shape' is set to 'delta'. Will
                              be ignored if key 'id' is set and a preset value
                              used for the diameter or dipole. Default = 25.0.
                'orientation' [list or numpy array] If key 'shape' is set to
                              dipole, it refers to the orientation of the dipole
                              element unit vector whose magnitude is specified by
                              length. If key 'shape' is set to 'dish', it refers
                              to the position on the sky to which the dish is
                              pointed. For a dipole, this unit vector must be
                              provided in the local ENU coordinate system aligned
                              with the direction cosines coordinate system or in
                              the Alt-Az coordinate system.
                              This could be a 2-element vector (transverse
                              direction cosines) where the third (line-of-sight)
                              component is determined, or a 3-element vector
                              specifying all three direction cosines or a two-
                              element coordinate in Alt-Az system. If not provided
                              it defaults to an eastward pointing dipole. If key
                              'shape' is set to 'dish', the orientation refers
                              to the pointing center of the dish on the sky. It
                              can be provided in Alt-Az system as a two-element
                              vector or in the direction cosine coordinate
                              system as a two- or three-element vector. If not
                              set in the case of a dish element, it defaults to
                              zenith. The coordinate system is specified by
                              the key 'ocoords'
                'ocoords'     [scalar string] specifies the coordinate system
                              for key 'orientation'. Accepted values are 'altaz'
                              and 'dircos'.
                'groundplane' [scalar] height of telescope element above the
                              ground plane (in meteres). Default = None will
                              denote no ground plane effects.
                'ground_modify'
                              [dictionary] contains specifications to modify
                              the analytically computed ground plane pattern. If
                              absent, the ground plane computed will not be
                              modified. If set, it may contain the following
                              keys:
                              'scale' [scalar] positive value to scale the
                                      modifying factor with. If not set, the
                                      scale factor to the modification is unity.
                              'max'   [scalar] positive value to clip the
                                      modified and scaled values to. If not set,
                                      there is no upper limit

    layout      [dictionary] contains array layout information (on the full
                array even if only a subset of antennas or baselines are used
                in the simulation). It contains the following keys and
                information:
                'positions' [numpy array] Antenna positions (in m) as a
                            nant x 3 array in coordinates specified by key
                            'coords'
                'coords'    [string] Coordinate system in which antenna
                            positions are specified. Currently accepts 'ENU'
                            for local ENU system
                'labels'    [list or numpy array of strings] Unique string
                            identifiers for antennas. Must be of same length
                            as nant.
                'ids'       [list or numpy array of integers] Unique integer
                            identifiers for antennas. Must be of same length
                            as nant.

    timestamp   [list] List of timestamps during the observation

    t_acc       [list] Accumulation time (sec) corresponding to each timestamp

    t_obs       [scalar] Total observing duration (sec)

    Tsys        [scalar, list or numpy vector] System temperature in Kelvin. At
                end of the simulation, it will be a numpy array of size
                n_baselines x nchan x n_snaps.

    Tsysinfo    [list of dictionaries] Contains a list of system temperature
                information for each timestamp of observation. Each dictionary
                element in the list following keys and values:
                'Trx'      [scalar] Recevier temperature (in K) that is
                           applicable to all frequencies and baselines
                'Tant'     [dictionary] contains antenna temperature info
                           from which the antenna temperature is estimated.
                           Used only if the key 'Tnet' is absent or set to
                           None. It has the following keys and values:
                           'f0'      [scalar] Reference frequency (in Hz)
                                     from which antenna temperature will
                                     be estimated (see formula below)
                           'T0'      [scalar] Antenna temperature (in K) at
                                     the reference frequency specified in
                                     key 'f0'. See formula below.
                           'spindex' [scalar] Antenna temperature spectral
                                     index. See formula below.

                           Tsys = Trx + Tant['T0'] * (f/Tant['f0'])**spindex

                'Tnet'     [numpy array] Pre-computed Tsys (in K)
                           information that will be used directly to set the
                           Tsys. If specified, the information under keys
                           'Trx' and 'Tant' will be ignored. If a scalar
                           value is provided, it will be assumed to be
                           identical for all interferometers and all
                           frequencies. If a vector is provided whose length
                           is equal to the number of interferoemters, it
                           will be assumed identical for all frequencies. If
                           a vector is provided whose length is equal to the
                           number of frequency channels, it will be assumed
                           identical for all interferometers. If a 2D array
                           is provided, it should be of size
                           n_baselines x nchan. Tsys = Tnet

    vis_freq    [numpy array] The simulated complex visibility (in Jy or K)
                observed by each of the interferometers along frequency axis for
                each timestamp of observation per frequency channel. It is the
                sum of skyvis_freq and vis_noise_freq. It can be either directly
                initialized or simulated in observe(). Same dimensions as
                skyvis_freq.

    vis_lag     [numpy array] The simulated complex visibility (in Jy Hz or K Hz)
                along delay axis for each interferometer obtained by FFT of
                vis_freq along frequency axis. Same size as vis_noise_lag and
                skyis_lag. It is evaluated in member function delay_transform().

    vis_noise_freq
                [numpy array] Complex visibility noise (in Jy or K) generated
                using an rms of vis_rms_freq along frequency axis for each
                interferometer which is then added to the generated sky
                visibility. Same dimensions as skyvis_freq. Used in the member
                function observe(). Read its docstring for more details.

    vis_noise_lag
                [numpy array] Complex visibility noise (in Jy Hz or K Hz) along
                delay axis for each interferometer generated using an FFT of
                vis_noise_freq along frequency axis. Same size as vis_noise_freq.
                Created in the member function delay_transform(). Read its
                docstring for more details.

    vis_rms_freq
                [list of float] Theoretically estimated thermal noise rms (in Jy
                or K) in visibility measurements. Same size as vis_freq. This
                will be estimated and used to inject simulated noise when a call
                to member function observe() is made. Read the  docstring of
                observe() for more details. The noise rms is estimated from the
                instrument parameters as:
                (2 k T_sys / (A_eff x sqrt(2 x channel_width x t_acc))) / Jy, or
                T_sys / sqrt(2 x channel_width x t_acc)

    simparms_file
                [string] Full path to filename containing simulation parameters
                in YAML format

    Member functions:

    __init__()          Initializes an instance of class InterferometerArray

    observe()           Simulates an observing run with the interferometer
                        specifications and an external sky catalog thus producing
                        visibilities. The simulation generates visibilities
                        observed by the interferometer for the specified
                        parameters.

    observing_run()     Simulate an extended observing run in 'track' or 'drift'
                        mode, by an instance of the InterferometerArray class, of
                        the sky when a sky catalog is provided. The simulation
                        generates visibilities observed by the interferometer
                        array for the specified parameters. Uses member function
                        observe() and builds the observation from snapshots. The
                        timestamp for each snapshot is the current time at which
                        the snapshot is generated.

    generate_noise()    Generates thermal noise from attributes that describe
                        system parameters which can be added to sky visibilities

    add_noise()         Adds the thermal noise generated in member function
                        generate_noise() to the sky visibilities after
                        extracting and applying complex instrument gains
                        
    apply_gradients()   Apply the perturbations in combination with the 
                        gradients to determine perturbed visibilities

    duplicate_measurements()
                        Duplicate visibilities based on redundant baselines 
                        specified. This saves time when compared to simulating 
                        visibilities over redundant baselines. Thus, it is more 
                        efficient to simulate unique baselines and duplicate 
                        measurements for redundant baselines

    getThreePointCombinations()
                        Return all unique 3-point combinations of baselines

    getClosurePhase()   Get closure phases of visibilities from triplets of 
                        antennas

    rotate_visibilities()
                        Centers the phase of visibilities around any given phase
                        center. Project baseline vectors with respect to a
                        reference point on the sky. Essentially a wrapper to
                        member functions phase_centering() and
                        project_baselines()

    phase_centering()   Centers the phase of visibilities around any given phase
                        center.

    project_baselines() Project baseline vectors with respect to a reference
                        point on the sky. Assigns the projected baselines to the
                        attribute projected_baselines

    conjugate()         Flips the baseline vectors and conjugates the visibilies
                        for a specified subset of baselines.

    delay_transform()  Transforms the visibilities from frequency axis onto
                       delay (time) axis using an IFFT. This is performed for
                       noiseless sky visibilities, thermal noise in visibilities,
                       and observed visibilities.

    concatenate()      Concatenates different visibility data sets from instances
                       of class InterferometerArray along baseline, frequency or
                       time axis.

    save()             Saves the interferometer array information to disk in
                       HDF5, FITS, NPZ and UVFITS formats

    write_uvfits()     Saves the interferometer array information to disk in
                       UVFITS format

    ----------------------------------------------------------------------------
    """

    def __init__(self, labels, baselines, channels, telescope=None, eff_Q=0.89,
                 latitude=34.0790, longitude=0.0, altitude=0.0,
                 skycoords='radec', A_eff=NP.pi*(25.0/2)**2,
                 pointing_coords='hadec', layout=None, blgroupinfo=None,
                 baseline_coords='localenu', freq_scale=None, gaininfo=None,
                 init_file=None, simparms_file=None):

        """
        ------------------------------------------------------------------------
        Intialize the InterferometerArray class which manages information on a
        multi-element interferometer.

        Class attributes initialized are:
        labels, baselines, channels, telescope, latitude, longitude, altitude,
        skycoords, eff_Q, A_eff, pointing_coords, baseline_coords,
        baseline_lengths, channels, bp, bp_wts, freq_resolution, lags, lst,
        obs_catalog_indices, pointing_center, skyvis_freq, skyvis_lag,
        timestamp, t_acc, Tsys, Tsysinfo, vis_freq, vis_lag, t_obs, n_acc,
        vis_noise_freq, vis_noise_lag, vis_rms_freq, geometric_delays,
        projected_baselines, simparms_file, layout, gradient, gradient_mode,
        gaininfo, blgroups, bl_reversemap

        Read docstring of class InterferometerArray for details on these
        attributes.

        Keyword input(s):

        init_file    [string] Location of the initialization file from which an
                     instance of class InterferometerArray will be created.
                     File format must be compatible with the one saved to disk
                     by member function save().

        simparms_file
                     [string] Location of the simulation parameters in YAML
                     format that went into making the simulated data product

        Other input parameters have their usual meanings. Read the docstring of
        class InterferometerArray for details on these inputs.
        ------------------------------------------------------------------------
        """

        argument_init = False
        init_file_success = False
        if init_file is not None:
            try:
                with h5py.File(init_file+'.hdf5', 'r') as fileobj:
                    self.simparms_file = None
                    self.latitude = 0.0
                    self.longitude = 0.0
                    self.altitude = 0.0
                    self.skycoords = 'radec'
                    self.flux_unit = 'JY'
                    self.telescope = {}
                    self.telescope['shape'] = 'delta'
                    self.telescope['size'] = 1.0
                    self.telescope['groundplane'] = None
                    self.Tsysinfo = []
                    self.layout = {}
                    self.blgroups = None
                    self.bl_reversemap = None
                    self.lags = None
                    self.vis_lag = None
                    self.skyvis_lag = None
                    self.vis_noise_lag = None
                    self.gradient_mode = None
                    self.gradient = {}
                    self.gaininfo = None
                    for key in ['header', 'telescope_parms', 'spectral_info', 'simparms', 'antenna_element', 'timing', 'skyparms', 'array', 'layout', 'instrument', 'visibilities', 'gradients', 'gaininfo', 'blgroupinfo']:
                        try:
                            grp = fileobj[key]
                        except KeyError:
                            if key in ['gradients', 'gaininfo']:
                                pass
                            elif key not in ['simparms', 'blgroupinfo']:
                                raise KeyError('Key {0} not found in init_file'.format(key))
                        if key == 'header':
                            self.flux_unit = grp['flux_unit'].value
                        if key == 'telescope_parms':
                            if 'latitude' in grp:
                                self.latitude = grp['latitude'].value
                            if 'longitude' in grp:
                                self.longitude = grp['longitude'].value
                            if 'altitude' in grp:
                                self.altitude = grp['altitude'].value
                            if 'id' in grp:
                                self.telescope['id'] = grp['id'].value
                        if key == 'layout':
                            if 'positions' in grp:
                                self.layout['positions'] = grp['positions'].value
                            else:
                                raise KeyError('Antenna layout positions is missing')
                            try:
                                self.layout['coords'] = grp['positions'].attrs['coords']
                            except KeyError:
                                raise KeyError('Antenna layout position coordinate system is missing')
                            if 'labels' in grp:
                                self.layout['labels'] = grp['labels'].value
                            else:
                                raise KeyError('Layout antenna labels is missing')
                            if 'ids' in grp:
                                self.layout['ids'] = grp['ids'].value
                            else:
                                raise KeyError('Layout antenna ids is missing')
                        if key == 'antenna_element':
                            if 'shape' in grp:
                                self.telescope['shape'] = grp['shape'].value
                            if 'size' in grp:
                                self.telescope['size'] = grp['size'].value
                            if 'ocoords' in grp:
                                self.telescope['ocoords'] = grp['ocoords'].value
                            else:
                                raise KeyError('Keyword "ocoords" not found in init_file')
                            if 'orientation' in grp:
                                self.telescope['orientation'] = grp['orientation'].value.reshape(1,-1)
                            else:
                                raise KeyError('Key "orientation" not found in init_file')
                            if 'groundplane' in grp:
                                self.telescope['groundplane'] = grp['groundplane'].value

                        if key == 'simparms':
                            if 'simfile' in grp:
                                self.simparms_file = grp['simfile'].value
                        if key == 'spectral_info':
                            self.freq_resolution = grp['freq_resolution'].value
                            self.channels = grp['freqs'].value
                            if 'lags' in grp:
                                self.lags = grp['lags'].value
                            if 'bp' in grp:
                                self.bp = grp['bp'].value
                            else:
                                raise KeyError('Key "bp" not found in init_file')
                            if 'bp_wts' in grp:
                                self.bp_wts = grp['bp_wts'].value
                            else:
                                self.bp_wts = NP.ones_like(self.bp)
                            self.bp_wts = grp['bp_wts'].value
                        if key == 'skyparms':
                            if 'pointing_coords' in grp:
                                self.pointing_coords = grp['pointing_coords'].value
                            if 'phase_center_coords' in grp:
                                self.phase_center_coords = grp['phase_center_coords'].value
                            if 'skycoords' in grp:
                                self.skycoords = grp['skycoords'].value
                            self.lst = grp['LST'].value
                            self.pointing_center = grp['pointing_center'].value
                            self.phase_center = grp['phase_center'].value
                        if key == 'timing':
                            if 'timestamps' in grp:
                                self.timestamp = grp['timestamps'].value.tolist()
                            else:
                                raise KeyError('Key "timestamps" not found in init_file')
                            if 't_acc' in grp:
                                self.t_acc = grp['t_acc'].value.tolist()
                                self.t_obs = grp['t_obs'].value
                                self.n_acc = grp['n_acc'].value
                            else:
                                raise KeyError('Key "t_acc" not found in init_file')

                        if key == 'instrument':
                            if ('Trx' in grp) and ('Tant' in grp) and ('spindex' in grp) and ('Tnet' in grp):
                                for ti in xrange(grp['Trx'].value.size):
                                    tsysinfo = {}
                                    tsysinfo['Trx'] = grp['Trx'].value[ti]
                                    tsysinfo['Tant'] = {'Tant0': grp['Tant0'].value[ti], 'f0': grp['f0'].value[ti], 'spindex': grp['spindex'].value[ti]}
                                    tsysinfo['Tnet'] = None
                                    self.Tsysinfo += [tsysinfo]
                            if 'Tsys' in grp:
                                self.Tsys = grp['Tsys'].value
                            else:
                                raise KeyError('Key "Tsys" not found in init_file')
                            if 'effective_area' in grp:
                                self.A_eff = grp['effective_area'].value
                            else:
                                raise KeyError('Key "effective_area" not found in init_file')
                            if 'efficiency' in grp:
                                self.eff_Q = grp['efficiency'].value
                            else:
                                raise KeyError('Key "effeciency" not found in init_file')

                        if key == 'array':
                            if 'labels' in grp:
                                self.labels = grp['labels'].value
                            else:
                                self.labels = ['B{0:0d}'.format(i+1) for i in range(self.baseline_lengths.size)]
                            if 'baselines' in grp:
                                self.baselines = grp['baselines'].value
                                self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
                            else:
                                raise KeyError('Key "baselines" not found in init_file')
                            if 'baseline_coords' in grp:
                                self.baseline_coords = grp['baseline_coords'].value
                            else:
                                self.baseline_coords = 'localenu'
                            if 'projected_baselines' in grp:
                                self.projected_baselines = grp['projected_baselines'].value

                        if key == 'visibilities':
                            if 'freq_spectrum' in grp:
                                subgrp = grp['freq_spectrum']
                                if 'rms' in subgrp:
                                    self.vis_rms_freq = subgrp['rms'].value
                                else:
                                    self.vis_rms_freq = None
                                    # raise KeyError('Key "rms" not found in init_file')
                                if 'vis' in subgrp:
                                    self.vis_freq = subgrp['vis'].value
                                else:
                                    self.vis_freq = None
                                if 'skyvis' in subgrp:
                                    self.skyvis_freq = subgrp['skyvis'].value
                                else:
                                    raise KeyError('Key "skyvis" not found in init_file')
                                if 'noise' in subgrp:
                                    self.vis_noise_freq = subgrp['noise'].value
                                else:
                                    self.vis_noise_freq = None
                            else:
                                raise KeyError('Key "freq_spectrum" not found in init_file')
                            if 'delay_spectrum' in grp:
                                subgrp = grp['delay_spectrum']
                                if 'vis' in subgrp:
                                    self.vis_lag = subgrp['vis'].value
                                if 'skyvis' in subgrp:
                                    self.skyvis_lag = subgrp['skyvis'].value
                                if 'noise' in subgrp:
                                    self.vis_noise_lag = subgrp['noise'].value

                        if key == 'gradients':
                            if key in fileobj:
                                for gradkey in grp:
                                    self.gradient_mode = gradkey
                                    self.gradient[gradkey] = grp[gradkey].value

                        if key == 'gaininfo':
                            if key in fileobj:
                                self.gaininfo = GainInfo(init_file=grp['gainsfile'].value)

                        if key == 'blgroupinfo':
                            if key in fileobj:
                                self.blgroups = {}
                                self.bl_reversemap = {}
                                for blkey in grp['groups']:
                                    self.blgroups[ast.literal_eval(blkey)] = grp['groups'][blkey].value
                                for blkey in grp['reversemap']:
                                    self.bl_reversemap[ast.literal_eval(blkey)] = grp['reversemap'][blkey].value
                                    

            except IOError: # Check if a FITS file is available
                try:
                    hdulist = fits.open(init_file+'.fits')
                except IOError:
                    argument_init = True
                    print '\tinit_file provided but could not open the initialization file. Attempting to initialize with input parameters...'

                extnames = [hdulist[i].header['EXTNAME'] for i in xrange(1,len(hdulist))]

                self.simparms_file = None
                if 'simparms' in hdulist[0].header:
                    if isinstance(hdulist[0].header['simparms'], str):
                        self.simparms_file = hdulist[0].header['simparms']
                    else:
                        warnings.warn('\tInvalid specification found in header for simulation parameters file. Proceeding with None as default.')

                try:
                    self.gradient_mode = hdulist[0].header['gradient_mode']
                except KeyError:
                    self.gradient_mode = None
                    self.gradient = {}

                try:
                    self.freq_resolution = hdulist[0].header['freq_resolution']
                except KeyError:
                    hdulist.close()
                    raise KeyError('Keyword "freq_resolution" not found in header.')

                try:
                    self.latitude = hdulist[0].header['latitude']
                except KeyError:
                    print '\tKeyword "latitude" not found in header. Assuming 34.0790 degrees for attribute latitude.'
                    self.latitude = 34.0790

                try:
                    self.longitude = hdulist[0].header['longitude']
                except KeyError:
                    print '\tKeyword "longitude" not found in header. Assuming 0.0 degrees for attribute longitude.'
                    self.longitude = 0.0

                try:
                    self.altitude = hdulist[0].header['altitude']
                except KeyError:
                    print '\tKeyword "altitude" not found in header. Assuming 0m for attribute altitude.'
                    self.altitude = 0.0
                self.telescope = {}
                if 'telescope' in hdulist[0].header:
                    self.telescope['id'] = hdulist[0].header['telescope']

                try:
                    self.telescope['shape'] = hdulist[0].header['element_shape']
                except KeyError:
                    print '\tKeyword "element_shape" not found in header. Assuming "delta" for attribute antenna element shape.'
                    self.telescope['shape'] = 'delta'

                try:
                    self.telescope['size'] = hdulist[0].header['element_size']
                except KeyError:
                    print '\tKeyword "element_size" not found in header. Assuming 25.0m for attribute antenna element size.'
                    self.telescope['size'] = 1.0

                try:
                    self.telescope['ocoords'] = hdulist[0].header['element_ocoords']
                except KeyError:
                    raise KeyError('\tKeyword "element_ocoords" not found in header. No defaults.')

                try:
                    self.telescope['groundplane'] = hdulist[0].header['groundplane']
                except KeyError:
                    self.telescope['groundplane'] = None

                if 'ANTENNA ELEMENT ORIENTATION' not in extnames:
                    raise KeyError('No extension found containing information on element orientation.')
                else:
                    self.telescope['orientation'] = hdulist['ANTENNA ELEMENT ORIENTATION'].data.reshape(1,-1)

                try:
                    self.baseline_coords = hdulist[0].header['baseline_coords']
                except KeyError:
                    print '\tKeyword "baseline_coords" not found in header. Assuming "localenu" for attribute baseline_coords.'
                    self.baseline_coords = 'localenu'

                try:
                    self.pointing_coords = hdulist[0].header['pointing_coords']
                except KeyError:
                    print '\tKeyword "pointing_coords" not found in header. Assuming "hadec" for attribute pointing_coords.'
                    self.pointing_coords = 'hadec'

                try:
                    self.phase_center_coords = hdulist[0].header['phase_center_coords']
                except KeyError:
                    print '\tKeyword "phase_center_coords" not found in header. Assuming "hadec" for attribute phase_center_coords.'
                    self.phase_center_coords = 'hadec'

                try:
                    self.skycoords = hdulist[0].header['skycoords']
                except KeyError:
                    print '\tKeyword "skycoords" not found in header. Assuming "radec" for attribute skycoords.'
                    self.skycoords = 'radec'

                try:
                    self.flux_unit = hdulist[0].header['flux_unit']
                except KeyError:
                    print '\tKeyword "flux_unit" not found in header. Assuming "jy" for attribute flux_unit.'
                    self.flux_unit = 'JY'

                if 'POINTING AND PHASE CENTER INFO' not in extnames:
                    raise KeyError('No extension table found containing pointing information.')
                else:
                    self.lst = hdulist['POINTING AND PHASE CENTER INFO'].data['LST'].tolist()
                    self.pointing_center = NP.hstack((hdulist['POINTING AND PHASE CENTER INFO'].data['pointing_longitude'].reshape(-1,1), hdulist['POINTING AND PHASE CENTER INFO'].data['pointing_latitude'].reshape(-1,1)))
                    self.phase_center = NP.hstack((hdulist['POINTING AND PHASE CENTER INFO'].data['phase_center_longitude'].reshape(-1,1), hdulist['POINTING AND PHASE CENTER INFO'].data['phase_center_latitude'].reshape(-1,1)))

                if 'TIMESTAMPS' in extnames:
                    self.timestamp = hdulist['TIMESTAMPS'].data['timestamps'].tolist()
                else:
                    raise KeyError('Extension named "TIMESTAMPS" not found in init_file.')

                self.Tsysinfo = []
                if 'TSYSINFO' in extnames:
                    self.Tsysinfo = [{'Trx': elem['Trx'], 'Tant': {'T0': elem['Tant0'], 'f0': elem['f0'], 'spindex': elem['spindex']}, 'Tnet': None} for elem in hdulist['TSYSINFO'].data]

                if 'TSYS' in extnames:
                    self.Tsys = hdulist['Tsys'].data
                else:
                    raise KeyError('Extension named "Tsys" not found in init_file.')

                if 'BASELINES' in extnames:
                    self.baselines = hdulist['BASELINES'].data.reshape(-1,3)
                    self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
                else:
                    raise KeyError('Extension named "BASELINES" not found in init_file.')

                if 'PROJ_BASELINES' in extnames:
                    self.projected_baselines = hdulist['PROJ_BASELINES'].data

                if 'LABELS' in extnames:
                    # self.labels = hdulist['LABELS'].data.tolist()
                    a1 = hdulist['LABELS'].data['A1']
                    a2 = hdulist['LABELS'].data['A2']
                    self.labels = zip(a2,a1)
                else:
                    self.labels = ['B{0:0d}'.format(i+1) for i in range(self.baseline_lengths.size)]

                self.layout = {}
                if 'LAYOUT' in extnames:
                    for key in ['positions', 'ids', 'labels']:
                        self.layout[key] = hdulist['LAYOUT'].data[key]
                    self.layout['coords'] = hdulist['LAYOUT'].header['COORDS']

                if 'EFFECTIVE AREA' in extnames:
                    self.A_eff = hdulist['EFFECTIVE AREA'].data
                else:
                    raise KeyError('Extension named "EFFECTIVE AREA" not found in init_file.')

                if 'INTERFEROMETER EFFICIENCY' in extnames:
                    self.eff_Q = hdulist['INTERFEROMETER EFFICIENCY'].data
                else:
                    raise KeyError('Extension named "INTERFEROMETER EFFICIENCY" not found in init_file.')

                if 'SPECTRAL INFO' not in extnames:
                    raise KeyError('No extension table found containing spectral information.')
                else:
                    self.channels = hdulist['SPECTRAL INFO'].data['frequency']
                    try:
                        self.lags = hdulist['SPECTRAL INFO'].data['lag']
                    except KeyError:
                        self.lags = None

                if 'BANDPASS' in extnames:
                    self.bp = hdulist['BANDPASS'].data
                else:
                    raise KeyError('Extension named "BANDPASS" not found in init_file.')

                if 'BANDPASS_WEIGHTS' in extnames:
                    self.bp_wts = hdulist['BANDPASS_WEIGHTS'].data
                else:
                    self.bp_wts = NP.ones_like(self.bp)

                if 'T_ACC' in extnames:
                    self.t_acc = hdulist['t_acc'].data.tolist()
                    self.n_acc = len(self.t_acc)
                    self.t_obs = sum(self.t_acc)
                else:
                    raise KeyError('Extension named "T_ACC" not found in init_file.')

                if 'FREQ_CHANNEL_NOISE_RMS_VISIBILITY' in extnames:
                    self.vis_rms_freq = hdulist['freq_channel_noise_rms_visibility'].data
                else:
                    self.vis_rms_freq = None

                if 'REAL_FREQ_OBS_VISIBILITY' in extnames:
                    self.vis_freq = hdulist['real_freq_obs_visibility'].data
                    if 'IMAG_FREQ_OBS_VISIBILITY' in extnames:
                        self.vis_freq = self.vis_freq.astype(NP.complex128)
                        self.vis_freq += 1j * hdulist['imag_freq_obs_visibility'].data
                else:
                    self.vis_freq = None

                if 'REAL_FREQ_SKY_VISIBILITY' in extnames:
                    self.skyvis_freq = hdulist['real_freq_sky_visibility'].data
                    if 'IMAG_FREQ_SKY_VISIBILITY' in extnames:
                        self.skyvis_freq = self.skyvis_freq.astype(NP.complex128)
                        self.skyvis_freq += 1j * hdulist['imag_freq_sky_visibility'].data
                else:
                    raise KeyError('Extension named "REAL_FREQ_SKY_VISIBILITY" not found in init_file.')

                if 'REAL_FREQ_NOISE_VISIBILITY' in extnames:
                    self.vis_noise_freq = hdulist['real_freq_noise_visibility'].data
                    if 'IMAG_FREQ_NOISE_VISIBILITY' in extnames:
                        self.vis_noise_freq = self.vis_noise_freq.astype(NP.complex128)
                        self.vis_noise_freq += 1j * hdulist['imag_freq_noise_visibility'].data
                else:
                    self.vis_noise_freq = None

                if self.gradient_mode is not None:
                    self.gradient = {}
                    if 'real_freq_sky_visibility_gradient_wrt_{0}'.format(self.gradient_mode) in extnames:
                        self.gradient[self.gradient_mode] = hdulist['real_freq_sky_visibility_gradient_wrt_{0}'.format(self.gradient_mode)].data
                        if 'imag_freq_sky_visibility_gradient_wrt_{0}'.format(self.gradient_mode) in extnames:
                            self.gradient[self.gradient_mode] = self.gradient[self.gradient_mode].astype(NP.complex128)
                            self.gradient[self.gradient_mode] += 1j * hdulist['imag_freq_sky_visibility_gradient_wrt_{0}'.format(self.gradient_mode)].data

                try:
                    gainsfile = hdulist[0].header['gainsfile']
                except KeyError:
                    print '\tKeyword "gainsfile" not found in header. Assuming default unity gains.'
                    self.gaininfo = None
                else:
                    self.gaininfo = GainInfo(init_file=gainsfile, axes_order=['label', 'frequency', 'time'])

                if 'REAL_LAG_VISIBILITY' in extnames:
                    self.vis_lag = hdulist['real_lag_visibility'].data
                    if 'IMAG_LAG_VISIBILITY' in extnames:
                        self.vis_lag = self.vis_lag.astype(NP.complex128)
                        self.vis_lag += 1j * hdulist['imag_lag_visibility'].data
                else:
                    self.vis_lag = None

                if 'REAL_LAG_SKY_VISIBILITY' in extnames:
                    self.skyvis_lag = hdulist['real_lag_sky_visibility'].data
                    if 'IMAG_LAG_SKY_VISIBILITY' in extnames:
                        self.skyvis_lag = self.skyvis_lag.astype(NP.complex128)
                        self.skyvis_lag += 1j * hdulist['imag_lag_sky_visibility'].data
                else:
                    self.skyvis_lag = None

                if 'REAL_LAG_NOISE_VISIBILITY' in extnames:
                    self.vis_noise_lag = hdulist['real_lag_noise_visibility'].data
                    if 'IMAG_LAG_NOISE_VISIBILITY' in extnames:
                        self.vis_noise_lag = self.vis_noise_lag.astype(NP.complex128)
                        self.vis_noise_lag += 1j * hdulist['imag_lag_noise_visibility'].data
                else:
                    self.vis_noise_lag = None

                hdulist.close()
            init_file_success = True
            return
        else:
            argument_init = True

        if (not argument_init) and (not init_file_success):
            raise ValueError('Initialization failed with the use of init_file.')

        self.baselines = NP.asarray(baselines)
        if len(self.baselines.shape) == 1:
            if self.baselines.size == 2:
                self.baselines = NP.hstack((self.baselines.reshape(1,-1), NP.zeros(1)))
            elif self.baselines.size == 3:
                self.baselines = self.baselines.reshape(1,-1)
            else:
                raise ValueError('Baseline(s) must be a 2- or 3-column array.')
        elif len(self.baselines.shape) == 2:
            if self.baselines.shape[1] == 2:
                self.baselines = NP.hstack((self.baselines, NP.zeros(self.baselines.shape[0]).reshape(-1,1)))
            elif self.baselines.shape[1] != 3:
                raise ValueError('Baseline(s) must be a 2- or 3-column array')
        else:
            raise ValueError('Baseline(s) array contains more than 2 dimensions.')

        self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
        self.baseline_orientations = NP.angle(self.baselines[:,0] + 1j * self.baselines[:,1])
        self.projected_baselines = None

        if not isinstance(labels, (list, tuple)):
            raise TypeError('Interferometer array labels must be a list or tuple of unique identifiers')
        elif len(labels) != self.baselines.shape[0]:
            raise ValueError('Number of labels do not match the number of baselines specified.')
        else:
            self.labels = labels

        self.simparms_file = None
        if isinstance(simparms_file, str):
            self.simparms_file = simparms_file
        else:
            warnings.warn('\tInvalid specification found in header for simulation parameters file. Proceeding with None as default.')

        if isinstance(telescope, dict):
            self.telescope = telescope
        else:
            self.telescope = {}
            self.telescope['id'] = 'vla'
            self.telescope['shape'] = 'dish'
            self.telescope['size'] = 25.0
            self.telescope['ocoords'] = 'altaz'
            self.telescope['orientation'] = NP.asarray([90.0, 270.0]).reshape(1,-1)
            self.telescope['groundplane'] = None

        self.layout = {}
        if isinstance(layout, dict):
            if 'positions' in layout:
                if isinstance(layout['positions'], NP.ndarray):
                    if layout['positions'].ndim == 2:
                        if (layout['positions'].shape[1] == 2) or (layout['positions'].shape[1] == 3):
                            if layout['positions'].shape[1] == 2:
                                layout['positions'] = NP.hstack((layout['positions'], NP.zeros(layout['positions'].shape[0]).reshape(-1,1)))
                            self.layout['positions'] = layout['positions']
                        else:
                            raise ValueError('Incompatible shape in array layout')
                    else:
                        raise ValueError('Incompatible shape in array layout')
                else:
                    raise TypeError('Array layout positions must be a numpy array')
            else:
                raise KeyError('Array layout positions missing')
            if 'coords' in layout:
                if isinstance(layout['coords'], str):
                    self.layout['coords'] = layout['coords']
                else:
                    raise TypeError('Array layout coordinates must be a string')
            else:
                raise KeyError('Array layout coordinates missing')
            if 'labels' in layout:
                if isinstance(layout['labels'], (list,NP.ndarray)):
                    self.layout['labels'] = layout['labels']
                else:
                    raise TypeError('Array antenna labels must be a list or numpy array')
            else:
                raise KeyError('Array antenna labels missing')
            if 'ids' in layout:
                if isinstance(layout['ids'], (list,NP.ndarray)):
                    self.layout['ids'] = layout['ids']
                else:
                    raise TypeError('Array antenna ids must be a list or numpy array')
            else:
                raise KeyError('Array antenna ids missing')
            if (layout['positions'].shape[0] != layout['labels'].size) or (layout['ids'].size != layout['labels'].size):
                raise ValueError('Antenna layout positions, labels and IDs must all be for same number of antennas')

        self.blgroups = None
        self.bl_reversemap = None
        if blgroupinfo is not None:
            if not isinstance(blgroupinfo, dict):
                raise TypeError('Input blgroupinfo must be a dictionary')
            self.blgroups = blgroupinfo['groups']
            self.bl_reversemap = blgroupinfo['reversemap']

        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.vis_freq = None
        self.skyvis_freq = None
        self.vis_noise_freq = None
        self.gradient_mode = None
        self.gradient = {}
        self.gaininfo = None
        if gaininfo is not None:
            if not isinstance(gaininfo, GainInfo):
                raise TypeError('Input gaininfo must be an instance of class GainInfo')
            self.gaininfo = gaininfo

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            self.channels = NP.asarray(channels)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            self.channels = NP.asarray(channels) * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            self.channels = NP.asarray(channels) * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            self.channels = NP.asarray(channels) * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')

        self.bp = NP.ones((self.baselines.shape[0],self.channels.size)) # Inherent bandpass shape
        self.bp_wts = NP.ones((self.baselines.shape[0],self.channels.size)) # Additional bandpass weights
        self.lag_kernel = DSP.FT1D(self.bp*self.bp_wts, ax=1, inverse=True, use_real=False, shift=True)

        self.Tsys = NP.zeros((self.baselines.shape[0],self.channels.size))
        self.Tsysinfo = []

        self.flux_unit = 'JY'
        self.timestamp = []
        self.t_acc = []
        self.t_obs = 0.0
        self.n_acc = 0
        self.pointing_center = NP.empty([1,2])
        self.phase_center = NP.empty([1,2])
        self.lst = []

        if isinstance(eff_Q, (int, float)):
            if (eff_Q >= 0.0) or (eff_Q <= 1.0):
                self.eff_Q = eff_Q * NP.ones((self.baselines.shape[0], self.channels.size))
            else:
                raise ValueError('Efficiency value of interferometer is invalid.')
        elif isinstance(eff_Q, (list, tuple, NP.ndarray)):
            eff_Q = NP.asarray(eff_Q)
            if (NP.any(eff_Q < 0.0)) or (NP.any(eff_Q > 1.0)):
                raise ValueError('One or more values of eff_Q found to be outside the range [0,1].')
            if eff_Q.size == self.baselines.shape[0]:
                self.eff_Q = NP.repeat(eff_Q.reshape(-1,1), self.channels.size, axis=1)
            elif eff_Q.size == self.channels.size:
                self.eff_Q = NP.repeat(eff_Q.reshape(1,-1), self.channels.size, axis=0)
            elif eff_Q.size == self.baselines.shape[0]*self.channels.size:
                self.eff_Q = eff_Q.reshape(-1,self.channels.size)
            else:
                raise ValueError('Efficiency values of interferometers incompatible with the number of interferometers and/or frequency channels.')
        else:
            raise TypeError('Efficiency values of interferometers must be provided as a scalar, list, tuple or numpy array.')

        if isinstance(A_eff, (int, float)):
            if A_eff >= 0.0:
                self.A_eff = A_eff * NP.ones((self.baselines.shape[0], self.channels.size))
            else:
                raise ValueError('Negative value for effective area is invalid.')
        elif isinstance(A_eff, (list, tuple, NP.ndarray)):
            A_eff = NP.asarray(A_eff)
            if NP.any(A_eff < 0.0):
                raise ValueError('One or more values of A_eff found to be negative.')
            if A_eff.size == self.baselines.shape[0]:
                self.A_eff = NP.repeat(A_eff.reshape(-1,1), self.channels.size, axis=1)
            elif A_eff.size == self.channels.size:
                self.A_eff = NP.repeat(A_eff.reshape(1,-1), self.channels.size, axis=0)
            elif A_eff.size == self.baselines.shape[0]*self.channels.size:
                self.A_eff = A_eff.reshape(-1,self.channels.size)
            else:
                raise ValueError('Effective area(s) of interferometers incompatible with the number of interferometers and/or frequency channels.')
        else:
            raise TypeError('Effective area(s) of interferometers must be provided as a scalar, list, tuple or numpy array.')

        self.vis_rms_freq = None
        self.freq_resolution = self.channels[1] - self.channels[0]
        self.baseline_coords = baseline_coords
        self.lags = None
        self.skyvis_lag = None
        self.vis_noise_lag = None
        self.vis_lag = None
        self.obs_catalog_indices = []
        self.geometric_delays = []

        if (pointing_coords == 'radec') or (pointing_coords == 'hadec') or (pointing_coords == 'altaz'):
            self.pointing_coords = pointing_coords
            self.phase_center_coords = pointing_coords
        else:
            raise ValueError('Pointing center of the interferometer must be "radec", "hadec" or "altaz". Check inputs.')

        if (skycoords == 'radec') or (skycoords == 'hadec') or (skycoords == 'altaz'):
            self.skycoords = skycoords
        else:
            raise ValueError('Sky coordinates must be "radec", "hadec" or "altaz". Check inputs.')

        if (baseline_coords == 'equatorial') or (baseline_coords == 'localenu'):
            self.baseline_coords = baseline_coords
        else:
            raise ValueError('Baseline coordinates must be "equatorial" or "local". Check inputs.')

    #############################################################################

    def observe(self, timestamp, Tsysinfo, bandpass, pointing_center, skymodel,
                t_acc, pb_info=None, brightness_units=None, bpcorrect=None,
                roi_info=None, roi_radius=None, roi_center=None, lst=None,
                gradient_mode=None, memsave=False):

        """
        -------------------------------------------------------------------------
        Simulate a snapshot observation, by an instance of the
        InterferometerArray class, of the sky when a sky catalog is provided. The
        simulation generates visibilities observed by the interferometers for the
        specified parameters. See member function observing_run() for simulating
        an extended observing run in 'track' or 'drift' mode.

        Inputs:

        timestamp    [scalar] Timestamp associated with each integration in the
                     observation

        Tsysinfo     [dictionary] Contains system temperature information for
                     specified timestamp of observation. It contains the
                     following keys and values:
                     'Trx'      [scalar] Recevier temperature (in K) that is
                                applicable to all frequencies and baselines
                     'Tant'     [dictionary] contains antenna temperature info
                                from which the antenna temperature is estimated.
                                Used only if the key 'Tnet' is absent or set to
                                None. It has the following keys and values:
                                'f0'      [scalar] Reference frequency (in Hz)
                                          from which antenna temperature will
                                          be estimated (see formula below)
                                'T0'      [scalar] Antenna temperature (in K) at
                                          the reference frequency specified in
                                          key 'f0'. See formula below.
                                'spindex' [scalar] Antenna temperature spectral
                                          index. See formula below.

                                Tsys = Trx + Tant['T0'] * (f/Tant['f0'])**spindex

                     'Tnet'     [numpy array] Pre-computed Tsys (in K)
                                information that will be used directly to set the
                                Tsys. If specified, the information under keys
                                'Trx' and 'Tant' will be ignored. If a scalar
                                value is provided, it will be assumed to be
                                identical for all interferometers and all
                                frequencies. If a vector is provided whose length
                                is equal to the number of interferoemters, it
                                will be assumed identical for all frequencies. If
                                a vector is provided whose length is equal to the
                                number of frequency channels, it will be assumed
                                identical for all interferometers. If a 2D array
                                is provided, it should be of size
                                n_baselines x nchan. Tsys = Tnet

        bandpass     [numpy array] Bandpass weights associated with the
                     interferometers for the specified timestamp of observation

        pointing_center
                     [2-element numpy vector or list] Pointing center (latitude
                     and longitude) of the observation at a given timestamp.
                     This is where the telescopes will be phased up to as
                     reference. Coordinate system for the pointing_center is
                     specified by the attribute pointing_coords initialized in
                     __init__().

        skymodel     [instance of class SkyModel] It consists of source flux
                     densities, their positions, and spectral indices. Read
                     class SkyModel docstring for more information.

        t_acc        [scalar] Accumulation time (sec) corresponding to timestamp

        brightness_units
                     [string] Units of flux density in the catalog and for the
                     generated visibilities. Accepted values are 'Jy' (Jansky)
                     and 'K' (Kelvin for temperature). If None set, it defaults
                     to 'Jy'

        Keyword Inputs:

        roi_info     [instance of class ROI_parameters] It consists of indices
                     in the polskymodel object, polarized beams for different
                     baseline types for every time stamp that will be simulated

        roi_radius   [scalar] Radius of the region of interest (degrees) inside
                     which sources are to be observed. Default = 90 degrees,
                     which is the entire horizon.

        roi_center   [string] Center of the region of interest around which
                     roi_radius is used. Accepted values are 'pointing_center'
                     and 'zenith'. If set to None, it defaults to 'zenith'.

        lst          [scalar] LST (in degrees) associated with the timestamp

        gradient_mode
                     [string] If set to None, visibilities will be simulated as
                     usual. If set to string, both visibilities and visibility
                     gradients with respect to the quantity specified in the
                     string will be simulated. Currently accepted value is
                     'baseline'. Plan to incorporate gradients with respect to
                     'skypos' and 'frequency' as well in the future.

        memsave      [boolean] If set to True, enforce computations in single
                     precision, otherwise enforce double precision (default)
        ------------------------------------------------------------------------
        """

        if len(bandpass.shape) == 1:
            if bandpass.size != self.channels.size:
                raise ValueError('Specified bandpass incompatible with the number of frequency channels')

            if len(self.bp.shape) == 2:
                self.bp = NP.expand_dims(NP.repeat(bandpass.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
            else:
                self.bp = NP.dstack((self.bp, NP.repeat(bandpass.reshape(1,-1), self.baselines.shape[0], axis=0)))
        elif len(bandpass.shape) == 2:
            if bandpass.shape[1] != self.channels.size:
                raise ValueError('Specified bandpass incompatible with the number of frequency channels')
            elif bandpass.shape[0] != self.baselines.shape[0]:
                raise ValueError('Specified bandpass incompatible with the number of interferometers')

            if len(self.bp.shape) == 2:
                self.bp = NP.expand_dims(bandpass, axis=2)
            else:
                self.bp = NP.dstack((self.bp, bandpass))
        elif len(bandpass.shape) == 3:
            if bandpass.shape[1] != self.channels.size:
                raise ValueError('Specified bandpass incompatible with the number of frequency channels')
            elif bandpass.shape[0] != self.baselines.shape[0]:
                raise ValueError('Specified bandpass incompatible with the number of interferometers')
            elif bandpass.shape[2] != 1:
                raise ValueError('Bandpass can have only one layer for this instance of accumulation.')

            if len(self.bp.shape) == 2:
                self.bp = bandpass
            else:
                self.bp = NP.dstack((self.bp, bandpass))

        self.bp_wts = NP.ones_like(self.bp) # All additional bandpass shaping weights are set to unity.

        if isinstance(Tsysinfo, dict):
            set_Tsys = False
            if 'Tnet' in Tsysinfo:
                if Tsysinfo['Tnet'] is not None:
                    Tsys = Tsysinfo['Tnet']
                    set_Tsys = True
            if not set_Tsys:
                try:
                    Tsys = Tsysinfo['Trx'] + Tsysinfo['Tant']['T0'] * (self.channels/Tsysinfo['Tant']['f0']) ** Tsysinfo['Tant']['spindex']
                except KeyError:
                    raise KeyError('One or more keys not found in input Tsysinfo')
                Tsys = Tsys.reshape(1,-1) + NP.zeros(self.baselines.shape[0]).reshape(-1,1) # nbl x nchan
        else:
            raise TypeError('Input Tsysinfo must be a dictionary')

        self.Tsysinfo += [Tsysinfo]
        if bpcorrect is not None:
            if not isinstance(bpcorrect, NP.ndarray):
                raise TypeError('Input specifying bandpass correction must be a numpy array')
            if bpcorrect.size == self.channels.size:
                bpcorrect = bpcorrect.reshape(1,-1)
            elif bpcorrect.size == self.baselines.shape[0]:
                bpcorrect = bpcorrect.reshape(-1,1)
            elif bpcorrect.size == self.baselines.shape[0] * self.channels.size:
                bpcorrect = bpcorrect.reshape(-1,self.channels.size)
            else:
                raise ValueError('Input bpcorrect has dimensions incompatible with the number of baselines and frequencies')
            Tsys = Tsys * bpcorrect

        if isinstance(Tsys, (int,float)):
            if Tsys < 0.0:
                raise ValueError('Tsys found to be negative.')

            if len(self.Tsys.shape) == 2:
                self.Tsys = Tsys + NP.zeros((self.baselines.shape[0], self.channels.size, 1))
            else:
                self.Tsys = NP.dstack((self.Tsys, Tsys + NP.zeros((self.baselines.shape[0], self.channels.size, 1))))
        elif isinstance(Tsys, (list, tuple, NP.ndarray)):
            Tsys = NP.asarray(Tsys)
            if NP.any(Tsys < 0.0):
                raise ValueError('Tsys should be non-negative.')

            if Tsys.size == self.baselines.shape[0]:
                if self.Tsys.ndim == 2:
                    self.Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(-1,1), self.channels.size, axis=1), axis=2)
                elif self.Tsys.ndim == 3:
                    self.Tsys = NP.dstack((self.Tsys, NP.expand_dims(NP.repeat(Tsys.reshape(-1,1), self.channels.size, axis=1), axis=2)))
            elif Tsys.size == self.channels.size:
                if self.Tsys.ndim == 2:
                    self.Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
                elif self.Tsys.ndim == 3:
                    self.Tsys = NP.dstack((self.Tsys, NP.expand_dims(NP.repeat(Tsys.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)))
            elif Tsys.size == self.baselines.shape[0]*self.channels.size:
                if self.Tsys.ndim == 2:
                    self.Tsys = NP.expand_dims(Tsys.reshape(-1,self.channels.size), axis=2)
                elif self.Tsys.ndim == 3:
                    self.Tsys = NP.dstack((self.Tsys, NP.expand_dims(Tsys.reshape(-1,self.channels.size), axis=2)))
            else:
                raise ValueError('Specified Tsys has incompatible dimensions with the number of baselines and/or number of frequency channels.')
        else:
            raise TypeError('Tsys should be a scalar, list, tuple, or numpy array')

        # if (brightness_units is None) or (brightness_units=='Jy') or (brightness_units=='JY') or (brightness_units=='jy'):
        #     if self.vis_rms_freq is None:
        #         self.vis_rms_freq = 2.0 * FCNST.k / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.A_eff/self.eff_Q, axis=2) / CNST.Jy
        #     elif len(self.vis_rms_freq.shape) == 3:
        #         self.vis_rms_freq = NP.dstack((self.vis_rms_freq, 2.0 * FCNST.k / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.A_eff/self.eff_Q, axis=2)/CNST.Jy))
        #     self.flux_unit = 'JY'
        # elif (brightness_units=='K') or (brightness_units=='k'):
        #     if len(self.vis_rms_freq.shape) == 2:
        #         self.vis_rms_freq = 1 / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.eff_Q, axis=2)
        #     elif len(self.vis_rms_freq.shape) == 3:
        #         self.vis_rms_freq = NP.dstack((self.vis_rms_freq, 1 / NP.sqrt(2.0*t_acc*self.freq_resolution) * NP.expand_dims(self.Tsys[:,:,-1]/self.eff_Q, axis=2)))
        #     self.flux_unit = 'K'
        # else:
        #     raise ValueError('Invalid brightness temperature units specified.')

        if not self.timestamp:
            self.pointing_center = NP.asarray(pointing_center).reshape(1,-1)
            self.phase_center = NP.asarray(pointing_center).reshape(1,-1)
        else:
            self.pointing_center = NP.vstack((self.pointing_center, NP.asarray(pointing_center).reshape(1,-1)))
            self.phase_center = NP.vstack((self.phase_center, NP.asarray(pointing_center).reshape(1,-1)))

        pointing_lon = self.pointing_center[-1,0]
        pointing_lat = self.pointing_center[-1,1]

        if self.skycoords == 'radec':
            if self.pointing_coords == 'hadec':
                if lst is not None:
                    pointing_lon = lst - self.pointing_center[-1,0]
                    pointing_lat = self.pointing_center[-1,1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in RA-Dec format while pointing center is in HA-Dec format.')
            elif self.pointing_coords == 'altaz':
                pointing_lonlat = GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = lst - pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]
        elif self.skycoords == 'hadec':
            if self.pointing_coords == 'radec':
                if lst is not None:
                    pointing_lon = lst - self.pointing_center[-1,0]
                    pointing_lat = self.pointing_center[-1,1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in RA-Dec format while pointing center is in HA-Dec format.')
            elif self.pointing_coords == 'altaz':
                pointing_lonlat = lst - GEOM.altaz2hadec(self.pointing_center[-1,:], self.latitude, units='degrees')
                pointing_lon = pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]
        else:
            if self.pointing_coords == 'radec':
                if lst is not None:
                    pointing_lonlat = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
                    pointing_lon = pointing_lonlat[0]
                    pointing_lat = pointing_lonlat[1]
                else:
                    raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')
            elif self.pointing_coords == 'hadec':
                pointing_lonlat = GEOM.hadec2altaz(self.pointing_center,
                                                   self.latitude,
                                                   units='degrees')
                pointing_lon = pointing_lonlat[0]
                pointing_lat = pointing_lonlat[1]

        baselines_in_local_frame = self.baselines
        if self.baseline_coords == 'equatorial':
            baselines_in_local_frame = GEOM.xyz2enu(self.baselines, self.latitude, 'degrees')

        pc_altaz = self.pointing_center[-1,:] # Convert pointing center to Alt-Az coordinates
        if self.pointing_coords == 'hadec':
            pc_altaz = GEOM.hadec2altaz(self.pointing_center[-1,:], self.latitude, units='degrees')
        elif self.pointing_coords == 'radec':
            if lst is not None:
                pc_altaz = GEOM.hadec2altaz(NP.asarray([lst-self.pointing_center[-1,0], self.pointing_center[-1,1]]), self.latitude, units='degrees')
            else:
                raise ValueError('LST must be provided. Sky coordinates are in Alt-Az format while pointing center is in RA-Dec format.')

        pc_dircos = GEOM.altaz2dircos(pc_altaz, 'degrees') # Convert pointing center to direction cosine coordinates
        pc_delay_offsets = DLY.geometric_delay(baselines_in_local_frame, pc_dircos, altaz=False, hadec=False, dircos=True, latitude=self.latitude)
        if memsave:
            pc_delay_offsets = pc_delay_offsets.astype(NP.float32)

        # pointing_phase = 2.0 * NP.pi * NP.repeat(NP.dot(baselines_in_local_frame, pc_dircos.reshape(-1,1)), self.channels.size, axis=1) * NP.repeat(self.channels.reshape(1,-1), self.baselines.shape[0], axis=0)/FCNST.c

        if not isinstance(skymodel, SM.SkyModel):
            raise TypeError('skymodel should be an instance of class SkyModel.')

        if self.skycoords == 'hadec':
            skypos_altaz = GEOM.hadec2altaz(skymodel.location, self.latitude, units='degrees')
        elif self.skycoords == 'radec':
            skypos_altaz = GEOM.hadec2altaz(NP.hstack((NP.asarray(lst-skymodel.location[:,0]).reshape(-1,1), skymodel.location[:,1].reshape(-1,1))), self.latitude, units='degrees')

        pb = None
        if roi_info is not None:
            if ('ind' not in roi_info) or ('pbeam' not in roi_info):
                raise KeyError('Both "ind" and "pbeam" keys must be present in dictionary roi_info')

            if (roi_info['ind'] is not None) and (roi_info['pbeam'] is not None):
                try:
                    pb = roi_info['pbeam'].reshape(-1,len(self.channels))
                except ValueError:
                    raise ValueError('Number of columns of primary beam in key "pbeam" of dictionary roi_info must be equal to number of frequency channels.')

                if NP.asarray(roi_info['ind']).size == pb.shape[0]:
                    m2 = roi_info['ind']
                else:
                    raise ValueError('Values in keys ind and pbeam in must carry same number of elements.')
        else:
            if roi_radius is None:
                roi_radius = 90.0

            if roi_center is None:
                roi_center = 'zenith'
            elif (roi_center != 'zenith') and (roi_center != 'pointing_center'):
                raise ValueError('Center of region of interest, roi_center, must be set to "zenith" or "pointing_center".')

            if roi_center == 'pointing_center':
                m1, m2, d12 = GEOM.spherematch(pointing_lon, pointing_lat, skymodel.location[:,0], skymodel.location[:,1], roi_radius, maxmatches=0)
            else: # roi_center = 'zenith'
                m2 = NP.arange(skypos_altaz.shape[0])
                m2 = m2[NP.where(skypos_altaz[:,0] >= 90.0-roi_radius)] # select sources whose altitude (angle above horizon) is 90-roi_radius

        if memsave:
            datatype = NP.complex64
        else:
            datatype = NP.complex128
        skyvis = NP.zeros( (self.baselines.shape[0], self.channels.size), dtype=datatype)
        if len(m2) != 0:
            skypos_altaz_roi = skypos_altaz[m2,:]
            coords_str = 'altaz'

            skymodel_subset = skymodel.subset(indices=m2)
            fluxes = skymodel_subset.generate_spectrum()

            if pb is None:
                pb = PB.primary_beam_generator(skypos_altaz_roi, self.channels/1.0e9, skyunits='altaz', telescope=self.telescope, pointing_info=pb_info, pointing_center=pc_altaz, freq_scale='GHz')

            pbfluxes = pb * fluxes
            geometric_delays = DLY.geometric_delay(baselines_in_local_frame, skypos_altaz_roi, altaz=(coords_str=='altaz'), hadec=(coords_str=='hadec'), latitude=self.latitude)

            vis_wts = None
            if skymodel_subset.src_shape is not None:
                eps = 1.0e-13
                f0 = self.channels[int(0.5*self.channels.size)]
                wl0 = FCNST.c / f0
                wl = FCNST.c / self.channels
                skypos_dircos_roi = GEOM.altaz2dircos(skypos_altaz_roi, units='degrees')
                # projected_spatial_frequencies = NP.sqrt(self.baseline_lengths.reshape(1,-1)**2 - (FCNST.c * geometric_delays)**2) / wl0
                projected_spatial_frequencies = NP.sqrt(self.baseline_lengths.reshape(1,-1,1)**2 - (FCNST.c * geometric_delays[:,:,NP.newaxis])**2) / wl.reshape(1,1,-1)

                src_FWHM = NP.sqrt(skymodel_subset.src_shape[:,0] * skymodel_subset.src_shape[:,1])
                src_FWHM_dircos = 2.0 * NP.sin(0.5*NP.radians(src_FWHM)).reshape(-1,1) # assuming the projected baseline is perpendicular to source direction
                # src_sigma_spatial_frequencies = 2.0 * NP.sqrt(2.0 * NP.log(2.0)) / (2 * NP.pi * src_FWHM_dircos)  # estimate 1
                src_sigma_spatial_frequencies = 1.0 / NP.sqrt(2.0*NP.log(2.0)) / src_FWHM_dircos  # estimate 2 created by constraint that at lambda/D_proj, visibility weights are half

	        # # Tried deriving below an alternate expression but previous expression for src_FWHM_dircos seems better
                # dtheta_radial = NP.radians(src_FWHM).reshape(-1,1)
                # dtheta_circum = NP.radians(src_FWHM).reshape(-1,1)
                # src_FWHM_dircos = NP.sqrt(skypos_dircos_roi[:,2].reshape(-1,1)**2 * dtheta_radial**2 + dtheta_circum**2) / NP.sqrt(2.0) # from 2D error propagation (another approximation to commented expression above for the same quantity). Add in quadrature and divide by sqrt(2) to get radius of error circle
                # arbitrary_factor_for_src_width = NP.sqrt(2.0) # An arbitrary factor that can be adjusted based on what the longest baseline measures for a source of certain finite width
                # src_sigma_spatial_frequencies = 2.0 * NP.sqrt(2.0 * NP.log(2.0)) / (2 * NP.pi * src_FWHM_dircos) * arbitrary_factor_for_src_width

                # extended_sources_flag = 1/NP.clip(projected_spatial_frequencies, 0.5, NP.amax(projected_spatial_frequencies)) < src_FWHM_dircos

                vis_wts = NP.ones_like(projected_spatial_frequencies)
                # vis_wts = NP.exp(-0.5 * (projected_spatial_frequencies/src_sigma_spatial_frequencies)**2)
                vis_wts = NP.exp(-0.5 * (projected_spatial_frequencies/src_sigma_spatial_frequencies[:,:,NP.newaxis])**2) # nsrc x nbl x nchan

            if memsave:
                pbfluxes = pbfluxes.astype(NP.float32, copy=False)
                self.geometric_delays = self.geometric_delays + [geometric_delays.astype(NP.float32)]
                if vis_wts is not None:
                    vis_wts = vis_wts.astype(NP.float32, copy=False)
            else:
                self.geometric_delays = self.geometric_delays + [geometric_delays]

            # memory_available = psutil.phymem_usage().available
            memory_available = psutil.virtual_memory().available

            if gradient_mode is None:
                if memsave:
                    memory_required = len(m2) * self.channels.size * self.baselines.shape[0] * 4.0 * 2 # bytes, 4 bytes per float, factor 2 is because the phase involves complex values
                else:
                    memory_required = len(m2) * self.channels.size * self.baselines.shape[0] * 8.0 * 2 # bytes, 8 bytes per float, factor 2 is because the phase involves complex values
            else:
                if not isinstance(gradient_mode, str):
                    raise TypeError('Input gradient_mode must be a string')
                if gradient_mode.lower() not in ['baseline', 'skypos', 'frequency']:
                    raise ValueError('Invalid value specified in input gradient_mode')
                if self.gradient_mode is None:
                    self.gradient_mode = gradient_mode
                if gradient_mode.lower() == 'baseline':
                    skyvis_gradient = NP.zeros((3, self.baselines.shape[0], self.channels.size), dtype=datatype)
                    if memsave:
                        memory_required = 3 * len(m2) * self.channels.size * self.baselines.shape[0] * 4.0 * 2 # bytes, 4 bytes per float, factor 2 is because the phase involves complex values, factor 3 because of three vector components of the gradient
                    else:
                        memory_required = 3 * len(m2) * self.channels.size * self.baselines.shape[0] * 8.0 * 2 # bytes, 8 bytes per float, factor 2 is because the phase involves complex values, factor 3 because of three vector components of the gradient

            if float(memory_available) > memory_required:
                if memsave:
                    phase_matrix = NP.exp(-1j * NP.asarray(2.0 * NP.pi).astype(NP.float32) *  (self.geometric_delays[-1][:,:,NP.newaxis].astype(NP.float32) - pc_delay_offsets.astype(NP.float32).reshape(1,-1,1)) * self.channels.astype(NP.float32).reshape(1,1,-1)).astype(NP.complex64)
                    if vis_wts is not None:
                        # phase_matrix *= vis_wts[:,:,NP.newaxis]
                        phase_matrix *= vis_wts
                    skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * phase_matrix, axis=0) # SUM(nsrc x nbl x nchan, axis=0) = nbl x nchan
                    if gradient_mode is not None:
                        if gradient_mode.lower() == 'baseline':
                            skyvis_gradient = NP.sum(skypos_dircos_roi[:,:,NP.newaxis,NP.newaxis].astype(NP.float32) * pbfluxes[:,NP.newaxis,NP.newaxis,:] * phase_matrix[:,NP.newaxis,:,:], axis=0) # SUM(nsrc x 3 x nbl x nchan, axis=0) = 3 x nbl x nchan
                else:
                    phase_matrix = 2.0 * NP.pi * (self.geometric_delays[-1][:,:,NP.newaxis].astype(NP.float64) - pc_delay_offsets.astype(NP.float64).reshape(1,-1,1)) * self.channels.astype(NP.float64).reshape(1,1,-1)
                    if vis_wts is not None:
                        # skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * NP.exp(-1j*phase_matrix) * vis_wts[:,:,NP.newaxis], axis=0) # Don't apply bandpass here
                        skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * NP.exp(-1j*phase_matrix) * vis_wts, axis=0) # SUM(nsrc x nbl x nchan, axis=0) = nbl x nchan
                        if gradient_mode is not None:
                            if gradient_mode.lower() == 'baseline':
                                skyvis_gradient = NP.sum(skypos_dircos_roi[:,:,NP.newaxis,NP.newaxis].astype(NP.float64) * pbfluxes[:,NP.newaxis,NP.newaxis,:] * NP.exp(-1j*phase_matrix[:,NP.newaxis,:,:]) * vis_wts[:,NP.newaxis,:,:], axis=0) # SUM(nsrc x 3 x nbl x nchan, axis=0) = 3 x nbl x nchan
                    else:
                        skyvis = NP.sum(pbfluxes[:,NP.newaxis,:] * NP.exp(-1j*phase_matrix), axis=0) # SUM(nsrc x nbl x nchan, axis=0) = nbl x nchan
                        if gradient_mode is not None:
                            if gradient_mode.lower() == 'baseline':
                                skyvis_gradient = NP.sum(skypos_dircos_roi[:,:,NP.newaxis,NP.newaxis].astype(NP.float64) * pbfluxes[:,NP.newaxis,NP.newaxis,:] * NP.exp(-1j*phase_matrix[:,NP.newaxis,:,:]), axis=0) # SUM(nsrc x 3 x nbl x nchan, axis=0) = 3 x nbl x nchan
            else:
                print '\t\tDetecting memory shortage. Serializing over sky direction.'
                downsize_factor = NP.ceil(memory_required/float(memory_available))
                n_src_stepsize = int(len(m2)/downsize_factor)
                src_indices = range(0,len(m2),n_src_stepsize)
                if memsave:
                    print '\t\tEnforcing single precision computations.'
                    for i in xrange(len(src_indices)):
                        phase_matrix = NP.exp(-1j * NP.asarray(2.0 * NP.pi).astype(NP.float32) * (self.geometric_delays[-1][src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis].astype(NP.float32) - pc_delay_offsets.astype(NP.float32).reshape(1,-1,1)) * self.channels.astype(NP.float32).reshape(1,1,-1)).astype(NP.complex64, copy=False)
                        if vis_wts is not None:
                            phase_matrix *= vis_wts[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,:].astype(NP.float32)
                            # phase_matrix *= vis_wts[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis].astype(NP.float32)

                        phase_matrix *= pbfluxes[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),NP.newaxis,:].astype(NP.float32)
                        skyvis += NP.sum(phase_matrix, axis=0)
                        if gradient_mode is not None:
                            if gradient_mode.lower() == 'baseline':
                                skyvis_gradient += NP.sum(skypos_dircos_roi[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis,NP.newaxis].astype(NP.float32) * phase_matrix[:,NP.newaxis,:,:], axis=0)
                else:
                    for i in xrange(len(src_indices)):
                        phase_matrix = NP.exp(-1j * NP.asarray(2.0 * NP.pi).astype(NP.float64) * (self.geometric_delays[-1][src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis].astype(NP.float64) - pc_delay_offsets.astype(NP.float64).reshape(1,-1,1)) * self.channels.astype(NP.float64).reshape(1,1,-1)).astype(NP.complex128, copy=False)
                        if vis_wts is not None:
                            phase_matrix *= vis_wts[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,:].astype(NP.float64)

                        phase_matrix *= pbfluxes[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),NP.newaxis,:].astype(NP.float64)
                        skyvis += NP.sum(phase_matrix, axis=0)
                        if gradient_mode is not None:
                            if gradient_mode.lower() == 'baseline':
                                skyvis_gradient += NP.sum(skypos_dircos_roi[src_indices[i]:min(src_indices[i]+n_src_stepsize,len(m2)),:,NP.newaxis,NP.newaxis].astype(NP.float64) * phase_matrix[:,NP.newaxis,:,:], axis=0)
            self.obs_catalog_indices = self.obs_catalog_indices + [m2]
        else:
            print 'No sources found in the catalog within matching radius. Simply populating the observed visibilities and/or gradients with noise.'
            if gradient_mode is not None:
                if gradient_mode.lower() == 'baseline':
                    skyvis_gradient = NP.zeros( (3, self.baselines.shape[0], self.channels.size), dtype=datatype)

        if self.timestamp == []:
            self.skyvis_freq = skyvis[:,:,NP.newaxis]
            if gradient_mode is not None:
                if gradient_mode.lower() == 'baseline':
                    self.gradient[gradient_mode] = skyvis_gradient[:,:,:,NP.newaxis]
        else:
            self.skyvis_freq = NP.dstack((self.skyvis_freq, skyvis[:,:,NP.newaxis]))
            if gradient_mode is not None:
                if gradient_mode.lower() == 'baseline':
                    self.gradient[gradient_mode] = NP.concatenate((self.gradient[gradient_mode], skyvis_gradient[:,:,:,NP.newaxis]), axis=3)

        self.timestamp = self.timestamp + [timestamp]
        self.t_acc = self.t_acc + [t_acc]
        self.t_obs += t_acc
        self.n_acc += 1
        self.lst = self.lst + [lst]

    ############################################################################

    def observing_run(self, pointing_init, skymodel, t_acc, duration, channels,
                      bpass, Tsys, lst_init, roi_radius=None, roi_center=None,
                      mode='track', pointing_coords=None, freq_scale=None,
                      brightness_units=None, verbose=True, memsave=False):

        """
        -------------------------------------------------------------------------
        Simulate an extended observing run in 'track' or 'drift' mode, by an
        instance of the InterferometerArray class, of the sky when a sky catalog
        is provided. The simulation generates visibilities observed by the
        interferometer array for the specified parameters. Uses member function
        observe() and builds the observation from snapshots. The timestamp for
        each snapshot is the current time at which the snapshot is generated.

        Inputs:

        pointing_init [2-element list or numpy array] The inital pointing
                      of the telescope at the start of the observing run.
                      This is where the telescopes will be initially phased up to
                      as reference. Coordinate system for the pointing_center is
                      specified by the input pointing_coords

        skymodel      [instance of class SkyModel] It consists of source flux
                      densities, their positions, and spectral indices. Read
                      class SkyModel docstring for more information.

        t_acc         [scalar] Accumulation time (sec) corresponding to timestamp

        brightness_units
                      [string] Units of flux density in the catalog and for the
                      generated visibilities. Accepted values are 'Jy' (Jansky)
                      and 'K' (Kelvin for temperature). If None set, it defaults
                      to 'Jy'

        duration      [scalar] Duration of observation in seconds

        channels      [list or numpy vector] frequency channels in units as
                      specified in freq_scale

        bpass         [list, list of lists or numpy array] Bandpass weights in
                      the form of M x N array or list of N-element lists. N must
                      equal the number of channels. If M=1, the same bandpass
                      will be used in all the snapshots for the entire
                      observation, otherwise M must equal the number of
                      snapshots which is int(duration/t_acc)

        Tsys          [scalar, list or numpy array] System temperature (in K). If
                      a scalar is provided, the same Tsys will be used in all the
                      snapshots for the duration of the observation. If a list or
                      numpy array is provided, the number of elements must equal
                      the number of snapshots which is int(duration/t_int)

        lst_init      [scalar] Initial LST (in degrees) at the beginning of the
                      observing run corresponding to pointing_init

        Keyword Inputs:

        roi_radius    [scalar] Radius of the region of interest (degrees) inside
                      which sources are to be observed. Default = 90 degrees,
                      which is the entire horizon.

        roi_center    [string] Center of the region of interest around which
                      roi_radius is used. Accepted values are 'pointing_center'
                      and 'zenith'. If set to None, it defaults to 'zenith'.

        freq_scale    [string] Units of frequencies specified in channels.
                      Accepted values are 'Hz', 'hz', 'khz', 'kHz', 'mhz',
                      'MHz', 'GHz' and 'ghz'. If None provided, defaults to 'Hz'

        mode          [string] Mode of observation. Accepted values are 'track'
                      and 'drift'. If using 'track', pointing center is fixed to
                      a specific point on the sky coordinate frame. If using
                      'drift', pointing center is fixed to a specific point on
                      the antenna's reference frame.

        pointing_coords
                      [string] Coordinate system for pointing_init. Accepted
                      values are 'radec', 'hadec' and 'altaz'. If None provided,
                      default is set based on observing mode. If mode='track',
                      pointing_coords defaults to 'radec', and if mode='drift',
                      it defaults to 'hadec'

        verbose       [boolean] If set to True, prints progress and diagnostic
                      messages. Default = True
        ------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing an observing run...\n'
            print '\tVerifying input arguments to observing_run()...'

        try:
            pointing_init, skymodel, t_acc, duration, bpass, Tsys, lst_init
        except NameError:
            raise NameError('One or more of pointing_init, skymodel, t_acc, duration, bpass, Tsys, lst_init not specified.')

        if isinstance(pointing_init, list):
            pointing_init = NP.asarray(pointing_init)
        elif not isinstance(pointing_init, NP.ndarray):
            raise TypeError('pointing_init must be a list or numpy array.')

        if pointing_init.size != 2:
            raise ValueError('pointing_init must be a 2-element vector.')
        pointing_init = pointing_init.ravel()

        if not isinstance(skymodel, SM.SkyModel):
            raise TypeError('skymodel must be an instance of class SkyModel.')

        if not isinstance(t_acc, (int, float)):
            raise TypeError('t_acc must be a scalar integer or float.')

        if t_acc <= 0.0:
            raise ValueError('t_acc must be positive.')

        if not isinstance(duration, (int, float)):
            raise TypeError('duration must be a scalar integer or float.')

        if duration <= t_acc:
            if verbose:
                print '\t\tDuration specified to be shorter than t_acc. Will set it equal to t_acc'
            duration = t_acc

        n_acc = int(duration / t_acc)
        if verbose:
            print '\t\tObserving run will have {0} accumulations.'.format(n_acc)

        if isinstance(channels, list):
            channels = NP.asarray(channels)
        elif not isinstance(channels, NP.ndarray):
            raise TypeError('channels must be a list or numpy array')

        if (freq_scale is None) or (freq_scale == 'Hz') or (freq_scale == 'hz'):
            channels = NP.asarray(channels)
        elif freq_scale == 'GHz' or freq_scale == 'ghz':
            channels = channels * 1.0e9
        elif freq_scale == 'MHz' or freq_scale == 'mhz':
            channels = channels * 1.0e6
        elif freq_scale == 'kHz' or freq_scale == 'khz':
            channels = channels * 1.0e3
        else:
            raise ValueError('Frequency units must be "GHz", "MHz", "kHz" or "Hz". If not set, it defaults to "Hz"')

        if isinstance(bpass, (list, tuple, NP.ndarray)):
            bpass = NP.asarray(bpass)
        else:
            raise TypeError('bpass must be a list, tuple or numpy array')

        if bpass.size == self.channels.size:
            bpass = NP.expand_dims(NP.repeat(bpass.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
            if verbose:
                print '\t\tSame bandpass will be applied to all baselines and all accumulations in the observing run.'
        elif bpass.size == self.baselines.shape[0] * self.channels.size:
            bpass = NP.expand_dims(bpass.reshape(-1,self.channels.size), axis=2)
            if verbose:
                print '\t\tSame bandpass will be applied to all accumulations in the observing run.'
        elif bpass.size == self.baselines.shape[0] * self.channels.size * n_acc:
            bpass = bpass.reshape(-1,self.channels.size,n_acc)
        else:
            raise ValueError('Dimensions of bpass incompatible with the number of frequency channels, baselines and number of accumulations.')

        if isinstance(Tsys, (int, float, list, tuple, NP.ndarray)):
            Tsys = NP.asarray(Tsys).reshape(-1)
        else:
            raise TypeError('Tsys must be a scalar, list, tuple or numpy array')

        if Tsys.size == 1:
            if verbose:
                print '\t\tTsys = {0:.1f} K will be assumed for all frequencies, baselines, and accumulations.'.format(Tsys[0])
            Tsys = Tsys + NP.zeros((self.baselines.shape[0], self.channels.size, 1))
        elif Tsys.size == self.channels.size:
            Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2)
            if verbose:
                print '\t\tSame Tsys will be assumed for all baselines and all accumulations in the observing run.'
        elif Tsys.size == self.baselines.shape[0]:
            Tsys = NP.expand_dims(NP.repeat(Tsys.reshape(-1,1), self.channels.size, axis=1), axis=2)
            if verbose:
                print '\t\tSame Tsys will be assumed for all frequency channels and all accumulations in the observing run.'
        elif Tsys.size == self.baselines.shape[0] * self.channels.size:
            Tsys = NP.expand_dims(Tsys.reshape(-1,self.channels.size), axis=2)
            if verbose:
                print '\t\tSame Tsys will be assumed for all accumulations in the observing run.'
        elif Tsys.size == self.baselines.shape[0] * self.channels.size * n_acc:
            Tsys = Tsys.reshape(-1,self.channels.size,n_acc)
        else:
            raise ValueError('Dimensions of Tsys incompatible with the number of frequency channels, baselines and number of accumulations.')

        if not isinstance(lst_init, (int, float)):
            raise TypeError('Starting LST should be a scalar')

        if verbose:
            print '\tVerified input arguments.'
            print '\tProceeding to schedule the observing run...'

        lst = (lst_init + (t_acc/3.6e3) * NP.arange(n_acc)) * 15.0 # in degrees
        if verbose:
            print '\tCreated LST range for observing run.'

        if mode == 'track':
            if pointing_coords == 'hadec':
                pointing = NP.asarray([lst_init - pointing_init[0], pointing_init[1]])
            elif (pointing_coords == 'radec') or (pointing_coords is None):
                pointing = pointing_init
            elif pointing_coords == 'altaz':
                hadec = GEOM.altaz2hadec(pointing_init, self.latitude, units='degrees')
                pointing = NP.asarray([lst_init - hadec[0], hadec[1]])
            else:
                raise ValueError('pointing_coords can only be set to "hadec", "radec" or "altaz".')
            self.pointing_coords = 'radec'
            self.phase_center_coords = 'radec'
        elif mode == 'drift':
            if pointing_coords == 'radec':
                pointing = NP.asarray([lst_init - pointing_init[0], pointing_init[1]])
            elif (pointing_coords == 'hadec') or (pointing_coords is None):
                pointing = pointing_init
            elif pointing_coords == 'altaz':
                pointing = GEOM.altaz2hadec(pointing_init, self.latitude, units='degrees')
            else:
                raise ValueError('pointing_coords can only be set to "hadec", "radec" or "altaz".')
            self.pointing_coords = 'hadec'
            self.phase_center_coords = 'hadec'

        if verbose:
            print '\tPreparing to observe in {0} mode'.format(mode)

        if verbose:
            milestones = range(max(1,int(n_acc/10)), int(n_acc), max(1,int(n_acc/10)))
            progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=n_acc).start()
        for i in range(n_acc):
            # if (verbose) and (i in milestones):
            #     print '\t\tObserving run {0:.1f} % complete...'.format(100.0*i/n_acc)
            timestamp = str(DT.datetime.now())
            self.observe(timestamp, Tsys[:,:,i%Tsys.shape[2]],
                         bpass[:,:,i%bpass.shape[2]], pointing, skymodel,
                         t_acc, brightness_units=brightness_units,
                         roi_radius=roi_radius, roi_center=roi_center,
                         lst=lst[i], memsave=memsave)
            if verbose:
                progress.update(i+1)

        if verbose:
            progress.finish()

        # if verbose:
        #     print '\t\tObserving run 100 % complete.'

        self.t_obs = duration
        self.n_acc = n_acc
        if verbose:
            print 'Observing run completed successfully.'

    #############################################################################

    def generate_noise(self):

        """
        -------------------------------------------------------------------------
        Generates thermal noise from attributes that describe system parameters
        which can be added to sky visibilities
        -------------------------------------------------------------------------
        """

        eff_Q = self.eff_Q
        A_eff = self.A_eff
        t_acc = NP.asarray(self.t_acc)

        if len(eff_Q.shape) == 2:
            eff_Q = eff_Q[:,:,NP.newaxis]
        if len(A_eff.shape) == 2:
            A_eff = A_eff[:,:,NP.newaxis]
        t_acc = t_acc[NP.newaxis,NP.newaxis,:]

        if (self.flux_unit == 'JY') or (self.flux_unit == 'jy') or (self.flux_unit == 'Jy'):
            self.vis_rms_freq = 2.0 * FCNST.k / NP.sqrt(2.0*t_acc*self.freq_resolution) * (self.Tsys/A_eff/eff_Q) / CNST.Jy
        elif (self.flux_unit == 'K') or (self.flux_unit == 'k'):
            self.vis_rms_freq = 1 / NP.sqrt(2.0*t_acc*self.freq_resolution) * self.Tsys/eff_Q
        else:
            raise ValueError('Flux density units can only be in Jy or K.')

        self.vis_noise_freq = self.vis_rms_freq / NP.sqrt(2.0) * (NP.random.randn(self.baselines.shape[0], self.channels.size, len(self.timestamp)) + 1j * NP.random.randn(self.baselines.shape[0], self.channels.size, len(self.timestamp))) # sqrt(2.0) is to split equal uncertainty into real and imaginary parts

    #############################################################################

    def add_noise(self):

        """
        -------------------------------------------------------------------------
        Adds the thermal noise generated in member function generate_noise() to
        the sky visibilities after extracting and applying complex instrument
        gains
        -------------------------------------------------------------------------
        """

        gains = 1.0
        if self.gaininfo is not None:
            try:
                gains = self.gaininfo.spline_gains(self.labels, freqs=self.channels, times=NP.asarray(self.timestamp))
            except IndexError:
                try:
                    gains = self.gaininfo.spline_gains(self.labels, freqs=self.channels, times=NP.asarray(self.timestamp)-self.timestamp[0])
                except IndexError:
                    try:
                        gains = self.gaininfo.nearest_gains(self.labels, freqs=self.channels, times=NP.asarray(self.timestamp))
                    except:
                        warnings.warn('Interpolation and nearest neighbour logic failed. Proceeding with default unity gains')
        else:
            warnings.warn('Gain table absent. Proceeding with default unity gains')

        self.vis_freq = gains * self.skyvis_freq + self.vis_noise_freq

    #############################################################################

    def apply_gradients(self, gradient_mode=None, perturbations=None):

        """
        -------------------------------------------------------------------------
        Apply the perturbations in combination with the gradients to determine
        perturbed visibilities

        Inputs:

        perturbations   [dictionary] Contains perturbations on one of the 
                        following quantities (specified as keys):
                        'baseline'  [numpy array] nseed x 3 x nbl baseline 
                                    perturbations (in same units as attribute 
                                    baselines). The first dimension denotes the 
                                    number of realizations, the second denotes
                                    the x-, y- and z-axes and the third 
                                    denotes the number of baselines. It can also
                                    handle arrays of shapes (n1, n2, ..., 3, nbl)
        
        gradient_mode   [string] Specifies the quantity on which perturbations
                        are provided and perturbed visibilities to be computed.
                        This string must be one of the keys in the input 
                        dictionary perturbations and must be found in the
                        attribute gradient_mode and gradient. Currently 
                        accepted values are 'baseline'

        Output:

        Perturbed visibilities as a n1 x n2 x ... x nbl x nchan x ntimes 
        complex array
        -------------------------------------------------------------------------
        """

        if gradient_mode is None:
            gradient_mode = self.gradient_mode

        if perturbations is None:
            perturbations = {gradient_mode: NP.zeros((1,1,1))}

        if self.gradient_mode is None:
            raise AttributeError('No gradient attribute found')
        else:
            if not self.gradient:
                raise AttributeError('No gradient attribute found')

        if not isinstance(perturbations, dict):
            raise TypeError('Input perturbations must be a dictionary')

        if not isinstance(gradient_mode, str):
            raise TypeError('Input gradient_mode must be a string')

        if gradient_mode not in ['baseline']:
            raise KeyError('Specified gradient mode {0} not currently supported'.format(gradient_mode))

        if gradient_mode not in perturbations:
            raise KeyError('{0} key not found in input perturbations'.format(gradient_key))

        if gradient_mode != self.gradient_mode:
            raise ValueError('Specified gradient mode {0} not found in attribute'.format(gradient_mode))

        if not isinstance(perturbations[gradient_mode], NP.ndarray):
            raise TypeError('Perturbations must be specified as a numpy array')

        if perturbations[gradient_mode].ndim == 2:
            perturbations[gradient_mode] = perturbations[gradient_mode][NP.newaxis,...]
        if perturbations[gradient_mode].ndim < 2:
            raise ValueError('Perturbations must be two--dimensions or higher')

        inpshape = perturbations[gradient_mode].shape
        if perturbations[gradient_mode].ndim > 3:
            perturbations[gradient_mode] = perturbations[gradient_mode].reshape(-1,inpshape[-2],inpshape[-1])

        if perturbations[gradient_mode].shape[2] != self.gradient[self.gradient_mode].shape[1]:
            raise ValueError('Number of {0} perturbations not equal to that in the gradient attribute'.format(gradient_mode))

        if perturbations[gradient_mode].shape[1] == 1:
            warnings.warn('Only {0}-dimensional coordinates specified. Proceeding with zero perturbations in other coordinate axes.'.format(perturbations[gradient_mode].shape[1]))
            perturbations[gradient_mode] = NP.hstack((perturbations[gradient_mode], NP.zeros((perturbations[gradient_mode].shape[0],2,perturbations[gradient_mode].shape[2])))) # nseed x 3 x nbl
        elif perturbations[gradient_mode].shape[1] == 2:
            warnings.warn('Only {0}-dimensional coordinates specified. Proceeding with zero perturbations in other coordinate axes.'.format(perturbations[gradient_mode].shape[1])) 
            perturbations[gradient_mode] = NP.hstack((perturbations[gradient_mode], NP.zeros((perturbations[gradient_mode].shape[0],1,perturbations[gradient_mode].shape[2])))) # nseed x 3 x nbl
        elif perturbations[gradient_mode].shape[1] > 3:
            warnings.warn('{0}-dimensional coordinates specified. Proceeding with only the first three dimensions of coordinate axes.'.format(3))
            perturbations[gradient_mode] = perturbations[gradient_mode][:,:3,:] # nseed x 3 x nbl

        wl = FCNST.c / self.channels
        if gradient_mode == 'baseline':
            delta_skyvis_freq = -1j * 2.0 * NP.pi / wl.reshape(1,1,-1,1) * NP.sum(perturbations[gradient_mode][...,NP.newaxis,NP.newaxis] * self.gradient[gradient_mode][NP.newaxis,...], axis=1) # nseed x nbl x nchan x ntimes
            
        outshape = list(inpshape[:-2])
        outshape += [self.labels.size, self.channels.size, self.lst.size]
        outshape = tuple(outshape)
        delta_skyvis_freq = delta_skyvis_freq.reshape(outshape)
        return delta_skyvis_freq

    #############################################################################

    def duplicate_measurements(self, blgroups=None):

        """
        -------------------------------------------------------------------------
        Duplicate visibilities based on redundant baselines specified. This saves
        time when compared to simulating visibilities over redundant baselines.
        Thus, it is more efficient to simulate unique baselines and duplicate
        measurements for redundant baselines

        Inputs:

        blgroups    [dictionary] Dictionary of baseline groups where the keys are
                    tuples containing baseline labels. Under each key is a numpy
                    recarray of baseline labels that are redundant and fall under 
                    the baseline label key. Any number of sets of redundant 
                    measurements can be duplicated in this depending on the 
                    baseline label keys and recarrays specified here. It results
                    in updating attributes where a new number of baselines are 
                    formed from original baselines and new redundant baselines.
                    If set to None (default), attribute blgroups will be used to
                    create redundant sets
        -------------------------------------------------------------------------
        """

        if blgroups is None:
            blgroups = self.blgroups
        if not isinstance(blgroups, dict):
            raise TypeError('Input blgroups must be a dictionary')

        if self.bl_reversemap is None:
            nbl = NP.sum(NP.asarray([len(blgroups[blkey]) for blkey in blgroups]))
        else:
            nbl = len(self.bl_reversemap)

        if self.labels.size < nbl:
            label_keys = NP.asarray(blgroups.keys(), dtype=self.labels.dtype)
            for label_key in label_keys:
                if label_key not in self.labels:
                    if NP.asarray([tuple(reversed(label_key))], dtype=self.labels.dtype)[0] not in self.labels:
                        raise KeyError('Input label {0} not found in attribute labels'.format(label_key))
                    else:
                        label_key = NP.asarray([tuple(reversed(label_key))], dtype=self.labels.dtype)[0]
                if label_key not in blgroups[tuple(label_key)]:
                    blgroups[tuple(label_key)] += [label_key]
    
            uniq_inplabels = []
            num_list = []
            for label in self.labels:
                if label in label_keys:
                    num_list += [blgroups[tuple(label)].size]
                    for lbl in blgroups[tuple(label)]:
                        if tuple(lbl) not in uniq_inplabels:
                            uniq_inplabels += [tuple(lbl)]
                        else:
                            raise ValueError('Label {0} repeated in more than one baseline group'.format(lbl))
                else:
                    num_list += [1]
                    uniq_inplabels += [tuple(label)]
            if len(num_list) != len(self.labels):
                raise ValueError('Fatal error in counting and matching labels in input blgroups')
            if self.skyvis_freq is not None:
                self.skyvis_freq = NP.repeat(self.skyvis_freq, num_list, axis=0)
            if self.gradient_mode is not None:
                self.gradient[self.gradient_mode] = NP.repeat(self.gradient[self.gradient_mode], num_list, axis=1)
            self.labels = NP.asarray(uniq_inplabels, dtype=self.labels.dtype)
            self.baselines = NP.repeat(self.baselines, num_list, axis=0)
            self.baseline_lengths = NP.repeat(self.baseline_lengths, num_list)
            if self.Tsys.shape[0] > 1:
                self.Tsys = NP.repeat(self.Tsys, num_list, axis=0)
            if self.eff_Q.shape[0] > 1:
                self.eff_Q = NP.repeat(self.eff_Q, num_list, axis=0)
            if self.A_eff.shape[0] > 1:
                self.A_eff = NP.repeat(self.A_eff, num_list, axis=0)
            if self.bp.shape[0] > 1:
                self.bp = NP.repeat(self.bp, num_list, axis=0)
            if self.bp_wts.shape[0] > 1:
                self.bp_wts = NP.repeat(self.bp_wts, num_list, axis=0)
            self.generate_noise()
            self.add_noise()

    #############################################################################

    def getThreePointCombinations(self):

        """
        -------------------------------------------------------------------------
        Return all unique 3-point combinations of baselines

        Output:

        Tuple containing two lists. The first list is a list of triplet tuples of
        antenna ids in the form [(a1,a2,a3), (a1,a4,a6), ...], the second list 
        is a list of triplet tuples of baselines as encoded as strings
        -------------------------------------------------------------------------
        """

        bl = self.baselines + 0.0 # to avoid any weird negative sign before 0.0
        blstr = ['{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(lo) for lo in bl]
        bltriplets = []
        anttriplets = []
        for aind1,aid1 in enumerate(self.layout['ids']):
            for aind2,aid2 in enumerate(self.layout['ids']):
                bl12 = self.layout['positions'][aind2] - self.layout['positions'][aind1]
                bl12 += 0.0 # to avoid any weird negative sign before 0.0
                bl12[NP.abs(bl12) < 1e-10] = 0.0
                bl12_len = NP.sqrt(NP.sum(bl12**2))
                if bl12_len > 0.0:
                    bl12str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl12)
                    if bl12str not in blstr:
                        bl12 *= -1
                        bl12 += 0.0 # to avoid any weird negative sign before 0.0
                        bl12str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl12)
                        if bl12str not in blstr:
                            raise IndexError('A baseline not found in reference baselines')
                    for aind3,aid3 in enumerate(self.layout['ids']):
                        bl23 = self.layout['positions'][aind3] - self.layout['positions'][aind2]
                        bl31 = self.layout['positions'][aind1] - self.layout['positions'][aind3]
                        bl23 += 0.0 # to avoid any weird negative sign before 0.0
                        bl31 += 0.0 # to avoid any weird negative sign before 0.0
                        bl23[NP.abs(bl23) < 1e-10] = 0.0
                        bl31[NP.abs(bl31) < 1e-10] = 0.0
                        bl23_len = NP.sqrt(NP.sum(bl23**2))
                        bl31_len = NP.sqrt(NP.sum(bl31**2))
                        if (bl23_len > 0.0) and (bl31_len > 0.0):
                            bl23str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl23)
                            if bl23str not in blstr:
                                bl23 *= -1
                                bl23 += 0.0 # to avoid any weird negative sign before 0.0
                                bl23str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl23)
                                if bl23str not in blstr:
                                    raise IndexError('A baseline not found in reference baselines')
        
                            bl31str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl31)
                            if bl31str not in blstr:
                                bl31 *= -1
                                bl31 += 0.0 # to avoid any weird negative sign before 0.0
                                bl31str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl31)
                                if bl31str not in blstr:
                                    raise IndexError('A baseline not found in reference baselines')
                            list123_str = [bl12str, bl23str, bl31str]
                            if len(list123_str) == 3:
                                if len(bltriplets) == 0:
                                    bltriplets += [list123_str]
                                    anttriplets += [(aid1, aid2, aid3)]
                                else:
                                    found = False
                                    ind = 0
                                    while (not found) and (ind < len(bltriplets)):
                                        bltriplet = bltriplets[ind]
                                        if NP.setdiff1d(list123_str, bltriplet).size == 0:
                                            found = True
                                        else:
                                            ind += 1
                                    if not found:
                                        bltriplets += [list123_str]
                                        anttriplets += [(aid1, aid2, aid3)]

        return (anttriplets, bltriplets)             

    #############################################################################

    def getClosurePhase(self, antenna_triplets=None):

        """
        -------------------------------------------------------------------------
        Get closure phases of visibilities from triplets of antennas. 

        Inputs:

        antenna_triplets
                    [list of tuples] List of antenna ID triplets where each 
                    triplet is given as a tuple. If set to None (default), all
                    the unique triplets based on the antenna layout attribute
                    in class InterferometerArray

        Output:

        Dictionary containing closure phase information under the following keys
        and values:
        'closure_phase_skyvis'  [numpy array] Closure phases (in radians) for
                                the given antenna triplets from the noiseless 
                                visibilities. It is of shape
                                ntriplets x nchan x ntimes
        'closure_phase_vis'     [numpy array] Closure phases (in radians) for
                                the given antenna triplets for noisy 
                                visibilities. It is of shape
                                ntriplets x nchan x ntimes
        'antenna_triplets'      [list of tuples] List of three-element tuples of 
                                antenna IDs for which the closure phases are
                                calculated.
        -------------------------------------------------------------------------
        """

        if antenna_triplets is None:
            antenna_triplets, bltriplets = self.getThreePointCombinations()

        if not isinstance(antenna_triplets, list):
            raise TypeError('Input antenna triplets must be a list of triplet tuples')

        phase_skyvis123 = []
        phase_vis123 = []
        for anttriplet in antenna_triplets:
            a1, a2, a3 = anttriplet
            a1 = str(a1)
            a2 = str(a2)
            a3 = str(a3)
            bl12_id = (a2, a1)
            conj12 = False
            if bl12_id in self.bl_reversemap:
                bl12_id_ref = self.bl_reversemap[bl12_id]
            elif tuple(reversed(bl12_id)) in self.bl_reversemap:
                bl12_id_ref = self.bl_reversemap[tuple(reversed(bl12_id))]
                conj12 = True
            else:
                raise ValueError('Baseline ({0[0]:0d}, {0[1]:0d}) not found in simulated baselines'.format(bl12_id))
            ind12 = NP.where(self.labels == bl12_id_ref)[0][0]
            if not conj12:
                skyvis12 = self.skyvis_freq[ind12,:,:]
                vis12 = self.vis_freq[ind12,:,:]
            else:
                skyvis12 = self.skyvis_freq[ind12,:,:].conj()
                vis12 = self.vis_freq[ind12,:,:].conj()

            bl23_id = (a3, a2)
            conj23 = False
            if bl23_id in self.bl_reversemap:
                bl23_id_ref = self.bl_reversemap[bl23_id]
            elif tuple(reversed(bl23_id)) in self.bl_reversemap:
                bl23_id_ref = self.bl_reversemap[tuple(reversed(bl23_id))]
                conj23 = True
            else:
                raise ValueError('Baseline ({0[0]:0d}, {0[1]:0d}) not found in simulated baselines'.format(bl23_id))
            ind23 = NP.where(self.labels == bl23_id_ref)[0][0]
            if not conj23:
                skyvis23 = self.skyvis_freq[ind23,:,:]
                vis23 = self.vis_freq[ind23,:,:]
            else:
                skyvis23 = self.skyvis_freq[ind23,:,:].conj()
                vis23 = self.vis_freq[ind23,:,:].conj()

            bl31_id = (a1, a3)
            conj31 = False
            if bl31_id in self.bl_reversemap:
                bl31_id_ref = self.bl_reversemap[bl31_id]
            elif tuple(reversed(bl31_id)) in self.bl_reversemap:
                bl31_id_ref = self.bl_reversemap[tuple(reversed(bl31_id))]
                conj31 = True
            else:
                raise ValueError('Baseline ({0[0]:0d}, {0[1]:0d}) not found in simulated baselines'.format(bl31_id))
            ind31 = NP.where(self.labels == bl31_id_ref)[0][0]
            if not conj31:
                skyvis31 = self.skyvis_freq[ind31,:,:]
                vis31 = self.vis_freq[ind31,:,:]
            else:
                skyvis31 = self.skyvis_freq[ind31,:,:].conj()
                vis31 = self.vis_freq[ind31,:,:].conj()

            phase_skyvis123 += [NP.angle(skyvis12*skyvis23*skyvis31)]
            phase_vis123 += [NP.angle(vis12*vis23*vis31)]

        return {'closure_phase_skyvis': NP.asarray(phase_skyvis123), 'closure_phase_vis': NP.asarray(phase_vis123), 'antenna_triplets': antenna_triplets}

    #############################################################################

    def rotate_visibilities(self, ref_point, do_delay_transform=False,
                            verbose=True):

        """
        -------------------------------------------------------------------------
        Centers the phase of visibilities around any given phase center.
        Project baseline vectors with respect to a reference point on the sky.
        Essentially a wrapper to member functions phase_centering() and
        project_baselines()

        Input(s):

        ref_point   [dictionary] Contains information about the reference
                    position to which projected baselines and rotated
                    visibilities are to be computed. No defaults. It must be
                    contain the following keys with the following values:
                    'coords'    [string] Refers to the coordinate system in
                                which value in key 'location' is specified in.
                                Accepted values are 'radec', 'hadec', 'altaz'
                                and 'dircos'
                    'location'  [numpy array] Must be a Mx2 (if value in key
                                'coords' is set to 'radec', 'hadec', 'altaz' or
                                'dircos') or Mx3 (if value in key 'coords' is
                                set to 'dircos'). M can be 1 or equal to number
                                of timestamps. If M=1, the same reference point
                                in the same coordinate system will be repeated
                                for all tiemstamps. If value under key 'coords'
                                is set to 'radec', 'hadec' or 'altaz', the
                                value under this key 'location' must be in
                                units of degrees.

        do_delay_transform
                      [boolean] If set to True (default), also recompute the
                      delay transform after the visibilities are rotated to the
                      new phase center

        verbose:      [boolean] If set to True (default), prints progress and
                      diagnostic messages.
        -------------------------------------------------------------------------
        """

        try:
            ref_point
        except NameError:
            raise NameError('Input ref_point must be provided')
        if ref_point is None:
            raise ValueError('Invalid input specified in ref_point')
        elif not isinstance(ref_point, dict):
            raise TypeError('Input ref_point must be a dictionary')
        else:
            if ('location' not in ref_point) or ('coords' not in ref_point):
                raise KeyError('Both keys "location" and "coords" must be specified in input dictionary ref_point')
            self.phase_centering(ref_point, do_delay_transform=do_delay_transform, verbose=verbose)
            self.project_baselines(ref_point)

    #############################################################################

    def phase_centering(self, ref_point, do_delay_transform=False, verbose=True):

        """
        -------------------------------------------------------------------------
        Centers the phase of visibilities around any given phase center.

        Inputs:

        ref_point   [dictionary] Contains information about the reference
                    position to which projected baselines and rotated
                    visibilities are to be computed. No defaults. It must be
                    contain the following keys with the following values:
                    'coords'    [string] Refers to the coordinate system in
                                which value in key 'location' is specified in.
                                Accepted values are 'radec', 'hadec', 'altaz'
                                and 'dircos'
                    'location'  [numpy array] Must be a Mx2 (if value in key
                                'coords' is set to 'radec', 'hadec', 'altaz' or
                                'dircos') or Mx3 (if value in key 'coords' is
                                set to 'dircos'). M can be 1 or equal to number
                                of timestamps. If M=1, the same reference point
                                in the same coordinate system will be repeated
                                for all tiemstamps. If value under key 'coords'
                                is set to 'radec', 'hadec' or 'altaz', the
                                value under this key 'location' must be in
                                units of degrees.

        do_delay_transform
                      [boolean] If set to True, also recompute the delay
                      transform after the visibilities are rotated to the new
                      phase center. If set to False (default), this is skipped

        verbose:      [boolean] If set to True (default), prints progress and
                      diagnostic messages.
        -------------------------------------------------------------------------
        """

        try:
            ref_point
        except NameError:
            raise NameError('Input ref_point must be provided')
        if ref_point is None:
            raise ValueError('Invalid input specified in ref_point')
        elif not isinstance(ref_point, dict):
            raise TypeError('Input ref_point must be a dictionary')
        else:
            if ('location' not in ref_point) or ('coords' not in ref_point):
                raise KeyError('Both keys "location" and "coords" must be specified in input dictionary ref_point')

        phase_center = ref_point['location']
        phase_center_coords = ref_point['coords']

        if phase_center is None:
            raise ValueError('Valid phase center not specified in input ref_point')
        elif not isinstance(phase_center, NP.ndarray):
            raise TypeError('Phase center must be a numpy array')
        elif phase_center.shape[0] == 1:
            phase_center = NP.repeat(phase_center, len(self.lst), axis=0)
        elif phase_center.shape[0] != len(self.lst):
            raise ValueError('One phase center must be provided for every timestamp.')

        phase_center_current = self.phase_center + 0.0
        phase_center_new = phase_center + 0.0
        phase_center_coords_current = self.phase_center_coords + ''
        phase_center_coords_new = phase_center_coords + ''
        phase_center_temp = phase_center_new + 0.0
        phase_center_coords_temp = phase_center_coords_new + ''

        if phase_center_coords_new is None:
            raise NameError('Coordinates of phase center not provided.')
        elif phase_center_coords_new == 'dircos':
            if (phase_center_new.shape[1] < 2) or (phase_center_new.shape[1] > 3):
                raise ValueError('Dimensions incompatible for direction cosine positions')
            if NP.any(NP.sqrt(NP.sum(phase_center_new**2, axis=1)) > 1.0):
                raise ValueError('direction cosines found to be exceeding unit magnitude.')
            if phase_center_new.shape[1] == 2:
                n = 1.0 - NP.sqrt(NP.sum(phase_center_new**2, axis=1))
                phase_center_new = NP.hstack((phase_center_new, n.reshape(-1,1)))
            phase_center_temp = phase_center_new + 0.0
            phase_center_coords_temp = 'dircos'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.dircos2altaz(phase_center_temp, units='degrees')
                phase_center_coords_temp = 'altaz'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.altaz2hadec(phase_center_temp, self.latitude, units='degrees')
                phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'radec'
            if phase_center_coords_temp != phase_center_coords_current:
                raise ValueError('Pointing coordinates of interferometer array instance invalid.')
        elif phase_center_coords_new == 'altaz':
            phase_center_temp = phase_center_new + 0.0
            phase_center_coords_temp = 'altaz'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.altaz2hadec(phase_center_temp, self.latitude, units='degrees')
                phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'radec'
            if phase_center_coords_temp != phase_center_coords_current:
                raise ValueError('Pointing coordinates of interferometer array instance invalid.')
            phase_center_coords_temp = phase_center_coords_current + ''
            phase_center_new = GEOM.altaz2dircos(phase_center_new, units='degrees')
        elif phase_center_coords_new == 'hadec':
            phase_center_temp = phase_center_new + 0.0
            phase_center_coords_temp = 'hadec'
            if phase_center_coords_temp != phase_center_coords_current:
                if self.pointing_coords == 'radec':
                    phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                    phase_center_coords_temp = 'radec'
                else:
                    phase_center_temp = GEOM.hadec2altaz(phase_center_temp, self.latitude, units='degrees')
                    phase_center_coords_temp = 'altaz'
                    if phase_center_coords_temp != phase_center_coords_current:
                        phase_center_temp = GEOM.altaz2dircos(phase_center_temp, units='degrees')
                        phase_center_coords_temp = 'dircos'
                        if phase_center_coords_temp != phase_center_coords_current:
                            raise ValueError('Pointing coordinates of interferometer array instance invalid.')
            phase_center_new = GEOM.hadec2altaz(phase_center_new, self.latitude, units='degrees')
            phase_center_new = GEOM.altaz2dircos(phase_center_new, units='degrees')
        elif phase_center_coords_new == 'radec':
            phase_center_temp = phase_center_new + 0.0
            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp[:,0] = self.lst - phase_center_temp[:,0]
                phase_center_coords_temp = 'hadec'

            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.hadec2altaz(phase_center_temp, self.latitude, units='degrees')
                phase_center_coords_temp = 'altaz'

            if phase_center_coords_temp != phase_center_coords_current:
                phase_center_temp = GEOM.altaz2dircos(phase_center_temp, units='degrees')
                phase_center_coords_temp = 'dircos'

            if phase_center_coords_temp != phase_center_coords_current:
                raise ValueError('Pointing coordinates of interferometer array instance invalid.')

            phase_center_new[:,0] = self.lst - phase_center_new[:,0]
            phase_center_new = GEOM.hadec2altaz(phase_center_new, self.latitude, units='degrees')
            phase_center_new = GEOM.altaz2dircos(phase_center_new, units='degrees')
        else:
            raise ValueError('Invalid phase center coordinate system specified')

        phase_center_current_temp = phase_center_current + 0.0
        phase_center_coords_current_temp = phase_center_coords_current + ''
        if phase_center_coords_current_temp == 'radec':
            phase_center_current_temp[:,0] = self.lst - phase_center_current_temp[:,0]
            phase_center_coords_current_temp = 'hadec'
        if phase_center_coords_current_temp == 'hadec':
            phase_center_current_temp = GEOM.hadec2altaz(phase_center_current_temp, self.latitude, units='degrees')
            phase_center_coords_current_temp = 'altaz'
        if phase_center_coords_current_temp == 'altaz':
            phase_center_current_temp = GEOM.altaz2dircos(phase_center_current_temp, units='degrees')
            phase_center_coords_current_temp = 'dircos'

        pos_diff_dircos = phase_center_current_temp - phase_center_new
        b_dot_l = NP.dot(self.baselines, pos_diff_dircos.T)

        self.phase_center = phase_center_temp + 0.0
        self.phase_center_coords = phase_center_coords_temp + ''

        self.skyvis_freq = self.skyvis_freq * NP.exp(-1j * 2 * NP.pi * b_dot_l[:,NP.newaxis,:] * self.channels.reshape(1,-1,1) / FCNST.c)
        if self.vis_freq is not None:
            self.vis_freq = self.vis_freq * NP.exp(-1j * 2 * NP.pi * b_dot_l[:,NP.newaxis,:] * self.channels.reshape(1,-1,1) / FCNST.c)
        if self.vis_noise_freq is not None:
            self.vis_noise_freq = self.vis_noise_freq * NP.exp(-1j * 2 * NP.pi * b_dot_l[:,NP.newaxis,:] * self.channels.reshape(1,-1,1) / FCNST.c)
        if do_delay_transform:
            self.delay_transform()
            print 'Running delay_transform() with defaults inside phase_centering() after rotating visibility phases. Run delay_transform() again with appropriate inputs.'

    #############################################################################

    def project_baselines(self, ref_point):

        """
        ------------------------------------------------------------------------
        Project baseline vectors with respect to a reference point on the sky.
        Assigns the projected baselines to the attribute projected_baselines

        Input(s):

        ref_point   [dictionary] Contains information about the reference
                    position to which projected baselines are to be computed.
                    No defaults. It must be contain the following keys with the
                    following values:
                    'coords'    [string] Refers to the coordinate system in
                                which value in key 'location' is specified in.
                                Accepted values are 'radec', 'hadec', 'altaz'
                                and 'dircos'
                    'location'  [numpy array] Must be a Mx2 (if value in key
                                'coords' is set to 'radec', 'hadec', 'altaz' or
                                'dircos') or Mx3 (if value in key 'coords' is
                                set to 'dircos'). M can be 1 or equal to number
                                of timestamps. If M=1, the same reference point
                                in the same coordinate system will be repeated
                                for all tiemstamps. If value under key 'coords'
                                is set to 'radec', 'hadec' or 'altaz', the
                                value under this key 'location' must be in
                                units of degrees.
        ------------------------------------------------------------------------
        """

        try:
            ref_point
        except NameError:
            raise NameError('Input ref_point must be provided')
        if ref_point is None:
            raise ValueError('Invalid input specified in ref_point')
        elif not isinstance(ref_point, dict):
            raise TypeError('Input ref_point must be a dictionary')
        else:
            if ('location' not in ref_point) or ('coords' not in ref_point):
                raise KeyError('Both keys "location" and "coords" must be specified in input dictionary ref_point')

        phase_center = ref_point['location']
        phase_center_coords = ref_point['coords']
        if not isinstance(phase_center, NP.ndarray):
            raise TypeError('The specified reference point must be a numpy array')
        if not isinstance(phase_center_coords, str):
            raise TypeError('The specified coordinates of the reference point must be a string')
        if phase_center_coords not in ['radec', 'hadec', 'altaz', 'dircos']:
            raise ValueError('Specified coordinates of reference point invalid')
        if phase_center.ndim == 1:
            phase_center = phase_center.reshape(1,-1)
        if phase_center.ndim > 2:
            raise ValueError('Reference point has invalid dimensions')
        if (phase_center.shape[0] != self.n_acc) and (phase_center.shape[0] != 1):
            raise ValueError('Reference point has dimensions incompatible with the number of timestamps')
        if phase_center.shape[0] == 1:
            phase_center = phase_center + NP.zeros(self.n_acc).reshape(-1,1)
        if phase_center_coords == 'radec':
            if phase_center.shape[1] != 2:
                raise ValueError('Reference point has invalid dimensions')
            ha = NP.asarray(self.lst) - phase_center[:,0]
            dec = phase_center[:,1]
        elif phase_center_coords == 'hadec':
            if phase_center.shape[1] != 2:
                raise ValueError('Reference point has invalid dimensions')
            ha = phase_center[:,0]
            dec = phase_center[:,1]
        elif phase_center_coords == 'altaz':
            if phase_center.shape[1] != 2:
                raise ValueError('Reference point has invalid dimensions')
            hadec = GEOM.altaz2hadec(phase_center, self.latitude, units='degrees')
            ha = hadec[:,0]
            dec = hadec[:,1]
        else: # phase_center_coords = 'dircos'
            if (phase_center.shape[1] < 2) or (phase_center.shape[1] > 3):
                raise ValueError('Reference point has invalid dimensions')
            if NP.any(NP.sqrt(NP.sum(phase_center**2, axis=1)) > 1.0):
                raise ValueError('direction cosines found to be exceeding unit magnitude.')
            if NP.any(NP.max(NP.abs(phase_center), axis=1) > 1.0):
                raise ValueError('direction cosines found to be exceeding unit magnitude.')
            if phase_center.shape[1] == 2:
                n = 1.0 - NP.sqrt(NP.sum(phase_center**2, axis=1))
                phase_center = NP.hstack((phase_center, n.reshape(-1,1)))
            altaz = GEOM.dircos2altaz(phase_center, units='degrees')
            hadec = GEOM.altaz2hadec(phase_center, self.latitude, units='degrees')
            ha = hadec[:,0]
            dec = hadec[:,1]
        ha = NP.radians(ha).ravel()
        dec = NP.radians(dec).ravel()

        eq_baselines = GEOM.enu2xyz(self.baselines, self.latitude, units='degrees')
        rot_matrix = NP.asarray([[NP.sin(ha),               NP.cos(ha),             NP.zeros(ha.size)],
                                 [-NP.sin(dec)*NP.cos(ha), NP.sin(dec)*NP.sin(ha), NP.cos(dec)],
                                 [NP.cos(dec)*NP.cos(ha), -NP.cos(dec)*NP.sin(ha), NP.sin(dec)]])
        if rot_matrix.ndim == 2:
            rot_matrix = rot_matrix[:,:,NP.newaxis] # To ensure correct dot product is obtained in the next step
        self.projected_baselines = NP.dot(eq_baselines, rot_matrix) # (n_bl x [3]).(3 x [3] x n_acc) -> n_bl x (first 3) x n_acc

        # proj_baselines = NP.empty((eq_baselines.shape[0], eq_baselines.shape[1], len(self.lst)))
        # for i in xrange(len(self.lst)):
        #     rot_matrix = NP.asarray([[NP.sin(ha[i]),               NP.cos(ha[i]),             0.0],
        #                              [-NP.sin(dec[i])*NP.cos(ha[i]), NP.sin(dec[i])*NP.sin(ha[i]), NP.cos(dec[i])],
        #                              [NP.cos(dec[i])*NP.cos(ha[i]), -NP.cos(dec[i])*NP.sin(ha[i]), NP.sin(dec[i])]])

        #     proj_baselines[:,:,i] = NP.dot(eq_baselines, rot_matrix.T)

        # self.projected_baselines = proj_baselines

    #############################################################################

    def conjugate(self, ind=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Flips the baseline vectors and conjugates the visibilies for a specified
        subset of baselines.

        Inputs:

        ind      [scalar, list or numpy array] Indices pointing to specific
                 baseline vectors which need to be flipped. Default = None means
                 no flipping or conjugation. If all baselines are to be
                 flipped, either provide all the indices in ind or set ind="all"

        verbose  [boolean] If set to True (default), print diagnostic and
                 progress messages. If set to False, no such messages are
                 printed.
        ------------------------------------------------------------------------
        """

        if ind is not None:
            if isinstance(ind, str):
                if ind != 'all':
                    raise ValueError('Value of ind must be "all" if set to string')
                ind = NP.arange(self.baselines.shape[0])
            elif isinstance(ind, int):
                ind = [ind]
            elif isinstance(ind, NP.ndarray):
                ind = ind.tolist()
            elif not isinstance(ind, list):
                raise TypeError('ind must be string "all", scalar interger, list or numpy array')

            ind = NP.asarray(ind)
            if NP.any(ind >= self.baselines.shape[0]):
                raise IndexError('Out of range indices found.')

            self.labels = [tuple(reversed(self.labels[i])) if i in ind else self.labels[i] for i in xrange(len(self.labels))]
            self.baselines[ind,:] = -self.baselines[ind,:]
            self.baseline_orientations = NP.angle(self.baselines[:,0] + 1j * self.baselines[:,1])
            if self.vis_freq is not None:
                self.vis_freq[ind,:,:] = self.vis_freq[ind,:,:].conj()
            if self.skyvis_freq is not None:
                self.skyvis_freq[ind,:,:] = self.skyvis_freq[ind,:,:].conj()
            if self.vis_noise_freq is not None:
                self.vis_noise_freq[ind,:,:] = self.vis_noise_freq[ind,:,:].conj()
            if self.projected_baselines is not None:
                self.projected_baselines[ind,:,:] = -self.projected_baselines[ind,:,:]

            if verbose:
                print 'Certain baselines have been flipped and their visibilities conjugated. Use delay_transform() to update the delay spectra.'

    #############################################################################

    def delay_transform(self, pad=1.0, freq_wts=None, verbose=True):

        """
        ------------------------------------------------------------------------
        Transforms the visibilities from frequency axis onto delay (time) axis
        using an IFFT. This is performed for noiseless sky visibilities, thermal
        noise in visibilities, and observed visibilities.

        Inputs:

        pad         [scalar] Non-negative scalar indicating padding fraction
                    relative to the number of frequency channels. For e.g., a
                    pad of 1.0 pads the frequency axis with zeros of the same
                    width as the number of channels. After the delay transform,
                    the transformed visibilities are downsampled by a factor of
                    1+pad. If a negative value is specified, delay transform
                    will be performed with no padding

        freq_wts    [numpy vector or array] window shaping to be applied before
                    computing delay transform. It can either be a vector or size
                    equal to the number of channels (which will be applied to all
                    time instances for all baselines), or a nchan x n_snapshots
                    numpy array which will be applied to all baselines, or a
                    n_baselines x nchan numpy array which will be applied to all
                    timestamps, or a n_baselines x nchan x n_snapshots numpy
                    array. Default (None) will not apply windowing and only the
                    inherent bandpass will be used.

        verbose     [boolean] If set to True (default), print diagnostic and
                    progress messages. If set to False, no such messages are
                    printed.
        ------------------------------------------------------------------------
        """

        if verbose:
            print 'Preparing to compute delay transform...\n\tChecking input parameters for compatibility...'

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        if freq_wts is not None:
            if freq_wts.size == self.channels.size:
                freq_wts = NP.repeat(NP.expand_dims(NP.repeat(freq_wts.reshape(1,-1), self.baselines.shape[0], axis=0), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.channels.size * self.n_acc:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(self.channels.size, -1), axis=0), self.baselines.shape[0], axis=0)
            elif freq_wts.size == self.channels.size * self.baselines.shape[0]:
                freq_wts = NP.repeat(NP.expand_dims(freq_wts.reshape(-1, self.channels.size), axis=2), self.n_acc, axis=2)
            elif freq_wts.size == self.channels.size * self.baselines.shape[0] * self.n_acc:
                freq_wts = freq_wts.reshape(self.baselines.shape[0], self.channels.size, self.n_acc)
            else:
                raise ValueError('window shape dimensions incompatible with number of channels and/or number of tiemstamps.')
            self.bp_wts = freq_wts
            if verbose:
                print '\tFrequency window weights assigned.'

        if verbose:
            print '\tInput parameters have been verified to be compatible.\n\tProceeding to compute delay transform.'

        self.lags = DSP.spectral_axis(self.channels.size, delx=self.freq_resolution, use_real=False, shift=True)
        if pad == 0.0:
            self.vis_lag = DSP.FT1D(self.vis_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            self.skyvis_lag = DSP.FT1D(self.skyvis_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            self.vis_noise_lag = DSP.FT1D(self.vis_noise_freq * self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            self.lag_kernel = DSP.FT1D(self.bp * self.bp_wts, ax=1, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            if verbose:
                print '\tDelay transform computed without padding.'
        else:
            npad = int(self.channels.size * pad)
            self.vis_lag = DSP.FT1D(NP.pad(self.vis_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution
            self.skyvis_lag = DSP.FT1D(NP.pad(self.skyvis_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution
            self.vis_noise_lag = DSP.FT1D(NP.pad(self.vis_noise_freq * self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution
            self.lag_kernel = DSP.FT1D(NP.pad(self.bp * self.bp_wts, ((0,0),(0,npad),(0,0)), mode='constant'), ax=1, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution

            if verbose:
                print '\tDelay transform computed with padding fraction {0:.1f}'.format(pad)
            self.vis_lag = DSP.downsampler(self.vis_lag, 1+pad, axis=1)
            self.skyvis_lag = DSP.downsampler(self.skyvis_lag, 1+pad, axis=1)
            self.vis_noise_lag = DSP.downsampler(self.vis_noise_lag, 1+pad, axis=1)
            self.lag_kernel = DSP.downsampler(self.lag_kernel, 1+pad, axis=1)
            if verbose:
                print '\tDelay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'delay_transform() completed successfully.'

    #############################################################################

    def multi_window_delay_transform(self, bw_eff, freq_center=None, shape=None,
                                     pad=1.0, verbose=True):

        """
        ------------------------------------------------------------------------
        Computes delay transform on multiple frequency windows with specified
        weights

        Inputs:

        bw_eff       [scalar, list, numpy array] Effective bandwidths of the
                     selected frequency windows. If a scalar is provided, the
                     same will be applied to all frequency windows.

        freq_center  [scalar, list, numpy array] Frequency centers of the
                     selected frequency windows. If a scalar is provided, the
                     same will be applied to all frequency windows. Default=None
                     uses the center frequency from the class attribute named
                     channels

        shape        [string] specifies frequency window shape. Accepted values
                     are 'rect' or 'RECT' (for rectangular), 'bnw' and 'BNW'
                     (for Blackman-Nuttall), and 'bhw' or 'BHW' (for Blackman-
                     Harris). Default=None sets it to 'rect' (rectangular
                     window)

        pad          [scalar] Non-negative scalar indicating padding fraction
                     relative to the number of frequency channels. For e.g., a
                     pad of 1.0 pads the frequency axis with zeros of the same
                     width as the number of channels. After the delay transform,
                     the transformed visibilities are downsampled by a factor of
                     1+pad. If a negative value is specified, delay transform
                     will be performed with no padding

        verbose      [boolean] If set to True (default), print diagnostic and
                     progress messages. If set to False, no such messages are
                     printed.

        Output:

        A dictionary containing information under the following keys:
        skyvis_lag        Numpy array of pure sky visibilities delay spectra of
                          size n_bl x n_windows x nchan x n_snaps

        vis_noise_lag     Numpy array of noise delay spectra of size
                          size n_bl x n_windows x nchan x n_snaps

        lag_kernel        Numpy array of delay kernel of size
                          size n_bl x n_windows x nchan x n_snaps

        lag_corr_length   Numpy array of correlation length (in units of number
                          of delay samples) due to convolving kernel in delay
                          space. This is the number by which the delay spectra
                          obtained have to be downsampled by to get independent
                          samples of delay spectra
        ------------------------------------------------------------------------
        """

        try:
            bw_eff
        except NameError:
            raise NameError('Effective bandwidth must be specified')
        else:
            if not isinstance(bw_eff, (int, float, list, NP.ndarray)):
                raise TypeError('Effective bandwidth must be a scalar, list or numpy array')
            bw_eff = NP.asarray(bw_eff).reshape(-1)
            if NP.any(bw_eff <= 0.0):
                raise ValueError('All values in effective bandwidth must be strictly positive')

        if freq_center is None:
            freq_center = NP.asarray(self.channels[int(0.5*self.channels.size)]).reshape(-1)
        elif isinstance(freq_center, (int, float, list, NP.ndarray)):
            freq_center = NP.asarray(freq_center).reshape(-1)
            if NP.any((freq_center <= self.channels.min()) | (freq_center >= self.channels.max())):
                raise ValueError('Frequency centers must lie strictly inside the observing band')
        else:
            raise TypeError('Frequency center(s) must be scalar, list or numpy array')

        if (bw_eff.size == 1) and (freq_center.size > 1):
            bw_eff = NP.repeat(bw_eff, freq_center.size)
        elif (bw_eff.size > 1) and (freq_center.size == 1):
            freq_center = NP.repeat(freq_center, bw_eff.size)
        elif bw_eff.size != freq_center.size:
            raise ValueError('Effective bandwidth(s) and frequency center(s) must have same number of elements')

        if shape is not None:
            if not isinstance(shape, str):
                raise TypeError('Window shape must be a string')
            if shape not in ['rect', 'bhw', 'bnw', 'RECT', 'BHW', 'BNW']:
                raise ValueError('Invalid value for window shape specified.')
        else:
            shape = 'rect'

        if not isinstance(pad, (int, float)):
            raise TypeError('pad fraction must be a scalar value.')
        if pad < 0.0:
            pad = 0.0
            if verbose:
                print '\tPad fraction found to be negative. Resetting to 0.0 (no padding will be applied).'

        freq_wts = NP.empty((bw_eff.size, self.channels.size))
        frac_width = DSP.window_N2width(n_window=None, shape=shape)
        window_loss_factor = 1 / frac_width
        n_window = NP.round(window_loss_factor * bw_eff / self.freq_resolution).astype(NP.int)

        ind_freq_center, ind_channels, dfrequency = LKP.find_1NN(self.channels.reshape(-1,1), freq_center.reshape(-1,1), distance_ULIM=0.5*self.freq_resolution, remove_oob=True)
        sortind = NP.argsort(ind_channels)
        ind_freq_center = ind_freq_center[sortind]
        ind_channels = ind_channels[sortind]
        dfrequency = dfrequency[sortind]
        n_window = n_window[sortind]

        for i,ind_chan in enumerate(ind_channels):
            window = DSP.windowing(n_window[i], shape=shape, centering=True)
            window_chans = self.channels[ind_chan] + self.freq_resolution * (NP.arange(n_window[i]) - int(n_window[i]/2))
            ind_window_chans, ind_chans, dfreq = LKP.find_1NN(self.channels.reshape(-1,1), window_chans.reshape(-1,1), distance_ULIM=0.5*self.freq_resolution, remove_oob=True)
            sind = NP.argsort(ind_window_chans)
            ind_window_chans = ind_window_chans[sind]
            ind_chans = ind_chans[sind]
            dfreq = dfreq[sind]
            window = window[ind_window_chans]
            window = NP.pad(window, ((ind_chans.min(), self.channels.size-1-ind_chans.max())), mode='constant', constant_values=((0.0,0.0)))
            freq_wts[i,:] = window

        lags = DSP.spectral_axis(self.channels.size, delx=self.freq_resolution, use_real=False, shift=True)
        if pad == 0.0:
            skyvis_lag = DSP.FT1D(self.skyvis_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ax=2, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            vis_noise_lag = DSP.FT1D(self.vis_noise_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ax=2, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            lag_kernel = DSP.FT1D(self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ax=2, inverse=True, use_real=False, shift=True) * self.channels.size * self.freq_resolution
            if verbose:
                print '\tMulti-window delay transform computed without padding.'
        else:
            npad = int(self.channels.size * pad)
            skyvis_lag = DSP.FT1D(NP.pad(self.skyvis_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution
            vis_noise_lag = DSP.FT1D(NP.pad(self.vis_noise_freq[:,NP.newaxis,:,:] * self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution
            lag_kernel = DSP.FT1D(NP.pad(self.bp[:,NP.newaxis,:,:] * freq_wts[NP.newaxis,:,:,NP.newaxis], ((0,0),(0,0),(0,npad),(0,0)), mode='constant'), ax=2, inverse=True, use_real=False, shift=True) * (npad + self.channels.size) * self.freq_resolution

            if verbose:
                print '\tMulti-window delay transform computed with padding fraction {0:.1f}'.format(pad)
            skyvis_lag = DSP.downsampler(skyvis_lag, 1+pad, axis=2)
            vis_noise_lag = DSP.downsampler(vis_noise_lag, 1+pad, axis=2)
            lag_kernel = DSP.downsampler(lag_kernel, 1+pad, axis=2)
            if verbose:
                print '\tMulti-window delay transform products downsampled by factor of {0:.1f}'.format(1+pad)
                print 'multi_window_delay_transform() completed successfully.'

        return {'skyvis_lag': skyvis_lag, 'vis_noise_lag': vis_noise_lag, 'lag_kernel': lag_kernel, 'lag_corr_length': self.channels.size / NP.sum(freq_wts, axis=1)}

    #############################################################################

    def concatenate(self, others, axis):

        """
        -------------------------------------------------------------------------
        Concatenates different visibility data sets from instances of class
        InterferometerArray along baseline, frequency or time axis.

        Inputs:

        others       [instance of class Interferometer Array or list of such
                     instances] Instance or list of instances of class
                     InterferometerArray whose visibility data have to be
                     concatenated to the current instance.

        axis         [scalar] Axis along which visibility data sets are to be
                     concatenated. Accepted values are 0 (concatenate along
                     baseline axis), 1 (concatenate frequency channels), or 2
                     (concatenate along time/snapshot axis). No default
        -------------------------------------------------------------------------
        """

        try:
            others, axis
        except NameError:
            raise NameError('An instance of class InterferometerArray or a list of such instances and the axis along which they are to be concatenated must be provided.')

        if isinstance(others, list):
            for other in others:
                if not isinstance(other, InterferometerArray):
                    raise TypeError('The interferometer array data to be concatenated must be an instance of class InterferometerArray or a list of such instances')
            loo = [self]+others
        elif isinstance(others, InterferometerArray):
            loo = [self, others]
        elif not isinstance(other, InterferometerArray):
            raise TypeError('The interferometer array data to be concatenated must be an instance of class InterferometerArray or a list of such instances')

        if not isinstance(axis, int):
            raise TypeError('axis must be an integer')

        self_shape = self.skyvis_freq.shape

        if axis >= len(self_shape):
            raise ValueError('Specified axis not found in the visibility data.')
        elif axis == -1:
            axis = len(self_shape) - 1
        elif axis < -1:
            raise ValueError('Specified axis not found in the visibility data.')

        self.skyvis_freq = NP.concatenate(tuple([elem.skyvis_freq for elem in loo]), axis=axis)
        if self.vis_freq is not None:
            self.vis_freq = NP.concatenate(tuple([elem.vis_freq for elem in loo]), axis=axis)
        if self.vis_noise_freq is not None:
            self.vis_noise_freq = NP.concatenate(tuple([elem.vis_noise_freq for elem in loo]), axis=axis)
        if self.vis_rms_freq is not None:
            self.vis_rms_freq  = NP.concatenate(tuple([elem.vis_rms_freq for elem in loo]), axis=axis)
        self.bp = NP.concatenate(tuple([elem.bp for elem in loo]), axis=axis)
        self.bp_wts = NP.concatenate(tuple([elem.bp_wts for elem in loo]), axis=axis)
        self.Tsys = NP.concatenate(tuple([elem.Tsys for elem in loo]), axis=axis)
        if self.gradient_mode is not None:
            self.gradient[self.gradient_mode] = NP.concatenate(tuple([elem.gradient[self.gradient_mode] for elem in loo]), axis=axis+1)

        if not self.Tsysinfo:
            for elem in loo:
                if elem.Tsysinfo:
                    self.Tsysinfo = elem.Tsysinfo
        if axis != 1:
            if self.skyvis_lag is not None:
                self.skyvis_lag = NP.concatenate(tuple([elem.skyvis_lag for elem in loo]), axis=axis)
            if self.vis_lag is not None:
                self.vis_lag = NP.concatenate(tuple([elem.vis_lag for elem in loo]), axis=axis)
            if self.vis_noise_lag is not None:
                self.vis_noise_lag = NP.concatenate(tuple([elem.vis_noise_lag for elem in loo]), axis=axis)

        if axis == 0: # baseline axis
            for elem in loo:
                if elem.baseline_coords != self.baseline_coords:
                    raise ValueError('Coordinate systems for the baseline vectors are mismatched.')
            self.baselines = NP.vstack(tuple([elem.baselines for elem in loo]))
            self.baseline_lengths = NP.sqrt(NP.sum(self.baselines**2, axis=1))
            self.baseline_orientations = NP.angle(self.baselines[:,0] + 1j * self.baselines[:,1])
            self.projected_baselines = NP.vstack(tuple([elem.projected_baselines for elem in loo]))
            self.labels = [label for elem in loo for label in elem.labels]
            self.A_eff = NP.vstack(tuple([elem.A_eff for elem in loo]))
            self.eff_Q = NP.vstack(tuple([elem.eff_Q for elem in loo]))
        elif axis == 1: # Frequency axis
            self.channels = NP.hstack(tuple([elem.channels for elem in loo]))
            self.A_eff = NP.hstack(tuple([elem.A_eff for elem in loo]))
            self.eff_Q = NP.hstack(tuple([elem.eff_Q for elem in loo]))
            # self.delay_transform()
        elif axis == 2: # time axis
            # self.timestamp = [timestamp for elem in loo for timestamp in elem.timestamp]
            self.t_acc = [t_acc for elem in loo for t_acc in elem.t_acc]
            self.n_acc = len(self.t_acc)
            self.t_obs = sum(self.t_acc)
            self.pointing_center = NP.vstack(tuple([elem.pointing_center for elem in loo]))
            self.phase_center = NP.vstack(tuple([elem.phase_center for elem in loo]))
            self.lst = [lst for elem in loo for lst in elem.lst]
            self.timestamp = [timestamp for elem in loo for timestamp in elem.timestamp]
            self.Tsysinfo = [Tsysinfo for elem in loo for Tsysinfo in elem.Tsysinfo]

    #############################################################################

    def save(self, outfile, fmt='HDF5', tabtype='BinTableHDU', npz=True,
             overwrite=False, uvfits_parms=None, verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the interferometer array information to disk in HDF5, FITS, NPZ
        and UVFITS formats

        Inputs:

        outfile      [string] Filename with full path to be saved to. Will be
                     appended with '.hdf5' or '.fits' extension depending on
                     input keyword fmt. If input npz is set to True, the
                     simulated visibilities will also get stored in '.npz'
                     format. Depending on parameters in uvfits_parms, three
                     UVFITS files will also be created whose names will be
                     outfile+'-noiseless', outfile+'-noisy' and
                     'outfile+'-noise' appended with '.uvfits'

        Keyword Input(s):

        fmt          [string] string specifying the format of the output.
                     Accepted values are 'HDF5' (default) and 'FITS'.
                     The file names will be appended with '.hdf5' or '.fits'
                     respectively

        tabtype      [string] indicates table type for one of the extensions in
                     the FITS file. Allowed values are 'BinTableHDU' and
                     'TableHDU' for binary and ascii tables respectively. Default
                     is 'BinTableHDU'. Only applies if input fmt is set to 'FITS'

        npz          [boolean] True (default) indicates a numpy NPZ format file
                     is created in addition to the FITS file to store essential
                     attributes of the class InterferometerArray for easy
                     handing over of python files

        overwrite    [boolean] True indicates overwrite even if a file already
                     exists. Default = False (does not overwrite). Beware this
                     may not work reliably for UVFITS output when uvfits_method
                     is set to None or 'uvdata' and hence always better to make
                     sure the output file does not exist already

        uvfits_parms [dictionary] specifies basic parameters required for
                     saving in UVFITS format. If set to None (default), the
                     data will not be saved in UVFITS format. To save in UVFITS
                     format, the following keys and values are required:
                     'ref_point'    [dictionary] Contains information about the
                                    reference position to which projected
                                    baselines and rotated visibilities are to
                                    be computed. Default=None (no additional
                                    phasing will be performed). It must be
                                    contain the following keys with the
                                    following values:
                                    'coords'    [string] Refers to the
                                                coordinate system in which value
                                                in key 'location' is specified
                                                in. Accepted values are 'radec',
                                                'hadec', 'altaz' and 'dircos'
                                    'location'  [numpy array] Must be a Mx2 (if
                                                value in key 'coords' is set to
                                                'radec', 'hadec', 'altaz' or
                                                'dircos') or Mx3 (if value in
                                                key 'coords' is set to
                                                'dircos'). M can be 1 or equal
                                                to number of timestamps. If M=1,
                                                the same reference point in the
                                                same coordinate system will be
                                                repeated for all tiemstamps. If
                                                value under key 'coords' is set
                                                to 'radec', 'hadec' or 'altaz',
                                                the value under this key
                                                'location' must be in units of
                                                degrees.
                     'method'       [string] specifies method to be used in
                                    saving in UVFITS format. Accepted values are
                                    'uvdata', 'uvfits' or None (default). If set
                                    to 'uvdata', the UVFITS writer in uvdata
                                    module is used. If set to 'uvfits', the
                                    in-house UVFITS writer is used. If set to
                                    None, first uvdata module will be attempted
                                    but if it fails then the in-house UVFITS
                                    writer will be tried.

        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('No filename provided. Aborting InterferometerArray.save()...')

        if fmt.lower() not in ['hdf5', 'fits']:
            raise ValueError('Invalid output file format specified')
        if fmt.lower() == 'hdf5':
            filename = outfile + '.' + fmt.lower()
        if fmt.lower() == 'fits':
            filename = outfile + '.' + fmt.lower()

        if verbose:
            print '\nSaving information about interferometer...'

        if fmt.lower() == 'fits':
            use_ascii = False
            if tabtype == 'TableHDU':
                use_ascii = True

            hdulist = []

            hdulist += [fits.PrimaryHDU()]
            hdulist[0].header['latitude'] = (self.latitude, 'Latitude of interferometer')
            hdulist[0].header['longitude'] = (self.longitude, 'Longitude of interferometer')
            hdulist[0].header['altitude'] = (self.altitude, 'Altitude of interferometer')
            hdulist[0].header['baseline_coords'] = (self.baseline_coords, 'Baseline coordinate system')
            hdulist[0].header['freq_resolution'] = (self.freq_resolution, 'Frequency Resolution (Hz)')
            hdulist[0].header['pointing_coords'] = (self.pointing_coords, 'Pointing coordinate system')
            hdulist[0].header['phase_center_coords'] = (self.phase_center_coords, 'Phase center coordinate system')
            hdulist[0].header['skycoords'] = (self.skycoords, 'Sky coordinate system')
            if 'id' in self.telescope:
                hdulist[0].header['telescope'] = (self.telescope['id'], 'Telescope Name')
            if self.telescope['groundplane'] is not None:
                hdulist[0].header['groundplane'] = (self.telescope['groundplane'], 'Ground plane height')
            if self.simparms_file is not None:
                hdulist[0].header['simparms'] = (self.simparms_file, 'YAML file with simulation parameters')
            if self.gradient_mode is not None:
                hdulist[0].header['gradient_mode'] = (self.gradient_mode, 'Visibility Gradient Mode')
            if self.gaininfo is not None:
                hdulist[0].header['gainsfile'] = (outfile+'.gains.hdf5', 'Gains File')
            hdulist[0].header['element_shape'] = (self.telescope['shape'], 'Antenna element shape')
            hdulist[0].header['element_size'] = (self.telescope['size'], 'Antenna element size')
            hdulist[0].header['element_ocoords'] = (self.telescope['ocoords'], 'Antenna element orientation coordinates')
            hdulist[0].header['t_obs'] = (self.t_obs, 'Observing duration (s)')
            hdulist[0].header['n_acc'] = (self.n_acc, 'Number of accumulations')
            hdulist[0].header['flux_unit'] = (self.flux_unit, 'Unit of flux density')
            hdulist[0].header['EXTNAME'] = 'PRIMARY'

            if verbose:
                print '\tCreated a primary HDU.'

            hdulist += [fits.ImageHDU(self.telescope['orientation'], name='Antenna element orientation')]
            if verbose:
                print '\tCreated an extension for antenna element orientation.'

            cols = []
            if self.lst:
                cols += [fits.Column(name='LST', format='D', array=NP.asarray(self.lst).ravel())]
                cols += [fits.Column(name='pointing_longitude', format='D', array=self.pointing_center[:,0])]
                cols += [fits.Column(name='pointing_latitude', format='D', array=self.pointing_center[:,1])]
                cols += [fits.Column(name='phase_center_longitude', format='D', array=self.phase_center[:,0])]
                cols += [fits.Column(name='phase_center_latitude', format='D', array=self.phase_center[:,1])]

            columns = _astropy_columns(cols, tabtype=tabtype)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'POINTING AND PHASE CENTER INFO')
            hdulist += [tbhdu]
            if verbose:
                print '\tCreated pointing and phase center information table.'

            label_lengths = [len(label[0]) for label in self.labels]
            maxlen = max(label_lengths)
            labels = NP.asarray(self.labels, dtype=[('A2', '|S{0:0d}'.format(maxlen)), ('A1', '|S{0:0d}'.format(maxlen))])
            cols = []
            cols += [fits.Column(name='A1', format='{0:0d}A'.format(maxlen), array=labels['A1'])]
            cols += [fits.Column(name='A2', format='{0:0d}A'.format(maxlen), array=labels['A2'])]
            # cols += [fits.Column(name='labels', format='5A', array=NP.asarray(self.labels))]

            columns = _astropy_columns(cols, tabtype=tabtype)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'LABELS')
            hdulist += [tbhdu]
            if verbose:
                print '\tCreated extension table containing baseline labels.'

            hdulist += [fits.ImageHDU(self.baselines, name='baselines')]
            if verbose:
                print '\tCreated an extension for baseline vectors.'

            if self.projected_baselines is not None:
                hdulist += [fits.ImageHDU(self.projected_baselines, name='proj_baselines')]
                if verbose:
                    print '\tCreated an extension for projected baseline vectors.'

            if self.layout:
                label_lengths = [len(label) for label in self.layout['labels']]
                maxlen = max(label_lengths)
                cols = []
                cols += [fits.Column(name='labels', format='{0:0d}A'.format(maxlen), array=self.layout['labels'])]
                cols += [fits.Column(name='ids', format='J', array=self.layout['ids'])]
                cols += [fits.Column(name='positions', format='3D', array=self.layout['positions'])]
                columns = _astropy_columns(cols, tabtype=tabtype)
                tbhdu = fits.new_table(columns)
                tbhdu.header.set('EXTNAME', 'LAYOUT')
                tbhdu.header.set('COORDS', self.layout['coords'])
                hdulist += [tbhdu]

            hdulist += [fits.ImageHDU(self.A_eff, name='Effective area')]
            if verbose:
                print '\tCreated an extension for effective area.'

            hdulist += [fits.ImageHDU(self.eff_Q, name='Interferometer efficiency')]
            if verbose:
                print '\tCreated an extension for interferometer efficiency.'

            cols = []
            cols += [fits.Column(name='frequency', format='D', array=self.channels)]
            if self.lags is not None:
                cols += [fits.Column(name='lag', format='D', array=self.lags)]

            columns = _astropy_columns(cols, tabtype=tabtype)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'SPECTRAL INFO')
            hdulist += [tbhdu]
            if verbose:
                print '\tCreated spectral information table.'

            if self.t_acc:
                hdulist += [fits.ImageHDU(self.t_acc, name='t_acc')]
                if verbose:
                    print '\tCreated an extension for accumulation times.'

            cols = []
            if isinstance(self.timestamp[0], str):
                cols += [fits.Column(name='timestamps', format='24A', array=NP.asarray(self.timestamp))]
            elif isinstance(self.timestamp[0], float):
                cols += [fits.Column(name='timestamps', format='D', array=NP.asarray(self.timestamp))]
            else:
                raise TypeError('Invalid data type for timestamps')

            columns = _astropy_columns(cols, tabtype=tabtype)

            tbhdu = fits.new_table(columns)
            tbhdu.header.set('EXTNAME', 'TIMESTAMPS')
            hdulist += [tbhdu]
            if verbose:
                print '\tCreated extension table containing timestamps.'

            if self.Tsysinfo:
                cols = []
                cols += [fits.Column(name='Trx', format='D', array=NP.asarray([elem['Trx'] for elem in self.Tsysinfo], dtype=NP.float))]
                cols += [fits.Column(name='Tant0', format='D', array=NP.asarray([elem['Tant']['T0'] for elem in self.Tsysinfo], dtype=NP.float))]
                cols += [fits.Column(name='f0', format='D', array=NP.asarray([elem['Tant']['f0'] for elem in self.Tsysinfo], dtype=NP.float))]
                cols += [fits.Column(name='spindex', format='D', array=NP.asarray([elem['Tant']['spindex'] for elem in self.Tsysinfo], dtype=NP.float))]
                columns = _astropy_columns(cols, tabtype=tabtype)
                tbhdu = fits.new_table(columns)
                tbhdu.header.set('EXTNAME', 'TSYSINFO')
                hdulist += [tbhdu]

            hdulist += [fits.ImageHDU(self.Tsys, name='Tsys')]
            if verbose:
                print '\tCreated an extension for Tsys.'

            if self.vis_rms_freq is not None:
                hdulist += [fits.ImageHDU(self.vis_rms_freq, name='freq_channel_noise_rms_visibility')]
                if verbose:
                    print '\tCreated an extension for simulated visibility noise rms per channel.'

            if self.vis_freq is not None:
                hdulist += [fits.ImageHDU(self.vis_freq.real, name='real_freq_obs_visibility')]
                hdulist += [fits.ImageHDU(self.vis_freq.imag, name='imag_freq_obs_visibility')]
                if verbose:
                    print '\tCreated extensions for real and imaginary parts of observed visibility frequency spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_freq.shape)

            if self.skyvis_freq is not None:
                hdulist += [fits.ImageHDU(self.skyvis_freq.real, name='real_freq_sky_visibility')]
                hdulist += [fits.ImageHDU(self.skyvis_freq.imag, name='imag_freq_sky_visibility')]
                if verbose:
                    print '\tCreated extensions for real and imaginary parts of noiseless sky visibility frequency spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.skyvis_freq.shape)

            if self.vis_noise_freq is not None:
                hdulist += [fits.ImageHDU(self.vis_noise_freq.real, name='real_freq_noise_visibility')]
                hdulist += [fits.ImageHDU(self.vis_noise_freq.imag, name='imag_freq_noise_visibility')]
                if verbose:
                    print '\tCreated extensions for real and imaginary parts of visibility noise frequency spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_noise_freq.shape)

            if self.gradient_mode is not None:
                for gradkey in self.gradient:
                    hdulist += [fits.ImageHDU(self.gradient[gradkey].real, name='real_freq_sky_visibility_gradient_wrt_{0}'.format(gradkey))]
                    hdulist += [fits.ImageHDU(self.gradient[gradkey].imag, name='imag_freq_sky_visibility_gradient_wrt_{0}'.format(gradkey))]
                    if verbose:
                        print '\tCreated extensions for real and imaginary parts of gradient of sky visibility frequency spectrum wrt {0} of size {1[0]} x {1[1]} x {1[2]} x {1[3]}'.format(gradkey, self.gradient[gradkey].shape)

            hdulist += [fits.ImageHDU(self.bp, name='bandpass')]
            if verbose:
                print '\tCreated an extension for bandpass functions of size {0[0]} x {0[1]} x {0[2]} as a function of baseline,  frequency, and snapshot instance'.format(self.bp.shape)

            hdulist += [fits.ImageHDU(self.bp_wts, name='bandpass_weights')]
            if verbose:
                print '\tCreated an extension for bandpass weights of size {0[0]} x {0[1]} x {0[2]} as a function of baseline,  frequency, and snapshot instance'.format(self.bp_wts.shape)

            # hdulist += [fits.ImageHDU(self.lag_kernel.real, name='lag_kernel_real')]
            # hdulist += [fits.ImageHDU(self.lag_kernel.imag, name='lag_kernel_imag')]
            # if verbose:
            #     print '\tCreated an extension for impulse response of frequency bandpass shape of size {0[0]} x {0[1]} x {0[2]} as a function of baseline, lags, and snapshot instance'.format(self.lag_kernel.shape)

            if self.vis_lag is not None:
                hdulist += [fits.ImageHDU(self.vis_lag.real, name='real_lag_visibility')]
                hdulist += [fits.ImageHDU(self.vis_lag.imag, name='imag_lag_visibility')]
                if verbose:
                    print '\tCreated extensions for real and imaginary parts of observed visibility delay spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_lag.shape)

            if self.skyvis_lag is not None:
                hdulist += [fits.ImageHDU(self.skyvis_lag.real, name='real_lag_sky_visibility')]
                hdulist += [fits.ImageHDU(self.skyvis_lag.imag, name='imag_lag_sky_visibility')]
                if verbose:
                    print '\tCreated extensions for real and imaginary parts of noiseless sky visibility delay spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.skyvis_lag.shape)

            if self.vis_noise_lag is not None:
                hdulist += [fits.ImageHDU(self.vis_noise_lag.real, name='real_lag_noise_visibility')]
                hdulist += [fits.ImageHDU(self.vis_noise_lag.imag, name='imag_lag_noise_visibility')]
                if verbose:
                    print '\tCreated extensions for real and imaginary parts of visibility noise delay spectrum of size {0[0]} x {0[1]} x {0[2]}'.format(self.vis_noise_lag.shape)

            if verbose:
                print '\tNow writing FITS file to disk...'
            hdu = fits.HDUList(hdulist)
            hdu.writeto(filename, clobber=overwrite)
            if self.gaininfo is not None:
                self.gaininfo.write_gaintable(outfile+'.gains.hdf5')

        elif fmt.lower() == 'hdf5':
            if overwrite:
                write_str = 'w'
            else:
                write_str = 'w-'
            with h5py.File(filename, write_str) as fileobj:
                hdr_group = fileobj.create_group('header')
                hdr_group['flux_unit'] = self.flux_unit
                tlscp_group = fileobj.create_group('telescope_parms')
                tlscp_group['latitude'] = self.latitude
                tlscp_group['longitude'] = self.longitude
                tlscp_group['altitude'] = self.altitude
                tlscp_group['latitude'].attrs['units'] = 'deg'
                tlscp_group['longitude'].attrs['units'] = 'deg'
                tlscp_group['altitude'].attrs['units'] = 'm'
                if 'id' in self.telescope:
                    tlscp_group['id'] = self.telescope['id']
                spec_group = fileobj.create_group('spectral_info')
                spec_group['freq_resolution'] = self.freq_resolution
                spec_group['freq_resolution'].attrs['units'] = 'Hz'
                spec_group['freqs'] = self.channels
                spec_group['freqs'].attrs['units'] = 'Hz'
                if self.lags is not None:
                    spec_group['lags'] = self.lags
                    spec_group['lags'].attrs['units'] = 's'
                spec_group['bp'] = self.bp
                spec_group['bp_wts'] = self.bp_wts
                if self.simparms_file is not None:
                    sim_group = fileobj.create_group('simparms')
                    sim_group['simfile'] = self.simparms_file
                antelem_group = fileobj.create_group('antenna_element')
                antelem_group['shape'] = self.telescope['shape']
                antelem_group['size'] = self.telescope['size']
                antelem_group['size'].attrs['units'] = 'm'
                antelem_group['ocoords'] = self.telescope['ocoords']
                antelem_group['orientation'] = self.telescope['orientation']
                if self.telescope['ocoords'] != 'dircos':
                    antelem_group['orientation'].attrs['units'] = 'deg'
                if 'groundplane' in self.telescope:
                    if self.telescope['groundplane'] is not None:
                        antelem_group['groundplane'] = self.telescope['groundplane']
                if self.layout:
                    layout_group = fileobj.create_group('layout')
                    layout_group['positions'] = self.layout['positions']
                    layout_group['positions'].attrs['units'] = 'm'
                    layout_group['positions'].attrs['coords'] = self.layout['coords']
                    layout_group['labels'] = self.layout['labels']
                    layout_group['ids'] = self.layout['ids']
                timing_group = fileobj.create_group('timing')
                timing_group['t_obs'] = self.t_obs
                timing_group['n_acc'] = self.n_acc
                if self.t_acc:
                    timing_group['t_acc'] = self.t_acc
                timing_group['timestamps'] = NP.asarray(self.timestamp)
                sky_group = fileobj.create_group('skyparms')
                sky_group['pointing_coords'] = self.pointing_coords
                sky_group['phase_center_coords'] = self.phase_center_coords
                sky_group['skycoords'] = self.skycoords
                sky_group['LST'] = NP.asarray(self.lst).ravel()
                sky_group['LST'].attrs['units'] = 'deg'
                sky_group['pointing_center'] = self.pointing_center
                sky_group['phase_center'] = self.phase_center
                array_group = fileobj.create_group('array')
                label_lengths = [len(label[0]) for label in self.labels]
                maxlen = max(label_lengths)
                labels = NP.asarray(self.labels, dtype=[('A2', '|S{0:0d}'.format(maxlen)), ('A1', '|S{0:0d}'.format(maxlen))])
                array_group['labels'] = labels
                array_group['baselines'] = self.baselines
                array_group['baseline_coords'] = self.baseline_coords
                array_group['baselines'].attrs['coords'] = 'local-ENU'
                array_group['baselines'].attrs['units'] = 'm'
                array_group['projected_baselines'] = self.baselines
                array_group['baselines'].attrs['coords'] = 'eq-XYZ'
                array_group['baselines'].attrs['units'] = 'm'
                instr_group = fileobj.create_group('instrument')
                instr_group['effective_area'] = self.A_eff
                instr_group['effective_area'].attrs['units'] = 'm^2'
                instr_group['efficiency'] = self.eff_Q
                if self.Tsysinfo:
                    instr_group['Trx'] = NP.asarray([elem['Trx'] for elem in self.Tsysinfo], dtype=NP.float)
                    instr_group['Tant0'] = NP.asarray([elem['Tant']['T0'] for elem in self.Tsysinfo], dtype=NP.float)
                    instr_group['f0'] = NP.asarray([elem['Tant']['f0'] for elem in self.Tsysinfo], dtype=NP.float)
                    instr_group['spindex'] = NP.asarray([elem['Tant']['spindex'] for elem in self.Tsysinfo], dtype=NP.float)
                    instr_group['Trx'].attrs['units'] = 'K'
                    instr_group['Tant0'].attrs['units'] = 'K'
                    instr_group['f0'].attrs['units'] = 'Hz'
                instr_group['Tsys'] = self.Tsys
                instr_group['Tsys'].attrs['units'] = 'K'
                vis_group = fileobj.create_group('visibilities')
                visfreq_group = vis_group.create_group('freq_spectrum')
                if self.vis_rms_freq is not None:
                    visfreq_group['rms'] = self.vis_rms_freq
                    visfreq_group['rms'].attrs['units'] = 'Jy'
                if self.vis_freq is not None:
                    visfreq_group['vis'] = self.vis_freq
                    visfreq_group['vis'].attrs['units'] = 'Jy'
                if self.skyvis_freq is not None:
                    visfreq_group['skyvis'] = self.skyvis_freq
                    visfreq_group['skyvis'].attrs['units'] = 'Jy'
                if self.vis_noise_freq is not None:
                    visfreq_group['noise'] = self.vis_noise_freq
                    visfreq_group['noise'].attrs['units'] = 'Jy'
                vislags_group = vis_group.create_group('delay_spectrum')
                if self.vis_lag is not None:
                    vislags_group['vis'] = self.vis_lag
                    vislags_group['vis'].attrs['units'] = 'Jy Hz'
                if self.skyvis_lag is not None:
                    vislags_group['skyvis'] = self.skyvis_lag
                    vislags_group['skyvis'].attrs['units'] = 'Jy Hz'
                if self.vis_noise_lag is not None:
                    vislags_group['noise'] = self.vis_noise_lag
                    vislags_group['noise'].attrs['units'] = 'Jy Hz'
                if self.gradient_mode is not None:
                    visgradient_group = fileobj.create_group('gradients')
                    for gradkey in self.gradient:
                        visgradient_group[gradkey] = self.gradient[gradkey]
                if self.gaininfo is not None:
                    gains_group = fileobj.create_group('gaininfo')
                    gains_group['gainsfile'] = outfile+'.gains.hdf5'
                    self.gaininfo.write_gaintable(gains_group['gainsfile'].value)
                if self.blgroups is not None:
                    blinfo = fileobj.create_group('blgroupinfo')
                    blgrp = blinfo.create_group('groups')
                    for blkey in self.blgroups:
                        blgrp[str(blkey)] = self.blgroups[blkey]
                    revmap = blinfo.create_group('reversemap')
                    for blkey in self.bl_reversemap:
                        revmap[str(blkey)] = self.bl_reversemap[blkey]
                    
        if verbose:
            print '\tInterferometer array information written successfully to file on disk:\n\t\t{0}\n'.format(filename)

        if npz:
            if (self.vis_freq is not None) and (self.vis_noise_freq is not None):
                NP.savez_compressed(outfile+'.npz', skyvis_freq=self.skyvis_freq, vis_freq=self.vis_freq, vis_noise_freq=self.vis_noise_freq, lst=self.lst, freq=self.channels, timestamp=self.timestamp, bl=self.baselines, bl_length=self.baseline_lengths)
            else:
                NP.savez_compressed(outfile+'.npz', skyvis_freq=self.skyvis_freq, lst=self.lst, freq=self.channels, timestamp=self.timestamp, bl=self.baselines, bl_length=self.baseline_lengths)
            if verbose:
                print '\tInterferometer array information written successfully to NPZ file on disk:\n\t\t{0}\n'.format(outfile+'.npz')

        if uvfits_parms is not None:
            self.write_uvfits(outfile, uvfits_parms=uvfits_parms, overwrite=overwrite, verbose=verbose)

    #############################################################################

    def write_uvfits(self, outfile, uvfits_parms=None, overwrite=False,
                     verbose=True):

        """
        -------------------------------------------------------------------------
        Saves the interferometer array information to disk in UVFITS format

        Inputs:

        outfile      [string] Filename with full path to be saved to. Three
                     UVFITS files will also be created whose names will be
                     outfile+'-noiseless', outfile+'-noisy' and
                     'outfile+'-noise' appended with '.uvfits'

        Keyword Input(s):

        uvfits_parms [dictionary] specifies basic parameters required for
                     saving in UVFITS format. If set to None (default), the
                     data will not be saved in UVFITS format. To save in UVFITS
                     format, the following keys and values are required:
                     'ref_point'    [dictionary] Contains information about the
                                    reference position to which projected
                                    baselines and rotated visibilities are to
                                    be computed. Default=None (no additional
                                    phasing will be performed). It must be
                                    contain the following keys with the
                                    following values:
                                    'coords'    [string] Refers to the
                                                coordinate system in which value
                                                in key 'location' is specified
                                                in. Accepted values are 'radec',
                                                'hadec', 'altaz' and 'dircos'
                                    'location'  [numpy array] Must be a Mx2 (if
                                                value in key 'coords' is set to
                                                'radec', 'hadec', 'altaz' or
                                                'dircos') or Mx3 (if value in
                                                key 'coords' is set to
                                                'dircos'). M can be 1 or equal
                                                to number of timestamps. If M=1,
                                                the same reference point in the
                                                same coordinate system will be
                                                repeated for all tiemstamps. If
                                                value under key 'coords' is set
                                                to 'radec', 'hadec' or 'altaz',
                                                the value under this key
                                                'location' must be in units of
                                                degrees.
                     'method'       [string] specifies method to be used in
                                    saving in UVFITS format. Accepted values are
                                    'uvdata', 'uvfits' or None (default). If set
                                    to 'uvdata', the UVFITS writer in uvdata
                                    module is used. If set to 'uvfits', the
                                    in-house UVFITS writer is used. If set to
                                    None, first uvdata module will be attempted
                                    but if it fails then the in-house UVFITS
                                    writer will be tried.

        overwrite    [boolean] True indicates overwrite even if a file already
                     exists. Default = False (does not overwrite). Beware this
                     may not work reliably if uvfits_method is set to None or
                     'uvdata' and hence always better to make sure the output
                     file does not exist already

        verbose      [boolean] If True (default), prints diagnostic and progress
                     messages. If False, suppress printing such messages.
        -------------------------------------------------------------------------
        """

        if uvfits_parms is not None:
            if not isinstance(uvfits_parms, dict):
                raise TypeError('Input uvfits_parms must be a dictionary')
            if 'ref_point' not in uvfits_parms:
                uvfits_parms['ref_point'] = None
            if 'method' not in uvfits_parms:
                uvfits_parms['method'] = None
            dataobj = InterferometerData(self, ref_point=uvfits_parms['ref_point'])
            for datakey in dataobj.infodict['data_array']:
                dataobj.write(outfile+'-{0}.uvfits'.format(datakey), datatype=datakey, fmt='UVFITS', uvfits_method=uvfits_parms['method'], overwrite=overwrite)

#################################################################################

class ApertureSynthesis(object):

    """
    ----------------------------------------------------------------------------
    Class to manage aperture synthesis of visibility measurements of a
    multi-element interferometer array.

    Attributes:

    ia          [instance of class InterferometerArray] Instance of class
                InterferometerArray created at the time of instantiating an
                object of class ApertureSynthesis

    baselines:  [M x 3 Numpy array] The baseline vectors associated with the
                M interferometers in SI units. The coordinate system of these
                vectors is local East, North, Up system

    blxyz       [M x 3 Numpy array] The baseline vectors associated with the
                M interferometers in SI units. The coordinate system of these
                vectors is X, Y, Z in equatorial coordinates

    uvw_lambda  [M x 3 x Nt numpy array] Baseline vectors phased to the phase
                center of each accummulation. M is the number of baselines, Nt
                is the number of accumulations and 3 denotes U, V and W
                components. This is in units of physical distance (usually in m)

    uvw         [M x 3 x Nch x Nt numpy array] Baseline vectors phased to the
                phase center of each accummulation at each frequency. M is the
                number of baselines, Nt is the number of accumulations, Nch is
                the number of frequency channels, and 3 denotes U, V and W
                components. This is uvw_lambda / wavelength and in units of
                number of wavelengths

    blc         [numpy array] 3-element numpy array specifying bottom left
                corner of the grid coincident with bottom left interferometer
                location in UVW coordinate system (same units as uvw)

    trc         [numpy array] 3-element numpy array specifying top right
                corner of the grid coincident with top right interferometer
                location in UVW coordinate system (same units as uvw)

    grid_blc    [numpy array] 3-element numpy array specifying bottom left
                corner of the grid in UVW coordinate system including any
                padding used (same units as uvw)

    grid_trc    [numpy array] 2-element numpy array specifying top right
                corner of the grid in UVW coordinate system including any
                padding used (same units as uvw)

    gridu       [numpy array] 3-dimensional numpy meshgrid array specifying
                grid u-locations in units of uvw in the UVW coordinate system
                whose corners are specified by attributes grid_blc and grid_trc

    gridv       [numpy array] 3-dimensional numpy meshgrid array specifying
                grid v-locations in units of uvw in the UVW coordinate system
                whose corners are specified by attributes grid_blc and grid_trc

    gridw       [numpy array] 3-dimensional numpy meshgrid array specifying
                grid w-locations in units of uvw in the UVW coordinate system
                whose corners are specified by attributes grid_blc and grid_trc

    grid_ready  [boolean] set to True if the gridding has been performed,
                False if grid is not available yet. Set to False in case
                blc, trc, grid_blc or grid_trc is updated indicating gridding
                is to be perfomed again

    f           [numpy vector] frequency channels in Hz

    df          [scalar] Frequency resolution (in Hz)

    latitude    [Scalar] Latitude of the interferometer's location. Default
                is 34.0790 degrees North corresponding to that of the VLA.

    lst         [list] List of LST (in degrees) for each timestamp

    n_acc       [scalar] Number of accumulations

    pointing_center
                [2-column numpy array] Pointing center (latitude and
                longitude) of the observation at a given timestamp. This is
                where the telescopes will be phased up to as reference.
                Coordinate system for the pointing_center is specified by another
                attribute pointing_coords.

    phase_center
                [2-column numpy array] Phase center (latitude and
                longitude) of the observation at a given timestamp. This is
                where the telescopes will be phased up to as reference.
                Coordinate system for the phase_center is specified by another
                attribute phase_center_coords.

    pointing_coords
                [string] Coordinate system for telescope pointing. Accepted
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz'
                (Altitude-Azimuth). Default = 'hadec'.

    phase_center_coords
                [string] Coordinate system for array phase center. Accepted
                values are 'radec' (RA-Dec), 'hadec' (HA-Dec) or 'altaz'
                (Altitude-Azimuth). Default = 'hadec'.

    timestamp   [list] List of timestamps during the observation

    Member functions:

    __init__()      Initialize an instance of class ApertureSynthesis which
                    manages information on a aperture synthesis with an
                    interferometer array.

    genUVW()        Generate U, V, W (in units of number of wavelengths) by
                    phasing the baseline vectors to the phase centers of each
                    pointing at all frequencies

    reorderUVW()    Reorder U, V, W (in units of number of wavelengths) of shape
                    nbl x 3 x nchan x n_acc to 3 x (nbl x nchan x n_acc)

    setUVWgrid()    Set up U, V, W grid (in units of number of wavelengths)
                    based on the synthesized U, V, W
    ----------------------------------------------------------------------------
    """

    def __init__(self, interferometer_array=None):

        """
        ------------------------------------------------------------------------
        Intialize the ApertureSynthesis class which manages information on a
        aperture synthesis with an interferometer array.

        Class attributes initialized are:
        ia, f, df, lst, timestamp, baselines, blxyz, phase_center, n_acc,
        phase_center_coords, pointing_center, pointing_coords, latitude, blc,
        trc, grid_blc, grid_trc, grid_ready, uvw, uvw_lambda, gridu, gridv,
        gridw

        Read docstring of class ApertureSynthesis for details on these
        attributes.

        Keyword input(s):

        interferometer_array
                     [instance of class InterferometerArray] Instance of class
                     InterferometerArray used to initialize an instance of
                     class ApertureSynthesis
        ------------------------------------------------------------------------
        """

        if interferometer_array is not None:
            if isinstance(interferometer_array, InterferometerArray):
                self.ia = interferometer_array
            else:
                raise TypeError('Input interferometer_array must be an instance of class InterferoemterArray')
        else:
            raise NameError('No input interferometer_array provided')

        self.f = self.ia.channels
        self.df = interferometer_array.freq_resolution
        self.n_acc = interferometer_array.n_acc
        self.lst = interferometer_array.lst
        self.phase_center = interferometer_array.phase_center
        self.pointing_center = interferometer_array.pointing_center
        self.phase_center_coords = interferometer_array.phase_center_coords
        self.pointing_coords = interferometer_array.pointing_coords
        self.baselines = interferometer_array.baselines
        self.timestamp = interferometer_array.timestamp
        self.latitude = interferometer_array.latitude
        self.blxyz = GEOM.enu2xyz(self.baselines, self.latitude, units='degrees')
        self.uvw_lambda = None
        self.uvw = None
        self.blc = NP.zeros(2)
        self.trc = NP.zeros(2)
        self.grid_blc = NP.zeros(2)
        self.grid_trc = NP.zeros(2)
        self.gridu, self.gridv, self.gridw = None, None, None
        self.grid_ready = False

    #############################################################################

    def genUVW(self):

        """
        ------------------------------------------------------------------------
        Generate U, V, W (in units of number of wavelengths) by phasing the
        baseline vectors to the phase centers of each pointing at all
        frequencies
        ------------------------------------------------------------------------
        """

        if self.phase_center_coords == 'hadec':
            pc_hadec = self.phase_center
        elif self.phase_center_coords == 'radec':
            pc_hadec = NP.hstack((NP.asarray(self.lst).reshape(-1,1), NP.zeros(len(self.lst)).reshape(-1,1)))
        elif self.phase_center_coords == 'altaz':
            pc_altaz = self.phase_center
            pc_hadec = GEOM.altaz2hadec(pc_altaz, self.latitude, units='degrees')
        else:
            raise ValueError('Attribute phase_center_coords must be set to one of "hadec", "radec" or "altaz"')

        pc_hadec = NP.radians(pc_hadec)
        ha = pc_hadec[:,0]
        dec = pc_hadec[:,1]
        rotmat = NP.asarray([[NP.sin(ha), NP.cos(ha), NP.zeros_like(ha)],
                            [-NP.sin(dec)*NP.cos(ha), NP.sin(dec)*NP.sin(ha), NP.cos(dec)],
                            [NP.cos(dec)*NP.cos(ha), -NP.cos(dec)*NP.sin(ha), NP.sin(dec)]])
        self.uvw_lambda = NP.tensordot(self.blxyz, rotmat, axes=[1,1])
        wl = FCNST.c / self.f
        self.uvw = self.uvw_lambda[:,:,NP.newaxis,:] / wl.reshape(1,1,-1,1)

    #############################################################################

    def reorderUVW(self):

        """
        ------------------------------------------------------------------------
        Reorder U, V, W (in units of number of wavelengths) of shape
        nbl x 3 x nchan x n_acc to 3 x (nbl x nchan x n_acc)
        ------------------------------------------------------------------------
        """

        reorderedUVW = NP.swapaxes(self.uvw, 0, 1) # now 3 x Nbl x nchan x n_acc
        reorderedUVW = reorderedUVW.reshape(3,-1) # now 3 x (Nbl x nchan x n_acc)
        return reorderedUVW

    #############################################################################

    def setUVWgrid(self, spacing=0.5, pad=None, pow2=True):

        """
        ------------------------------------------------------------------------
        Routine to produce a grid based on the UVW spacings of the
        interferometer array

        Inputs:

        spacing     [Scalar] Positive value indicating the upper limit on grid
                    spacing in uvw-coordinates desirable at the lowest
                    wavelength (max frequency). Default = 0.5

        pad         [List] Padding to be applied around the locations
                    before forming a grid. List elements should be positive. If
                    it is a one-element list, the element is applicable to all
                    x, y and z axes. If list contains four or more elements,
                    only the first three elements are considered one for each
                    axis. Default = None (no padding).

        pow2        [Boolean] If set to True, the grid is forced to have a size
                    a next power of 2 relative to the actual size required. If
                    False, gridding is done with the appropriate size as
                    determined by spacing. Default = True.
        ------------------------------------------------------------------------
        """

        if self.uvw is None:
            self.genUVW()

        uvw = self.reorderUVW()
        blc = NP.amin(uvw, axis=1)
        trc = NP.amax(uvw, axis=1)

        self.trc = NP.amax(NP.abs(NP.vstack((blc, trc))), axis=0)
        self.blc = -1 * self.trc

        self.gridu, self.gridv, self.gridw = GRD.grid_3d([(self.blc[0], self.trc[0]), (self.blc[1], self.trc[1]), (self.blc[2], self.trc[2])], pad=pad, spacing=spacing, pow2=True)

        self.grid_blc = NP.asarray([self.gridu.min(), self.gridv.min(), self.gridw.min()])
        self.grid_trc = NP.asarray([self.gridu.max(), self.gridv.max(), self.gridw.max()])
        self.grid_ready = True

################################################################################

class InterferometerData(object):

    """
    ----------------------------------------------------------------------------
    Class to act as an interface between PRISim object and external data
    formats.

    Attributes:

    infodict    [dictionary] Dictionary consisting of many attributes loaded
                from the PRISim object. This will be used to convert to info
                required in external data formats

    Member functions:

    __init__()  Initialize an instance of class InterferometerData

    createUVData()
                Create an instance of class UVData

    write()     Write an instance of class InterferometerData into specified
                formats. Currently writes in UVFITS format
    ----------------------------------------------------------------------------
    """

    def __init__(self, prisim_object, ref_point=None):

        """
        ------------------------------------------------------------------------
        Initialize an instance of class InterferometerData.

        Class attributes initialized are:
        infodict

        Inputs:

        prisim_object
                    [instance of class InterferometerArray] Instance of
                    class InterferometerArray used to initialize an
                    instance of class InterferometerData.

        ref_point   [dictionary] Contains information about the reference
                    position to which projected baselines and rotated
                    visibilities are to be computed. Default=None (no additional
                    phasing will be performed). It must be contain the following
                    keys with the following values:
                    'coords'    [string] Refers to the coordinate system in
                                which value in key 'location' is specified in.
                                Accepted values are 'radec', 'hadec', 'altaz'
                                and 'dircos'
                    'location'  [numpy array] Must be a Mx2 (if value in key
                                'coords' is set to 'radec', 'hadec', 'altaz' or
                                'dircos') or Mx3 (if value in key 'coords' is
                                set to 'dircos'). M can be 1 or equal to number
                                of timestamps. If M=1, the same reference point
                                in the same coordinate system will be repeated
                                for all tiemstamps. If value under key 'coords'
                                is set to 'radec', 'hadec' or 'altaz', the
                                value under this key 'location' must be in
                                units of degrees.
        ------------------------------------------------------------------------
        """

        try:
            prisim_object
        except NameError:
            raise NameError('Input prisim_object not specified')
        if ref_point is not None:
            prisim_object.rotate_visibilities(ref_point)
        if not isinstance(prisim_object, InterferometerArray):
            raise TypeError('Inout prisim_object must be an instance of class InterferometerArray')
        datatypes = ['noiseless', 'noisy', 'noise']
        visibilities = {key: None for key in datatypes}
        for key in visibilities:
            # Conjugate visibilities for compatibility with UVFITS and CASA imager
            if key == 'noiseless':
                visibilities[key] = prisim_object.skyvis_freq.conj()
            if key == 'noisy':
                if prisim_object.vis_freq is not None:
                    visibilities[key] = prisim_object.vis_freq.conj()
            if key == 'noise':
                if prisim_object.vis_noise_freq is not None:
                    visibilities[key] = prisim_object.vis_noise_freq.conj()

        self.infodict = {}
        self.infodict['Ntimes'] = prisim_object.n_acc
        self.infodict['Nbls'] = prisim_object.baselines.shape[0]
        self.infodict['Nblts'] = self.infodict['Nbls'] * self.infodict['Ntimes']
        self.infodict['Nfreqs'] = prisim_object.channels.size
        self.infodict['Npols'] = 1
        self.infodict['Nspws'] = 1
        self.infodict['data_array'] = {'noiseless': None, 'noisy': None, 'noise': None}
        for key in visibilities:
            if visibilities[key] is not None:
                self.infodict['data_array'][key] = NP.transpose(NP.transpose(visibilities[key], (2,0,1)).reshape(self.infodict['Nblts'], self.infodict['Nfreqs'], self.infodict['Nspws'], self.infodict['Npols']), (0,2,1,3)) # (Nbls, Nfreqs, Ntimes) -> (Ntimes, Nbls, Nfreqs) -> (Nblts, Nfreqs, Nspws=1, Npols=1) -> (Nblts, Nspws=1, Nfreqs, Npols=1)
        self.infodict['vis_units'] = 'Jy'
        self.infodict['nsample_array'] = NP.ones((self.infodict['Nblts'], self.infodict['Nspws'], self.infodict['Nfreqs'], self.infodict['Npols']))
        self.infodict['flag_array'] = NP.zeros((self.infodict['Nblts'], self.infodict['Nspws'], self.infodict['Nfreqs'], self.infodict['Npols']), dtype=NP.bool)
        self.infodict['spw_array'] = NP.arange(self.infodict['Nspws'])
        self.infodict['uvw_array'] = NP.transpose(prisim_object.projected_baselines, (2,0,1)).reshape(self.infodict['Nblts'], 3)
        time_array = NP.asarray(prisim_object.timestamp).reshape(-1,1) + NP.zeros(self.infodict['Nbls']).reshape(1,-1)
        self.infodict['time_array'] = time_array.ravel()
        lst_array = NP.radians(NP.asarray(prisim_object.lst).reshape(-1,1)) + NP.zeros(self.infodict['Nbls']).reshape(1,-1)
        self.infodict['lst_array'] = lst_array.ravel()

        labels_A1 = prisim_object.labels['A1']
        labels_A2 = prisim_object.labels['A2']
        if prisim_object.layout:
            id_A1 = [prisim_object.layout['ids'][prisim_object.layout['labels'].tolist().index(albl)] for albl in labels_A1]
            id_A2 = [prisim_object.layout['ids'][prisim_object.layout['labels'].tolist().index(albl)] for albl in labels_A2]
            id_A1 = NP.asarray(id_A1, dtype=int)
            id_A2 = NP.asarray(id_A2, dtype=int)
        else:
            try:
                id_A1 = prisim_object.labels['A1'].astype(NP.int)
                id_A2 = prisim_object.labels['A2'].astype(NP.int)
            except ValueError:
                raise ValueError('Could not convert antenna labels to numbers')
        ant_1_array = id_A1
        ant_2_array = id_A2
        ant_1_array = ant_1_array.reshape(1,-1) + NP.zeros(self.infodict['Ntimes'], dtype=NP.int).reshape(-1,1)
        ant_2_array = ant_2_array.reshape(1,-1) + NP.zeros(self.infodict['Ntimes'], dtype=NP.int).reshape(-1,1)

        self.infodict['ant_1_array'] = ant_1_array.ravel()
        self.infodict['ant_2_array'] = ant_2_array.ravel()
        self.infodict['baseline_array'] = 2048 * (self.infodict['ant_2_array'] + 1) + (self.infodict['ant_1_array'] + 1) + 2**16
        self.infodict['freq_array'] = prisim_object.channels.reshape(self.infodict['Nspws'],-1)
        self.infodict['polarization_array'] = NP.asarray([-5]).reshape(self.infodict['Npols']) # stokes 1:4 (I,Q,U,V); circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)
        self.infodict['integration_time'] = prisim_object.t_acc[0]
        self.infodict['channel_width'] = prisim_object.freq_resolution

        # ----- Observation information ------
        pointing_center = prisim_object.pointing_center
        pointing_coords = prisim_object.pointing_coords
        if pointing_coords == 'dircos':
            pointing_center_dircos = pointing_center
            pointing_center_altaz = GEOM.dircos2altaz(pointing_center_dircos, units='degrees')
            pointing_center_hadec = GEOM.altaz2hadec(pointing_center_altaz, prisim_object.latitude, units='degrees')
            pointing_center_ra = NP.asarray(prisim_object.lst) - pointing_center_hadec[:,0]
            pointing_center_radec = NP.hstack((pointing_center_ra.reshape(-1,1), pointing_center_hadec[:,1].reshape(-1,1)))
            pointing_coords = 'radec'
        elif pointing_coords == 'altaz':
            pointing_center_altaz = pointing_center
            pointing_center_hadec = GEOM.altaz2hadec(pointing_center_altaz, prisim_object.latitude, units='degrees')
            pointing_center_ra = NP.asarray(prisim_object.lst) - pointing_center_hadec[:,0]
            pointing_center_radec = NP.hstack((pointing_center_ra.reshape(-1,1), pointing_center_hadec[:,1].reshape(-1,1)))
            pointing_coords = 'radec'
        elif pointing_coords == 'hadec':
            pointing_center_hadec = pointing_center
            pointing_center_ra = NP.asarray(prisim_object.lst) - pointing_center_hadec[:,0]
            pointing_center_radec = NP.hstack((pointing_center_ra.reshape(-1,1), pointing_center_hadec[:,1].reshape(-1,1)))
            pointing_coords = 'radec'
        elif pointing_coords == 'radec':
            pointing_center_radec = pointing_center
        else:
            raise ValueError('Invalid pointing center coordinates')

        phase_center = prisim_object.phase_center
        phase_center_coords = prisim_object.phase_center_coords
        if phase_center_coords == 'dircos':
            phase_center_dircos = phase_center
            phase_center_altaz = GEOM.dircos2altaz(phase_center_dircos, units='degrees')
            phase_center_hadec = GEOM.altaz2hadec(phase_center_altaz, prisim_object.latitude, units='degrees')
            phase_center_ra = NP.asarray(prisim_object.lst) - phase_center_hadec[:,0]
            phase_center_radec = NP.hstack((phase_center_ra.reshape(-1,1), phase_center_hadec[:,1].reshape(-1,1)))
            phase_center_coords = 'radec'
        elif phase_center_coords == 'altaz':
            phase_center_altaz = phase_center
            phase_center_hadec = GEOM.altaz2hadec(phase_center_altaz, prisim_object.latitude, units='degrees')
            phase_center_ra = NP.asarray(prisim_object.lst) - phase_center_hadec[:,0]
            phase_center_radec = NP.hstack((phase_center_ra.reshape(-1,1), phase_center_hadec[:,1].reshape(-1,1)))
            phase_center_coords = 'radec'
        elif phase_center_coords == 'hadec':
            phase_center_hadec = phase_center
            phase_center_ra = NP.asarray(prisim_object.lst) - phase_center_hadec[:,0]
            phase_center_radec = NP.hstack((phase_center_ra.reshape(-1,1), phase_center_hadec[:,1].reshape(-1,1)))
            phase_center_coords = 'radec'
        elif phase_center_coords == 'radec':
            phase_center_radec = phase_center
        else:
            raise ValueError('Invalid phase center coordinates')

        pointing_centers = SkyCoord(ra=pointing_center_radec[:,0], dec=pointing_center_radec[:,1], frame='icrs', unit='deg')
        phase_centers = SkyCoord(ra=phase_center_radec[:,0], dec=phase_center_radec[:,1], frame='icrs', unit='deg')
        pointing_center_obscenter = pointing_centers[int(prisim_object.n_acc/2)]
        phase_center_obscenter = phase_centers[int(prisim_object.n_acc/2)]

        self.infodict['object_name'] = 'J{0}{1}'.format(pointing_center_obscenter.ra.to_string(sep='', precision=2, pad=True), pointing_center_obscenter.dec.to_string(sep='', precision=2, alwayssign=True, pad=True))
        if 'id' not in prisim_object.telescope:
            self.infodict['telescope_name'] = 'custom'
        else:
            self.infodict['telescope_name'] = prisim_object.telescope['id']
        self.infodict['instrument'] = self.infodict['telescope_name']
        self.infodict['telescope_location'] = NP.asarray([prisim_object.latitude, prisim_object.longitude, prisim_object.altitude])
        self.infodict['history'] = 'PRISim'

        self.infodict['phase_center_epoch'] = 2000.0
        is_phased = NP.allclose(phase_centers.ra.value, phase_centers.ra.value[::-1]) and NP.allclose(phase_centers.dec.value, phase_centers.dec.value[::-1])
        self.infodict['is_phased'] = is_phased

        # ----- antenna information ------
        self.infodict['Nants_data'] = len(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2']))
        if prisim_object.layout:
            # self.infodict['Nants_telescope'] = len(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2']))
            self.infodict['Nants_telescope'] = prisim_object.layout['ids'].size
        else:
            self.infodict['Nants_telescope'] = self.infodict['Nants_data']

        if prisim_object.layout:
            self.infodict['antenna_names'] = prisim_object.layout['labels']
            self.infodict['antenna_numbers'] = prisim_object.layout['ids']
        else:
            self.infodict['antenna_names'] = NP.asarray(list(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2'])))
            try:
                self.infodict['antenna_numbers'] = NP.asarray(list(set(prisim_object.labels['A1']) | set(prisim_object.labels['A2']))).astype(NP.int)
            except ValueError:
                raise ValueError('Count not convert antenna labels to numbers')

        # ----- Optional information ------
        self.infodict['dateobs'] = Time(prisim_object.timestamp[0], format='jd', scale='utc').iso
        self.infodict['phase_center_ra'] = NP.radians(phase_center_obscenter.ra.value)
        self.infodict['phase_center_dec'] = NP.radians(phase_center_obscenter.dec.value)
        self.infodict['antenna_positions'] = NP.zeros((self.infodict['Nants_telescope'],3), dtype=NP.float)
        if hasattr(prisim_object, 'layout'):
            if prisim_object.layout:
                if not isinstance(prisim_object.layout['positions'], NP.ndarray):
                    warnings.warn('Antenna positions must be a numpy array. Proceeding with default values.')
                else:
                    if prisim_object.layout['positions'].shape != (self.infodict['Nants_telescope'],3):
                        warnings.warn('Number of antennas in prisim_object found to be incompatible with number of unique antennas found. Proceeding with default values.')
                    else:
                        self.infodict['antenna_positions'] = prisim_object.layout['positions']

        self.infodict['gst0'] = 0.0
        self.infodict['rdate'] = ''
        self.infodict['earth_omega'] = 360.985
        self.infodict['dut1'] = 0.0
        self.infodict['timesys'] = 'UTC'

    #############################################################################

    def createUVData(self, datatype='noiseless'):

        """
        ------------------------------------------------------------------------
        Create an instance of class UVData.

        Inputs:

        datatype    [string] Specifies which visibilities are to be used in
                    creating the UVData object. Accepted values are 'noiseless'
                    (default) for noiseless pure-sky visibilities, 'noisy' for
                    sky visibilities to which noise has been added, or 'noise'
                    for pure noise visibilities.

        Outputs:

        dataobj     [instance of class UVData] an instance of class UVData
                    containing visibilities of type specified in datatype. This
                    object can be used to write to some common external formats
                    such as UVFITS, etc.
        ------------------------------------------------------------------------
        """

        if not uvdata_module_found:
            raise ImportError('uvdata module not found')

        if datatype not in ['noiseless', 'noisy', 'noise']:
            raise ValueError('Invalid input datatype specified')

        attributes_of_uvdata = ['Ntimes', 'Nbls', 'Nblts', 'Nfreqs', 'Npols', 'Nspws', 'data_array', 'vis_units', 'nsample_array', 'flag_array', 'spw_array', 'uvw_array', 'time_array', 'lst_array', 'ant_1_array', 'ant_2_array', 'baseline_array', 'freq_array', 'polarization_array', 'integration_time', 'channel_width', 'object_name', 'telescope_name', 'instrument', 'telescope_location', 'history', 'phase_center_epoch', 'is_phased', 'phase_type', 'Nants_data', 'Nants_telescope', 'antenna_names', 'antenna_numbers', 'dateobs', 'phase_center_ra', 'phase_center_dec']
        dataobj = UVData()
        for attrkey in attributes_of_uvdata:
            if attrkey == 'telescope_location':
                x, y, z = GEOM.lla2ecef(*self.infodict[attrkey], units='degrees')
                setattr(dataobj, attrkey, NP.asarray([x[0],y[0],z[0]]))
            elif attrkey == 'phase_type':
                if self.infodict['is_phased']:
                    setattr(dataobj, attrkey, 'phased')
                else:
                    setattr(dataobj, attrkey, 'drift')
            elif attrkey != 'data_array':
                setattr(dataobj, attrkey, self.infodict[attrkey])
            else:
                if datatype in self.infodict[attrkey]:
                    if self.infodict[attrkey][datatype] is not None:
                        setattr(dataobj, attrkey, self.infodict[attrkey][datatype])
                    else:
                        raise KeyError('Data of specified datatype not found in InterferometerData object')
                else:
                    raise KeyError('Specified datatype not found in InterferometerData object')

        return dataobj

    #############################################################################

    def _blnum_to_antnums(self, blnum):
        if self.infodict['Nants_telescope'] > 2048:
            raise StandardError('error Nants={Nants}>2048 not supported'.format(Nants=self.infodict['Nants_telescope']))
        if NP.min(blnum) > 2**16:
            i = (blnum - 2**16) % 2048 - 1
            j = (blnum - 2**16 - (i + 1)) / 2048 - 1
        else:
            i = (blnum) % 256 - 1
            j = (blnum - (i + 1)) / 256 - 1
        return NP.int32(i), NP.int32(j)

    #############################################################################

    def _antnums_to_blnum(self, i, j, attempt256=False):
        # set the attempt256 keyword to True to (try to) use the older
        # 256 standard used in many uvfits files
        # (will use 2048 standard if there are more than 256 antennas)
        i, j = NP.int64((i, j))
        if self.infodict['Nants_telescope'] > 2048:
            raise StandardError('cannot convert i,j to a baseline index '
                                'with Nants={Nants}>2048.'
                                .format(Nants=self.infodict['Nants_telescope']))
        if attempt256:
            if (NP.max(i) < 255 and NP.max(j) < 255):
                return 256 * (j + 1) + (i + 1)
            else:
                print('Max antnums are {} and {}'.format(NP.max(i), NP.max(j)))
                message = 'antnums_to_baseline: found > 256 antennas, using ' \
                          '2048 baseline indexing. Beware compatibility ' \
                          'with CASA etc'
                warnings.warn(message)

        return NP.int64(2048 * (j + 1) + (i + 1) + 2**16)

    #############################################################################

    def write(self, outfile, datatype='noiseless', fmt='UVFITS',
              uvfits_method=None, overwrite=False):

        """
        ------------------------------------------------------------------------
        Write an instance of class InterferometerData into specified formats.
        Currently writes in UVFITS format

        Inputs:

        outfile     [string] Filename into which data will be written

        datatype    [string] Specifies which visibilities are to be used in
                    creating the UVData object. Accepted values are 'noiseless'
                    (default) for noiseless pure-sky visibilities, 'noisy' for
                    sky visibilities to which noise has been added, or 'noise'
                    for pure noise visibilities.

        fmt         [string] Output file format. Currently accepted values are
                    'UVFITS'

        uvfits_method
                    [string] Method using which UVFITS output is produced.
                    Accepted values are 'uvdata', 'uvfits' or None (default).
                    If set to 'uvdata', the UVFITS writer in uvdata module is
                    used. If set to 'uvfits', the in-house UVFITS writer is
                    used. If set to None, first uvdata module will be attempted
                    but if it fails then the in-house UVFITS writer will be
                    tried.

        overwrite   [boolean] True indicates overwrite even if a file already
                    exists. Default = False (does not overwrite). Beware this
                    may not work reliably if uvfits_method is set to None or
                    'uvdata' and hence always better to make sure the output
                    file does not exist already
        ------------------------------------------------------------------------
        """

        try:
            outfile
        except NameError:
            raise NameError('Output filename not specified')

        if not isinstance(outfile, str):
            raise TypeError('Output filename must be a string')

        if datatype not in ['noiseless', 'noisy', 'noise']:
            raise ValueError('Invalid input datatype specified')

        if fmt.lower() not in ['uvfits']:
            raise ValueError('Output format not supported')

        if fmt.lower() == 'uvfits':
            write_successful = False
            if uvfits_method not in [None, 'uvfits', 'uvdata']:
                uvfits_method = None
            if (uvfits_method is None) or (uvfits_method == 'uvdata'):
                try:
                    uvdataobj = self.createUVData(datatype=datatype)
                    uvdataobj.write_uvfits(outfile, spoof_nonessential=True)
                except Exception as xption1:
                    write_successful = False
                    if uvfits_method == 'uvdata':
                        warnings.warn('Output through UVData module did not work due to the following exception:')
                        raise xption1
                    else:
                        warnings.warn('Output through UVData module did not work. Trying with built-in UVFITS writer')
                else:
                    write_successful = True
                    print 'Data successfully written using uvdata module to {0}'.format(outfile)
                    return

            # Try with in-house UVFITS writer
            try:
                weights_array = self.infodict['nsample_array'] * NP.where(self.infodict['flag_array'], -1, 1)
                data_array = self.infodict['data_array'][datatype][:, NP.newaxis, NP.newaxis, :, :, :, NP.newaxis]
                weights_array = weights_array[:, NP.newaxis, NP.newaxis, :, :, :, NP.newaxis]
                # uvfits_array_data shape will be  (Nblts,1,1,[Nspws],Nfreqs,Npols,3)
                uvfits_array_data = NP.concatenate([data_array.real, data_array.imag, weights_array], axis=6)

                uvw_array_sec = self.infodict['uvw_array'] / FCNST.c
                # jd_midnight = NP.floor(self.infodict['time_array'][0] - 0.5) + 0.5
                tzero = NP.float32(self.infodict['time_array'][0])

                # uvfits convention is that time_array + relevant PZERO = actual JD
                # We are setting PZERO4 = float32(first time of observation)
                time_array = NP.float32(self.infodict['time_array'] - NP.float64(tzero))

                int_time_array = (NP.zeros_like((time_array), dtype=NP.float) + self.infodict['integration_time'])
                baselines_use = self._antnums_to_blnum(self.infodict['ant_1_array'], self.infodict['ant_2_array'], attempt256=True)
                # Set up dictionaries for populating hdu
                # Note that uvfits antenna arrays are 1-indexed so we add 1
                # to our 0-indexed arrays
                group_parameter_dict = {'UU      ': uvw_array_sec[:, 0],
                                        'VV      ': uvw_array_sec[:, 1],
                                        'WW      ': uvw_array_sec[:, 2],
                                        'DATE    ': time_array,
                                        'BASELINE': baselines_use,
                                        'ANTENNA1': self.infodict['ant_1_array'] + 1,
                                        'ANTENNA2': self.infodict['ant_2_array'] + 1,
                                        'SUBARRAY': NP.ones_like(self.infodict['ant_1_array']),
                                        'INTTIM': int_time_array}
                pscal_dict = {'UU      ': 1.0, 'VV      ': 1.0, 'WW      ': 1.0,
                              'DATE    ': 1.0, 'BASELINE': 1.0, 'ANTENNA1': 1.0,
                              'ANTENNA2': 1.0, 'SUBARRAY': 1.0, 'INTTIM': 1.0}
                pzero_dict = {'UU      ': 0.0, 'VV      ': 0.0, 'WW      ': 0.0,
                              'DATE    ': tzero, 'BASELINE': 0.0, 'ANTENNA1': 0.0,
                              'ANTENNA2': 0.0, 'SUBARRAY': 0.0, 'INTTIM': 0.0}

                # list contains arrays of [u,v,w,date,baseline];
                # each array has shape (Nblts)
                if (NP.max(self.infodict['ant_1_array']) < 255 and
                        NP.max(self.infodict['ant_2_array']) < 255):
                    # if the number of antennas is less than 256 then include both the
                    # baseline array and the antenna arrays in the group parameters.
                    # Otherwise just use the antenna arrays
                    parnames_use = ['UU      ', 'VV      ', 'WW      ',
                                    'DATE    ', 'BASELINE', 'ANTENNA1',
                                    'ANTENNA2', 'SUBARRAY', 'INTTIM']
                else:
                    parnames_use = ['UU      ', 'VV      ', 'WW      ', 'DATE    ',
                                    'ANTENNA1', 'ANTENNA2', 'SUBARRAY', 'INTTIM']

                group_parameter_list = [group_parameter_dict[parname] for
                                        parname in parnames_use]
                hdu = fits.GroupData(uvfits_array_data, parnames=parnames_use,
                                     pardata=group_parameter_list, bitpix=-32)
                hdu = fits.GroupsHDU(hdu)

                for i, key in enumerate(parnames_use):
                    hdu.header['PSCAL' + str(i + 1) + '  '] = pscal_dict[key]
                    hdu.header['PZERO' + str(i + 1) + '  '] = pzero_dict[key]

                # ISO string of first time in self.infodict['time_array']

                # hdu.header['DATE-OBS'] = Time(self.infodict['time_array'][0], scale='utc', format='jd').iso
                hdu.header['DATE-OBS'] = self.infodict['dateobs']

                hdu.header['CTYPE2  '] = 'COMPLEX '
                hdu.header['CRVAL2  '] = 1.0
                hdu.header['CRPIX2  '] = 1.0
                hdu.header['CDELT2  '] = 1.0

                hdu.header['CTYPE3  '] = 'STOKES  '
                hdu.header['CRVAL3  '] = self.infodict['polarization_array'][0]
                hdu.header['CRPIX3  '] = 1.0
                try:
                    hdu.header['CDELT3  '] = NP.diff(self.infodict['polarization_array'])[0]
                except(IndexError):
                    hdu.header['CDELT3  '] = 1.0

                hdu.header['CTYPE4  '] = 'FREQ    '
                hdu.header['CRVAL4  '] = self.infodict['freq_array'][0, 0]
                hdu.header['CRPIX4  '] = 1.0
                hdu.header['CDELT4  '] = NP.diff(self.infodict['freq_array'][0])[0]

                hdu.header['CTYPE5  '] = 'IF      '
                hdu.header['CRVAL5  '] = 1.0
                hdu.header['CRPIX5  '] = 1.0
                hdu.header['CDELT5  '] = 1.0

                hdu.header['CTYPE6  '] = 'RA'
                hdu.header['CRVAL6  '] = NP.degrees(self.infodict['phase_center_ra'])

                hdu.header['CTYPE7  '] = 'DEC'
                hdu.header['CRVAL7  '] = NP.degrees(self.infodict['phase_center_dec'])

                hdu.header['BUNIT   '] = self.infodict['vis_units']
                hdu.header['BSCALE  '] = 1.0
                hdu.header['BZERO   '] = 0.0

                hdu.header['OBJECT  '] = self.infodict['object_name']
                hdu.header['TELESCOP'] = self.infodict['telescope_name']
                hdu.header['LAT     '] = self.infodict['telescope_location'][0]
                hdu.header['LON     '] = self.infodict['telescope_location'][1]
                hdu.header['ALT     '] = self.infodict['telescope_location'][2]
                hdu.header['INSTRUME'] = self.infodict['instrument']
                hdu.header['EPOCH   '] = float(self.infodict['phase_center_epoch'])

                for line in self.infodict['history'].splitlines():
                    hdu.header.add_history(line)

                # ADD the ANTENNA table
                staxof = NP.zeros(self.infodict['Nants_telescope'])

                # 0 specifies alt-az, 6 would specify a phased array
                mntsta = NP.zeros(self.infodict['Nants_telescope'])

                # beware, X can mean just about anything
                poltya = NP.full((self.infodict['Nants_telescope']), 'X', dtype=NP.object_)
                polaa = [90.0] + NP.zeros(self.infodict['Nants_telescope'])
                poltyb = NP.full((self.infodict['Nants_telescope']), 'Y', dtype=NP.object_)
                polab = [0.0] + NP.zeros(self.infodict['Nants_telescope'])

                col1 = fits.Column(name='ANNAME', format='8A',
                                   array=self.infodict['antenna_names'])
                col2 = fits.Column(name='STABXYZ', format='3D',
                                   array=self.infodict['antenna_positions'])
                # convert to 1-indexed from 0-indexed indicies
                col3 = fits.Column(name='NOSTA', format='1J',
                                   array=self.infodict['antenna_numbers'] + 1)
                col4 = fits.Column(name='MNTSTA', format='1J', array=mntsta)
                col5 = fits.Column(name='STAXOF', format='1E', array=staxof)
                col6 = fits.Column(name='POLTYA', format='1A', array=poltya)
                col7 = fits.Column(name='POLAA', format='1E', array=polaa)
                # col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
                col9 = fits.Column(name='POLTYB', format='1A', array=poltyb)
                col10 = fits.Column(name='POLAB', format='1E', array=polab)
                # col11 = fits.Column(name='POLCALB', format='3E', array=polcalb)
                # note ORBPARM is technically required, but we didn't put it in

                cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col9, col10])
                ant_hdu = fits.BinTableHDU.from_columns(cols)
                ant_hdu.header['EXTNAME'] = 'AIPS AN'
                ant_hdu.header['EXTVER'] = 1

                # write XYZ coordinates if not already defined
                ant_hdu.header['ARRAYX'] = self.infodict['telescope_location'][0]
                ant_hdu.header['ARRAYY'] = self.infodict['telescope_location'][1]
                ant_hdu.header['ARRAYZ'] = self.infodict['telescope_location'][2]
                # ant_hdu.header['FRAME'] = 'ITRF'
                ant_hdu.header['FRAME'] = None
                ant_hdu.header['GSTIA0'] = self.infodict['gst0']
                ant_hdu.header['FREQ'] = self.infodict['freq_array'][0, 0]
                ant_hdu.header['RDATE'] = self.infodict['rdate']
                ant_hdu.header['UT1UTC'] = self.infodict['dut1']

                ant_hdu.header['TIMSYS'] = self.infodict['timesys']
                if self.infodict['timesys'] == 'IAT':
                    warnings.warn('This file has an "IAT" time system. Files of '
                                  'this type are not properly supported')
                ant_hdu.header['ARRNAM'] = self.infodict['telescope_name']
                ant_hdu.header['NO_IF'] = self.infodict['Nspws']
                ant_hdu.header['DEGPDY'] = self.infodict['earth_omega']
                # ant_hdu.header['IATUTC'] = 35.

                # set mandatory parameters which are not supported by this object
                # (or that we just don't understand)
                ant_hdu.header['NUMORB'] = 0

                # note: Bart had this set to 3. We've set it 0 after aips 117. -jph
                ant_hdu.header['NOPCAL'] = 0

                ant_hdu.header['POLTYPE'] = 'X-Y LIN'

                # note: we do not support the concept of "frequency setups"
                # -- lists of spws given in a SU table.
                ant_hdu.header['FREQID'] = -1

                # if there are offsets in images, this could be the culprit
                ant_hdu.header['POLARX'] = 0.0
                ant_hdu.header['POLARY'] = 0.0

                ant_hdu.header['DATUTC'] = 0  # ONLY UTC SUPPORTED

                # we always output right handed coordinates
                ant_hdu.header['XYZHAND'] = 'RIGHT'

                # ADD the FQ table
                # skipping for now and limiting to a single spw

                # write the file
                hdulist = fits.HDUList(hdus=[hdu, ant_hdu])
                hdulist.writeto(outfile, clobber=overwrite)
            except Exception as xption2:
                print xption2
                raise IOError('Could not write to UVFITS file')
            else:
                write_successful = True
                print 'Data successfully written using in-house uvfits writer to {0}'.format(outfile)
                return

#################################################################################
