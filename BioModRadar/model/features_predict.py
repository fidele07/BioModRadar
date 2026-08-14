import os
import numpy as np
from ..radar import read_radar_data
from .mask_gates import (mask_gates_outside_ranges,
                         mask_non_biological)
from .depolarization import depolarization_ratio
from .texture import (compute_ndimage_median,
                      compute_texture_ndimage_stdev)
from .smoothing import smoothing_sweeps_ndimage_generic_filter

def build_features_predict(file_path, volume_type,
                           sweeps, fields_dict,
                           spatial_stat_fields=True,
                           texture_fields=True,
                           dr_thres=-12,
                           rho_thres=0.95,
                           ref_thres=35):
    """Create a DataFrame of a radar fields.

    Create a pandas data frame from the specified radar fields
    to be used to train the models.

    Parameters
    ----------
    dir_radar: str
        Full path to the file contacting the radar data.
    volume_type: str
        The type of the radar data to read. Valid values are: ``'cfradial'``, 
        ``'mdv'``, ``'odim-h5'``, ``'nexrad-archive'`` and ``'rwanda-odim-h5'``.
    sweeps: array-like or list of integers
        Sweeps (0-based) to extract.
    fields_dict: dict
    Dictionary mapping the fields to read.
    Format:

        {
            'ref': '<field name for reflectivity>',
            'zdr': '<field name for differential reflectivity>',
            'rho': '<field name for correlation coefficient>', 
            'phi': '<field name for differential phase>', 
            'vel': '<field name for radial velocity>', 
            'sw': '<field name for spectrum width>',
            ......
        }

    spatial_stat_fields: array-like or bool, optional
        List of fields a spatial aggregation (median smoothing) with a moving window will be performed.
        If ``True`` all the fields in scans will be spatially smoothed, ``False`` no smoothing performed.
    texture_fields: array-like or bool, optional
        List of fields the texture will be computed.
        If ``True`` the texture of all the fields in scans will be computed, ``False`` no texture computed.
    dr_thres: float, optional
        Threshold of the depolarization ratio to separate biological of non-biological.
        Above this value the gate is considered as biological.
    rho_thres: float, optional
        Threshold for the correlation coefficient to exclude precipitation.
        Above this value the gate is considered as precipitation.
    ref_thres: float, optional
        Threshold for the reflectivity to exclude precipitation.
        Above this value the gate is considered as precipitation.

    Returns
    -------
    radar : Radar
        Radar object which contains all the fields from ``fields_dict`` and the computed fields.
    fields: list
        List of all fields in the radar object.

    """
    if not os.path.exists(file_path):
       raise FileNotFoundError(f'File containing the bird statistics not found.')

    if not isinstance(sweeps, (list, np.ndarray)):
        raise TypeError('sweeps must be a list or np.ndarray.')

    if fields_dict is not None:
        if not isinstance(fields_dict, dict):
            raise TypeError('sweeps must be a dictionary.')
        else:
            fields_r = ['ref', 'zdr', 'rho', 'phi', 'vel', 'sw',
                        'kdp', 'snr', 'ncp', 'cmd']
            fields_keys = list(fields_dict.keys())
            res = [f for f in fields_keys if f not in fields_r]
            if len(res) > 0:
                raise KeyError(f'Unknown dictionary keys: {", ".join(res)}')

    field_list = [fields_dict[f] for f in fields_dict]

    smooth_median = False
    if isinstance(spatial_stat_fields, bool):
        if spatial_stat_fields:
            field_stat = field_list
            smooth_median = True
    elif isinstance(spatial_stat_fields, (list, np.ndarray)):
        res = [f for f in spatial_stat_fields if f not in field_list]
        if len(res) > 0:
            raise ValueError(f'Unknown "spatial_stat_fields": {", ".join(res)}')
        field_stat = list(spatial_stat_fields)
        smooth_median = True
    else:
        raise TypeError('Unknown "spatial_stat_fields" type')

    compute_texture = False
    if isinstance(texture_fields, bool):
        if texture_fields:
            field_tex = field_list
            compute_texture = True
    elif isinstance(texture_fields, (list, np.ndarray)):
        res = [f for f in texture_fields if f not in field_list]
        if len(res) > 0:
            raise ValueError(f'Unknown "texture_fields": {", ".join(res)}')
        field_tex = list(texture_fields)
        compute_texture = True
    else:
        raise TypeError('Unknown "texture_fields" type')

    radar = read_radar_data(file_path, sweeps, volume_type, fields_dict)
    if radar is None:
        raise Exception(f'Enable to read: {file_path}')

    # the returned radar contains only the extracted sweeps (re-indexed
    # from 0); scans with fewer sweeps than requested would otherwise
    # crash downstream with "invalid sweeps indices"
    sweeps = np.arange(radar.nsweeps)

    radar = mask_gates_outside_ranges(radar, 1000., 300000.)
    radar = depolarization_ratio(radar, sweeps,
                                 fields_dict['zdr'],
                                 fields_dict['rho'],
                                 fields_dict['ref'],
                                 dr_thres=dr_thres,
                                 rho_thres=rho_thres,
                                 ref_thres=ref_thres,
                                 use_blob_mask=True,
                                 blob_min_size=25,
                                 blob_connectivity=2,
                                 despeckle_class=True,
                                 despeckle_size=9)
    radar = mask_non_biological(radar, fields=field_list)

    field_med = []
    if smooth_median:
        radar = smoothing_sweeps_ndimage_generic_filter(radar, 
                                                        compute_ndimage_median,
                                                        pad=1,
                                                        min_gates=6, 
                                                        fields=field_stat,
                                                        fields_name_suffix='MED')
        field_med = [f'{k}_MED' for k in field_stat]
    field_tex1 = []
    if compute_texture:
        radar = smoothing_sweeps_ndimage_generic_filter(radar,
                                                        compute_texture_ndimage_stdev,
                                                        pad=1,
                                                        min_gates=6,
                                                        fields=field_tex,
                                                        fields_name_suffix='TEX')
        field_tex1 = [f'{k}_TEX' for k in field_tex]
    field_tex2 = []
    if compute_texture and smooth_median:
        radar = smoothing_sweeps_ndimage_generic_filter(radar,
                                                        compute_texture_ndimage_stdev,
                                                        pad=1,
                                                        min_gates=6,
                                                        fields=field_med,
                                                        fields_name_suffix='TEX')
        field_tex2 = [f'{k}_TEX' for k in field_med]
    fields = field_list + field_med + field_tex1 + field_tex2 + ['DR', 'DR_CLASS']

    return radar, fields
