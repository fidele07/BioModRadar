import os
import numpy as np
import pandas as pd
from ..radar import (get_scans_list_bird_insect, read_radar_data)
from .mask_gates import (mask_gates_outside_ranges,
                         mask_non_biological)
from .depolarization import depolarization_ratio
from .texture import (compute_ndimage_median,
                      compute_texture_ndimage_stdev)
from .smoothing import smoothing_sweeps_ndimage_generic_filter
from .features_create import create_feature_table_bboxs

def build_features_table_train(dir_radar, file_format,
                               volume_type, scans_list_file,
                               sweeps, fields_dict,
                               dir_format=None,
                               spatial_stat_fields=True,
                               texture_fields=True):
    """Create a DataFrame of a radar fields.

    Create a pandas data frame from the specified radar fields
    to be used to train the models.

    Parameters
    ----------
    dir_radar: str
        Full path to the directory contacting the radar data.
    file_format: string
        The date format of the scan file name, in POSIX date and time format.
        Example: ``"KHTX%Y%m%d_%H%M%S_V06.gz"``
    volume_type: str
        The type of the radar data to read. Valid values are: ``'cfradial'``, 
        ``'mdv'``, ``'odim-h5'``, ``'nexrad-archive'`` and ``'rwanda-odim-h5'``.
    scans_list_file: str
        Full path to the csv (Comma Separated Values) file containing 
        the list of scans to be read. Each line is formed by the date of the scan, 
        it must be the date in the scan file name (format: yyyy-mm-dd hh:mm:ss), 
        the bounding boxes of the areas labeled as insect (start with ``n;``), 
        then the areas labeled as bird (start with ``b;``),
        the bounding box in the format ``west,south/east,north``. If there are many
        bounding boxes, they can be separated by a semicolon. For a scan there is no
        insect or bird observed, it can be left empty by just putting an empty quotes. 
        Example::
        
            date,insect,bird
            2015-08-11 10:30:15,"","b;-118,20/-74,78;-75,64/-36,109"
            2015-08-11 10:34:42,"","b;-135,14/-68,84"
            2015-08-11 11:55:01,"n;-49,-50/66,49",""
            2015-08-11 11:59:26,"n;-68,-47/69,25","b;-122,42/-7,186;-130,-89/-58,9"

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
                .........
                'kdp': '<Specific differential phase>',
                'ncp': <>, 
            }

    dir_format: str, optional
        The date format of the directory contacting the scans, in POSIX date and time format,
        if the scans are stored by date.
        Example:

            /20201110/120403.mdv ->  dir_format="%Y%m%d", file_format="%H%M%S.mdv"
            /2021/08/11/KHTX20150811_000555_V06 -> dir_format="%Y/%m/%d" file_format="KHTX%Y%m%d_%H%M%S_V06"

    spatial_stat_fields: array-like or bool, optional
        List of fields a spatial aggregation (median smoothing) with a moving window will be performed.
        If ``True`` all the fields in scans will be spatially smoothed, ``False`` no smoothing performed.
    texture_fields: array-like or bool, optional
        List of fields the texture will be computed.
        If ``True`` the texture of all the fields in scans will be computed, ``False`` no texture computed.

    Returns
    -------
    df : pd.DataFrame
        Pandas data frame.

    """
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

    scans_list = get_scans_list_bird_insect(scans_list_file)

    df_data = []
    for d in scans_list:
        file = d['date'].strftime(file_format)
        if dir_format is not None:
            dirf = d['date'].strftime(dir_format)
            file_path = os.path.join(dir_radar, dirf, file)
        else:
            file_path = os.path.join(dir_radar, file)

        print(file_path)

        radar = read_radar_data(file_path, sweeps, volume_type, fields_dict)
        if radar is None: continue

        # the returned radar contains only the extracted sweeps
        # (re-indexed from 0)
        sweeps_r = np.arange(radar.nsweeps)

        radar = mask_gates_outside_ranges(radar, 1000., 300000.)
        radar = depolarization_ratio(radar, sweeps_r,
                                     fields_dict['zdr'],
                                     fields_dict['rho'],
                                     fields_dict['ref'],
                                     dr_thres=-12,
                                     rho_thres=0.85,
                                     ref_thres=30,
                                     use_blob_mask=True,
                                     blob_min_size=50,
                                     blob_connectivity=2,
                                     despeckle_class=True,
                                     despeckle_size=25)
        radar = mask_non_biological(radar, fields=field_list)

        field_med = []
        if smooth_median:
            radar = smoothing_sweeps_ndimage_generic_filter(radar, 
                                                            compute_ndimage_median,
                                                            pad=2,
                                                            min_gates=10, 
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

        fields = field_list + field_med + field_tex1 + field_tex2
        insect_data = create_feature_table_bboxs(radar, sweeps_r, fields, d['insect'])
        insect_data['label'] = 0
        bird_data = create_feature_table_bboxs(radar, sweeps_r, fields, d['bird'])
        bird_data['label'] = 1
        scan_data = pd.concat([bird_data, insect_data])
        df_data += [scan_data]

    df_data = pd.concat(df_data, axis=0)
    df_data = df_data.reset_index(drop=True)

    return df_data
