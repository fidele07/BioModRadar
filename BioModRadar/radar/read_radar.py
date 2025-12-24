import numpy as np
from scipy.interpolate import NearestNDInterpolator
import pyart

def read_radar_data(file_path, sweeps=None,
                    volume_type='cfradial',
                    fields_dict=None):
    # sweeps = [0, 1, 2, 3, 4, 5]
    # fields_dict = {'ref': 'REF', 'zdr': 'ZDR', 'rho': 'RHO',
    #                'phi': 'PHI', 'vel': 'VEL', 'sw': 'SW'}
    try:
        if volume_type == 'cfradial':
            radar = pyart.io.read_cfradial(file_path,
                                           file_field_names=True,
                                           delay_field_loading=True)
        elif volume_type == 'mdv':
            radar = pyart.io.read_mdv(file_path,
                                      file_field_names=True,
                                      delay_field_loading=True)
        elif volume_type == 'odim-h5':
            radar = pyart.aux_io.read_odim_h5(file_path,
                                              file_field_names=True,
                                              delay_field_loading=True)
        elif volume_type == 'rwanda-odim-h5':
            radar = pyart.aux_io.read_odim_h5(file_path,
                                              file_field_names=True,
                                              delay_field_loading=True)
            radar = _reduce_sweeps_rwanda_odim_hdf5(radar)
        elif volume_type == 'nexrad-archive':
            radar = pyart.io.read_nexrad_archive(file_path,
                                                 file_field_names=True,
                                                 delay_field_loading=True)
            radar = _reduce_sweeps_nexrad_level2(radar)
        else:
            raise TypeError(f'Unknown volume_type {volume_type}')
    except Exception as e:
        print(e)
        print(f'Enable to read: {file_path}')
        return None

    if sweeps is not None:
        if not isinstance(sweeps, (list, np.ndarray)):
            sweeps = [sweeps]
        if len(sweeps) > radar.nsweeps:
            sweeps = np.arange(radar.nsweeps)
    else:
        sweeps = np.arange(radar.nsweeps)

    radar = radar.extract_sweeps(sweeps)
    radar_fields = list(radar.fields)
    if fields_dict is not None:
        include_fields = [fields_dict[f] for f in fields_dict]
        include_fields = [f for f in include_fields if f is not None]
        if len(include_fields) > 0:
            for fl in include_fields:
                if fl not in radar_fields:
                    raise KeyError(f'No field named "{fl}" found.')
            delete_fields = [fl for fl in radar_fields if fl not in include_fields]
            if len(delete_fields) > 0:
                for fl in delete_fields:
                    del radar.fields[fl]
    return radar

def _reduce_sweeps_rwanda_odim_hdf5(radar):
    keep = np.arange(radar.nsweeps - 3)
    radar = radar.extract_sweeps(sweeps=keep)
    fields = list(radar.fields.keys())
    radar = pyart.util.subset_radar(
        radar,
        field_names=fields,
        ele_min=0.5,
        ele_max=32.
    )
    sweep_number = radar.sweep_number['data']
    sweep_number = np.sort(sweep_number)
    radar.sweep_number['data'] = sweep_number
    return radar

def _reduce_sweeps_nexrad_level2(radar):
    sweep_start = radar.sweep_start_ray_index['data']
    sweep_end = radar.sweep_end_ray_index['data']
    fixed_angle = radar.fixed_angle['data']
    fixed_angle1 = np.unique(fixed_angle)
    kp_swp = []
    rm_swp = []
    for s in fixed_angle1:
        dup = fixed_angle == s
        if sum(dup) > 1:
            ix = np.where(dup)
            kp_swp += [ix[0][0]]
            rm_swp += [ix[0][1]]
    # kp_swp = [0, 2]
    # rm_swp = [1, 3]

    keep = []
    remove = []
    for j in range(len(kp_swp)):
       i = kp_swp[j]
       l = rm_swp[j]
       keep += [np.arange(sweep_start[i], sweep_end[i] + 1)]
       remove += [np.arange(sweep_start[l], sweep_end[l] + 1)]
    repl = [keep, remove]

    for j in range(len(repl[0])):
        iold = repl[0][j]
        inew = repl[1][j]
        # for field in ['REF', 'VEL', 'SW']:
        for field in ['VEL', 'SW']:
            field_data = _interp_sweeps_nexrad_level2(
                    radar.range['data'],
                    radar.azimuth['data'][iold],
                    radar.azimuth['data'][inew],
                    radar.fields[field]['data'][inew, :]
                )
            radar.fields[field]['data'][iold, :] = field_data
        radar.instrument_parameters['unambiguous_range']['data'][iold] = radar.instrument_parameters['unambiguous_range']['data'][inew]
        radar.instrument_parameters['nyquist_velocity']['data'][iold] = radar.instrument_parameters['nyquist_velocity']['data'][inew]

    radar_fields = list(radar.fields)
    rm_ix = np.array(repl[1]).ravel()

    for field in radar_fields:
        data = radar.fields[field]['data'].copy()
        data = data.filled(fill_value=np.nan)
        data = np.delete(data, rm_ix, axis=0)
        data = np.ma.masked_where(np.isnan(data), data)
        radar.fields[field]['data'] = data

    radar.azimuth['data'] = np.delete(radar.azimuth['data'], rm_ix, axis=0)
    radar.elevation['data'] = np.delete(radar.elevation['data'], rm_ix, axis=0)
    radar.instrument_parameters['unambiguous_range']['data'] = np.delete(radar.instrument_parameters['unambiguous_range']['data'], rm_ix, axis=0)
    radar.instrument_parameters['nyquist_velocity']['data'] = np.delete(radar.instrument_parameters['nyquist_velocity']['data'], rm_ix, axis=0)
    radar.time['data'] = np.delete(radar.time['data'], rm_ix, axis=0)

    radar.nrays = radar.nrays - len(np.array(repl[0]).ravel())
    radar.nsweeps = radar.nsweeps - len(repl[0])
    radar.sweep_number['data'] = np.arange(radar.nsweeps)

    radar.fixed_angle['data'] = np.delete(radar.fixed_angle['data'], rm_swp, axis=0)
    radar.sweep_mode['data'] = np.delete(radar.sweep_mode['data'], rm_swp, axis=0)

    sweep_number = sweep_end - sweep_start + 1
    sweep_number = np.delete(sweep_number, rm_swp, axis=0)
    sweep_start = np.cumsum(sweep_number, axis=0)
    sweep_start = np.insert(sweep_start, 0, 0)
    sweep_start = sweep_start[:-1]
    sweep_end = sweep_start + sweep_number - 1

    radar.sweep_start_ray_index['data'] = sweep_start
    radar.sweep_end_ray_index['data'] = sweep_end

    return radar

def _interp_sweeps_nexrad_level2(
                                range_grid,
                                azimuth_grid,
                                azimuth_new,
                                field_data_new
                            ):
    rg, az = np.meshgrid(range_grid, azimuth_new)
    az_flattened = az.flatten()
    rg_flattened = rg.flatten()
    fd = field_data_new.filled(fill_value=np.nan)
    fd_flattened = fd.flatten()

    f_interp = NearestNDInterpolator(
        (az_flattened, rg_flattened),
        fd_flattened,
        rescale=True
    )

    rg_g, az_g = np.meshgrid(range_grid, azimuth_grid)
    az_grid = az_g.flatten()
    rg_grid = rg_g.flatten()
    fd_new = f_interp(az_grid, rg_grid)
    fd_new = fd_new.reshape(rg_g.shape)
    fd_new = np.ma.masked_invalid(fd_new)
    fd_new = fd_new.astype(field_data_new.dtype)

    return fd_new
