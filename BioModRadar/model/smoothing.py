import warnings
import numpy as np
from scipy import ndimage as ndi
import pyart

from .texture import (compute_ndimage_median, compute_ndimage_mean,
                      compute_texture_ndimage_stdev)

# Window callbacks with a vectorized NaN-aware equivalent. The per-gate
# generic_filter callback path costs minutes per sweep; the vectorized
# path computes the same result in well under a second.
_VECTORIZED_STATS = {
    compute_ndimage_median: np.nanmedian,
    compute_ndimage_mean: np.nanmean,
    compute_texture_ndimage_stdev: np.nanstd,
}

def _windowed_stat_vectorized(tmp, pad, wind_size, nan_reduce, min_gates):
    """Vectorized equivalent of the compute_ndimage_* window callbacks:
    NaN when the center gate is NaN or fewer than min_gates valid values
    fall in the window; otherwise the reduction over valid values."""
    windows = np.lib.stride_tricks.sliding_window_view(tmp, wind_size)
    center = tmp[pad:-pad, pad:-pad]
    count = np.sum(~np.isnan(windows), axis=(-2, -1))
    if nan_reduce is np.nanmedian:
        # sort-based median of valid values: NaNs sort to the end, so the
        # median of the first `count` sorted values is the nanmedian —
        # much faster than np.nanmedian on windowed views
        flat = np.sort(windows.reshape(*count.shape, -1), axis=-1)
        c = np.maximum(count, 1)
        lo = np.take_along_axis(flat, ((c - 1) // 2)[..., None], axis=-1)[..., 0]
        hi = np.take_along_axis(flat, (c // 2)[..., None], axis=-1)[..., 0]
        res = (lo + hi) / 2.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            res = nan_reduce(windows, axis=(-2, -1))
    res[np.isnan(center) | (count < min_gates)] = np.nan
    return res

def smoothing_sweeps_ndimage_generic_filter(radar, stat_function, pad=1,
                                            fields=None, fields_name_suffix='STAT',
                                            dBZ_fields=None, **kwargs):
    if fields is None:
        radar_fields = list(radar.fields)
    else:
        if type(fields) is list:
            radar_fields = fields
        else:
            raise TypeError('"fields" must be a list.')

    # skip fields whose smoothed output already exists on the radar
    radar_fields = [
        f for f in radar_fields
        if f'{f}_{fields_name_suffix}' not in radar.fields
    ]

    pad_size = 2 * pad + 1
    wind_size = (pad_size, pad_size)

    nan_reduce = _VECTORIZED_STATS.get(stat_function)
    vectorized = nan_reduce is not None and set(kwargs) <= {'min_gates'}

    nsweep = radar.nsweeps
    sweep_start = radar.sweep_start_ray_index['data']
    sweep_end = radar.sweep_end_ray_index['data']

    for field in radar_fields:
        field_data = radar.fields[field]['data'].copy()

        if dBZ_fields is not None:
            if type(dBZ_fields) is list:
                if field in dBZ_fields:
                    field_data = 10**(field_data / 10.0)
            else:
                raise TypeError('"dBZ_fields" must be a list.')

        field_data.fill_value = np.nan
        field_data = field_data.filled()
        field_res = field_data.copy() * np.nan

        for s in range(nsweep):
            tmp = field_data[sweep_start[s]:(sweep_end[s] + 1), :]
            range_edge1 = tmp[:, :pad] * np.nan
            range_edge2 = tmp[:, -pad:] * np.nan
            tmp = np.concatenate((range_edge1, tmp, range_edge2), axis=1)
            tmp = np.concatenate((tmp[-pad:, :] , tmp, tmp[:pad, :]), axis=0)
            if vectorized:
                res_int = _windowed_stat_vectorized(
                    tmp, pad, wind_size, nan_reduce,
                    kwargs.get('min_gates', 0))
                field_res[sweep_start[s]:(sweep_end[s] + 1), :] = res_int
                continue
            res = ndi.generic_filter(tmp, stat_function, size=wind_size, extra_keywords=kwargs, mode='nearest')
            field_res[sweep_start[s]:(sweep_end[s] + 1), :] = res[pad:-pad, pad:-pad]

        field_res = np.ma.masked_where(np.isnan(field_res), field_res)
        field_res.fill_value = -9999
        out_field = f'{field}_{fields_name_suffix}'
        radar.add_field_like(field, out_field, field_res, replace_existing=True)

    return radar

def smoothing_sweeps_moving_window(radar, stat_function, pad=1,
                                   fields=None, fields_name_suffix='STAT',
                                   dBZ_fields=None, **kwargs):
    if fields is None:
        radar_fields = list(radar.fields)
    else:
        if type(fields) is list:
            radar_fields = fields
        else:
            raise TypeError('"fields" must be a list.')

    tmp_fields = []
    for field in radar_fields:
        out_field = f'{field}_{fields_name_suffix}'
        if out_field in radar_fields:
            tmp_fields += [field]
        else:
            continue
    if len(tmp_fields) > 0:
        radar_fields = tmp_fields

    nsweep = radar.nsweeps
    ngates = radar.ngates
    sweep_start = radar.sweep_start_ray_index['data']
    sweep_end = radar.sweep_end_ray_index['data']

    sweep_index = []
    for s in range(nsweep):
        nrow = sweep_end[s] - sweep_start[s] + 1
        sweep_index += [get_padding_index(pad, nrow, ngates)]

    for field in radar_fields:
        field_data = radar.fields[field]['data'].copy()

        if dBZ_fields is not None:
            if type(dBZ_fields) is list:
                if field in dBZ_fields:
                    field_data = 10**(field_data / 10.0)
            else:
                raise TypeError('"dBZ_fields" must be a list.')

        field_data.fill_value = np.nan
        field_data = field_data.filled()
        field_res = field_data.copy() * np.nan

        for s in range(nsweep):
            tmp = field_data[sweep_start[s]:(sweep_end[s] + 1), :]
            res = tmp * np.nan
            tmp = np.concatenate((res[:, :pad], tmp, res[:, -pad:]), axis=1)
            tmp = np.concatenate((tmp[-pad:, :] , tmp, tmp[:pad, :]), axis=0)

            for ix in sweep_index[s]:
                x = tmp[ix['ix_az'], ix['ix_rg']]
                x = stat_function(x, **kwargs)
                res[ix['az'], ix['rg']] = x
            
            field_res[sweep_start[s]:(sweep_end[s] + 1), :] = res

        field_res = np.ma.masked_where(np.isnan(field_res), field_res)
        field_res.fill_value = -9999
        out_field = f'{field}_{fields_name_suffix}'
        radar.add_field_like(field, out_field, field_res, replace_existing=True)

    return radar

def smoothing_rays_moving_window(radar, stat_function,
                                 wind_size=7, fields=None,
                                 overwrite_fields=False,
                                 fields_name_suffix='SRMV', **kwargs):
    hwd = int((wind_size - 1) / 2)
    lwd = hwd + 1 if wind_size % 2 == 0 else hwd

    if fields is None:
        radar_fields = list(radar.fields)
    else:
        if type(fields) is list:
            radar_fields = fields
        else:
            raise TypeError('"fields" must be a list.')

    tmp_fields = []
    for field in radar_fields:
        out_field = f'{field}_{fields_name_suffix}'
        if out_field in radar_fields:
            tmp_fields += [field]
        else:
            continue
    if len(tmp_fields) > 0:
        radar_fields = tmp_fields

    for field in radar_fields:
        tmp = radar.fields[field]['data'].copy()
        res = np.ma.zeros(tmp.shape)
        for ray in range(res.shape[0]):
            roll_win = pyart.util.rolling_window(tmp[ray, :], wind_size)
            ray_stat = stat_function(roll_win, **kwargs)
            res[ray, hwd:-lwd] = ray_stat
            res[ray, -lwd:] = np.ones(lwd) * ray_stat[-1]
            res[ray, 0:hwd] = np.ones(hwd) * ray_stat[0]

        res = np.ma.masked_where(np.isnan(res), res)
        res.fill_value = -9999
        if overwrite_fields:
            radar.fields[field]['data'] = res
        else:
            out_field = f'{field}_{fields_name_suffix}'
            radar.add_field_like(field, out_field, res, replace_existing=True)

    return radar

def get_padding_index(pad, nrow, ncol):
    # wind_size = 9 = 3 * 3 = ((2 * pad) + 1)^2
    index = []
    for az in range(pad, nrow + pad):
        ir = np.arange(-pad, pad + 1) + az
        for rg in range(pad, ncol + pad):
            ic = np.arange(-pad, pad + 1) + rg
            ix = {'az': az - pad,
                  'rg': rg - pad,
                  'ix_az': ir[:, None],
                  'ix_rg': ic[None, :]}
            index += [ix]
    return index

### for each ray on all sweeps
def compute_stat_interval_window(radar, stat_function, fields=None, min_valid=10,
                                 bin_size=10, range_min=10, range_max=200, 
                                 **kwargs):
    if fields is None:
        radar_fields = list(radar.fields)
    else:
        if type(fields) is list:
            radar_fields = fields
        else:
            raise TypeError('"fields" must be a list.')

    ranges = radar.range['data']
    bin_size = bin_size * 1000.
    range_min = range_min * 1000.
    range_max = range_max * 1000.

    bins = np.arange(range_min, range_max, bin_size)
    index = np.digitize(ranges, bins, right=False)
    index = [ranges[index == i] for i in range(1, len(bins) + 1)]
    index[-1] = index[-1][index[-1] < range_max]
    index = [np.where(np.in1d(ranges, i))[0] for i in index]

    # nsweep = radar.nsweeps
    sweep_start = radar.sweep_start_ray_index['data']
    sweep_end = radar.sweep_end_ray_index['data']

    drays = sweep_end - sweep_start + 1
    lray = (drays - drays[0]) == 0
    sweep_start = sweep_start[lray]
    sweep_end = sweep_end[lray]
    nsweep = np.sum(lray).item()

    data_fields = {}
    for field in radar_fields:
        tmp = radar.fields[field]['data'].copy()
        tmp = tmp.filled(fill_value=np.nan)
        tmp = np.array([tmp[sweep_start[s]:(sweep_end[s] + 1), :] for s in range(nsweep)])
        res = np.full((tmp.shape[1], len(index)), np.nan)
        for ray in range(res.shape[0]):
            ray_stat = np.array([stat_function(tmp[:, ray, i], **kwargs) for i in index])
            bins_valid = np.array([np.sum(~np.isnan(tmp[:, ray, i])) for i in index])
            ray_stat[bins_valid < min_valid] = np.nan
            res[ray, :] = ray_stat
        
        data_fields[field] = res

    return data_fields

def compute_stat_aggregation(data_dict, stat_function, min_valid=3, fields=None):
    if fields is None:
        fields_dict = list(data_dict[0].keys())
    else:
        if type(fields) is list:
            fields_dict = fields
        else:
            raise TypeError('"fields" must be a list.')

    data_fields = {}
    for field in fields_dict:
        tmp = [data[field] for data in data_dict]
        tmp = np.array(tmp)
        data_valid = np.sum(~np.isnan(tmp), axis=0)
        data_stat = stat_function(tmp, axis=0)
        data_stat[data_valid < min_valid] = np.nan
        data_fields[field] = data_stat

    return data_fields
