import numpy as np
from .blob_seg import radar_blob_segmentation
import pyart

def depolarization_ratio(radar, sweeps, zdr_field, rho_field, ref_field,
                         dr_thres=-12, rho_thres=0.95, ref_thres=35,
                         use_blob_mask=False, blob_min_size=100, blob_connectivity=2,
                         despeckle_class=False, despeckle_size=25):
    zdr = radar.fields[zdr_field]['data'].copy()
    rho = radar.fields[rho_field]['data'].copy()
    ref = radar.fields[ref_field]['data'].copy()
    if use_blob_mask:
        mask_zdr = radar_blob_segmentation(radar, sweeps, zdr_field, None,
                                           blob_min_size, blob_connectivity)
        zdr = np.ma.masked_array(zdr, mask=mask_zdr.fields['blob_mask']['data'])
        mask_rho = radar_blob_segmentation(radar, sweeps, rho_field, None,
                                           blob_min_size, blob_connectivity)
        rho = np.ma.masked_array(rho, mask=mask_rho.fields['blob_mask']['data'])
        mask_ref = radar_blob_segmentation(radar, sweeps, ref_field, -15,
                                           blob_min_size, blob_connectivity)
        ref = np.ma.masked_array(ref, mask=mask_ref.fields['blob_mask']['data'])

    # ZDR is in dB; the depolarization ratio formula requires the
    # linear differential reflectivity. Using dB directly makes
    # sqrt(zdr) invalid for negative ZDR (common for birds), masking
    # those gates and preferentially removing bird echoes.
    zdr_lin = np.ma.power(10., zdr / 10.)
    dr_n = (zdr_lin + 1 - 2 * np.ma.sqrt(zdr_lin) * rho)
    dr_d = (zdr_lin + 1 + 2 * np.ma.sqrt(zdr_lin) * rho)
    dr = 10 * np.log10(dr_n/dr_d)
    radar.add_field_like(ref_field, 'DR', dr, replace_existing=True)

    dr_class = np.empty(dr.shape, dtype=np.int16)
    dr_class = np.ma.masked_array(dr_class, mask=dr.mask)

    weather = np.logical_or(np.logical_or(dr <= dr_thres, ref > ref_thres), rho > rho_thres)
    dr_class[weather] = 0
    biology = np.logical_and(np.logical_and(dr > dr_thres, ~np.ma.getmask(dr)), ~weather)
    dr_class[biology] = 1

    dr_class_dict = {
        'data': dr_class,
        'units': '',
        'long_name': 'Depolarization Classification: weather=0, biology=1',
        '_FillValue': -9999,
        'standard_name': 'depolarization_class',
    }
    radar.add_field('DR_CLASS', dr_class_dict, replace_existing=True)

    if despeckle_class:
        dr_despeckeld = pyart.correct.despeckle_field(radar, 'DR_CLASS',
                                                     size=despeckle_size,
                                                     threshold=(0, 1))
        dr_class = np.ma.masked_where(dr_despeckeld.gate_excluded, dr_class)
        radar.fields['DR_CLASS']['data'] = dr_class

    return radar

def compute_dr(radar, zdr_field, rho_field):
    zdr = radar.fields[zdr_field]['data'].copy()
    rho = radar.fields[rho_field]['data'].copy()
    zdr = np.ma.masked_where(np.isnan(zdr), zdr)
    rho = np.ma.masked_where(np.isnan(rho), rho)
    zdr_lin = np.ma.power(10., zdr / 10.)
    dr_n = (zdr_lin + 1 - 2 * np.ma.sqrt(zdr_lin) * rho)
    dr_d = (zdr_lin + 1 + 2 * np.ma.sqrt(zdr_lin) * rho)
    dr = 10 * np.log10(dr_n/dr_d)
    radar.add_field_like(zdr_field, 'DR', dr, replace_existing=True)
    return radar
