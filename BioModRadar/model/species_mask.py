"""Species-masked polar volumes for dual-pol vertical profiling (Step 2).

Writes two copies of a Rwanda ODIM volume in which every gate NOT
classified as the requested species is blanked to `nodata`, so that a
subsequent vol2bird run retrieves a truly species-specific profile:

    bird volume   -> gates with BIO_CLASS == 1 kept
    insect volume -> gates with BIO_CLASS == 0 kept

Gates that are unclassified (weather, low confidence under prob_thres,
missing data) are excluded from BOTH volumes.

Assumptions (documented for review):
- Ray/bin ordering in the ODIM datasets matches the order produced by
  ``read_radar_data`` for the same file (Py-ART's ODIM reader preserves
  file order); a shape guard blanks any dataset that does not line up
  instead of risking cross-species leakage.
- ODIM datasets whose elevation does not match an extracted sweep
  (duplicate sweeps removed by the Rwanda reader) are blanked entirely.

CLI (used by BioVPRadar's wrapper via system2):

    python -m BioModRadar.model.species_mask SRC DST_BIRD DST_INSECT MODEL
"""
import shutil
import sys

import h5py
import numpy as np

from .features_predict import build_features_predict
from .models_predict import predict_ML_models

# production configuration (mirrors BioProcRadar rwanda_bioclass)
FIELDS_DICT = {'ref': 'DBZH', 'zdr': 'ZDR', 'rho': 'RHOHV',
               'phi': 'PHIDP', 'vel': 'VRADH', 'sw': 'WRADH'}
FEATURES = ['DBZH_MED', 'ZDR_MED', 'RHOHV_MED',
            'PHIDP_MED', 'VRADH_MED', 'WRADH_MED']
# moments vol2bird consumes; blanking these removes the gate from profiling
MASK_QUANTITIES = {'DBZH', 'VRADH', 'WRADH'}


def write_species_masked_volumes(
        src, dst_bird, dst_insect, file_model,
        volume_type='rwanda-odim-h5', sweeps=None,
        prob_thres=0.6):
    """Classify `src` per gate and write bird-/insect-only ODIM copies."""
    if sweeps is None:
        sweeps = np.arange(0, 11)
    radar, _ = build_features_predict(
        src, volume_type, sweeps, FIELDS_DICT,
        spatial_stat_fields=True, texture_fields=False,
        dr_thres=-12, rho_thres=0.9, ref_thres=30
    )
    radar = predict_ML_models(
        radar, FEATURES, file_model, prob_thres=prob_thres
    )
    bio = radar.fields['BIO_CLASS']['data'].filled(-1)

    for dst, keep in ((dst_bird, 1), (dst_insect, 0)):
        shutil.copyfile(src, dst)
        _mask_odim_file(dst, radar, bio, keep)
    return 0


def _mask_odim_file(path, radar, bio, keep):
    fixed_angles = radar.fixed_angle['data']
    with h5py.File(path, 'r+') as h:
        for name in sorted(k for k in h.keys() if k.startswith('dataset')):
            elangle = float(h[name]['where'].attrs['elangle'])
            match = np.where(np.isclose(fixed_angles, elangle, atol=0.05))[0]
            keep_mask = None
            if len(match) > 0:
                sl = radar.get_slice(int(match[0]))
                keep_mask = bio[sl] == keep

            for dname in sorted(k for k in h[name].keys()
                                if k.startswith('data')):
                grp = h[name][dname]
                quantity = grp['what'].attrs['quantity']
                if isinstance(quantity, bytes):
                    quantity = quantity.decode()
                if quantity not in MASK_QUANTITIES:
                    continue
                nodata = grp['what'].attrs['nodata']
                arr = grp['data'][...]
                if keep_mask is None or keep_mask.shape != arr.shape:
                    # unmatched or misaligned sweep: blank it entirely
                    # rather than risking cross-species leakage
                    arr[:] = nodata
                else:
                    arr[~keep_mask] = nodata
                grp['data'][...] = arr


def main(argv):
    if len(argv) != 5:
        sys.stderr.write(
            'usage: python -m BioModRadar.model.species_mask '
            'SRC DST_BIRD DST_INSECT MODEL\n'
        )
        return 2
    return write_species_masked_volumes(
        argv[1], argv[2], argv[3], argv[4]
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv))
