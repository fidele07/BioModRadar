"""Species-split polar volumes for dual-pol vertical profiling (Step 2).

Pipeline order (fixed by design — "classify first, RCS last"):

1. Weather/biology separation at gate level (DR_CLASS from the
   depolarization-ratio pre-filter, weather-ONLY exclusion: the relaxed
   library thresholds rho>0.95 / ref>35 dBZ are used, not the stricter
   display settings).
2. Every remaining biological gate gets the transfer-learned dual-pol
   classifier's continuous P(bird).
3. Two ODIM copies are written with the biological reflectivity SPLIT
   proportionally (soft assignment): linear Z_bird = Z * P(bird),
   Z_insect = Z * (1 - P(bird)). Gates whose features are incomplete
   (smoothing window edges) are split by the scan's mean P(bird) —
   NO gate is dropped for low confidence; the 60% confidence cut is a
   map-display device and must never starve profiling (this is what
   collapsed the 2026-08-17 attempt).
4. vol2bird then profiles each copy independently; the species RCS is
   applied only at that final step, outside this module. RCS is never
   used to classify.

Conservation: by construction Z_bird + Z_insect == Z_bio per gate; the
CLI measures the residual actually left after ODIM 8/16-bit quantization
and prints it as the last stdout line:

    CONSERVATION {"eta_bio": ..., "eta_bird": ..., "eta_insect": ...,
                  "residual_pct": ..., ...}

The caller must flag scans whose |residual_pct| exceeds 5.

CLI (used by BioVPRadar's wrapper via system2):

    python -m BioModRadar.model.species_mask SRC DST_BIRD DST_INSECT MODEL
"""
import json
import shutil
import sys

import h5py
import numpy as np

from .features_predict import build_features_predict
from .models_predict import predict_ML_proba

FIELDS_DICT = {'ref': 'DBZH', 'zdr': 'ZDR', 'rho': 'RHOHV',
               'phi': 'PHIDP', 'vel': 'VRADH', 'sw': 'WRADH'}
FEATURES = ['DBZH_MED', 'ZDR_MED', 'RHOHV_MED',
            'PHIDP_MED', 'VRADH_MED', 'WRADH_MED']
# moments vol2bird consumes
REFL_QUANTITY = 'DBZH'
VELO_QUANTITIES = {'VRADH', 'WRADH'}
# a species keeps a gate when its share of the gate's linear Z is at
# least this much; below it the shifted dB underflows the encoding
MIN_FRAC = 0.01


def write_species_masked_volumes(
        src, dst_bird, dst_insect, file_model,
        volume_type='rwanda-odim-h5', sweeps=None):
    """Split `src` into bird-/insect-weighted ODIM copies; return stats."""
    if sweeps is None:
        sweeps = np.arange(0, 11)
    # weather-only exclusion: library-default thresholds, NOT the
    # stricter display configuration (rho 0.9 / ref 30 kept only 29% of
    # echo gates on a migration night)
    radar, _ = build_features_predict(
        src, volume_type, sweeps, FIELDS_DICT,
        spatial_stat_fields=True, texture_fields=False,
        dr_thres=-12, rho_thres=0.95, ref_thres=35
    )
    bio = radar.fields['DR_CLASS']['data']
    bio_mask = np.asarray(bio.filled(0) == 1)

    p_bird = predict_ML_proba(radar, FEATURES, file_model)
    classified = (~p_bird.mask) & bio_mask
    # scan prior for biological gates the classifier could not score
    # (incomplete feature windows): the mean P(bird) of scored gates
    prior = float(p_bird[classified].mean()) if classified.any() else 0.5
    p = np.where(classified, p_bird.filled(prior), prior)
    p = np.clip(p, 0.0, 1.0)

    stats = {
        'gates_bio': int(bio_mask.sum()),
        'gates_scored': int(classified.sum()),
        'prior_p_bird': round(prior, 4),
        'eta_bio': 0.0, 'eta_bird': 0.0, 'eta_insect': 0.0,
        # ALL valid echo (weather included) on dropped duplicate/wrong
        # sweeps - context, not lost biology (both pipelines drop them)
        'eta_echo_dropped_sweeps': 0.0,
    }
    for species, dst, frac in (('bird', dst_bird, p),
                               ('insect', dst_insect, 1.0 - p)):
        shutil.copyfile(src, dst)
        stats[f'eta_{species}'] = _split_odim_file(
            dst, radar, bio_mask, frac, stats,
            measure_bio=species == 'bird'
        )

    eta_bio = stats['eta_bio']
    split_sum = stats['eta_bird'] + stats['eta_insect']
    stats['residual_pct'] = round(
        100.0 * (split_sum - eta_bio) / eta_bio, 3
    ) if eta_bio > 0 else 0.0
    return stats


def _split_odim_file(path, radar, bio_mask, frac, stats, measure_bio):
    """Weight one ODIM copy by `frac`; return its summed linear Z."""
    fixed_angles = radar.fixed_angle['data']
    eta_sum = 0.0
    with h5py.File(path, 'r+') as h:
        for name in sorted(k for k in h.keys() if k.startswith('dataset')):
            elangle = float(h[name]['where'].attrs['elangle'])
            match = np.where(np.isclose(fixed_angles, elangle, atol=0.05))[0]
            sweep_bio = None
            sweep_frac = None
            if len(match) > 0:
                sl = radar.get_slice(int(match[0]))
                sweep_bio = bio_mask[sl]
                sweep_frac = frac[sl]
                # Py-ART does NOT preserve the file's ray order (measured:
                # its rays start at -180 deg vs the file's 0 deg — a 180 deg
                # rotation that scrambled every masked volume). Re-index
                # the mask into ODIM row order via each ray's azimuth:
                # ODIM rows are uniform in azimuth with row 0 at north.
                nrays = sweep_bio.shape[0]
                az = np.asarray(radar.azimuth['data'][sl]) % 360.0
                rows = np.rint(az / (360.0 / nrays)).astype(int) % nrays
                bio_f = np.zeros_like(sweep_bio)
                frac_f = np.zeros_like(sweep_frac)
                bio_f[rows] = sweep_bio
                frac_f[rows] = sweep_frac
                sweep_bio, sweep_frac = bio_f, frac_f

            for dname in sorted(k for k in h[name].keys()
                                if k.startswith('data')):
                grp = h[name][dname]
                quantity = grp['what'].attrs['quantity']
                if isinstance(quantity, bytes):
                    quantity = quantity.decode()
                if quantity != REFL_QUANTITY and quantity not in VELO_QUANTITIES:
                    continue
                nodata = grp['what'].attrs['nodata']
                undetect = grp['what'].attrs.get('undetect', nodata)
                gain = float(grp['what'].attrs.get('gain', 1.0)) or 1.0
                arr = grp['data'][...]
                valid = (arr != nodata) & (arr != undetect)

                if sweep_bio is None or sweep_bio.shape != arr.shape:
                    # unmatched or misaligned sweep: blank it entirely
                    # rather than risking cross-species leakage; account
                    # for the biology structurally lost this way
                    if measure_bio and quantity == REFL_QUANTITY:
                        offset = float(grp['what'].attrs.get('offset', 0.0))
                        db = gain * arr[valid].astype('float64') + offset
                        stats['eta_echo_dropped_sweeps'] += float(
                            np.sum(10.0 ** (db / 10.0))
                        )
                    arr[:] = nodata
                    grp['data'][...] = arr
                    continue

                keep = sweep_bio & (sweep_frac >= MIN_FRAC) & valid
                if quantity == REFL_QUANTITY:
                    offset = float(grp['what'].attrs.get('offset', 0.0))
                    if measure_bio:
                        db_bio = gain * arr[sweep_bio & valid].astype('float64') + offset
                        stats['eta_bio'] += float(np.sum(10.0 ** (db_bio / 10.0)))
                    # soft split in LINEAR Z: dB shift = 10*log10(frac),
                    # applied in raw encoding units (delta_dB / gain)
                    shift_raw = 10.0 * np.log10(
                        np.where(keep, sweep_frac, 1.0)
                    ) / gain
                    new = np.rint(arr.astype('float64') + shift_raw)
                    lo = float(min(nodata, undetect)) + 1.0
                    hi = float(np.iinfo(arr.dtype).max
                               if np.issubdtype(arr.dtype, np.integer)
                               else np.finfo(arr.dtype).max)
                    underflow = new < lo
                    new = np.clip(new, lo, hi).astype(arr.dtype)
                    # DENOMINATOR SEMANTICS: vol2bird's eta/dens are
                    # means over ALL sampled gates. Excluded-but-measured
                    # gates (other species' share, weather, no echo) must
                    # stay in the average as ZERO signal -> undetect, not
                    # nodata; nodata would shrink the denominator to the
                    # kept gates and inflate densities ~an order of
                    # magnitude (measured: 18x on 2025-11-15). Only gates
                    # the radar never measured remain nodata.
                    new[~keep | underflow] = undetect
                    new[arr == nodata] = nodata
                    db_out = gain * new[keep & ~underflow].astype('float64') + offset
                    eta_sum += float(np.sum(10.0 ** (db_out / 10.0)))
                    grp['data'][...] = new
                else:
                    # velocities are ensemble properties, not additive:
                    # both species keep them where they keep the gate;
                    # elsewhere velocity is UNKNOWN (nodata) — writing
                    # undetect would poison the VVP fit with fake zeros
                    arr[~keep] = nodata
                    grp['data'][...] = arr
    return eta_sum


def main(argv):
    if len(argv) != 5:
        sys.stderr.write(
            'usage: python -m BioModRadar.model.species_mask '
            'SRC DST_BIRD DST_INSECT MODEL\n'
        )
        return 2
    stats = write_species_masked_volumes(argv[1], argv[2], argv[3], argv[4])
    sys.stdout.write('CONSERVATION ' + json.dumps(stats) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
