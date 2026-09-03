"""Per-altitude-layer bird share for dual-pol vertical profiling (Step 2).

Same "classify first, RCS last" pipeline as species_mask, but instead of
writing two starved polar volumes for two separate vol2bird runs (whose
per-layer sample minima delete sparse scans, and whose split velocity
fields ruin the wind fit), this computes the per-height-ring bird share
of biological linear reflectivity by HARD gate classification:

    bird gates:    P(bird) >= PROB_THRES        (confident bird)
    insect gates:  P(bird) <= 1 - PROB_THRES    (confident insect)
    ambiguous:     everything else, incl. gates the model cannot score

    share_bird(h) = (Z_bird + Z_ambig * r(h)) / (Z_bird + Z_insect + Z_ambig)
    r(h) = Z_bird / (Z_bird + Z_insect), falling back to the scan-wide
           classified ratio when a ring has no confidently classified Z

A probability-weighted soft split (the previous formula) put a nonzero
slice of BOTH species in every layer because RandomForest probabilities
are almost never exactly 0 or 1 — insect profiles were then always a
scaled shadow of bird profiles and vice versa. Hard assignment restores
the agreed pipeline (classify each gate, extract the species' own gates,
quantify each class from them): a scan with no confidently-insect gate
yields an identically-zero insect profile. Ambiguous reflectivity is
apportioned by the classified ratio so conservation stays exact:
eta_bird + eta_insect == eta per layer by construction.

The caller runs vol2bird ONCE on the normally-filtered total volume
(well-conditioned eta, velocities, sd_vvp) and splits the retrieved
profile: eta_bird = eta * share, eta_insect = eta * (1 - share).

CLI:  python -m BioModRadar.model.species_share SRC MODEL
Prints one line:  SHARES {json}
  heights: ring lower edges (m ASL), share_bird, z_sum, gates per ring,
  z_bird/z_insect/z_ambig diagnostics, gates_bio, gates_scored,
  prior_p_bird (scan-wide classified bird ratio).
"""
import json
import sys

import numpy as np

from .features_predict import build_features_predict
from .models_predict import predict_ML_proba

FIELDS_DICT = {'ref': 'DBZH', 'zdr': 'ZDR', 'rho': 'RHOHV',
               'phi': 'PHIDP', 'vel': 'VRADH', 'sw': 'WRADH'}
FEATURES = ['DBZH_MED', 'ZDR_MED', 'RHOHV_MED',
            'PHIDP_MED', 'VRADH_MED', 'WRADH_MED']
LAYER_M = 100.0
MAX_ALT_M = 10000.0
# Confidence for a hard gate class; matches rwanda_bioclass prob_thres.
PROB_THRES = 0.6


def compute_species_shares(src, file_model,
                           volume_type='rwanda-odim-h5', sweeps=None):
    if sweeps is None:
        sweeps = np.arange(0, 11)
    radar, _ = build_features_predict(
        src, volume_type, sweeps, FIELDS_DICT,
        spatial_stat_fields=True, texture_fields=False,
        dr_thres=-12, rho_thres=0.95, ref_thres=35
    )
    bio_mask = np.asarray(radar.fields['DR_CLASS']['data'].filled(0) == 1)
    dbz = radar.fields['DBZH']['data'].filled(np.nan)
    p_bird = predict_ML_proba(radar, FEATURES, file_model)
    classified = (~p_bird.mask) & bio_mask
    p = p_bird.filled(0.5)

    # hard gate classes at the bioclass confidence threshold; everything
    # the model cannot score confidently (or at all) is ambiguous
    hard_bird = classified & (p >= PROB_THRES)
    hard_insect = classified & (p <= 1.0 - PROB_THRES)

    # gate altitude ASL from Py-ART georeferencing
    alt = radar.gate_altitude['data']
    use = bio_mask & np.isfinite(dbz)
    z = 10.0 ** (dbz[use] / 10.0)
    h = alt[use]
    is_b = hard_bird[use]
    is_i = hard_insect[use]

    edges = np.arange(0.0, MAX_ALT_M + LAYER_M, LAYER_M)
    idx = np.clip(np.digitize(h, edges) - 1, 0, len(edges) - 2)
    n_rings = len(edges) - 1
    zb = np.zeros(n_rings)
    zi = np.zeros(n_rings)
    za = np.zeros(n_rings)
    n_gates = np.zeros(n_rings, dtype=int)
    np.add.at(zb, idx, np.where(is_b, z, 0.0))
    np.add.at(zi, idx, np.where(is_i, z, 0.0))
    np.add.at(za, idx, np.where(~(is_b | is_i), z, 0.0))
    np.add.at(n_gates, idx, 1)
    z_sum = zb + zi + za

    # scan-wide classified bird ratio: the prior for rings (and layers on
    # the R side) without confidently classified reflectivity
    zb_tot, zi_tot = float(zb.sum()), float(zi.sum())
    r_scan = zb_tot / (zb_tot + zi_tot) if (zb_tot + zi_tot) > 0 else 0.5

    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(zb + zi > 0, zb / (zb + zi), r_scan)
        share = np.where(z_sum > 0, (zb + za * r) / z_sum, np.nan)

    keep = z_sum > 0
    return {
        'heights': edges[:-1][keep].tolist(),
        'share_bird': np.round(share[keep], 5).tolist(),
        'z_sum': np.round(z_sum[keep], 4).tolist(),
        'z_bird': np.round(zb[keep], 4).tolist(),
        'z_insect': np.round(zi[keep], 4).tolist(),
        'z_ambig': np.round(za[keep], 4).tolist(),
        'gates': n_gates[keep].tolist(),
        'gates_bio': int(bio_mask.sum()),
        'gates_scored': int(classified.sum()),
        'prior_p_bird': round(r_scan, 4),
    }


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            'usage: python -m BioModRadar.model.species_share SRC MODEL\n')
        return 2
    out = compute_species_shares(argv[1], argv[2])
    sys.stdout.write('SHARES ' + json.dumps(out) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
