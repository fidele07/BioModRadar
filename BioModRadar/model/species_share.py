"""Per-altitude-layer bird share for dual-pol vertical profiling (Step 2).

Same "classify first, RCS last" pipeline as species_mask, but instead of
writing two starved polar volumes for two separate vol2bird runs (whose
per-layer sample minima delete sparse scans, and whose split velocity
fields ruin the wind fit), this computes the per-height-ring share of
biological linear reflectivity that the classifier attributes to birds:

    share_bird(h) = sum(Z * P(bird)) / sum(Z)   over bio gates in ring h

The caller runs vol2bird ONCE on the normally-filtered total volume
(well-conditioned eta, velocities, sd_vvp) and splits the retrieved
profile: eta_bird = eta * share, eta_insect = eta * (1 - share) —
conservation is exact per layer by construction. Gates the classifier
cannot score use the scan prior, identical to species_mask.

CLI:  python -m BioModRadar.model.species_share SRC MODEL
Prints one line:  SHARES {json}
  heights: ring lower edges (m ASL), share_bird, z_sum, gates per ring,
  gates_bio, gates_scored, prior_p_bird.
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
    prior = float(p_bird[classified].mean()) if classified.any() else 0.5
    p = np.clip(np.where(classified, p_bird.filled(prior), prior), 0.0, 1.0)

    # gate altitude ASL from Py-ART georeferencing
    alt = radar.gate_altitude['data']
    use = bio_mask & np.isfinite(dbz)
    z = 10.0 ** (dbz[use] / 10.0)
    pb = p[use]
    h = alt[use]

    edges = np.arange(0.0, MAX_ALT_M + LAYER_M, LAYER_M)
    idx = np.clip(np.digitize(h, edges) - 1, 0, len(edges) - 2)
    n_rings = len(edges) - 1
    z_sum = np.zeros(n_rings)
    zp_sum = np.zeros(n_rings)
    n_gates = np.zeros(n_rings, dtype=int)
    np.add.at(z_sum, idx, z)
    np.add.at(zp_sum, idx, z * pb)
    np.add.at(n_gates, idx, 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(z_sum > 0, zp_sum / z_sum, np.nan)

    keep = z_sum > 0
    return {
        'heights': edges[:-1][keep].tolist(),
        'share_bird': np.round(share[keep], 5).tolist(),
        'z_sum': np.round(z_sum[keep], 4).tolist(),
        'gates': n_gates[keep].tolist(),
        'gates_bio': int(bio_mask.sum()),
        'gates_scored': int(classified.sum()),
        'prior_p_bird': round(prior, 4),
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
