import os
import re
import copy
import warnings
import numpy as np
import pandas as pd
import joblib
from .models_fit import compute_fuzzy_scores

# def predict_fuzzy_logic(radar, features, file_model):
#     if not os.path.exists(file_model):
#        raise FileNotFoundError(f'File containing the fuzzy model not found.')

#     model = joblib.load(file_model)

# ## model fitting,
# features = ['REFH_MED', 'PHID_MED', 'RHOV_MED', 'ZDR_MED', 'VV_MED', 'WD_MED']
# ## predict, diffente radar
# fields_dict = {'ref': 'DBZH', 'zdr': 'ZDR', 'rho': 'RHOHV',
#                'phi': 'PHIDP', 'vel': 'VRADH', 'sw': 'WRADH'}
# features = ['DBZH_MED', 'PHIDP_MED', 'RHOHV_MED', 'ZDR_MED', 'VRADH_MED', 'WRADH_MED']
# ## in case features are not ordered
# features = ['ZDR_MED', 'RHOHV_MED', 'DBZH_MED', 'PHIDP_MED', 'WRADH_MED', 'VRADH_MED']

#######

def predict_fuzzy_logic(radar, features, file_stats_bird, file_stats_insect, fields_dict=None):
    if not os.path.exists(file_stats_bird):
       raise FileNotFoundError(f'File containing the bird statistics not found.')
    if not os.path.exists(file_stats_insect):
       raise FileNotFoundError(f'File containing the insect statistics not found.')

    features_stat = copy.deepcopy(features)
    if fields_dict is not None:
        for k, v in fields_dict.items():
            for i in range(len(features_stat)):
                features_stat[i] = re.sub(v, k.upper(), features_stat[i])

    stats_bird = pd.read_csv(file_stats_bird, index_col=0)
    stats_bird = stats_bird[features_stat].T.to_dict(orient='list')
    stats_insect = pd.read_csv(file_stats_insect, index_col=0)
    stats_insect = stats_insect[features_stat].T.to_dict(orient='list')
    data = np.array([radar.fields[f]['data'].filled(np.nan).ravel() for f in features]).T

    scores_bird = compute_fuzzy_scores(data, stats_bird)
    scores_insect = compute_fuzzy_scores(data, stats_insect)

    pred = np.full(data.shape[0], 2, dtype=np.int16)
    pred[scores_bird > scores_insect] = 1
    pred[scores_insect > scores_bird] = 0
    pred = pred.reshape(radar.fields['DR']['data'].shape)
    pred = np.ma.masked_where(pred == 2, pred)

    radar_c = copy.deepcopy(radar)
    radar_fields = list(radar_c.fields)
    delete_fields = [fl for fl in radar_fields if fl not in ['DR', 'DR_CLASS']]
    for fl in delete_fields:
        del radar_c.fields[fl]

    bio_class_dict = {
        'data': pred,
        'units': '',
        'long_name': 'Biological Classification: insect=0, bird=1',
        '_FillValue': -9999,
        'standard_name': 'biological_class',
    }
    radar_c.add_field('BIO_CLASS', bio_class_dict, replace_existing=True)

    return radar_c

def predict_ML_models(radar, features, file_model, prob_thres=None):
    """Classify biological gates as insect (0) or bird (1).

    Parameters
    ----------
    prob_thres: float or None, optional
        Minimum class probability (0.5-1.0) required to keep a
        prediction. Gates below the threshold are masked (transparent)
        instead of being painted with a low-confidence class. Only
        applied when the model supports ``predict_proba``
        (RandomForest, DecisionTree); ignored for RidgeClassifier.
        ``None`` (default) keeps every prediction. The output remains
        strictly 2-class: insect=0, bird=1, everything else masked.
    """
    if not os.path.exists(file_model):
       raise FileNotFoundError(f'File containing the ML model not found.')

    model = joblib.load(file_model)
    features = _align_features_order(features, model.get('features'))
    data = np.array([radar.fields[f]['data'].filled(np.nan).ravel() for f in features]).T
    if model['scaler'] is not None:
        # transform (NOT fit_transform): the scaler must keep the
        # mean/std learned from the training data, otherwise every
        # scan is re-centered onto the training centroid and the
        # classification becomes relative-within-scan (systematic bias).
        data = model['scaler'].transform(data)

    pred = np.full(data.shape[0], 2, dtype=np.int16)
    mask = _mask_data_ML(data)
    if np.any(mask):
        pred[mask] = model['model'].predict(data[mask])
        if prob_thres is not None and hasattr(model['model'], 'predict_proba'):
            proba = model['model'].predict_proba(data[mask])
            conf = proba.max(axis=1)
            pred_m = pred[mask]
            pred_m[conf < prob_thres] = 2
            pred[mask] = pred_m
    pred = pred.reshape(radar.fields['DR']['data'].shape)
    pred = np.ma.masked_where(pred == 2, pred)

    radar_c = copy.deepcopy(radar)
    radar_fields = list(radar_c.fields)
    delete_fields = [fl for fl in radar_fields if fl not in ['DR', 'DR_CLASS']]
    for fl in delete_fields:
        del radar_c.fields[fl]

    bio_class_dict = {
        'data': pred,
        'units': '',
        'long_name': 'Biological Classification: insect=0, bird=1',
        '_FillValue': -9999,
        'standard_name': 'biological_class',
    }
    radar_c.add_field('BIO_CLASS', bio_class_dict, replace_existing=True)

    return radar_c

def predict_ML_proba(radar, features, file_model, fallback_raw=True):
    """Per-gate bird probability P(bird) for soft species assignment.

    Unlike ``predict_ML_models`` (hard 2-class output with an optional
    confidence cut, meant for map display), this returns the classifier's
    continuous P(bird) for every gate whose features are complete, and
    keeps NO confidence threshold: profiling must not discard biological
    signal, it splits it proportionally instead.

    fallback_raw: the '*_MED' features are median-smoothed and need a
    populated window (min_gates), so sparse echo — precisely the scans
    where every gate counts — loses feature coverage (3-45% scored on
    the validation suite). When True, a missing smoothed feature falls
    back to the gate's RAW field value, trading a little smoothing for
    scoring coverage. The scaler/model see the same feature space.

    Returns a masked array shaped like the radar sweeps: P(bird) in
    [0, 1]; masked where features are incomplete (caller decides the
    prior for those gates).
    """
    if not os.path.exists(file_model):
        raise FileNotFoundError('File containing the ML model not found.')

    model = joblib.load(file_model)
    if not hasattr(model['model'], 'predict_proba'):
        raise TypeError(
            'Soft species assignment requires a probabilistic model '
            f"(got {type(model['model']).__name__})."
        )
    features = _align_features_order(features, model.get('features'))
    cols = []
    for f in features:
        col = radar.fields[f]['data'].filled(np.nan).ravel()
        raw_name = f[:-4] if f.endswith('_MED') else None
        if fallback_raw and raw_name and raw_name in radar.fields:
            raw = radar.fields[raw_name]['data'].filled(np.nan).ravel()
            col = np.where(np.isnan(col), raw, col)
        cols.append(col)
    data = np.array(cols).T
    if model['scaler'] is not None:
        data = model['scaler'].transform(data)

    p_bird = np.full(data.shape[0], np.nan)
    mask = _mask_data_ML(data)
    if np.any(mask):
        proba = model['model'].predict_proba(data[mask])
        classes = list(model['model'].classes_)
        p_bird[mask] = proba[:, classes.index(1)]
    p_bird = p_bird.reshape(radar.fields['DR']['data'].shape)
    return np.ma.masked_invalid(p_bird)


def _align_features_order(features, model_features):
    """Ensure the feature matrix columns match the training order.

    The model was fitted on ``model_features`` in a fixed order.
    If the caller supplies the same names in a different order,
    reorder them. If the names differ (e.g. different radar field
    naming), assume the caller provides them positionally in the
    training order, and warn.
    """
    if model_features is None:
        return list(features)
    model_features = list(model_features)
    features = list(features)
    if len(model_features) != len(features):
        raise ValueError(
            'Number of features does not match the trained model: '
            f'model expects {len(model_features)} features '
            f'({", ".join(model_features)}), got {len(features)} '
            f'({", ".join(features)}).'
        )
    if set(model_features) == set(features):
        return model_features
    warnings.warn(
        'Feature names differ from the trained model '
        f'(model: {", ".join(model_features)}; '
        f'supplied: {", ".join(features)}). Assuming the supplied '
        'features are already ordered as in training.'
    )
    return features

def _mask_data_ML(data):
    mask_2d = ~np.isnan(data)
    mask_1d = np.full(data.shape[0], True, dtype=bool)
    for i in range(data.shape[1]):
        mask_1d = np.logical_and(mask_1d, mask_2d[:, i])

    return mask_1d
