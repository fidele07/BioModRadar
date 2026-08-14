"""Diagnostic for a deployed bird/insect model pickle.

Run on the machine that hosts the model (path from
BioConfigRadar/config/config_datasets.yaml -> models: dir/job):

    python verify_model.py /path/to/model.joblib

Checks:
  1. model['features'] - the exact names AND order the model was fit on.
     The production caller (BioProcRadar rwanda_bioclass.py) must supply
     the same features; since the fix, predict_ML_models reorders
     matching names automatically, but different names are only mapped
     positionally.
  2. model['model'].classes_ - must be [0 1] (0=insect, 1=bird).
  3. model['scaler'] - if present, prints the training means/stds the
     (fixed) transform will now use.
"""
import sys
import joblib
import numpy as np

PRODUCTION_FEATURES = ['DBZH', 'PHIDP', 'RHOHV', 'ZDR', 'VRADH', 'WRADH']

def main(path):
    model = joblib.load(path)
    print(f'pickle: {path}')
    print(f'keys:   {list(model.keys())}')

    clf = model.get('model')
    print(f'\nclassifier: {type(clf).__name__}')
    if hasattr(clf, 'classes_'):
        print(f'classes_:   {clf.classes_}   (expected [0 1]; 0=insect, 1=bird)')
        if not np.array_equal(clf.classes_, [0, 1]):
            print('  !! UNEXPECTED CLASS LABELS - investigate the training labels')

    feats = model.get('features')
    print(f'\nmodel features (training order): {feats}')
    if feats is None:
        print('  !! no feature list stored - predictions rely entirely on caller order')
    else:
        if list(feats) == PRODUCTION_FEATURES:
            print('  OK: identical names and order to the production caller.')
        elif set(feats) == set(PRODUCTION_FEATURES):
            print('  OK-ish: same names, different order.')
            print('  The fixed predict_ML_models reorders these automatically;')
            print('  the OLD code silently permuted the columns -> garbage output.')
        else:
            print('  !! NAME MISMATCH vs production caller:')
            print(f'     caller sends: {PRODUCTION_FEATURES}')
            print('     If these are *_MED or other-radar names, the model was')
            print('     trained on different (e.g. smoothed) inputs than it is')
            print('     served -> systematic misclassification. Retrain with the')
            print('     production feature set, or serve the training features.')

    scaler = model.get('scaler')
    if scaler is None:
        print('\nscaler: None (tree model; the fit_transform bug was inert here)')
    else:
        print(f'\nscaler: {type(scaler).__name__}')
        print('  The OLD predict code called fit_transform per scan, re-centering')
        print('  every scan onto the training centroid (systematic class bias).')
        print('  The FIXED code uses these stored training statistics:')
        if hasattr(scaler, 'mean_'):
            names = feats if feats is not None else range(len(scaler.mean_))
            for n, m, s in zip(names, scaler.mean_, np.sqrt(scaler.var_)):
                print(f'    {n:>12}: mean={m:10.3f}  std={s:10.3f}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python verify_model.py /path/to/model.joblib')
    main(sys.argv[1])
