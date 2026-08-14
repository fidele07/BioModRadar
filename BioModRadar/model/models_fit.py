import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

def confusion_matrix_1(y_true, y_pred):
    ### Compute confusion matrix components for bird detection
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # True Positives (Bird correctly classified)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # True Negatives (Insect correctly classified)
    tn = np.sum((y_pred == 0) & (y_true == 0))
    # False Positives (Insect classified as Bird)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    # False Negatives (Bird classified as Insect)
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tn, fp, fn, tp

def confusion_matrix_2(y_true, y_pred):
    ### Compute confusion matrix components for bird detection
    conf_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_mat.ravel()
    return tn, fp, fn, tp

def model_metrics(y_true, y_pred, y_score):
    # tn, fp, fn, tp = confusion_matrix_2(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix_1(y_true, y_pred)

    # acc = accuracy_score(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_score)
    metrics = {'acc': acc, 'tpr': tpr,
               'tnr': tnr, 'auc': auc}
    cmatrix = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    return {'metrics': metrics, 'cmatrix': cmatrix}

def gaussian_membership(x, stats):
    p = np.exp(-((x - stats['mean'])**2) / (2 * stats['std']**2))
    # d = (stats['sigma'] * np.sqrt(2 * np.pi))
    # p = np.exp(-((x - stats['mean'])**2) / (2 * stats['sigma']**2)) / d
    return p

def membership_fuzzy_stats(data, fields):
    stats = {}
    for f in fields:
        x = data[f]
        mean = np.nanmean(x)
        std = np.nanstd(x)
        sigma = 1.06 * std * len(x)**(-1/5)
        stats[f] = {'mean': np.round(mean, 4),
                    'std': np.round(std, 4),
                    'sigma': np.round(sigma, 4)}
    return  pd.DataFrame(stats)

def membership_fuzzy_train(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    n = data.shape[0]
    sigma = 1.06 * std * n**(-1/5)
    return {'mean': mean,
            'std': std,
            'sigma': sigma}

def compute_fuzzy_scores(data, stats):
    prod = np.full_like(data[:, 0], 1.)
    for j in range(data.shape[1]):
        stat = {s: stats[s][j] for s in stats}
        prod = prod * gaussian_membership(data[:, j], stat)
    return prod

def train_FuzzyLogic_models(data, features, label='label'):
    X = data[features].values
    y = data[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                  test_size=0.33, random_state=42, stratify=y)
    # membership statistics
    stats_bird = membership_fuzzy_train(X_train[y_train == 1, :])
    stats_insect = membership_fuzzy_train(X_train[y_train == 0, :])

    # Compute fuzzy scores
    scores_bird = compute_fuzzy_scores(X_test, stats_bird)
    scores_insect = compute_fuzzy_scores(X_test, stats_insect)

    # Classification: 1 = bird, 0 = insect, 2 = unclassified
    y_pred = np.full(y_test.shape, 2, dtype=np.int16)
    y_pred[scores_bird > scores_insect] = 1
    y_pred[scores_insect > scores_bird] = 0

    ix_bird = y_test == 1
    ix_insect = y_test == 0

    bird_scores_bird = scores_bird[ix_bird]
    insect_scores_bird = scores_bird[ix_insect]

    bird_scores_insect = scores_insect[ix_bird]
    insect_scores_insect =  scores_insect[ix_insect]

    # build fuzzy scores (bird probability)
    p_scores_bird = bird_scores_bird / (bird_scores_bird + insect_scores_bird + 1e-6)
    p_scores_insect = bird_scores_insect / (bird_scores_insect + insect_scores_insect + 1e-6)

    y_scores = np.zeros(y_test.shape) 
    y_scores[ix_bird] = p_scores_bird
    y_scores[ix_insect] = p_scores_insect
    stats = model_metrics(y_test, y_pred, y_scores)

    stats_bird = membership_fuzzy_stats(data[data[label] == 1], features)
    stats_insect = membership_fuzzy_stats(data[data[label] == 0], features)
    models = {'bird': stats_bird, 'insect': stats_insect, 'features': features}
    return models, stats

def train_RidgeClassifier_models(data, features, label='label', **kwargs):
    X = data[features].values
    y = data[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                  test_size=0.33, random_state=42, stratify=y)
    ridge, scaler = _train_RidgeClassifier(X_train, y_train, **kwargs)
    X_test_s = scaler.transform(X_test)
    y_pred = ridge.predict(X_test_s)
    y_scores = ridge.decision_function(X_test_s)
    stats = model_metrics(y_test, y_pred, y_scores)

    ridge, scaler = _train_RidgeClassifier(X, y, **kwargs)
    models = {'model': ridge, 'scaler': scaler, 'features': features}
    return models, stats

def _train_RidgeClassifier(X, y, **kwargs):
    ## Classifier using Ridge regression
    ## require normalization (or scaling) 
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    ridge = RidgeClassifier(alpha=1.0, random_state=42, **kwargs)
    ridge.fit(X_s, y)
    return ridge, scaler

def train_DecisionTree_models(data, features, scale=False,
                              label='label', **kwargs):
    X = data[features].values
    y = data[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                  test_size=0.33, random_state=42, stratify=y)
    dtree, scaler = _train_DecisionTree(X_train, y_train, scale, **kwargs)
    if scaler is not None:
        X_test_s = scaler.transform(X_test)
        y_pred = dtree.predict(X_test_s)
        y_scores = dtree.predict_proba(X_test_s)[:, 1]
    else:
        y_pred = dtree.predict(X_test)
        y_scores = dtree.predict_proba(X_test)[:, 1]
    stats = model_metrics(y_test, y_pred, y_scores)

    dtree, scaler = _train_DecisionTree(X, y, scale, **kwargs)
    models = {'model': dtree, 'scaler': scaler, 'features': features}
    return models, stats

def _train_DecisionTree(X, y, scale, **kwargs):
    ## decision tree classifier
    ## doesn't require scaling usually
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None

    dtree = DecisionTreeClassifier(max_depth=10, random_state=42, **kwargs)
    dtree.fit(X, y)
    return dtree, scaler

def train_RandomForest_models(data, features, scale=False,
                              label='label', **kwargs):
    X = data[features].values
    y = data[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                  test_size=0.33, random_state=42, stratify=y)
    rforest, scaler = _train_RandomForest(X_train, y_train, scale, **kwargs)
    if scaler is not None:
        X_test_s = scaler.transform(X_test)
        y_pred = rforest.predict(X_test_s)
        y_scores = rforest.predict_proba(X_test_s)[:, 1]
    else:
        y_pred = rforest.predict(X_test)
        y_scores = rforest.predict_proba(X_test)[:, 1]
    stats = model_metrics(y_test, y_pred, y_scores)

    rforest, scaler = _train_RandomForest(X, y, scale, **kwargs)
    models = {'model': rforest, 'scaler': scaler, 'features': features}
    return models, stats

def _train_RandomForest(X, y, scale, **kwargs):
    ## random forest classifier
    ## doesn't require scaling usually
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None

    rforest = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     random_state=42, **kwargs)
    rforest.fit(X, y)
    return rforest, scaler
