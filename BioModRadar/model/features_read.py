import numpy as np
import pandas as pd

def read_features_dataframe(file_csv, f_insect=1, f_bird=1):
    data = pd.read_csv(file_csv)

    ix_insect = np.array(data[data['label'] == 0].index)
    ix_bird = np.array(data[data['label'] == 1].index)
    n_insect = len(ix_insect)
    n_bird = len(ix_bird)

    np.random.seed(42)
    if n_insect > n_bird:
        nf = int(np.floor(f_insect * n_bird))
        if nf > n_insect:
            nf = n_insect
        ix = np.random.choice(n_insect, nf, replace=False)
        ix_insect = ix_insect[ix]
    else:
        nf = int(np.floor(f_bird * n_insect))
        if nf > n_bird:
            nf = n_bird
        ix = np.random.choice(n_bird, nf, replace=False)
        ix_bird = ix_bird[ix]

    data_instect = data.iloc[ix_insect]
    data_instect = data_instect.reset_index(drop=True)
    data_bird = data.iloc[ix_bird]
    data_bird = data_bird.reset_index(drop=True)

    return data_bird, data_instect


