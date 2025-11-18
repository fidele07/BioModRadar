import os
import gzip
import numpy as np
import xarray as xr
import xradar as xd

xr.set_options(use_new_combine_kwarg_defaults=True)

def read_xradar_data(file_path, sweeps=None,
                     volume_type='cfradial',
                     fields_dict=None):
    try:
        if volume_type == 'cfradial':
            dtree = xd.io.open_cfradial1_datatree(file_path)
        elif volume_type == 'odim-h5':
            dtree = xd.io.open_odim_datatree(file_path)
        elif volume_type == 'rwanda-odim-h5':
            dtree = xd.io.open_odim_datatree(file_path)
            dtree = _reduce_sweeps_rwanda_odim_hdf5(dtree)
        elif volume_type == 'nexrad-archive':
            pth, ext = os.path.splitext(file_path)
            if ext == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    file_tmp = f.read()
                dtree = xd.io.open_nexradlevel2_datatree(file_tmp)
            else:
                dtree = xd.io.open_nexradlevel2_datatree(file_path)
            dtree = _reduce_sweeps_nexrad_level2(dtree)
        else:
            raise TypeError(f'Unknown volume_type {volume_type}')
    except Exception as e:
        print(e)
        print(f'Enable to read: {file_path}')
        return None

    # sweeps and fields
    return dtree

def _reduce_sweeps_rwanda_odim_hdf5(dtree):
    sweeps, fixed_angles = _get_sweeps_fixed_angles(dtree)
    u_fixed_angles = np.unique(fixed_angles)
    if len(fixed_angles) != len(u_fixed_angles):
        rm_sweeps = np.setdiff1d(sweeps, sweeps[:-3])
        dtree = dtree.drop_nodes(rm_sweeps)
    return dtree

def _reduce_sweeps_nexrad_level2(dtree):
    sweeps, fixed_angles = _get_sweeps_fixed_angles(dtree)
    u_fixed_angles = np.unique(fixed_angles)
    if len(fixed_angles) != len(u_fixed_angles):
        kp_swp = []
        rm_swp = []
        for s in u_fixed_angles:
            dup = fixed_angles == s
            if sum(dup) > 1:
                ix = np.where(dup)
                kp_swp += [f'sweep_{ix[0][0]}']
                rm_swp += [f'sweep_{ix[0][1]}']

        for j in range(len(kp_swp)):
            kp_ds = dtree[kp_swp[j]].ds
            rm_ds = dtree[rm_swp[j]].ds
            for f in ['VRADH', 'WRADH']:
                f_data = rm_ds[f].interp(
                            azimuth=kp_ds.azimuth.values,
                            range=kp_ds.range.values,
                            method='nearest'
                        )
                dtree[kp_swp[j]][f] = f_data
            dtree = dtree.drop_nodes(rm_swp[j])

    sweeps, fixed_angles = _get_sweeps_fixed_angles(dtree)
    index = np.argsort(fixed_angles)
    sweeps_new = np.array(['sweep_' + str(i) for i in index])
    sweeps_name = dict(zip(sweeps, sweeps_new))
    dtree_name = np.array(list(dtree.keys()))
    dtree_keep = np.setdiff1d(dtree_name, sweeps)
    new_tree = xr.DataTree()
    for n in dtree_keep:
        new_tree[n] = dtree[n]
    
    for old, new in sweeps_name.items():
        new_tree[new] = dtree[old]

    new_tree.attrs = dtree.attrs
    return new_tree

def _get_sweeps_fixed_angles(dtree):
    sweeps = xd.util.get_sweep_keys(dtree)
    fixed_angles = []
    for s in sweeps:
       fixed_angles += [dtree[s]['sweep_fixed_angle'].values]

    return np.array(sweeps), np.array(fixed_angles)


