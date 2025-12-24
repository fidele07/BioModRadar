import os
import gzip
import numpy as np
import xarray as xr
import xradar as xd

xr.set_options(use_new_combine_kwarg_defaults=True)

def read_xradar_data(file_path, sweeps=None,
                     volume_type='cfradial',
                     fields_dict=None):
    # sweeps = [0, 1, 2, 3, 4, 5]
    # fields_dict = {'ref': 'DBZH', 'zdr': 'ZDR', 'rho': 'RHOHV',
    #                'phi': 'PHIDP', 'vel': 'VRADH', 'sw': 'WRADH',
    #                'kdp': 'KDP'}
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

    dtree = _xradar_order_sweeps(dtree)
    dtree = _xradar_extract_sweeps(dtree, sweeps)
    if fields_dict is not None:
        dtree = _xradar_extract_fields(dtree, fields_dict)
    return dtree

def _reduce_sweeps_rwanda_odim_hdf5(dtree):
    dtree = dtree.copy()
    sweeps, fixed_angles = _get_sweeps_fixed_angles(dtree)
    v_fixed_angles = np.array([32., 22., 15., 11., 8., 6.,
                               4.5, 3.5, 2.5, 1.5, 0.5])
    n_swp = len(v_fixed_angles)
    if not np.array_equal(fixed_angles, v_fixed_angles):
        new_sweeps = []
        ix_sweeps = []
        for x in v_fixed_angles:
            ix = np.where(fixed_angles == x)
            if len(ix[0]) == 0:
                raise Exception('Invalid fixed angles.')
            ix = ix[0][0]
            ix_sweeps += [ix]
            new_sweeps += [sweeps[ix]]
        new_sweeps = np.array(new_sweeps)
        rm_sweeps = np.setdiff1d(sweeps, new_sweeps)
        dtree = dtree.drop_nodes(rm_sweeps)

        dtree_name = np.array(list(dtree.keys()))
        new_tree = xr.DataTree()

        for n in dtree_name:
            if 'sweep' in dtree[n].dims:
                node_dt = dtree[n].to_dataset()
                if 'sweep' in node_dt.dims:
                    new_tree[n] = dtree[n].isel(sweep=slice(0, n_swp))
                else:
                    new_tree[n] = xr.DataTree(dataset=node_dt, name=n)
            else:
                new_tree[n] = dtree[n]

    new_tree.attrs = dtree.attrs
    return new_tree

def _reduce_sweeps_nexrad_level2(dtree):
    dtree = dtree.copy()
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
    return dtree

def _get_sweeps_fixed_angles(dtree):
    sweeps = xd.util.get_sweep_keys(dtree)
    fixed_angles = []
    for s in sweeps:
       fixed_angles += [dtree[s]['sweep_fixed_angle'].values]

    return np.array(sweeps), np.array(fixed_angles)

def _xradar_order_sweeps(dtree):
    sweeps, fixed_angles = _get_sweeps_fixed_angles(dtree)
    index = np.argsort(fixed_angles)
    sweeps_new = np.array(['sweep_' + str(i) for i in index])
    sweeps_name = dict(zip(sweeps, sweeps_new))
    sweeps_number = dict(zip(sweeps, index))
    dtree_name = np.array(list(dtree.keys()))
    dtree_keep = np.setdiff1d(dtree_name, sweeps)
    new_tree = xr.DataTree()

    for n in dtree_keep:
        new_tree[n] = dtree[n]
    
    for old, new in sweeps_name.items():
        new_tree[new] = dtree[old]
        new_tree[new].sweep_number.values = sweeps_number[old]

    new_tree.attrs = dtree.attrs
    if 'sweep_fixed_angle' in list(new_tree.keys()):
        fixed_angles = fixed_angles[index]
        new_tree.sweep_fixed_angle.values = fixed_angles
    return new_tree

def _xradar_extract_sweeps(dtree, sweeps):
    sweeps_old, fixed_angles = _get_sweeps_fixed_angles(dtree)
    nsweeps = len(sweeps_old)
    if sweeps is not None:
        if not isinstance(sweeps, (list, np.ndarray)):
            sweeps = [sweeps]
        if len(sweeps) > nsweeps:
            sweeps = np.arange(nsweeps)
    else:
        sweeps = np.arange(nsweeps)

    sweeps_new = np.array(['sweep_' + str(i) for i in sweeps])
    dtree_name = np.array(list(dtree.keys()))
    dtree_keep = np.setdiff1d(dtree_name, sweeps_old)
    new_tree = xr.DataTree()

    for n in dtree_keep:
        if 'sweep' in dtree[n].dims:
            node_dt = dtree[n].to_dataset()
            if 'sweep' in node_dt.dims:
                new_tree[n] = dtree[n].isel(sweep=sweeps)
            else:
                new_tree[n] = xr.DataTree(dataset=node_dt, name=n)
        else:
            new_tree[n] = dtree[n]

    for new in sweeps_new:
        node_dt = dtree[new].to_dataset()
        new_tree[new] = xr.DataTree(dataset=node_dt, name=new)

    new_tree.attrs = dtree.attrs
    return new_tree

def _xradar_extract_fields(dtree, fields_dict):
    sweeps, fixed_angles = _get_sweeps_fixed_angles(dtree)
    dtree_fields = list(dtree[sweeps[0]].data_vars.keys())
    include_fields = [fields_dict[f] for f in fields_dict]
    include_fields = [f for f in include_fields if f is not None]
    dtree_vars = ['sweep_mode', 'sweep_number', 'prt_mode',
                  'follow_mode', 'sweep_fixed_angle']
    new_tree = dtree
    if len(include_fields) > 0:
        include_fields += dtree_vars
        for fl in include_fields:
            if fl not in dtree_fields:
                raise KeyError(f'No field named "{fl}" found.')
        delete_fields = [fl for fl in dtree_fields if fl not in include_fields]
        if len(delete_fields) > 0:
            new_tree = xr.DataTree()
            for n in list(dtree.keys()):
                if n not in sweeps:
                    new_tree[n] = dtree[n]
            new_tree.attrs = dtree.attrs

            for swp in sweeps:
                node_dt = dtree[swp].to_dataset()
                node_dt = node_dt.drop_vars(delete_fields)
                new_tree[swp] = xr.DataTree(dataset=node_dt, name=swp)
    return new_tree
