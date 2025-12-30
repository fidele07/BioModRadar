import numpy as np
import pyart

def grid_radar_data(radar, fields, gridding_algo='map_gates_to_grid',
                   weighting_function='Nearest', min_radius=200.,
                   x_lim=(-250000.,250000.), y_lim=(-250000.,250000.), z_lim=(0., 5000.),
                   horizontal_res=500., vertical_res=250., **kwargs):
    def compute_number_of_points(extent, resolution):
        return int((extent[1] - extent[0])/resolution) + 1

    z_grid_points = compute_number_of_points(z_lim, vertical_res)
    x_grid_points = compute_number_of_points(x_lim, horizontal_res)
    y_grid_points = compute_number_of_points(y_lim, horizontal_res)

    if type(fields) is not list:
        fields = [fields]

    grid = pyart.map.grid_from_radars(radar, fields=fields,
                grid_shape=(z_grid_points, y_grid_points, x_grid_points),
                grid_limits=(z_lim, y_lim, x_lim),
                gridding_algo=gridding_algo,
                weighting_function=weighting_function,
                min_radius=min_radius, **kwargs)
    # grid.time['data'] = np.array([radar.time['data'][-1]])
    return grid
