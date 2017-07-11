import numpy as np
from ..handler.images import ImageObj

# Method collection for dynamic analysis
def seed_coords(tractobj, start_point, end_point):
    data = tractobj.dataobj
    coords = np.argwhere(data == 1)
    coords = map(list, coords)
    coords.remove(start_point)
    seed_coords = []

    x, y, z = start_point
    cubic = data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
    cubic_corrds = np.argwhere(cubic == 1)
    cubic_results = []

    for dx, dy, dz in cubic_corrds:
        cubic_results.append([x + dx - 1, y + dy - 1, z + dz - 1])
    for cubic_corrd in cubic_results:
        if list(cubic_corrd) in (start_point, end_point):
            pass
        else:
            seed_coords.append(cubic_corrd)

    for i in range(len(coords)):
        x, y, z = seed_coords[-1]
        cubic = data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        cubic_corrds = np.argwhere(cubic == 1)
        cubic_results = []

        for dx, dy, dz in cubic_corrds:
            cubic_results.append([x + dx - 1, y + dy - 1, z + dz - 1])
        for cubic_corrd in cubic_results:
            if list(cubic_corrd) in seed_coords:
                pass
            else:
                seed_coords.append(cubic_corrd)
    return seed_coords


def gen_travel_seed(tractobj, start_point, end_point, filename=None):
    seed_crds = seed_coords(tractobj, start_point, end_point)
    shape = list(tractobj.shape[:])
    shape.append(len(seed_crds))
    data = np.zeros(shape, np.int16)
    for i, coord in enumerate(seed_crds):
        x, y, z = seed_crds
        data[x, y, z, i] = 1
        data[x, y, z + 1, i] = 1
        data[x + 1, y, z, i] = 1
        data[x + 1, y, z + 1, i] = 1
    travelseed_obj = ImageObj(data, tractobj.affine)
    if filename:
        travelseed_obj.to_filename(filename)
    return travelseed_obj