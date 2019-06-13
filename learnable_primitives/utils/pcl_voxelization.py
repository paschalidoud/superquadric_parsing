import numpy as np


def get_voxel_grid(bbox, grid_shape):
    """Given a bounding box and the dimensionality of a grid generate a grid of
    voxels and return their centers.

    Arguments:
    ----------
        bbox: array(shape=(6, 1), dtype=np.float32)
              The min and max of the corners of the bbox that encloses the
              scene
        grid_shape: array(shape(3,), dtype=int32)
                    The dimensions of the voxel grid used to discretize the
                    scene
    Returns:
    --------
        voxel_grid: array(shape=(3,)+grid_shape)
                    The centers of the voxels
    """
    # Make sure that we have the appropriate inputs
    assert bbox.shape[0] == 6
    assert bbox.shape[1] == 1

    xyz = [
        np.linspace(s, e, c, endpoint=True, dtype=np.float32)
        for s, e, c in
        zip(bbox[:3], bbox[3:], grid_shape)
    ]
    bin_size = np.array([xyzi[1]-xyzi[0] for xyzi in xyz]).reshape(3, 1, 1, 1)
    return np.stack(np.meshgrid(*xyz, indexing="ij")) + bin_size/2
