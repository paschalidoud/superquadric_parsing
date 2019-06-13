import numpy as np
import torch

from .primitives import transform_to_primitives_centric_system,\
    inside_outside_function, points_to_cuboid_distances


def broadcast_cross(a, b):
    a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
    b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2]

    s1 = a2*b3 - a3*b2
    s2 = a3*b1 - a1*b3
    s3 = a1*b2 - a2*b1

    return torch.stack([s1, s2, s3], dim=-1)


def inside_mesh(points, triangles):
    """Return a boolean mask that is true for all the points that are inside
    the mesh defined by the triangles.

    Arguments
    ---------
        points: array Nx3 the 3d points
        triangles: array Mx3x3 the 3 3d points defining each of the M triangles
    """
    # Establish a point that is outside the mesh
    out = (triangles.min(0)[0].min(0)[0] - 1).reshape(1, 3)

    # Get a normalized ray direction
    rays = points - out
    ray_norms = torch.sqrt((rays**2).sum(dim=1, keepdim=True))
    rays /= ray_norms

    # Calculate the edges that share point 0 of the triangle
    edges1 = triangles[:, 1] - triangles[:, 0]
    edges2 = triangles[:, 2] - triangles[:, 0]

    # Start the calculation of the determinant (see Moeller & Trumbore 1997)
    pvec = broadcast_cross(rays.reshape(1, -1, 3), edges2.reshape(-1, 1, 3))
    dets = (edges1.reshape(-1, 1, 3) * pvec).sum(dim=-1)
    inv_dets = 1.0 / dets

    # Calculate U
    tvec = out - triangles[:, 0]
    u = (tvec.reshape(-1, 1, 3) * pvec).sum(dim=-1) * inv_dets

    # Calculate V
    qvec = torch.cross(tvec, edges1)
    v = (rays.reshape(1, -1, 3) * qvec.reshape(-1, 1, 3)).sum(dim=-1)
    v = v * inv_dets

    # Compute all intersections
    intersections = (u > 0) * (v > 0) * ((u + v) < 1)

    # Compute the lengths of the intersections
    ts = (edges2 * qvec).sum(dim=1).reshape(-1, 1) * inv_dets

    # We need points that starting from `out` the line intersects an odd number
    # of triangles.
    mask = ((intersections * (ts <= ray_norms.reshape(1, -1))).sum(dim=0) % 2)
    mask = mask == 1

    return mask


def _test_inside_mesh():
    # Firstly lets make a cube and an in cube function
    from scipy.spatial import ConvexHull

    def in_cube(P):
        def b(p):
            return (0.3 < p) * (p < 0.6)
        return b(P[:, 0]) * b(P[:, 1]) * b(P[:, 2])
    cube = torch.FloatTensor([
               [0.3, 0.3, 0.3],
               [0.3, 0.3, 0.6],
               [0.3, 0.6, 0.3],
               [0.3, 0.6, 0.6],
               [0.6, 0.3, 0.3],
               [0.6, 0.3, 0.6],
               [0.6, 0.6, 0.3],
               [0.6, 0.6, 0.6]
           ])
    h = ConvexHull(cube.numpy())
    triangles = cube[h.simplices.ravel()].reshape(12, 3, 3)

    # Random points in [0, 1]^3
    P = torch.rand(100000, 3)

    mask1 = in_cube(P)
    mask2 = inside_mesh(P, triangles)

    # The ratio of the volume of the unit cube to the 0.3^3 cube is 0.027.
    d1 = 0.027 - mask1.sum().item() / 100000
    d2 = 0.027 - mask2.sum().item() / 100000

    assert -0.0005 < d1 < 0.0005
    assert -0.0005 < d2 < 0.0005


def inside_sqs(P, y_hat, use_cuboids=False, use_sq=False, prob_threshold=0.5):
    """
    Arguments:
    ---------
        P: Tensor with size BxNx3, with N points
        y_hat: List of Tensors containing the predictions of the network
        use_cuboids: when True use cuboids as primitives
        use_sq: when True use SQ as primitives
    Returns:
        M: Boolean mask with points are inside the SQs
    """
    # Make sure that everything has the right shape
    assert P.shape[-1] == 3

    # Declare some variables
    B = P.shape[0]  # batch size
    N = P.shape[1]  # number of points per sample
    M = y_hat[0].shape[1]  # number of primitives

    probs = y_hat[0]
    translations = y_hat[1].view(B, M, 3)
    rotations = y_hat[2].view(B, M, 4)
    shapes = y_hat[3].view(B, M, 3)
    epsilons = y_hat[4].view(B, M, 2)

    # Transform the 3D points from world-coordinates to primitive-centric
    # coordinates
    X_transformed = transform_to_primitives_centric_system(
        P,
        translations,
        rotations
    )
    assert X_transformed.shape == (B, N, M, 3)
    if use_sq:
        F = inside_outside_function(
            X_transformed,
            shapes,
            epsilons
        )
        inside = F <= 1
    elif use_cuboids:
        F = points_to_cuboid_distances(X_transformed, shapes)
        inside = F <= 0

    probs_mask = probs.unsqueeze(1) > prob_threshold
    assert inside.shape == (B, N, M)
    inside = inside * probs_mask

    # For every row if a column is 1.0 then that point is inside the SQs
    return inside.any(dim=-1)
