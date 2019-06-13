
import numpy as np
import torch


def fexp(x, p):
    return torch.sign(x)*(torch.abs(x)**p)


def cuboid_inside_outside_function(X, shape_params, epsilon=0.25):
    """
    Arguments:
    ----------
        X: Tensor with size BxNxMx3, containing the 3D points, where B is the
           batch size and N is the number of points
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives
        epsilon: int, the shape of the SQ along the latitude and longitude

    Returns:
    ---------
        F: Tensor with size BxNxM, containing the values of the
           inside-outside function
    """
    # Make sure that both tensors have the right shape
    assert X.shape[0] == shape_params.shape[0]  # batch size
    assert X.shape[2] == shape_params.shape[1]  # number of primitives
    assert X.shape[-1] == 3  # 3D points

    # Tensor that holds the values of the inside-outside function
    F = shape_params.new_zeros(X.shape[:-1])
    shape_params = shape_params.unsqueeze(1)
    for i in range(3):
        F += (X[:, :, :, i] / shape_params[:, :, :, i])**(2.0/epsilon)

    return F**(epsilon)


def inside_outside_function(X, shape_params, epsilons):
    """
    Arguments:
    ----------
        X: Tensor with size BxNxMx3, containing the 3D points, where B is the
           batch size and N is the number of points
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives
        epsilons: Tensor with size BxMx2, containing the shape along the
                  longitude and the latitude for the M primitives

    Returns:
    ---------
        F: Tensor with size BxNxM, containing the values of the
           inside-outside function
    """
    B = X.shape[0]  # batch_size
    N = X.shape[1]  # number of points on target object
    M = X.shape[2]  # number of primitives

    # Make sure that both tensors have the right shape
    assert shape_params.shape[0] == B  # batch size
    assert epsilons.shape[0] == B  # batch size
    assert shape_params.shape[1] == M  # number of primitives
    assert shape_params.shape[1] == epsilons.shape[1]
    assert shape_params.shape[-1] == 3  # number of shape parameters
    assert epsilons.shape[-1] == 2  # number of shape parameters
    assert X.shape[-1] == 3  # 3D points

    # Declare some variables
    a1 = shape_params[:, :, 0].unsqueeze(1)  # size Bx1xM
    a2 = shape_params[:, :, 1].unsqueeze(1)  # size Bx1xM
    a3 = shape_params[:, :, 2].unsqueeze(1)  # size Bx1xM
    e1 = epsilons[:, :, 0].unsqueeze(1)  # size Bx1xM
    e2 = epsilons[:, :, 1].unsqueeze(1)  # size Bx1xM

    # Add a small constant to points that are completely dead center to avoid
    # numerical issues in computing the gradient
    # zeros = X == 0
    # X[zeros] = X[zeros] + 1e-6
    X = ((X > 0).float() * 2 - 1) * torch.max(torch.abs(X), X.new_tensor(1e-6))

    F = ((X[:, :, :, 0] / a1)**2)**(1./e2)
    F += ((X[:, :, :, 1] / a2)**2)**(1./e2)
    F = F**(e2 / e1)
    F += ((X[:, :, :, 2] / a3)**2)**(1./e1)

    # Sanity check to make sure that we have the expected size
    assert F.shape == (B, N, M)
    return F**e1
    # return F


def points_to_cuboid_distances(X, shape_params):
    """
    Arguments:
    ----------
        X: Tensor with size BxNxMx3, containing the 3D points, where B is the
           batch size and N is the number of points
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives

    Returns:
    ---------
        F: Tensor with size BxNxM, containing the distances of each point to
           every primitive
    """
    # Make sure that everything has the right size
    assert X.shape[0] == shape_params.shape[0]  # batch size
    assert X.shape[2] == shape_params.shape[1]  # number of primitives
    assert X.shape[-1] == 3  # 3D points

    # The distance between a point (x, y, z) to a cuboid with dimensions
    # (a1, a2, a3) is sqrt(max(0, abs(x) - a1)^2 + max(0, abs(y) - a2)^2 +
    # max(0, abs(z) - a3)^2). Technically, F=0 for all points either inside or
    # on the surface of the primitive, while we only want F=0 for the points on
    # the surface of the cuboid.
    F = (torch.max(
        X.abs() - shape_params.unsqueeze(1),
        torch.zeros_like(X)
    )**2).sum(-1)

    return F


def euler_angles_to_rotation_matrices(angles):
    """
    Arguments:
    ---------
        angles: Tensor with size Kx3, where K is the number of Euler angles we
                want to transform to rotation matrices

    Returns:
    -------
        rotation_matrices: Tensor with size Kx3x3, that contains the computed
                           rotation matrices
    """
    K = angles.shape[0]
    # Allocate memory for a Tensor of size Kx3x3 that will hold the rotation
    # matrix along the x-axis
    r_x = angles.new_zeros((K, 3, 3))
    r_x[:, 0, 0] = 1.0
    c = torch.cos(angles[:, 0])
    s = torch.sin(angles[:, 0])
    r_x[torch.arange(K), 1, 1] = c
    r_x[torch.arange(K), 2, 2] = c
    r_x[torch.arange(K), 1, 2] = -s
    r_x[torch.arange(K), 2, 1] = s

    # Similar for the rotation matrices along the y-axis and z-axis
    r_y = angles.new_zeros((K, 3, 3))
    r_y[:, 1, 1] = 1.0
    c = torch.cos(angles[:, 1])
    s = torch.sin(angles[:, 1])
    r_y[torch.arange(K), 0, 0] = c
    r_y[torch.arange(K), 2, 2] = c
    r_y[torch.arange(K), 2, 0] = -s
    r_y[torch.arange(K), 0, 2] = s

    r_z = angles.new_zeros((K, 3, 3))
    r_z[:, 2, 2] = 1.0
    c = torch.cos(angles[:, 2])
    s = torch.sin(angles[:, 2])
    r_z[torch.arange(K), 0, 0] = c
    r_z[torch.arange(K), 1, 1] = c
    r_z[torch.arange(K), 0, 1] = -s
    r_z[torch.arange(K), 1, 0] = s

    return r_z.bmm(r_y.bmm(r_x))


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size Kx4, where K is the number of quaternions
                     we want to transform to rotation matrices

    Returns:
    -------
        rotation_matrices: Tensor with size Kx3x3, that contains the computed
                           rotation matrices
    """
    K = quaternions.shape[0]
    # Allocate memory for a Tensor of size Kx3x3 that will hold the rotation
    # matrix along the x-axis
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1]**2
    yy = quaternions[:, 2]**2
    zz = quaternions[:, 3]**2
    ww = quaternions[:, 0]**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R


def transform_to_primitives_centric_system(X, translations, rotation_angles):
    """
    Arguments:
    ----------
        X: Tensor with size BxNx3, containing the 3D points, where B is the
           batch size and N is the number of points
        translations: Tensor with size BxMx3, containing the translation
                      vectors for the M primitives
        rotation_angles: Tensor with size BxMx4 containing the 4 quaternion
                         values for the M primitives

    Returns:
    --------
        X_transformed: Tensor with size BxNxMx3 containing the N points
                       transformed in the M primitive centric coordinate
                       systems.
    """
    # Make sure that all tensors have the right shape
    assert X.shape[0] == translations.shape[0]
    assert translations.shape[0] == rotation_angles.shape[0]
    assert translations.shape[1] == rotation_angles.shape[1]
    assert X.shape[-1] == 3
    assert translations.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4

    # Subtract the translation and get X_transformed with size BxNxMx3
    X_transformed = X.unsqueeze(2) - translations.unsqueeze(1)

    # R = euler_angles_to_rotation_matrices(rotation_angles.view(-1, 3)).view(
    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )

    # Let as denote a point x_p in the primitive-centric coordinate system and
    # its corresponding point in the world coordinate system x_w. We denote the
    # transformation from the point in the world coordinate system to a point
    # in the primitive-centric coordinate system as x_p = R(x_w - t)
    X_transformed = R.unsqueeze(1).matmul(X_transformed.unsqueeze(-1))

    X_signs = (X_transformed > 0).float() * 2 - 1
    X_abs = X_transformed.abs()
    X_transformed = X_signs * torch.max(X_abs, X_abs.new_tensor(1e-5))

    return X_transformed.squeeze(-1)


def transform_to_world_coordinates_system(X_SQ, translations, rotation_angles):
    """
    Arguments:
    ----------
        X_SQ: Tensor with size BxMxSx3, containing the 3D points, where B is
              the batch size, M is the number of primitives and S is the number
              of points on each primitive-centric system
        translations: Tensor with size BxMx3, containing the translation
                      vectors for the M primitives
        rotation_angles: Tensor with size BxMx3 containing the 3 Euler angles
                         for the M primitives

    Returns:
    --------
        X_SQ_w: Tensor with size BxMxSx3 containing the N points
                transformed in the M primitive centric coordinate
                systems.
    """
    # Make sure that all tensors have the right shape
    assert X_SQ.shape[0] == translations.shape[0]
    assert translations.shape[0] == rotation_angles.shape[0]
    assert translations.shape[1] == rotation_angles.shape[1]
    assert X_SQ.shape[1] == translations.shape[1]
    assert X_SQ.shape[-1] == 3
    assert translations.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4

    # Compute the rotation matrices to every primitive centric coordinate
    # system (R has size BxMx3x3)
    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )
    # We need the R.T to get the rotation matrix from the primitive-centric
    # coordinate system to the world coordinate system.
    R_t = torch.einsum("...ij->...ji", (R,))
    assert R.shape == R_t.shape

    X_SQ_w = R.unsqueeze(2).matmul(X_SQ.unsqueeze(-1))
    X_SQ_w = X_SQ_w.squeeze(-1) + translations.unsqueeze(2)

    return X_SQ_w


def deform(X, shape_params, tapering_params, bending_params=None):
    """
    Arguments:
    ----------
        X: Tensor with size BxMxSx3 containing the S points
           sampled on the surfaces of each SQ
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives
        tapering_params: Tensor with size BxMx2, containing the tapering_params
                         for every primitive
        bending_params: Tensor with size BxMx2, containing the bending_params
                        for every primitive
    Returns:
    --------
        X_deformed: Tensor with size BxMxSx3 containing the N points
                    transformed in the M primitive centric coordinate
                    systems after the deformations.
    """
    B, M, S, _ = X.shape
    # Make sure that all tensors have the right shape
    assert X.shape[0] == shape_params.shape[0]  # batch size
    assert X.shape[0] == tapering_params.shape[0]  # batch size
    assert shape_params.shape[-1] == 3
    assert tapering_params.shape[-1] == 2
    assert X.shape[1] == shape_params.shape[1]
    assert X.shape[1] == tapering_params.shape[1]

    # Compute the two linear tapering functions
    K = tapering_params / shape_params[:, :, -1].unsqueeze(-1)
    assert tapering_params.shape == K.shape
    f = K.unsqueeze(2) * X[:, :, :, -1].unsqueeze(-1) + 1.0
    assert f.shape == (B, M, S, 2)
    f = torch.cat([
        f,
        f.new_ones(B, M, S, 1)
    ], -1)
    assert f.shape == X.shape
    X_d = X * f
    X_d = apply_bending(X_d, bending_params)

    return X_d


def apply_bending(X, bending_params):
    """
    Arguments:
    ----------
        X: Tensor with size BxMxSx3 containing the S points
           sampled on the surfaces of each SQ
        bending_params: Tensor with size BxMx2, containing the bending_params
                        for every primitive
    Returns:
    --------
        X_d: Tensor with size BxMxSx3 containing the N points
             transformed in the M primitive centric coordinate
             systems after the deformations.
    """
    # If there no bending params specified return the input as is
    if bending_params is None:
        return X

    B, M, S, _ = X.shape
    # Make sure that all tensors have the right shape
    assert X.shape[0] == bending_params.shape[0]  # batch size
    assert bending_params.shape[-1] == 2

    # Apply the bending operation
    bending_params = bending_params.unsqueeze(2)  # BXMX2 -> BxMx1x2
    k = bending_params[:, :, :, 0].unsqueeze(-1)  # BxMx1x1
    a = bending_params[:, :, :, 1].unsqueeze(-1)  # BxMx1x1

    b = torch.atan2(X[:, :, :, 1].unsqueeze(-1), X[:, :, :, 0].unsqueeze(-1))
    assert b.shape == (B, M, S, 1)
    r = torch.sqrt(
        X_d[:, :, :, 0].unsqueeze(-1)**2 + X_d[:, :, :, 1].unsqueeze(-1)**2
    ) * torch.cos(a - b)
    assert r.shape == (B, M, S, 1)
    k_inv = 1 / k  # BxMx1x1
    gamma = X_d[:, :, :, -1].unsqueeze(-1) / k
    R = k_inv - (k_inv - r) * torch.cos(gamma)
    assert R.shape == (B, M, S, 1)

    X_d = X.new_zeros(X.shape)
    X_d[:, :, :, 0] = X_d[:, :, :, 0] + (R - r)*torch.cos(a)
    X_d[:, :, :, 1] = X_d[:, :, :, 1] + (R - r)*torch.sin(a)
    X_d[:, :, :, 2] = (k_inv - r)*(R - r)*torch.sin(gamma)

    return X_d


def distance(F, shape_params=None, use_chamfer=False):
    """
    Arguments:
    ----------
        F: Tensor of size BxNxM, with the values of the inside-outside function
           for the N points w.r.t. the M primitives
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives
    Returns:
    --------
        C: Tensor of size BxNxM, with the distance between points and
           primitives
        primitive_idxs: Tensor of size BxNxM, with the indices of the
                        primitives in the original tensor F
    """
    # Minimization of the distances between points and primitives
    if use_chamfer:
        C = (F-1.0)**2.0
    else:
        a1a2a3 = torch.sqrt(shape_params.prod(-1)).unsqueeze(1)
        # C = torch.max(a1a2a3*(F - 1.0), torch.zeros_like(F))
        # C = torch.max(torch.sqrt(F) - 1.0, torch.zeros_like(F))
        C = torch.max((F - 1.0), torch.zeros_like(F))

    return torch.sort(C, dim=-1)


def ray_plane_intersections(P, V, normals, exp1, exp2):
    """
    Find the interesection between a set of rays and a set of planes. Rays are
    defined as two points and normals as points and planes.

    We we want to compute
    rs = n (Vo - Po)
         -----------
         n (P1 - Po)
    n and Vo define the plane and Po and P1 the ray

    Arguments:
    ----------
        P: Tensor of size BxMx?x3 containing the start of each ray (P1 - Po)
        V: Tensor of size BxMxSxNx3 with the differences between the ray_starts
           and the points of the planes (Vo - Po)
        normals: Tensor of size BxMx?x3 N normals transformed in the M
           primitive-centric coordinate systems
    Returns:
    --------
        r: Tensor of size BxMxSxN with the squared_distances
    """
    B, M, S, N, _ = V.shape

    t1 = torch.einsum(exp1, [normals, V])
    t2 = torch.einsum(exp2, [normals, P])
    rs = torch.div(t1, t2)
    assert rs.shape == (B, M, S, N)

    return torch.pow(rs, 2)


def beta_stirling(x, y):
    sqrt2pi = float(np.sqrt(2*np.pi))
    return sqrt2pi * (x**(x-0.5) * y**(y-0.5)) / (x+y)**(x+y-0.5)


def sq_volumes(parameters):
    a1a2a3 = parameters[3].view(-1, 3).prod(-1)
    e = parameters[4].view(-1, 2)
    e1 = e[:, 0]
    e2 = e[:, 1]
    e1e2 = e.prod(-1)
    b1 = beta_stirling(e1/2 + 1, e1)
    b2 = beta_stirling(e2/2, e2/2)
    volumes = 2 * a1a2a3 * e1e2 * b1 * b2
    return volumes


def sq_areas(shapes, epsilons):
    """Approximate area of the superquadric.

    We use Knud Thomsen's formula for ellipsoids.
    """
    p = 1.6075
    a = shapes[:, :, 0]
    b = shapes[:, :, 1]
    c = shapes[:, :, 2]

    return 4 * np.pi * (((a*b)**p + (a*c)**p + (b*c)**p)/3)**(1/p)


def sample_points_inside_primitives(X_SQ, N, rotations, translations):
    """Sample points inside the primitives, given S points on their surface

    Arguments:
    ----------
        X_SQ: Tensor of size BxMxSx3 containing S points sampled on the surface
              of each primitive
        rotations: Tensor of size BxMx4 containing the quaternions of the SQs
        translations: Tensor of size BxMx4 containing the translation vectors
                      of the SQs
        N: number of points to be generated internally in each primitive

    Returns:
    --------
        X_world: Tensor of size BxMxNx3 containing N points sampled uniformly
                 inside and on the surface of the SQs
    """
    B, M, S, _ = X_SQ.shape
    assert rotations.shape == (B, M, 4)
    assert translations.shape == (B, M, 3)

    # Create points inside the primitives
    device = X_SQ.device
    batch = (torch.arange(B*M*N) / (M*N)).view(B, M, N).to(device)
    prim = ((torch.arange(B*M*N) / N) % M).view(B, M, N).to(device)
    pointsA = torch.randint(0, S, (B, M, N), dtype=torch.long).to(device)
    pointsB = torch.randint(0, S, (B, M, N), dtype=torch.long).to(device)
    t = torch.rand(B, M, N, 1).to(device)
    X_a = X_SQ[batch, prim, pointsA]
    X_b = X_SQ[batch, prim, pointsB]
    X = X_a + t * (X_b-X_a)

    # Transform the points to world coordinates
    # R = quaternions_to_rotation_matrices(rotations.view(-1, 4))
    # R = R.view(B, M, 3, 3)
    # X_world = X.view(B, M, N, 1, 3).matmul(R.view(B, M, 1, 3, 3))
    # X_world = X_world.view(B, M, N, 3)
    # X_world = X_world + translations.view(B, M, 1, 3)
    X_world = transform_to_world_coordinates_system(
        X,
        translations,
        rotations
    )
    assert X_world.shape == (B, M, N, 3)

    return X_world
