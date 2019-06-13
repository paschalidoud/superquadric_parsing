import numpy as np

import torch

from .primitives import fexp, cuboid_inside_outside_function, \
    inside_outside_function, points_to_cuboid_distances, \
    transform_to_primitives_centric_system, deform, sq_volumes
from .regularizers import get as get_regularizer


def sampling_from_parametric_space_to_equivalent_points(
    shape_params,
    epsilons,
    sq_sampler
):
    """
    Given the sampling steps in the parametric space, we want to ge the actual
    3D points.

    Arguments:
    ----------
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives
        epsilons: Tensor with size BxMx2, containing the shape along the
                  latitude and the longitude for the M primitives

    Returns:
    ---------
        P: Tensor of size BxMxSx3 that contains S sampled points from the
           surface of each primitive
        N: Tensor of size BxMxSx3 that contains the normals of the S sampled
           points from the surface of each primitive
    """
    # Allocate memory to store the sampling steps
    B = shape_params.shape[0]  # batch size
    M = shape_params.shape[1]  # number of primitives
    S = sq_sampler.n_samples

    etas, omegas = sq_sampler.sample_on_batch(
        shape_params.detach().cpu().numpy(),
        epsilons.detach().cpu().numpy()
    )
    # Make sure we don't get nan for gradients
    etas[etas == 0] += 1e-6
    omegas[omegas == 0] += 1e-6

    # Move to tensors
    etas = shape_params.new_tensor(etas)
    omegas = shape_params.new_tensor(omegas)

    # Make sure that all tensors have the right shape
    a1 = shape_params[:, :, 0].unsqueeze(-1)  # size BxMx1
    a2 = shape_params[:, :, 1].unsqueeze(-1)  # size BxMx1
    a3 = shape_params[:, :, 2].unsqueeze(-1)  # size BxMx1
    e1 = epsilons[:, :, 0].unsqueeze(-1)  # size BxMx1
    e2 = epsilons[:, :, 1].unsqueeze(-1)  # size BxMx1

    x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
    y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
    z = a3 * fexp(torch.sin(etas), e1)

    # Make sure we don't get INFs
    # x[torch.abs(x) <= 1e-9] = 1e-9
    # y[torch.abs(y) <= 1e-9] = 1e-9
    # z[torch.abs(z) <= 1e-9] = 1e-9
    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

    # Compute the normals of the SQs
    nx = (torch.cos(etas)**2) * (torch.cos(omegas)**2) / x
    ny = (torch.cos(etas)**2) * (torch.sin(omegas)**2) / y
    nz = (torch.sin(etas)**2) / z

    return torch.stack([x, y, z], -1), torch.stack([nx, ny, nz], -1)


def sample_uniformly_from_cubes_surface(shape_params, epsilons, sampler):
    """
    Given the sampling steps in the parametric space, we want to ge the actual
    3D points on the surface of the cube.

    Arguments:
    ----------
        shape_params: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives

    Returns:
    ---------
        P: Tensor of size BxMxSx3 that contains S sampled points from the
           surface of each primitive
    """
    # TODO: Make sure that this is the proper way to do this!
    # Check the device of the angles and move all the tensors to that device
    device = shape_params.device

    # Allocate memory to store the sampling steps
    B = shape_params.shape[0]  # batch size
    M = shape_params.shape[1]  # number of primitives
    S = sampler.n_samples
    N = S/6

    X_SQ = torch.zeros(B, M, S, 3).to(device)

    for b in range(B):
        for m in range(M):
            x_max = shape_params[b, m, 0]
            y_max = shape_params[b, m, 1]
            z_max = shape_params[b, m, 2]
            x_min = -x_max
            y_min = -y_max
            z_min = -z_max

            X_SQ[b, m] = torch.stack([
                torch.stack([
                    torch.ones((N, 1)).to(device)*x_min,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.ones((N, 1)).to(device)*x_max,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.ones((N, 1)).to(device)*y_min,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.ones((N, 1)).to(device)*y_max,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.ones((N, 1)).to(device)*z_min,
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.ones((N, 1)).to(device)*z_max,
                ], dim=-1).squeeze()
            ]).view(-1, 3)

    normals = X_SQ.new_zeros(X_SQ.shape)
    normals[:, :, 0*N:1*N, 0] = -1
    normals[:, :, 1*N:2*N, 0] = 1
    normals[:, :, 2*N:3*N, 1] = -1
    normals[:, :, 3*N:4*N, 1] = 1
    normals[:, :, 4*N:5*N, 2] = -1
    normals[:, :, 5*N:6*N, 2] = 1

    # make sure that X_SQ has the expected shape
    assert X_SQ.shape == (B, M, S, 3)
    return X_SQ, normals


def euclidean_dual_loss(
    y_hat,
    y_target,
    regularizer_terms,
    sampler,
    options
):
    """
    Arguments:
    ----------
        y_hat: List of Tensors containing the predictions of the network
        y_target: Tensor with size BxNx6 with the N points from the target
                  object and their corresponding normals
        regularizer_terms: dictionary with the various regularizers, on the
                           volume of the primitives, the Bernoullis etc.
        sampler: An object of either CuboidSampler or EqualDistanceSampler
                 depending on the type of the primitive we are using
        options: A dictionary with various options

    Returns:
    --------
        the loss
    """
    # If use_cuboids is true then use 3D cuboids as geometric primitives. If
    # use_sq is true use SQs as geometric primitives. If none of the above is
    # true the default geometric primitive is cuboidal superquadrics, namely
    # SQs with \epsilon_1=\epsilon_2=0.25
    use_cuboids = options.get("use_cuboids", False)
    use_sq = options.get("use_sq", False)

    use_chamfer = options.get("use_chamfer", False)
    loss_weights = options.get(
        "loss_weights",
        {"pcl_to_prim_weight": 1.0, "prim_to_pcl_weight": 1.0}
    )

    gt_normals = y_target[:, :, 3:6]
    gt_points = y_target[:, :, :3]
    # Make sure that everything has the right shape
    assert gt_points.shape[-1] == 3

    # Declare some variables
    B = gt_points.shape[0]  # batch size
    N = gt_points.shape[1]  # number of points per sample
    M = y_hat[0].shape[1]  # number of primitives
    S = sampler.n_samples  # number of points sampled from the SQ

    probs = y_hat[0].view(B, M)
    translations = y_hat[1].view(B, M, 3)
    rotations = y_hat[2].view(B, M, 4)
    shapes = y_hat[3].view(B, M, 3)
    epsilons = y_hat[4].view(B, M, 2)
    tapering_params = y_hat[5].view(B, M, 2)

    # Transform the 3D points from world-coordinates to primitive-centric
    # coordinates with size BxNxMx3
    X_transformed = transform_to_primitives_centric_system(
        gt_points,
        translations,
        rotations
    )

    # Based on the shape of the primitive, do the sampling either on the
    # surface of the SQ or on the surface of the cuboid
    if use_cuboids:
        sample_points_on_surface = sample_uniformly_from_cubes_surface
    else:
        sample_points_on_surface =\
            sampling_from_parametric_space_to_equivalent_points

    # Get the coordinates of the sampled points on the surfaces of the SQs,
    # with size BxMxSx3
    X_SQ, normals = sample_points_on_surface(
        shapes,
        epsilons,
        sampler
    )
    X_SQ = deform(X_SQ, shapes, tapering_params)

    # Make the normals unit vectors
    normals_norm = normals.norm(dim=-1).view(B, M, S, 1)
    normals = normals / normals_norm

    # Make sure that everything has the right size
    assert X_SQ.shape == (B, M, S, 3)
    assert normals.shape == (B, M, S, 3)
    assert X_transformed.shape == (B, N, M, 3)
    # Make sure that the normals are unit vectors
    assert torch.sqrt(torch.sum(normals ** 2, -1)).sum() == B*M*S

    # Compute the pairwise Euclidean distances between points sampled on the
    # surface of the SQ (X_SQ) with points sampled on the surface of the target
    # object (X_transformed)
    # In the code we do everything at once, but this comment helps understand
    # what we are actually doing
    # t = X_transformed.permute(0, 2, 1, 3)  # now X_transformed has size
    # BxMxNx3
    # xx_sq = X_sq.unsqueeze(3)  # now xx_sq has size BxMxSx1x3
    # t = t.unsqueeze(2)  # now t has size BxMx1xNx3
    V = (X_SQ.unsqueeze(3) - (X_transformed.permute(0, 2, 1, 3)).unsqueeze(2))
    assert V.shape == (B, M, S, N, 3)
    # Now we can compute the distances from every point in the surface of the
    # SQ to every point on the target object transformed in every
    # primitive-based coordinate system
    # D = torch.sum((xx_sq - t)**2, -1)  # D has size BxMxSxN
    # TODO: Should I add the SQRT, now we are computing the squared distances
    D = torch.sum((V)**2, -1)
    assert D.shape == (B, M, S, N)

    pcl_to_prim, inside, debug_stats = pcl_to_prim_loss(
        [probs, translations, rotations, shapes, epsilons, tapering_params],
        X_transformed,
        D,
        use_cuboids,
        use_sq,
        use_chamfer
    )
    assert inside is None or inside.shape == (B, N, M)

    prim_to_pcl = prim_to_pcl_loss(
        y_hat,
        V,
        normals,
        inside,
        D,
        use_chamfer
    )

    # Compute any regularizer terms
    regularizers = get_regularizer_term(
        y_hat,
        debug_stats["F"],
        X_SQ,
        regularizer_terms
    )

    reg_values = get_regularizer_weights(
        regularizers,
        regularizer_terms
    )
    debug_stats["regularizer_terms"] = reg_values
    debug_stats["pcl_to_prim_loss"] = pcl_to_prim
    debug_stats["prim_to_pcl_loss"] = prim_to_pcl

    # Sum up the regularization terms
    regs = sum(reg_values.values())
    w1 = loss_weights["pcl_to_prim_weight"]
    w2 = loss_weights["prim_to_pcl_weight"]
    return w1 * pcl_to_prim + w2 * prim_to_pcl + regs, debug_stats


def pcl_to_prim_loss(
    y_hat,
    X_transformed,
    D,
    use_cuboids=False,
    use_sq=False,
    use_chamfer=False
):
    """
    Arguments:
    ----------
        y_hat: List of Tensors containing the predictions of the network
        X_transformed: Tensor with size BxNxMx3 with the N points from the
                       target object transformed in the M primitive-centric
                       coordinate systems
        D: Tensor of size BxMxSxN that contains the pairwise distances between
           points on the surface of the SQ to the points on the target object
        use_cuboids: when True use cuboids as geometric primitives
        use_sq: when True use superquadrics as geometric primitives
        use_chamfer: when True compute the Chamfer distance
    """
    # Declare some variables
    B = X_transformed.shape[0]  # batch size
    N = X_transformed.shape[1]  # number of points per sample
    M = X_transformed.shape[2]  # number of primitives

    shapes = y_hat[3].view(B, M, 3)
    epsilons = y_hat[4].view(B, M, 2)
    probs = y_hat[0]

    # Get the relative position of points with respect to the SQs using the
    # inside-outside function
    F = shapes.new_tensor(0)
    inside = None
    if not use_chamfer:
        if use_cuboids:
            F = points_to_cuboid_distances(X_transformed, shapes)
            inside = F <= 0
        elif use_sq:
            F = inside_outside_function(
                X_transformed,
                shapes,
                epsilons
            )
            inside = F <= 1
        else:
            # If no argument is given (use_sq and use_cuboids) the default
            # geometric primitives are cuboidal superquadrics, namely
            # with \epsilon_1=\epsilon_2=0.25
            F = cuboid_inside_outside_function(
                X_transformed,
                shapes,
                epsilon=0.25
            )
            inside = F <= 1

    D = torch.min(D, 2)[0].permute(0, 2, 1)  # size BxNxM
    assert D.shape == (B, N, M)

    if not use_chamfer:
        D[inside] = 0.0
    distances, idxs = torch.sort(D, dim=-1)

    # Start by computing the cumulative product
    # Sort based on the indices
    probs = torch.cat([
        probs[i].take(idxs[i]).unsqueeze(0) for i in range(len(idxs))
    ])
    neg_cumprod = torch.cumprod(1-probs, dim=-1)
    neg_cumprod = torch.cat(
        [neg_cumprod.new_ones((B, N, 1)), neg_cumprod[:, :, :-1]],
        dim=-1
    )

    # minprob[i, j, k] is the probability that for sample i and point j the
    # k-th primitive has the minimum loss
    minprob = probs.mul(neg_cumprod)

    loss = torch.einsum("ijk,ijk->", [distances, minprob])
    loss = loss / B / N

    # Return some debug statistics
    debug_stats = {}
    debug_stats["F"] = F
    debug_stats["distances"] = distances
    debug_stats["minprob"] = minprob
    debug_stats["neg_cumprod"] = neg_cumprod
    return loss, inside, debug_stats


def prim_to_pcl_loss(
    y_hat,
    V,
    normals,
    inside,
    D,
    use_chamfer=False
):
    """
    Arguments:
    ----------
        y_hat: List of Tensors containing the predictions of the network
        V: Tensor with size BxMxSxN3 with the vectors from the points on SQs to
           the points on the target's object surface.
        normals: Tensor with size BxMxSx3 with the normals at every sampled
                 points on the surfaces of the M primitives
        inside: A mask containing 1 if a point is inside the corresponding
                shape
        D: Tensor of size BxMxSxN that contains the pairwise distances between
           points on the surface of the SQ to the points on the target object
    """
    B = V.shape[0]  # batch size
    M = V.shape[1]  # number of primitives
    S = V.shape[2]  # number of points sampled on the SQ
    N = V.shape[3]  # number of points sampled on the target object
    probs = y_hat[0]

    assert D.shape == (B, M, S, N)

    # We need to compute the distance to the closest point from the target
    # object for every point S
    # min_D = D.min(-1)[0] # min_D has size BxMxS
    if not use_chamfer:
        outside = (1-inside).permute(0, 2, 1).unsqueeze(2).float()
        assert outside.shape == (B, M, 1, N)
        D = D + (outside*1e30)
    # Compute the minimum distances D, with size BxMxS
    D = D.min(-1)[0]
    D[D >= 1e30] = 0.0
    assert D.shape == (B, M, S)

    # Compute an approximate area of the superellipsoid as if it were an
    # ellipsoid
    shapes = y_hat[3].view(B, M, 3)
    area = 4 * np.pi * (
        (shapes[:, :, 0] * shapes[:, :, 1])**1.6 / 3 +
        (shapes[:, :, 0] * shapes[:, :, 2])**1.6 / 3 +
        (shapes[:, :, 1] * shapes[:, :, 2])**1.6 / 3
    )**0.625
    area = M * area / area.sum(dim=-1, keepdim=True)

    # loss = torch.einsum("ij,ij,ij->", [torch.max(D, -1)[0], probs, volumes])
    # loss = torch.einsum("ij,ij,ij->", [torch.mean(D, -1), probs, volumes])
    # loss = torch.einsum("ij,ij->", [torch.max(D, -1)[0], probs])
    loss = torch.einsum("ij,ij,ij->", [torch.mean(D, -1), probs, area])
    loss = loss / B / M

    return loss


def get_regularizer_term(
    parameters,
    F,
    X_SQ,
    regularizer_terms,
    transition_matrix=None
):
    regularizers = [
        "sparsity_regularizer",
        "bernoulli_regularizer",
        "entropy_bernoulli_regularizer",
        "parsimony_regularizer",
        "overlapping_regularizer"
    ]
    if regularizer_terms["regularizer_type"] is None:
        regularizer_terms["regularizer_type"] = []

    return {
        r: get_regularizer(
            r if r in regularizer_terms["regularizer_type"] else "",
            parameters,
            F,
            X_SQ,
            regularizer_terms
        )
        for r in regularizers
    }


def get_regularizer_weights(regularizers, regularizer_terms):
    # Ensures that the expected number of primitives lies between a minimum and
    # a maximum number of primitives.
    bernoulli_reg = regularizers["bernoulli_regularizer"] *\
        regularizer_terms["bernoulli_regularizer_weight"]
    # Ensures that the bernoullis will be either 1.0 or 0.0 and not 0.5
    entropy_bernoulli_reg = regularizers["entropy_bernoulli_regularizer"] *\
        regularizer_terms["entropy_bernoulli_regularizer_weight"]
    # Minimizes the expected number of primitives
    parsimony_reg = regularizers["parsimony_regularizer"] *\
        regularizer_terms["parsimony_regularizer_weight"]
    # Ensures that primitves do not intersect with each other using the F
    # function
    overlapping_reg = regularizers["overlapping_regularizer"] *\
        regularizer_terms["overlapping_regularizer_weight"]
    # Similar to the bernoulli_regularizer. Again we want to ensure that the
    # expected number of primitives will be between a minimum an a maximum
    # number of primitives.
    sparsity_reg = regularizers["sparsity_regularizer"] *\
        regularizer_terms["sparsity_regularizer_weight"]

    reg_values = {
        "sparsity_regularizer": sparsity_reg,
        "overlapping_regularizer": overlapping_reg,
        "parsimony_regularizer": parsimony_reg,
        "entropy_bernoulli_regularizer": entropy_bernoulli_reg,
        "bernoulli_regularizer": bernoulli_reg
    }

    return reg_values
