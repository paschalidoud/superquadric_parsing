#!/usr/bin/env python

import os

import numpy as np
import torch
from pyquaternion import Quaternion

from shapes import Shape, Cuboid
from learnable_primitives.loss_functions import\
    euclidean_dual_loss
from learnable_primitives.equal_distance_sampler_sq import\
    EqualDistanceSamplerSQ
from learnable_primitives.primitives import\
    euler_angles_to_rotation_matrices, quaternions_to_rotation_matrices
from learnable_primitives.mesh import MeshFromOBJ

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
rc('text', usetex=True)
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc("font", size=10, family="serif")

def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)

def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z


def get_sq(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)

    # Get an array of size 3x10000 that contains the points of the SQ
    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t
    # print "R:", R
    # print "t:", t
    # print "e:", [e1, e2]

    x_tr = points_transformed[0].reshape(n_samples, n_samples)
    y_tr = points_transformed[1].reshape(n_samples, n_samples)
    z_tr = points_transformed[2].reshape(n_samples, n_samples)

    return x_tr, y_tr, z_tr, points_transformed


def cubes_inside(x_1, y_1, z_1, x_2, y_2, z_2):
    # cube 1
    x_min = -x_1
    x_max = x_1
    y_min = -y_1
    y_max = y_1
    z_min = -z_1
    z_max = z_1
    c1 = Cuboid(x_min, x_max, y_min, y_max, z_min, z_max)
    c1.translate(np.array([[0.2, 0.2, 0.0]]).T)

    # cube 2
    x_min = -x_2
    x_max = x_2
    y_min = -y_2
    y_max = y_2
    z_min = -z_2
    z_max = z_2
    c2 = Cuboid(x_min, x_max, y_min, y_max, z_min, z_max)

    # Denote some useful variables
    B = 1  # batch size
    M = 2  # number of primitives
    shapes = torch.zeros(B, M*3, dtype=torch.float, requires_grad=True)
    shapes[0, 0] = 0.1
    shapes[0, 1] = 0.1
    shapes[0, 2] = z_1
    shapes[0, 3] = 0.1
    shapes[0, 4] = 0.1
    shapes[0, 5] = z_2
    # probs
    probs = torch.ones(B, M, dtype=torch.float, requires_grad=True)
    translations = torch.zeros(B, M*3, dtype=torch.float, requires_grad=True)
    translations[0, 0] = 0.2
    translations[0, 1] = 0.6
    translations[0, 3:5] = -0.5
    #translations = 0.5 - torch.randn(B, M*3, dtype=torch.float, requires_grad=True)
    quaternions = torch.zeros(B, M*4, dtype=torch.float, requires_grad=True)
    quaternions[0, 0] = 1.0
    quaternions[0, 4] = 1.0
    #for i in range(2):
    #    q = Quaternion.random()
    #    for j in range(4):
    #        quaternions[0, 4*i+j] = q[j]

    epsilons = torch.ones(B, M*2, dtype=torch.float, requires_grad=True) * 0.25

    y_hat = [probs, translations, quaternions, shapes, epsilons]
    y_hat = [
        torch.tensor(yi.data, requires_grad=True)
        for yi in y_hat
    ]

    return c1, c2, y_hat


def get_translation(shapes):
    p_min = np.min([
        s.points.min(axis=1)
        for s in shapes
    ], axis=0)
    p_max = np.max([
        s.points.max(axis=1)
        for s in shapes
    ], axis=0)

    return (p_max - p_min) / 2 + p_min


def get_rectangle(cube, T, **kwargs):
    corner1 = cube.points[:, 0]
    corner2 = cube.points[:, -1]

    return Rectangle(
        corner1-T,
        *(corner2-corner1),
        **kwargs
    )


if __name__ == "__main__":
    c1, c2, y_hat = cubes_inside(0.2, 0.2, 0.1, 0.1, 0.1, 0.1)
    T = get_translation([c1, c2])
    probs, translations, quats, shapes, epsilons = y_hat
    c = Shape.from_shapes([c1, c2])
    c.save_as_mesh("/tmp/mesh.obj", "obj")
    m = MeshFromOBJ("/tmp/mesh.obj")
    y_target = torch.from_numpy(
        m.sample_faces(1000).astype(np.float32)
    ).float().unsqueeze(0)

    # A sampler instance
    e = EqualDistanceSamplerSQ(200)
    # Compute the loss for the current experiment
    l_weights = {
        "coverage_loss_weight": 1.0,
        "consistency_loss_weight": 1.0,
    }
    reg_terms = {
        "regularizer_type": [],
        "shapes_regularizer_weight": 0.0,
        "bernoulli_with_sparsity_regularizer_weight": 0.0,
        "bernoulli_regularizer_weight": 0.0,
        "entropy_bernoulli_weight": 0.0,
        "partition_regularizer_weight": 0.0,
        "parsimony_regularizer_weight": 0.0,
        "overlapping_regularizer_weight": 0.0,
        "minimum_number_of_primitives": 0.0
    }

    
    use_simple_cuboids = False
    use_sq = False
    lr = 1e-1  # learning rate
    n_iters = 300 # number of gradient_descent iterations
    optim = torch.optim.SGD([translations, shapes], lr=lr, momentum=0.9)
    for k in range(n_iters):
        optim.zero_grad()
        loss, debug_stats = euclidean_dual_loss(
            y_hat,
            y_target,
            reg_terms,
            e,
            l_weights,
            use_simple_cuboids,
            use_sq,
            False
        )
        loss.backward()
        print "It: %d - loss: %f - cnst_loss:%f - cvrg_loss:%f" %(
            k, loss, debug_stats[-1], debug_stats[-2]
        )

        if (k % 10) == 0:
            fig = plt.figure(figsize=(4, 3))
            axis = plt.gca()
            axis.add_patch(get_rectangle(c1, T, edgecolor='g', fill=None, alpha=1, label="target", linewidth=2))
            axis.add_patch(get_rectangle(c2, T, edgecolor='g', fill=None, alpha=1, linewidth=2))
            axis.add_patch(
                Rectangle(
                    (translations[0, 0] - shapes[0, 0], translations[0, 1] - shapes[0, 1]),
                    2*shapes[0, 0],
                    2*shapes[0, 1],
                    edgecolor='r',
                    fill=None,
                    linestyle='--', alpha=1, linewidth=2, label="primitive 1"
                ))
            axis.add_patch(
                Rectangle((translations[0, 3] - shapes[0, 3], translations[0, 4] - shapes[0, 4]),
                2*shapes[0, 3], 2*shapes[0, 4], edgecolor='b', fill=None, linestyle='--', alpha=1, linewidth=2, label="primitive 2")
           )
            plt.xlim((-0.75, 0.75))
            plt.ylim((-0.75, 0.75))
            plt.legend(loc="upper left")
            # plt.title(r"\textbf{Chamfer distance (Ours)}")
            # plt.savefig("/tmp/empirical_test/chamfer_iter_%05d.png" % (k,), bbox_inches="tight")
            # plt.savefig("/tmp/empirical_test/chamfer_iter_%05d.pdf" % (k,), bbox_inches="tight")
            plt.title(r"\textbf{Truncated distance (Tulsiani et al.)}")
            plt.savefig("/tmp/empirical_test/trunc_iter_%05d.png" % (k,), bbox_inches="tight")
            plt.savefig("/tmp/empirical_test/trunc_iter_%05d.pdf" % (k,), bbox_inches="tight")
            plt.close()

        # Update everything
        optim.step()
        shapes.data.clamp_(min=0.01, max=0.5)
