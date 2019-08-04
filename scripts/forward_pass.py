#!/usr/bin/env python
"""Script used to perform a forward pass using a previously trained model and
visualize the corresponding primitives
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from arguments import add_voxelizer_parameters, add_nn_parameters, \
     add_dataset_parameters, add_gaussian_noise_layer_parameters, \
     voxelizer_shape, add_tsdf_fusion_parameters, \
     add_loss_options_parameters, add_loss_parameters
from utils import get_colors, store_primitive_parameters

from learnable_primitives.common.dataset import get_dataset_type,\
    compose_transformations
from learnable_primitives.common.model_factory import DatasetBuilder
from learnable_primitives.equal_distance_sampler_sq import\
    EqualDistanceSamplerSQ
from learnable_primitives.models import NetworkParameters
from learnable_primitives.loss_functions import euclidean_dual_loss
from learnable_primitives.primitives import\
    euler_angles_to_rotation_matrices, quaternions_to_rotation_matrices
from learnable_primitives.voxelizers import VoxelizerFactory

# Import mayavi
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)
from mayavi import mlab

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)


def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z


def get_sq(a1, a2, a3, e1, e2, R, t, Kx, Ky, n_samples=100):
    """Computes a SQ given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)

    # Apply the deformations
    fx = Kx * z / a3
    fx += 1
    fy = Ky * z / a3
    fy += 1
    fz = 1

    x = x * fx
    y = y * fy
    z = z * fz

    # Get an array of size 3x10000 that contains the points of the SQ
    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t
    print "R:", R
    print "t:", t
    print "e:", [e1, e2]
    print "K:", [Kx, Ky]

    x_tr = points_transformed[0].reshape(n_samples, n_samples)
    y_tr = points_transformed[1].reshape(n_samples, n_samples)
    z_tr = points_transformed[2].reshape(n_samples, n_samples)

    return x_tr, y_tr, z_tr, points_transformed


def get_cuboid(a1, a2, a3, e1, e2, R, t, n_samples=100):
    """Computes a cube given a set of parameters and saves it into a np array
    """
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)

    X = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1]
    ], dtype=np.float32)
    X[X == 1.0] = a1
    X[X == 0.0] = -a1

    Y = np.array([
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ], dtype=np.float32)
    Y[Y == 1.0] = a2
    Y[Y == 0.0] = -a2

    Z = np.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1]
    ], dtype=np.float32)
    Z[Z == 1.0] = a3
    Z[Z == 0.0] = -a3

    points = np.stack([X, Y, Z]).reshape(3, -1)
    points_transformed = R.T.dot(points) + t
    print "R:", R
    print "t:", t

    assert points.shape == (3, 18)

    x_tr = points_transformed[0].reshape(2, 9)
    y_tr = points_transformed[1].reshape(2, 9)
    z_tr = points_transformed[2].reshape(2, 9)
    return x_tr, y_tr, z_tr, points_transformed


def get_shape_configuration(use_cuboids):
    if use_cuboids:
        return get_cuboid
    else:
        return get_sq


def is_inside(pcl1, pcl2, threshold):
    # Check the percentage of points that lie inside another pointcloud and if
    # they exceed a threashold return True, else return False
    assert pcl1.shape[0] == 3
    assert pcl2.shape[0] == 3
    # for every point in pcl2 check whether it lies inside pcl1
    minimum = pcl1.min(1)
    maximum = pcl1.max(1)

    c1 = np.logical_and(
        pcl2[0, :] <= maximum[0],
        pcl2[0, :] >= minimum[0],
    )
    c2 = np.logical_and(
        pcl2[1, :] <= maximum[1],
        pcl2[1, :] >= minimum[1],
    )
    c3 = np.logical_and(
        pcl2[2, :] <= maximum[2],
        pcl2[2, :] >= minimum[2],
    )
    c4 = np.logical_and(c1, np.logical_and(c2, c3)).sum()
    return float(c4) / pcl1.shape[1] > threshold


def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--tsdf_directory",
        default="",
        help="Path to the directory containing the precomputed tsdf files"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )

    parser.add_argument(
        "--n_primitives",
        type=int,
        default=32,
        help="Number of primitives"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )
    parser.add_argument(
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--title",
        default="Fooo",
        help="Title on the plot"
    )
    parser.add_argument(
        "--save_image_to",
        default="/tmp/image_0.png",
        help="Path to image"
    )
    parser.add_argument(
        "--with_animation",
        action="store_true",
        help="Add animation"
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=0,
        help="Epoch at which this model was captured"
    )

    add_dataset_parameters(parser)
    add_nn_parameters(parser)
    add_voxelizer_parameters(parser)
    add_tsdf_fusion_parameters(parser)
    add_gaussian_noise_layer_parameters(parser)
    add_loss_parameters(parser)
    add_loss_options_parameters(parser)
    args = parser.parse_args(argv)

    # A sampler instance
    e = EqualDistanceSamplerSQ(200)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print "Running code on ", device

    losses = {
        "euclidean_dual_loss": euclidean_dual_loss
    }
    loss_factory = losses[args.loss_type]

    # Create a factory that returns the appropriate voxelizer based on the
    # input argument
    voxelizer_factory = VoxelizerFactory(
        args.voxelizer_factory,
        np.array(voxelizer_shape(args)),
        args.save_voxels_to
    )

    # Create a dataset instance to generate the samples for training
    dataset = get_dataset_type(args.loss_type)(
        (DatasetBuilder()
            .with_dataset(args.dataset_type)
            .filter_tags(args.model_tags)
            .build(args.dataset_directory)),
        voxelizer_factory,
        args.n_points_from_mesh,
        transform=compose_transformations(voxelizer_factory)
    )

    # TODO: Change batch_size in dataloader
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    network_params = NetworkParameters.from_options(args)
    # Build the model to be used for testing
    model = network_params.network(network_params)
    # Move model to device to be used
    model.to(device)
    if args.weight_file is not None:
        # Load the model parameters of the previously trained model
        model.load_state_dict(
            torch.load(args.weight_file, map_location=device)
        )
    model.eval()

    colors = get_colors(args.n_primitives)
    for sample in dataloader:
        X, y_target = sample
        X, y_target = X.to(device), y_target.to(device)

        # Do the forward pass and estimate the primitive parameters
        y_hat = model(X)

        M = args.n_primitives  # number of primitives
        probs = y_hat[0].to("cpu").detach().numpy()
        # Transform the Euler angles to rotation matrices
        if y_hat[2].shape[1] == 3:
            R = euler_angles_to_rotation_matrices(
                y_hat[2].view(-1, 3)
            ).to("cpu").detach()
        else:
            R = quaternions_to_rotation_matrices(
                    y_hat[2].view(-1, 4)
                ).to("cpu").detach()
            # get also the raw quaternions
            quats = y_hat[2].view(-1, 4).to("cpu").detach().numpy()
        translations = y_hat[1].to("cpu").view(args.n_primitives, 3)
        translations = translations.detach().numpy()

        shapes = y_hat[3].to("cpu").view(args.n_primitives, 3).detach().numpy()
        epsilons = y_hat[4].to("cpu").view(
            args.n_primitives, 2
        ).detach().numpy()
        taperings = y_hat[5].to("cpu").view(
            args.n_primitives, 2
        ).detach().numpy()

        pts = y_target[:, :, :3].to("cpu")
        pts_labels = y_target[:, :, -1].to("cpu").squeeze().numpy()
        pts = pts.squeeze().detach().numpy().T

        on_prims = 0
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        mlab.view(azimuth=0.0, elevation=0.0, distance=2)
        # Uncomment to visualize the points sampled from the target mesh
        # t = np.array([1.2, 0, 0]).reshape(3, -1)
        # pts_n = pts + t
        #     mlab.points3d(
        #        # pts_n[0], pts_n[1], pts_n[2],
        #        pts[0], pts[1], pts[2],
        #        scale_factor=0.03, color=(0.8, 0.8, 0.8)
        #     )
        for i in range(args.n_primitives):
            x_tr, y_tr, z_tr, prim_pts =\
                get_shape_configuration(args.use_cuboids)(
                    shapes[i, 0],
                    shapes[i, 1],
                    shapes[i, 2],
                    epsilons[i, 0],
                    epsilons[i, 1],
                    R[i].numpy(),
                    translations[i].reshape(-1, 1),
                    taperings[i, 0],
                    taperings[i, 1]
                )
            store_primitive_parameters(
                size=tuple(shapes[i]),
                shape=tuple(epsilons[i]),
                rotation=tuple(quats[i]),
                location=tuple(translations[i]),
                tapering=tuple(taperings[i]),
                probability=(probs[0, i],),
                color=(colors[i % len(colors)]) + (1.0,),
                filepath=os.path.join(
                    args.output_directory,
                    "primitive_%d.p" %(i,)
                )
            )
            if probs[0, i] >= args.prob_threshold:
                on_prims += 1
                mlab.mesh(
                    x_tr,
                    y_tr,
                    z_tr,
                    color=tuple(colors[i % len(colors)]),
                    opacity=1.0
                )

        if args.with_animation:
            cnt = 0
            for az in range(0, 360, 1):
                cnt += 1
                mlab.view(azimuth=az, elevation=0.0, distance=2)
                mlab.savefig(
                    os.path.join(
                        args.output_directory,
                        "img_%04d.png" % (cnt,)
                    )
                )
        for i in range(args.n_primitives):
            print i, probs[0, i]

        print "Using %d primitives out of %d" % (on_prims, args.n_primitives)
        mlab.show()


if __name__ == "__main__":
    main(sys.argv[1:])
