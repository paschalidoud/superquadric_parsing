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
     voxelizer_shape, add_loss_options_parameters, add_loss_parameters
from utils import get_colors, store_primitive_parameters
from visualization_utils import points_on_sq_surface, points_on_cuboid, \
    save_prediction_as_ply

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

from mayavi import mlab


def get_shape_configuration(use_cuboids):
    if use_cuboids:
        return points_on_cuboid
    else:
        return points_on_sq_surface


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
        "--save_prediction_as_mesh",
        action="store_true",
        help="When true store prediction as a mesh"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--with_animation",
        action="store_true",
        help="Add animation"
    )

    add_dataset_parameters(parser)
    add_nn_parameters(parser)
    add_voxelizer_parameters(parser)
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
    print("Running code on ", device)

    # Create a factory that returns the appropriate voxelizer based on the
    # input argument
    voxelizer_factory = VoxelizerFactory(
        args.voxelizer_factory,
        np.array(voxelizer_shape(args)),
        args.save_voxels_to
    )

    # Create a dataset instance to generate the samples for training
    dataset = get_dataset_type("euclidean_dual_loss")(
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

        # Keep track of the files containing the parameters of each primitive
        primitive_files = []
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

            # Dump the parameters of each primitive as a dictionary
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
                primitive_files.append(
                    os.path.join(args.output_directory, "primitive_%d.p" % (i,))
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
            print(i, probs[0, i])

        print("Using %d primitives out of %d" % (on_prims, args.n_primitives))
        mlab.show()

        if args.save_prediction_as_mesh:
            print("Saving prediction as mesh....")
            save_prediction_as_ply(
                primitive_files,
                os.path.join(args.output_directory, "primitives.ply")
            )
            print("Saved prediction as ply file in {}".format(
                os.path.join(args.output_directory, "primitives.ply")
            ))


if __name__ == "__main__":
    main(sys.argv[1:])
