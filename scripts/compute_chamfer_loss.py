#!/usr/bin/env python
"""Script used to compute the Chamfer distance on a set of pre-trained models
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from pyquaternion import Quaternion

from arguments import add_voxelizer_parameters, add_nn_parameters,\
     add_dataset_parameters, add_gaussian_noise_layer_parameters,\
     voxelizer_shape, add_loss_options_parameters, \
     add_loss_parameters, get_loss_options

from learnable_primitives.common.dataset import get_dataset_type,\
    compose_transformations
from learnable_primitives.common.model_factory import DatasetBuilder
from learnable_primitives.equal_distance_sampler_sq import\
    EqualDistanceSamplerSQ
from learnable_primitives.models import NetworkParameters
from learnable_primitives.loss_functions import euclidean_dual_loss
from learnable_primitives.voxelizers import VoxelizerFactory
from learnable_primitives.utils.progbar import Progbar


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
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
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
        n_bbox=args.n_bbox,
        n_surface=args.n_surface,
        equal=args.equal,
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

    losses = []
    pcl_to_prim_losses = []
    prim_to_pcl_losses = []

    prog = Progbar(len(dataloader))
    i = 0
    for sample in dataloader:
        X, y_target = sample
        X, y_target = X.to(device), y_target.to(device)

        # Do the forward pass and estimate the primitive parameters
        y_hat = model(X)

        reg_terms = {
            "regularizer_type": [],
            "bernoulli_regularizer_weight": 0.0,
            "entropy_bernoulli_regularizer_weight": 0.0,
            "parsimony_regularizer_weight": 0.0,
            "overlapping_regularizer_weight": 0.0,
            "sparsity_regularizer_weight": 0.0,
        }
        loss, debug_stats = euclidean_dual_loss(
            y_hat,
            y_target,
            reg_terms,
            e,
            get_loss_options(args)
        )

        if not np.isnan(loss.item()):
            losses.append(loss.item())
            pcl_to_prim_losses.append(debug_stats["pcl_to_prim_loss"].item())
            prim_to_pcl_losses.append(debug_stats["prim_to_pcl_loss"].item())
        # Update progress bar
        prog.update(i+1)
        i += 1
    np.savetxt(
        os.path.join(args.output_directory, "losses.txt"),
        losses
    )

    np.savetxt(
        os.path.join(args.output_directory, "pcl_to_prim_losses.txt"),
        pcl_to_prim_losses
    )
    np.savetxt(
        os.path.join(args.output_directory, "prim_to_pcl_losses.txt"),
        prim_to_pcl_losses
    )
    np.savetxt(
        os.path.join(args.output_directory, "mean_std_losses.txt"),
        [np.mean(losses), np.std(losses),
        np.mean(pcl_to_prim_losses), np.std(pcl_to_prim_losses),
        np.mean(prim_to_pcl_losses), np.std(prim_to_pcl_losses)]
    )

    print("loss: %.7f +/- %.7f - pcl_to_prim_loss %.7f +/- %.7f - prim_to_pcl_loss %.7f +/- %.7f" %(
        np.mean(losses),
        np.std(losses),
        np.mean(pcl_to_prim_losses),
        np.std(pcl_to_prim_losses),
        np.mean(prim_to_pcl_losses),
        np.std(prim_to_pcl_losses)
    ))

if __name__ == "__main__":
    main(sys.argv[1:])
