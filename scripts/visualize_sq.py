#!/usr/bin/env python
"""Script for visualizing superquadrics."""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('text', usetex=True)
plt.rc("font", size=10, family="serif")

from arguments import add_sq_mesh_sampler_parameters

from learnable_primitives.equal_distance_sampler_sq import \
    EqualDistanceSamplerSQ
from visualization_utils import sq_surface


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize superquadrics given a set of parameters"
    )
    parser.add_argument(
        "--save_image_to",
        default=None,
        help="Path to save the generated image"
    )
    parser.add_argument(
        "--size",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0.55,0.55,0.55",
        help="Size of the superquadric"
    )
    parser.add_argument(
        "--shape",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0.35,0.35",
        help="Shape of the superquadric"
    )
    parser.add_argument(
        "--n_points_from_sq_mesh",
        type=int,
        default=1000,
        help="Number of points to be sampled from the SQ"
    )
    parser.add_argument(
        "--with_mesh",
        action="store_true",
        help="When true visualize the sampled points as a mesh"
    )
    args = parser.parse_args(argv)

    # Create an object that will sample points in equal distances on the
    # surface of the primitive
    e = EqualDistanceSamplerSQ(
        args.n_points_from_sq_mesh
    )
    a1, a2, a3 = args.size
    size = np.array([[[a1, a2, a3]]], dtype=np.float32)
    e1, e2, = args.shape
    shape = np.array([[[e1, e2]]], dtype=np.float32)

    etas, omegas = e.sample_on_batch(size, shape)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, etas.ravel(), omegas.ravel())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.65])
    ax.set_zlim([-0.65, 0.65])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    plt.title("Superquadric with size:(%0.3f, %0.3f, %0.3f) and shape:(%0.3f, %0.3f)" % (a1, a2, a3, e1, e2))
    # Uncomment this if you want to save the SQ as png
    if args.save_image_to is not None:
        plt.subplots_adjust()
        plt.savefig(args.save_image_to)
    else:
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
