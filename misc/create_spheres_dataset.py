#!/usr/bin/env python
"""Script used to generate a spheres dataset following the same format as
ShapeNet.
"""
import argparse
import os
import sys

import numpy as np
from pyquaternion import Quaternion

from shapes import Sphere, Ellipsoid, Shape

from learnable_primitives.mesh import MeshFromOBJ
from learnable_primitives.utils.progbar import Progbar


def build_sequentially_attaching_sheres(N):
    """
        Arguments:
        ---------
            N: number of shapes to be build
    """
    def t(s):
        return s[0]

    def r(s):
        return s[1]

    def overlap(s1, s2):
        d = np.sqrt(np.sum((t(s1)-t(s2))**2))
        return d < r(s1) + r(s2)

    def random_point(s, d=0):
        v = np.random.randn(3)
        v = v / np.sqrt((v**2).sum())
        return t(s) + v*(r(s)+d)

    spheres = [(np.zeros(3), np.random.rand()*0.4 + 0.1)]
    while len(spheres) < N:
        s1 = spheres[np.random.choice(len(spheres))]
        s2r = np.random.rand()*0.4 + 0.1
        s2c = random_point(s1, s2r)
        s2 = (s2c, s2r)
        if not any(overlap(s, s2) for s in spheres):
            spheres.append(s2)

    return [Sphere(r(s)).translate(t(s)[:, np.newaxis]) for s in spheres]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate a cuboid dataset"
    )
    parser.add_argument(
        "output_directory",
        help="Save the dataset in this directory"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of training samples to be generated"
    )
    parser.add_argument(
        "--max_n_shapes_per_samples",
        type=int,
        default=4,
        help="Number of shapes per sample"
    )
    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create a directory based on the type of the shapes inside the output
    # directory
    output_directory = os.path.join(
        args.output_directory,
        "spheres_dataset"
    )
    print "Saving models to %s" % (output_directory,)

    prog = Progbar(args.n_samples)
    for i in range(args.n_samples):
        prims = build_sequentially_attaching_sheres(
            np.random.choice(np.arange(2, args.max_n_shapes_per_samples))
        )
        c = Shape.from_shapes(prims)
        # Create subdirectory to save the sample
        base_dir = os.path.join(output_directory, "%05d" % (i,), "models")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # print base_dir
        # Save as obj file
        c.save_as_mesh(os.path.join(base_dir, "model_normalized.obj"), "obj")
        c.save_as_mesh(os.path.join(base_dir, "model_normalized.ply"), "ply")
        c.save_as_pointcloud(
            os.path.join(base_dir, "model_normalized_pcl.obj"), "obj"
        )
        prog.update(i + 1)


if __name__ == "__main__":
    main(sys.argv[1:])
