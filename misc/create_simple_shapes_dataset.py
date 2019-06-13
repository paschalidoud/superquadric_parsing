#!/usr/bin/env python
"""Script used to generate a cuboid dataset with cubes and rectangles under
various shapes, rotations, translations following the general format of
ShapeNet.
"""
import argparse
import random
import os
from string import ascii_letters, digits
import sys

import numpy as np
from progress.bar import Bar
from pyquaternion import Quaternion

from shapes import Shape, Cuboid, Ellipsoid

from learnable_primitives.mesh import MeshFromOBJ


def get_single_cube(minimum, maximum):
    minimum = minimum[0]
    maximum = maximum[0]
    r = minimum + np.random.rand() * (maximum-minimum)

    return Cuboid(-r, r, -r, r, -r, r)


def get_single_rectangle(minimum, maximum):
    minimum = np.array(minimum)
    maximum = np.array(maximum)
    rs = minimum + np.random.rand(3) * (maximum - minimum)

    return Cuboid(-rs[0], rs[0], -rs[1], rs[1], -rs[2], rs[2])


def adjacent_cubes(R):
    x_max1, y_max1, z_max1 = tuple(np.random.rand(3))
    x_max2, y_max2, z_max2 = tuple(np.random.rand(3))
    c1 = Cuboid(-x_max1, x_max1, -y_max1, y_max1, -z_max1, z_max1)
    c2 = Cuboid(-x_max2, x_max2, -y_max2, y_max2, -z_max2, z_max2)
    t1 = np.array([
        [0.0, y_max2 + y_max1, 0.0],
        [x_max2 + x_max1, 0.0, 0.0],
        [0.0, 0.0, z_max2 + z_max1]
    ])
    t = t1[np.random.choice(np.arange(3))].reshape(3, 1)
    c2.translate(t)
    c1.rotate(R)
    c2.rotate(R)
    return c1, c2


def multiple_cubes(R1, R2, t):
    x_max1, y_max1, z_max1 = tuple(np.random.rand(3))
    x_max2, y_max2, z_max2 = tuple(np.random.rand(3))
    c1 = Cuboid(-x_max1, x_max1, -y_max1, y_max1, -z_max1, z_max1)
    c2 = Cuboid(-x_max2, x_max2, -y_max2, y_max2, -z_max2, z_max2)
    c1.rotate(R1)
    c2.translate(t)
    c2.rotate(R2)
    #c2.translate(R2.dot(t))
    return c1, c2


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
        "--shapes_type",
        default="cubes",
        choices=[
            "cubes",
            "cubes_translated",
            "cubes_rotated_translated",
            "cubes_rotated",
            "rectangles",
            "rectangles_translated",
            "rectangles_rotated",
            "rectangles_rotated_translated",
            "ellipsoid",
            "random"
        ],
        help="The type of the shapes in every sample"
    )
    parser.add_argument(
        "--n_shapes_per_samples",
        type=int,
        default=1,
        help="Number of shapes per sample"
    )
    parser.add_argument(
        "--maximum",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0.5,0.5,0.5",
        help="Maximum size along every axis"
    )
    parser.add_argument(
        "--minimum",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0.13,0.13,0.13",
        help="Maximum size along every axis"
    )
    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create a directory based on the type of the shapes inside the output
    # directory
    output_directory = os.path.join(
        args.output_directory,
        args.shapes_type
    )

    ranges = None
    if "cubes" in args.shapes_type:
        # Make sure that the maximum and minimum range are equal along each
        # axis
        assert args.maximum[0] == args.maximum[1]
        assert args.maximum[1] == args.maximum[2]
        assert args.minimum[0] == args.minimum[1]
        assert args.minimum[1] == args.minimum[2]
        ranges = np.linspace(
            args.minimum[0],
            args.maximum[0],
            10,
            endpoint=False
        )

    # elif "rectangles" in args.shapes_type:
    else:
        ranges = [
            np.linspace(args.minimum[0], args.maximum[0], 10, endpoint=False),
            np.linspace(args.minimum[1], args.maximum[1], 10, endpoint=False),
            np.linspace(args.minimum[2], args.maximum[2], 10, endpoint=False),
        ]

    bar = Bar("Generating %d cuboids" % (args.n_samples,), max=args.n_samples)
    c = None
    for i in range(args.n_samples):
        if "cubes" in args.shapes_type:
            c = get_single_cube(args.minimum, args.maximum)
        if "rectangles" in args.shapes_type:
            c = get_single_rectangle(args.minimum, args.maximum)

        if "translated" in args.shapes_type:
            t = 0.3*np.random.random((3, 1))
            c.translate(t)

        if "rotated" in args.shapes_type:
            q = Quaternion.random()
            R = q.rotation_matrix
            c.rotate(R)

        if "ellipsoid" in args.shapes_type:
            abc = np.random.random((3, 1))
            c1 = Ellipsoid(abc[0], abc[1], abc[2])
            c2 = Ellipsoid(abc[0], abc[1], abc[2])
            c3 = Ellipsoid(abc[0], abc[1], abc[2])
            q = Quaternion.random()
            R = q.rotation_matrix
            c2.rotate(R)
            q = Quaternion.random()
            R = q.rotation_matrix
            c3.rotate(R)
            # t = 0.3*np.random.random((3, 1))
            # c1.translate(t)
            c = Shape.from_shapes([c1, c2, c3])

        if "random" in args.shapes_type:
            #if random.choice((True, False)):
            #if True:
            #   q = Quaternion.random()
            #   c1, c2 = adjacent_cubes(q.rotation_matrix)
            #else:
            if True:
                q1 = Quaternion.random()
                q2 = Quaternion.random()
                c1, c2 = multiple_cubes(
                    q1.rotation_matrix,
                    q2.rotation_matrix,
                    3.5*np.random.random((3, 1))
                )
            # q = Quaternion.random()
            # c1, c2 = adjacent_cubes(q.rotation_matrix)
            # q1 = Quaternion.random()
            # x_max1, y_max1, z_max1 = tuple(np.random.rand(3))
            # c3 = Cuboid(-x_max1, x_max1, -y_max1, y_max1, -z_max1, z_max1)
            # c3.rotate(q1.rotation_matrix)
            # c3.translate(np.random.random((3,1)).reshape(3, -1))
            c = Shape.from_shapes([c1, c2])

        # Create subdirectory to save the sample
        folder_name = ''.join([
            random.choice(ascii_letters + digits) for n in xrange(32)
        ])
        base_dir = os.path.join(output_directory, folder_name, "models")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # print base_dir
        # Save as obj file
        c.save_as_mesh(os.path.join(base_dir, "model_normalized.obj"), "obj")
        c.save_as_mesh(os.path.join(base_dir, "model_normalized.ply"), "ply")
        c.save_as_pointcloud(
            os.path.join(base_dir, "model_normalized_pcl.obj"), "obj"
        )
        if "translated" in args.shapes_type:
            print os.path.join(base_dir, "model_normalized_pcl.obj"), t.T
        if "rotated" in args.shapes_type:
            print os.path.join(base_dir, "model_normalized_pcl.obj"), q
        bar.next()

    for i in os.listdir(output_directory):
        x = os.path.join(output_directory, i, "models/model_normalized.obj")
        m = MeshFromOBJ(x)
        print x, m.points.max(-1)


if __name__ == "__main__":
    main(sys.argv[1:])
