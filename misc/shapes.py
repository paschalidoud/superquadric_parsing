from itertools import permutations
import math

from csg.core import CSG, Polygon as CSGPolygon, Vertex as CSGVertex
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull
import trimesh

from learnable_primitives.pointcloud import Pointcloud


class Shape(object):
    """A wrapper class for shapes"""
    def __init__(self, points, face_idxs):
        self._points = points
        self._faces_idxs = face_idxs

    @property
    def points(self):
        if self._points is None:
            raise NotImplementedError()
        return self._points

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            raise NotImplementedError()
        return self._faces_idxs

    def save_as_mesh(self, filename, format="ply"):
        m = trimesh.Trimesh(vertices=self.points.T, faces=self.faces_idxs)
        # Make sure that the face orinetations are ok
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fix_winding(m)
        assert m.is_winding_consistent == True
        m.export(filename)

    def sample_faces(self, N=1000):
        m = trimesh.Trimesh(vertices=self.points.T, faces=self.faces_idxs)
        # Make sure that the face orinetations are ok
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fix_winding(m)
        assert m.is_winding_consistent == True
        P, t = trimesh.sample.sample_surface(m, N)
        return np.hstack([
            P, m.face_normals[t, :]
        ])

    def save_as_pointcloud(self, filename, format="ply"):
        pcl = Pointcloud(self.points)
        if format == "ply":
            pcl.save_ply(filename)
        else:
            pcl.save_obj(filename)

    def rotate(self, R):
        """ 3x3 rotation matrix that will rotate the points
        """
        # Make sure that the rotation matrix has the right shape
        assert R.shape == (3, 3)
        self._points = R.T.dot(self.points)

        return self

    def translate(self, t):
        # Make sure thate everything has the right shape
        assert t.shape[0] == 3
        assert t.shape[1] == 1
        self._points = self.points + t

        return self

    def to_csg(self):
        points = self.points
        polygons = [
            CSGPolygon([
                CSGVertex(pos=points[:, i])
                for i in face
            ])
            for face in self.faces_idxs
        ]

        return CSG.fromPolygons(polygons)

    @classmethod
    def from_csg(cls, csg_object):
        points, face_idxs, n = csg_object.toVerticesAndPolygons()
        triangles = []
        for face in face_idxs:
            if len(face) == 3:
                triangles.append(face)
            else:
                for j in range(2, len(face)):
                    triangles.append((face[0], face[j-1], face[j]))

        return cls(
            np.array(points).T,
            np.array(triangles)
        )

    @classmethod
    def from_shapes(cls, shapes):
        # Make sure that the input is a list of shapes
        isinstance(shapes, list)
        points = []
        triangles = []
        for i, s in enumerate(shapes):
            if len(points) == 0:
                triangles.append(s.faces_idxs)
            else:
                triangles.append(s.faces_idxs + i*points[-1].shape[1])
            points.append(s.points)

        return cls(
            np.hstack(points),
            np.vstack(triangles)
        )

    @staticmethod
    def get_orientation_of_face(points, face_idxs):
        # face_idxs corresponds to the indices of a single face
        assert len(face_idxs.shape) == 1
        assert face_idxs.shape[0] == 3

        x = np.vstack([
            points.T[face_idxs, 0].T,
            points.T[face_idxs, 1].T,
            points.T[face_idxs, 2].T
        ]).T

        # Based on the Wikipedia article
        # https://en.wikipedia.org/wiki/Curve_orientation
        # If the determinant is negative, then the polygon is oriented
        # clockwise. If the determinant is positive, the polygon is oriented
        # counterclockwise
        return np.linalg.det(x)

    @staticmethod
    def fix_orientation_of_face(points, face_idxs):
        # face_idxs corresponds to the indices of a single face
        assert len(face_idxs.shape) == 1
        assert face_idxs.shape[0] == 3

        # Iterate over all possible permutations
        for item in permutations(face_idxs, face_idxs.shape[0]):
            t = np.array(item)
            orientation = Shape.get_orientation_of_face(points, t)
            if orientation < 0:
                pass
            else:
                return t


class ConvexShape(Shape):
    """A wrapper class for convex shapes"""
    def __init__(self, points):
        self._points = points

        # Contains the convexhull of the set of points (see cv property)
        self._cv = None
        # Contains the faces_idxs (see face_idxs Shape property)
        self._faces_idxs = None

    @property
    def points(self):
        return self._points

    @property
    def cv(self):
        if self._cv is None:
            self._cv = ConvexHull(self.points.T)
        return self._cv

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            self._faces_idxs = np.array(self.cv.simplices)
            self._make_consistent_orientation_of_faces()
        return self._faces_idxs

    def _make_consistent_orientation_of_faces(self):
        for i, face_idxs in zip(xrange(self.cv.nsimplex), self.faces_idxs):
            # Compute the orientation for the current face
            orientation = Shape.get_orientation_of_face(self.points, face_idxs)
            if orientation < 0:
                # if the orientation is negative, permute the face_idxs to make
                # it positive
                self._faces_idxs[i] =\
                    Shape.fix_orientation_of_face(self.points, face_idxs)


class Cuboid(ConvexShape):
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        super(Cuboid, self).__init__(self._create_points(
            x_min, x_max, y_min, y_max, z_min, z_max
        ))

    def _create_points(self, x_min, x_max, y_min, y_max, z_min, z_max):
        vertices = np.array([
            [x_min, x_max],
            [y_min, y_max],
            [z_min, z_max]
        ])
        idxs = np.array([
            [i, j, k]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ])

        return vertices[np.tile(range(3), (8, 1)), idxs].T

    @staticmethod
    def keep_points_inside_cube(x, y, z, x_min, x_max,
                                y_min, y_max, z_min, z_max):
        c1 = np.logical_and(
            np.logical_and(x >= x_min, x <= x_max),
            np.logical_and(y >= y_min, y <= y_max)
        )
        c2 = np.logical_and(
            np.logical_and(y >= y_min, y <= y_max),
            np.logical_and(z >= z_min, z <= z_max)
        )
        c3 = np.logical_and(
            np.logical_and(x >= x_min, x <= x_max),
            np.logical_and(z >= z_min, z <= z_max)
        )
        c4 = np.logical_and(c1, z == z_min)
        c5 = np.logical_and(c1, z == z_max)
        c6 = np.logical_and(c2, x == x_min)
        c7 = np.logical_and(c2, x == x_max)
        c8 = np.logical_and(c3, y == y_min)
        c9 = np.logical_and(c3, y == y_max)
        return np.logical_or(
            np.logical_or(np.logical_or(c5, c6), np.logical_or(c7, c8)),
            np.logical_or(c4, c9)
            )


class Sphere(ConvexShape):
    def __init__(self, radius):
        self._radius = radius
        super(Sphere, self).__init__(Sphere.fibonacci_sphere(self._radius))

    @property
    def radius(self):
        return self._radius

    @staticmethod
    def fibonacci_sphere(radius, samples=100):
        # From stackoverflow on How to evenly distribute N points on a sphere
        points = []
        offset = 2./samples
        increment = math.pi * (3. - math.sqrt(5.))

        # Point in the unit sphere
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - pow(y, 2))

            phi = ((i + 1) % samples) * increment

            x = math.cos(phi) * r
            z = math.sin(phi) * r

            points.append([x, y, z])

        return np.array(points).T * radius


class Ellipsoid(ConvexShape):
    def __init__(self, a, b, c):
        super(Ellipsoid, self).__init__(
            self._create_points(a, b, c)
        )

    def _create_points(self, a, b, c):
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        phi = np.linspace(-np.pi, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = a * np.cos(theta) * np.cos(phi)
        y = b * np.cos(theta) * np.sin(phi)
        z = c * np.sin(theta)
        points = np.stack([x, y, z]).reshape(3, -1)

        return points
