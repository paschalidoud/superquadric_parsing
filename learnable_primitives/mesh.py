"""Create a Mesh object by parsing a file either in .ply format or .obj format
"""
import os
import sys

import numpy as np
import trimesh
import re

from .pointcloud import PointcloudFromOBJ, Pointcloud, PointcloudFromOFF


class Mesh(object):
    """A collection of vertices, normals and faces that construct a 3D mesh of
    an object.
    """
    def __init__(self, points, normals, faces_idxs):
        self._normals = normals
        self._faces_idxs = faces_idxs
        self._face_normals = None

        # Pointcloud object that will hold the points
        self._pointcloud = Pointcloud(points)
        self._faces = None

        self._bbox = None

    @staticmethod
    def get_face_normals(faces):
        assert faces.shape[1] == 9
        A = faces[:, :3]
        B = faces[:, 3:6]
        C = faces[:, 6:]
        AB = B - A
        AC = C - A
        n = np.cross(AB, AC)
        n /= np.linalg.norm(n, axis=-1)[:, np.newaxis]
        # Make sure that n are unit vectors
        assert np.linalg.norm(n, axis=-1).sum() == n.shape[0]
        return n

    @staticmethod
    def get_face_normals_from_file(mesh_file):
        m = trimesh.load(mesh_file)
        # Make sure that the face orinetations are ok
        trimesh.repair.fix_normals(m, multibody=True)
        trimesh.repair.fix_winding(m)
        assert m.is_winding_consistent == True
        return m.face_normals

    @property
    def pointcloud(self):
        return self._pointcloud

    @property
    def points(self):
        return self.pointcloud.points

    @property
    def normals(self):
        return self._normals

    @property
    def faces(self):
        if self._faces is None:
            self._faces = np.array([np.hstack((
                    self.points.T[idx[0]],
                    np.hstack((self.points.T[idx[1]], self.points.T[idx[2]]))
                 )) for idx in self.faces_idxs.T
            ])
        return self._faces

    @property
    def faces_idxs(self):
        return self._faces_idxs

    @property
    def face_normals(self):
        raise NotImplementedError()

    @property
    def bounding_box(self):
        if self._bbox is None:
            self._bbox = np.vstack([
                self.points.min(-1),
                self.points.max(-1)
            ])
            assert self._bbox.shape == (2, 3)
        return self._bbox

    def sample_points(self, N):
        return self.pointcloud.sample(N)

    def save_obj(self, file):
        with open(file, "w") as f:
            f.write("# OBJ file\n")
            for v in self.points.T:
                f.write("v %.4f %.4f %.4f\n" % tuple(v.tolist()))
            for fidx in self.faces_idxs.T:
                f.write("f")
                for i in fidx:
                    f.write(" %d" % (i+1))
                f.write("\n")

    def save_ply_as_ascii(self, file):
        # Number of points (vertices)
        N = self.points.shape[1]
        # Number of faces
        F = self.faces_idxs.shape[1]

        with open(file, "w") as f:
            f.write(("ply\nformat ascii 1.0\ncomment Raynet"
                     " pointcloud!\nelement vertex %d\nproperty float x\n"
                     "property float y\nproperty float z\nelement face %d\n"
                     "property list uchar int vertex_indices\nend_header\n")
                    % (N, F))
            for p in self.points.T:
                f.write(" ".join(map(str, [p[0], p[1], p[2]])) + "\n")

            idxs = np.hstack([
                3*np.ones(self.faces_idxs.shape[1]).reshape(-1, 1),
                self.faces_idxs.T
            ]).astype(int)
            for p in idxs:
                f.write(" ".join(map(str, [p[0], p[1], p[2], p[3]])) + "\n")

    def save_ply(self, file):
        # Number of points (vertices)
        N = self.points.shape[1]
        # Number of faces
        F = self.faces_idxs.shape[1]

        with open(file, "w") as f:
            f.write(("ply\nformat binary_%s_endian 1.0\ncomment Raynet"
                     " pointcloud!\nelement vertex %d\nproperty float x\n"
                     "property float y\nproperty float z\nelement face %d\n"
                     "property list uchar int vertex_index\nend_header\n") % (
                     sys.byteorder, N, F))
            self.points.T.astype(np.float32).tofile(f)

            # TODO: Make this properly :-)
            for p in self.faces_idxs.T:
                np.array([3]).astype(np.uint8).tofile(f)
                np.array(p).astype(np.int32).tofile(f)

    def _sample_faces(self, N):
        # Change the shape of self.points and self.faces_idxs to match the
        # implementation
        vertices = self.points.T
        faces = self.faces_idxs.T

        n_faces = faces.shape[0]
        # Calculate all face areas, for total sum and cumsum sample uniformly
        # into the fractional size array, and then sample uniformly in that
        # triangle
        v0s = vertices[faces[:, 0], :]
        v1s = vertices[faces[:, 1], :] - v0s
        v2s = vertices[faces[:, 2], :] - v0s

        areas = np.power(np.sum(np.power(np.cross(v2s, v1s), 2), 1), 0.5)
        triangle_idxs = np.random.choice(
            len(areas),
            N,
            p=areas/np.sum(areas)
        )

        v1_fracs = np.random.rand(N, 1)
        v2_fracs = np.random.rand(N, 1)
        frac_out = (v1_fracs + v2_fracs > 1)
        v1_fracs[frac_out] = 1 - v1_fracs[frac_out]
        v2_fracs[frac_out] = 1 - v2_fracs[frac_out]

        P = v0s[triangle_idxs, :]
        P += v1_fracs * v1s[triangle_idxs, :]
        P += v2_fracs * v2s[triangle_idxs, :]

        return P, triangle_idxs

    def sample_faces(self, N=10000):
        P, t = self._sample_faces(N)
        return np.hstack([P, self.face_normals[t, :]])


class MeshFromOBJ(Mesh):
    """Construct a Mesh Object from an OBJ file
    """
    def __init__(self, obj_file):
        self.obj_file = obj_file
        # Raise Exception in case the given file does not exist
        if not os.path.exists(obj_file):
            raise IOException("File does not exist : %s" % (obj_file,))

        self._normals = None
        self._faces = None
        self._faces_idxs = None
        self._face_normals = None

        # Pointcloud object that will hold the points
        self._pointcloud = None

        self._bbox = None

    @property
    def pointcloud(self):
        if self._pointcloud is None:
            self._pointcloud = PointcloudFromOBJ(self.obj_file)
        return self._pointcloud

    @staticmethod
    def parse_data(obj_file):
        # List to keep the unprocessed lines parsed from the file
        lines = []
        with open(obj_file, "r") as f:
            lines = f.readlines()
            # Remove lines containing change of line
            lines = [_f for _f in [x.strip("\r\n") for x in lines] if _f]

            # Keep only the lines that start with the letter v, that
            # correspond to vertices
            vertices = [k for k in lines if k.startswith("v ")]
            # Remove the unwanted "v" in front of every row and transform
            # it to float
            points = np.array([
                    list(map(float, k.strip().split(" ")[1:])) for k in vertices
            ])

            # Keep only the lines that start with the bigramm vn, that
            # correspond to normals
            normals = [k for k in lines if k.startswith("vn")]
            # Remove the unwanted "v" in front of every row and transform
            # it to float
            normals = np.array([
                    list(map(float, k.strip().split(" ")[1:])) for k in normals
            ])

            # Keep only the lines that start with the letter f, that
            # correspond to faces
            f = [k for k in lines if k.startswith("f")]
            # Remove all empty strings from the list and split it
            t = [[_f for _f in x if _f] for x in [k.strip().split(" ") for k in f]]
            # Remove preficx "f"
            f_clean = [[x for x in k if "f" not in x] for k in t]

            faces_idxs = []
            normals_idxs = []
            for ff in f_clean:
                # Every row in f has the following format v1//vn1 v2//vn2
                # v3//vn3, where v* corresponds the vertex index while vn*
                # corresponds to the normal index.
                faces_idxs.append([re.split("/+", i)[0] for i in ff])
                normals_idxs.append([re.split("/+", i)[-1] for i in ff])
            faces_idxs = np.array([list(map(int, x)) for x in faces_idxs])
            normals_idxs = np.array([list(map(int, x)) for x in normals_idxs])
            # Remove 1 to make it compatible with the zero notation
            faces_idxs = faces_idxs - 1
            normals_idxs = normals_idxs - 1

            return points.T, normals.T, faces_idxs.T, normals_idxs.T

    @property
    def normals(self):
        if self._normals is None:
            _, self._normals, _, _ = MeshFromOBJ.parse_data(self.obj_file)
        return self._normals

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            _, _, self._faces_idxs, _ = MeshFromOBJ.parse_data(self.obj_file)
        return self._faces_idxs

    @property
    def face_normals(self):
        if self._face_normals is None:
            self._face_normals = Mesh.get_face_normals_from_file(self.obj_file)
        return self._face_normals


class MeshFromOFF(Mesh):
    """Construct a Mesh Object from an OFF file
    """
    def __init__(self, off_file):
        self.off_file = off_file

        # Raise Exception in case the given file does not exist
        if not os.path.exists(off_file):
            raise IOException("File does not exist : %s" % (off_file,))
        self._normals = None
        self._faces = None
        self._faces_idxs = None

        # Pointcloud object that will hold the points
        self._pointcloud = None

        self._bbox = None

    @property
    def pointcloud(self):
        if self._pointcloud is None:
            self._pointcloud = PointcloudFromOFF(self.off_file)
        return self._pointcloud

    @staticmethod
    def parse_data(off_file):
        with open(off_file, "r") as f:
            # Read the lines and drop the first to remove the "OFF"
            lines = f.readlines()[1:]
            # Parse the number of vertices and the number of faces from the
            # first line
            n_vertices, n_faces, _ = list(map(int, lines[0].strip().split()))
            vertices = lines[1:n_vertices+1]
            assert len(vertices) == n_vertices
            points = np.array([
                list(map(float, vi))
                for vi in
                [v.strip().split() for v in vertices]
            ])

            faces_idxs = lines[n_vertices+1:]
            assert len(faces_idxs) == n_faces
            faces_idxs = np.array([
                list(map(int, vi))
                for vi in
                [v.strip().split()[1:] for v in faces_idxs]
            ])

            return points.T, None, faces_idxs.T

    @property
    def normals(self):
        if self._normals is None:
            _, self._normals, _ = MeshFromOFF.parse_data(self.off_file)
        return self._normals

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            _, _, self._faces_idxs = MeshFromOFF.parse_data(self.off_file)
        return self._faces_idxs


class Trimesh(Mesh):
    "Wrapper when using the trimesh library"
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        # Raise Exception in case the given file does not exist
        if not os.path.exists(mesh_file):
            raise ValueError("File does not exist : %s" % (mesh_file,))
        self._mesh = None

        self._normals = None
        self._faces = None
        self._faces_idxs = None
        self._face_normals = None

        # Pointcloud object that will hold the points
        self._pointcloud = None
        self._points = None

        self._bbox = None

    def _normalize_points(self):
        """Make sure that points lie in the unit cube."""
        points = self.mesh.vertices.T
        mins = np.min(points, axis=1, keepdims=True)
        steps = np.max(points, axis=1, keepdims=True) - mins
        points -= mins + steps/2
        if steps.max() > 1:
            points /= steps.max()

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = trimesh.load(self.mesh_file)
            # Make sure that the face orinetations are ok
            # trimesh.repair.fix_normals(self.mesh, multibody=True)
            # trimesh.repair.fix_winding(self.mesh)
            # assert self.mesh.is_winding_consistent == True
            # Normalize the points to be in the unit cube
            self._normalize_points()

        return self._mesh

    def contains(self, points):
        return self.mesh.contains(points)

    @property
    def faces_idxs(self):
        if self._faces_idxs is None:
            self._faces_idxs = self.mesh.faces.T
        return self._faces_idxs

    @property
    def face_normals(self):
        if self._face_normals is None:
            self._face_normals = self.mesh.face_normals
        return self._face_normals

    @property
    def points(self):
        if self._points is None:
            self._points = self.mesh.vertices.T

        return self._points

    @property
    def normals(self):
        if self._normals is None:
            self._normals = self.mesh.vertex_normals.T
        return self._normals

    @property
    def pointcloud(self):
        if self._pointcloud is None:
            self._pointcloud = Pointcloud(self.points)
        return self._pointcloud

    @property
    def bounding_box(self):
        if self._bbox is None:
            # A numpy array of size 2x3 containing the bbox that contains the
            # mesh
            self._bbox = self.mesh.bounds
        return self._bbox

    def sample_faces(self, N=10000):
        P, t = trimesh.sample.sample_surface(self.mesh, N)
        return np.hstack([
            P, self.face_normals[t, :]
        ])


class MeshFactory(object):
    """Static factory methods collected under the MeshFactory namespace."""
    @staticmethod
    def from_file(filepath):
        return Trimesh(filepath)
