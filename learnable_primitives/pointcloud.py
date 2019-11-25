"""Create a PointCloud object by parsing a point cloud from either a .ply file
or an .obj file.
"""

import sys

import numpy as np
from sklearn.neighbors import KDTree

from matplotlib.cm import get_cmap


class PLYHeader(object):
    """Parse a PLY file header into an object"""
    class Element(object):
        def __init__(self, name, count, properties):
            assert len(properties) > 0
            self.name = name
            self.count = count
            self.properties = properties

        @property
        def bytes(self):
            return sum(p.bytes for p in self.properties)

    class Property(object):
        def __init__(self, name, type):
            self.name = name
            self.type = type

        @property
        def bytes(self):
            return {
                "float": 4,
                "uchar": 1,
                "int": 4
            }[self.type]

    def __init__(self, fileobj):
        assert fileobj.readline().strip() == "ply"

        lines = []
        while True:
            l = fileobj.readline()
            if "end_header" in l:
                break
            lines.append(l)

        # Version and format
        identifier, format, version = lines[0].split()
        assert identifier == "format"
        self.is_ascii = "ascii" in format
        self.version = float(version)
        self.little_endian = "little" in format
        lines.pop(0)

        # Comments
        self.comments = [
            x.split(" ", 1)[1]
            for x in lines
            if x.startswith("comment")
        ]

        # Elements
        lines = [l for l in lines if not l.startswith("comment")]
        elements = []
        while lines:
            identifier, name, count = lines[0].split()
            assert identifier == "element"
            count = int(count)
            lines.pop(0)

            properties = []
            while lines:
                identifier, type, name = lines[0].split()
                if identifier != "property":
                    break
                properties.append(self.Property(name, type))
                lines.pop(0)
            elements.append(self.Element(name, count, properties))
        self.elements = elements


class Pointcloud(object):
    """A collection of ND (usually 3D) points that can be searched over and
    saved into a file."""
    def __init__(self, points):
        assert points.shape[0] == 3

        self._points = points
        self._normalize()

    def _normalize(self):
        """Normalize the points so that they are in the unit cube."""
        points = self._points
        mins = np.min(points, axis=1, keepdims=True)
        steps = np.max(points, axis=1, keepdims=True) - mins
        points -= mins + steps/2
        if steps.max() > 1:
            points /= steps.max()

    @property
    def points(self):
        return self._points

    def sample(self, N):
        return self.points[
            :,
            np.random.choice(np.arange(self.points.shape[1]), N)
        ]

    def _add_header(self, N):
        return [
            "ply",
            "format binary_%s_endian 1.0" % (sys.byteorder,),
            "comment Raynet pointcloud!",
            "element vertex %d" % (N,),
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n"
        ]

    def save_obj(self, file):
        with open(file, "w") as f:
            f.write("# OBJ file\n")
            for v in self.points.T:
                f.write("v %.4f %.4f %.4f\n" % tuple(v.tolist()))

    def save_ply(self, file):
        N = self.points.shape[1]
        with open(file, "w") as f:
            header = self._add_header(N)
            f.write("\n".join(header[:7] + header[-1:]))
            self.points.T.astype(np.float32).tofile(f)

    def save_colored_ply(self, file, intensities, colormap="jet"):
        # Get the colormap based on the input
        cmap = get_cmap(colormap)
        # Based on the selected colormap get the the colors for every point
        intensities = intensities / 2
        colors = cmap(intensities.ravel())[:, :-1]
        # The color values need to be uchar
        colors = (colors * 255).astype(np.uint8)

        N = self.points.shape[1]
        # idxs = np.arange(N)[intensities.ravel() < 1.0]
        idxs = np.arange(N)
        with open(file, "w") as f:
            f.write("\n".join(self._add_header(len(idxs))))
            cnt = 0
            # for point, color in zip(self.points.T, colors):
            for i in idxs:
                point = self.points.T[i]
                color = colors[i]
                point.astype(np.float32).tofile(f)
                color.tofile(f)
                cnt += 1

    def save(self, file):
        np.save(file, self.points)

    def filter(self, mask):
        self._points = mask.filter(self.points)

    def index(self, leaf_size=40, metric="minkowski"):
        if hasattr(self, "_index"):
            return

        # NOTE: scikit-learn expects points (samples, features) while we use
        # the more traditional (features, samples)
        self._index = KDTree(self.points.T, leaf_size, metric)

    def nearest_neighbors(self, X, k=1, return_distances=True):
        return self._index.query(X.T, k, return_distances)


class PointcloudFromPLY(Pointcloud):
    """Create a point cloud from a .PLY file
    """
    def __init__(self, ply_file):
        self.ply_file = ply_file
        self._points = None

    @property
    def points(self):
        if self._points is None:
            with open(self.ply_file, "rb") as f:
                header = PLYHeader(f)
                assert len(header.elements) == 1
                el = header.elements[0]
                assert all(p.type == "float" for p in el.properties[:3])

                # Read the data and place one element per line and skip all the
                # extra elements
                data = np.fromfile(f, dtype=np.uint8)
                data = data.reshape(-1, header.elements[0].bytes)
                data =\
                    data[:, :sum(p.bytes for p in el.properties[:3])].ravel()

                # Reread in the correct byte-order
                order = "<" if header.little_endian else ">"
                dtype = order + "f4"
                self._points = np.frombuffer(data.data, dtype=dtype).T
                self._normalize()

        return self._points


class PointcloudFromOBJ(Pointcloud):
    """Create a point cloud from a .OBJ file
    """
    def __init__(self, obj_file):
        self.obj_file = obj_file
        self._points = None

    @property
    def points(self):
        if self._points is None:
            # List to keep the unprocessed lines parsed from the file
            lines = []
            with open(self.obj_file, "r") as f:
                lines = f.readlines()
                # Remove lines containing change of line
                lines = [_f for _f in [x.strip("\r\n") for x in lines] if _f]

                # Keep only the lines that start with the letter v, that
                # corresponds to vertices
                vertices = [k for k in lines if k.startswith("v ")]
                # Remove the unwanted "v" in front of every row and transform
                # it to float
                self._points = np.array([
                        list(map(float, k.strip().split(" ")[1:])) for k in vertices
                ]).T
                self._normalize()

        return self._points


class PointcloudFromOFF(Pointcloud):
    """Create a point cloud from a .OBJ file
    """
    def __init__(self, off_file):
        self.off_file = off_file
        self._points = None

    @property
    def points(self):
        if self._points is None:
            with open(self.off_file, "r") as fp:
                # Read the lines and drop the first to remove the "OFF"
                lines = fp.readlines()[1:]
                # Parse the number of vertices and the number of faces from the
                # first line
                n_vertices, n_faces, _ = list(map(int, lines[0].strip().split()))
                vertices = lines[1:n_vertices+1]
                assert len(vertices) == n_vertices
                self._points = np.array([
                    list(map(float, vi))
                    for vi in
                    [v.strip().split() for v in vertices]
                ]).T
                self._normalize()

        return self._points
