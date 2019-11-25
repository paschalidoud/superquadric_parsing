import numpy as np
import re
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
import os
import pickle
from PIL import Image

from ..mesh import MeshFactory


class BaseModel(object):
    """BaseModel class is wrapper for all models, independent of dataset. Every
       model has a unique model_tag, mesh_file and Mesh object. Optionally, it
       can also have a tsdf file.
    """
    def __init__(self, tag, path_to_mesh_file, path_to_tsdf_file,
                 mesh=None, images_dir=None):
        self._path_to_tsdf_file = path_to_tsdf_file
        self._path_to_mesh_file = path_to_mesh_file
        self._tag = tag
        self.images_dir = images_dir

        # A Mesh object
        self._gt_mesh = mesh
        # A numpy array containing the TSDF
        self._tsdf = None
        # A list containing the images
        self._images = []
        self._points_on_surface = None
        # Variable to cache mesh internal points
        self._internal_points = None

    @property
    def tag(self):
        return self._tag

    @property
    def path_to_mesh_file(self):
        return self._path_to_mesh_file

    @property
    def path_to_tsdf_file(self):
        return self._path_to_tsdf_file

    @property
    def groundtruth_mesh(self):
        if self._gt_mesh is None:
            self._gt_mesh = MeshFactory.from_file(self._path_to_mesh_file)
        return self._gt_mesh

    @groundtruth_mesh.setter
    def groundtruth_mesh(self, mesh):
        if self._gt_mesh is not None:
            raise RuntimeError("Trying to overwrite a mesh")
        self._gt_mesh = mesh

    @property
    def tsdf(self):
        if self._tsdf is None:
            # Check that the file exists
            if not os.path.isfile(self._path_to_tsdf_file):
                raise Exception(
                    "The mesh path %s does not exist" % (
                        self._path_to_tsdf_file
                    )
                )
            self._tsdf = np.load(self._path_to_tsdf_file)
        return self._tsdf

    def _n_images(self):
        return len(os.listdir(self.images_dir))

    @property
    def images(self):
        if not self._images:
            for path_to_img in sorted(os.listdir(self.images_dir)):
                if path_to_img.endswith((".jpg", ".png")):
                    image_path = os.path.join(self.images_dir, path_to_img)
                    self._images.append(
                        np.array(Image.open(image_path).convert("RGB"))
                    )
        return self._images

    @property
    def random_image(self):
        ri = np.random.choice(len(self.images))
        return self.images[ri]


class ModelsCollection(object):
    def __len__(self):
        raise NotImplementedError()

    def _get_model(self, i):
        raise NotImplementedError()

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self._get_model(i)


class ModelsSubset(ModelsCollection):
    def __init__(self, collection, subset):
        self._collection = collection
        self._subset = subset

    def __len__(self):
        return len(self._subset)

    def _get_sample(self, i):
        return self._collection[self._subset[i]]

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self._get_sample(i)


class TagSubset(ModelsSubset):
    def __init__(self, collection, tags):
        subset = [i for (i, m) in enumerate(collection) if m.tag in tags]
        super(TagSubset, self).__init__(collection, subset)


class ShapeNetQuad(ModelsCollection):
    def __init__(self, base_dir):
        self._tags = sorted(os.listdir(base_dir))
        self._paths = [os.path.join(base_dir, x) for x in self._tags]

        print("Found {} 'ShapeNetQuad' models".format(len(self)))

    def __len__(self):
        return len(self._paths)

    def _get_model(self, i):
        return BaseModel(
            self._tags[i],
            os.path.join(self._paths[i], "model.obj"),
            None
        )


class ShapeNetV1(ModelsCollection):
    def __init__(self, base_dir):
        self._tags = sorted(
            x for x in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, x))
        )
        self._paths = [os.path.join(base_dir, x) for x in self._tags]

        print("Found {} 'ShapeNetV1' models".format(len(self)))

    def __len__(self):
        return len(self._paths)

    def _get_model(self, i):
        return BaseModel(
            self._tags[i],
            os.path.join(self._paths[i], "model_watertight.off"),
            None,
            images_dir=os.path.join(self._paths[i], "img_choy2016")
        )


class ShapeNetV2(ModelsCollection):
    def __init__(self, base_dir):
        self._tags = sorted(
            x for x in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, x))
        )
        self._paths = [os.path.join(base_dir, x) for x in self._tags]

        print("Found {} 'ShapeNetV2' models".format(len(self)))

    def __len__(self):
        return len(self._paths)

    def _get_model(self, i):
        return BaseModel(
            self._tags[i],
            os.path.join(self._paths[i], "models", "model_normalized.obj"),
            None
        )


class SurrealHumanBodies(ModelsCollection):
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._tags = sorted({x[:6] for x in os.listdir(base_dir)})

        print("Found {} 'Surreal' models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return BaseModel(
            self._tags[i],
            os.path.join(self._base_dir, "{}.obj".format(self._tags[i])),
            None
        )


class DynamicFaust(ModelsCollection):
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._paths = sorted([
            d
            for d in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, d))
        ])

        self._tags = sorted([
            "{}:{}".format(d, l[:-4]) for d in self._paths
            for l in os.listdir(os.path.join(self._base_dir, d, "mesh_seq"))
        ])

        print("Found {} 'Surreal' models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        tag_parts = self._tags[i].split(":")
        model_dir = os.path.join(self._base_dir, tag_parts[0])
        return BaseModel(
            self._tags[i],
            os.path.join(model_dir, "mesh_seq", "{}.obj".format(tag_parts[1])),
            None
        )


class MeshCache(ModelsCollection):
    """Cache the meshes from a collection and give them to the model before
    returning it."""
    def __init__(self, collection):
        self._collection = collection
        self._meshes = [None]*len(collection)

    def __len__(self):
        return len(self._collection)

    def _get_model(self, i):
        model = self._collection._get_model(i)
        if self._meshes[i] is not None:
            model.groundtruth_mesh = self._meshes[i]
        else:
            self._meshes[i] = model.groundtruth_mesh

        return model


class LRUCache(ModelsCollection):
    def __init__(self, collection, n=2000):
        self._collection = collection
        self._model_getter = lru_cache(n)(self._inner_get_model)

    def __len__(self):
        return len(self._collection)

    def _inner_get_model(self, i):
        return self._collection._get_model(i)

    def _get_model(self, i):
        return self._model_getter(i)


def model_factory(dataset_type):
    return {
        "shapenet_quad": ShapeNetQuad,
        "shapenet_v1": ShapeNetV1,
        "shapenet_v2": ShapeNetV2,
        "surreal_bodies": SurrealHumanBodies,
        "dynamic_faust": DynamicFaust
    }[dataset_type]


class DatasetBuilder(object):
    def __init__(self):
        self._dataset_class = None
        self._cache_meshes = False
        self._lru_cache = 0
        self._tags = []

    def with_dataset(self, dataset):
        self._dataset_class = model_factory(dataset)
        return self

    def with_cache_meshes(self):
        self._cache_meshes = True
        return self

    def without_cache_meshes(self):
        self._cache_meshes = False
        return self

    def lru_cache(self, n=2000):
        self._lru_cache = n
        return self

    def filter_tags(self, tags):
        self._tags = tags
        return self

    def build(self, base_dir):
        dataset = self._dataset_class(base_dir)
        if self._cache_meshes:
            dataset = MeshCache(dataset)
        if self._lru_cache > 0:
            dataset = LRUCache(dataset, self._lru_cache)
        if self._tags:
            dataset = TagSubset(dataset, self._tags)

        return dataset
