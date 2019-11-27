import pandas as pd
import pickle

from functools import reduce
import seaborn as sns
sns.set()


def parse_train_test_splits(train_test_splits_file, model_tags):
    splits = {}
    if not train_test_splits_file.endswith("csv"):
        raise Exception("Input file %s is not csv" % (train_test_splits_file,))
    df = pd.read_csv(
        train_test_splits_file,
        names=["id", "synsetId", "subSynsetId", "modelId", "split"]
    )
    keep_from_model = reduce(
        lambda a, x: a | (df["synsetId"] in x),
        model_tags,
        False
    )
    # Keep only the rows from the model we want
    df_from_model = df[keep_from_model]

    train_idxs = df_from_model["split"] == "train"
    splits["train"] = df_from_model[train_idxs].modelId.values.tolist()
    test_idxs = df_from_model["split"] == "test"
    splits["test"] = df_from_model[test_idxs].modelId.values.tolist()
    val_idxs = df_from_model["split"] == "val"
    splits["val"] = df_from_model[val_idxs].modelId.values.tolist()

    return splits


def get_colors(M):
    return sns.color_palette("Paired")


def store_primitive_parameters(
    size,
    shape,
    rotation,
    location,
    tapering,
    probability,
    color,
    filepath
):
    primitive_params = dict(
        size=size,
        shape=shape,
        rotation=rotation,
        location=location,
        tapering=tapering,
        probability=probability,
        color=color
    )
    pickle.dump(
        primitive_params,
        open(filepath, "wb")
    )
