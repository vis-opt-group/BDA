"""
Module for loading datasets
"""

import os
import sys
import threading
from collections import defaultdict, OrderedDict
from functools import reduce
from os.path import join

import h5py
import numpy as np
import scipy as sp
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from boml import load_data as dl
from boml.load_data.datasets import MetaDataset
from boml.load_data.datasets.dl_utils import (
    get_rand_state,
    vstack,
    get_data,
    stack_or_concat,
    get_targets,
    as_list,
    get_indices_balanced_classes,
    test_if_balanced,
    to_one_hot_enc,
    merge_dicts,
    as_tuple_or_list,
)

try:
    from sklearn.datasets import make_classification, make_regression
except ImportError as _err:
    print(_err, file=sys.stderr)
    make_regression, make_classification = None, None

from_env = os.getenv("DATASETS_FOLDER")
if from_env:
    DATA_FOLDER = from_env
else:
    print(
        "Environment variable DATASETS_FOLDER not found. Variables HELP_WIN and HELP_UBUNTU contain info."
    )
    DATA_FOLDER = os.getcwd()
    _COMMON_BEGIN = (
        "You can set environment variable DATASETS_FOLDER to"
        "specify root folder in which you store various datasets. \n"
    )
    _COMMON_END = """\n
    You can also skip this step... \n
    In this case all load_* methods take a FOLDER path as first argument. \n
    Bye."""
    HELP_UBUNTU = (
        _COMMON_BEGIN
        + """
    Bash command is: export DATASETS_FOLDER='absolute/path/to/dataset/folder \n
    Remember! To add the global variable kinda permanently in your system you should add export command in
    bash.bashrc file located in etc folder, if you want to set it globally, or .bashrc in your home directory
    if you want to set it only locally.
    """
        + _COMMON_END
    )

    HELP_WIN = (
        _COMMON_BEGIN
        + """
    Cmd command is: Set DATASETS_FOLDER absolute/path/to/dataset/folder  for one session. \n
    To set it permanently use SetX instead of Set (and probably reboot system)
    """
        + _COMMON_END
    )

print("Data folder is", DATA_FOLDER)

# kind of private
TIMIT_DIR = os.path.join(DATA_FOLDER, "timit4python")
XRMB_DIR = os.path.join(DATA_FOLDER, "XRMB")
IROS15_BASE_FOLDER = os.path.join(
    DATA_FOLDER, os.path.join("dls_collaboration", "Learning")
)

# easy to find!
IRIS_TRAINING = os.path.join(DATA_FOLDER, "iris", "training.csv")
IRIS_TEST = os.path.join(DATA_FOLDER, "iris", "test.csv")
MNIST_DIR = os.path.join(DATA_FOLDER, "mnist_data")
CALTECH101_30_DIR = os.path.join(DATA_FOLDER, "caltech101-30")
CALTECH101_DIR = os.path.join(DATA_FOLDER, "caltech")
CENSUS_TRAIN = os.path.join(DATA_FOLDER, "census", "train.csv")
CENSUS_TEST = os.path.join(DATA_FOLDER, "census", "test.csv")
CIFAR10_DIR = os.path.join(DATA_FOLDER, "CIFAR-10")
CIFAR100_DIR = os.path.join(DATA_FOLDER, "CIFAR-100")
REALSIM = os.path.join(DATA_FOLDER, "realsim")

# scikit learn datasets
SCIKIT_LEARN_DATA = os.path.join(DATA_FOLDER, "scikit_learn_data")

IMAGENET_BASE_FOLDER = join(DATA_FOLDER, "imagenet")
MINI_IMAGENET_FOLDER = join(DATA_FOLDER, join("imagenet", "mini_v1"))
MINI_IMAGENET_FOLDER_RES84 = join(DATA_FOLDER, join("imagenet", "mini_res84"))
MINI_IMAGENET_FOLDER_V2 = join(DATA_FOLDER, join("imagenet", "mini_v2"))
MINI_IMAGENET_FOLDER_V3 = join(DATA_FOLDER, join("imagenet", "mini_v3"))

OMNIGLOT_RESIZED = join(DATA_FOLDER, "omniglot_resized")


def balanced_choice_wr(a, num, rand=None):
    rand = get_rand_state(rand)
    lst = [len(a)] * (num // len(a)) + [num % len(a)]
    return np.concatenate([rand.choice(a, size=(d,), replace=False) for d in lst])


def mnist(
    folder=None, one_hot=True, partitions=None, filters=None, maps=None, shuffle=False
):
    if not folder:
        folder = MNIST_DIR
    datasets = read_data_sets(folder, one_hot=one_hot)
    train = dl.Dataset(datasets.train.images, datasets.train.labels, name="MNIST")
    validation = dl.Dataset(
        datasets.validation.images, datasets.validation.labels, name="MNIST"
    )
    test = dl.Dataset(datasets.test.images, datasets.test.labels, name="MNIST")
    res = [train, validation, test]
    if partitions:
        res = redivide_data(
            res,
            partition_proportions=partitions,
            filters=filters,
            maps=maps,
            shuffle=shuffle,
        )
        res += [None] * (3 - len(res))
    return dl.Datasets.from_list(res)


def omni_light(folder=join(DATA_FOLDER, "omniglot-light"), add_bias=False):
    """
    Extract from omniglot dataset with rotated images, 100 classes,
    3 examples per class in training set
    3 examples per class in validation set
    15 examples per class in test set
    """
    file = h5py.File(os.path.join(folder, "omni-light.h5"), "r")
    return dl.Datasets.from_list(
        [
            dl.Dataset(
                np.array(file["X_ft_tr"]),
                np.array(file["Y_tr"]),
                info={"original images": np.array(file["X_orig_tr"])},
                add_bias=add_bias,
            ),
            dl.Dataset(
                np.array(file["X_ft_val"]),
                np.array(file["Y_val"]),
                info={"original images": np.array(file["X_orig_val"])},
                add_bias=add_bias,
            ),
            dl.Dataset(
                np.array(file["X_ft_test"]),
                np.array(file["Y_test"]),
                info={"original images": np.array(file["X_orig_test"])},
                add_bias=add_bias,
            ),
        ]
    )


load_omni_light = omni_light


class OmniglotMetaDataset(MetaDataset):
    def __init__(
        self,
        info=None,
        rotations=None,
        name="Omniglot",
        num_classes=None,
        num_examples=None,
    ):
        super().__init__(
            info, name=name, num_classes=num_classes, num_examples=num_examples
        )
        self._loaded_images = defaultdict(lambda: {})
        self._rotations = rotations or [0, 90, 180, 270]
        self.load_all()

    def generate_datasets(self, rand=None, num_classes=None, num_examples=None):
        rand = dl.get_rand_state(rand)

        if not num_examples:
            num_examples = self.kwargs["num_examples"]
        if not num_classes:
            num_classes = self.kwargs["num_classes"]

        clss = self._loaded_images if self._loaded_images else self.info["classes"]

        random_classes = rand.choice(
            list(clss.keys()), size=(num_classes,), replace=False
        )
        rand_class_dict = {rnd: k for k, rnd in enumerate(random_classes)}

        _dts = []
        for ns in as_tuple_or_list(num_examples):
            classes = balanced_choice_wr(random_classes, ns, rand)

            all_images = {cls: list(clss[cls]) for cls in classes}
            data, targets, sample_info = [], [], []
            for c in classes:
                rand.shuffle(all_images[c])
                img_name = all_images[c][0]
                all_images[c].remove(img_name)
                sample_info.append({"name": img_name, "label": c})
                data.append(clss[c][img_name])
                targets.append(rand_class_dict[c])

            if self.info["one_hot_enc"]:
                targets = dl.to_one_hot_enc(targets, dimension=num_classes)

            _dts.append(
                dl.Dataset(
                    data=np.array(np.stack(data)),
                    target=targets,
                    sample_info=sample_info,
                    info={"all_classes": random_classes},
                )
            )
        return dl.Datasets.from_list(_dts)

    def load_all(self):
        from imageio import imread
        from scipy.ndimage.interpolation import rotate

        _cls = self.info["classes"]
        _base_folder = self.info["base_folder"]

        for c in _cls:
            all_images = list(_cls[c])
            for img_name in all_images:
                img = imread(join(_base_folder, join(c, img_name)))
                img = 1.0 - np.reshape(img, (28, 28, 1)) / 255.0
                for rot in self._rotations:
                    img = rotate(img, rot, reshape=False)
                    self._loaded_images[c + os.path.sep + "rot_" + str(rot)][
                        img_name
                    ] = img


def meta_omniglot(
    folder=OMNIGLOT_RESIZED,
    std_num_classes=None,
    examples_train=None,
    examples_test=None,
    one_hot_enc=True,
    _rand=0,
    n_splits=None,
):
    """
    Loading function for Omniglot dataset in learning-to-learn version. Use image data as obtained from
    https://github.com/cbfinn/maml/blob/master/data/omniglot_resized/resize_images.py

    :param folder: root folder name.
    :param std_num_classes: standard number of classes for N-way classification
    :param examples_train:standard number of examples to be picked in each generated per classes for training
    (eg .1 shot, examples_train=1)
    :param examples_test: standard number of examples to be picked in each generated per classes for testing
    :param one_hot_enc: one hot encoding
    :param _rand: random seed or RandomState for generate training, validation, testing meta-datasets
                    split
    :param n_splits: num classes per split
    :return: a Datasets of MetaDataset s
    """
    std_num_examples = (
        examples_train * std_num_classes,
        examples_test * std_num_classes,
    )
    alphabets = os.listdir(folder)

    labels_and_images = OrderedDict()
    for alphabet in alphabets:
        base_folder = join(folder, alphabet)
        label_names = os.listdir(base_folder)  # all characters in one alphabet
        labels_and_images.update(
            {
                alphabet + os.path.sep + ln: os.listdir(join(base_folder, ln))
                # all examples of each character
                for ln in label_names
            }
        )

    # divide between training validation and test meta-datasets
    _rand = dl.get_rand_state(_rand)
    all_clss = list(labels_and_images.keys())
    _rand.shuffle(all_clss)
    n_splits = n_splits or (0, 1200, 1300, len(all_clss))

    meta_dts = []
    for start, end in zip(n_splits, n_splits[1:]):
        meta_dts.append(
            OmniglotMetaDataset(
                info={
                    "base_folder": folder,
                    "classes": {k: labels_and_images[k] for k in all_clss[start:end]},
                    "one_hot_enc": one_hot_enc,
                },
                num_classes=std_num_classes,
                num_examples=std_num_examples,
            )
        )

    return dl.Datasets.from_list(meta_dts)


def meta_omniglot_v2(
    folder=OMNIGLOT_RESIZED,
    std_num_classes=None,
    examples_train=None,
    examples_test=None,
    one_hot_enc=True,
    _rand=0,
    n_splits=None,
):
    """
    Loading function for Omniglot dataset in learning-to-learn version. Use image data as obtained from
    https://github.com/cbfinn/maml/blob/master/data/omniglot_resized/resize_images.py

    :param folder: root folder name.
    :param std_num_classes: standard number of classes for N-way classification
    :param examples_train:standard number of examples to be picked in each generated per classes for training
    (eg .1 shot, examples_train=1)
    :param examples_test: standard number of examples to be picked in each generated per classes for testing
    :param one_hot_enc: one hot encoding
    :param _rand: random seed or RandomState for generate training, validation, testing meta-datasets
                    split
    :param n_splits: num classes per split
    :return: a Datasets of MetaDataset s
    """

    class OmniglotMetaDataset(MetaDataset):
        def __init__(
            self,
            info=None,
            rotations=None,
            name="Omniglot",
            num_classes=None,
            num_examples=None,
        ):
            super().__init__(
                info, name=name, num_classes=num_classes, num_examples=num_examples
            )
            self._loaded_images = defaultdict(lambda: {})
            self._rotations = rotations or [0, 90, 180, 270]
            self._img_array = None
            self.load_all()

        def generate_datasets(self, rand=None, num_classes=None, num_examples=None):
            rand = dl.get_rand_state(rand)

            if not num_examples:
                num_examples = self.kwargs["num_examples"]
            if not num_classes:
                num_classes = self.kwargs["num_classes"]

            clss = self._loaded_images if self._loaded_images else self.info["classes"]

            random_classes = rand.choice(
                list(clss.keys()), size=(num_classes,), replace=False
            )
            rand_class_dict = {rnd: k for k, rnd in enumerate(random_classes)}

            _dts = []
            for ns in dl.as_tuple_or_list(num_examples):
                classes = balanced_choice_wr(random_classes, ns, rand)

                all_images = {cls: list(clss[cls]) for cls in classes}
                indices, targets = [], []
                for c in classes:
                    rand.shuffle(all_images[c])
                    img_name = all_images[c][0]
                    all_images[c].remove(img_name)
                    # sample_info.append({'name': img_name, 'label': c})
                    indices.append(clss[c][img_name])
                    targets.append(rand_class_dict[c])

                if self.info["one_hot_enc"]:
                    targets = dl.to_one_hot_enc(targets, dimension=num_classes)

                data = self._img_array[indices]

                _dts.append(dl.Dataset(data=data, target=targets))
            return dl.Datasets.from_list(_dts)

        def load_all(self):
            from scipy.ndimage import imread
            from scipy.ndimage.interpolation import rotate

            _cls = self.info["classes"]
            _base_folder = self.info["base_folder"]

            _id = 0
            flat_data = []
            flat_targets = []
            for c in _cls:
                all_images = list(_cls[c])
                for img_name in all_images:
                    img = imread(join(_base_folder, join(c, img_name)))
                    img = 1.0 - np.reshape(img, (28, 28, 1)) / 255.0
                    for rot in self._rotations:
                        img = rotate(img, rot, reshape=False)
                        self._loaded_images[c + os.path.sep + "rot_" + str(rot)][
                            img_name
                        ] = _id
                        _id += 1
                        flat_data.append(img)
                        # flat_targets maybe... no flat targets... they depend on the episode!!

            self._img_array = np.stack(flat_data)

            # end of class

    std_num_examples = (
        examples_train * std_num_classes,
        examples_test * std_num_classes,
    )
    alphabets = os.listdir(folder)

    labels_and_images = OrderedDict()
    for alphabet in alphabets:
        base_folder = join(folder, alphabet)
        label_names = os.listdir(base_folder)  # all characters in one alphabet
        labels_and_images.update(
            {
                alphabet + os.path.sep + ln: os.listdir(join(base_folder, ln))
                # all examples of each character
                for ln in label_names
            }
        )

    # divide between training validation and test meta-datasets
    _rand = dl.get_rand_state(_rand)
    all_clss = list(labels_and_images.keys())
    _rand.shuffle(all_clss)
    n_splits = n_splits or (0, 1200, 1300, len(all_clss))

    meta_dts = []
    for start, end in zip(n_splits, n_splits[1:]):
        meta_dts.append(
            OmniglotMetaDataset(
                info={
                    "base_folder": folder,
                    "classes": {k: labels_and_images[k] for k in all_clss[start:end]},
                    "one_hot_enc": one_hot_enc,
                },
                num_classes=std_num_classes,
                num_examples=std_num_examples,
            )
        )

    return dl.Datasets.from_list(meta_dts)


class ImageNetMetaDataset(MetaDataset):
    def __init__(
        self,
        info=None,
        name="MiniImagenet",
        num_classes=None,
        num_examples=None,
        h5=False,
    ):

        super().__init__(
            info, name=name, num_classes=num_classes, num_examples=num_examples
        )
        self._loaded_images = defaultdict(lambda: {})
        self._threads = []
        self.h5 = h5

    def load_all_images(self):
        if self.h5:
            _file = self.info["file"]
            h5m = h5py.File(_file, "r")
            img_per_class = 600
            for j in range(len(h5m["X"])):
                self._loaded_images[j // img_per_class][j] = (
                    np.array(h5m["X"][j], dtype=np.float32) / 255.0
                )
                # images were stored as int
        else:
            from imageio import imread

            # from scipy.misc import imresize
            _cls = self.info["classes"]
            _base_folder = self.info["base_folder"]

            def _load_class(c):
                all_images = list(_cls[c])
                for img_name in all_images:
                    img = imread(join(_base_folder, join(c, img_name)))
                    if self.info["resize"]:
                        # noinspection PyTypeChecker
                        # img = imresize(img, size=(self.info['resize'], self.info['resize'], 3)) / 255.
                        img = (
                            np.array(
                                Image.fromarray(img).resize(
                                    size=(self.info["resize"], self.info["resize"])
                                )
                            )
                            / 255.0
                        )
                    self._loaded_images[c][img_name] = img

            for cls in _cls:
                self._threads.append(
                    threading.Thread(target=lambda: _load_class(cls), daemon=True)
                )
                self._threads[-1].start()

    def check_loaded_images(self, n_min):
        return self._loaded_images and all(
            [len(v) >= n_min for v in self._loaded_images.values()]
        )

    def generate_datasets(
        self, rand=None, num_classes=None, num_examples=None, wait_for_n_min=None
    ):

        rand = dl.get_rand_state(rand)

        if wait_for_n_min:
            import time

            while not self.check_loaded_images(wait_for_n_min):
                time.sleep(5)

        if not num_examples:
            num_examples = self.kwargs["num_examples"]
        if not num_classes:
            num_classes = self.kwargs["num_classes"]

        clss = self._loaded_images if self._loaded_images else self.info["classes"]

        random_classes = rand.choice(
            list(clss.keys()), size=(num_classes,), replace=False
        )
        rand_class_dict = {rnd: k for k, rnd in enumerate(random_classes)}

        _dts = []
        for ns in dl.as_tuple_or_list(num_examples):
            classes = balanced_choice_wr(random_classes, ns, rand)

            all_images = {cls: list(clss[cls]) for cls in classes}
            data, targets, sample_info = [], [], []
            for c in classes:
                rand.shuffle(all_images[c])
                img_name = all_images[c][0]
                all_images[c].remove(img_name)
                sample_info.append({"name": img_name, "label": c})

                if self._loaded_images:
                    data.append(clss[c][img_name])
                else:
                    from imageio import imread

                    data.append(
                        np.array(
                            Image.fromarray(
                                imread(
                                    join(self.info["base_folder"], join(c, img_name))
                                )
                            ).resize(size=(self.info["resize"], self.info["resize"]))
                        )
                        / 255.0
                    )
                targets.append(rand_class_dict[c])

            if self.info["one_hot_enc"]:
                targets = to_one_hot_enc(targets, dimension=num_classes)

            _dts.append(
                dl.Dataset(
                    data=np.array(np.stack(data)),
                    target=targets,
                    sample_info=sample_info,
                    info={"all_classes": random_classes},
                )
            )
        return dl.Datasets.from_list(_dts)

    def all_data(self, partition_proportions=None, seed=None):
        if not self._loaded_images:
            self.load_all_images()
            while not self.check_loaded_images(600):
                import time

                time.sleep(5)
        data, targets = [], []
        for k, c in enumerate(sorted(self._loaded_images)):
            data += list(self._loaded_images[c].values())
            targets += [k] * 600
        if self.info["one_hot_enc"]:
            targets = dl.to_one_hot_enc(targets, dimension=len(self._loaded_images))
        _dts = [
            dl.Dataset(
                data=np.stack(data), target=np.array(targets), name="MiniImagenet_full"
            )
        ]
        if seed:
            np.random.seed(seed)
        if partition_proportions:
            _dts = redivide_data(
                _dts, partition_proportions=partition_proportions, shuffle=True
            )
        return dl.Datasets.from_list(_dts)


def meta_mini_imagenet(
    folder=MINI_IMAGENET_FOLDER_V3,
    sub_folders=None,
    std_num_classes=None,
    examples_train=None,
    examples_test=None,
    resize=84,
    one_hot_enc=True,
    load_all_images=True,
    h5=False,
):
    """
    Load a meta-datasets from Mini-ImageNet. Returns a Datasets of MetaDataset s,

    :param folder: base folder
    :param sub_folders: optional sub-folders in which data is locate
    :param std_num_classes: standard number of classes to be included in each generated per dataset
                            (can be None, default)
    :param examples_train:standard number of examples to be picked in each generated per classes for training
    :param examples_test: standard number of examples to be picked in each generated per classes for testing
    :param resize:  resizing dimension
    :param one_hot_enc:
    :param load_all_images:
    :param h5:  True (default) to use HDF5 files, when False search for JPEG images.
    :return:
    """
    std_num_examples = (
        examples_train * std_num_classes,
        examples_test * std_num_classes,
    )

    if sub_folders is None:
        sub_folders = ["train", "val", "test"]
    meta_dts = []
    for ds in sub_folders:
        if not h5:
            base_folder = join(folder, ds)
            label_names = os.listdir(base_folder)
            labels_and_images = {
                ln: os.listdir(join(base_folder, ln)) for ln in label_names
            }
            meta_dts.append(
                ImageNetMetaDataset(
                    info={
                        "base_folder": base_folder,
                        "classes": labels_and_images,
                        "resize": resize,
                        "one_hot_enc": one_hot_enc,
                    },
                    num_classes=std_num_classes,
                    num_examples=std_num_examples,
                    h5=False,
                )
            )
        else:
            file = join(folder, ds + ".h5")
            meta_dts.append(
                ImageNetMetaDataset(
                    info={"file": file, "one_hot_enc": one_hot_enc},
                    num_classes=std_num_classes,
                    num_examples=std_num_examples,
                    h5=True,
                )
            )

    dts = dl.Datasets.from_list(meta_dts)
    if load_all_images:
        import time

        [_d.load_all_images() for _d in dts]
        _check_available = lambda min_num: [
            _d.check_loaded_images(min_num) for _d in dts
        ]
        while not all(_check_available(15)):
            time.sleep(
                1
            )  # be sure that there are at least 15 images per class in each meta-dataset
    return dts


def random_classification_datasets(
    n_samples,
    features=100,
    classes=2,
    informative=0.1,
    partition_proportions=(0.5, 0.3),
    rnd=None,
    one_hot=True,
    **mk_cls_kwargs
):
    rnd_state = dl.get_rand_state(rnd)
    X, Y = make_classification(
        n_samples, features, n_classes=classes, random_state=rnd_state, **mk_cls_kwargs
    )
    if one_hot:
        Y = to_one_hot_enc(Y)

    print("range of Y", np.min(Y), np.max(Y))
    info = merge_dicts({"informative": informative, "random_seed": rnd}, mk_cls_kwargs)
    name = dl.em_utils.name_from_dict(info, "w")
    dt = dl.Dataset(X, Y, name=name, info=info)
    datasets = dl.Datasets.from_list(redivide_data([dt], partition_proportions))
    print(
        "conditioning of X^T X",
        np.linalg.cond(datasets.train.data.T @ datasets.train.data),
    )
    return datasets


def random_regression_datasets(
    n_samples,
    features=100,
    outs=1,
    informative=0.1,
    partition_proportions=(0.5, 0.3),
    rnd=None,
    **mk_rgr_kwargs
):
    rnd_state = dl.get_rand_state(rnd)
    X, Y, w = make_regression(
        n_samples,
        features,
        int(features * informative),
        outs,
        random_state=rnd_state,
        coef=True,
        **mk_rgr_kwargs
    )
    if outs == 1:
        Y = np.reshape(Y, (n_samples, 1))

    print("range of Y", np.min(Y), np.max(Y))
    info = merge_dicts(
        {"informative": informative, "random_seed": rnd, "w": w}, mk_rgr_kwargs
    )
    name = dl.em_utils.name_from_dict(info, "w")
    dt = dl.Dataset(X, Y, name=name, info=info)
    datasets = dl.Datasets.from_list(redivide_data([dt], partition_proportions))
    print(
        "conditioning of X^T X",
        np.linalg.cond(datasets.train.data.T @ datasets.train.data),
    )
    return datasets


def redivide_data(
    datasets,
    partition_proportions=None,
    shuffle=False,
    filters=None,
    maps=None,
    balance_classes=False,
    rand=None,
):
    """
    Function that redivides datasets. Can be use also to shuffle or filter or map examples.

    :param rand:
    :param balance_classes: # TODO RICCARDO
    :param datasets: original datasets, instances of class Dataset (works with get_data and get_targets for
                        compatibility with mnist datasets
    :param partition_proportions: (optional, default None)  list of fractions that can either sum up to 1 or less
                                    then one, in which case one additional partition is created with
                                    proportion 1 - sum(partition proportions).
                                    If None it will retain the same proportion of samples found in datasets
    :param shuffle: (optional, default False) if True shuffles the examples
    :param filters: (optional, default None) filter or list of filters: functions with signature
                        (data, target, index) -> boolean (accept or reject the sample)
    :param maps: (optional, default None) map or list of maps: functions with signature
                        (data, target, index) ->  (new_data, new_target) (maps the old sample to a new one,
                        possibly also to more
                        than one sample, for data augmentation)
    :return: a list of datasets of length equal to the (possibly augmented) partition_proportion
    """

    rnd = get_rand_state(rand)

    all_data = vstack([get_data(d) for d in datasets])
    all_labels = stack_or_concat([get_targets(d) for d in datasets])

    all_infos = np.concatenate([d.sample_info for d in datasets])

    N = all_data.shape[0]

    if partition_proportions:  # argument check
        partition_proportions = list(
            [partition_proportions]
            if isinstance(partition_proportions, float)
            else partition_proportions
        )
        sum_proportions = sum(partition_proportions)
        assert sum_proportions <= 1, (
            "partition proportions must sum up to at most one: %d" % sum_proportions
        )
        if sum_proportions < 1.0:
            partition_proportions += [1.0 - sum_proportions]
    else:
        partition_proportions = [1.0 * get_data(d).shape[0] / N for d in datasets]

    if shuffle:
        if sp and isinstance(all_data, sp.sparse.csr.csr_matrix):
            raise NotImplementedError()
        # if sk_shuffle:  # TODO this does not work!!! find a way to shuffle these matrices while
        # keeping compatibility with tensorflow!
        #     all_data, all_labels, all_infos = sk_shuffle(all_data, all_labels, all_infos)
        # else:
        permutation = np.arange(all_data.shape[0])
        rnd.shuffle(permutation)

        all_data = all_data[permutation]
        all_labels = np.array(all_labels[permutation])
        all_infos = np.array(all_infos[permutation])

    if filters:
        if sp and isinstance(all_data, sp.sparse.csr.csr_matrix):
            raise NotImplementedError()
        filters = as_list(filters)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for fiat in filters:
            data_triple = [
                xy for i, xy in enumerate(data_triple) if fiat(xy[0], xy[1], xy[2], i)
            ]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    if maps:
        if sp and isinstance(all_data, sp.sparse.csr.csr_matrix):
            raise NotImplementedError()
        maps = as_list(maps)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for _map in maps:
            data_triple = [
                _map(xy[0], xy[1], xy[2], i) for i, xy in enumerate(data_triple)
            ]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    N = all_data.shape[0]
    assert N == all_labels.shape[0]

    calculated_partitions = reduce(
        lambda v1, v2: v1 + [sum(v1) + v2],
        [int(N * prp) for prp in partition_proportions],
        [0],
    )
    calculated_partitions[-1] = N

    print(
        "datasets.redivide_data:, computed partitions numbers -",
        calculated_partitions,
        "len all",
        N,
        end=" ",
    )

    new_general_info_dict = {}
    for data in datasets:
        new_general_info_dict = {**new_general_info_dict, **data.info}

        if balance_classes:
            new_datasets = []
            forbidden_indices = np.empty(0, dtype=np.int64)
            for d1, d2 in zip(calculated_partitions[:-1], calculated_partitions[1:-1]):
                indices = np.array(
                    get_indices_balanced_classes(d2 - d1, all_labels, forbidden_indices)
                )
                dataset = dl.Dataset(
                    data=all_data[indices],
                    target=all_labels[indices],
                    sample_info=all_infos[indices],
                    info=new_general_info_dict,
                )
                new_datasets.append(dataset)
                forbidden_indices = np.append(forbidden_indices, indices)
                test_if_balanced(dataset)
            remaining_indices = np.array(
                list(set(list(range(N))) - set(forbidden_indices))
            )
            new_datasets.append(
                dl.Dataset(
                    data=all_data[remaining_indices],
                    target=all_labels[remaining_indices],
                    sample_info=all_infos[remaining_indices],
                    info=new_general_info_dict,
                )
            )
        else:
            new_datasets = [
                dl.Dataset(
                    data=all_data[d1:d2],
                    target=all_labels[d1:d2],
                    sample_info=all_infos[d1:d2],
                    info=new_general_info_dict,
                )
                for d1, d2 in zip(calculated_partitions, calculated_partitions[1:])
            ]

        print("DONE")

        return new_datasets


if __name__ == "__main__":
    # pass
    # mmi = meta_mini_imagenet()
    # # st = mmi.train.generate_datasets(num_classes=10, num_examples=(123, 39))
    # d1 = mmi.train.all_data(seed=0)
    # print(d1.train.dim_data, d1.train.dim_target)
    #
    # mmiii = meta_mini_imagenet()
    # # st = mmi.train.generate_datasets(num_classes=10, num_examples=(123, 39))
    # d2 = mmiii.train.all_data(seed=0)
    # print(d2.train.dim_data, d2.train.dim_target)
    #
    # print(np.equal(d1.train.data[0], d2.train.data[0]))

    res = meta_omniglot_v2(std_num_classes=5, std_num_examples=(10, 20))
    lst = []
    while True:
        lst.append(res.train.generate_datasets())
        # print(dt.train.data)
        # print(res)
