"""
Module that contains datasets classes for managing data.
"""
import sys

import numpy as np
import scipy.sparse as sc_sp
import tensorflow as tf

try:
    import intervaltree as it
except ImportError:
    it = None

from boml.load_data.datasets.dl_utils import (
    maybe_cast_to_scalar,
    pad,
    stack_or_concat,
    vstack,
    convert_sparse_matrix_to_sparse_tensor,
    merge_dicts,
    get_rand_state,
    maybe_call,
)


class Datasets:
    """
    Simple object for standard datasets. Has the field `train` `validation` and `test` and support indexing
    """

    def __init__(self, train=None, validation=None, test=None):
        self.train = train
        self.validation = validation
        self.test = test
        self._lst = [train, validation, test]

    def setting(self):
        return {
            k: v.setting() if hasattr(v, "setting") else None
            for k, v in vars(self).items()
        }

    def __getitem__(self, item):
        return self._lst[item]

    def __len__(self):
        return len([_ for _ in self._lst if _ is not None])

    @property
    def name(self):
        return (
            self.train.name
        )  # could be that different datasets have different names....

    @staticmethod
    def from_list(list_of_datasets):
        """
        Generates a `Datasets` object from a list.

        :param list_of_datasets: list containing from one to three dataset
        :return:
        """
        train, valid, test = None, None, None
        train = list_of_datasets[0]
        if len(list_of_datasets) > 3:
            print("There are more then 3 Datasets here...")
            return list_of_datasets
        if len(list_of_datasets) > 1:
            test = list_of_datasets[-1]
            if len(list_of_datasets) == 3:
                valid = list_of_datasets[1]
        return Datasets(train, valid, test)

    @staticmethod
    def stack(*datasets_s):
        """
        Stack some datasets calling stack for each dataset.

        :param datasets_s:
        :return: a new dataset
        """
        return Datasets.from_list(
            [
                Dataset.stack(*[d[k] for d in datasets_s if d[k] is not None])
                for k in range(3)
            ]
        )


NAMED_SUPPLIER = {}


class Dataset:
    """
    Class for managing a single dataset, includes data and target fields and has some utility functions.
     It allows also to convert the dataset into tensors and to store additional information both on a
     per-example basis and general infos.
    """

    def __init__(
        self, data, target, sample_info=None, info=None, name=None, add_bias=False
    ):
        """

        :param data: Numpy array containing data
        :param target: Numpy array containing targets
        :param sample_info: either an array of dicts or a single dict, in which case it is cast to array of
                                  dicts.
        :param info: (optional) dictionary with further info about the dataset
        """
        self._tensor_mode = False
        self._data = data
        self._add_bias = add_bias
        if self._add_bias:
            assert isinstance(
                self.dim_data, int
            ), "Add bias not defined for non vector data"
            self._data = np.hstack((self.data, np.ones((self.num_examples, 1))))

        self._target = target
        if self._data is not None:  # in meta-dataset data and target can be unspecified
            if sample_info is None:
                sample_info = {}
            self.sample_info = (
                np.array([sample_info] * self.num_examples)
                if isinstance(sample_info, dict)
                else sample_info
            )

            assert self.num_examples == len(self.sample_info), (
                str(self.num_examples) + " " + str(len(self.sample_info))
            )
            assert self.num_examples == self._shape(self._target)[0]

        self.info = info or {}
        self.info.setdefault("_name", name)

    @property
    def name(self):
        return self.info["_name"]

    @property
    def bias(self):
        return self._add_bias

    def _shape(self, what):
        return what.get_shape().as_list() if self._tensor_mode else what.shape

    def setting(self):
        """
        for save setting purposes, does not save the actual data

        :return:
        """
        return {
            "num_examples": self.num_examples,
            "dim_data": self.dim_data,
            "dim_target": self.dim_target,
            "info": self.info,
        }

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def num_examples(self):
        """

        :return: Number of examples in this dataset
        """
        return self._shape(self.data)[0]

    @property
    def dim_data(self):
        """

        :return: The data dimensionality as an integer, if input are vectors, or a tuple in the general case
        """
        return maybe_cast_to_scalar(self._shape(self.data)[1:])

    @property
    def dim_target(self):
        """

        :return: The target dimensionality as an integer, if targets are vectors, or a tuple in the general case
        """
        shape = self._shape(self.target)
        return 1 if len(shape) == 1 else maybe_cast_to_scalar(shape[1:])

    def convert_to_tensor(self, keep_sparse=True):
        SPARSE_SCIPY_MATRICES = (sc_sp.csr.csr_matrix, sc_sp.coo.coo_matrix)
        matrices = ["_data", "_target"]
        for att in matrices:
            if keep_sparse and isinstance(
                self.__getattribute__(att), SPARSE_SCIPY_MATRICES
            ):
                self.__setattr__(
                    att,
                    convert_sparse_matrix_to_sparse_tensor(self.__getattribute__(att)),
                )
            else:
                self.__setattr__(
                    att,
                    tf.convert_to_tensor(self.__getattribute__(att), dtype=tf.float32),
                )
        self._tensor_mode = True

    def create_supplier(self, x, y, other_feeds=None, name=None):
        """
        Return a standard feed dictionary for this dataset.

        :param name: if not None, register this supplier in dict NAMED_SUPPLIERS (this can be useful for instance
                        when recording with rf.Saver)
        :param x: placeholder for data
        :param y: placeholder for target
        :param other_feeds: optional other feeds (dictionary or None)
        :return: a callable.
        """
        if not other_feeds:
            other_feeds = {}

        # noinspection PyUnusedLocal
        def _supplier(step=None):
            """

            :param step: unused, just for making it compatible with `HG` and `Saver`
            :return: the feed dictionary
            """
            if isinstance(self.data, WindowedData):
                data = self.data.generate_all()

            return {**{x: self.data, y: self.target}, **other_feeds}

        if name:
            NAMED_SUPPLIER[name] = _supplier

        return _supplier

    @staticmethod
    def stack(*datasets):
        """
        Assuming that the datasets have same structure, stacks data, targets and other info

        :param datasets:
        :return: stacked dataset
        """
        return Dataset(
            data=vstack([d.data for d in datasets]),
            target=stack_or_concat([d.target for d in datasets]),
            sample_info=np.concatenate([d.sample_info for d in datasets]),
            info={
                k: [d.info.get(k, None) for d in datasets]
                for k in merge_dicts(*[d.info for d in datasets])
            },
        )


class MetaDataset(Dataset):
    def __init__(self, info=None, name="MetaDataset", *args, **kwargs):
        super().__init__(None, None, None, info, name=name)
        self.args = args
        self.kwargs = kwargs

    def generate_datasets(self, rand=None, *args, **kwargs):
        """
        Generates and returns a single Datasets (possibly composed by training, validation and test sets)
        according to args and kwargs

        :param rand: random seed, state or None
        :param args:
        :param kwargs:
        :return: `Datasets`
        """
        raise NotImplementedError()

    def generate(self, count, batch_size=1, rand=None, *args, **kwargs):
        """
        Generator of datasets

        :param rand: random seed, state or None
        :param count: number of datasets to generate
        :param batch_size: number of episodes to generate at each call
        :param args:
        :param kwargs:
        :return: one or a list of Datasets objects
        """
        if not args:
            args = self.args
        if not kwargs:
            kwargs = self.kwargs
        rand = get_rand_state(rand)
        for _ in range(count):
            if batch_size == 1:
                yield self.generate_datasets(rand=rand, *args, **kwargs)
            else:
                yield self.generate_batch(batch_size, rand=rand, *args, **kwargs)

    def generate_batch(self, batch_size, rand=None, *args, **kwargs):
        """
        Generates a batch of Datasets
        """
        return [
            self.generate_datasets(rand, *args, **kwargs) for _ in range(batch_size)
        ]

    @property
    def dim_data(self, *args, **kwargs):
        return self.generate_datasets(*args, **kwargs).train.dim_data

    @property
    def dim_target(self, *args, **kwargs):
        return self.generate_datasets(*args, **kwargs).train.dim_target


class WindowedData(object):
    def __init__(self, data, row_sentence_bounds, window=5, process_all=False):
        """
        Class for managing windowed input data (like TIMIT).

        :param data: Numpy matrix. Each row should be an example data
        :param row_sentence_bounds:  Numpy matrix with bounds for padding. TODO add default NONE
        :param window: half-window size
        :param process_all: (default False) if True adds context to all data at object initialization.
                            Otherwise the windowed data is created in runtime.
        """
        assert it is not None, "NEED PACKAGE INTERVALTREE!"
        self.window = window
        self.data = data
        base_shape = self.data.shape
        self.shape = (base_shape[0], (2 * self.window + 1) * base_shape[1])
        self.tree = it.IntervalTree(
            [it.Interval(int(e[0]), int(e[1]) + 1) for e in row_sentence_bounds]
        )
        if process_all:
            print("adding context to all the dataset", end="- ")
            self.data = self.generate_all()
            print("DONE")
        self.process_all = process_all

    def generate_all(self):
        return self[:]

    def __getitem__(self, item):
        if hasattr(self, "process_all") and self.process_all:  # keep attr check!
            return self.data[item]
        if isinstance(item, int):
            return self.get_context(item=item)
        if isinstance(item, tuple):
            if len(item) == 2:
                rows, columns = item
                if isinstance(rows, int) and isinstance(columns, int):
                    # do you want the particular element?
                    return self.get_context(item=rows)[columns]
            else:
                raise TypeError("NOT IMPLEMENTED <|>")
            if isinstance(rows, slice):
                rows = range(*rows.indices(self.shape[0]))
            return np.vstack([self.get_context(r) for r in rows])[:, columns]
        else:
            if isinstance(item, slice):
                item = range(*item.indices(self.shape[0]))
            return np.vstack([self.get_context(r) for r in item])

    def __len__(self):
        return self.shape[0]

    def get_context(self, item):
        interval = list(self.tree[item])[0]
        # print(interval)
        left, right = interval[0], interval[1]
        left_pad = max(self.window + left - item, 0)
        right_pad = max(
            0, self.window - min(right, len(self) - 1) + item
        )  # this is to cope with reduce datasets
        # print(left, right, item)

        # print(left_pad, right_pad)
        base = np.concatenate(
            self.data[
                item - self.window + left_pad : item + self.window + 1 - right_pad
            ]
        )
        if left_pad:
            base = np.concatenate([pad(self.data[item], left_pad), base])
        if right_pad:
            base = np.concatenate([base, pad(self.data[item], right_pad)])
        return base


class ExampleVisiting:
    def __init__(self, dataset, batch_size, epochs=None, rnd=None):
        """
        Class for stochastic sampling of data points. It is most useful for feeding examples for the the
        training ops of `ReverseHG` or `ForwardHG`. Most notably, if the number of epochs is specified,
        the class takes track of the examples per mini-batches which is important for the backward pass
        of `ReverseHG` method.

        :param dataset: instance of `Dataset` class
        :param batch_size:
        :param epochs: number of epochs (can be None, in which case examples are
                        fed continuously)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.T = int(np.ceil(dataset.num_examples / batch_size))
        if self.epochs:
            self.T *= self.epochs

        self.rnd = get_rand_state(rnd)

        self.training_schedule = None
        self.iter_per_epoch = int(dataset.num_examples / batch_size)

    def setting(self):
        excluded = ["training_schedule", "datasets"]
        dictionary = {k: v for k, v in vars(self).items() if k not in excluded}
        if hasattr(self.dataset, "setting"):
            dictionary["dataset"] = self.dataset.setting()
        return dictionary

    def generate_visiting_scheme(self):
        """
        Generates and stores example visiting scheme, as a numpy array of integers.

        :return: self
        """

        def all_indices_shuffled():
            _res = list(range(self.dataset.num_examples))
            self.rnd.shuffle(_res)
            return _res

        # noinspection PyUnusedLocal
        _tmp_ts = np.concatenate(
            [all_indices_shuffled() for _ in range(self.epochs or 1)]
        )
        self.training_schedule = (
            _tmp_ts
            if self.training_schedule is None
            else np.concatenate([self.training_schedule, _tmp_ts])
        )  # do not discard previous schedule,
        # this should allow backward passes of arbitrary length

        return self

    def create_supplier(self, x, y, *other_feeds, name=None):
        return self.create_feed_dict_supplier(x, y, *other_feeds, name=name)

    def create_feed_dict_supplier(self, x, y, *other_feeds, name=None):
        """

        :param name: optional name for this supplier
        :param x: placeholder for independent variable
        :param y: placeholder for dependent variable
        :param other_feeds: dictionary of other feeds (e.g. dropout factor, ...) to add to the input output
                            feed_dict
        :return: a function that generates a feed_dict with the right signature for Reverse and Forward HyperGradient
                    classes
        """

        def _training_supplier(step=None):
            nonlocal other_feeds

            if step >= self.T:
                if step % self.T == 0:
                    if self.epochs:
                        print(
                            "WARNING: End of the training scheme reached."
                            "Generating another scheme.",
                            file=sys.stderr,
                        )
                    self.generate_visiting_scheme()
                step %= self.T

            if self.training_schedule is None:
                # print('visiting scheme not yet generated!')
                self.generate_visiting_scheme()

            # noinspection PyTypeChecker
            nb = self.training_schedule[
                step
                * self.batch_size : min(
                    (step + 1) * self.batch_size, len(self.training_schedule)
                )
            ]

            bx = self.dataset.data[nb, :]
            by = self.dataset.target[nb, :]

            # if lambda_feeds:  # this was previous implementation... dunno for what it was used for
            #     lambda_processed_feeds = {k: v(nb) for k, v in lambda_feeds.items()}  previous implementation...
            #  looks like lambda was
            # else:
            #     lambda_processed_feeds = {}
            return merge_dicts(
                {x: bx, y: by}, *[maybe_call(of, step) for of in other_feeds]
            )

        if name:
            NAMED_SUPPLIER[name] = _training_supplier

        return _training_supplier
