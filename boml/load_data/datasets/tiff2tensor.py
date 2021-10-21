# from progressbar import ProgressBar, Percentage, Bar, ETA
import os
import sys

import h5py
import numpy as np
from PIL import Image

from boml.load_data.datasets.load_full_dataset import MINI_IMAGENET_FOLDER_RES84


def img_to_array_old(image_file, resize=84, flip=False, convert_to_gray=False):
    img_z = Image.open(image_file)
    # img_z.convert('RGB')  # not useful at all...
    if resize:
        img_z = img_z.resize((resize, resize), Image.LANCZOS)
    if (
        flip
    ):  # when reading a stack saved in vaa3d format Y coordinates are reversed (used by save_subst
        img_z = img_z.transpose(Image.FLIP_TOP_BOTTOM)
    if convert_to_gray:
        img_z = img_z.convert("L")
    ary = np.array(img_z)
    ary = (
        np.stack((ary,) * 3, axis=2) if ary.ndim == 2 else ary
    )  # some images are black and white
    return (
        ary if ary.shape[-1] == 3 else ary[:, :, :3]
    )  # some images have an alpha channel... (unbelievable...)


def img_to_array(imfile, resize=84):
    from scipy.misc import imresize, imread

    img = imread(imfile, mode="RGB")
    if resize:
        # noinspection PyTypeChecker
        img = imresize(img, size=(resize, resize, 3))
    return img


def images_to_tensor(files, tensor):
    # widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
    # pbar = ProgressBar(widgets=widgets, maxval=len(files))
    # pbar.start()
    for i, f in enumerate(files):
        print(f)
        tensor[i, :] = img_to_array(f)
        # pbar.update(i + 1)
        # pbar.finish()


def tensor_to_images(prefix, tensor):
    n = tensor.shape[0]
    # widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
    # pbar = ProgressBar(widgets=widgets, maxval=n)
    # pbar.start()
    for z in range(n):
        img = Image.fromarray(tensor[z].astype(np.int8), mode="L")
        img.save(prefix + "%03d" % z + ".tif")
        # pbar.update(z + 1)
        # pbar.finish()


def convert_mini_imagenet(_folder=None, resize=None):
    for st in ["train", "val", "test"]:
        folder = os.path.join(_folder or MINI_IMAGENET_FOLDER_RES84, st)
        classes = os.listdir(folder)
        files = []
        for c in classes:
            files += [
                os.path.join(folder, os.path.join(c, f))
                for f in os.listdir(os.path.join(folder, c))
            ]
        # print(files)
        n = len(files)
        img_shape = img_to_array(files[0], resize=resize).shape

        img = img_to_array(files[0], resize=resize)
        print(img)
        print(img.dtype)

        h5file = h5py.File(
            os.path.join(_folder or MINI_IMAGENET_FOLDER_RES84, "%s.h5" % st), "w"
        )
        X = h5file.create_dataset(
            "X", (n, img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8
        )
        images_to_tensor(files, X)


if __name__ == "__main__":
    convert_mini_imagenet(sys.argv[1] if len(sys.argv) > 1 else None, resize=None)
    # base = '../vasculature_data'
    # orig_files = ['{}/orig/097000_164000_08600000{:0>2d}.tif'.format(base, z) for z in range(100)]
    # target_files = ['{}/GT/097000_164000_086000-labels-stack-intero-smooth00{:0>2d}.tif'.format(base, z) for z in range(100)]
    #
    # assert(len(orig_files) == len(target_files))
    # n = len(orig_files)
    # img_shape = img_to_array(orig_files[0]).shape
    #
    # h5file = h5py.File('vasculature.h5', 'w')
    # X = h5file.create_dataset("X", (n,img_shape[0],img_shape[1]), h5py.h5t.NATIVE_FLOAT)
    # y = h5file.create_dataset("y", (n,img_shape[0],img_shape[1]), h5py.h5t.NATIVE_FLOAT)
    # images_to_tensor(orig_files, X, 65535.)
    # images_to_tensor(target_files, y, 255.)
    # h5file.close()
    #
    # base = '../vasculature_data'
    # orig_files = ['{}/orig/097000_164000_08600000{:0>2d}.tif'.format(base, z) for z in range(100)]
    # target_files = ['{}/GT/097000_164000_086000-labels-stack-intero-smooth00{:0>2d}.tif'.format(base, z) for z in
    #                 range(100)]
