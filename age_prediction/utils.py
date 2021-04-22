"""
Utility functions
"""
# Standard library imports
import datetime
import numpy
import os

# Third party imports
from torchvision import transforms
from PIL import Image
from skimage.exposure import rescale_intensity


def _get_current_time(strft=False):
    if strft:
        return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")
    else:
        return datetime.datetime.now()


def _path_to_string(path):
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def _convert_img_plot(img):
    img = rescale_intensity(img.cpu().detach().numpy(), out_range=(0, 255))
    im = Image.fromarray(numpy.float32(img)).convert('LA')
    pil_to_tensor = transforms.ToTensor()(im)
    return pil_to_tensor


def _is_tuple_or_list(x):
    return isinstance(x, (tuple, list))


def _process_array_argument(x):
    if not _is_tuple_or_list(x):
        x = [x]
    return x
