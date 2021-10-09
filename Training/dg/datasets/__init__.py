from __future__ import absolute_import
import warnings

from .pairs import Pair
from .isc_10k import ISC10K
from .isc_100k import ISC100K
from .isc_100k_256_big import ISC100K_256_big
from .isc_100k_256_big_opa import ISC100K_256_big_opa
from .isc_100k_512_big import ISC100K_512_big
from .isc_100k_512 import ISC100K_512
from .isc_100k_384_big import ISC100K_384_big
from .isc_100k_384 import ISC100K_384

from .isc_100k_256 import ISC100K_256
from .isc_100k_256_olp6 import ISC100K_256_olp6
from .isc_100k_256_olp4 import ISC100K_256_olp4
from .isc_100k_256_olp6_35 import ISC100K_256_olp6_35
from .isc_100k_256_olp4_35 import ISC100K_256_olp4_35

from .isc_100k_256 import ISC100K_256
from .isc_100k_256 import ISC100K_256

from .isc_100k_256_color import ISC100K_256_color
from .isc_100k_256_big_edge import ISC100K_256_big_edge
from .isc_100k_256_big_u import ISC100K_256_big_u
from .isc_100k_256_big_blur import ISC100K_256_big_blur
from .isc_100k_256_big_blur_bw import ISC100K_256_big_blur_bw
from .isc_100k_256_big_ff import ISC100K_256_big_ff
from .isc_100k_256_big_ff_bw import ISC100K_256_big_ff_bw
from .isc_100k_256_big_bw import ISC100K_256_big_bw
from .isc_100k_256_big_color_p4_V2 import ISC100K_256_big_color_p4_V2
from .isc_100k_256_big_color_p4 import ISC100K_256_big_color_p4
from .isc_100k_256_big_color_p4_bw import ISC100K_256_big_color_p4_bw
from .isc_500k_256_big import ISC500K_256_big
from .isc_500k_256 import ISC500K_256
from .isc_500k_256_big_V2 import ISC500K_256_big_V2
from .isc_100k_256_big_dark import ISC100K_256_big_dark


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'msmt17': MSMT17,
    'randperson_subset': RandPerson,
    'isc_10k': ISC10K,
    'isc_100k': ISC100K,
    'isc_100k_256_big': ISC100K_256_big,
    'isc_100k_256_big_opa': ISC100K_256_big_opa,
    'isc_100k_512_big': ISC100K_512_big,
    'isc_100k_512': ISC100K_512,
    'isc_100k_384_big': ISC100K_384_big,
    'isc_100k_384': ISC100K_384,
    'isc_100k_256': ISC100K_256,
    'isc_100k_256_olp6': ISC100K_256_olp6,
    'isc_100k_256_olp4': ISC100K_256_olp4,
    'isc_100k_256_olp6_35': ISC100K_256_olp6_35,
    'isc_100k_256_olp4_35': ISC100K_256_olp4_35,
    'isc_100k_256_color': ISC100K_256_color,
    'isc_100k_256_big_edge': ISC100K_256_big_edge,
    'isc_100k_256_big_u': ISC100K_256_big_u,
    'isc_100k_256_big_blur': ISC100K_256_big_blur,
    'isc_100k_256_big_blur_bw': ISC100K_256_big_blur_bw,
    'isc_100k_256_big_ff': ISC100K_256_big_ff,
    'isc_100k_256_big_ff_bw': ISC100K_256_big_ff_bw,
    'isc_100k_256_big_bw': ISC100K_256_big_bw,
    'isc_100k_256_big_color_p4_V2': ISC100K_256_big_color_p4_V2,
    'isc_100k_256_big_color_p4': ISC100K_256_big_color_p4,
    'isc_100k_256_big_color_p4_bw': ISC100K_256_big_color_p4_bw,
    'isc_500k_256_big': ISC500K_256_big,
    'isc_500k_256': ISC500K_256,
    'isc_500k_256_big_V2': ISC500K_256_big_V2,
    'isc_100k_256_big_dark': ISC100K_256_big_dark,
    'pairs': Pair
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
