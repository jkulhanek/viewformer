from typing import Optional

from .interiornet import InteriorNetLoader
from .dataset import DatasetLoader
from .sevenscenes import SevenScenesLoader
from .colors import ColorsLoader
from .co3d import CO3DLoader
from .shapenet import ShapenetLoader
from .sm7 import SM7Loader
from viewformer.data._common import ShuffledLoader, FixedSequenceSizeLoader, ChangedImageSizeLoader


_registry = dict()


def register_loader(loader_class):
    name = loader_class.__name__.lower()[:-len('Loader')]

    class _Wrapped(loader_class):
        def __init__(self,
                     shuffle_sequences: Optional[bool] = None,
                     shuffle_sequence_items: Optional[bool] = None,
                     shuffle: Optional[bool] = None,
                     sequence_size: Optional[int] = None,
                     image_size: int = None,
                     seed: int = None,
                     **kwargs):
            raise NotImplementedError()
            
        def __new__(self,
                    shuffle_sequences: Optional[bool] = None,
                    shuffle_sequence_items: Optional[bool] = None,
                    shuffle: Optional[bool] = None,
                    sequence_size: Optional[int] = None,
                    image_size: int = None,
                    seed: int = None,
                    **kwargs):
            if seed is not None:
                kwargs['seed'] = seed
            seed = seed if seed is not None else 42
            custom_resize = getattr(loader_class, '_custom_resize', False)
            custom_shuffle = getattr(loader_class, '_custom_shuffle', False)
            custom_sequence_size = getattr(loader_class, '_custom_sequence_size', False)
            if custom_resize:
                kwargs['image_size'] = image_size
            if custom_sequence_size:
                kwargs['sequence_size'] = sequence_size
            if shuffle is not None:
                assert shuffle_sequence_items is None
                assert shuffle_sequences is None
                shuffle_sequence_items = shuffle_sequences = shuffle
            else:
                assert shuffle is None
                shuffle_sequence_items = shuffle_sequence_items or False
                shuffle_sequences = shuffle_sequences or False

            if custom_shuffle:
                loader = loader_class(shuffle_sequences=shuffle_sequences,
                                      shuffle_sequence_items=shuffle_sequence_items,
                                      sequence_size=sequence_size,
                                      seed=seed, **kwargs)
            else:
                loader = loader_class(**kwargs)
                if shuffle_sequence_items:
                    loader = ShuffledLoader(loader, seed, shuffle_sequence_items=True)
                if sequence_size is not None and not custom_sequence_size:
                    loader = FixedSequenceSizeLoader(loader, sequence_size)
                if shuffle_sequences:
                    loader = ShuffledLoader(loader, seed, shuffle_sequences=True)
            if image_size is not None and not custom_resize:
                loader = ChangedImageSizeLoader(loader, image_size)
            return loader

    _registry[name] = _Wrapped
    return _Wrapped


def build(name, *args, **kwargs):
    return _registry[name](*args, **kwargs)


def get_loader(name):
    return _registry[name]


def get_loader_names():
    return list(_registry.keys())


def get_loaders():
    return _registry


DatasetLoader = register_loader(DatasetLoader)
InteriorNetLoader = register_loader(InteriorNetLoader)
SevenScenesLoader = register_loader(SevenScenesLoader)
ColorsLoader = register_loader(ColorsLoader)
CO3DLoader = register_loader(CO3DLoader)
ShapenetLoader = register_loader(ShapenetLoader)
SM7Loader = register_loader(SM7Loader)
