import warnings
import os
from io import BytesIO
import json
from random import Random
from tfrecord.reader import tfrecord_iterator
from tfrecord import example_pb2
import torch
from itertools import chain
from functools import partial
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import pytorch_lightning as pl
from viewformer.data._common import expand_path, get_dataset_url, get_dataset_info
from viewformer.data._common import resize_th


def convert_image_dtype(image, dtype):
    if image.dtype == dtype:
        return image
    assert dtype in [torch.float32, torch.uint8]
    assert image.dtype in [torch.float32, torch.uint8]
    if dtype == torch.float32:
        return image.to(torch.float32) * 255.
    elif dtype == torch.uint8:
        return image.div(255).to(torch.uint8)


def get_dataset_info(path):
    with open(os.path.join(path, 'info.json'), 'r') as f:
        dataset_info = json.load(f)
    return dataset_info


def get_dataset_url(path, split, dataset_info):
    dataset_name = dataset_info['name']
    size = dataset_info[f'{split}_size']

    if f'{split}_url' in dataset_info:
        return dataset_info[f'{split}_url']

    if path.startswith('~'):
        path = os.path.expanduser(path)
    return f'{path}/{dataset_name}-{split}-{{000001..{size:06d}}}-of-{size:06d}'


def worker_shard_selection(shards):
    assert isinstance(shards, list)
    assert isinstance(shards[0], str)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(shards) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(shards)}")
        return shards[wid::num_workers]
    else:
        return shards


def shard_selection(shards, distributed_info=None):
    if distributed_info is not None:
        gr, ws = distributed_info
        if len(shards) < ws:
            warnings.warn('There are not enough shards.')
            warnings.warn('Some data will be duplicated!')
            ws = len(shards)
            gr = gr % len(shards)
        shards = list(shards[gr::ws])
    shards = worker_shard_selection(shards)
    return shards


def apply(dataset, fn):
    def iter_fn():
        return fn(dataset())
    return iter_fn


def map_fn(dataset, fn):
    return apply(dataset, partial(map, fn))


def flat_map(dataset, fn):
    return apply(dataset, lambda dataset: chain(*(fn(x) for x in dataset)))


def make_infinite(dataset):
    def iter_fn():
        while True:
            for x in dataset():
                yield x
    return iter_fn


def repeat(dataset, repeat=None):
    if repeat is None:
        return dataset

    def iter_fn():
        c = repeat
        while c == -1 or c > 0:
            for x in dataset():
                yield x
            if c > 0:
                c -= 1
    return iter_fn


def fixed_seed_shuffle(seed=42):
    rng = Random(seed)

    def shuffle(items):
        items = list(items)
        rng.shuffle(items)
        return items

    return shuffle


def local_shuffle_iterator(dataset, queue_size):
    seed = 42
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % np.iinfo(np.uint32).max
    rng = Random(seed)

    def iter_fn():
        iterator = dataset()
        buffer = []
        try:
            for _ in range(queue_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass

        while buffer:
            index = rng.randrange(len(buffer))
            try:
                item = buffer[index]
                buffer[index] = next(iterator)
                yield item
            except StopIteration:
                yield buffer.pop(index)
    return iter_fn


def batch(dataset, batch_size: int, drop_last: bool = False):
    def iter_fn():
        buffer = []

        def _stack(xs):
            if isinstance(xs[0], dict):
                return {k: _stack([x[k] for x in xs]) for k in xs[0].keys()}
            if isinstance(xs[0], (str, bytes)):
                return list(xs)
            return torch.stack(xs, 0)

        for x in dataset():
            buffer.append(x)
            if len(buffer) >= batch_size:
                yield _stack(buffer)
                buffer = []
        if len(buffer) > 0:
            yield _stack(buffer)
            buffer = []
    return iter_fn


def limit(dataset, limit: int = None):
    if limit is None:
        return dataset
    return apply(dataset, lambda dataset: (x for x, _ in zip(dataset, range(limit))))


class ImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, shard_paths,
                 batch_size: int, image_size: int, repeat: int = None,
                 distributed_info=None, shuffle: bool = False,
                 transform=None,
                 limit: int = None):
        super().__init__()
        self.shard_paths = shard_paths
        self.shard_selection = partial(shard_selection, distributed_info=distributed_info)
        self.batch_size = batch_size
        self.image_size = image_size
        self.repeat = repeat
        self.transform = transform
        self.shuffle = shuffle
        self.local_batch_size = self.batch_size
        if distributed_info is not None:
            assert self.batch_size % distributed_info[1] == 0, "Batch size must be divisible by the number of nodes in DDP"
            self.local_batch_size = batch_size // distributed_info[1]
        self.limit = limit
        self._dataset = self._build_dataset()

    def _decode_example(self, record):
        example = example_pb2.Example()
        example.ParseFromString(record)
        frames = example.features.feature['frames'].ListFields()[0]
        frames = frames[1].value
        return frames

    def _parse_example(self, frames):
        images = []
        for array in frames:
            pic = Image.open(BytesIO(array)).convert('RGB')
            img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
            images.append(img.permute((2, 0, 1)))
        images = torch.stack(images, 0).contiguous()
        images = images.to(dtype=torch.float32).div(255)
        if self.image_size is not None:
            images = convert_image_dtype(resize_th(images, self.image_size), torch.float32)
        return images

    def _build_dataset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)

        dataset = lambda: self.shard_paths
        if self.shuffle:
            dataset = apply(dataset, fixed_seed_shuffle())
        dataset = apply(dataset, self.shard_selection)
        dataset = flat_map(dataset, lambda x: tfrecord_iterator(data_path=f'{x}.tfrecord'))

        # Decode and unbatch
        dataset = flat_map(dataset, self._decode_example)
        dataset = repeat(dataset, self.repeat)

        # Shuffle and batch
        # NOTE: we shuffle all splits
        dataset = local_shuffle_iterator(dataset, 1000)
        dataset = batch(dataset, self.local_batch_size)
        dataset = map_fn(dataset, self._parse_example)
        if self.transform is not None:
            dataset = map_fn(dataset, self.transform)
        dataset = limit(dataset, self.limit)
        return dataset

    def __iter__(self):
        return iter(self._dataset())


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, batch_size: int, num_workers: int = 8, transform=None,
                 num_val_workers: int = None,
                 limit_val_batches=None, limit_train_batches=None, image_size: int = None):
        super().__init__()
        self.num_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers
        self.batch_size = batch_size
        self.dataset = dataset
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.image_size = image_size
        self.transform = transform

    def _get_shards(self, is_train):
        shards = []
        for dataset in self.dataset.split(','):
            dataset_info = get_dataset_info(dataset)
            split = 'train'
            if not is_train:
                split = 'val' if 'val' in dataset_info.get('splits', []) else 'test'

            shards.extend(expand_path(get_dataset_url(dataset, split, dataset_info)))
        return shards

    def setup(self, stage=None):
        distributed_info = self.trainer.global_rank, self.trainer.world_size
        self.train_dataset = ImageDataset(self._get_shards(True),
                                          batch_size=self.batch_size,
                                          image_size=self.image_size,
                                          repeat=-1,
                                          distributed_info=distributed_info,
                                          shuffle=True,
                                          transform=self.transform,
                                          limit=self.limit_train_batches)

        self.test_dataset = ImageDataset(self._get_shards(False),
                                         batch_size=self.batch_size,
                                         image_size=self.image_size,
                                         repeat=-1,
                                         distributed_info=distributed_info,
                                         shuffle=True,
                                         transform=self.transform,
                                         limit=self.limit_val_batches)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_val_workers, pin_memory=True, batch_size=None)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_val_workers, pin_memory=True, batch_size=None)
