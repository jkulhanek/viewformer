import re
import random
import numpy as np
from functools import lru_cache
import sys
import json
import shutil
import fnmatch
import tempfile
import os
import copy
from aparse import Literal
from typing import Union, List
from tqdm import tqdm
from viewformer.utils import SplitIndices
from viewformer.utils import unique, batch_slice, batch_len


def resize_th(th_images, image_size, method=None):
    '''
    Note, this resize operation was used when generating our original datasets
    in order to reproduce the results, it has to be the same
    '''
    if method is not None:
        assert method in ['nearest', 'bilinear']
    if th_images.shape[-2] == image_size:
        return th_images

    import torch
    if th_images.dtype == torch.uint8:
        th_images = th_images.to(torch.float32) / 255.
    assert th_images.dtype == torch.float32
    if method is None:
        if image_size > th_images.shape[-2]:
            method = 'nearest'
        else:
            method = 'bilinear'
    if method == 'nearest':
        th_images = torch.nn.functional.interpolate(th_images, (image_size, image_size), mode='nearest')
    else:
        th_images = torch.nn.functional.interpolate(th_images, (image_size, image_size), mode='bilinear', align_corners=False)
    th_images = th_images.clamp_(0, 1)
    th_images = (th_images * 255.).to(torch.uint8)
    return th_images


def resize(images, image_size, method=None):
    '''
    Note, this resize operation was used when generating our original datasets
    in order to reproduce the results, it has to be the same
    '''
    if method is not None:
        assert method in ['nearest', 'bilinear']
    if images.shape[-2] == image_size:
        return images

    import torch
    th_images = torch.from_numpy(images).permute(0, 3, 1, 2)
    th_images = resize_th(th_images, image_size, method)
    return th_images.permute(0, 2, 3, 1).numpy()


Framework = Literal['tf', 'th']
DatasetFormat = Literal['tf', 'webd']


class ChangedImageSizeLoader:
    def __init__(self, inner, image_size):
        self.inner = inner
        self.image_size = image_size

    @property
    def sequence_size(self):
        return getattr(self.inner, 'sequence_size', None)

    def num_images_per_sequence(self):
        return self.inner.num_images_per_sequence()

    def __getitem__(self, idx):
        item = self.inner[idx]
        if self.image_size is not None and item['frames'].shape[-2] != self.image_size:
            if 'frames' in item:
                item['frames'] = resize(np.array(item['frames']), self.image_size)
        return item

    def __len__(self):
        return len(self.inner)


class FixedSequenceSizeLoader:
    def __init__(self, inner, sequence_size):
        self.inner = inner
        self.sequence_size = sequence_size

    def __len__(self):
        return len(self.num_images_per_sequence())

    @lru_cache()
    def num_images_per_sequence(self):
        return sum(([self.sequence_size] * (x // self.sequence_size) for x in self.inner.num_images_per_sequence()), [])

    @lru_cache()
    def _cum_seq_offset_map(self):
        return [(inner_i, i * self.sequence_size) for inner_i, x in enumerate(self.inner.num_images_per_sequence()) for i in range(x // self.sequence_size)]

    @lru_cache(maxsize=1)
    def _get_inner(self, idx):
        return self.inner[idx]

    def __getitem__(self, idx):
        if idx > 0:
            inner_idx, local_offset = self._cum_seq_offset_map()[idx]
        else:
            # Fast load first batch for visualize utils
            inner_idx, local_offset = 0, 0
        return batch_slice(self._get_inner(inner_idx), slice(local_offset, local_offset + self.sequence_size))


class LazyArray:
    def __init__(self, array, map_fn):
        self.array = array
        self.map_fn = map_fn

    def __getitem__(self, idx):
        if isinstance(idx, list):
            indices = [self.array[x] for x in idx]
            return np.stack(list(map(self.map_fn, indices)), 0)
        else:
            indices = self.array[idx]
            if isinstance(idx, slice):
                return np.stack(list(map(self.map_fn, indices)), 0)
            else:
                return self.map_fn(indices)

    def __len__(self):
        return len(self.array)

    @property
    @lru_cache()
    def shape(self):
        return (len(self.array),) + tuple(self.map_fn(self.array[0]).shape)

    def __array__(self):
        return np.stack(list(map(self.map_fn, self.array)))


class ShuffledLoader:
    def __init__(self, inner,
                 seed: int = 42,
                 shuffle_sequence_items: bool = False,
                 shuffle_sequences: bool = False):
        self.inner = inner
        self.seed = seed
        self.shuffle_sequences = shuffle_sequences
        self.shuffle_sequence_items = shuffle_sequence_items
        if hasattr(inner, 'sequence_size'):
            self.sequence_size = inner.sequence_size

    @lru_cache()
    def _sequence_indices(self):
        indices = list(range(len(self)))
        if self.shuffle_sequences:
            random.Random(self.seed).shuffle(indices)
        return indices

    def __len__(self):
        return len(self.inner)

    def num_images_per_sequence(self):
        inner_images_per_sequence = self.inner.num_images_per_sequence()
        if self.shuffle_sequences:
            return [inner_images_per_sequence[x] for x in self._sequence_indices()]
        return inner_images_per_sequence

    def _items_take_indices(self, items, indices):
        if isinstance(items, LazyArray):
            return LazyArray([items.array[x] for x in indices], items.map_fn)
        elif isinstance(items, str):
            return items
        return items[indices]

    def __getitem__(self, idx):
        if self.shuffle_sequences:
            idx = self._sequence_indices()[idx]
        batch = self.inner[idx]
        if self.shuffle_sequence_items:
            indices = list(range(batch_len(batch)))
            random.Random(self.seed * len(self) + idx).shuffle(indices)
            batch = {
                k: self._items_take_indices(v, indices)
                for k, v in batch.items()
            }
        return batch


def _get_shard_map(num_images_per_sequence, max_images_per_shard, max_sequences_per_shard):
    shards = []
    current_shard_imgs, current_shard_seqs = 0, 0
    shard_offset = 0
    for num_img in num_images_per_sequence:
        current_shard_imgs += num_img
        current_shard_seqs += 1
        if (max_images_per_shard is not None and current_shard_imgs >= max_images_per_shard) \
                or (max_sequences_per_shard is not None and current_shard_seqs >= max_sequences_per_shard):
            shards.append((current_shard_seqs, current_shard_imgs, shard_offset))
            shard_offset += current_shard_seqs
            current_shard_imgs, current_shard_seqs = 0, 0
    if current_shard_seqs > 0:
        shards.append((current_shard_seqs, current_shard_imgs, shard_offset))
    return shards


def all_same(iterable):
    value = None
    for i, x in enumerate(iterable):
        if i > 0 and x != value:
            return False
        value = x
    return True


class _ProxyList(list):
    def __init__(self, indices, inner):
        super().__init__(indices)
        self.inner = inner

    def __getitem__(self, idx):
        return self.inner[self[idx]]

    def __setitem__(self, idx, val):
        self.inner[self[idx]] = val

    def __delitem__(self, *args, **kwargs):
        raise NotImplementedError()

    def __iter__(self):
        for x in super().__iter__():
            yield self.inner[x]


def write_dataset_info(path, dataset_info, allow_incompatible_config=False):
    info = dict()
    if os.path.exists(path):
        with open(path, mode='r') as f:
            info = json.load(f)
    orig_info = dict(info)
    info.update(dataset_info)
    if not allow_incompatible_config:
        for key, val in orig_info.items():
            if info[key] != val and key != 'splits':
                raise RuntimeError(f'Cannot override dataset because dataset config is different:\n{json.dumps(orig_info, sort_keys=True)}\n!=\n{json.dumps(info, sort_keys=True)}')
    info['splits'] = sorted(set(dataset_info['splits'] + orig_info.get('splits', [])))
    with open(path, mode='w+') as f:
        json.dump(info, f, sort_keys=True)


def build_index(path, num_images_per_sequence, shard_seqs):
    with open(path, 'w+') as f:
        for shard_id, (seqs, images, offset) in enumerate(shard_seqs):
            for seq_id in range(offset, seqs + offset):
                f.write(f'{shard_id + 1:06d} {num_images_per_sequence[seq_id]}\n')


def generate_dataset_from_loader(
        loader,
        split: str,
        output_path: str,
        max_images_per_shard: int = None,
        max_sequences_per_shard: int = None,
        drop_last: bool = False,
        shards: Union[int, List[int]] = None,
        features: List[str] = None,
        seed: int = 42,
        allow_incompatible_config: bool = False,
        format: DatasetFormat = 'tf'):
    from .tfrecord_dataset import write_shard
    assert max_images_per_shard is not None or max_sequences_per_shard is not None
    num_images_per_sequence = loader.num_images_per_sequence()
    shard_seqs = _get_shard_map(num_images_per_sequence, max_images_per_shard, max_sequences_per_shard)
    dataset_info = dict()
    if drop_last:
        assert max_images_per_shard is None
        assert max_sequences_per_shard is not None
        if shard_seqs[-1][0] < max_sequences_per_shard:
            num_images_per_sequence = num_images_per_sequence[:-shard_seqs[-1][0]]
            shard_seqs = shard_seqs[:-1]
    first_batch = next(iter(loader))
    if features is None:
        features = list(first_batch.keys())
        if 'cameras' in first_batch and first_batch['cameras'].shape[-1] == 5:
            features.remove('cameras')
            features.append('cameras-gqn')
    num_all_shards = len(shard_seqs)
    dataset_info['frame_size'] = first_batch['frames'].shape[-2]
    dataset_info['features'] = features

    sequence_size = getattr(loader, 'sequence_size', None)
    dataset_info[f'{split}_sequence_size'] = sequence_size
    dataset_info[f'{split}_size'] = num_all_shards
    dataset_info['splits'] = [split]
    dataset_info[f'{split}_max_images_per_shard'] = max_images_per_shard
    dataset_info[f'{split}_max_sequences_per_shard'] = max_sequences_per_shard
    dataset_info[f'{split}_num_images'] = sum(x[1] for x in shard_seqs)
    dataset_info[f'{split}_num_sequences'] = sum(x[0] for x in shard_seqs)
    if all_same(x[0] for x in shard_seqs):
        dataset_info[f'{split}_num_sequences_per_shard'] = next(iter(shard_seqs))[0]
    if all_same(x[1] for x in shard_seqs):
        dataset_info[f'{split}_num_images_per_shard'] = next(iter(shard_seqs))[1]
    dataset_info['format'] = format
    dataset_path, dataset_info['name'] = os.path.split(output_path)
    os.makedirs(dataset_path, exist_ok=True)
    if shards is None:
        shard_indices_plus_1 = SplitIndices(range(1, num_all_shards + 1))
    else:
        shard_indices_plus_1 = list(SplitIndices(shards).restrict(SplitIndices(range(1, num_all_shards + 1))))
    if 1 in shard_indices_plus_1:
        # First writer writes the dataset info as well
        write_dataset_info(os.path.join(dataset_path, 'info.json'), dataset_info, allow_incompatible_config=allow_incompatible_config)
        build_index(f'{output_path}-{split}.index', num_images_per_sequence, shard_seqs)
    for shard_id_plus_1 in shard_indices_plus_1:
        num_seqs, num_img, seq_offset = shard_seqs[shard_id_plus_1 - 1]
        shard_sequences = _ProxyList([seq_offset + i for i in range(num_seqs)], loader)
        shard_sequences = tqdm(shard_sequences, desc=f'generating shard [{shard_id_plus_1}/{num_all_shards}]')
        shard_path = f'{output_path}-{split}-{shard_id_plus_1:06d}-of-{num_all_shards:06d}'
        write_shard(shard_path, shard_sequences, features)


def read_shards(shard_paths, info, _decode_image=True, output_tf_dataset=False, **kwargs):
    '''
    NOTE: this function is not efficient and should not be used for training
    for training use an appropriate load_dataset function
    '''
    from .tfrecord_dataset import read_shards
    dataset = read_shards(shard_paths, info, **kwargs, _decode_image=_decode_image)
    if output_tf_dataset:
        return dataset
    else:
        return dataset.as_numpy_iterator()


def get_dataset_url(path, split, dataset_info):
    dataset_name = dataset_info['name']
    size = dataset_info[f'{split}_size']

    if f'{split}_url' in dataset_info:
        return dataset_info[f'{split}_url']

    if path.startswith('~'):
        path = os.path.expanduser(path)
    return f'{path}/{dataset_name}-{split}-{{000001..{size:06d}}}-of-{size:06d}'


def get_dataset_info(path):
    with open(os.path.join(path, 'info.json'), 'r') as f:
        dataset_info = json.load(f)
    return dataset_info


def read_dataset(dataset_path, split: str, output_tf_dataset=False, shards=None, **kwargs):
    '''
    NOTE: this function is not efficient and should not be used for training
    for training use an appropriate load_dataset function
    '''
    info = get_dataset_info(dataset_path)
    name = info['name']
    size = info[f'{split}_size']
    if shards is None:
        shards = list(range(1, size + 1))
    else:
        shards = [i for i in shards if i >= 1 and i <= size]
    if info['format'] == 'tf':
        shards = [f'{dataset_path}/{name}-{split}-{i:06d}-of-{size:06d}.tfrecord' for i in shards]
    elif info['format'] == 'webd':
        shards = [f'{dataset_path}/{name}-{split}-{i:06d}-of-{size:06d}.tar' for i in shards]
    else:
        raise RuntimeError(f'dataset format {info["format"]} not in "tf", "webd"')

    return read_shards(shards, info, output_tf_dataset=output_tf_dataset, **kwargs)


def transform_dataset(dataset_path, output_path: str, transformer, shards=None,
                      framework: Framework = None,
                      splits: List[str] = None):
    from .tfrecord_dataset import get_dataset_info, write_shard, build_shard_index
    if framework is None:
        framework = getattr(transformer, 'framework', 'th')

    old_dataset_info = get_dataset_info(dataset_path)
    dataset_info = dict(**old_dataset_info)
    new_dataset_info = copy.copy(dataset_info)
    new_dataset_info['features'] = transformer.output_features(dataset_info.get('features', None))
    new_dataset_info['format'] = 'tf'
    if hasattr(transformer, 'update_dataset_info'):
        new_dataset_info = transformer.update_dataset_info(new_dataset_info)

    splits = splits if splits is not None else dataset_info.get('splits', ['test', 'train'])
    os.makedirs(output_path, exist_ok=True)
    if shards is None or 1 in shards:
        # First shard writes info
        write_dataset_info(os.path.join(output_path, 'info.json'), new_dataset_info, allow_incompatible_config=True)

    for split in splits:
        size = dataset_info[f'{split}_size']
        if shards is not None:
            shard_list = list(SplitIndices(range(1, size + 1)).restrict(SplitIndices(shards)))
        else:
            shard_list = list(range(1, size + 1))

        if 1 in shard_list:
            # First shard writes index
            if os.path.exists(f'{dataset_path}/{dataset_info["name"]}-{split}.index'):
                shutil.copy(f'{dataset_path}/{dataset_info["name"]}-{split}.index', f'{output_path}/{dataset_info["name"]}-{split}.index')

        for shard_id in tqdm(shard_list, desc=f'generating {split}'):
            dataset = read_dataset(dataset_path, split, output_tf_dataset=framework == 'tf',
                                   shards=[shard_id],
                                   image_size=getattr(transformer, 'image_size', None))
            transformed_iterator = transformer(split, dataset)
            write_shard(f'{output_path}/{dataset_info["name"]}-{split}-{shard_id:06d}-of-{size:06d}', transformed_iterator,
                        features=new_dataset_info['features'])
            build_shard_index(
                f'{output_path}/{dataset_info["name"]}-{split}-{shard_id:06d}-of-{size:06d}.tfrecord',
                f'{output_path}/{dataset_info["name"]}-{split}-{shard_id:06d}-of-{size:06d}.index')


class LazyFeatures(dict):
    def __getitem__(self, k):
        value = super().__getitem__(k)
        if callable(value):
            value = value()
            self[k] = value
        return value

    def items(self):
        for k, _ in super().items():
            yield k, self[k]


class ArchiveStoreContext:
    _current_context = None

    def __init__(self):
        self._dir = None
        self.path = None

    def __enter__(self):
        self._old_context = ArchiveStoreContext._current_context
        ArchiveStoreContext._current_context = self
        self._dir = tempfile.TemporaryDirectory()
        self.path = self._dir.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        ArchiveStoreContext._current_context = self._old_context
        self._dir.__exit__(*args, **kwargs)
        self.path = None
        self._old_context = None

    @staticmethod
    def current_context():
        return ArchiveStoreContext._current_context

    @staticmethod
    def get_any():
        if ArchiveStoreContext._current_context is None:
            return ArchiveStoreContext()
        else:
            class NoopContext:
                def __enter__(self):
                    return self

                def __exit__(self, *args, **kwargs):
                    pass

            return NoopContext()


class ArchiveStore:
    def __init__(self, file):
        if isinstance(file, str):
            self.filename = file
            self.file = None
        elif hasattr(file, 'read'):
            self.filename = getattr(file, 'name')
            if self.filename.endswith('.zip'):
                self.filename = self.filename[:-len('.zip')]
            if self.filename.endswith('.tar.gz'):
                self.filename = self.filename[:-len('.tar.gz')]
            self.file = file
        self._ctx = None
        self._path = None
        self._filelist = None
        self._archive_use_prefix = False
        self._archive = None

    def exists(self, filename):
        pass

    @staticmethod
    def with_context():
        return ArchiveStoreContext()

    def __enter__(self):
        if os.path.exists(f'{self.filename}.zip'):
            import zipfile
            if not os.access(os.path.dirname(self.filename), os.W_OK):
                raise NotImplementedError()
            archive_name = os.path.split(self.filename)[1]
            try:
                self._archive = zipfile.ZipFile(f'{self.filename}.zip', 'r').__enter__()
            except Exception as e:
                print(f'Invalid archive file "{self.filename}.zip"', file=sys.stderr)
                raise e
            self._archive_prefix = ''
            filelist = [x.filename for x in self._archive.filelist]
            if all('/./' in x for x in filelist):
                strp = next(iter(filelist))
                strp = strp[:strp.find('/./') + 3]
                self._archive_prefix += strp
                filelist = [x[len(strp):] for x in filelist]
            self._archive_prefix += (archive_name + '/') if all(x.startswith(archive_name + '/') for x in filelist) else ''
            self._filelist = [x[len(self._archive_prefix):] for x in filelist]
            self._path = self.filename
        elif os.path.exists(f'{self.filename}.tar.gz'):
            raise RuntimeError(f'Tar is not supported, please convert all tar files to zip: {self.filename}')
        else:
            raise RuntimeError(f'File not found {self.filename}')
        return self

    def open(self, file, mode='r'):
        if not os.path.exists(os.path.join(self._path, file)):
            os.makedirs(os.path.dirname(os.path.join(self._path, file)), exist_ok=True)
            member = self._archive.getinfo(self._archive_prefix + file)
            with open(os.path.join(self._path, file), 'wb+') as f, \
                    self._archive.open(member) as sf:
                shutil.copyfileobj(sf, f)
        return open(os.path.join(self._path, file), mode)

    def glob(self, pattern):
        return fnmatch.filter(self._filelist, pattern)

    def ls(self, path):
        return [x.rstrip('/') for x in self._filelist if x.startswith(path) and '/' not in x[len(path):-1] and x != '']

    def __exit__(self, *args, **kwargs):
        self._filelist = None
        self._path = None
        self.close()

    def close(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    @staticmethod
    def list_archives(path):
        files = os.listdir(path)
        files = unique(x[:-len('.zip')] if x.endswith('.zip') else (x[:-len('.tar.gz')] if x.endswith('.tar.gz') else x) for x in files)
        files = sorted(files)
        return files


def expand_path(path, return_shard_ids=False):
    paths = []
    range_val = ''

    def match(x):
        nonlocal range_val
        range_val = x.group(1)
        return '{}'

    path = re.sub(r'{(.+)}', match, path)
    if range_val:
        if ':' in range_val:
            indices = SplitIndices(range_val)
            form = '{:0' + str(len(range_val.split(':')[0])) + '}'
            for i in indices:
                val = form.format(i)
                if not return_shard_ids:
                    paths.append(path.format(val))
                else:
                    paths.append((i, path.format(val)))
        else:
            start, end = range_val.split('..')
            form = '{:0' + str(len(start)) + '}'
            start, end = tuple(map(int, (start, end)))
            for i in range(start, end + 1):
                val = form.format(i)
                if not return_shard_ids:
                    paths.append(path.format(val))
                else:
                    paths.append((i, path.format(val)))
    else:
        if return_shard_ids:
            raise NotImplementedError()
        paths.append(path)
    return paths
