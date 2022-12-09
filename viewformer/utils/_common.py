import os
import shutil
import inspect
import logging
import tarfile
import sys
from dataclasses import dataclass
from functools import partial
from collections import Counter


class SplitIndices:
    def __init__(self, indices):
        if isinstance(indices, range):
            self._indices = f'{indices.start}:{indices.stop}:{indices.step}'
        elif isinstance(indices, list):
            self._indices = ','.join(str(x) for x in indices)
        elif isinstance(indices, SplitIndices):
            self._indices = indices._indices
        else:
            self._indices = indices

    @classmethod
    def from_str(cls, str_val):
        return SplitIndices(str_val)

    def __repr__(self):
        return self._indices

    def __str__(self):
        return self._indices

    def restrict(self, b):
        vals = []
        if not isinstance(b, SplitIndices):
            b = SplitIndices(b)
        limit = b.left_limit()
        for x in self._indices.split(','):
            xx = [int(a) if a else None for a in x.split(':')]
            if len(xx) == 1:
                if xx[0] in b:
                    vals.append(xx[0])
            elif len(xx) == 2:
                xx.append(None)
            if len(xx) == 3:
                cur = xx[0]
                if cur is None:
                    cur = 0
                while (xx[1] is None or cur < xx[1]) and cur < limit:
                    if cur in b:
                        vals.append(cur)
                    cur += 1 if xx[2] is None else xx[2]
        return SplitIndices(','.join(map(str, vals)))

    def __contains__(self, val):
        for x in self._indices.split(','):
            xx = [int(a) if a else None for a in x.split(':')]
            if len(xx) == 1:
                if val == xx[0]:
                    return True
                else:
                    continue
            if len(xx) == 2:
                step = 1
            else:
                step = xx[-1]
            start, stop = xx[:2]
            if start is None:
                start = 0
            if (val - start) % step == 0 and (stop is None or val < stop) and (start is None or val >= start):
                return True
        return False

    def left_limit(self):
        max_v = -float('inf')
        for x in self._indices.split(','):
            xx = [int(a) if a else None for a in x.split(':')]
            if len(xx) == 1:
                max_v = max(max_v, xx[0] + 1)
            if xx[1] is None:
                return float('inf')
            return xx[1]
        return max_v

    def __iter__(self):
        if self._indices == '':
            return
        for x in self._indices.split(','):
            xx = [int(a) if a else None for a in x.split(':')]
            if len(xx) == 1:
                yield xx[0]
            elif len(xx) == 2:
                xx.append(None)
            if len(xx) == 3:
                cur = xx[0]
                if cur is None:
                    cur = 0
                while xx[1] is None or cur < xx[1]:
                    yield cur
                    cur += 1 if xx[2] is None else xx[2]


def is_torch_model(checkpoint):
    return checkpoint.endswith('.pth') or checkpoint.endswith('.ckpt')


def batch_slice(x, ind):
    if isinstance(x, tuple):
        return tuple(map(partial(batch_slice, ind=ind), x))
    elif isinstance(x, dict):
        return x.__class__([(k, batch_slice(v, ind)) for k, v in x.items()])
    return x[ind]


def batch_len(x):
    if isinstance(x, tuple):
        return batch_len(x[0])
    elif isinstance(x, dict):
        return batch_len(next(iter(x.values())))
    # return x.shape[0]
    return len(x)


def dict_replace(d, key, value):
    d = dict(**d)
    d[key] = value
    return d


def single(iterator):
    value = None
    for x in iterator:
        if value is not None:
            raise RuntimeError('Iterable contains more than one item')
        value = (x,)
    if value is None:
        raise StopIteration('Iterable contains no items')
    return value[0]


def unique(iterable):
    outputted = set()
    for x in iterable:
        if x not in outputted:
            outputted.add(x)
            yield x


def pull_checkpoint(checkpoint, override=False):
    import requests
    from tqdm import tqdm

    path = f'https://data.ciirc.cvut.cz/public/projects/2022ViewFormer/checkpoints/{checkpoint}.tar.gz'

    basename = os.path.split(path)[1][:-len('.tar.gz')]
    local_path = os.path.expanduser(f'~/.cache/viewformer/{basename}')
    if os.path.exists(local_path):
        if override:
            shutil.rmtree(local_path)
        else:
            return local_path
    os.makedirs(local_path, exist_ok=True)

    response = requests.get(path, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    if response.status_code != 200:
        raise Exception(f'Model {checkpoint} not found')

    stream = response.raw
    _old_read = stream.read

    def _read(size):
        progress_bar.update(size)
        return _old_read(size)
    setattr(stream, 'read', _read)

    with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar, \
            tarfile.open(fileobj=stream, mode='r') as tfile:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tfile, local_path)
    return local_path
