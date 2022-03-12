from random import Random
from viewformer.data._common import get_dataset_info, read_dataset
from viewformer.utils import batch_slice, batch_len
from queue import PriorityQueue
from itertools import chain
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache()


def get_sequence_shuffled_dataset(dataset, seed: int = 42):
    class iterator:
        def __iter__(self):
            rng = Random(seed)
            for data in dataset:
                permutation = list(range(batch_len(data)))
                rng.shuffle(permutation)
                yield batch_slice(data, permutation)
    return iterator()


def get_locally_shuffled_dataset(dataset, buffer_size, seed: int = 42):
    class iterator:
        def __iter__(self):
            rng = Random(seed)
            queue = PriorityQueue(buffer_size)
            dataset_iter = iter(dataset)
            for data, _ in zip(dataset_iter, range(buffer_size)):
                queue.put((rng.random(), data))

            for data in dataset_iter:
                _, out = queue.get()
                yield out
                queue.put((rng.random(), data))

            while not queue.empty():
                _, out = queue.get()
                yield out
    return iterator()


def limit_sequence_size(dataset, sequence_size):
    class iterator:
        def __iter__(self):
            for data in dataset:
                size = batch_len(data)
                for i in range(size // sequence_size):
                    yield batch_slice(data, slice(i * sequence_size, (i + 1) * sequence_size))
    return iterator()


class DatasetLoader:
    _custom_shuffle = True

    def __init__(self, path: str, split: str = 'train', shuffle_sequences: bool = False,
                 sequence_size: int = None,
                 shuffle_sequence_items: bool = False, shuffle_buffer_size: int = 10000,
                 seed: int = 42,
                 image_size: int = None, **kwargs):
        self.dataset_info = get_dataset_info(path)
        self.path = path
        self.split = split
        self.num_sequences = self.dataset_info.get(f'{split}_num_sequences')
        if image_size is not None:
            kwargs['image_size'] = image_size
        self.shuffle_sequence_items = shuffle_sequence_items
        self.shuffle_buffer_size = shuffle_buffer_size
        self.sequence_size = sequence_size
        self.dataset = read_dataset(path, split, output_tf_dataset=False, **kwargs)
        if shuffle_sequence_items:
            self.dataset = get_sequence_shuffled_dataset(self.dataset, seed)
        if sequence_size is not None:
            self.dataset = limit_sequence_size(self.dataset, sequence_size)
            self.num_sequences = sum(x // self.sequence_size for x in self.num_images_per_sequence())
        if shuffle_sequences:
            self.dataset = get_locally_shuffled_dataset(self.dataset, shuffle_buffer_size, seed)
        self._iterator_cache = None

    @cache
    def num_images_per_sequence(self):
        if f'{self.split}_sequence_size' in self.dataset_info and self.dataset_info[f'{self.split}_sequence_size'] is not None:
            image_per_sequence = [self.dataset_info.get(f'{self.split}_sequence_size')] * self.dataset_info.get(f'{self.split}_num_sequences')
        else:
            image_per_sequence = [int(x.rstrip('\n').split(' ')[-1]) for x in open(f'{self.path}/{self.dataset_info["name"]}-{self.split}.index', 'r')]
        if self.sequence_size is None:
            return image_per_sequence
        return list(chain(*([self.sequence_size] * (x // self.sequence_size) for x in image_per_sequence)))

    @cache
    def __len__(self):
        return len(self.num_images_per_sequence())

    def _get_batch(self, i):
        if self._iterator_cache is None or self._iterator_cache[0] > i:
            iterator = iter(self.dataset)
            self._iterator_cache = 0, iterator, next(iterator)

        current = self._iterator_cache[-1]
        for j in range(i - self._iterator_cache[0]):
            idx, iterator, current = self._iterator_cache
            current = next(iterator)
            self._iterator_cache = idx + 1, iterator, current
        return current

    def __getitem__(self, i):
        current = self._get_batch(i)
        return current


if __name__ == '__main__':
    import sys
    loader = DatasetLoader(sys.argv[1])
    loader[0]
    breakpoint()
