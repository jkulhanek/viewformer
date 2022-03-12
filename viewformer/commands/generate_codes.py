#!/usr/bin/env python
from aparse import click
from typing import List
from viewformer.utils import SplitIndices
from viewformer.data import transform_dataset

# Use memory growth for tf
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except ImportError:
    pass


class LatentCodeTransformer:
    def _convert_image_type(self, image):
        if image.dtype == 'uint8':
            image = (image.astype('float32') / 255.) * 2. - 1.
        if image.shape[-1] == 3:
            image = image.transpose((0, 3, 1, 2))
        return image

    def update_dataset_info(self, dataset_info):
        dataset_info['token_image_size'] = self.image_size // self.model.config.stride
        self.dataset_info = dataset_info
        return dataset_info

    def __init__(self, model, batch_size: int = None, device=None):
        if device is not None:
            model = model.to(device)
        self.model = model
        self.image_size = model.config.image_size
        self.batch_size = batch_size if batch_size is not None else model.config.batch_size
        self.device = device

    def output_features(self, features):
        if features is not None and 'cameras-gqn' in features:
            return ['codes', 'cameras-gqn']
        else:
            return ['codes', 'cameras']

    def __call__(self, split, dataset):
        import torch
        import webdataset as wds

        with torch.no_grad():
            dataset = wds.filters.map_(dataset, lambda x: (torch.from_numpy(x['cameras']), torch.from_numpy(self._convert_image_type(x['frames'])), [len(x['frames'])] * len(x['frames'])))
            dataset = wds.filters.unbatched_(dataset)
            dataset = wds.filters.batched_(dataset, self.batch_size)

            past_cameras = None
            past_codes = None

            def update_cummulative_variable(past, value, sequence_sizes):
                sequence_sizes = list(sequence_sizes)
                output = []
                if past is not None:
                    value = torch.cat([past, value], 0)
                    sequence_sizes = ([sequence_sizes[0]] * len(past)) + sequence_sizes
                while len(sequence_sizes) > 0 and len(value) >= sequence_sizes[0]:
                    output.append(value[:sequence_sizes[0]])
                    value = value[sequence_sizes[0]:]
                    sequence_sizes = sequence_sizes[sequence_sizes[0]:]
                past = value
                return past, output

            if hasattr(self.model, 'encode'):
                predict_step = lambda x: self.model.encode(x.to(self.device))[-1].detach().cpu()
            else:
                predict_step = lambda x: self.model(x.to(self.device))[-1].detach().cpu()
            for batch_id, (cameras, frames, sequence_sizes) in enumerate(dataset):
                codes = predict_step(frames)
                past_codes, codes = update_cummulative_variable(past_codes, codes, sequence_sizes)
                past_cameras, cameras = update_cummulative_variable(past_cameras, cameras, sequence_sizes)
                for cur_cameras, cur_codes in zip(cameras, codes):
                    yield dict(cameras=cur_cameras, codes=cur_codes)


@click.command('generate-codes')
def main(dataset: str, output: str, model: str,
         shards: SplitIndices = None,
         batch_size: int = None,
         splits: List[str] = None,
         profile_batch_id: int = None, use_gpu: bool = True):
    import torch
    from viewformer.utils.torch import load_model
    device = 'cpu' if not use_gpu or torch.cuda.device_count() == 0 else 'cuda'
    device = torch.device(device)
    model = load_model(model)
    transformer = LatentCodeTransformer(model, batch_size=batch_size, device=device)
    transform_dataset(dataset, output, transformer,
                      splits=splits,
                      shards=shards)


if __name__ == '__main__':
    main()
