import os
import shutil
import json
from random import Random
from functools import reduce, lru_cache
from itertools import groupby
from tqdm import tqdm
from viewformer.data.tfrecord_dataset import build_shard_index


def _shuffle_split(path, output_path, dataset_info, split, seed):
    assert os.path.exists(os.path.join(path, f'{dataset_info["name"]}-{split}.index'))

    @lru_cache()
    def _get_shard_index(idx):
        index_file = os.path.join(path, f'{dataset_info["name"]}-{split}-{idx:06d}-of-{dataset_info[f"{split}_size"]:06d}.index')
        tfrecord_file = os.path.join(path, f'{dataset_info["name"]}-{split}-{idx:06d}-of-{dataset_info[f"{split}_size"]:06d}.tfrecord')
        if not os.path.exists(index_file):
            build_shard_index(tfrecord_file, index_file)
        return [tuple(map(int, x.rstrip('\n').split(' '))) for x in open(index_file, 'r')]

    index = (x.rstrip('\n').split(' ') for x in open(os.path.join(path, f'{dataset_info["name"]}-{split}.index'), 'r'))
    index = ((i, int(shard.lstrip('0')), int(seq_len)) for i, (shard, seq_len) in enumerate(index))
    index = (list(x) for _, x in groupby(index, key=lambda x: x[1]))
    index = [x + (i,) for split in index for i, x in enumerate(split)]
    os.makedirs(output_path, exist_ok=True)

    rng = Random(seed)
    rng.shuffle(index)
    max_images_per_shard = dataset_info[f'{split}_max_images_per_shard']
    max_sequences_per_shard = dataset_info[f'{split}_max_sequences_per_shard']
    output_index = []
    current_shard_id = 0
    current_shard_len = 0
    current_shard_seq = 0
    for _, _, seq_len, shard_local_id in index:
        output_index.append((current_shard_id + 1, seq_len))
        current_shard_len += seq_len
        current_shard_seq += 1
        if (max_sequences_per_shard is not None and current_shard_seq >= max_sequences_per_shard) or \
                (max_images_per_shard is not None and current_shard_len >= max_images_per_shard):
            current_shard_id += 1
            current_shard_len = 0
            current_shard_seq = 0

    output_stream = tqdm(zip(output_index, index), desc=f'shuffling {split}', total=len(index))
    output_offset = 0
    for shard_id, sequences in groupby(output_stream, key=lambda x: x[0][0]):
        tfrecord_file = os.path.join(output_path, f'{dataset_info["name"]}-{split}-{shard_id:06d}-of-{dataset_info[f"{split}_size"]:06d}.tfrecord')
        index_file = os.path.join(output_path, f'{dataset_info["name"]}-{split}-{shard_id:06d}-of-{dataset_info[f"{split}_size"]:06d}.index')
        with open(tfrecord_file, 'wb+') as tf_f, \
                open(index_file, 'w+') as index_f:
            for (_, seq_len), (_, i_shard_id, _, i_shard_local_id) in sequences:
                index_f.write(f'{shard_id:06d} {seq_len}\n')
                with open(os.path.join(path, f'{dataset_info["name"]}-{split}-{i_shard_id:06d}-of-{dataset_info[f"{split}_size"]:06d}.tfrecord'), 'rb') as input_f:
                    start, record_len = _get_shard_index(i_shard_id)[i_shard_local_id]
                    input_f.seek(start)
                    record_bytes = input_f.read(record_len)
                index_f.write(f'{output_offset} {record_len}\n')
                tf_f.write(record_bytes)
                output_offset += record_len

    if shard_id != dataset_info[f'{split}_size']:
        dataset_info[f'{split}_size'] = shard_id
        with open(os.path.join(output_path, 'info.json'), 'w') as info_f:
            json.dump(dataset_info, info_f)


def shuffle_dataset(path, output_path, seed: int = 42):
    # Copy dataset info
    if os.path.exists(output_path):
        raise RuntimeError(f'Output path {output_path} already exists')

    os.makedirs(output_path, exist_ok=True)
    shutil.copy(os.path.join(path, 'info.json'), os.path.join(output_path, 'info.json'))
    dataset_info = json.load(open(os.path.join(path, 'info.json'), 'r'))
    splits = dataset_info['splits']
    for split in splits:
        local_seed = seed ^ (reduce(lambda a, x: a * ord(x), split, 1) % 31)
        _shuffle_split(path, output_path, dataset_info, split, local_seed)
