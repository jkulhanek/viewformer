import os
import json
import struct
import math
from typing import List, Union, Callable, Any
from functools import partial
import tensorflow as tf
import tqdm
from tensorflow.data import Dataset
from viewformer.utils import SplitIndices, dict_replace
from viewformer.utils.geometry_tf import quaternion_multiply, make_quaternion_y, make_quaternion_x
from viewformer.utils.geometry_tf import quaternion_to_euler
from ._common import get_dataset_url, get_dataset_info, expand_path


def loader_to_dataset(loader):
    assert len(loader) > 0
    first_batch = loader[0]
    types = {k: str(x.dtype) for k, x in first_batch.items()}
    shapes = {k: x.shape for k, x in first_batch.items()}
    dataset = tf.data.Dataset.from_generator(lambda: loader, output_types=types, output_shapes=shapes)
    return dataset


def generate_dataset_handle_existing_settings(path, settings, ignore=None):
    if ignore is None:
        ignore = {'features'}
    else:
        ignore = set(ignore)
    path = os.path.join(path, 'info.json')
    if os.path.exists(path):
        with tf.io.gfile.GFile(path, mode='r') as f:
            old_settings = json.load(f)
            old_settings_str = json.dumps({k: v for k, v in old_settings.items() if not k.endswith('_size') and k not in ignore}, sort_keys=True, indent=2)

        settings_str = json.dumps({k: v for k, v in settings.items() if not k.endswith('_size') and k not in ignore}, sort_keys=True, indent=2)
        if old_settings_str == settings_str:
            return  # Ok, we can override the dataset
        else:
            while True:
                print('There already exists a dataset with the same name, but different parameters')
                print('old parameters:')
                print(old_settings_str)
                print('new parameters:')
                print(settings_str)
                print()
                resp = input('Do you want to override it? [y/n]\n')
                if resp.lower() == 'y':
                    tf.io.gfile.rmtree(os.path.dirname(path))
                    tf.io.gfile.makedirs(os.path.dirname(path))
                    break
                elif resp.lower() == 'n':
                    exit(0)


def transform_viewpoint(v):
    y, p = tf.split(v[..., 3:], 2, axis=-1)

    # position, [yaw, pitch]
    view_vector = [v[..., :3], tf.cos(y), tf.sin(y), tf.cos(p), tf.sin(p)]
    v_hat = tf.concat(view_vector, axis=-1)
    return v_hat


def transform_image(x):
    return x * 2 - 1


def _load_dataset(load_dataset_fn, path: str, split: str, batch_size: int):
    # def generate_distributed_dataset(input_context: tf.distribute.InputContext):
    #     dataset = Dataset.from_tensor_slices(paths)
    #     local_batch_size = input_context.get_per_replica_batch_size(batch_size)
    #     dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    #     if split == 'train':
    #         dataset = dataset.shuffle(1000)
    #     dataset = dataset.interleave(tf.data.TFRecordDataset,
    #                                  cycle_length=num_workers, block_length=1)
    #     dataset = load_dataset_fn(dataset, info, split)
    #     dataset = dataset.batch(local_batch_size)
    #     dataset = dataset.prefetch(prefetch)
    #     return dataset

    # strategy = tf.distribute.get_strategy()
    # return strategy.distribute_datasets_from_function(generate_distributed_dataset)
    pass


def load_image_dataset(path: str, batch_size: int, image_size: int, repeat: int = None):
    info = get_dataset_info(path)
    assert info['frame_size'] == image_size, f'Dataset has a different image size: {info["frame_size"]} != {image_size}'

    def load_split(split):
        paths = [x + '.tfrecord' for x in expand_path(get_dataset_url(path, split, info))]
        feature_description = {
            'frames': tf.io.RaggedFeature(tf.string),
        }

        def parse_example(x):
            x = tf.io.parse_example(x, feature_description)
            return x['frames']

        def preprocess_data(frame):
            frame = tf.io.decode_image(frame, dtype=tf.float32)
            frame = tf.ensure_shape(frame, (info['frame_size'], info['frame_size'], 3))
            frame = transform_image(frame)
            return frame

        def _load_dataset(input_context):
            dataset = Dataset.from_tensor_slices(paths)
            local_batch_size = input_context.get_per_replica_batch_size(batch_size)
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            if split == 'train':
                dataset = dataset.shuffle(1000)
            d = dataset.interleave(tf.data.TFRecordDataset,
                                   cycle_length=tf.data.AUTOTUNE, block_length=1)
            d = d.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            d = d.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
            d = d.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

            # Note, we shuffle both the training and the validation sets
            d = d.shuffle(1000)
            if repeat is not None:
                d = d.repeat(repeat)
            d = d.batch(local_batch_size)
            d = d.prefetch(tf.data.AUTOTUNE)
            return d

        strategy = tf.distribute.get_strategy()
        return strategy.distribute_datasets_from_function(_load_dataset)

    return tuple(map(load_split, ('train', 'test')))


def load_token_dataset(path: str, batch_size: int, sequence_size: int, token_image_size: int, repeat: int = None, max_samples_per_environment: int = -1, transform=None):
    info = get_dataset_info(path.split(',')[0])
    poses_num_dim = 5 if 'cameras-gqn' in info.get('features', set()) else 7

    def load_split(training):
        # shuffle = split == 'train'
        paths = []
        for dpath in path.split(','):
            info = get_dataset_info(dpath)
            split = 'train' if training else ('val' if 'val' in info.get('splits', []) else 'test')
            paths.extend([x + '.tfrecord' for x in expand_path(get_dataset_url(dpath, split, info))])

        feature_description = {
            'cameras': tf.io.RaggedFeature(tf.float32),
            'codes': tf.io.RaggedFeature(tf.int64),
        }

        def parse_example(x):
            x = tf.io.parse_example(x, feature_description)
            poses = tf.reshape(x['cameras'], [-1, poses_num_dim])
            if poses_num_dim == 5:
                poses = fix_legacy_gqn_cameras(poses)
            tokens = tf.reshape(x['codes'], [-1, token_image_size, token_image_size])

            # Shuffle train environments
            # Note, we should also shuffle dev
            indices = tf.range(start=0, limit=tf.shape(poses)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            poses = tf.gather(poses, shuffled_indices)
            tokens = tf.gather(tokens, shuffled_indices)
            return tf.data.Dataset.from_tensors((poses, tokens)).unbatch().batch(sequence_size, drop_remainder=True)

        def _load_dataset(input_context):
            dataset = Dataset.from_tensor_slices(paths)
            local_batch_size = input_context.get_per_replica_batch_size(batch_size)
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            if training:
                dataset = dataset.shuffle(1000)
            d = dataset.interleave(tf.data.TFRecordDataset,
                                   cycle_length=tf.data.AUTOTUNE, block_length=1)
            d = d.interleave(parse_example, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)

            # Sample multiple queries per environment
            def transform_environment(x, y):
                env_d = (tf.data.Dataset.from_tensor_slices((x, y))
                           .shuffle(1000)
                           .batch(sequence_size, drop_remainder=True)
                           .take(max_samples_per_environment))
                if transform is not None:
                    env_d = env_d.map(partial(transform, split='train' if training else 'test'))
                return env_d

            d = d.flat_map(transform_environment)
            d = d.shuffle(1000)
            if repeat is not None:
                d = d.repeat(repeat)
            d = d.batch(local_batch_size)
            d = d.prefetch(tf.data.AUTOTUNE)
            return d

        strategy = tf.distribute.get_strategy()
        return strategy.distribute_datasets_from_function(_load_dataset)

    return tuple(map(load_split, (True, False)))


def format_image(image):
    if len(tf.shape(image)) > 1 and tf.shape(image)[-3] == 3:
        image = tf.transpose(image, (0, 2, 3, 1))
    return image


def fix_legacy_gqn_cameras(poses, position_multiplier=1.0):
    x, y, z, yaw, pitch = tf.unstack(poses, 5, axis=-1)
    return tf.concat(
        (position_multiplier * tf.stack([y, -z, -x], axis=-1),
         quaternion_multiply(make_quaternion_y(math.pi - yaw), make_quaternion_x(pitch))),
        -1)


def get_legacy_gqn_representation(cameras):
    xyz, quaternion = tf.split(cameras, [3, 4], axis=-1)
    x, y, z = tf.unstack(xyz, 3, axis=-1)
    rx, ry, rz = tf.unstack(quaternion_to_euler(quaternion), 3, axis=-1)
    ry = ((math.pi - ry) + math.pi) % (2 * math.pi) - math.pi
    return tf.stack([-z, x, -y, ry, rx], axis=-1)


def read_shards(shard_paths, info, image_size=None,
                features=None, _decode_image=True, shuffle_sequences: bool = False,
                split=None):
    if split is None:
        split = os.path.split(next(iter(shard_paths)))[-1][len(info['name'] + '-'):]
        split = split[:split.rindex('-of')]
        split = split[:split.rindex('-')]
    sequence_size = info.get(f'{split}_sequence_size', None)
    if features is None:
        features = info.get('features', {'cameras', 'frames'})
    if 'codes' in features or 'code_probs' in features or 'code_probs_truncated' in features:
        token_image_size = info['token_image_size']
    if image_size is not None:
        assert info['frame_size'] == image_size, f'Dataset has a different image size: {info["frame_size"]} != {image_size}'

    # Prepare dataset
    feature_description = dict()
    if 'cameras' in features or 'cameras-gqn' in features:
        poses_num_dim = 5 if 'cameras-gqn' in features else 7
        if sequence_size is None:
            feature_description['cameras'] = tf.io.RaggedFeature(tf.float32)
        else:
            feature_description['cameras'] = tf.io.FixedLenFeature([sequence_size * poses_num_dim], tf.float32)
    if 'codes' in features:
        if sequence_size is None:
            feature_description['codes'] = tf.io.RaggedFeature(tf.int64)
        else:
            feature_description['codes'] = tf.io.FixedLenFeature([sequence_size * info['token_image_size'] ** 2], tf.int64)
    if 'images' in features or 'frames' in features:
        if sequence_size is None:
            feature_description['frames'] = tf.io.RaggedFeature(tf.string)
        else:
            feature_description['frames'] = tf.io.FixedLenFeature([sequence_size], tf.string)

    def parse_example(x):
        output = tf.io.parse_example(x, feature_description)
        if 'cameras' in features or 'cameras-gqn' in features:
            poses = tf.reshape(output['cameras'], [-1, poses_num_dim])
            if poses_num_dim == 5:
                poses = fix_legacy_gqn_cameras(poses)
            output['cameras'] = poses
        if 'codes' in features:
            tokens = tf.reshape(output['codes'], [-1, token_image_size, token_image_size])
            output['codes'] = tokens
        if 'frames' in features or 'images' in features:
            frame_size = info['frame_size']
            if _decode_image:
                output['frames'] = tf.map_fn(partial(tf.io.decode_image, dtype=tf.uint8, expand_animations=False), output['frames'], fn_output_signature=tf.uint8)
        return output

    dataset = tf.data.TFRecordDataset(shard_paths)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def get_shard_filename(path, split, shard_id, size):
    return f'{path}-{split}-{shard_id:06d}-of-{size:06d}.tfrecord'


def build_shard_index(tfrecord_file: str, index_file: str) -> None:
    infile = open(tfrecord_file, "rb")
    outfile = open(index_file, "w")

    while True:
        current = infile.tell()
        byte_len = infile.read(8)
        if len(byte_len) == 0:
            break
        infile.read(4)
        proto_len = struct.unpack("q", byte_len)[0]
        infile.read(proto_len)
        infile.read(4)
        outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
    infile.close()
    outfile.close()


def write_shard(path, data, features: List[str]):
    with tf.io.TFRecordWriter(f'{path}.tfrecord.tmp') as current_writer:
        for i, sequence in enumerate(data):
            feature = dict()
            if 'cameras' in features or 'cameras-gqn' in features:
                cameras = tf.convert_to_tensor(sequence['cameras'])
                if hasattr(cameras, 'numpy'):
                    cameras = cameras.numpy()
                feature['cameras'] = tf.train.Feature(float_list=tf.train.FloatList(value=cameras.reshape([-1])))
            if 'codes' in features:
                value = tf.convert_to_tensor(sequence['codes'])
                if value.dtype == 'int32':
                    value = tf.cast(value, tf.int64)
                if hasattr(value, 'numpy'):
                    value = value.numpy()
                feature['codes'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(value, [-1])))
            if 'frames' in features:
                value = tf.convert_to_tensor(sequence['frames'])
                value = format_image(value)
                if hasattr(value[0], 'dtype') and value[0].dtype == 'uint8':
                    value = [x.numpy() for x in tf.map_fn(tf.image.encode_jpeg, value, dtype=tf.string)]
                feature['frames'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            current_writer.write(example.SerializeToString())
        current_writer.flush()
    try:
        build_shard_index(f'{path}.tfrecord.tmp', f'{path}.index')
    except Exception:
        print(f'Failed to create index for shard: {path}.tfrecord')
    tf.io.gfile.rename(f'{path}.tfrecord.tmp', f'{path}.tfrecord', overwrite=True)
