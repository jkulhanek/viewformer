from aparse import click, ConditionalType
import os
import tqdm
import json
from typing import Optional
from collections import OrderedDict
import tensorflow as tf
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import get_loaders
from viewformer.evaluate.evaluate_transformer import Evaluator, to_relative_cameras, from_relative_cameras, resize_tf, normalize_cameras


class MultiContextEvaluator:
    def __init__(self, sequence_size: int, image_size: Optional[int] = None):
        self.sequence_size = sequence_size
        self._evaluators = [Evaluator(image_size=image_size) for _ in range(sequence_size - 1)]

    def update_state(self, ground_truth_cameras, generated_cameras, ground_truth_images, generated_images):
        for i, (gen_cam, gen_img) in enumerate(zip(tf.unstack(generated_cameras, axis=1), tf.unstack(generated_images, axis=1))):
            if i == 0:
                continue
            self._evaluators[i - 1].update_state(ground_truth_cameras, gen_cam, ground_truth_images, gen_img)

    def get_progress_bar_info(self):
        return self._evaluators[-1].get_progress_bar_info()

    def result(self):
        return OrderedDict((
            (f'ctx{i + 1:02d}', x.result())
            for i, x in enumerate(self._evaluators)))


def generate_batch_predictions(transformer_model, codebook_model, images, cameras):
    ground_truth_cameras = cameras[:, -1]
    if transformer_model.config.augment_poses == 'relative':
        # Rotate poses for relative augment
        cameras, transform = to_relative_cameras(cameras)
    cameras = normalize_cameras(cameras)

    with tf.name_scope('encode'):
        def encode(images):
            fimages = resize_tf(images, codebook_model.config.image_size)
            fimages = tf.image.convert_image_dtype(fimages, tf.float32)
            fimages = fimages * 2 - 1
            codes = codebook_model.encode(fimages)[-1]  # [N, H', W']
            codes = tf.cast(codes, dtype=tf.int32)
            return codes

        # codes = tf.ragged.map_flat_values(encode, codes)
        batch_size, seq_len, *im_dim = tf.unstack(tf.shape(images), 5)
        code_len = transformer_model.config.token_image_size
        codes = tf.reshape(encode(tf.reshape(images, [batch_size * seq_len] + list(im_dim))), [batch_size, seq_len, code_len, code_len])

    # Generate image tokens
    with tf.name_scope('transformer_generate_images'):
        # Remove prediction information
        input_ids = tf.concat([codes[:, :-1], tf.fill(tf.shape(codes[:, :1]),
                                                      tf.constant(transformer_model.mask_token, dtype=codes.dtype))], 1)
        context_cameras = tf.concat([cameras[:, :-1], tf.zeros_like(cameras[:, :1])], 1)

        # Task specific outputs
        image_generation_query_cameras = tf.tile(cameras[:, -1:], [1, tf.shape(cameras)[1], 1])
        localization_query_tokens = tf.tile(codes[:, -1:], [1, tf.shape(cameras)[1], 1, 1])

        # Generate outputs
        output = transformer_model(dict(input_ids=input_ids,
                                        poses=context_cameras,
                                        localization_tokens=localization_query_tokens,
                                        output_poses=image_generation_query_cameras), training=False)

        # Format output
        generated_codes = tf.argmax(output['logits'], -1)
        generated_cameras = transformer_model.reduce_cameras(output['pose_prediction'], -2)

    # Decode images
    with tf.name_scope('decode_images'):
        batch_size, seq_len, token_image_shape = tf.split(tf.shape(generated_codes), (1, 1, 2), 0)
        generated_images = codebook_model.decode_code(tf.reshape(generated_codes, tf.concat((batch_size * seq_len, token_image_shape), 0)))
        generated_images = tf.clip_by_value(generated_images, -1, 1)
        generated_images = tf.image.convert_image_dtype(generated_images / 2 + 0.5, tf.uint8)
        generated_images = tf.reshape(generated_images, tf.concat((batch_size, seq_len, tf.shape(generated_images)[-3:]), 0))

    # Erase relative transform
    if transformer_model.config.augment_poses == 'relative':
        generated_cameras = from_relative_cameras(generated_cameras, transform)

    return dict(
        ground_truth_images=images[:, -1],
        generated_images=generated_images,
        ground_truth_cameras=ground_truth_cameras,
        generated_cameras=generated_cameras)


def build_store_predictions(job_dir, limit: int = None):
    os.makedirs(job_dir, exist_ok=True)
    i = 0

    def store_predictions(ground_truth_cameras, generated_cameras, ground_truth_images, generated_images, postfix: str = '', ctx=None):
        nonlocal i
        for bi, (gt_cam, gen_cam_batch, gt_img, gen_img_batch) in enumerate(zip(ground_truth_cameras,
                                                                            generated_cameras,
                                                                            ground_truth_images,
                                                                            generated_images)):
            if limit != -1 and i >= limit:
                return
            tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gt{postfix}.png')), tf.io.encode_png(gt_img))
            tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gt{postfix}.cam')), tf.io.serialize_tensor(gt_cam))
            for ctx_size, (gen_cam, gen_img) in enumerate(zip(gen_cam_batch, gen_img_batch)):
                tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gen@{ctx_size:02d}{postfix}.png')), tf.io.encode_png(gen_img))
                tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gen@{ctx_size:02d}{postfix}.cam')), tf.io.serialize_tensor(gen_cam))

            ctx_dir = os.path.join(job_dir, f'{i:08d}-ctx{postfix}')
            if ctx is not None:
                os.makedirs(ctx_dir, exist_ok=True)
                for j, ctx_img in enumerate(ctx[bi]):
                    tf.io.write_file(tf.constant(os.path.join(ctx_dir, f'{j:02}.png')), tf.io.encode_png(ctx_img))
            i += 1
    return store_predictions


def print_metrics(metrics, precision=4):
    yheader = list(metrics.keys())
    xheader = list(next(iter(metrics.values())).keys())
    num_format = f'{{0:.{precision}f}}'
    table_vals = [[num_format.format(metrics[ctx_size][metric]) for metric in xheader] for ctx_size in yheader]

    table = [[metric] + vals for metric, vals in zip(yheader, table_vals)]
    cell_lens = [max(len(row[j]) for row in (table + [[''] + xheader])) for j in range(len(table[0]))]
    row_format = '  '.join([f'{{{i}: >{l}}}' if i != 0 else f'{{{i}: <{l}}}' for i, l in enumerate(cell_lens)])
    prefix = ' '
    print(prefix + row_format.format('', *xheader))
    print(prefix + '  '.join('-' * cl for cl in cell_lens))
    for row in table:
        print(prefix + row_format.format(*row))


#
# Types used in argument parsing
#
def _loader_switch_cls(cls):
    class Loader(cls):
        # Disable image_size argument in loader classes
        def __init__(self, *args, image_size=None, **kwargs):
            raise NotImplementedError()

        def __new__(_cls, *args, **kwargs):
            # Return callback to construct Loader on the Fly
            return lambda image_size: cls(*args, **kwargs, image_size=image_size)
    return Loader


LoaderSwitch = ConditionalType('Loader', {k: _loader_switch_cls(v) for k, v in get_loaders().items()}, default='dataset')


@click.command('evaluate')
def main(loader: LoaderSwitch,
         transformer_model: str,
         codebook_model: str,
         job_dir: str,
         batch_size: int,
         num_eval_sequences: Optional[int] = None,
         pose_multiplier: Optional[float] = None,
         sequence_size: Optional[int] = None,
         num_store_images: int = 100,
         store_ctx: bool = False,
         image_size: Optional[int] = None):
    transformer_config = dict()
    if pose_multiplier is not None:
        transformer_config['pose_multiplier'] = pose_multiplier
    transformer_model = load_model(transformer_model, **transformer_config)
    codebook_model = load_model(codebook_model)
    if sequence_size is None:
        sequence_size = transformer_model.config.sequence_size
    loader = loader(codebook_model.config.image_size)
    store_predictions = build_store_predictions(job_dir, num_store_images)
    evaluator = MultiContextEvaluator(sequence_size, image_size=image_size)
    dataset = tf.data.Dataset.from_generator(lambda: ((x['frames'][:sequence_size], x['cameras'][:sequence_size]) for x in loader),
                                             output_types=(tf.uint8, tf.float32))
    num_eval_sequences = num_eval_sequences if num_eval_sequences is not None else len(loader)
    dataset = dataset.take(num_eval_sequences)
    dataset = dataset.batch(batch_size)

    with tqdm.tqdm(total=(num_eval_sequences + batch_size - 1) // batch_size, desc='evaluating') as progress:
        for frames, cameras in tqdm.tqdm(dataset):
            batch_prediction = generate_batch_predictions(transformer_model, codebook_model, frames, cameras)
            evaluator.update_state(**batch_prediction)
            if store_ctx:
                batch_prediction['ctx'] = frames[:, :-1]
            store_predictions(**batch_prediction)
            progress.set_postfix(evaluator.get_progress_bar_info())
            progress.update()
    result = evaluator.result()
    with open(os.path.join(job_dir, 'results.json'), 'w+') as f:
        json.dump(result, f)
    print('Results:')
    print_metrics(result)


if __name__ == '__main__':
    main()
