from aparse import click, ConditionalType
import os
import tqdm
import json
import numpy as np
from PIL import Image
from typing import Optional
from typing import List
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import get_loaders
import tensorflow as tf
from viewformer.evaluate.evaluate_transformer_multictx import MultiContextEvaluator, print_metrics, to_relative_cameras, resize_tf, normalize_cameras, from_relative_cameras


def transformer_predict(cameras, codes, *, transformer_model):
    if transformer_model.config.augment_poses == 'relative':
        # Rotate poses for relative augment
        cameras, transform = to_relative_cameras(cameras)
    cameras = normalize_cameras(cameras)

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

    # Erase relative transform
    if transformer_model.config.augment_poses == 'relative':
        generated_cameras = from_relative_cameras(generated_cameras, transform)
    return generated_cameras, generated_codes


def run_with_batchsize(fn, batch_size, *args, **kwargs):
    total = len(args[0])
    splits = [min(batch_size, (total - i * batch_size)) for i in range((total + batch_size - 1) // batch_size)]
    outs = []
    for i, bs in enumerate(splits):
        largs = [x[i * batch_size: (i+1) * batch_size] for x in args]
        louts = fn(*largs, **kwargs)
        outs.append(louts)
    if isinstance(outs[0], tuple):
        return tuple(tf.concat([x[i] for x in outs], 0) for i in range(len(outs[0])))
    else:
        return tf.concat(outs, 0)


def encode_images(frames, *, codebook_model):
    with tf.name_scope('encode'):
        def encode(images):
            fimages = resize_tf(images, codebook_model.config.image_size)
            fimages = tf.image.convert_image_dtype(fimages, tf.float32)
            fimages = fimages * 2 - 1
            codes = codebook_model.encode(fimages)[-1]  # [N, H', W']
            codes = tf.cast(codes, dtype=tf.int32)
            return codes

        # codes = tf.ragged.map_flat_values(encode, codes)
        batch_size, seq_len, *im_dim = tf.unstack(tf.shape(frames), 5)
        codes = encode(tf.reshape(frames, [batch_size * seq_len] + list(im_dim)))
        code_len = tf.shape(codes)[-1]
        codes = tf.reshape(codes, [batch_size, seq_len, code_len, code_len])
    return codes


def decode_code(generated_codes, *, codebook_model):
    with tf.name_scope('decode_images'):
        batch_size, seq_len, token_image_shape = tf.split(tf.shape(generated_codes), (1, 1, 2), 0)
        generated_images = codebook_model.decode_code(tf.reshape(generated_codes, tf.concat((batch_size * seq_len, token_image_shape), 0)))
        generated_images = tf.clip_by_value(generated_images, -1, 1)
        generated_images = tf.image.convert_image_dtype(generated_images / 2 + 0.5, tf.uint8)
        generated_images = tf.reshape(generated_images, tf.concat((batch_size, seq_len, tf.shape(generated_images)[-3:]), 0))
    return generated_images


#
# Types used in argument parsing
#
def _loader_switch_cls(cls):
    class Loader(cls):
        # Disable arguments in loader classes
        def __init__(self, *args, image_size=None, shuffle=None, shuffle_sequence_items=None, shuffle_sequences=None, **kwargs):
            raise NotImplementedError()

        def __new__(_cls, *args, **kwargs):
            # Return callback to construct Loader on the Fly
            return lambda image_size: cls(*args, **kwargs, image_size=image_size, shuffle_sequences=False, shuffle_sequence_items=False)
    return Loader


LoaderSwitch = ConditionalType('Loader', {k: _loader_switch_cls(v) for k, v in get_loaders().items()}, default='dataset')


@click.command('evaluate-allimg')
def main(loader: LoaderSwitch,
         transformer_model: str,
         codebook_model: str,
         job_dir: str,
         context_views: List[int] = None,
         pose_multiplier: Optional[float] = None,
         image_size: Optional[int] = None):
    transformer_config = dict()
    if pose_multiplier is not None:
        transformer_config['pose_multiplier'] = pose_multiplier
    transformer_model = load_model(transformer_model, **transformer_config)
    codebook_model = load_model(codebook_model)
    loader = loader(codebook_model.config.image_size)
    n_context_views = len(context_views) if context_views is not None else (transformer_model.config.sequence_size - 1)
    evaluator = MultiContextEvaluator(n_context_views + 1, image_size=image_size)
    rng = np.random.default_rng(42)

    with tqdm.tqdm(total=len(loader)) as progress:
        for seq in loader:
            c_context_views = context_views
            if c_context_views is None:
                c_context_views = list(rng.choice(len(seq['frames']), (n_context_views,), replace=False))
            frames = np.array(seq['frames'])[np.newaxis, ...]
            cameras = np.stack(seq['cameras'])[np.newaxis, ...].astype('float32')
            frames, cameras = tf.convert_to_tensor(frames), tf.convert_to_tensor(cameras)
            codes = encode_images(frames, codebook_model=codebook_model)
            generated_cameras, generated_codes = [], []
            tcodes = np.concatenate([np.stack([codes[:, j] for j in c_context_views + [i]], 1) for i in range(len(seq['frames']))], 0)
            tcameras = np.concatenate([np.stack([cameras[:, j] for j in c_context_views + [i]], 1) for i in range(len(seq['frames']))], 0)
            generated_cameras, generated_codes = run_with_batchsize(transformer_predict, 128, tcameras, tcodes, transformer_model=transformer_model)

            # Decode images
            generated_images = run_with_batchsize(decode_code, 64, generated_codes, codebook_model=codebook_model)
            eval_frames = [x for x in range(len(generated_images)) if x not in c_context_views]
            evaluator.update_state(
                ground_truth_cameras=tf.stack([cameras[0, x] for x in eval_frames], 0),
                ground_truth_images=tf.stack([frames[0, x] for x in eval_frames], 0),
                generated_images=tf.stack([generated_images[x] for x in eval_frames], 0),
                generated_cameras=tf.stack([generated_cameras[x] for x in eval_frames], 0))
            for i in range(0, 1 + len(c_context_views)):
                os.makedirs(os.path.join(job_dir, 'gen_images', seq['sequence_id'], f'gen-{i:02}'), exist_ok=True)
            os.makedirs(os.path.join(job_dir, 'gen_images', seq['sequence_id'], 'gt'), exist_ok=True)
            os.makedirs(os.path.join(job_dir, 'gen_images', seq['sequence_id'], 'ctx'), exist_ok=True)
            for i, c in enumerate(c_context_views):
                Image.fromarray(frames[0, c].numpy()).save(os.path.join(job_dir, 'gen_images', seq['sequence_id'], 'ctx', f'{i:02}-{c:03}.png'))
            for i, c in enumerate(frames[0]):
                Image.fromarray(c.numpy()).save(os.path.join(job_dir, 'gen_images', seq['sequence_id'], 'gt', f'{i:03}.png'))
            for i, c in enumerate(generated_images):
                for j, d in enumerate(c):
                    Image.fromarray(d.numpy()).save(os.path.join(job_dir, 'gen_images', seq['sequence_id'], f'gen-{j:02}', f'{i:03}.png'))
            progress.set_postfix(evaluator.get_progress_bar_info())
            progress.update()

    result = evaluator.result()
    with open(os.path.join(job_dir, 'results.json'), 'w+') as f:
        json.dump(result, f)
    print('Results:')
    print_metrics(result)


if __name__ == '__main__':
    main()
