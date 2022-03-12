from aparse import click, ConditionalType
import os
import tqdm
import json
from typing import Optional
from collections import OrderedDict
from itertools import chain
import tensorflow as tf
from viewformer.utils import geometry_tf as geometry
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import get_loaders
from viewformer.utils.metrics import CameraOrientationError, CameraPositionError, PSNRMetric
from viewformer.utils.metrics import CameraPositionMedian, CameraOrientationMedian
from viewformer.utils.metrics import LPIPSMetric, SSIMMetric, ImageRMSE
from viewformer.data._common import resize


def resize_tf(images, size, method=None):
    return tf.convert_to_tensor(resize(images.numpy(), size, method=method))


class Evaluator:
    def __init__(self, image_size: int = None):
        self.image_size = image_size
        self._localization_metrics = [CameraOrientationError('loc-angle'),
                                      CameraPositionError('loc-dist'),
                                      CameraOrientationMedian('loc-angle-med'),
                                      CameraPositionMedian('loc-dist-med')]
        self._image_generation_metrics = [
            tf.keras.metrics.MeanSquaredError('mse'),
            ImageRMSE('rmse'),
            tf.keras.metrics.MeanAbsoluteError('mae'),
            PSNRMetric('psnr'),
            LPIPSMetric('vgg', name='lpips'),
            SSIMMetric('ssim')]

    def update_with_image(self, ground_truth_images, generated_images):
        image_size = self.image_size
        if image_size is None:
            image_size = tf.maximum(tf.shape(ground_truth_images)[-2], tf.shape(generated_images)[-2])
        ground_truth_images = resize_tf(ground_truth_images, image_size)
        if tf.shape(generated_images)[-2] != image_size:
            # When upsampling generated image, we will use bilinear as well
            generated_images = resize_tf(generated_images, image_size, 'bilinear')
        for metric in self._image_generation_metrics:
            metric.update_state(ground_truth_images, generated_images)

    def update_with_camera(self, ground_truth_cameras, generated_cameras):
        for metric in self._localization_metrics:
            metric.update_state(ground_truth_cameras, generated_cameras)

    def update_state(self, ground_truth_cameras, generated_cameras, ground_truth_images, generated_images):
        self.update_with_image(ground_truth_images, generated_images)
        self.update_with_camera(ground_truth_cameras, generated_cameras)

    def get_progress_bar_info(self):
        return OrderedDict([
            ('img_rgbl1', float(next((x for x in self._image_generation_metrics if x.name == 'mae')).result())),
            ('img_lpips', float(next((x for x in self._image_generation_metrics if x.name == 'lpips')).result())),
            ('cam_loc', float(next((x for x in self._localization_metrics if x.name == 'loc-dist')).result())),
            ('cam_ang', float(next((x for x in self._localization_metrics if x.name == 'loc-angle')).result()))])

    def result(self):
        return OrderedDict((
            (m.name, float(m.result()))
            for m in chain(self._localization_metrics, self._image_generation_metrics)))


def to_relative_cameras(cameras):
    xyz, quaternion = tf.split(cameras, (3, 4), -1)
    transform_xyz = xyz[..., :1, :]
    transform_quaternion = quaternion[..., :1, :]
    rotation_inverse = geometry.quaternion_conjugate(transform_quaternion)
    xyz = xyz - transform_xyz
    xyz = geometry.quaternion_rotate(xyz, rotation_inverse)
    quaternion = geometry.quaternion_multiply(rotation_inverse, quaternion)
    return tf.concat((xyz, quaternion), -1), tf.concat((transform_xyz, transform_quaternion), -1)


def from_relative_cameras(cameras, transform):
    transform_xyz, transform_quaternion = tf.split(transform, (3, 4), -1)
    xyz, quaternion = tf.split(cameras, (3, 4), -1)
    quaternion = geometry.quaternion_multiply(transform_quaternion, quaternion)
    xyz = geometry.quaternion_rotate(xyz, transform_quaternion)
    xyz = xyz + transform_xyz
    return tf.concat((xyz, quaternion), -1)


def normalize_cameras(cameras):
    xyz, quaternion = tf.split(cameras, (3, 4), -1)
    quaternion = geometry.quaternion_normalize(quaternion)
    quaternion = geometry.quaternion_remove_sign(quaternion)
    return tf.concat((xyz, quaternion), -1)


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
        image_generation_input_ids = tf.concat([codes[:, :-1], tf.fill(tf.shape(codes[:, :1]),
                                                                       tf.constant(transformer_model.mask_token, dtype=codes.dtype))], 1)
        output = transformer_model(dict(input_ids=image_generation_input_ids, poses=cameras), training=False)
        generated_codes = tf.argmax(output['logits'], -1)[:, -1]

    # Decode images
    with tf.name_scope('decode_images'):
        generated_images = codebook_model.decode_code(generated_codes)
        generated_images = tf.clip_by_value(generated_images, -1, 1)
        generated_images = tf.image.convert_image_dtype(generated_images / 2 + 0.5, tf.uint8)

    # Generate cameras
    # If the model supports is
    # Otherwise, we return first context pose
    if transformer_model.use_localization:
        output = transformer_model(dict(input_ids=codes, poses=cameras[:, :-1]), training=False)
        generated_cameras = transformer_model.reduce_cameras(output['pose_prediction'][:, -1:], -2)
    else:
        generated_cameras = cameras[:, :1]
    if transformer_model.config.augment_poses == 'relative':
        generated_cameras = from_relative_cameras(generated_cameras, transform)

    return dict(
        ground_truth_images=images[:, -1],
        generated_images=generated_images,
        ground_truth_cameras=ground_truth_cameras,
        generated_cameras=generated_cameras[:, -1])


def build_store_predictions(job_dir, limit: int = None):
    os.makedirs(job_dir, exist_ok=True)
    # assert len(os.listdir(job_dir)) == 0, f'Evaluation directory {job_dir} is not empty'
    i = 0

    def store_predictions(ground_truth_cameras, generated_cameras, ground_truth_images, generated_images, postfix: str = '', ctx=None):
        nonlocal i
        for bi, (gt_cam, gen_cam, gt_img, gen_img) in enumerate(zip(ground_truth_cameras,
                                                                generated_cameras,
                                                                ground_truth_images,
                                                                generated_images)):
            if limit != -1 and i >= limit:
                return
            tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gen{postfix}.png')), tf.io.encode_png(gen_img))
            tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gt{postfix}.png')), tf.io.encode_png(gt_img))
            tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gen{postfix}.cam')), tf.io.serialize_tensor(gen_cam))
            tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gt{postfix}.cam')), tf.io.serialize_tensor(gt_cam))
            ctx_dir = os.path.join(job_dir, f'{i:08d}-ctx{postfix}')
            if ctx is not None:
                os.makedirs(ctx_dir, exist_ok=True)
                for j, ctx_img in enumerate(ctx[bi]):
                    tf.io.write_file(tf.constant(os.path.join(ctx_dir, f'{j:02}.png')), tf.io.encode_png(ctx_img))
            i += 1
    return store_predictions


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
    evaluator = Evaluator(image_size=image_size)
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
    for m, val in result.items():
        print(f'    {m}: {val:.6f}')


if __name__ == '__main__':
    main()
