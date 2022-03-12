from aparse import click, ConditionalType
import os
import tqdm
import json
from typing import Optional
from collections import OrderedDict
from itertools import chain
import tensorflow as tf
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import get_loaders
from viewformer.data._common import resize
from viewformer.utils.metrics import LPIPSMetric, SSIMMetric, PSNRMetric, ImageRMSE


def resize_tf(images, size, method=None):
    return tf.convert_to_tensor(resize(images.numpy(), size, method=method))


class Evaluator:
    def __init__(self, image_size: int = None):
        self.image_size = image_size
        self._image_generation_metrics = [
            tf.keras.metrics.MeanSquaredError('mse'),
            ImageRMSE('rmse'),
            tf.keras.metrics.MeanAbsoluteError('mae'),
            PSNRMetric('psnr'),
            LPIPSMetric('vgg', name='lpips'),
            SSIMMetric('ssim')]

    def update_state(self, ground_truth_images, generated_images):
        image_size = self.image_size
        if image_size is None:
            image_size = tf.maximum(tf.shape(ground_truth_images)[-2], tf.shape(generated_images)[-2])
        ground_truth_images = resize_tf(ground_truth_images, image_size)
        if tf.shape(generated_images)[-2] != image_size:
            # When upsampling generated image, we will use bilinear as well
            generated_images = resize_tf(generated_images, image_size, 'bilinear')
        for metric in self._image_generation_metrics:
            metric.update_state(ground_truth_images, generated_images)

    def get_progress_bar_info(self):
        return OrderedDict([
            ('img_rgbl1', float(next((x for x in self._image_generation_metrics if x.name == 'mae')).result())),
            ('img_lpips', float(next((x for x in self._image_generation_metrics if x.name == 'lpips')).result()))])

    def result(self):
        return OrderedDict((
            (m.name, float(m.result()))
            for m in chain(self._image_generation_metrics)))


def build_store_predictions(job_dir, limit: int = None):
    os.makedirs(job_dir, exist_ok=True)
    # assert len(os.listdir(job_dir)) == 0, f'Evaluation directory {job_dir} is not empty'
    i = 0

    def store_predictions(ground_truth_images, generated_images, postfix: str = ''):
        nonlocal i
        for gt_img, gen_img in zip(ground_truth_images, generated_images):
            if limit is None or limit == -1 or i < limit:
                tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gen{postfix}.png')), tf.io.encode_png(gen_img))
                tf.io.write_file(tf.constant(os.path.join(job_dir, f'{i:08d}-gt{postfix}.png')), tf.io.encode_png(gt_img))
            i += 1
    return store_predictions


def generate_batch_predictions(codebook_model, images):
    fimages = resize_tf(images, codebook_model.config.image_size)
    fimages = tf.image.convert_image_dtype(fimages, tf.float32) * 2 - 1
    codes = codebook_model.encode(fimages)[-1]  # [N, H', W']
    generated_images = codebook_model.decode_code(codes)
    generated_images = tf.clip_by_value(generated_images, -1, 1)
    generated_images = tf.image.convert_image_dtype(generated_images / 2 + 0.5, tf.uint8)

    return dict(
        ground_truth_images=images,
        generated_images=generated_images)


#
# Types used in argument parsing
#
def _loader_switch_cls(cls):
    class Loader(cls):
        # Disable image_size argument in loader classes
        def __init__(self, *args, image_size=None, sequence_size=None, **kwargs):
            raise NotImplementedError()

        def __new__(_cls, *args, **kwargs):
            # Return callback to construct Loader on the Fly
            return lambda image_size, sequence_size: cls(*args, **kwargs, image_size=image_size, sequence_size=sequence_size)
    return Loader


LoaderSwitch = ConditionalType('Loader', {k: _loader_switch_cls(v) for k, v in get_loaders().items()}, default='dataset')


@click.command('evaluate')
def main(loader: LoaderSwitch,
         codebook_model: str,
         job_dir: str,
         batch_size: int,
         num_eval_images: Optional[int] = None,
         num_store_images: int = -1,
         single_image_per_scene: bool = True,
         image_size: Optional[int] = None):
    codebook_model = load_model(codebook_model)
    if single_image_per_scene:
        loader = loader(None, None)
    else:
        loader = loader(None, 1)
    store_predictions = build_store_predictions(job_dir, num_store_images)
    evaluator = Evaluator(image_size=image_size)
    dataset = tf.data.Dataset.from_generator(lambda: (x['frames'] for x in loader),
                                             output_types=tf.uint8)
    if single_image_per_scene:
        dataset = dataset.map(lambda x: x[:1])
    dataset = dataset.unbatch()
    if num_eval_images is not None:
        dataset = dataset.take(num_eval_images)
    else:
        num_eval_images = sum(loader.num_images_per_sequence())
    dataset = dataset.batch(batch_size)

    with tqdm.tqdm(total=(num_eval_images + batch_size - 1) // batch_size, desc='evaluating') as progress:
        for batch in tqdm.tqdm(dataset):
            batch_prediction = generate_batch_predictions(codebook_model, batch)
            store_predictions(**batch_prediction)
            evaluator.update_state(**batch_prediction)
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
