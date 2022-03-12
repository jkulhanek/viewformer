import os
from aparse import click
import tqdm
import tensorflow as tf
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import DatasetLoader
from .evaluate_transformer import generate_batch_predictions, LoaderSwitch


@click.command('generate-gqn-images')
def main(dataset_path: str,
         job_dir: str,
         transformer_model: str,
         codebook_model: str):
    num_eval_sequences = 5
    transformer_model = load_model(transformer_model)
    codebook_model = load_model(codebook_model)
    loader = DatasetLoader(dataset_path, 'test', image_size=codebook_model.config.image_size)
    dataset = tf.data.Dataset.from_generator(lambda: loader,
                                             output_types={
                                                 'frames': tf.uint8,
                                                 'cameras': tf.float32})
    num_eval_sequences = num_eval_sequences if num_eval_sequences is not None else len(loader)
    dataset = dataset.take(num_eval_sequences)
    dataset = dataset.batch(1)
    for i, batch in enumerate(tqdm.tqdm(dataset, total=num_eval_sequences, desc='generating')):
        batch['frames'] = tf.concat((batch['frames'][:, :3], batch['frames'][:, -1:]), 1)
        batch['cameras'] = tf.concat((batch['cameras'][:, :3], batch['cameras'][:, -1:]), 1)
        batch_prediction = generate_batch_predictions(transformer_model, codebook_model, batch['frames'], batch['cameras'])
        for gt_image, gen_image in zip(batch_prediction['ground_truth_images'], batch_prediction['generated_images']):
            tf.io.write_file(os.path.join(job_dir, f'gen{i}.png'), tf.image.encode_png(tf.image.convert_image_dtype(gen_image, 'uint8')))
            tf.io.write_file(os.path.join(job_dir, f'gt{i}.png'), tf.image.encode_png(tf.image.convert_image_dtype(gt_image, 'uint8')))
        for j, img in enumerate(batch['frames'][0, :-1]):
            tf.io.write_file(os.path.join(job_dir, f'c{i}_{j}.png'), tf.image.encode_png(tf.image.convert_image_dtype(img, 'uint8')))


if __name__ == '__main__':
    main()
