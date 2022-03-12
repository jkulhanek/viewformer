#!/bin/env python
from aparse import click, ConditionalType, WithArgumentName, AllArguments
import os
import json
from functools import reduce, partial
import math
import tensorflow as tf
from viewformer.models import supported_config_dict
from viewformer.utils.tensorflow import load_model
from viewformer.models import AutoModel
from viewformer.data.tfrecord_dataset import load_token_dataset
from viewformer.utils import geometry_tf as geometry
from .utils import CustomLoggingCallback, ModelCheckpoint, get_strategy


class UseOptimizerIterationAsTrainStep(tf.keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.model._train_counter.assign(self.model.optimizer.iterations)


ModelSwitch = WithArgumentName(ConditionalType('ModelSwitch', supported_config_dict(), default='migt', prefix=None), 'model')


def chain(*fx):
    return lambda x, **kwargs: reduce(lambda x, f: f(x, **kwargs), (x for x in fx if x is not None), x)


def process_batch(cameras, tokens, augment, split):
    xyz, quaternion = tf.split(cameras, (3, 4), axis=-1)
    if augment == 'relative':
        rotation_inverse = geometry.quaternion_conjugate(quaternion[..., :1, :])
        xyz_orig = xyz[..., :1, :]
        xyz = xyz - xyz_orig
        xyz = geometry.quaternion_rotate(xyz, rotation_inverse)
        quaternion = geometry.quaternion_multiply(rotation_inverse, quaternion)
    elif augment == 'no' or split != 'train':
        pass
    elif augment == 'simple':
        xyz = xyz + tf.random.normal((1, 3), dtype=xyz.dtype)
        rotation = geometry.quaternion_multiply(
            geometry.make_quaternion_y(tf.random.uniform((1,), 0, 2 * math.pi, dtype=xyz.dtype)),
            geometry.quaternion_multiply(
                geometry.make_quaternion_x(tf.random.uniform((1,), 0, math.pi / 8, dtype=xyz.dtype)),
                geometry.make_quaternion_y(tf.random.uniform((1,), 0, 2 * math.pi, dtype=xyz.dtype)),
            ))

        xyz = geometry.quaternion_rotate(xyz, rotation)
        quaternion = geometry.quaternion_multiply(quaternion, rotation)
    elif augment == 'advanced':
        xyz = xyz + tf.random.normal((1, 3), dtype=xyz.dtype)
        rotation = geometry.make_quaternion_y(tf.random.uniform((1,), 0, 2 * math.pi, dtype=xyz.dtype))
        xyz = geometry.quaternion_rotate(xyz, rotation)
        quaternion = geometry.quaternion_multiply(quaternion, rotation)

    else:
        raise ValueError(f'Augment {augment} is not supported')

    quaternion = geometry.quaternion_normalize(quaternion)
    quaternion = geometry.quaternion_remove_sign(quaternion)
    cameras = tf.concat([xyz, quaternion], -1)
    return cameras, tokens


@click.command('train_transformer', soft_defaults=True)
def main(codebook_model: str,
         model_config: ModelSwitch,
         args: AllArguments,
         job_dir: str,
         dataset: str,
         batch_size: int,
         total_steps: int,
         validation_images: int = 32,
         max_samples_per_environment: int = -1,
         ddp: bool = False,
         tpu: bool = False,
         fp16: bool = False,
         wandb: bool = False,
         epochs: int = 100,
         n_embeddings=None):
    assert n_embeddings is None
    distributed_strategy = get_strategy(ddp, tpu)
    if wandb:
        import wandb
        hparams = {k: v for k, v in args.items() if not k.startswith('wandb_')}
        wandb.init(config=hparams, resume='allow')
        wandb.tensorboard.patch(root_logdir=job_dir)

    validation_steps = max(1, min((total_steps // epochs) // 10, 100))

    with distributed_strategy.scope():
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            job_dir,
            profile_batch=50,
            update_freq=10)  # Log every 10 batches

        codebook_model = load_model(codebook_model)
        model_config.n_embeddings = codebook_model.config.n_embed

        if fp16:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16' if not tpu else 'mixed_bfloat16')

        model = AutoModel.from_config(model_config)
        model.codebook_model = codebook_model
        model.compile()
        train_dataset, test_dataset = load_token_dataset(dataset,
                                                         batch_size=batch_size,
                                                         sequence_size=model.config.sequence_size,
                                                         token_image_size=model.config.token_image_size,
                                                         max_samples_per_environment=max_samples_per_environment,
                                                         repeat=-1,
                                                         transform=partial(process_batch, augment=model.config.augment_poses))
        model.fit(train_dataset, batch_size=batch_size,
                  callbacks=[
                      tf.keras.callbacks.experimental.BackupAndRestore(os.path.join(job_dir, '.backup')),
                      UseOptimizerIterationAsTrainStep(),
                      tensorboard_callback,
                      CustomLoggingCallback(test_dataset, args, tensorboard_callback, validation_images),
                      ModelCheckpoint(job_dir),
                  ],
                  epochs=epochs,
                  steps_per_epoch=total_steps // epochs,
                  validation_steps=validation_steps,
                  validation_data=test_dataset)


if __name__ == '__main__':
    main()
