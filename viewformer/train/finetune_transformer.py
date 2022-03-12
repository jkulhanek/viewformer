#!/bin/env python
from aparse import click, AllArguments
from typing import Optional
from functools import partial
import tensorflow as tf
from viewformer.utils.schedules import Schedule
from viewformer.models.utils import create_optimizer
from viewformer.utils.tensorflow import load_model
from viewformer.data.tfrecord_dataset import load_token_dataset
from .train_transformer import process_batch
from .utils import CustomLoggingCallback, ModelCheckpoint, get_strategy


@click.command('finetune_transformer', soft_defaults=True)
def main(codebook_model: str,
         checkpoint: str,
         args: AllArguments,
         job_dir: str,
         dataset: str,
         total_steps: int,
         validation_images: int = 32,
         learning_rate: float = 1e-5,
         pose_multiplier: Optional[float] = None,
         batch_size: Optional[int] = None,
         localization_weight: Schedule = None,
         sequence_size: Optional[int] = None,
         n_loss_skip: Optional[int] = None,
         weight_decay: Optional[float] = None,
         augment_poses: Optional[str] = None,
         gradient_clip_val: Optional[float] = None,
         epochs: int = 10,
         fp16: bool = False,
         wandb: bool = False):
    distributed_strategy = get_strategy(False, False)
    if wandb:

        import wandb
        hparams = {k: v for k, v in args.items() if not k.startswith('wandb_')}
        wandb.init(config=hparams)
        wandb.tensorboard.patch(root_logdir=job_dir)

    validation_steps = max(1, min((total_steps // epochs) // 10, 100))

    with distributed_strategy.scope():
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            job_dir,
            profile_batch=50,
            update_freq=10)  # Log every 10 batches

        codebook_model = load_model(codebook_model)

        if fp16:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')

        transformer_config = dict()
        if pose_multiplier is not None:
            transformer_config['pose_multiplier'] = pose_multiplier
        if localization_weight is not None:
            transformer_config['localization_weight'] = localization_weight
        if sequence_size is not None:
            transformer_config['sequence_size'] = sequence_size
        if n_loss_skip is not None:
            transformer_config['n_loss_skip'] = n_loss_skip
        if weight_decay is not None:
            transformer_config['weight_decay'] = weight_decay
        if gradient_clip_val is not None:
            transformer_config['gradient_clip_val'] = gradient_clip_val
        if augment_poses is not None:
            transformer_config['augment_poses'] = augment_poses

        model = load_model(checkpoint, restore_weights=False, **transformer_config)
        tf.keras.Model.__setattr__(model, 'codebook_model', None)
        if batch_size is None:
            batch_size = model.config.batch_size
        model.codebook_model = codebook_model

        # Note, this restores the model and the optimizer
        # and the learning rate schedules should work correctly
        warmup_steps = 2000
        optimizer, lr_schedule = create_optimizer(
            learning_rate, num_train_steps=total_steps,
            num_warmup_steps=warmup_steps, weight_decay_rate=model.config.weight_decay)
        model.compile(optimizer=optimizer)
        model.load_weights(checkpoint).expect_partial()
        lr_schedule.offset.assign(model.optimizer.iterations)

        train_dataset, test_dataset = load_token_dataset(dataset,
                                                         batch_size=batch_size,
                                                         sequence_size=model.config.sequence_size,
                                                         token_image_size=model.config.token_image_size,
                                                         repeat=-1,
                                                         transform=partial(process_batch, augment=model.config.augment_poses))
        model.fit(train_dataset, batch_size=batch_size,
                  callbacks=[
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
