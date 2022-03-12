import pytorch_lightning as pl
from aparse import click, AllArguments, ConditionalType, WithArgumentName
from viewformer.data.tfrecord_dataset_th import ImageDataModule
from viewformer.models import AutoModelTH as AutoModel, supported_config_dict
from . import logging_utils_th as logging_utils


def transform_image(batch):
    return batch.mul_(2).add_(-1)


ModelSwitch = WithArgumentName(ConditionalType('ModelSwitch', supported_config_dict(), default='vqgan', prefix=None), 'model')
ImageDataModule = WithArgumentName(ImageDataModule, None)


@click.command('train-codebook', soft_defaults=True)
def main(
        model_config: ModelSwitch,
        args: AllArguments,
        datamodule: ImageDataModule,
        job_dir: str,
        total_steps: int,
        gradient_clip_val: float,
        epochs: int = 100,
        num_gpus: int = 8,
        num_nodes: int = 1,
        profile: bool = False,
        log_graph: bool = False,
        resume_from_checkpoint: str = None,
        accumulate_grad_batches: int = 1,
        fp16: bool = False,
        wandb: bool = False):
    datamodule.transform = transform_image

    if wandb:
        logger = logging_utils.WandbLogger(log_graph=log_graph)
    else:
        logger = pl.loggers.TensorBoardLogger(save_dir=job_dir, log_graph=log_graph)
    kwargs = dict(num_nodes=num_nodes)
    if num_gpus > 0:
        kwargs.update(dict(gpus=num_gpus, accelerator='ddp'))

    # Split training to #epochs epochs
    limit_train_batches = 1 + total_steps // epochs
    if profile:
        profiler = pl.profiler.AdvancedProfiler()
    else:
        profiler = pl.profiler.PassThroughProfiler()
    if fp16:
        kwargs['precision'] = 16

    # Save every 5 epochs
    validation_steps = max(1, min((total_steps // epochs) // 10, 100))
    model_checkpoint = logging_utils.EpochModelCheckpoint(
        mode='min',
        # monitor='val/total_loss',
        period=5,
        dirpath=job_dir,
        save_last=True)

    trainer = pl.Trainer(
        weights_save_path=job_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        # max_steps=total_steps,
        max_epochs=epochs,
        # val_check_interval=int(limit_train_batches // 5),
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=limit_train_batches,  # validation at the end of an epoch
        gradient_clip_val=gradient_clip_val,
        limit_val_batches=validation_steps,
        limit_train_batches=limit_train_batches,
        # track_grad_norm=2,
        logger=logger,
        profiler=profiler,
        callbacks=[logging_utils.LogImageCallback(), pl.callbacks.LearningRateMonitor('step'), model_checkpoint], **kwargs)

    if hasattr(trainer.logger.experiment, 'wandb_experiment') and \
            hasattr(trainer.logger.experiment.wandb_experiment, 'config'):
        trainer.logger.experiment.wandb_experiment.config.update(args, allow_val_change=True)

    model = AutoModel.from_config(model_config, use_mixed_precision=fp16)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
