import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.warning_utils import WarningCache
from pytorch_lightning import Callback, Trainer
import pytorch_lightning as pl
from torchvision.utils import make_grid
import fsspec
import json

try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:  # pragma: no-cover
    wandb = None
    Run = None


class ExperimentTuple:
    def __init__(self, wandb_experiment: Run, tensorboard_experiment: SummaryWriter):
        self.wandb_experiment = wandb_experiment
        self.tensorboard_experiment = tensorboard_experiment

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return (self.wandb_experiment, self.tensorboard_experiment)[idx]

    def __getattr__(self, name):
        return getattr(self.tensorboard_experiment, name)

    def add_image(self, tag, img_tensor, global_step=None, *, label=None, **kwargs):
        self.tensorboard_experiment.add_image(tag, img_tensor, global_step=global_step, **kwargs)
        self.wandb_experiment.log({tag: [wandb.Image(img_tensor, caption=label or tag)]}, step=global_step)

    def add_images(self, tag, img_tensor, global_step=None, *, label=None, **kwargs):
        self.tensorboard_experiment.add_images(tag, img_tensor, global_step=global_step, **kwargs)
        self.wandb_experiment.log({tag: [wandb.Image(x, caption=f'{label or tag} {i}') for i, x in enumerate(torch.unbind(img_tensor))]},
                                  step=global_step)


class WandbLogger(LightningLoggerBase):
    LOGGER_JOIN_CHAR = '-'
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: bool = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: bool = False,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        experiment=None,
        prefix: str = '',
        **kwargs
    ):
        if wandb is None:
            raise ImportError('You want to use `wandb` logger which is not installed yet,'  # pragma: no-cover
                              ' install it with `pip install wandb`.')
        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._anonymous = 'allow' if anonymous else None
        self._id = version or id
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        self._kwargs = kwargs
        # logging multiple Trainer on a single W&B run (k-fold, resuming, etc)
        self._step_offset = 0
        self.hparams = {}
        self.warning_cache = WarningCache()

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state['_id'] = self._experiment.wandb_experiment.id if self._experiment is not None else None

        # cannot be pickled
        state['_experiment'] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            wandb_experiment = wandb.init(
                name=self._name, dir=self._save_dir, project=self._project, anonymous=self._anonymous,
                id=self._id, resume='allow', **self._kwargs) if wandb.run is None else wandb.run
            if getattr(wandb_experiment, "define_metric", None):
                wandb_experiment.define_metric("trainer/global_step")
                wandb_experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

            # offset logging step when resuming a run
            self._step_offset = wandb_experiment.step
            # save checkpoints in wandb dir to upload on W&B servers
            if self._log_model:
                self._save_dir = wandb_experiment.dir
            self._fs = get_filesystem(self.save_dir)

            tensorboard_experiment = SummaryWriter(log_dir=wandb_experiment.dir, **self._kwargs)
            self._experiment = ExperimentTuple(wandb_experiment, tensorboard_experiment)
        return self._experiment

    @rank_zero_only
    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        wandb_experiment, tensorboard_experiment = self.experiment
        wandb_experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        params = self._convert_params(params)
        wandb_experiment, tensorboard_experiment = self.experiment

        # store params to output
        self.hparams.update(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        # TensorBoard
        if metrics:
            metrics = self._add_prefix(metrics)
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                tensorboard_experiment.add_scalar(k, v, 0)
            exp, ssi, sei = hparams(params, metrics)
            writer = tensorboard_experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

        # Wandb
        wandb_experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        metrics = self._add_prefix(metrics)
        wandb_experiment, tensorboard_experiment = self.experiment

        # TensorBoard
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                tensorboard_experiment.add_scalars(k, v, step)
            else:
                try:
                    tensorboard_experiment.add_scalar(k, v, step)
                except Exception as e:
                    m = f'\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor.'
                    type(e)(e.message + m)

        # Wandb
        if step is not None:
            wandb_experiment.log({**metrics, "trainer/global_step": step})
        else:
            wandb_experiment.log(metrics)

    @rank_zero_only
    def log_graph(self, model: LightningModule, input_array=None):
        if self._log_graph:
            wandb_experiment, tensorboard_experiment = self.experiment
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                input_array = model.transfer_batch_to_device(input_array, model.device)
                tensorboard_experiment.add_graph(model, input_array)
            else:
                rank_zero_warn('Could not log computational graph since the'
                               ' `model.example_input_array` attribute is not set'
                               ' or `input_array` was not given',
                               UserWarning)

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._experiment.wandb_experiment.project_name() if self._experiment else self._name

    @property
    def version(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._experiment.wandb_experiment.id if self._experiment else self._id

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.flush()
        self.save()

        # offset future training logged on same W&B run
        if self._experiment is not None:
            self._step_offset = self._experiment.wandb_experiment.step

        # upload all checkpoints from saving dir
        if self._log_model:
            wandb.save(os.path.join(self.save_dir, "*.ckpt"))
        wandb.save(os.path.join(self.experiment.wandb_experiment.dir, self.NAME_HPARAMS_FILE))

    @rank_zero_only
    def save(self) -> None:
        # Initialize experiment
        _ = self.experiment

        super().save()

        # prepare the file path
        hparams_file = os.path.join(self.experiment.wandb_experiment.dir, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist
        if not os.path.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)


class LogImageCallback(Callback):
    def __init__(self, num_validation_images=32, grid_size=8):
        self.num_validation_images = num_validation_images
        self.grid_size = grid_size
        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer: Trainer, model, model_output, batch, batch_idx, *args, **kwargs):
        assert trainer.global_rank == 0

        if trainer.datamodule is not None and hasattr(trainer.datamodule, 'batch_size'):
            batch_size = trainer.datamodule.batch_size
        else:
            batch_size = trainer.val_dataloaders[0].batch_size
        generated_images = batch_idx * batch_size
        if generated_images >= self.num_validation_images:
            return

        logged_image = False
        experiment = trainer.logger.experiment

        if 'generated_image' in model_output:
            image = model_output['generated_image'][:(self.num_validation_images - generated_images)]
            if self.grid_size is None:
                experiment.add_images('test_generation', image, global_step=trainer.global_step)
            else:
                experiment.add_image('test_generation', make_grid(image, self.grid_size, padding=1), global_step=trainer.global_step)
            logged_image = True

        if 'reconstructed_image' in model_output:
            image = model_output['reconstructed_image'][:(self.num_validation_images - generated_images)]
            if self.grid_size is None:
                experiment.add_images('test_reconstruction', image, global_step=trainer.global_step)
            else:
                experiment.add_image('test_reconstruction', make_grid(image, self.grid_size, padding=1), global_step=trainer.global_step)
            logged_image = True

        if 'representation' in model_output:
            image = model_output['representation'][:(self.num_validation_images - generated_images)]
            if self.grid_size is None:
                experiment.add_images('test_representation', image, global_step=trainer.global_step)
            else:
                experiment.add_image('test_represenation', make_grid(image, self.grid_size, padding=1), global_step=trainer.global_step)
            logged_image = True

        if logged_image or 'ground_truth_image' in model_output:
            image = None
            if 'ground_truth_image' in model_output:
                image = model_output['ground_truth_image']
            else:
                image = batch
                if isinstance(image, dict):
                    image = batch['image'] if 'image' in batch else batch['query_image']
            image = image[:(self.num_validation_images - generated_images)]
            if self.grid_size is None:
                experiment.add_images('test_ground_truth', image, global_step=trainer.global_step)
            else:
                experiment.add_image('test_ground_truth', make_grid(image, self.grid_size, padding=1), global_step=trainer.global_step)


def EpochModelCheckpoint(*args, **kwargs):
    # NOTE: this is intentionally a function and not a class
    # It allows the type of the returned callback to be pl.callbacks.ModelCheckpoint
    # Which is serializable
    callback = pl.callbacks.ModelCheckpoint(*args, **kwargs)
    _on_validation_end = callback.on_validation_end
    _on_save_checkpoint = callback.on_save_checkpoint

    def on_validation_end(*args, **kwargs):
        return

    def on_save_checkpoint(trainer, module, *args):
        if hasattr(module, 'config'):
            with fsspec.open(os.path.join(callback.dirpath, 'config.json'), 'w+') as f:
                json.dump(module.config.asdict(), f)
                f.flush()

        return _on_save_checkpoint(trainer, module, *args)

    def on_train_epoch_end(trainer, module, _):
        _on_validation_end(trainer, module)

    callback.on_save_checkpoint = on_save_checkpoint
    callback.on_validation_end = on_validation_end
    callback.on_train_epoch_end = on_train_epoch_end
    return callback
