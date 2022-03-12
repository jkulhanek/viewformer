import tensorflow as tf
from typing import List, Callable
from tensorflow.python.ops import summary_ops_v2
from tensorboard.plugins.hparams import api as hp
import json
try:
    from tensorflow.python.keras.distribute import distributed_file_utils
except ImportError:
    # Old tf
    from tensorflow.python.distribute import distributed_file_utils


def shape_list(tensor: tf.Tensor) -> List[int]:
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def imgrid(imarray, cols=8, pad=1, row_major=True):
    """Lays out a [N, H, W, C] image array as a single image grid."""
    pad = int(pad)
    if pad < 0:
        raise ValueError('pad must be non-negative')
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = N // cols + int(N % cols != 0)
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = tf.pad(imarray, pad_arg, constant_values=0)
    H += pad
    W += pad
    grid = tf.reshape(imarray, [rows, cols, H, W, C])
    grid = tf.transpose(grid, [0, 2, 1, 3, 4])
    grid = tf.reshape(grid, [1, rows*H, cols*W, C])
    if pad:
        grid = grid[:, :-pad, :-pad]
    return grid


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, path):
        self._original_path = path
        self._path = None
        self._last_val_loss = None
        self._last_checkpoint = None

    @property
    def model_path(self):
        if self._path is None:
            distribute_strategy = tf.distribute.get_strategy()
            self._path = distributed_file_utils.write_dirpath(self._original_path, distribute_strategy)
        return self._path

    def set_model(self, model):
        self.model = model

        # Save config
        strategy = tf.distribute.get_strategy()
        if strategy.extended.should_checkpoint:
            if hasattr(self.model, 'config'):
                with tf.io.gfile.GFile(self._original_path + '/config.json', 'w+') as f:
                    json.dump(self.model.config.asdict(), f)
                    f.flush()

    def on_epoch_end(self, epoch, logs=None):
        if 'val_loss' in logs:
            val_loss = logs['val_loss']
            if self._last_val_loss is None or self._last_val_loss > val_loss:
                # Overwrite checkpoint
                if self._last_checkpoint is not None:
                    for fname in tf.io.gfile.glob(self.model_path + self._last_checkpoint + '*'):
                        tf.io.gfile.rmtree(fname)
                checkpoint_path = self._get_checkpoint_path(f'model.{epoch:03d}-{val_loss:.4f}')
                self.model.save_weights(self.model_path + checkpoint_path)
                self._last_val_loss = val_loss
                self._last_checkpoint = checkpoint_path

        for fname in tf.io.gfile.glob(self.model_path + self._get_checkpoint_path(f'model.{(epoch - 1):03d}-last') + '*'):
            tf.io.gfile.rmtree(fname)
        self.model.save_weights(self.model_path + self._get_checkpoint_path(f'model.{epoch:03d}-last'))

    def _get_checkpoint_path(self, name):
        return f'/weights.{name}'

    def on_train_end(self, logs=None):
        # last_save_checkpoint = self._get_checkpoint_path('model.last')
        # self.model.save_weights(self.model_path + last_save_checkpoint)
        pass


def serialize_hparams(hparams):
    if not isinstance(hparams, dict):
        hparams = vars(hparams)

    def fix_value(value):
        if isinstance(value, list):
            value = ','.join(map(str, value))
        return value
    return {k: fix_value(v) for k, v in hparams.items() if not k.startswith('wandb_') and fix_value(v) is not None}


class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, hparams, tensorboard_callback: tf.keras.callbacks.TensorBoard, validation_images: int = 32):
        self.test_dataset = test_dataset
        self.tensorboard_callback = tensorboard_callback
        self.validation_images = validation_images
        self.distribute_strategy = tf.distribute.get_strategy()
        self.hparams = serialize_hparams(hparams)
        self.predict_step = None

    def on_train_begin(self, *args, **kwargs):
        with summary_ops_v2.always_record_summaries():
            with self.tensorboard_callback._train_writer.as_default():
                hp.hparams(self.hparams)

    def make_predict_step(self):
        if self.predict_step is None:
            self.predict_step = tf.function(self.model.predict_step)
        return self.predict_step

    def on_epoch_end(self, epoch, logs=None):
        test_dataset_iterator = iter(self.test_dataset)
        data = test_dataset_iterator.get_next()
        pred = self.distribute_strategy.run(self.make_predict_step(), args=(data,))
        decoded_image = self.distribute_strategy.experimental_local_results(pred['decoded_image'])[0]
        if 'ground_truth_image' in pred:
            ground_truth_image = self.distribute_strategy.experimental_local_results(pred['ground_truth_image'])[0]
        else:
            ground_truth_image = self.distribute_strategy.experimental_local_results(data)[0]

        with summary_ops_v2.always_record_summaries():
            with self.tensorboard_callback._val_writer.as_default():
                summary_ops_v2.image('reconstructed_image', imgrid(
                    tf.clip_by_value((decoded_image[:self.validation_images, ...] + 1.) / 2., 0, 1)), step=epoch)
                summary_ops_v2.image('ground_truth_image', imgrid(
                    tf.clip_by_value((ground_truth_image[:self.validation_images, ...] + 1.) / 2., 0, 1)), step=epoch)


def get_strategy(ddp, tpu):
    distributed_strategy = tf.distribute.get_strategy()  # no-op
    if ddp:
        distributed_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    elif tpu:
        distributed_strategy = tf.distribute.TPUStrategy()
    else:
        distributed_strategy = tf.distribute.MirroredStrategy()
    return distributed_strategy


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.
    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
        offset: int = None
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name
        self.offset = tf.Variable(offset or 0, dtype=tf.int64)

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            step = tf.maximum(step - tf.cast(self.offset, step.dtype), 0)
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
            "offset": self.offset,
        }
