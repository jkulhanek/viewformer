import re
import os
import tensorflow as tf
import math
from typing import Callable, Optional, List, Union
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers


def angle_squared_difference(y_true, y_pred):
    pi = tf.constant(math.pi, dtype=y_true.dtype)
    diff = (y_true - y_pred + pi) % (2 * pi) - pi
    return diff ** 2


def angle_absolute_difference(y_true, y_pred):
    pi = tf.constant(math.pi, dtype=y_true.dtype)
    diff = (y_true - y_pred + pi) % (2 * pi) - pi
    return tf.abs(diff)


def pose_criterion(y_true, y_pred):
    poses_loss = tf.concat([tf.math.squared_difference(y_true[..., :3], y_pred[..., :3]),
                            angle_squared_difference(y_true[..., 3:], y_pred[..., 3:])], -1)
    return poses_loss


def pose_absolute_difference(y_true, y_pred):
    poses_loss = tf.concat([tf.abs(y_true[..., :3] - y_pred[..., :3]),
                            angle_absolute_difference(y_true[..., 3:], y_pred[..., 3:])], -1)
    return poses_loss


def reduce_angle_mean(angles, axis=-1):
    y = tf.reduce_mean(tf.sin(angles), axis)
    x = tf.reduce_mean(tf.cos(angles), axis)
    return tf.atan2(y, x)


class QuantizeEMA(layers.Layer):
    """Sonnet module representing the VQ-VAE layer.
    Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    The difference between VectorQuantizerEMA and VectorQuantizer is that
    this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. For
    most experiments the EMA version trains faster than the non-EMA version.
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Attributes:
      embedding_dim: integer representing the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
      num_embeddings: integer, the number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms (see
        equation 4 in the paper).
      decay: float, decay for the moving averages.
      epsilon: small float constant to avoid numerical instability.
    """

    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 decay=0.99,
                 epsilon=1e-5,
                 dtype=tf.float32,
                 name='vector_quantizer_ema'):
        """Initializes a VQ-VAE EMA module.
        Args:
          embedding_dim: integer representing the dimensionality of the tensors in
            the quantized space. Inputs to the modules must be in this format as
            well.
          num_embeddings: integer, the number of vectors in the quantized space.
          commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
          decay: float between 0 and 1, controls the speed of the Exponential Moving
            Averages.
          epsilon: small constant to aid numerical stability, default 1e-5.
          dtype: dtype for the embeddings variable, defaults to tf.float32.
          name: name of the module.
        """
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        if not 0 <= decay <= 1:
            raise ValueError('decay must be in range [0, 1]')
        self.decay = decay
        self.epsilon = epsilon

        embedding_shape = [embedding_dim, num_embeddings]
        self.embeddings = tf.Variable(tf.random.uniform(shape=embedding_shape, minval=-tf.sqrt(3.0), maxval=tf.sqrt(3.0), dtype=dtype), trainable=False, name=f'{self.name}/embeddings')

        self.ema_cluster_size_hidden = tf.Variable(tf.zeros([num_embeddings], dtype=dtype), trainable=False, name=f"{self.name}/ema_cluster_size_hidden")
        self.ema_dw_hidden = tf.Variable(tf.zeros_like(self.embeddings), trainable=False, name=f"{self.name}/ema_dw_hidden")
        self._counter = tf.Variable(
            0, trainable=False, dtype=tf.int64, name=f"{self.name}/counter")

    @property
    def ema_dw(self):
        value = self.ema_dw_hidden
        counter = tf.cast(self._counter, value.dtype)
        return value / (1. - tf.pow(self.decay, counter))

    @property
    def ema_cluster_size(self):
        value = self.ema_cluster_size_hidden
        counter = tf.cast(self._counter, value.dtype)
        return value / (1. - tf.pow(self.decay, counter))

    def call(self, inputs, training=False):
        """Connects the module to some inputs.
        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be flattened and treated as a large batch.
          is_training: boolean, whether this connection is to training data. When
            this is set to False, the internal moving average statistics will not be
            updated.
        """
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, self.embeddings) +
            tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype)

        # NB: if your code crashes with a reshape error on the line below about a
        # Tensor containing the wrong number of values, then the most likely cause
        # is that the input passed in does not have a final dimension equal to
        # self.embedding_dim. Ideally we would catch this with an Assert but that
        # creates various other problems related to device placement / TPUs.
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.embed_code(encoding_indices)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)

        if training:
            embed_onehot_sum = tf.reduce_sum(encodings, axis=0)
            embed_sum = tf.matmul(flat_inputs, encodings, transpose_a=True)

            ctx = tf.distribute.get_replica_context()
            if ctx is not None:
                ctx.all_reduce(tf.distribute.ReduceOp.SUM, embed_onehot_sum)
                ctx.all_reduce(tf.distribute.ReduceOp.SUM, embed_sum)

            # Update EMA
            self.ema_cluster_size_hidden.assign_add((embed_onehot_sum - self.ema_cluster_size_hidden) * (1 - self.decay))
            self.ema_dw_hidden.assign_add((embed_sum - self.ema_dw_hidden) * (1 - self.decay))
            self._counter.assign_add(1)
            updated_ema_cluster_size = self.ema_cluster_size
            updated_ema_dw = self.ema_dw

            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)

            normalised_updated_ema_w = updated_ema_dw / updated_ema_cluster_size
            self.embeddings.assign(normalised_updated_ema_w)

        # Straight Through Estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized, e_latent_loss, encoding_indices

    def embed_code(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(self.embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)


class Quantize(layers.Layer):
    """Sonnet module representing the VQ-VAE layer.
    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper - this variable is Beta).
    """

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 beta: float = 0.25,
                 dtype: tf.DType = tf.float32,
                 name: str = 'vector_quantizer'):
        """Initializes a VQ-VAE module.
        Args:
          embedding_dim: dimensionality of the tensors in the quantized space.
            Inputs to the modules must be in this format as well.
          num_embeddings: number of vectors in the quantized space.
          commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
          dtype: dtype for the embeddings variable, defaults to tf.float32.
          name: name of the module.
        """
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        embedding_shape = [embedding_dim, num_embeddings]
        rand_range = 1.0 / self.num_embeddings  # math.sqrt(3)
        self.embeddings = tf.Variable(tf.random.uniform(shape=embedding_shape, minval=-rand_range, maxval=rand_range, dtype=dtype), trainable=True, name=f'{self.name}/embeddings')

    def __call__(self, inputs, training=False):
        """Connects the module to some inputs.
        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be flattened and treated as a large batch.
          training: boolean, whether this connection is to training data.
        Returns:
          dict containing the following keys and values:
            quantize: Tensor containing the quantized version of the input.
            loss: Tensor containing the loss to optimize.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
            of the quantized space each input element was mapped to.
            encoding_indices: Tensor containing the discrete encoding indices, ie
            which element of the quantized space each input element was mapped to.
        """
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (
            tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
            2 * tf.matmul(flat_inputs, self.embeddings) +
            tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.embed_code(encoding_indices)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs))**2)
        loss = e_latent_loss + q_latent_loss * self.beta
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized, loss, encoding_indices

    def embed_code(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = tf.transpose(self.embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)


_LPIPS_URL = 'http://rail.eecs.berkeley.edu/models/lpips'


def load_lpips_model(net='alex', version='0.1', overwrite=False):
    path = os.path.join(os.path.expanduser('~/.lpips'), f'{net}-{version}')
    if not os.path.exists(f'{path}.pb') or overwrite:
        import torch
        from lpips import LPIPS
        import onnx
        import onnx_tf

        os.makedirs(os.path.dirname(path), exist_ok=True)
        lpips_th = LPIPS(net=net, version=version)
        sample_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        torch.onnx.export(lpips_th,
                          (sample_input, sample_input),
                          f'{path}.onnx',
                          dynamic_axes={
                              '0': {
                                  0: 'batch',
                                  2: 'height',
                                  3: 'width'},
                              '1': {
                                  0: 'batch',
                                  2: 'height',
                                  3: 'width'},
                              'output': {0: 'batch'}
                          })
        model = onnx.load(f'{path}.onnx')
        onnx_tf.backend.prepare(model).export_graph(f'{path}.pb')
    return tf.saved_model.load(f'{path}.pb')


def lpips(net='vgg'):
    model = load_lpips_model(net=net)

    def call(image_0, image_1):
        image_0 = tf.transpose(image_0, [0, 3, 1, 2])
        image_1 = tf.transpose(image_1, [0, 3, 1, 2])
        output = model(**{'0': image_0, '1': image_1})
        if isinstance(output, dict):
            output = next(iter(output.values()))
        return tf.reshape(output, [-1])
    return call


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


def create_optimizer(
    init_lr: float,
    num_train_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    weight_decay_rate: float = 0.0,
    include_in_weight_decay: Optional[List[str]] = None,
    use_cosine: bool = True,
):
    """
    Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.
    Args:
        init_lr (:obj:`float`):
            The desired learning rate at the end of the warmup phase.
        num_train_steps (:obj:`int`):
            The total number of training steps.
        num_warmup_steps (:obj:`int`):
            The number of warmup steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0):
            The final learning rate at the end of the linear decay will be :obj:`init_lr * min_lr_ratio`.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 to use in Adam.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 to use in Adam.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon to use in Adam.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to use.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters except bias and layer norm parameters.
    """
    # Implements linear decay of the learning rate.
    if use_cosine:
        lr_schedule = tf.keras.experimental.CosineDecay(init_lr, num_train_steps - num_warmup_steps)
    else:
        lr_schedule = init_lr
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )
    if weight_decay_rate > 0.0:
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=weight_decay_rate,
            beta_1=adam_beta1,
            beta_2=adam_beta2,
            epsilon=adam_epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            include_in_weight_decay=include_in_weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon
        )

    if mixed_precision.global_policy().compute_dtype != 'float32':
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # We return the optimizer and the LR scheduler in order to better track the
    # evolution of the LR independently of the optimizer.
    return optimizer, lr_schedule


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.
    Instead we want ot decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.
    Args:
        learning_rate (:obj:`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`, defaults to 1e-3):
            The learning rate to use or a schedule.
        beta_1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (:obj:`float`, `optional`, defaults to 1e-7):
            The epsilon parameter in Adam, which is a small constant for numerical stability.
        amsgrad (:obj:`bool`, `optional`, default to `False`):
            Whether to apply AMSGrad variant of this algorithm or not, see `On the Convergence of Adam and Beyond
            <https://arxiv.org/abs/1904.09237>`__.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in :obj:`exclude_from_weight_decay`).
        exclude_from_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            :obj:`include_in_weight_decay` is passed, the names in it will supersede this list.
        name (:obj:`str`, `optional`, defaults to 'AdamWeightDecay'):
            Optional name for the operations created when applying gradients.
        kwargs:
            Keyward arguments. Allowed to be {``clipnorm``, ``clipvalue``, ``lr``, ``decay``}. ``clipnorm`` is clip
            gradients by norm; ``clipvalue`` is clip gradients by value, ``decay`` is included for backward
            compatibility to allow time inverse decay of learning rate. ``lr`` is included for backward compatibility,
            recommended to use ``learning_rate`` instead.
    """

    def __init__(
        self,
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        weight_decay_rate: float = 0.0,
        include_in_weight_decay: Optional[List[str]] = None,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "AdamWeightDecay",
        **kwargs
    ):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"WarmUp": WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],
                use_locking=self._use_locking,
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True
