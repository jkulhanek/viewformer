from typing import Optional
import tensorflow as tf
from functools import partial
from collections import OrderedDict
from viewformer.utils.tensorflow import shape_list
from viewformer.utils.geometry_tf import quaternion_normalize, quaternion_remove_sign
from viewformer.utils.metrics import CameraOrientationError, CameraPositionError, PSNRMetric
from .branching_attention import compute_causal_block_multiend_attention
from .config import MIGTConfig
from .utils import create_optimizer


Gelu = partial(tf.keras.layers.Activation, tf.nn.gelu)
layer_norm_epsilon = 1e-5


class SharedEmbeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=tf.keras.initializers.TruncatedNormal(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids):
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


class MLP(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, d_output=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        if d_output is None:
            d_output = d_model
        self.c_fc = Conv1D(d_inner, d_model, name="c_fc", dtype=dtype)
        self.c_proj = Conv1D(d_output, d_inner, name="c_proj", dtype=dtype)
        self.act = Gelu(dtype=dtype)
        self.dropout = tf.keras.layers.Dropout(dropout, dtype=dtype)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class Conv1D(tf.keras.layers.Layer):
    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bs = shape_list(x)[:-1]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, bs + [self.nf])
        return x


def sparse_softmax_cross_entropy_with_logits(y, y_hat, label_smoothing=0.0):
    label_smoothing = tf.cast(tf.convert_to_tensor(label_smoothing), y_hat.dtype)
    n_classes = tf.shape(y_hat)[-1]
    y = tf.one_hot(tf.cast(y, tf.int32), n_classes, dtype=y_hat.dtype)
    y = y * (1. - label_smoothing) + (label_smoothing / tf.cast(n_classes, y_hat.dtype))
    return tf.nn.softmax_cross_entropy_with_logits(y, y_hat)


class DynamicLossWeightingCriterion(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(dtype='float32', **kwargs)
        self.pos_ori_weights = self.add_weight(
            'pos_ori_weights',
            dtype='float32',
            shape=(2,),
            initializer=tf.constant_initializer([0., -3.]))

    def call(self, position_loss, orientation_loss):
        losses = tf.stack([position_loss, orientation_loss], -1)
        loss = tf.reduce_sum(self.pos_ori_weights + tf.exp(-self.pos_ori_weights) * losses)
        metrics = dict(zip(['dynamic_loss_weight_pos', 'dynamic_loss_weight_ori'], tf.unstack(self.pos_ori_weights)))
        return loss, metrics


def quaternion_reduce_mean(quat, axis=-2):
    quat = quaternion_normalize(quat)
    quat = quaternion_remove_sign(quat)
    quat = tf.reduce_mean(quat, axis)
    quat = quaternion_normalize(quat)
    quat = quaternion_remove_sign(quat)
    return quat


class QuaternionPoseRepresentation(tf.keras.layers.Layer):
    def __init__(self, d_model, position_multiplier: float = 1.0, norm: int = 1):
        super().__init__(dtype='float32')
        self.position_multiplier = position_multiplier
        self.pose_classifier = MLP(d_model, d_model * 2, 0.0, d_output=7, name='pose_classifier', dtype='float32')
        self.norm = norm

    def get_model_input(self, x, pose_multiplier=None):
        tf.debugging.assert_equal(tf.shape(x)[-1], 7)
        xyz, quaternion = tf.split(x, (3, 4), axis=-1)
        xyz = xyz * self.position_multiplier
        if pose_multiplier is not None:
            xyz = xyz * self.expand_pose_multiplier(pose_multiplier, xyz)
        return tf.concat([xyz, quaternion], axis=-1)

    def expand_pose_multiplier(self, pose_multiplier, expand_as):
        return tf.reshape(pose_multiplier, tf.concat((tf.shape(expand_as)[:1], tf.ones(tf.rank(expand_as) - 1, dtype=tf.int32)), 0))

    def reduce(self, x, axis=-2):
        xyz, quat = tf.split(x, (3, 4), axis=-1)
        xyz = tf.reduce_mean(xyz, axis)
        quat = quaternion_reduce_mean(quat, axis)
        return tf.concat((xyz, quat), -1)

    def call(self, internal_x, y=None, skip_first=None, pose_multiplier=None, training: bool = None):
        internal_x = self.pose_classifier(internal_x, training=training)
        tf.debugging.assert_type(internal_x, tf.float32)
        xyz, quaternion = tf.split(internal_x, (3, 4), axis=-1)
        if pose_multiplier is not None:
            xyz = xyz / self.expand_pose_multiplier(pose_multiplier, xyz)
        quaternion_normalized = quaternion_normalize(quaternion)
        quaternion_normalized = quaternion_remove_sign(quaternion_normalized)
        output_x = tf.concat([xyz / self.position_multiplier, quaternion_normalized], -1)
        if y is not None:
            tf.debugging.assert_type(y, tf.float32)
            y = y * tf.constant([self.position_multiplier] * 3 + [1] * 4, dtype='float32')
            # position_loss = tf.norm(y[..., :3] - xyz, axis=-1, ord=self.norm)
            # orientation_loss = tf.norm(y[..., 3:] - quaternion, axis=-1, ord=self.norm)
            position_loss = tf.losses.mse(y[..., :3], xyz)
            orientation_loss = tf.losses.mse(y[..., 3:], quaternion)
            if skip_first is not None:
                position_loss = position_loss[:, skip_first:]
                orientation_loss = orientation_loss[:, skip_first:]
            position_loss = tf.reduce_mean(position_loss, [1, 2])
            orientation_loss = tf.reduce_mean(orientation_loss, [1, 2])
            return output_x, position_loss, orientation_loss
        else:
            return output_x


class BranchingAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_positions, n_head, dropout, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_head == 0
        self.n_head = n_head
        self.split_size = d_model

        self.c_attn = Conv1D(d_model * 3, d_model, name="c_attn")
        self.c_proj = Conv1D(d_model, d_model, name="c_proj")
        self._attn = compute_causal_block_multiend_attention
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.resid_dropout = tf.keras.layers.Dropout(dropout)

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 3, 1, 2, 4))  # (batch, head, seq_length, head_features)

    def _get_key_value_query(self, x):
        x = list(map(self.c_attn, x))
        return tuple(zip(*map(lambda y: map(self.split_heads, tf.split(y, 3, axis=-1)), x)))

    def call(self, x, training=False):
        v, q, k = self._get_key_value_query(x)
        a = self._attn(k, v, q, attn_dropout=self.attn_dropout, training=training)
        a = map(self.merge_heads, a)
        a = map(self.c_proj, a)
        a = list(map(partial(self.resid_dropout, training=training), a))
        return a


class Block(tf.keras.layers.Layer):
    def __init__(self, n_positions, d_model, n_head, dropout, d_inner=None, **kwargs):
        super().__init__(**kwargs)
        if d_inner is None:
            d_inner = 4 * d_model
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="ln_1")
        self.attn = BranchingAttention(d_model, n_positions, n_head, dropout, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="ln_2")
        self.mlp = MLP(d_model, d_inner, dropout, name="mlp")

    def call(self, x, training=False):
        a = list(map(self.ln_1, x))
        a = self.attn(a, training=training)
        x = [xx + aa for xx, aa in zip(x, a)]

        m = list(map(self.ln_2, x))
        m = list(map(partial(self.mlp, training=training), m))
        x = [xx + mm for xx, mm in zip(x, m)]
        return x


class MIGT(tf.keras.Model):
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias"]

    def __init__(self, config: MIGTConfig, **kwargs):
        super().__init__(**kwargs, autocast=False)
        self.n_image_tokens = config.token_image_size ** 2
        self.n_positions = n_positions = self.n_image_tokens * config.sequence_size
        self.n_embeddings = config.n_embeddings
        self.token_image_size = config.token_image_size
        self.d_model = config.d_model
        self.weight_decay = config.weight_decay

        # self.wpe = SharedEmbeddings(n_positions, config.d_model, initializer_range=0.02, name='wpe')
        self.pos_embed_size = config.d_model
        self.total_steps = config.total_steps
        self.mask_token = config.n_embeddings
        self.localization_token = config.n_embeddings + 1
        self.config = config
        self._codebook_model = None
        metrics = [
            tf.metrics.Mean('loss', dtype=tf.float32),
            tf.metrics.Mean('ce_loss', dtype=tf.float32),
            tf.metrics.Mean('acc', dtype=tf.float32),
            tf.metrics.Mean('localization_weight', dtype=tf.float32),
            PSNRMetric('psnr', dtype=tf.float32),
        ]
        self.num_special_tokens = 2
        self.localization_weight = config.localization_weight.with_total_steps(config.total_steps)
        self.use_localization = not self.localization_weight.is_zero()
        if self.use_localization:
            metrics.append(tf.metrics.Mean('pose_loss', dtype=tf.float32))
            metrics.append(tf.metrics.Mean('pose_pos_loss', dtype=tf.float32))
            metrics.append(tf.metrics.Mean('pose_ori_loss', dtype=tf.float32))
            metrics.append(CameraPositionError('pose_pos_err', dtype=tf.float32, allow_nan=False))
            metrics.append(CameraOrientationError('pose_ori_err', dtype=tf.float32, allow_nan=False))

        self.pose_criterion = QuaternionPoseRepresentation(config.d_model, position_multiplier=self.config.pose_multiplier)

        if self.config.use_dynamic_pose_loss:
            self.pose_loss_weighting_criterion = DynamicLossWeightingCriterion()
            metrics.append(tf.metrics.Mean('dynamic_loss_weight_pos', dtype=tf.float32))
            metrics.append(tf.metrics.Mean('dynamic_loss_weight_ori', dtype=tf.float32))
        else:
            self.pose_loss_weighting_criterion = lambda position_loss, rotation_loss: ((position_loss + rotation_loss), dict())

        self._metrics = list(metrics)
        self.metrics_dict = OrderedDict([(x.name, x) for x in self.metrics])
        self.wte = SharedEmbeddings(config.n_embeddings + self.num_special_tokens, self.pos_embed_size, initializer_range=0.02, name='wte', dtype='float32')
        self.drop = tf.keras.layers.Dropout(config.dropout)
        self.h = [Block(n_positions, config.d_model, config.n_head, config.dropout, name=f'h.{i}') for i in range(config.n_layer)]
        self.pose_embedding = MLP(7, config.d_model * 2, 0.0, d_output=self.pos_embed_size, name='pose_embedding', dtype='float32')
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="ln_f")

        # Initialize buffers and parameters
        init_input = dict(poses=tf.random.normal([1, config.sequence_size, 7], dtype=tf.float32),
                          input_ids=tf.random.uniform([1, config.sequence_size, config.token_image_size, config.token_image_size], maxval=4, dtype=tf.int64))
        self(init_input, training=False, compute_losses=True)

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.extend(self._metrics)
        return metrics

    def build(self, input_shape):
        with tf.name_scope("wpe"):
            # NOTE: the shape should be dynamic and the first dimension should be the following:
            #   self.config.token_image_size ** 2
            #   Current checkpoints, however, used static size of 256. Feel free to update it if
            #   you do not need to load existing checkpoints.
            self.wpe = self.add_weight(
                name="embeddings",
                shape=[256, self.pos_embed_size],
                initializer=tf.keras.initializers.TruncatedNormal(0.02),
            )

    @property
    def codebook_model(self):
        return object.__getattribute__(self, '_codebook_model')

    @codebook_model.setter
    def codebook_model(self, model):
        object.__setattr__(self, '_codebook_model', model)

    def _compute_accuracy(self, labels, logits):
        labels = labels[:, self.config.n_loss_skip:]
        pred = tf.argmax(logits, -1, output_type=labels.dtype)
        pred = pred[:, self.config.n_loss_skip:]
        tf.debugging.assert_equal(tf.shape(pred), tf.shape(labels))
        return tf.cast(pred == labels, tf.float32)

    def _merge_input_embeddings(self, *embeddings):
        return sum(embeddings)

    def _output_embedding(self, features):
        return self.wte(features, mode="linear")

    def call(self, inputs, compute_losses=False, training=False, **kwargs):
        poses = inputs['poses']
        input_ids = inputs['input_ids']
        original_input_shape = shape_list(input_ids)
        input_ids = tf.reshape(input_ids, shape_list(input_ids)[:2] + [-1])
        input_shape = shape_list(input_ids)
        localization_tokens = inputs.get('localization_tokens', None)
        output_poses = inputs.get('output_poses', None)
        tf.debugging.assert_type(poses, tf.float32)

        # input_ids: (batch, frame, image_position)
        # poses: (batch, frame, 7)
        if training:
            random_pose_multiplier = tf.constant(self.config.random_pose_multiplier, 'float32') ** tf.random.uniform((tf.shape(poses)[0],), -1, 1, dtype='float32')
        else:
            random_pose_multiplier = tf.ones((tf.shape(poses)[0]), 'float32')
        pose_embeddings = self.pose_embedding(self.pose_criterion.get_model_input(poses, pose_multiplier=random_pose_multiplier), training=training)
        pose_embeddings = tf.expand_dims(pose_embeddings, -2)
        tf.debugging.assert_type(pose_embeddings, tf.float32)

        position_ids = tf.range(0, self.config.token_image_size ** 2)[tf.newaxis, tf.newaxis, :]
        position_embeds = tf.gather(self.wpe, position_ids)

        inputs_embeds = self.wte(input_ids, mode="embedding")
        pose_embeddings = tf.cast(pose_embeddings, inputs_embeds.dtype)
        localization_embeds = None
        output_pose_embeddings = None
        gen_images_pointer, gen_poses_pointer = 0, 0

        position_embeds = tf.cast(position_embeds, dtype=inputs_embeds.dtype)
        pose_embeddings = tf.cast(pose_embeddings, dtype=inputs_embeds.dtype)
        loc_seq_size = tf.shape(inputs_embeds)[1] - tf.shape(pose_embeddings)[1]

        if compute_losses:
            if localization_tokens is None and self.use_localization:
                localization_tokens = input_ids
                localization_embeds = inputs_embeds
            if output_poses is None:
                output_poses = poses
                output_pose_embeddings = pose_embeddings
        if localization_tokens is not None and localization_embeds is None:
            localization_tokens = tf.reshape(localization_tokens, shape_list(localization_tokens)[:2] + [-1])
            localization_embeds = self.wte(localization_tokens, mode='embedding')
            localization_embeds = tf.cast(localization_embeds, dtype=inputs_embeds.dtype)
        if output_poses is not None and output_pose_embeddings is None:
            output_pose_embeddings = self.pose_embedding(self.pose_criterion.get_model_input(output_poses, pose_multiplier=random_pose_multiplier), training=training)
            output_pose_embeddings = tf.expand_dims(output_pose_embeddings, -2)
            output_pose_embeddings = tf.cast(output_pose_embeddings, dtype=inputs_embeds.dtype)

        if self.use_localization and (not compute_losses):
            localization_pose_embeds = tf.reshape(self.wte(tf.constant(self.localization_token, dtype=input_ids.dtype), mode='embedding'), [1, 1, 1, -1])
            localization_pose_embeds = tf.broadcast_to(localization_pose_embeds, [tf.shape(inputs_embeds)[0], loc_seq_size, tf.shape(pose_embeddings)[-2], tf.shape(localization_pose_embeds)[-1]])
            pose_embeddings = tf.concat([pose_embeddings, localization_pose_embeds], 1)

        hidden_states = [self._merge_input_embeddings(inputs_embeds, position_embeds, pose_embeddings)]
        if output_pose_embeddings is not None:
            mask_embeds = tf.reshape(self.wte(tf.constant(self.mask_token, dtype=input_ids.dtype), mode='embedding'), [1, 1, 1, -1])
            hidden_states.append(self._merge_input_embeddings(mask_embeds, position_embeds, output_pose_embeddings))
            gen_images_pointer = len(hidden_states) - 1

        if localization_embeds is not None:
            localization_token_embeds = tf.reshape(self.wte(tf.constant(self.localization_token, dtype=input_ids.dtype), mode='embedding'), [1, 1, 1, -1])
            hidden_states.append(self._merge_input_embeddings(localization_embeds, position_embeds, localization_token_embeds))
            gen_poses_pointer = len(hidden_states) - 1

        hidden_states = list(map(partial(self.drop, training=training), hidden_states))
        output_shape = input_shape + [shape_list(hidden_states[0])[-1]]
        for block in self.h:
            hidden_states = block(hidden_states, training=training)

        hidden_states = map(self.ln_f, hidden_states)
        hidden_states = [tf.reshape(x, output_shape) for x in hidden_states]

        # Add output embedding
        model_output = dict(
            hidden_states=hidden_states,
        )

        loss = 0
        lm_logits = self._output_embedding(hidden_states[gen_images_pointer])[..., :self.config.n_embeddings]
        if compute_losses:
            # The cross-entropy loss should be computed on float32 in mixed precision training
            if self.config.label_smoothing > 0:
                ce_loss = sparse_softmax_cross_entropy_with_logits(input_ids, lm_logits, label_smoothing=self.config.label_smoothing)
            else:
                ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(input_ids, lm_logits)
            ce_loss = ce_loss[:, self.config.n_loss_skip:]
            tf.debugging.assert_type(ce_loss, tf.float32)
            ce_loss = tf.reduce_mean(ce_loss, [1, 2])
            model_output['ce_loss'] = ce_loss
            loss += ce_loss * self.config.image_generation_weight

        if self.use_localization:
            poses_input = hidden_states[gen_poses_pointer]
            if compute_losses:
                gt_poses = tf.expand_dims(poses, -2)
                poses_out, pose_pos_loss, pose_ori_loss = self.pose_criterion(poses_input, gt_poses,
                                                                              skip_first=self.config.n_loss_skip,
                                                                              pose_multiplier=random_pose_multiplier,
                                                                              training=training)
                tf.debugging.assert_type(pose_pos_loss, tf.float32)
                tf.debugging.assert_type(pose_ori_loss, tf.float32)
                pose_loss, wc_metrics = self.pose_loss_weighting_criterion(pose_pos_loss, pose_ori_loss)
                for k, v in wc_metrics.items():
                    model_output[k] = v
                model_output['pose_loss'] = pose_loss
                model_output['pose_pos_loss'] = pose_pos_loss
                model_output['pose_ori_loss'] = pose_ori_loss
                localization_weight = self.localization_weight(self._train_counter, 'float32')
                loss += pose_loss * localization_weight
                model_output['localization_weight'] = localization_weight
            else:
                poses_out = self.pose_criterion(poses_input, training=training, pose_multiplier=random_pose_multiplier)
            model_output['pose_prediction'] = poses_out

        model_output['logits'] = tf.reshape(lm_logits, original_input_shape + [-1])
        model_output['loss'] = loss
        return model_output

    def compile(self, optimizer=None, **kwargs):
        if optimizer is None:
            warmup_steps = 2000
            optimizer, lr_schedule = create_optimizer(self.config.learning_rate, num_train_steps=self.config.total_steps,
                                                      num_warmup_steps=warmup_steps, weight_decay_rate=self.config.weight_decay)
        super().compile(optimizer=optimizer, **kwargs)

    def train_step(self, batch):
        poses, tokens = batch
        use_fp16 = hasattr(self.optimizer, 'get_scaled_loss')
        with tf.GradientTape() as tape:
            output_dict = self(dict(poses=poses, input_ids=tokens), compute_losses=True, training=True)
            loss = output_dict['loss']

            # Note: This should use tf.nn.compute_average_loss.
            #       The original code, however, used reduce_mean.
            #       On more devices, the learning rate should be scaled
            #       accordingly if compute_average_loss is used.
            # loss = tf.nn.compute_average_loss(loss, global_batch_size=self.config.batch_size)
            loss = tf.reduce_mean(loss)

            if use_fp16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

        grads = tape.gradient(scaled_loss, self.trainable_variables)
        if use_fp16:
            grads = self.optimizer.get_unscaled_gradients(grads)
        if self.config.gradient_clip_val is not None and self.config.gradient_clip_val > 0.:
            grads = [tf.clip_by_norm(g, self.config.gradient_clip_val) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for k, v in output_dict.items():
            if k in self.metrics_dict:
                self.metrics_dict[k].update_state(v)

        # Compute and update acc
        self.metrics_dict['acc'].update_state(self._compute_accuracy(tokens, output_dict['logits']))
        if 'pose_prediction' in output_dict:
            self.metrics_dict['pose_pos_err'].update_state(
                output_dict['pose_prediction'][:, self.config.n_loss_skip:],
                tf.expand_dims(poses, -2)[:, self.config.n_loss_skip:])
            self.metrics_dict['pose_ori_err'].update_state(
                output_dict['pose_prediction'][:, self.config.n_loss_skip:],
                tf.expand_dims(poses, -2)[:, self.config.n_loss_skip:])

        metrics = {m.name: m.result() for m in self.metrics}
        del metrics['psnr']
        return metrics

    def test_step(self, batch):
        poses, tokens = batch
        output_dict = self(dict(poses=poses, input_ids=tokens), compute_losses=True, training=False)
        for k, v in output_dict.items():
            if k in self.metrics_dict:
                self.metrics_dict[k].update_state(v)

        # Compute and update acc
        self.metrics_dict['acc'].update_state(self._compute_accuracy(tokens, output_dict['logits']))
        if 'pose_prediction' in output_dict:
            self.metrics_dict['pose_pos_err'].update_state(
                output_dict['pose_prediction'][:, self.config.n_loss_skip:],
                tf.expand_dims(poses, -2)[:, self.config.n_loss_skip:])
            self.metrics_dict['pose_ori_err'].update_state(
                output_dict['pose_prediction'][:, self.config.n_loss_skip:],
                tf.expand_dims(poses, -2)[:, self.config.n_loss_skip:])

        # Predict images
        gen_images = tf.clip_by_value(self._codebook_model.decode_code(tf.argmax(output_dict['logits'][:, -1], -1)) / 2 + 0.5, 0, 1)
        gt_images = tf.clip_by_value(self._codebook_model.decode_code(tokens[:, -1]) / 2 + 0.5, 0, 1)

        # Compute PSNR
        self.metrics_dict['psnr'].update_state(gt_images, gen_images)
        return {m.name: m.result() for m in self.metrics}

    def reduce_cameras(self, cameras, axis=-2):
        return self.pose_criterion.reduce(cameras, axis=axis)

    def predict_step(self, batch):
        poses, tokens = batch
        lm_logits = self(dict(poses=poses, input_ids=tokens), compute_losses=True, training=False)['logits']
        generated_tokens = tf.reshape(tf.argmax(lm_logits, -1), [-1, self.config.sequence_size, self.token_image_size, self.token_image_size])
        generated_tokens = tf.where(generated_tokens < self.n_embeddings, generated_tokens, tf.zeros_like(generated_tokens))
        decoded_image = self._codebook_model.decode_code(tf.reshape(generated_tokens, [-1, self.token_image_size, self.token_image_size]), training=False)
        ground_truth_image = self._codebook_model.decode_code(tf.reshape(tokens, [-1, self.token_image_size, self.token_image_size]), training=False)
        return {'decoded_image': decoded_image, 'latent_code': generated_tokens, 'ground_truth_image': ground_truth_image}
