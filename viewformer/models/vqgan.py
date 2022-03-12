import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from itertools import chain
from collections import OrderedDict
from viewformer.utils.tensorflow import shape_list
from viewformer.models import VQGANConfig
from .utils import QuantizeEMA, lpips


batchnorm_params = dict(epsilon=1e-5, momentum=0.9)


def _sequential_name(base_name):
    i = 0

    def name():
        nonlocal i
        i += 1
        return f'{base_name}.{i - 1}'
    return name


def nonlinearity(x):
    # swish
    return x*tf.sigmoid(x)


def Normalize(name=None):
    return tfa.layers.GroupNormalization(groups=32, epsilon=1e-6, center=True, scale=True, name=name)


class Upsample(layers.Layer):
    def __init__(self, in_channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.upsample = layers.UpSampling2D(interpolation='nearest')
        self.conv = tf.keras.Sequential([
            layers.ZeroPadding2D(),
            layers.Conv2D(in_channels, 3, name=f'{self.name}/conv')
        ])

    def call(self, x, training=False):
        x = self.upsample(x, training=training)
        x = self.conv(x, training=training)
        return x


class Downsample(layers.Layer):
    def __init__(self, in_channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv = tf.keras.Sequential([
            layers.ZeroPadding2D(((0, 1), (0, 1))),
            layers.Conv2D(in_channels, 3, strides=2, name=f'{self.name}/conv')
        ])

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        return x


class ResnetBlock(layers.Layer):
    def __init__(self, *, in_channels, out_channels=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        trunk = [
            Normalize(name=f'{self.name}/norm1'),
            layers.Activation(nonlinearity),
            layers.ZeroPadding2D(),
            layers.Conv2D(out_channels, 3, name=f'{self.name}/conv1'),
            Normalize(name=f'{self.name}/norm2'),
            layers.Activation(nonlinearity),
            layers.ZeroPadding2D(),
            layers.Conv2D(out_channels, 3, name=f'{self.name}/conv2')]

        if self.in_channels != self.out_channels:
            self.nin_shortcut = layers.Conv2D(out_channels, 1, name='nin_shortcut')
        self.trunk = tf.keras.Sequential(trunk)

    def call(self, x, training=False):
        h = self.trunk(x, training=training)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x, training=training)

        return x+h


class AttnBlock(layers.Layer):
    def __init__(self, in_channels, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_channels = in_channels

        self.norm = Normalize(name='norm')
        self.q = layers.Conv2D(in_channels, 1, name='q')
        self.k = layers.Conv2D(in_channels, 1, name='k')
        self.v = layers.Conv2D(in_channels, 1, name='v')
        self.proj_out = layers.Conv2D(in_channels, 1, name='proj_out')

    def call(self, x, training=False):
        h_ = x
        h_ = self.norm(h_, training=training)
        q = self.q(h_, training=training)
        k = self.k(h_, training=training)
        v = self.v(h_, training=training)

        # compute attention
        b, h, w, c = shape_list(q)
        q = tf.reshape(q, [b, h*w, c])
        k = tf.reshape(k, [b, h*w, c])
        w_ = tf.matmul(q, k, transpose_b=True)
        w_ = w_ * (int(c)**(-0.5))
        w_ = tf.nn.softmax(w_, axis=2)

        # attend to values
        v = tf.reshape(v, [b, h*w, c])
        h_ = tf.matmul(w_, v)
        h_ = tf.reshape(h_, [b, h, w, c])

        h_ = self.proj_out(h_, training=training)
        return x+h_


class Encoder(layers.Layer):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, in_channels,
                 image_size, z_channels, name=None, **kwargs):
        super().__init__(name=name)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = image_size
        self.in_channels = in_channels

        # downsampling
        self.down = tf.keras.Sequential([
            layers.ZeroPadding2D(),
            layers.Conv2D(self.ch, 3, name=f'{self.name}/conv_in')
        ])
        curr_res = image_size
        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_resolutions):
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                self.down.add(ResnetBlock(in_channels=block_in,
                                          out_channels=block_out,
                                          name=f'{self.name}/down.{i_level}/block.{i_block}'))
                block_in = block_out
                if curr_res in attn_resolutions:
                    self.down.add(AttnBlock(block_in, name=f'{self.name}/down.{i_level}/attn.{i_block}'))
            if i_level != self.num_resolutions-1:
                self.down.add(Downsample(block_in, name=f'{self.name}/down.{i_level}/downsample'))
                curr_res = curr_res // 2

        # middle
        self.mid = tf.keras.Sequential()
        self.mid.add(ResnetBlock(in_channels=block_in,
                                 out_channels=block_in,
                                 name=f'{self.name}/mid/block_1'))
        self.mid.add(AttnBlock(block_in, name=f'{self.name}/mid/attn_1'))
        self.mid.add(ResnetBlock(in_channels=block_in,
                                 out_channels=block_in,
                                 name=f'{self.name}/mid/block_2'))

        # end
        self.end = tf.keras.Sequential([
            Normalize(name=f'{self.name}/norm_out'),
            layers.Activation(nonlinearity),
            layers.ZeroPadding2D(),
            layers.Conv2D(z_channels, 3, name=f'{self.name}/conv_out')
        ])

    @property
    def last_layer(self):
        return self.end.layers[-1]

    def call(self, x, training=False):
        x = self.down(x, training=training)
        x = self.mid(x, training=training)
        x = self.end(x, training=training)
        return x


class Decoder(layers.Layer):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, in_channels,
                 image_size, z_channels, give_pre_end=False, name=None, **ignorekwargs):
        super().__init__(name=name)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = image_size
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = image_size // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = tf.keras.Sequential([
            layers.ZeroPadding2D(),
            layers.Conv2D(block_in, 3, name=f'{self.name}/conv_in')])

        # middle
        self.mid = tf.keras.Sequential()
        self.mid.add(ResnetBlock(in_channels=block_in,
                                 out_channels=block_in,
                                 name=f'{self.name}/mid/block_1'))
        self.mid.add(AttnBlock(block_in, name=f'{self.name}/mid/attn_1'))
        self.mid.add(ResnetBlock(in_channels=block_in,
                                 out_channels=block_in,
                                 name=f'{self.name}/mid/block_2'))

        # upsampling
        self.up = tf.keras.Sequential()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                self.up.add(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        name=f'{self.name}/up.{i_level}/block.{i_block}'))
                block_in = block_out
                if curr_res in attn_resolutions:
                    self.up.add(AttnBlock(block_in, name=f'{self.name}/up.{i_level}/attn.{i_block}'))
            if i_level != 0:
                self.up.add(Upsample(block_in, name=f'{self.name}/up.{i_level}/upsample'))
                curr_res = curr_res * 2

        # end
        self.end = tf.keras.Sequential([
            Normalize(name=f'{self.name}/norm_out'),
            layers.Activation(nonlinearity),
            layers.ZeroPadding2D(),
            layers.Conv2D(out_ch, 3, name=f'{self.name}/conv_out')])

    def call(self, z, training=False):
        h = self.conv_in(z, training=training)
        h = self.mid(h, training=training)
        h = self.up(h, training=training)
        if self.give_pre_end:
            return h
        return self.end(h, training=training)

    @property
    def last_layer(self):
        return self.end.layers[-1]


class VQGAN(tf.keras.Model):
    def __init__(self, config: VQGANConfig, name='vqgan', **kwargs):
        assert config.image_size % 8 == 0
        super().__init__(name=name, **kwargs)

        self.encoder = Encoder(**vars(config), name=f'{self.name}/encoder')
        self.decoder = Decoder(**vars(config), name=f'{self.name}/decoder')
        self.quantize = QuantizeEMA(config.embed_dim, config.n_embed, name=f'{self.name}/quantize')
        self.quant_conv = layers.Conv2D(config.embed_dim, 1, name='quant_conv')
        self.post_quant_conv = layers.Conv2D(config.z_channels, 1, name='post_quant_conv')
        self.learning_rate = config.learning_rate
        self.feature_image_size = config.image_size // (len(config.ch_mult) - 1)
        self.config = config

        self.perceptual_loss = lpips(net='vgg')

        self._metrics = [
            tf.metrics.Mean(x, dtype=tf.float32)
            for x in ['aeloss', 'mse', 'l_loss', "total_loss", "quant_loss", "rec_loss", "p_loss"]]
        self.metrics_dict = OrderedDict([(x.name, x) for x in self.metrics])

        # Initialize model
        init_input = tf.random.normal((1, config.image_size, config.image_size, 3), dtype=tf.float32)
        self(init_input, training=False)

    @property
    def metrics(self):
        return self._metrics

    def call(self, input, training=False):
        quant, diff, id = self.encode(input, training=training)
        post_quant = self.post_quant_conv(quant, training=training)
        dec = self.decoder(post_quant, training=training)
        return dec, diff, quant, id

    def encode(self, input, training=False):
        enc = self.encoder(input, training=training)
        enc = self.quant_conv(enc, training=training)
        quant, diff, id = self.quantize(enc, training=training)
        return quant, diff, id

    def decode_code(self, code, training=False):
        quant = self.quantize.embed_code(code)
        quant = self.post_quant_conv(quant, training=training)
        dec = self.decoder(quant, training=training)
        return dec

    @property
    def global_step(self):
        return self._train_counter

    def compile(self, **kwargs):
        if 'optimizer' not in kwargs:
            lr = self.learning_rate
            opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.9, epsilon=1e-8)
            kwargs['optimizer'] = opt
        super().compile(**kwargs)

    @property
    def generator_parameters(self):
        return list(chain(self.encoder.trainable_weights,
                          self.decoder.trainable_weights,
                          self.quantize.trainable_weights,
                          self.quant_conv.trainable_weights,
                          self.post_quant_conv.trainable_weights))

    def _compute_loss(self, codebook_loss, inputs, reconstructions):
        rec_loss = tf.reduce_mean(tf.abs(inputs - reconstructions), [-3, -2, -1])
        if self.config.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.config.perceptual_weight * p_loss
        else:
            p_loss = tf.constant(0., dtype=tf.float32, shape=(1,))

        loss = tf.reduce_mean(rec_loss) + self.codebook_weight * tf.reduce_mean(codebook_loss)
        log = {"total_loss": tf.reduce_mean(loss),
               "quant_loss": tf.reduce_mean(codebook_loss),
               "rec_loss": tf.reduce_mean(rec_loss),
               "p_loss": tf.reduce_mean(p_loss)}
        return loss, log

    def train_step(self, img):
        # Generator step
        with tf.GradientTape() as tape:
            out, latent_loss, *_ = self(img, training=True)
            aeloss, metrics = self._compute_loss(latent_loss, img, out)

        for k, v in metrics.items():
            self.metrics_dict[k].update_state(v)
        self.metrics_dict['aeloss'].update_state(aeloss)
        self.metrics_dict['mse'].update_state(tf.losses.mean_squared_error(img, out))
        self.metrics_dict['l_loss'].update_state(latent_loss)
        grads = tape.gradient(aeloss, self.generator_parameters)
        self.optimizer.apply_gradients(zip(grads, self.generator_parameters))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, img):
        out, latent_loss, *_ = self(img, training=False)
        aeloss, metrics = self._compute_loss(latent_loss, img, out)
        for k, v in metrics.items():
            self.metrics_dict[k].update_state(v)
        self.metrics_dict['aeloss'].update_state(aeloss)
        self.metrics_dict['mse'].update_state(tf.losses.mean_squared_error(img, out))
        self.metrics_dict['l_loss'].update_state(latent_loss)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, img):
        out, diff, quant, id = self(img, training=False)
        return {'decoded_image': out, 'latent_code': id}
