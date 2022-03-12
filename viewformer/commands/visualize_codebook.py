from aparse import click, ConditionalType
import tensorflow as tf
from viewformer.utils.visualization import EnvironmentViewerDataSource, MatplotlibViewer
from viewformer.data.loaders import get_loaders
from viewformer.utils import dict_replace
from viewformer.utils.tensorflow import load_model
from viewformer.data.tfrecord_dataset import loader_to_dataset


#
# Types used in argument parsing
#
def _loader_switch_cls(cls):
    class Loader(cls):
        # Disable image_size argument in loader classes
        def __init__(self, *args, image_size=None, **kwargs):
            raise NotImplementedError()

        def __new__(_cls, *args, **kwargs):
            # Return callback to construct Loader on the Fly
            return cls(*args, **kwargs, image_size=128)
    return Loader


LoaderSwitch = ConditionalType('Loader', {k: _loader_switch_cls(v) for k, v in get_loaders().items()}, default='dataset')


class Viewer(MatplotlibViewer):
    def render(self, batch, first_call):
        import tensorflow as tf
        inp, im_decoded = batch
        im_orig = inp['frames'] / 2 + 0.5
        im_decoded = im_decoded / 2 + 0.5
        im_diff = tf.abs(im_orig - im_decoded)

        if first_call:
            self._axs = self.fig.subplots(1, 3)
            self._horig = self._axs[0].imshow(im_orig)
            self._axs[0].axis('off')
            self._axs[0].set_title('Original image')
            self._hdec = self._axs[1].imshow(im_decoded)
            self._axs[1].axis('off')
            self._axs[1].set_title('Decoded image')
            self._hdiff = self._axs[2].imshow(im_diff)
            self._axs[2].axis('off')
            self._axs[2].set_title('Difference')
        else:
            self._horig.set_data(im_orig)
            self._hdec.set_data(im_decoded)
            self._hdiff.set_data(im_diff)

        cur_pose = inp['cameras']
        self.fig.suptitle(f'idx={self.data_source.local_index:d} pos=({cur_pose[0]:.3f}, {cur_pose[1]:.3f}, {cur_pose[3]:.3f})')


@click.command()
def main(source: LoaderSwitch, split: str = 'test', model: str = None, seed: int = 42):
    tf.random.set_seed(seed)
    codebook_model = load_model(model)

    def get_reconstructed_image(batch):
        x = tf.image.convert_image_dtype(batch['frames'], 'float32')
        x = codebook_model(x)[0]
        x = tf.clip_by_value(x, -1, 1)
        return x

    dataset = loader_to_dataset(source)
    dataset = dataset.map(lambda x: dict_replace(x, 'frames', 2 * tf.image.convert_image_dtype(x['frames'], 'float32') - 1))
    dataset = dataset.repeat(-1)
    data_source = EnvironmentViewerDataSource(dataset, get_reconstructed_image)
    viewer = Viewer(data_source)
    viewer.start()
