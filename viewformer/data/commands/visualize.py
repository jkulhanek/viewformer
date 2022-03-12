import math
import numpy as np
from aparse import click, ConditionalType
from viewformer.utils.visualization import MatplotlibViewer, EnvironmentViewerDataSource, pose_marker, np_imgrid
from viewformer.utils import batch_slice
from viewformer.utils.geometry import cameras_to_pose_euler
import viewformer.data


LoaderSwitch = ConditionalType('LoaderSwitch', viewformer.data.loaders.get_loaders(), prefix=False, default='dataset')


class MatplotlibImagesAndPoseRenderer(MatplotlibViewer):
    fig_size = (15, 5)

    def __init__(self, loader):
        class DataSource(EnvironmentViewerDataSource):
            def __init__(self, loader):
                super().__init__(loader, None)

            @property
            def data(self):
                if 'preview' not in self._current_batch:
                    try:
                        import tensorflow as tf
                        self._current_batch['preview'] = tf.image.resize(self._current_batch['frames'], (20, 20)).numpy().astype('uint8')
                    except ImportError:
                        self._current_batch['preview'] = self._current_batch['frames']
                return self._current_batch, batch_slice(self._current_batch, self._local_i)

            def _render(self, *args, **kwargs):
                pass

        self.marker_dir_len = 1
        super().__init__(DataSource(loader))

    @staticmethod
    def imgrid(imarray, cols=4, pad=1, row_major=True):
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
        imarray = np.pad(imarray, pad_arg, constant_values=0)
        H += pad
        W += pad
        grid = np.reshape(imarray, [rows, cols, H, W, C])
        grid = np.transpose(grid, [0, 2, 1, 3, 4])
        grid = np.reshape(grid, [rows*H, cols*W, C])
        if pad:
            grid = grid[:-pad, :-pad]
        return grid

    def render(self, data, is_first_call):
        data, single_item = data
        if is_first_call:
            axs = self._fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
            self._fig.tight_layout(pad=3.0)
            self._axs = axs

        self._axs[0].clear()
        self._axs[0].axis('off')
        local_frames = data['preview'].copy()
        local_frames[self.data_source.local_index] = 0
        self._axs[0].imshow(np_imgrid(local_frames)[0])
        self._axs[2].clear()
        self._axs[2].imshow(single_item['frames'])
        self._axs[1].clear()
        self._axs[1].axis('equal')
        self._axs[1].set_xlabel('x')
        self._axs[1].set_ylabel('z')
        self._axs[1].grid('on')
        cameras = cameras_to_pose_euler(data['cameras'])
        for i, row in enumerate(cameras):
            if i != self.data_source.local_index:
                x, y, z, rx, ry, rz = row
                self._draw_pose(ax=self._axs[1], pose=(x, z, math.pi / 2 - ry), color=(0.7, 0.7, 0.7))
        x, y, z, rx, ry, rz = cameras[self.data_source.local_index]
        self._draw_pose(ax=self._axs[1], pose=(x, z, math.pi / 2 - ry), color='r')
        self._fig.suptitle(f'{self.data_source.local_index}/{self.data_source.local_len} {data.get("sequence_id", "")}')
        label = f'x:{x:.2f} y:{y:.2f} z:{z:.2f} rx:{rx:.2f} ry:{ry:.2f} rz:{rz:.2f}'
        self._axs[2].text(0.5, -0.15, label, ha="center", transform=self._axs[2].transAxes)

    def _draw_pose(self, ax, pose, color='k'):
        x, y, yaw = pose
        po = ax.plot(x, y, marker=pose_marker(yaw), markersize=40, markerfacecolor='None').pop(0)
        po.set_color(color)
        return po

    def _set_pose_color(self, ph, color):
        ph[0].set_color(color)
        ph[1].set_color(color)

    def _update(self):
        self._fig.canvas.draw()


@click.command('visualize')
def main(loader: LoaderSwitch):
    renderer = MatplotlibImagesAndPoseRenderer(loader)
    renderer.show()


if __name__ == '__main__':
    main()
