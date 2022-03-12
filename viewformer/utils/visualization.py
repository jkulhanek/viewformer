import math
from ._common import batch_len, batch_slice
import numpy as np


def np_imgrid(imarray, cols=None, pad=1, row_major=True):
    """Lays out a [N, H, W, C] image array as a single image grid."""
    pad = int(pad)
    if pad < 0:
        raise ValueError('pad must be non-negative')
    N, H, W, *C = imarray.shape
    if cols is None:
        cols = int(math.ceil(math.sqrt(N)))
    else:
        cols = int(cols)
        assert cols >= 1
    rows = N // cols + int(N % cols != 0)
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad] + [0 for _ in C]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, constant_values=0)
    H += pad
    W += pad
    grid = imarray.reshape([rows, cols, H, W, *C])
    grid = grid.transpose([0, 2, 1, 3] + [4 + i for i, _ in enumerate(C)])
    grid = grid.reshape([1, rows * H, cols * W, *C])
    if pad:
        grid = grid[:, :-pad, :-pad]
    return grid


class EnvironmentViewerDataSource:
    def __init__(self, dataset, transform_batch):
        self.dataset = dataset
        self.transform_batch = transform_batch
        self._local_i = 0
        self._global_i = 0
        self._iterator = iter(dataset)
        self.next_environment()

    @property
    def local_index(self):
        return self._local_i

    @property
    def local_len(self):
        return len(self._current_cache)

    @property
    def global_index(self):
        return self._global_i

    def next_environment(self):
        self._local_i = 0
        self._current_batch = next(self._iterator)
        self._current_cache = [None] * batch_len(self._current_batch)
        self._global_i += 1
        self._render()

    def next(self):
        if self._local_i == batch_len(self._current_batch) - 1:
            return False
        self._local_i += 1
        self._render()
        return True

    def previous(self):
        if self._local_i == 0:
            return False
        self._local_i -= 1
        self._render()
        return True

    @property
    def data(self):
        return batch_slice(self._current_batch, self._local_i), self._current_cache[self._local_i]

    def _transform_batch(self, batch, i):
        value = self.transform_batch(batch_slice(batch, slice(i, i + 1)))
        value = batch_slice(value, 0)
        return value

    def _render(self):
        if self._current_cache[self._local_i] is None:
            value = self._transform_batch(self._current_batch, self._local_i)
            self._current_cache[self._local_i] = value
        return self._current_cache[self._local_i]


class MatplotlibViewer:
    fig_size = (10, 10)

    def __init__(self, data_source):
        self.data_source = data_source
        self._first_call = True

    def render(self, data, is_first_call=False):
        raise NotImplementedError()

    def on_key_press(self, event):
        if event.key == 'left':
            self.data_source.previous()
        elif event.key == 'right':
            self.data_source.next()
        elif event.key == 'n':
            self.data_source.next_environment()
        elif event.key == 'q':
            exit()
        self._redraw()

    @property
    def plt(self):
        return self._plt

    @property
    def fig(self):
        return self._fig

    def _redraw(self):
        if self._first_call:
            import matplotlib.pyplot as plt

            self._plt = plt
            self._fig = plt.figure(figsize=self.fig_size)
            self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.render(self.data_source.data, True)
            self._first_call = False
            self.plt.show()
        else:
            self.render(self.data_source.data, False)
            self.plt.draw()

    def show(self):
        self._first_call = True
        self._redraw()

    def start(self):
        self.show()


def pose_marker(rotation):
    from matplotlib.path import Path
    from matplotlib.markers import MarkerStyle
    vertices, codes = zip(*[
        ((1, 1), Path.MOVETO),
        ((1, -1), Path.LINETO),
        ((-1, -1), Path.LINETO),
        ((-1, 1), Path.LINETO),
        ((1, 1), Path.LINETO),
        ((3, 7), Path.LINETO),
        ((-3, 7), Path.LINETO),
        ((-1, 1), Path.LINETO),
    ])
    t = MarkerStyle(marker=Path(vertices, codes))
    t._transform = t.get_transform().rotate(rotation - math.pi / 2)
    return t
