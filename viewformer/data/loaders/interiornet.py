import math
import os
import random
import re
import numpy as np
import sys
from PIL import Image
from viewformer.utils import SplitIndices
from viewformer.utils.geometry import look_at_to_cameras
from viewformer.data._common import ArchiveStore
from viewformer.data._common import ShuffledLoader


class _InteriorNetLoader:
    _custom_shuffle = True

    def __init__(self, path: str, sequence_size: int = None, max_environments_per_scene: int = -1, seed: int = 42,
                 parts: SplitIndices = None, shuffle_sequence_items: bool = None, shuffle_sequences: bool = False, split: str = None):
        if parts is None:
            parts = SplitIndices('7')
        dataset_parts = parts.restrict(range(1, 8))
        assert max_environments_per_scene, 'Currently, only max_environments_per_scene=1 is supported'
        assert not shuffle_sequences

        assert split in {'train', 'test'}
        self.images_per_environment = sequence_size or 20
        self.max_environments_per_scene = max_environments_per_scene

        # NOTE: These ignored files can likely be removed
        # The following files were probably incorectly downloaded
        # We will keep them for reproducibility
        # NOTE: also, all tar files were converted to zip
        self._ignored_files = [
            '3FO4K5I8T7KR', '3FO4K5I8T7KR', '3FO4K3GYULI6', '3FO4K5I8T7KR',
            '3FO4K35GPEA7', '3FO4K6XVLSCH', '3FO4K33RY528', '3FO4JXJX64SU',
            '3FO4K5LPQL51', '3FO4K6YTSO3Y', '3FO4K6WXLP01', ]

        # NOTE first 3% are testing data

        self._environment_files = []
        self._hd16_len = 0
        self._hd7_len = 0
        self._images_per_scene = (3000, 20)
        self._environment_per_scene = tuple(
            min(max_environments_per_scene, x // self.images_per_environment)
            if max_environments_per_scene > 0
            else x // self.images_per_environment for x in self._images_per_scene)

        assert os.path.exists(os.path.join(path, 'GroundTruth_HD1-HD6')), 'Not a valid dataset, missing GroundTruth_HD1-HD6 folder'
        for i in sorted(dataset_parts):
            assert os.path.exists(os.path.join(path, f'HD{i}')), f'Not a valid dataset, missing HD{i} folder'
            part_files = [os.path.join(path, f'HD{i}', x) for x in ArchiveStore.list_archives(os.path.join(path, f'HD{i}')) if x not in self._ignored_files]
            part_files = sorted(part_files)
            if split is not None:
                num_test = int(math.ceil(len(part_files) * 0.03))
                if split == 'test':
                    part_files = part_files[:num_test]
                else:
                    part_files = part_files[num_test:]
            self._environment_files.extend(part_files)
            if i < 7:
                self._hd16_len += len(part_files)
            else:
                self._hd7_len += len(part_files)
        self._ctx = None
        self.shuffle_environment = shuffle_sequence_items

    def get_intrinsics(self):
        # Return (image_height, image_width, f_x, f_y, c_x, c_y)
        return (640, 480, 600, 600, 320, 240)

    def __len__(self):
        hd16_size, hd7_size = self._environment_per_scene
        return self._hd16_len * hd16_size + self._hd7_len * hd7_size

    def num_images_per_sequence(self):
        return [self.images_per_environment] * len(self)

    def _rotate_system(self, pos):
        x, y, z = np.moveaxis(pos, -1, 0)
        return np.stack((y, -z, -x), -1)

    def _convert_poses(self, poses):
        # first three elements, eye and next three, lookAt and the last there, up direction
        eye = self._rotate_system(poses[..., 0:3])
        lookat = self._rotate_system(poses[..., 3:6])
        up = self._rotate_system(poses[..., 6:9])
        return look_at_to_cameras(eye, lookat, up)

    def close(self):
        if self._ctx is not None:
            self._ctx.__exit__()
            self._ctx = None

    def _ensure_context(self):
        if self._ctx is None:
            self._ctx = ArchiveStore.with_context().__enter__()

    def __enter__(self, *args, **kwargs):
        self._ensure_context()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _parse_cam(self, file):
        last_id = None
        for line in file:
            line = line.rstrip('\n\r')
            vals = line.split()
            if vals[0].isnumeric():
                if last_id != vals[0]:
                    yield vals[0], np.array([float(x) for x in vals[1:]], dtype='float32')
                last_id = vals[0]

    def __getitem__(self, i):
        self._ensure_context()
        hd16_size, hd7_size = self._environment_per_scene
        if i >= self._hd16_len * hd16_size:
            env_i = (i - self._hd16_len * hd16_size) // hd7_size + self._hd16_len
            i = (i - self._hd16_len * hd16_size) % hd7_size
            is_hd16 = False
        else:
            env_i = i // hd16_size
            i = i % hd16_size
            is_hd16 = True
        fname = self._environment_files[env_i]
        images = []
        # depthmaps = []
        cameras = []
        data = []
        with ArchiveStore(fname) as archive:
            if is_hd16:
                par_dir, archivename = os.path.split(fname)
                par_dir = os.path.join(os.path.dirname(par_dir), 'GroundTruth_HD1-HD6')
                with ArchiveStore(os.path.join(par_dir, archivename)) as gt_archive:
                    subdirs = [re.match(r'^.*(\d+_\d+)$', x) for x in gt_archive.ls('')]
                    subdir_postfixes = [x.group(1) for x in subdirs if x is not None]
                    subdirs = [f'original_{x}/' for x in subdir_postfixes]
                    for subdir, postfix in zip(subdirs, subdir_postfixes):
                        with gt_archive.open(f'velocity_angular_{postfix}/cam0.render', 'r') as f:
                            for pose_id, pose_data in self._parse_cam(f):
                                data.append((subdir, pose_id, pose_data))
            else:
                with archive.open('cam0.render', 'r') as f:
                    for pose_id, pose_data in self._parse_cam(f):
                        data.append(('', pose_id, pose_data))

            rng = random.Random(env_i)
            if self.shuffle_environment:
                rng.shuffle(data)

            num_resamples = 0
            rng.seed(i)

            def try_add(i):
                nonlocal num_resamples
                subdir, pose_id, pose_data = data[i]
                try:
                    image = np.array(Image.open(archive.open(f'{subdir}cam0/data/{pose_id}.png', 'rb')).convert('RGB'))
                    # depthmap = np.array(Image.open(archive.open(f'{subdir}depth0/data/{pose_id}.png', 'rb')).convert('F'))
                    images.append(image)
                    # depthmaps.append(depthmap)
                    cameras.append(pose_data)
                except Exception as e:
                    print(f'Invalid image file "{subdir}cam0/data/{pose_id}.png" or "{subdir}depth0/data/{pose_id}.png" in archive {fname}', file=sys.stderr)
                    if num_resamples >= 1:
                        raise e
                    num_resamples += 1
                    try_add(rng.randrange(0, len(data)))
            for j in range(i * self.images_per_environment, (i + 1) * self.images_per_environment):
                try_add(j)

        output = dict()
        cameras = np.stack(cameras, 0)
        cameras = self._convert_poses(cameras)
        output['cameras'] = cameras
        output['frames'] = np.stack(images, 0)
        # output['depthmaps'] = np.stack(depthmaps, 0)
        return output


class InteriorNetLoader(_InteriorNetLoader):
    def __new__(cls, *args, shuffle_sequences: bool = None, **kwargs):
        loader = _InteriorNetLoader(*args, **kwargs)
        if shuffle_sequences:
            loader = ShuffledLoader(loader, kwargs.get('seed', 42), shuffle_sequences=True)
        return loader

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


if __name__ == '__main__':
    import sys
    ll = InteriorNetLoader(sys.argv[1], image_only=True)
    ll[0]
    breakpoint()
