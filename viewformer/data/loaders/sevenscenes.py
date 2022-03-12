import os
import fsspec
from collections import OrderedDict
import numpy as np
import sys
from typing import List
from PIL import Image
from viewformer.utils.geometry import rotation_matrix_to_quaternion, quaternion_normalize
from viewformer.data._common import LazyArray
from viewformer.data._common import ArchiveStore
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache()


ALL_SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


def rotation_matrix_to_euler_xyz(R):
    thetaY = np.arcsin(R[..., 0, 2])
    thetaX = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    thetaZ = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    euler1 = np.stack([thetaX, thetaY, thetaZ], -1)

    thetaY = -np.ones_like(R[..., 0, 0]) * np.pi / 2
    thetaX = -np.arctan2(R[..., 1, 0], R[..., 1, 1])
    thetaZ = np.zeros_like(R[..., 0, 0])
    euler2 = np.stack([thetaX, thetaY, thetaZ], -1)

    thetaY = np.ones_like(R[..., 0, 0]) * np.pi / 2
    thetaX = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    thetaZ = np.zeros_like(R[..., 0, 0])
    euler3 = np.stack([thetaX, thetaY, thetaZ], -1)

    euler12 = np.where(R[..., 0, 2:3] > -1, euler1, euler2)
    euler = np.where(R[..., 0, 2:3] < 1, euler12, euler3)
    return euler


class SevenScenesLoader:
    _images_per_scene = dict()

    def __init__(self, path: str, split: str = None, scenes: List[str] = None, seed=None, _load_file_paths=False):
        if scenes is None:
            scenes = ALL_SCENES
        self.scenes = scenes
        self.split = split

        # List files
        self.path = path
        self._load_file_paths = _load_file_paths

        for scene in scenes:
            assert os.path.exists(os.path.join(path, f'{scene}.zip')), f'Not a valid dataset, missing {scene}.zip file'

    @cache
    def get_seqs(self):
        splits = ['train', 'test']
        if self.split is not None:
            splits = [self.split]
        scene_seqs = OrderedDict()
        for scene in self.scenes:
            with ArchiveStore(os.path.join(self.path, f'{scene}')) as archive:
                seqs = []
                for split in splits:
                    with archive.open(f'{split.title()}Split.txt', 'r') as f:
                        for line in f:
                            line = line.rstrip('\n\r')
                            if line and line.startswith('sequence'):
                                seqs.append(int(line[len('sequence'):]))
            scene_seqs[scene] = seqs
        return scene_seqs

    @cache
    def num_images_per_sequence(self):
        return [len(self.read_scene_sequence_metadata(self.path, scene, seq)[0]) for scene, seqs in self.get_seqs().items() for seq in seqs]

    @cache
    def __len__(self):
        return sum(map(len, self.get_seqs().values()))

    def cam_to_world_to_poses(self, cam_to_world):
        t = cam_to_world[..., :-1, -1]
        R = cam_to_world[..., :-1, :-1]
        euler = rotation_matrix_to_euler_xyz(R)
        return np.concatenate([t, euler], -1)

    @staticmethod
    def camera_to_world_matrices_to_cameras(cam_to_world):
        position = cam_to_world[..., :-1, -1]
        R = cam_to_world[..., :-1, :-1]
        quaternion = rotation_matrix_to_quaternion(R)
        quaternion = quaternion_normalize(quaternion)
        return np.concatenate([position, quaternion], -1)

    def get_intrinsics(self):
        # Return (image_width, image_height, f_x, f_y, c_x, c_y)
        return (640, 480, 585, 585, 320, 240)

    @staticmethod
    @cache
    def read_scene_sequence_metadata(path, scene, seq):
        sup_archive = ArchiveStore(os.path.join(path, f'{scene}')).__enter__()
        archive = ArchiveStore(sup_archive.open(f'seq-{seq:02}.zip', 'r')).__enter__()
        seq_items = sorted(set(x[:x.index('.')] for x in archive.ls('') if '.' in x and 'thumbs' not in x.lower()))
        _open = archive.open

        camera_to_world_matrices = []
        for framename in seq_items:
            camera_to_world = np.ndarray((4, 4), dtype='float32')
            with _open(f'{framename}.pose.txt', 'r') as f:
                for r, line in enumerate(f):
                    line = line.strip('\r\n')
                    for i, val in enumerate(line.split()):
                        val = float(val)
                        camera_to_world[r, i] = val
            camera_to_world_matrices.append(camera_to_world)
        camera_to_world_matrices = np.stack(camera_to_world_matrices, 0)
        return seq_items, camera_to_world_matrices, _open

    def __getitem__(self, i):
        # Find split
        for scene, seqs in self.get_seqs().items():
            if i < len(seqs):
                break
            i -= len(seqs)
        else:
            raise StopIteration()
        seq = seqs[i]
        seq_items, camera_to_world_matrices, _open = self.read_scene_sequence_metadata(self.path, scene, seq)


        def get_depthmaps():
            return np.stack([np.array(Image.open(_open(f'{framename}.depth.png', 'rb')).convert('F')) for framename in seq_items], 0)

        output = dict()
        output['cameras'] = self.camera_to_world_matrices_to_cameras(camera_to_world_matrices)
        output['frames'] = LazyArray(seq_items, lambda framename: np.array(Image.open(_open(f'{framename}.color.png', 'rb')).convert('RGB')))
        if self._load_file_paths:
            output['frames_files'] = [f'seq-{seq:02}/{x}' for x in seq_items]
        # output['depthmaps'] = get_depthmaps
        return output


if __name__ == '__main__':
    import sys
    ll = SevenScenesLoader(sys.argv[1], limit_images=20)
    ll[0]
    breakpoint()
