import os
from collections import defaultdict
from functools import partial
import numpy as np
import sys
from typing import List
from PIL import Image
from viewformer.utils.geometry import rotation_matrix_to_quaternion, quaternion_normalize
from viewformer.data._common import LazyArray
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache()


ALL_CATEGORIES = ['cars', 'chairs']
_BLACKLIST = defaultdict(set)
_BLACKLIST['cars_train'] = {'4cce557de0c31a0e70a43c2d978e502e'}
_BLACKLIST['chairs_train'] = {
    # Missing files
    '18e5d3054fba58bf6e30a0dcfb43d654',
    '2a197b179994b877f63e8e405d49b8ce',
    '2be29f8ad81e2742eaf14273fa406ffc',
    '2cb0ac27f1cdb3f0b2db0181fdb9f615',
    '3d5053323021b1babbaf011bdbf27c0e',
    '4a671498c6e96238bf8db636a3460ee5',
    '4a89aad97f4c503068d1b9a1d97e2846',
    '738188ae01887d2349bb1cbbf9a4206',
    '8b552c23c064b96179368d1198f406e7',
    '9505568d7a277c7bdd7092ed47061a36',
    '9d0043b17b97ac694925bc492489de9c',
    'b46361e53253c07b6fa2cfca412075ea',
    'b88d8b5e5fbee4fa8336a02debb9923b',
    'c41fe0605cfe70571c25d54737ed5c8e',
    'cadf69f5353039e8593ebeeedbff73b',
    'chairs_2.0_train',
    'd323e6d19dc58526f2c0effc06a15c11',
    'e94befd51c02533b17b431cae0dd70ed',

    # Invalid poses
    '8f13ac6499dfcc83f381af8194aa4242',
    '7f8fc2fdc88e4ca1152b86a40777b4c',
    '49d6f3affe205cc4b04cb542e2c50eb4',
    'cbe006da89cca7ffd6bab114dd47e3f',
    '47d13a704da37b588fda227abcbd8611',
    '59c89dc89cf0d34e597976c675750537',
    '2d08a64e4a257e007135fc51795b4038',
    '752edd549ca958252b4875f731f71cd',
    'd5b9579151041cbd9b9f2eb77f5e247e',
}

_SEQ_SIZES = {
    'cars_train': (2151 - 1, 250),
    'cars_test': (704, 251),
    'chairs_train': (4613 - 27, 200),
    'chairs_test': (1317, 251),
}


class ShapenetLoader:
    _images_per_scene = dict()

    def __init__(self, path: str, split: str, categories: List[str] = None, seed=None, sequences=None):
        assert split in ['test', 'train']
        if categories is None:
            categories = ALL_CATEGORIES
        self.categories = categories
        self.split = split
        self.path = path
        self.sequences = sequences
        if len(self.categories) == 1:
            _, self.sequence_size = _SEQ_SIZES[f'{self.categories[0]}_{self.split}']

    def num_images_per_sequence(self):
        if self.sequences is not None:
            return sum(([_SEQ_SIZES[f'{x}_{self.split}'][-1]] * len(self._get_seqs(x)) for x in self.categories), [])
        return sum(([ln] * num for num, ln in (_SEQ_SIZES[f'{x}_{self.split}'] for x in self.categories)), [])

    @cache
    def __len__(self):
        if self.sequences is not None:
            return sum(len(self._get_seqs(x)) for x in self.categories)
        return sum(num for num, ln in (_SEQ_SIZES[f'{x}_{self.split}'] for x in self.categories))

    @staticmethod
    def camera_to_world_matrices_to_cameras(cam_to_world):
        position = cam_to_world[..., :-1, -1]
        R = cam_to_world[..., :-1, :-1]
        quaternion = rotation_matrix_to_quaternion(R)
        quaternion = quaternion_normalize(quaternion)
        return np.concatenate([position, quaternion], -1)

    @cache
    def _get_seqs(self, category):
        xs = os.listdir(os.path.join(self.path, f'{category}_{self.split}'))
        if self.sequences is not None:
            xs = set(xs)
            xs = [x for x in self.sequences if x in xs]
        else:
            xs = [x for x in xs if x not in _BLACKLIST[f'{category}_{self.split}']]
            xs.sort()
        return xs

    def read_camera(self, category, seq_name, i):
        with open(os.path.join(self.path, f'{category}_{self.split}', seq_name, 'pose', f'{i:06d}.txt'), 'r') as f:
            camera_to_world_matrix = np.array(list(map(float, f.read().strip().split())), dtype=np.float32)
        camera_to_world_matrix = camera_to_world_matrix.reshape((4, 4))
        return self.camera_to_world_matrices_to_cameras(camera_to_world_matrix)

    def read_image(self, category, seq_name, i):
        return np.array(Image.open(os.path.join(self.path, f'{category}_{self.split}', seq_name, 'rgb', f'{i:06d}.png')).convert('RGB'))

    def __getitem__(self, i):
        # Find split
        for cat in self.categories:
            num, ln = _SEQ_SIZES[f'{cat}_{self.split}']
            if i < num:
                break
            i -= num
        else:
            raise StopIteration()
        indices = list(range(ln))
        seq_name = self._get_seqs(cat)[i]
        output = dict()
        output['cameras'] = LazyArray(indices, partial(self.read_camera, cat, seq_name))
        output['frames'] = LazyArray(indices, partial(self.read_image, cat, seq_name))
        output['sequence_id'] = seq_name
        return output


if __name__ == '__main__':
    ll = ShapenetLoader(sys.argv[1])
    ll[0]
    breakpoint()
