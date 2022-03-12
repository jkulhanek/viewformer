import numpy as np
from functools import lru_cache, reduce
from PIL import Image, ImageDraw


class ColorsLoader:
    _custom_resize = True
    _custom_sequence_size = True

    def __init__(self, split: str, num_sequences: int = 1000, sequence_size: int = 20, seed: int = 42, image_size: int = 128):
        self.split = split
        self.seed = seed
        self.sequence_size = sequence_size
        self.num_sequences = num_sequences
        self.image_size = image_size

    def __len__(self):
        return self.num_sequences

    def num_images_per_sequence(self):
        return [self.sequence_size] * self.num_sequences

    @lru_cache(maxsize=1)
    def __getitem__(self, idx):
        rng_seed = self.seed ^ idx ^ (reduce(lambda a, x: a * ord(x), self.split, 1) % 31)
        gen = np.random.RandomState(rng_seed)
        env_color = gen.randint(0, 255, (3,), dtype=np.uint8)
        poses = gen.uniform(size=(self.sequence_size, 3)).astype(np.float32)
        poses = np.concatenate([poses, np.ones((self.sequence_size, 4), dtype=poses.dtype) * np.array([0, 0, 1, 0], dtype=poses.dtype)], -1)
        frames = []
        for pose in poses:
            image = Image.new('RGB', (self.image_size, self.image_size), tuple(env_color))
            d = ImageDraw.Draw(image)
            x, y = pose[0] * self.image_size, pose[2] * self.image_size
            d.ellipse([int(x - self.image_size // 6), int(y - self.image_size // 6), int(x + self.image_size // 6), int(y + self.image_size // 6)],
                      fill=tuple(255 - env_color))
            frames.append(np.array(image))
        poses[..., :3] = poses[..., :3] * 2 - 1
        frames = np.stack(frames, 0)
        return dict(cameras=poses, frames=frames)
