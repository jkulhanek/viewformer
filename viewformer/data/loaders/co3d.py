import logging
from itertools import groupby, accumulate
from functools import partial
from typing import List, Union
import copy
import os
import io
import numpy as np
import torch
from viewformer.utils.geometry import rotation_matrix_to_quaternion, quaternion_normalize
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache()


CO3D_CATEGORIES = list(reversed([
    "baseballbat", "banana", "bicycle", "microwave", "tv",
    "cellphone", "toilet", "hairdryer", "couch", "kite", "pizza",
    "umbrella", "wineglass", "laptop",
    "hotdog", "stopsign", "frisbee", "baseballglove",
    "cup", "parkingmeter", "backpack", "toyplane", "toybus",
    "handbag", "chair", "keyboard", "car", "motorcycle",
    "carrot", "bottle", "sandwich", "remote", "bowl", "skateboard",
    "toaster", "mouse", "toytrain", "book", "toytruck",
    "orange", "broccoli", "plant", "teddybear",
    "suitcase", "bench", "ball", "cake",
    "vase", "hydrant", "apple", "donut",
]))


def _invalid_color_error_msg(bg_color) -> str:
    return (
        f"Invalid bg_color={bg_color}. Plese set bg_color to a 3-element"
        + " tensor. or a string (white | black), or a float."
    )


def mask_background(image_rgb: torch.Tensor, mask_fg: torch.Tensor, ch_axis=-3):
    """
    Mask the background input image tensor `image_rgb` with `bg_color`.
    The background regions are obtained from the binary foreground segmentation
    mask `mask_fg`.
    """
    tgt_view = [1] * len(image_rgb.shape)
    tgt_view[ch_axis] = 3
    # obtain the background color tensor
    bg_color_t = image_rgb.new_zeros(tgt_view)
    # cast to the image_rgb's type
    mask_fg = mask_fg.type_as(image_rgb)
    # mask the bg
    image_masked = mask_fg * image_rgb + (1 - mask_fg) * bg_color_t
    return image_masked


def co3d_val_dataset(DATASET_CONFIGS, Co3dDataset, path, category, unseen=False):
    frame_file = os.path.join(path, category, "frame_annotations.jgz")
    sequence_file = os.path.join(path, category, "sequence_annotations.jgz")
    subset_lists_file = os.path.join(path, category, "set_lists.json")
    params = {
        **copy.deepcopy(DATASET_CONFIGS["default"]),
        "frame_annotations_file": frame_file,
        "sequence_annotations_file": sequence_file,
        "subset_lists_file": subset_lists_file,
        "dataset_root": path,
        "limit_to": -1,
        "limit_sequences_to": -1,
        "n_frames_per_sequence": -1,
        "subsets": ['test_known' if not unseen else 'test_unseen'],
        "load_point_clouds": False,
        "mask_images": False,
        "mask_depths": False,
        "pick_sequence": [],
    }
    return Co3dDataset(**params)


class CO3DLoader:
    _images_per_scene = dict()

    def __init__(self, path: str, split: str = None, categories: List[str] = None, mask_images: bool = True):
        assert split in ['test', 'train', 'val']
        self._installed = False
        self.install()

        if categories is None:
            categories = CO3D_CATEGORIES
        self.categories = categories
        self.split = split
        self.path = path
        self.mask_images = mask_images

        # List files
        self.path = path

    def install(self):
        if not self._installed:
            if not os.path.exists(os.path.expanduser('~/.cache/viewformer/co3d')):
                import urllib.request
                import zipfile
                import shutil
                os.makedirs(os.path.expanduser('~/.cache/viewformer'), exist_ok=True)
                with urllib.request.urlopen('https://github.com/facebookresearch/co3d/archive/d4895dd3976b1c6afb9e9221c047f67c678eaf08.zip') as f:
                    with io.BytesIO(f.read()) as bytes_io:
                        f.close()
                        with zipfile.ZipFile(bytes_io, 'r') as archive:
                            archive.extractall(os.path.expanduser('~/.cache/viewformer'))
                shutil.move(os.path.expanduser('~/.cache/viewformer/co3d-d4895dd3976b1c6afb9e9221c047f67c678eaf08'), os.path.expanduser('~/.cache/viewformer/co3d'))
                logging.info(f'CO3D installed to "{os.path.expanduser("~/.cache/viewformer/co3d")}"')

            def use_dataset_zoo():
                class ctx:
                    def __enter__(self):
                        import sys
                        sys.path.insert(0, os.path.expanduser('~/.cache/viewformer/co3d'))
                        from dataset.dataset_zoo import dataset_zoo
                        return dataset_zoo

                    def __exit__(self, *args, **kwargs):
                        import sys
                        sys.path.remove(os.path.expanduser('~/.cache/viewformer/co3d'))
                return ctx()

            def use_val_dataset():
                class ctx:
                    def __enter__(self):
                        import sys
                        sys.path.insert(0, os.path.expanduser('~/.cache/viewformer/co3d'))
                        from dataset.co3d_dataset import Co3dDataset
                        from dataset.dataset_zoo import DATASET_CONFIGS
                        val_dataset = partial(co3d_val_dataset, Co3dDataset=Co3dDataset, DATASET_CONFIGS=DATASET_CONFIGS)
                        return val_dataset

                    def __exit__(self, *args, **kwargs):
                        import sys
                        sys.path.remove(os.path.expanduser('~/.cache/viewformer/co3d'))
                return ctx()

            self.use_val_dataset = use_val_dataset
            self.use_dataset_zoo = use_dataset_zoo
            self._installed = True

    @staticmethod
    def world_to_camera_to_cameras(cam_to_world):
        world_to_cam = np.linalg.inv(cam_to_world)

        # Change coordinate system
        # in PyTorch3d, z points to screen, y up and x to left
        # in our dataset, z points to screen, y down and x to right
        world_to_cam[..., :2, :] *= -1

        R = world_to_cam[..., :-1, :-1]
        position = world_to_cam[..., :-1, -1]

        quaternion = rotation_matrix_to_quaternion(R)
        quaternion = quaternion_normalize(quaternion)
        return np.concatenate([position, quaternion], -1)

    @cache
    def _dataset(self):
        from torch.utils.data.dataset import ConcatDataset
        if self.split == 'train':
            with self.use_dataset_zoo() as dataset_zoo:
                return ConcatDataset([dataset_zoo('co3d_multisequence', self.path, category=c)[self.split] for c in self.categories])
        else:
            unseen = self.split == 'test'
            with self.use_val_dataset() as val_dataset:
                return ConcatDataset([val_dataset(path=self.path, category=c, unseen=unseen) for c in self.categories])

    def __len__(self):
        return len(self.num_images_per_sequence())

    @cache
    def num_images_per_sequence(self):
        return [sum(1 for _ in x) for _, x in groupby((x for d in self._dataset().datasets for x in d.frame_annots), key=lambda x: x['frame_annotation'].sequence_name)]

    @cache
    def _cum_images_per_sequence(self):
        return [0] + list(accumulate(self.num_images_per_sequence()[:-1]))

    def get_intrinsics(self):
        # Return (image_width, image_height, f_x, f_y, c_x, c_y)
        return (800, 800, 400, 400, 400, 400)

    def _process_rgb_image(self, frame_data):
        # threshold the masks to make ground truth binary masks
        if self.mask_images:
            mask_fg = frame_data.fg_probability >= 0.5
            image_rgb_masked = mask_background(
                frame_data.image_rgb,
                mask_fg,
            )
            return image_rgb_masked
        else:
            return frame_data.image_rgb

    def __getitem__(self, i):
        start = self._cum_images_per_sequence()[i]
        seq_len = self.num_images_per_sequence()[i]
        data = [self._dataset()[i] for i in range(start, start + seq_len)]
        images = torch.stack([self._process_rgb_image(x) for x in data], 0).permute(0, 2, 3, 1).numpy()
        images = (images * 255.).astype(np.uint8)
        depthmaps = torch.stack([x.depth_map * (x.fg_probability > 0.5).float() for x in data], 0).permute(0, 2, 3, 1).numpy()
        world_to_camera_to_cameras_matrices = torch.cat([x.camera.get_world_to_view_transform().get_matrix().permute(0, 2, 1) for x in data], 0).numpy()
        cameras = self.world_to_camera_to_cameras(world_to_camera_to_cameras_matrices)

        output = dict()
        output['cameras'] = cameras
        output['frames'] = images
        output['depthmaps'] = depthmaps
        output['sequence_id'] = data[0].sequence_name
        return output
