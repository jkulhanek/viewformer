import logging
from itertools import groupby, accumulate
from functools import partial
from typing import List, Union, Optional
import copy
import os
import io
import json
import numpy as np
from PIL import Image
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


class CO3Dv2Loader:
    _images_per_scene = dict()
    _custom_resize = True

    def __init__(self, path: str, split: str = None, categories: List[str] = None, sequence_set: str = "fewview_train", image_size: Optional[int] = None):
        assert split in ['test', 'train', 'val']
        assert image_size is not None
        self._installed = False
        self.install()

        if categories is None:
            categories = CO3D_CATEGORIES
        self.categories = categories
        self.split = split
        self.path = path
        self.sequence_set = sequence_set
        self.image_size = image_size

    def install(self):
        if not self._installed:
            if not os.path.exists(os.path.expanduser('~/.cache/viewformer/co3dv2')):
                import urllib.request
                import zipfile
                import shutil
                os.makedirs(os.path.expanduser('~/.cache/viewformer'), exist_ok=True)
                with urllib.request.urlopen('https://github.com/facebookresearch/co3d/archive/c5c6c8ab1b39c70c4661581b84e0b2a5dfab1f64.zip') as f:
                    with io.BytesIO(f.read()) as bytes_io:
                        f.close()
                        with zipfile.ZipFile(bytes_io, 'r') as archive:
                            archive.extractall(os.path.expanduser('~/.cache/viewformer'))
                shutil.move(os.path.expanduser('~/.cache/viewformer/co3d-c5c6c8ab1b39c70c4661581b84e0b2a5dfab1f64'), os.path.expanduser('~/.cache/viewformer/co3dv2'))
                logging.info(f'CO3Dv2 installed to "{os.path.expanduser("~/.cache/viewformer/co3dv2")}"')

            def use_co3d_data_types():
                class ctx:
                    def __enter__(self):
                        import sys
                        sys.path.insert(0, os.path.expanduser('~/.cache/viewformer/co3dv2'))
                        from co3d.dataset import data_types
                        return data_types

                    def __exit__(self, *args, **kwargs):
                        import sys
                        sys.path.remove(os.path.expanduser('~/.cache/viewformer/co3dv2'))
                return ctx()

            self.use_co3d_data_types = use_co3d_data_types
            self._installed = True

    @staticmethod
    def world_to_camera_matrix_to_cameras(R, position):
        R = np.array(R)
        position = np.array(position)

        # We should invert world_to_camera to camera_to_world
        # But the pytorch3D uses right multiplication x @ worldmat
        # Therefore the rotation matrix is already transposed.

        # Change coordinate system
        # in PyTorch3d, z points to screen, y up and x to left
        # in our dataset, z points to screen, y down and x to right
        R[:2, :] *= -1
        position[:2] *= -1
        quaternion = rotation_matrix_to_quaternion(R)
        quaternion = quaternion_normalize(quaternion)
        return np.concatenate([position, quaternion], -1)

    @cache
    def _dataset(self):
        frame_annotations = []
        with self.use_co3d_data_types() as data_types:
            for i, c in enumerate(self.categories):
                print(f"Loading CO3D category {c} [{i+1}/{len(self.categories)}].")
                category_frame_annotations = data_types.load_dataclass_jgzip(
                    f"{self.path}/{c}/frame_annotations.jgz", List[data_types.FrameAnnotation]
                )
                frame_annotation_map = {(x.sequence_name, x.frame_number): x for x in category_frame_annotations}
                json_path = f"{self.path}/{c}/set_lists.json"
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        try:
                            data_list = json.load(f)
                        except Exception as e:
                            print(f"Invalid file {json_path}")
                            raise e
                        data_list = data_list[self.sequence_set]
                else:
                    json_path = f"{self.path}/{c}/set_lists/set_lists_{self.sequence_set}.json"
                    with open(json_path, "r") as f:
                        try:
                            data_list = json.load(f)
                        except Exception as e:
                            print(f"Invalid file {json_path}")
                            raise e

                        data_list = data_list[self.split]
                for seq_name, frame_num, _ in data_list:
                    frame_annotations.append(frame_annotation_map[(seq_name, frame_num)])
        return frame_annotations

    def __len__(self):
        return len(self.num_images_per_sequence())

    @cache
    def num_images_per_sequence(self):
        return [sum(1 for _ in x) for _, x in groupby(self._dataset() , key=lambda x: x.sequence_name)]

    @cache
    def _cum_images_per_sequence(self):
        return [0] + list(accumulate(self.num_images_per_sequence()[:-1]))

    def get_intrinsics(self):
        # Return (image_width, image_height, f_x, f_y, c_x, c_y)
        return (800, 800, 400, 400, 400, 400)

    def _process_rgb_image(self, image, mask):
        # threshold the masks to make ground truth binary masks
        mask_fg = mask > 127
        image_rgb_masked = np.where(mask_fg[..., None], image, np.zeros_like(image))
        return np.concatenate([image_rgb_masked, mask[..., None]], -1)

    def _load_image(self, image_path):
        image_path = os.path.join(self.path, image_path)
        try:
            image = np.asarray(Image.open(image_path).resize((self.image_size, self.image_size))) 
        except Exception as e:
            print(f"Failed to load image {image_path}")
            raise e
        return image


    def __getitem__(self, i):
        start = self._cum_images_per_sequence()[i]
        seq_len = self.num_images_per_sequence()[i]
        data = [self._dataset()[i] for i in range(start, start + seq_len)]

        images = [self._load_image(x.image.path) for x in data]
        masks = [self._load_image(x.mask.path) for x in data]
        frames = np.stack([self._process_rgb_image(image, mask) for image, mask in zip(images, masks)], 0)
        cameras = np.stack([
            self.world_to_camera_matrix_to_cameras(x.viewpoint.R, x.viewpoint.T) for x in data
        ], 0)

        output = dict()
        output['cameras'] = cameras
        output['frames'] = frames
        output['sequence_id'] = data[0].sequence_name
        return output
