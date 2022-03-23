import os
import json
import random
from collections import defaultdict, OrderedDict
from aparse import Literal
from aparse import click
from typing import List, Optional
import numpy as np
import tensorflow as tf
import tqdm
from viewformer.utils.geometry import quaternion_average
from viewformer.data.loaders import SevenScenesLoader
from viewformer.data.loaders.sevenscenes import ALL_SCENES
from viewformer.utils.metrics import CameraOrientationError, CameraPositionError, CameraOrientationMedian, CameraPositionMedian
from viewformer.utils import geometry_tf as geometry


class Evaluator:
    def __init__(self):
        self._localization_metrics = [CameraOrientationError('loc-angle'),
                                      CameraPositionError('loc-dist'),
                                      CameraOrientationMedian('loc-angle-med'),
                                      CameraPositionMedian('loc-dist-med')]

    def update_with_camera(self, ground_truth_cameras, generated_cameras):
        for metric in self._localization_metrics:
            metric.update_state(ground_truth_cameras, generated_cameras)

    def update_state(self, ground_truth_cameras, generated_cameras):
        self.update_with_camera(ground_truth_cameras, generated_cameras)

    def get_progress_bar_info(self):
        return OrderedDict([
            ('cam_loc', float(next((x for x in self._localization_metrics if x.name == 'loc-dist')).result())),
            ('cam_ang', float(next((x for x in self._localization_metrics if x.name == 'loc-angle')).result()))])

    def result(self):
        return OrderedDict((
            (m.name, float(m.result()))
            for m in self._localization_metrics))


def compute_camera_distances(db_cameras, camera, position):
    assert camera.shape == (1, 7,)
    if position:
        return tf.norm(db_cameras[..., :3] - camera[..., :3], axis=-1)
    else:
        x1 = geometry.quaternion_normalize(db_cameras[..., 3:])
        x2 = geometry.quaternion_normalize(camera[..., 3:])
        diff = geometry.quaternion_multiply(x1, geometry.quaternion_conjugate(x2))
        return 2 * tf.asin(tf.linalg.norm(diff[..., 1:], axis=-1))


class SceneLookup:
    def __init__(self, path, scene, image_size=None):
        self.path = path
        self.scene = scene
        loader = SevenScenesLoader(path=path, split='train', scenes=[scene], image_size=image_size, _load_file_paths=True)
        cameras = []
        self.files = []
        for batch in loader:
            cameras.append(batch['cameras'])
            self.files.extend((x + '.color.png' for x in batch['frames_files']))
        self.cameras = np.concatenate(cameras, 0)
        self._lookup = {x: i for i, x in enumerate(self.files)}

    def __getitem__(self, name):
        idx = self._lookup[name]
        return self.cameras[idx]

    def __len__(self):
        return len(self.files)


def load_image_match_map(image_match_map_filepath):
    top_map = defaultdict(list)
    with open(image_match_map_filepath, 'r') as f:
        for line in f:
            fr, to = line.strip('\n\r').split()
            top_map[fr].append(to)
    return top_map


def generate_batch_predictions_baseline(cameras, baseline):
    ctx_cameras, gt_cameras = cameras[0, :-1], cameras[:, -1]
    if baseline == 'mean':
        xyz, quat = tf.split(ctx_cameras, (3, 4), -1)
        xyz = tf.reduce_mean(xyz, 0)
        quat = tf.convert_to_tensor(quaternion_average(quat.numpy()))
        pred_cameras = tf.concat((xyz, quat), -1)
    else:
        assert baseline in ['position_oracle', 'orientation_oracle']
        pred = tf.argmin(compute_camera_distances(ctx_cameras, gt_cameras, baseline == 'position_oracle'), 0)
        pred_cameras = ctx_cameras[pred]
    return dict(
        ground_truth_cameras=gt_cameras,
        generated_cameras=tf.expand_dims(pred_cameras, 0))


@click.command('evaluate-sevenscenes-baseline')
def main(path: str,
         job_dir: str,
         image_match_map: Optional[str] = None,
         scenes: List[str] = None,
         sequence_size: Optional[int] = None,
         num_eval_sequences: Optional[int] = 1000,
         top_n_matched_images: int = 0,
         baseline: Literal['orientation_oracle', 'position_oracle', 'mean'] = 'position_oracle'):
    if scenes is None:
        scenes = ALL_SCENES
    if top_n_matched_images > 0:
        assert image_match_map is not None

    all_results = dict()
    for scene in scenes:
        if image_match_map is not None:
            top_match_map = load_image_match_map(image_match_map.format(scene=scene))
        scene_lookup = SceneLookup(path, scene, 128)
        # db_cameras = tf.convert_to_tensor(scene_lookup.cameras)

        def build_batch(batch):
            gt_cameras = batch['cameras']
            ctx = []
            if image_match_map is not None:
                ctx = top_match_map[batch['frames_files'][0] + '.color.png'][:top_n_matched_images]
            ctx += random.sample(scene_lookup.files, 19 - len(ctx))
            ctx_cameras = np.stack([scene_lookup[x] for x in ctx], 0)
            cameras = np.concatenate((ctx_cameras, gt_cameras), 0)[np.newaxis, ...]
            return tf.convert_to_tensor(cameras)

        evaluator = Evaluator()
        test_loader = SevenScenesLoader(path=path, split='test',
                                        sequence_size=1,
                                        image_size=128,
                                        scenes=[scene],
                                        _load_file_paths=True)
        random_indices = random.Random(42).sample(list(range(len(test_loader))), min(len(test_loader), num_eval_sequences))
        with tqdm.tqdm(total=len(random_indices), desc=f'evaluating {scene}') as progress:
            for index in tqdm.tqdm(random_indices):
                cameras = build_batch(test_loader[index])
                batch_prediction = generate_batch_predictions_baseline(cameras, baseline)
                evaluator.update_state(**batch_prediction)
                progress.set_postfix(evaluator.get_progress_bar_info())
                progress.update()
        result = evaluator.result()
        all_results[scene] = result
        print('Results:')
        for m, val in result.items():
            print(f'    {m}: {val:.6f}')
        os.makedirs(os.path.join(job_dir, scene), exist_ok=True)
        with open(os.path.join(job_dir, scene, 'results.json'), 'w+') as f:
            json.dump(result, f)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, 'results.json'), 'w+') as f:
        json.dump(all_results, f)


if __name__ == '__main__':
    main()
