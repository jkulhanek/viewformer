import os
import json
import random
from collections import defaultdict
from aparse import Literal
from aparse import click
from typing import List, Optional
import numpy as np
import tensorflow as tf
import tqdm
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import SevenScenesLoader
from viewformer.data.loaders.sevenscenes import ALL_SCENES
from viewformer.evaluate.evaluate_transformer import generate_batch_predictions, Evaluator, build_store_predictions
from viewformer.utils.geometry_tf import quaternion_multiply
from viewformer.evaluate.evaluate_transformer import to_relative_cameras, resize_tf, from_relative_cameras, normalize_cameras
from viewformer.utils import geometry_tf as geometry


def generate_other_viewpoints(camera):
    max_offset = 1.  # maximum of 1 meter
    max_rad_offset = 0.3  # maximum rotation difference of 0.3 rad

    pos_offset = tf.random.uniform(tf.shape(camera[..., :3]), -1, 1, dtype=camera.dtype)
    pos_offset = tf.math.l2_normalize(pos_offset)
    quaternion_axis = tf.random.uniform(tf.shape(camera[..., :3]), -1, 1, dtype=camera.dtype)
    quaternion_axis = tf.math.l2_normalize(quaternion_axis)
    pos_offset *= tf.random.uniform(tf.shape(camera[..., :1]), 0, max_offset, dtype=camera.dtype)
    angle = tf.random.uniform(tf.shape(camera[..., :1]), 0, max_rad_offset, dtype=camera.dtype)
    quaternion_rot = tf.concat((tf.cos(angle / 2), tf.sin(angle / 2) * quaternion_axis), -1)
    xyz, quaternion = tf.split(camera, (3, 4), -1)
    new_pose = tf.concat((pos_offset + xyz, geometry.quaternion_normalize(quaternion_multiply(quaternion_rot, quaternion))), -1)
    return new_pose


def compute_camera_distances(db_cameras, camera):
    assert camera.shape == (1, 7,)
    pos_distances = tf.norm(db_cameras[..., :3] - camera[..., :3], axis=-1)
    x1 = geometry.quaternion_normalize(db_cameras[..., 3:])
    x2 = geometry.quaternion_normalize(camera[..., 3:])
    diff = geometry.quaternion_multiply(x1, geometry.quaternion_conjugate(x2))
    quat_distances = 2 * tf.asin(tf.linalg.norm(diff[..., 1:], axis=-1))

    # Coefficient chosen arbitrary
    return pos_distances * 0.3 + quat_distances


class SceneLookup:
    def __init__(self, path, scene, image_size=None):
        self.path = path
        self.scene = scene
        loader = SevenScenesLoader(path=path, split='train', scenes=[scene], image_size=image_size, _load_file_paths=True)
        cameras = []
        self.files = []
        self.image_loaders = []
        for batch in loader:
            cameras.append(batch['cameras'])
            self.files.extend((x + '.color.png' for x in batch['frames_files']))
            self.image_loaders.extend(batch['frames'])
        self.cameras = np.concatenate(cameras, 0)
        self._lookup = {x: i for i, x in enumerate(self.files)}

    def __getitem__(self, name):
        idx = self._lookup[name]
        return self.cameras[idx], self.image_loaders[idx]

    def __len__(self):
        return len(self.files)


def load_image_match_map(image_match_map_filepath):
    top_map = defaultdict(list)
    with open(image_match_map_filepath, 'r') as f:
        for line in f:
            fr, to = line.strip('\n\r').split()
            top_map[fr].append(to)
    return top_map


def generate_batch_predictions_using_generated_images(transformer_model, codebook_model, images, cameras, num_gen_ctx=5):
    ground_truth_cameras = cameras[:, -1]

    if transformer_model.config.augment_poses == 'relative':
        # Rotate poses for relative augment
        cameras, transform = to_relative_cameras(cameras)
    cameras = normalize_cameras(cameras)

    with tf.name_scope('encode'):
        def encode(images):
            fimages = resize_tf(images, codebook_model.config.image_size)
            fimages = tf.image.convert_image_dtype(fimages, tf.float32)
            fimages = fimages * 2 - 1
            codes = codebook_model.encode(fimages)[-1]  # [N, H', W']
            codes = tf.cast(codes, dtype=tf.int32)
            return codes

        # codes = tf.ragged.map_flat_values(encode, codes)
        batch_size, seq_len, *im_dim = tf.unstack(tf.shape(images), 5)
        code_len = transformer_model.config.token_image_size
        codes = tf.reshape(encode(tf.reshape(images, [batch_size * seq_len] + list(im_dim))), [batch_size, seq_len, code_len, code_len])

    # Generate first camera estimates
    output = transformer_model(dict(input_ids=codes, poses=cameras[:, :-1]), training=False)
    generated_cameras = transformer_model.reduce_cameras(output['pose_prediction'][:, -1:], -2)

    # Generate new images from altered poses
    new_cameras = generate_other_viewpoints(tf.tile(generated_cameras[:, -1:], (num_gen_ctx, 1, 1)))
    new_cameras = normalize_cameras(new_cameras)
    with tf.name_scope('transformer_generate_images'):
        image_generation_input_ids = tf.concat([codes[:, :-1], tf.fill(tf.shape(codes[:, :1]),
                                                                       tf.constant(transformer_model.mask_token, dtype=codes.dtype))], 1)
        output = transformer_model(dict(
            input_ids=tf.tile(image_generation_input_ids, (num_gen_ctx, 1, 1, 1)),
            poses=tf.concat((
                tf.tile(cameras[:, :-1], (num_gen_ctx, 1, 1)),
                new_cameras
            ), 1)
        ), training=False)
        new_codes = tf.cast(tf.argmax(output['logits'], -1)[:, -1], tf.int32)
    codes = tf.concat((
        codes[:, :-num_gen_ctx],
        tf.expand_dims(new_codes, 0)
    ), 1)
    cameras = tf.concat((
        cameras[:, :-num_gen_ctx],
        tf.reshape(new_cameras, (1, num_gen_ctx, -1))
    ), 1)

    # Generate final images
    with tf.name_scope('transformer_generate_images'):
        image_generation_input_ids = tf.concat([codes[:, :-1], tf.fill(tf.shape(codes[:, :1]),
                                                                       tf.constant(transformer_model.mask_token, dtype=codes.dtype))], 1)
        output = transformer_model(dict(input_ids=image_generation_input_ids, poses=cameras), training=False)
        generated_codes = tf.argmax(output['logits'], -1)[:, -1]

    # Decode images
    with tf.name_scope('decode_images'):
        generated_images = codebook_model.decode_code(generated_codes)
        generated_images = tf.clip_by_value(generated_images, -1, 1)
        generated_images = tf.image.convert_image_dtype(generated_images / 2 + 0.5, tf.uint8)

    # Generate pose estimates
    output = transformer_model(dict(input_ids=codes, poses=cameras[:, :-1]), training=False)
    generated_cameras = transformer_model.reduce_cameras(output['pose_prediction'][:, -1:], -2)

    # Transform generated cameras to original space
    if transformer_model.config.augment_poses == 'relative':
        generated_cameras = from_relative_cameras(generated_cameras, transform)

    return dict(
        ground_truth_images=images[:, -1],
        generated_images=generated_images,
        ground_truth_cameras=ground_truth_cameras,
        generated_cameras=generated_cameras[:, -1])


def generate_batch_predictions_using_pose_refinement(scene_lookup, db_cameras, transformer_model, codebook_model, images, cameras, num_gen_ctx=9):
    gt_cameras, gt_frames = cameras[:, -1], images[:, -1]

    if transformer_model.config.augment_poses == 'relative':
        # Rotate poses for relative augment
        cameras, transform = to_relative_cameras(cameras)
    cameras = normalize_cameras(cameras)

    with tf.name_scope('encode'):
        def encode(images):
            fimages = resize_tf(images, codebook_model.config.image_size)
            fimages = tf.image.convert_image_dtype(fimages, tf.float32)
            fimages = fimages * 2 - 1
            codes = codebook_model.encode(fimages)[-1]  # [N, H', W']
            codes = tf.cast(codes, dtype=tf.int32)
            return codes

        # codes = tf.ragged.map_flat_values(encode, codes)
        batch_size, seq_len, *im_dim = tf.unstack(tf.shape(images), 5)
        code_len = transformer_model.config.token_image_size
        codes = tf.reshape(encode(tf.reshape(images, [batch_size * seq_len] + list(im_dim))), [batch_size, seq_len, code_len, code_len])

    # Generate first camera estimates
    output = transformer_model(dict(input_ids=codes, poses=cameras[:, :-1]), training=False)
    generated_cameras = transformer_model.reduce_cameras(output['pose_prediction'][:, -1:], -2)

    # Transform generated cameras to original space
    if transformer_model.config.augment_poses == 'relative':
        generated_cameras = from_relative_cameras(generated_cameras, transform)

    # Now, we find the closest matching poses in the database
    distances = compute_camera_distances(db_cameras, generated_cameras[:, 0, :])
    top_files = tf.argsort(distances, axis=-1, direction='ASCENDING')[:num_gen_ctx]
    files = [scene_lookup.files[x] for x in top_files]
    files += random.sample(scene_lookup.files, 19 - len(files))
    ctx_cameras, ctx_frames = tuple(np.stack(y, 0) for y in zip(*(scene_lookup[x] for x in files)))
    ctx_cameras, ctx_frames = tf.convert_to_tensor(ctx_cameras), tf.convert_to_tensor(ctx_frames)

    cameras = tf.expand_dims(tf.concat((ctx_cameras, gt_cameras), 0), 0)
    frames = tf.expand_dims(tf.concat((ctx_frames, gt_frames), 0), 0)
    return generate_batch_predictions(transformer_model, codebook_model, frames, cameras)


@click.command('evaluate-sevenscenes')
def main(path: str,
         transformer_model: str,
         codebook_model: str,
         job_dir: str,
         batch_size: int,
         image_match_map: Optional[str] = None,
         scenes: List[str] = None,
         sequence_size: Optional[int] = None,
         num_eval_sequences: Optional[int] = None,
         num_store_images: int = 100,
         top_n_matched_images: int = 0,
         num_gen_ctx: int = 0,
         generation_procedure: Literal['standard', 'generated_images', 'pose_refinement'] = 'standard',
         pose_multiplier: Optional[float] = None):
    if scenes is None:
        scenes = ALL_SCENES
    if top_n_matched_images > 0:
        assert image_match_map is not None

    codebook_model = load_model(codebook_model)
    all_results = dict()
    model = None
    for scene in scenes:
        if image_match_map is not None:
            top_match_map = load_image_match_map(image_match_map.format(scene=scene))
        scene_lookup = SceneLookup(path, scene, 128)
        db_cameras = tf.convert_to_tensor(scene_lookup.cameras)
        if model is None or transformer_model.format(scene=scene) != transformer_model:
            model_kwargs = dict()
            if pose_multiplier is not None:
                model_kwargs['pose_multiplier'] = pose_multiplier
            model = load_model(transformer_model.format(scene=scene), **model_kwargs)

        def build_batch(batch):
            gt_frames = batch['frames']
            gt_cameras = batch['cameras']
            ctx = []
            if image_match_map is not None:
                ctx = top_match_map[batch['frames_files'][0] + '.color.png'][:top_n_matched_images]
            ctx += random.sample(scene_lookup.files, 19 - len(ctx))
            ctx_cameras, ctx_frames = tuple(np.stack(y, 0) for y in zip(*(scene_lookup[x] for x in ctx)))
            cameras = np.concatenate((ctx_cameras, gt_cameras), 0)[np.newaxis, ...]
            frames = np.concatenate((ctx_frames, gt_frames), 0)[np.newaxis, ...]
            return tf.convert_to_tensor(cameras), tf.convert_to_tensor(frames)

        store_predictions = build_store_predictions(os.path.join(job_dir, scene), num_store_images)
        evaluator = Evaluator(image_size=128)
        test_loader = SevenScenesLoader(path=path, split='test',
                                        sequence_size=1,
                                        image_size=128,
                                        scenes=[scene],
                                        _load_file_paths=True)
        if num_eval_sequences is not None and num_eval_sequences > 0:
            random_indices = random.Random(42).sample(list(range(len(test_loader))), min(len(test_loader), num_eval_sequences))
        else:
            random_indices = list(range(len(test_loader)))
            random.Random(42).shuffle(random_indices)
        with tqdm.tqdm(total=len(random_indices), desc=f'evaluating {scene}') as progress:
            for index in tqdm.tqdm(random_indices):
                cameras, frames = build_batch(test_loader[index])
                if generation_procedure == 'standard':
                    batch_prediction = generate_batch_predictions(model, codebook_model, frames, cameras)
                elif generation_procedure == 'generated_images':
                    batch_prediction = generate_batch_predictions_using_generated_images(model, codebook_model, frames, cameras, num_gen_ctx=num_gen_ctx)
                elif generation_procedure == 'pose_refinement':
                    batch_prediction = generate_batch_predictions_using_pose_refinement(scene_lookup, db_cameras, model, codebook_model,
                                                                                        frames, cameras, num_gen_ctx=num_gen_ctx)
                evaluator.update_state(**batch_prediction)
                store_predictions(**batch_prediction)
                progress.set_postfix(evaluator.get_progress_bar_info())
                progress.update()
        result = evaluator.result()
        all_results[scene] = result
        print(f'Results on {scene}:')
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
