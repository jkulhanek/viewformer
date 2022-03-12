import os
import json
import random
from aparse import click
from typing import List, Optional
import numpy as np
import tensorflow as tf
import tqdm
from viewformer.utils.tensorflow import load_model
from viewformer.data.loaders import SevenScenesLoader
from viewformer.data.loaders.sevenscenes import ALL_SCENES
from viewformer.evaluate.evaluate_transformer_multictx import generate_batch_predictions, build_store_predictions, MultiContextEvaluator, print_metrics
from viewformer.evaluate.evaluate_sevenscenes import SceneLookup


@click.command('evaluate-sevenscenes-multictx')
def main(path: str,
         transformer_model: str,
         codebook_model: str,
         job_dir: str,
         batch_size: int,
         scenes: List[str] = None,
         num_eval_sequences: Optional[int] = 100,
         store_ctx: bool = True,
         num_store_images: int = 100):
    if scenes is None:
        scenes = ALL_SCENES
    codebook_model = load_model(codebook_model)
    all_results = dict()
    model = None
    for scene in scenes:
        scene_lookup = SceneLookup(path, scene, 128)
        if model is None or transformer_model.format(scene=scene) != transformer_model:
            model = load_model(transformer_model.format(scene=scene))

        def build_batch(batch):
            gt_frames = batch['frames']
            gt_cameras = batch['cameras']
            ctx = random.sample(scene_lookup.files, 19)
            ctx_cameras, ctx_frames = tuple(np.stack(y, 0) for y in zip(*(scene_lookup[x] for x in ctx)))
            cameras = np.concatenate((ctx_cameras, gt_cameras), 0)[np.newaxis, ...]
            frames = np.concatenate((ctx_frames, gt_frames), 0)[np.newaxis, ...]
            return tf.convert_to_tensor(cameras), tf.convert_to_tensor(frames)

        store_predictions = build_store_predictions(os.path.join(job_dir, scene), num_store_images)
        evaluator = MultiContextEvaluator(20, image_size=128)
        test_loader = SevenScenesLoader(path=path, split='test',
                                        sequence_size=1,
                                        image_size=128,
                                        scenes=[scene],
                                        _load_file_paths=True)
        random_indices = random.Random(42).sample(list(range(len(test_loader))), min(len(test_loader), num_eval_sequences))
        with tqdm.tqdm(total=len(random_indices), desc=f'evaluating {scene}') as progress:
            for index in tqdm.tqdm(random_indices):
                cameras, frames = build_batch(test_loader[index])
                batch_prediction = generate_batch_predictions(model, codebook_model, frames, cameras)
                evaluator.update_state(**batch_prediction)
                if store_ctx:
                    batch_prediction['ctx'] = frames[:, :-1]
                store_predictions(**batch_prediction)
                progress.set_postfix(evaluator.get_progress_bar_info())
                progress.update()
        result = evaluator.result()
        all_results[scene] = result
        print(f'Results on {scene}:')
        print_metrics(result)
        os.makedirs(os.path.join(job_dir, scene), exist_ok=True)
        with open(os.path.join(job_dir, scene, 'results.json'), 'w+') as f:
            json.dump(result, f)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, 'results.json'), 'w+') as f:
        json.dump(all_results, f)


if __name__ == '__main__':
    main()
