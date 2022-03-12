# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from aparse import click
import copy
import os

from typing import Optional, List

import torch
from tqdm import tqdm
import numpy as np
import lpips
import json

import tensorflow as tf
# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from viewformer.evaluate.evaluate_transformer import generate_batch_predictions  # noqa: E402
from viewformer.utils.tensorflow import load_model  # noqa: E402
from viewformer.data.loaders import CO3DLoader  # noqa: E402


@click.command('evaluate-co3d')
def main(path: str, codebook_model: str, transformer_model: str, job_dir: str = '.', categories: List[str] = None):
    transformer_model = load_model(transformer_model)
    codebook_model = load_model(codebook_model)

    _loader = CO3DLoader(path=path, split='test', mask_images=True)
    with _loader.use_dataset_zoo():
        from dataset.dataset_zoo import CO3D_CATEGORIES, dataset_zoo
        from dataset.dataloader_zoo import dataloader_zoo
        from eval_demo import _print_aggregate_results, _get_all_source_cameras, aggregate_nvs_results
        from evaluation.evaluate_new_view_synthesis import NewViewSynthesisPrediction

        from evaluation.evaluate_new_view_synthesis import (
            eval_batch,
            summarize_nvs_eval_results,
            pretty_print_nvs_metrics,
        )

        """
        Evaluates new view synthesis metrics for multisequence tasks for several categories.
        The evaluation is conducted on the same data as in [1] and, hence, the results
        are directly comparable to the numbers reported in [1].
        References:
            [1] J. Reizenstein, R. Shapovalov, P. Henzler, L. Sbordone,
                    P. Labatut, D. Novotny:
                Common Objects in 3D: Large-Scale Learning
                    and Evaluation of Real-life 3D Category Reconstruction
        """

        def generate_predictions(loader, transformer_model, codebook_model, frame_data):
            images = loader._process_rgb_image(frame_data).permute(0, 2, 3, 1).numpy()
            images = (images * 255.).astype(np.uint8)
            world_to_camera_to_cameras_matrices = frame_data.camera.get_world_to_view_transform().get_matrix().permute(0, 2, 1).numpy()
            cameras = loader.world_to_camera_to_cameras(world_to_camera_to_cameras_matrices)
            images = np.concatenate((images[1:], images[:1]), 0)[np.newaxis, ...]
            cameras = np.concatenate((cameras[1:], cameras[:1]), 0)[np.newaxis, ...]
            preds = generate_batch_predictions(transformer_model, codebook_model, images, cameras)
            predicted_images = torch.from_numpy(preds['generated_images'].numpy()).to(torch.float32).permute(0, 3, 1, 2) / 255.
            nvs_prediction = NewViewSynthesisPrediction(
                image_render=predicted_images,
                mask_render=torch.ones((1, 1) + predicted_images.shape[-2:], dtype=torch.float32),
                depth_render=torch.zeros((1, 1) + predicted_images.shape[-2:], dtype=torch.float32))
            return nvs_prediction

        def evaluate_for_category(
            category: str = "apple",
            bg_color: str = "black",
            single_sequence_id: Optional[int] = None,
            num_workers: int = 16,
        ):
            """
            Evaluates new view synthesis metrics of a simple depth-based image rendering
            (DBIR) model for a given task, category, and sequence (in case task=='singlesequence').
            Args:
                category: Object category.
                bg_color: Background color of the renders.
                single_sequence_id: The ID of the evaluiation sequence for the singlesequence task.
                num_workers: The number of workers for the employed dataloaders.
            Returns:
                category_result: A dictionary of quantitative metrics.
            """

            task = "multisequence"
            torch.manual_seed(42)

            datasets = dataset_zoo(
                dataset_root=path,
                category=category,
                assert_single_seq=task == "singlesequence",
                dataset_name=f"co3d_{task}",
                test_on_train=False,
                load_point_clouds=True,
                test_restrict_sequence_id=single_sequence_id,
            )

            dataloaders = dataloader_zoo(
                datasets,
                dataset_name=f"co3d_{task}",
            )

            test_dataset = datasets["test"]
            test_dataloader = dataloaders["test"]

            if task == "singlesequence":
                # all_source_cameras are needed for evaluation of the
                # target camera difficulty
                sequence_name = test_dataset.frame_annots[0]["frame_annotation"].seqence_name
                all_source_cameras = _get_all_source_cameras(
                    test_dataset, sequence_name, num_workers=num_workers
                )
            else:
                all_source_cameras = None

            # init the simple DBIR model

            # init the lpips model for eval
            lpips_model = lpips.LPIPS(net="vgg")
            lpips_model = lpips_model.cuda()

            per_batch_eval_results = []
            for bi, frame_data in enumerate(tqdm(test_dataloader)):
                preds = generate_predictions(_loader, transformer_model, codebook_model, frame_data)
                nvs_prediction = copy.deepcopy(preds)
                per_batch_eval_results.append(
                    eval_batch(
                        frame_data,
                        nvs_prediction,
                        bg_color=bg_color,
                        lpips_model=lpips_model,
                        source_cameras=all_source_cameras,
                    )
                )

            category_result_flat, category_result = summarize_nvs_eval_results(
                per_batch_eval_results, task
            )

            return category_result["results"]

        task_results = {}
        task = 'multisequence'
        task_results[task] = []
        if categories is None:
            categories = CO3D_CATEGORIES[: (20 if task == "singlesequence" else 10)]
        for category in categories:
            for single_sequence_id in (0, 1) if task == "singlesequence" else (None,):
                category_result = evaluate_for_category(
                    category, single_sequence_id=single_sequence_id
                )
                print("")
                print(
                    f"Results for task={task}; category={category};" + (
                        f" sequence={single_sequence_id}:"
                        if single_sequence_id is not None
                        else ":"
                    )
                )
                pretty_print_nvs_metrics(category_result)
                print("")

                task_results[task].append(category_result)
            _print_aggregate_results(task, task_results)

        for task in task_results:
            _print_aggregate_results(task, task_results)
        with open(os.path.join(job_dir, 'results-co3d.json'), 'w+') as f:
            json.dump(task_results, f)

        mp = {f"{x['subset']}_{x['subsubset']}": x['metrics'] for x in aggregate_nvs_results(task_results['multisequence'])}
        with open(os.path.join(job_dir, 'results.json'), 'w+') as f:
            json.dump(mp, f)


if __name__ == "__main__":
    main()
