import os
from aparse import click
from co3d.challenge.co3d_submission import CO3DSubmission, CO3DTask, CO3DSequenceSet
from co3d.dataset import data_types
import viewformer
from aclick.utils import Literal
from viewformer.evaluate.evaluate_transformer import generate_batch_predictions  # noqa: E402
from viewformer.utils.tensorflow import load_model  # noqa: E402
from viewformer.data.loaders.co3dv2 import CO3Dv2Loader
from viewformer.evaluate.evaluate_transformer import generate_batch_predictions  # noqa: E402
import argparse
from typing import List
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache()


@click.command('evaluate-co3d')
def main(
        dataset_root: str,
        output: str = "co3d_submission",
        split: Literal["dev", "test"] = "dev",
        codebook_model: str = "co3dv2-all-codebook-th",
        transformer_model: str = "co3dv2-all-noloc-transformer-tf"):
    transformer_model = load_model(transformer_model)
    codebook_model = load_model(codebook_model)
    task = CO3DTask.FEW_VIEW
    sequence_set = getattr(CO3DSequenceSet, split.upper())

    submission = CO3DSubmission(
        task=task,
        sequence_set=sequence_set,
        output_folder=output,
        dataset_root=dataset_root,
    )
    eval_batches_map = submission.get_eval_batches_map()
    class Loader(CO3Dv2Loader):
        @cache
        def _dataset(self):
            return []
        
    loader = Loader(path=dataset_root, image_size=128, split="test")

    def load_image(image_path):
        image_path = os.path.join(loader.path, image_path)
        image = np.asarray(Image.open(image_path)) 
        return image

    def make_batch(data):
        images = [loader._load_image(x.image.path) for x in data]
        masks = [loader._load_image(x.mask.path) for x in data]
        depths = [load_image(x.depth.path) for x in data]
        frames = np.stack([loader._process_rgb_image(image, mask) for image, mask in zip(images, masks)], 0)
        cameras = np.stack([
            loader.world_to_camera_matrix_to_cameras(x.viewpoint.R, x.viewpoint.T) for x in data
        ], 0)

        output = dict()
        output['cameras'] = cameras
        output['frames'] = frames
        output['sequence_id'] = data[0].sequence_name
        output['depths'] = depths
        return output


    def predict_new_view(batch):
        images = batch["frames"].astype(np.float32) / 255.
        cameras = batch["cameras"].astype(np.float32)
        images = np.concatenate((images[1:], images[:1]), 0)[np.newaxis, ...]
        cameras = np.concatenate((cameras[1:], cameras[:1]), 0)[np.newaxis, ...]
        preds = generate_batch_predictions(transformer_model, codebook_model, images, cameras)
        return preds['generated_images'].numpy().astype(np.float32) / 255.

    # iterate over evaluation subsets and categories
    num_eval_batches = sum(map(len, eval_batches_map.values()))
    with tqdm(total=num_eval_batches) as progress:
        for (category, subset_name), eval_batches in eval_batches_map.items():
            category_frame_annotations = data_types.load_dataclass_jgzip(
                f"{loader.path}/{category}/frame_annotations.jgz", List[data_types.FrameAnnotation]
            )
            frame_annotation_map = {(x.sequence_name, x.frame_number): x for x in category_frame_annotations}

            # iterate over all evaluation examples of a given category and subset
            for eval_batch in eval_batches:
                # parse the evaluation sequence name and target frame number from eval_batch
                sequence_name, frame_number = eval_batch[0][:2]

                # `predict_new_view` is a user-defined function which generates
                # the test view (corresponding to the first element of the eval batch)
                prediction_batch = make_batch([
                    frame_annotation_map[(x, y)] for x, y, _ in eval_batch
                ])
                depth = prediction_batch["depths"][-1][None]
                is1, is2 = depth.shape[1:]
                images = predict_new_view(prediction_batch)
                th_images = torch.from_numpy(images).permute(0, 3,  1, 2)
                th_images = torch.nn.functional.interpolate(th_images, (is1, is2), mode='bilinear', align_corners=False)
                th_images = th_images.clamp_(0, 1)[-1] # .permute(0, 2, 3, 1)[-1]
                image = th_images.numpy()
                image, mask = image[:3], image[3:]
                mask = (mask > 0.5).astype(image.dtype)

                # add the render to the submission
                submission.add_result(
                    category=category,
                    subset_name=subset_name,
                    sequence_name=sequence_name,
                    frame_number=frame_number,
                    image=image,
                    mask=mask,
                    depth=depth,
                )
                progress.update()
