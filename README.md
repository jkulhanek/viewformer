# ViewFormer: NeRF-free Neural Rendering from Few Images Using Transformers
Official implementation of ViewFormer. 
ViewFormer is a NeRF-free neural rendering model based on the transformer architecture.
The model is capable of both novel view synthesis and camera pose estimation.
It is evaluated on previously unseen 3D scenes.


[Paper](https://arxiv.org/pdf/2203.10157.pdf)&nbsp;&nbsp;&nbsp;
[Web](https://jkulhanek.github.io/viewformer)&nbsp;&nbsp;&nbsp;
[Demo](https://colab.research.google.com/github/jkulhanek/viewformer/blob/master/notebooks/viewformer-playground.ipynb)
 
<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg?style=for-the-badge)](https://colab.research.google.com/github/jkulhanek/viewformer/blob/master/notebooks/viewformer-playground.ipynb)

## Getting started
Start by creating a python 3.8 venv. From the activated environment,
you can run the following command in the directory containing `setup.py`:
```bash
pip install -e .
```

## Getting datasets
In this section, we describe how you can prepare the data for training. We assume that you
have your environment ready and you want to store the dataset into `{output path}` directory.

### Shepard-Metzler-Parts-7
Please, first visit [https://github.com/deepmind/gqn-datasets](https://github.com/deepmind/gqn-datasets).
```bash
viewformer-cli dataset generate \
    --loader sm7 \
    --image-size 128 \
    --output {output path}/sm7 \
    --max-sequences-per-shard 2000 \
    --split train

viewformer-cli dataset generate \
    --loader sm7 \
    --image-size 128 \
    --output {output path}/sm7 \
    --max-sequences-per-shard 2000 \
    --split test
```

### InteriorNet
Download the dataset into the directory `{source}` by following the instruction here: [https://interiornet.org/](https://interiornet.org/). Then, proceed as follows:
```bash
viewformer-cli dataset generate \
    --loader interiornet \
    --path {source} \
    --image-size 128  \
    --output {output path}/interiornet \
    --max-sequences-per-shard 50 \
    --shuffle \
    --split train

viewformer-cli dataset generate \
    --loader interiornet \
    --path {source} \
    --image-size 128  \
    --output {output path}/interiornet \
    --max-sequences-per-shard 50 \
    --shuffle \
    --split test
```

### Common Objects in 3D
Download the dataset into the directory `{source}` by following the instruction here: [https://ai.facebook.com/datasets/CO3D-dataset](https://ai.facebook.com/datasets/CO3D-dataset).

Install the following dependencies: `plyfile>=0.7.4 pytorch3d`.
Then, generate the dataset for 10 categories as follows:
```bash
viewformer-cli dataset generate \
    --loader co3d \
    --path {source} \
    --image-size 128  \
    --output {output path}/co3d \
    --max-images-per-shard 6000 \
    --shuffle \
    --categories "plant,teddybear,suitcase,bench,ball,cake,vase,hydrant,apple,donut" \
    --split train

viewformer-cli dataset generate \
    --loader co3d \
    --path {source} \
    --image-size 128  \
    --output {output path}/co3d \
    --max-images-per-shard 6000 \
    --shuffle \
    --categories "plant,teddybear,suitcase,bench,ball,cake,vase,hydrant,apple,donut" \
    --split val
```

Alternatively, generate the full dataset as follows:
```bash
viewformer-cli dataset generate \
    --loader co3d \
    --path {source} \
    --image-size 128  \
    --output {output path}/co3d \
    --max-images-per-shard 6000 \
    --shuffle \
    --split train

viewformer-cli dataset generate \
    --loader co3d \
    --path {source} \
    --image-size 128  \
    --output {output path}/co3d \
    --max-images-per-shard 6000 \
    --shuffle \
    --split val
```

### ShapeNet cars and chairs dataset
Download and extract the SRN datasets into the directory `{source}`. The files can be found here: [https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90](https://drive.google.com/drive/folders/1OkYgeRcIcLOFu1ft5mRODWNQaPJ0ps90).

Then, generate the dataset as follows:
```bash
viewformer-cli dataset generate \
    --loader shapenet \
    --path {source} \
    --image-size 128  \
    --output {output path}/shapenet-{category}/shapenet \
    --categories {category} \
    --max-sequences-per-shard 50 \
    --shuffle \
    --split train

viewformer-cli dataset generate \
    --loader shapenet \
    --path {source} \
    --image-size 128  \
    --output {output path}/shapenet-{category}/shapenet \
    --categories {category} \
    --max-sequences-per-shard 50 \
    --shuffle \
    --split test
```
where {category} is either `cars` or `chairs`.

### Faster preprocessing
In order to make the preprocessing faster, you can add `--shards {process id}/{num processes}` to the command and run multiple instances of the command in multiple processes.


## Training the codebook model
The codebook model training uses the PyTorch framework, but the resulting model can be loaded by both TensorFlow and PyTorch. The training code was also prepared for TensorFlow framework, but in order to get the same results as published in the paper, PyTorch code should be used. To train the codebook model on 8 GPUs, run the following code:
```bash
viewformer-cli train codebook \
    --job-dir . \
    --dataset "{dataset path}" \
    --num-gpus 8 \
    --batch-size 352 \
    --n-embed 1024 \
    --learning-rate 1.584e-3 \
    --total-steps 200000
```
Replace `{dataset path}` by the real dataset path. Note that you can use more than one dataset. In that case, the dataset paths should be separated by a comma. Also, if the size of dataset is not large enough to support sharding, you can reduce the number of data loading workers by using `--num-val-workers` and `--num-workers` arguments. The argument `--job-dir` specifies the path where the resulting model and logs will be stored. You can also use the `--wandb` flag, that enables logging to wandb.

### Finetuning the codebook model
If you want to finetune an existing codebook model, add `--resume-from-checkpoint "{checkpoint path}"` to the command and increase the number of total steps.


## Transforming the dataset into the code representation
Before the transformer model can be trained, the dataset has to be transformed into the code representation. This can be achieved by running the following command (on a single GPU):
```bash
viewformer-cli generate-codes \
    --model "{codebook model checkpoint}" \
    --dataset "{dataset path}" \
    --output "{code dataset path}" \
    --batch-size 64 
```
We assume that the codebook model checkpoint path (ending with `.ckpt`) is `{codebook model checkpoint}` and the original dataset is stored in `{dataset path}`. The resulting dataset will be stored in `{code dataset path}`.

## Training the transformer model
To train the models with the same hyper-parameters as in the paper, run the commands from the following sections based on the target dataset. We assume that the codebook model checkpoint path (ending with `.ckpt`) is `{codebook model checkpoint}` and the associated code dataset is located in `{code dataset path}`. All commands use 8 GPUs (in our case 8 NVIDIA A100 GPUs).

### InteriorNet training
```bash
viewformer-cli train transformer \
    --dataset "{code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --sequence-size 20 \
    --n-loss-skip 4 \
    --batch-size 40 \
    --fp16 \
    --total-steps 200000 \
    --localization-weight 5. \
    --learning-rate 8e-5 \
    --weight-decay 0.01 \
    --job-dir . \
    --pose-multiplier 1.
```
For the variant without localization, use `--localization-weight 0`. Similarly, for the variant without novel view synthesis, use `--image-generation-weight 0`.

### CO3D finetuning
In order to finetune the model for 10 categories, use the following command:
```bash
viewformer-cli train finetune-transformer \
    --dataset "{code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --sequence-size 10 \
    --n-loss-skip 1 \
    --batch-size 80 \
    --fp16 \
    --localization-weight 5 \
    --learning-rate 1e-4 \
    --total-steps 40000 \
    --epochs 40 \
    --weight-decay 0.05 \
    --job-dir . \
    --pose-multiplier 0.05 \
    --checkpoint "{interiornet transformer model checkpoint}"
```
Here `{interiornet transformer model checkpoint}` is the path to the InteriorNet checkpoint (usually ending with `weights.model.099-last`). For the variant without localization, use `--localization-weight 0`.


For all categories and including localization:
```bash
viewformer-cli train finetune-transformer \
    --dataset "{code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --sequence-size 10 \
    --n-loss-skip 1 \
    --batch-size 40 \
    --localization-weight 5 \
    --gradient-clip-val 1. \
    --learning-rate 1e-4 \
    --total-steps 100000 \
    --epochs 100 \
    --weight-decay 0.05 \
    --job-dir . \
    --pose-multiplier 0.05 \
    --checkpoint "{interiornet transformer model checkpoint}"
```
Here `{interiornet transformer model checkpoint}` is the path to the InteriorNet checkpoint (usually ending with `weights.model.099-last`).

For all categories without localization:
```bash
viewformer-cli train finetune-transformer \
    --dataset "{code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --sequence-size 10 \
    --n-loss-skip 1 \
    --batch-size 40 \
    --localization-weight 5 \
    --learning-rate 1e-4 \
    --total-steps 100000 \
    --epochs 100 \
    --weight-decay 0.05 \
    --job-dir . \
    --pose-multiplier 0.05 \
    --checkpoint "{interiornet transformer model checkpoint}"
```
Here `{interiornet transformer model checkpoint}` is the path to the InteriorNet checkpoint (usually ending with `weights.model.099-last`).

### 7-Scenes finetuning
```bash
viewformer-cli train finetune-transformer \
    --dataset "{code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --localization-weight 5 \
    --pose-multiplier 5. \
    --batch-size 40 \
    --fp16 \
    --learning-rate 1e-5 \
    --job-dir .  \
    --total-steps 10000 \
    --epochs 10 \
    --checkpoint "{interiornet transformer model checkpoint}"
```
Here `{interiornet transformer model checkpoint}` is the path to the InteriorNet checkpoint (usually ending with `weights.model.099-last`).

### ShapeNet finetuning
```bash
viewformer-cli train finetune-transformer \
    --dataset "{cars code dataset path},{chairs code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --localization-weight 1 \
    --pose-multiplier 1 \
    --n-loss-skip 1 \
    --sequence-size 4 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --gradient-clip-val 1 \
    --job-dir .  \
    --total-steps 100000 \
    --epochs 100 \
    --weight-decay 0.05 \
    --checkpoint "{interiornet transformer model checkpoint}"
```
Here `{interiornet transformer model checkpoint}` is the path to the InteriorNet checkpoint (usually ending with `weights.model.099-last`).


### SM7 training
```bash
viewformer-cli train transformer \
    --dataset "{code dataset path}" \
    --codebook-model "{codebook model checkpoint}" \
    --sequence-size 6 \
    --n-loss-skip 1 \
    --batch-size 128 \
    --fp16 \
    --total-steps 120000 \
    --localization-weight "cosine(0,1,120000)" \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --job-dir . \
    --pose-multiplier 0.2
```
You can safely replace the cosine schedule for localization weight with a constant term.

## Evaluation
### Codebook evaluation
In order to evaluate the codebook model, run the following:
```bash
viewformer-cli evaluate codebook \
    --codebook-model "{codebook model checkpoint}" \
    --loader-path "{dataset path}" \
    --loader dataset \
    --loader-split test \
    --batch-size 64 \
    --image-size 128 \
    --num-store-images 0 \
    --num-eval-images 1000 \
    --job-dir . 
```
Note that `--image-size` argument controls the image size used for computing the metrics. You can change it to a different value.

### General transformer evaluation
In order to evaluate the transformer model, run the following:
```bash
viewformer-cli evaluate transformer \
    --codebook-model "{codebook model checkpoint}" \
    --transformer-model "{transformer model checkpoint}" \
    --loader-path "{dataset path}" \
    --loader dataset \
    --loader-split test \
    --batch-size 1 \
    --image-size 128 \
    --job-dir . \
    --num-eval-sequences 1000
```
Optionally, you can use `--sequence-size` to control the context size used for evaluation.
Note that `--image-size` argument controls the image size used for computing the metrics. You can change it to a different value.

### Transformer evaluation with different context sizes
In order to evaluate the transformer model with multiple context sizes, run the following:
```bash
viewformer-cli evaluate transformer-multictx \
    --codebook-model "{codebook model checkpoint}" \
    --transformer-model "{transformer model checkpoint}" \
    --loader-path "{dataset path}" \
    --loader dataset \
    --loader-split test \
    --batch-size 1 \
    --image-size 128 \
    --job-dir . \
    --num-eval-sequences 1000
```
Note that `--image-size` argument controls the image size used for computing the metrics. You can change it to a different value.

### CO3D evaluation
In order to evaluate the transformer model on the CO3D dataset, run the following:
```bash
viewformer-cli evaluate \
    --codebook-model "{codebook model checkpoint}" \
    --transformer-model "{transformer model checkpoint}" \
    --path {original CO3D root}
    --job-dir . 
```

### 7-Scenes evaluation
In order to evaluate the transformer model on the 7-Scenes dataset, run the following:
```bash
viewformer-cli evaluate 7scenes \
    --codebook-model "{codebook model checkpoint}" \
    --transformer-model "{transformer model checkpoint}" \
    --path {original 7-Scenes root}
    --batch-size 1
    --job-dir .
    --num-store-images 0
    --top-n-matched-images 10
    --image-match-map {path to top10 matched images}
```
You can change `--top-n-matched-images` to 0 if you don't want to use top 10 closest images in the context. `{path to top10 matched images}` as a path to the file containing the map between most similar images from the test and the train sets. Each line is in the format `{relative test image path} {relative train image path}`.

## Thanks
We would like to express our sincere gratitude to the authors of the following repositories, that we used in our code:
- [DeepMind Sonnet](https://github.com/deepmind/sonnet) 
- [Official implementation of Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers)
