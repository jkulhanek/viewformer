from typing import Optional
from aparse import ConditionalType, click
from viewformer.data.loaders import get_loaders
from viewformer.data import generate_dataset_from_loader
from viewformer.utils import SplitIndices
from viewformer.data import DatasetFormat


LoaderSwitch = ConditionalType('Loader', get_loaders(), prefix=False)


@click.command('generate')
def main(loader: LoaderSwitch,
         output: str,
         split: str,
         max_images_per_shard: Optional[int] = None,
         max_sequences_per_shard: Optional[int] = None,
         shards: SplitIndices = None,
         drop_last: bool = False,
         allow_incompatible_config: bool = False,
         format: DatasetFormat = 'tf'):
    # Generate Dataset
    generate_dataset_from_loader(
        loader, split, output_path=output,
        max_images_per_shard=max_images_per_shard,
        max_sequences_per_shard=max_sequences_per_shard,
        allow_incompatible_config=allow_incompatible_config,
        shards=shards, format=format)


if __name__ == '__main__':
    main()
