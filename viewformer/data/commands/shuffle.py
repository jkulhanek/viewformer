import os
import json
from aparse import click
from viewformer.data.tfrecord_shuffle import shuffle_dataset


@click.command('shuffle')
def main(input: str, output: str):
    dataset_info = json.load(open(os.path.join(input, 'info.json'), 'r'))
    shuffle_dataset(input, output)


if __name__ == '__main__':
    main()
