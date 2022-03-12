import click
from viewformer.utils import pull_checkpoint


@click.command('download-model')
@click.argument('checkpoint', type=str)
def main(checkpoint: str):
    print(f'Downloading checkpoint {checkpoint}')
    pull_checkpoint(checkpoint, override=True)
    print(f'Checkpoint {checkpoint} downloaded')
