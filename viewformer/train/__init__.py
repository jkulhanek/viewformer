import importlib
from aparse import click
from viewformer.utils.click import LazyGroup


@click.group('train', cls=LazyGroup)
def main():
    pass


train_transformer = main.add_command('viewformer.train.train_transformer', 'transformer')
train_codebook = main.add_command('viewformer.train.train_codebook_th', 'codebook')
train_codebook = main.add_command('viewformer.train.finetune_transformer', 'finetune-transformer')

if __name__ == '__main__':
    main()
