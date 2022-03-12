from aparse import click
from viewformer.utils.click import LazyGroup


@click.group(cls=LazyGroup)
def main():
    pass


@main.group(cls=LazyGroup)
def dataset():
    pass


@main.group(cls=LazyGroup)
def visualize():
    pass


@main.group(cls=LazyGroup)
def model():
    pass


@main.group(cls=LazyGroup)
def evaluate():
    pass


dataset.add_command('viewformer.data.commands.visualize', 'visualize')
dataset.add_command('viewformer.data.commands.generate', 'generate')
dataset.add_command('viewformer.data.commands.shuffle', 'shuffle')

visualize.add_command('viewformer.commands.visualize_codebook', 'codebook')

model.add_command('viewformer.commands.model_info', 'info')

evaluate.add_command("viewformer.evaluate.evaluate_transformer", "transformer")
evaluate.add_command("viewformer.evaluate.evaluate_transformer_multictx", "transformer-multictx")
evaluate.add_command("viewformer.evaluate.evaluate_transformer_multictx_allimg", "transformer-multictx-allimg")
evaluate.add_command("viewformer.evaluate.evaluate_codebook", "codebook")
evaluate.add_command("viewformer.evaluate.evaluate_sevenscenes", "7scenes")
evaluate.add_command("viewformer.evaluate.evaluate_sevenscenes_baseline", "7scenes-baseline")
evaluate.add_command("viewformer.evaluate.evaluate_sevenscenes_multictx", "7scenes-multictx")
evaluate.add_command("viewformer.evaluate.evaluate_co3d", "co3d")
evaluate.add_command("viewformer.evaluate.generate_gqn_images", "generate-gqn-images")

main.add_command("viewformer.train", "train")
main.add_command("viewformer.commands.generate_codes", 'generate-codes')
main.add_command("viewformer.commands.download_model", 'download-model')

if __name__ == '__main__':
    main()
