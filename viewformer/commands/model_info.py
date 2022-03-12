from aparse import click, ConditionalType, WithArgumentName, Literal
from viewformer.models import supported_config_dict


ModelSwitch = WithArgumentName(ConditionalType('ModelSwitch', supported_config_dict(), default='vqgan', prefix=None), 'model')


@click.command('model-info')
def main(config: ModelSwitch = None, checkpoint: str = None, framework: Literal['tf', 'th'] = 'tf', parameter_count_depth: int = None):
    if framework == 'tf':
        from viewformer.models import AutoModel
        import tensorflow as tf
        model = AutoModel.from_config(config)

        def count_params(m, name, depth=None):
            num_params = sum(tf.size(x) for x in m.variables)
            num_trainable_params = sum(tf.size(x) for x in m.trainable_variables)
            children = []
            if depth is None or depth > 0:
                children = [count_params(x, x.name, depth - 1 if depth is not None else None) for x in getattr(m, 'layers', getattr(m, 'submodules', []))]
            return (name, num_params, num_trainable_params, children)

        _, num_params, num_trainable_params, children = count_params(model, None, parameter_count_depth)
    else:
        if checkpoint is None:
            from viewformer.models import AutoModelTH as AutoModel
            model = AutoModel.from_config(config)
        else:
            from viewformer.utils.torch import load_model
            model = load_model(checkpoint)

        def count_params(m, name, depth=None):
            num_params = sum(x.numel() for x in m.parameters())
            num_trainable_params = sum(x.numel() for x in m.parameters() if x.requires_grad)
            children = []
            if depth is None or depth > 0:
                children = [count_params(x, name, depth - 1 if depth is not None else None) for name, x in m.named_children()]
            return (name, num_params, num_trainable_params, children)

        _, num_params, num_trainable_params, children = count_params(model, None, parameter_count_depth)

    def print_children(c, offset=''):
        name, num_params, num_trainable_params, children = c
        print(f'{offset}{name} ({num_trainable_params}/{num_params})')
        for c2 in children:
            print_children(c2, offset + '  ')

    for c in children:
        print_children(c)
    print(f'Total number of parameters: {num_params}')
    print(f'Total number of trainable parameters: {num_trainable_params}')
