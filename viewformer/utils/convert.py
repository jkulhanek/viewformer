import re
from collections import OrderedDict


def _replace_end(string, end, new_end):
    if string == end:
        return new_end
    if string.endswith('.' + end):
        return string[:-len('.' + end)] + '.' + new_end
    return string


def zip_th_tf_parameters(torch_module, tf_module, permute_torch=False):
    def replace_tf_key(key):
        key = key.replace('/', '.').rstrip(':0')
        key = _replace_end(key, 'moving_mean', 'running_mean')
        key = _replace_end(key, 'moving_variance', 'running_var')
        key = _replace_end(key, 'kernel', 'weight')
        key = _replace_end(key, 'beta', 'bias')
        key = _replace_end(key, 'gamma', 'weight')
        return key

    def endswith(string, end):
        if string == end:
            return True
        if string.endswith('.' + end):
            return True
        return False

    weights = torch_module.state_dict()
    variables = tf_module.variables
    ignored = set()
    if hasattr(tf_module, '_metrics'):
        # Remove metrics from weights
        delete_vars = {v.ref() for x in tf_module._metrics for v in x.variables}
        variables = [v for v in variables if v.ref() not in delete_vars]

    tf_weights = {replace_tf_key(v.name): v for v in variables}
    # print([x.name for x in tf_module.variables])
    try:
        tf_model_name, _ = next(iter(tf_weights.keys())).split('.', 1)
        if all(x.startswith(f'{tf_model_name}.') for x in tf_weights.keys()):
            tf_weights = {k[len(f'{tf_model_name}.'):]: v for k, v in tf_weights.items()}
    except StopIteration:
        pass

    if hasattr(tf_module, '_ignore_checkpoint_attributes'):
        pattern = re.compile('|'.join(map(lambda x: f'({x})', tf_module._ignore_checkpoint_attributes)))
        ignored = {k for k in weights.keys() if pattern.match(k)}
    # print([x for x in tf_weights.keys()])
    # print([x for x in weights.keys()])
    unmatched = set(tf_weights.keys()).difference(set(weights.keys()) - ignored)
    # print(unmatched)
    assert len(unmatched) == 0, f'There are some unmatched keys in model parameters ({len(unmatched)}), e.g., ' + ', '.join(list(unmatched)[:4])
    for k, val in weights.items():
        if k in ignored:
            continue
        if endswith(k, 'num_batches_tracked'):
            # This property is ignored in tensorflow
            continue
        if permute_torch:
            if endswith(k, 'weight') and len(val.shape) == 4:
                # val = val.permute(2, 3, 1, 0)
                val = val.permute(2, 3, 1, 0)
                pass
            elif endswith(k, 'weight') and len(val.shape) == 2:
                val = val.permute(1, 0)
        assert k in tf_weights, f'There are some weights ({k}) not found on the tf_module'
        matching_weight = tf_weights[k]
        assert matching_weight.shape == val.shape, f'Shape does not match for parameter {k}, {matching_weight.shape} != {val.shape}'
        yield val, matching_weight


def convert_weights_th_to_tf(torch_module, tf_module):
    for th_weight, tf_variable in zip_th_tf_parameters(torch_module, tf_module, permute_torch=True):
        tf_variable.assign(th_weight.clone().numpy())


def convert_model_th_to_tf(torch_module, checkpoint):
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    class TFStoredModel:
        def __init__(self, model, output_type, output_names):
            self.model = model
            self.output_type = output_type
            self.output_names = output_names

        @staticmethod
        def _nhwc_to_nchw(x):
            if isinstance(x, list):
                return [TFStoredModel._nhwc_to_nchw(y) for y in x]
            if isinstance(x, tuple):
                return tuple(TFStoredModel._nhwc_to_nchw(y) for y in x)
            if isinstance(x, dict):
                return x.__class__([(k, TFStoredModel._nhwc_to_nchw(y)) for k, y in x.items()])
            if len(tf.shape(x)) == 4 and tf.shape(x)[-1] == 3:
                return tf.transpose(x, (0, 3, 1, 2))
            return x

        @staticmethod
        def _nchw_to_nhwc(x):
            if isinstance(x, list):
                return [TFStoredModel._nchw_to_nhwc(y) for y in x]
            if isinstance(x, tuple):
                return tuple(TFStoredModel._nchw_to_nhwc(y) for y in x)
            if isinstance(x, dict):
                return {k: TFStoredModel._nchw_to_nhwc(y) for k, y in x.items()}
            if len(tf.shape(x)) == 4 and tf.shape(x)[-3] == 3:
                return tf.transpose(x, (0, 2, 3, 1))
            return x

        def __call__(self, *args, **kwargs):
            args = self._nhwc_to_nchw(args)
            kwargs = self._nhwc_to_nchw(kwargs)
            output = self.model.signatures['serving_default'](*args, **kwargs)
            output = self._nchw_to_nhwc(output)
            len_output = len(output.keys())
            output = tuple(output[f'output_{i}'] for i in range(len_output))
            if self.output_type == list:
                return list(output)
            elif self.output_type is None:
                return output[0]
            elif self.output_type == tuple:
                return output
            else:
                assert self.output_type in {dict, OrderedDict}
                return self.output_type(zip(self.output_names, output))

    onnx_path = f'{checkpoint}.onnx'
    output_type = torch_module.to_onnx(onnx_path, export_params=True)
    onnx_model = onnx.load(onnx_path)
    output_names = [x.name for x in onnx_model.graph.output]
    if output_type is None and len(output_names) > 0:
        output_type = tuple
    assert output_type in {OrderedDict, dict, tuple, list, None}
    model = prepare(onnx_model, auto_cast=True)
    model.export_graph(f'{checkpoint}.pb')
    path = f'{checkpoint}.pb'

    model = tf.saved_model.load(path)
    model = TFStoredModel(model, output_type=output_type, output_names=output_names)
    if hasattr(torch_module, 'config'):
        setattr(model, 'config', torch_module.config)
    return model
