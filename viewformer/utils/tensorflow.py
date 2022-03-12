from typing import List
import os
import json
import tensorflow as tf
import logging
from viewformer.models import AutoModel, load_config, ModelNotFoundError

_logger = logging.getLogger(__name__)


def shape_list(tensor: tf.Tensor) -> List[int]:
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def load_model(checkpoint, restore_weights: bool = True, **kwargs):
    model_path, checkpoint = os.path.split(checkpoint)
    if not tf.io.gfile.exists(os.path.join(model_path, 'config.json')) and '/' not in model_path:
        # It could be a network checkpoint
        from viewformer.utils import pull_checkpoint

        model_path = pull_checkpoint(checkpoint)
        if os.path.exists(os.path.join(model_path, 'model.index')):
            checkpoint = 'model'
        else:
            checkpoint = 'model.ckpt'  # Torch checkpoint

    with tf.io.gfile.GFile(os.path.join(model_path, 'config.json'), mode='r') as f:
        config = json.load(f)
        config.update(kwargs)
        config = load_config(config)

    is_th = checkpoint.endswith('.pth') or checkpoint.endswith('.ckpt')
    if is_th:
        _logger.warn('the loaded model is a PyTorch checkpoint')

        from viewformer.models import AutoModelTH
        import torch
        from viewformer.utils.convert import convert_weights_th_to_tf

        th_model = AutoModelTH.from_config(config)
        checkpoint_data = torch.load(
            tf.io.gfile.GFile(os.path.join(model_path, checkpoint), mode='rb'), map_location=torch.device('cpu'))
        th_model.load_state_dict(checkpoint_data['state_dict'])
        try:
            model = AutoModel.from_config(th_model.config)
            convert_weights_th_to_tf(th_model, model)
            return model
        except ModelNotFoundError:
            _logger.warn('the loaded model is not implemented for TensorFlow, we will try to load it using ONNX')
            from viewformer.utils.convert import convert_model_th_to_tf
            return convert_model_th_to_tf(th_model, checkpoint)
    else:
        model = AutoModel.from_config(config)
        if restore_weights:
            status = model.load_weights(os.path.join(model_path, checkpoint))
            if hasattr(status, 'expect_partial'):
                status.expect_partial()
    return model


def batch_cat(*x, axis=0):
    if isinstance(x[0], tuple):
        return tuple(batch_cat(*y, axis=axis) for y in zip(*x))
    elif isinstance(x[0], dict):
        return x[0].__class__([(k, batch_cat(*[y[k] for y in x], axis=axis)) for k in x[0].keys()])
    return tf.concat(x, axis=axis)
