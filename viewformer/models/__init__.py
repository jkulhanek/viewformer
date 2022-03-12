import dataclasses
from .config import ModelConfig
from .config import *  # noqa: F401, F403
import importlib

_TH_REPOSITORY = {}

_TF_REPOSITORY = {}


class ModelNotFoundError(RuntimeError):
    pass


class AutoModel:
    @staticmethod
    def from_config(config, **kwargs) -> 'tensorflow.keras.Model':
        package = config.model.lower()
        package, cls_name = _TF_REPOSITORY.get(config.model, (package, None))
        try:
            module = importlib.import_module(f'.{package}', __package__)
        except ImportError:
            raise ModelNotFoundError(f'Model {config.model} not found')

        cls = None
        if cls_name is not None:
            cls = getattr(module, cls_name)
        else:
            try:
                cls = next(iter((x for name, x in vars(module).items() if name.lower() == config.model)))
            except StopIteration:
                pass
        if cls is None:
            raise ModelNotFoundError(f'Model {config.model} not found')
        return cls(config, **kwargs)


class AutoModelTH:
    @staticmethod
    def from_config(config, **kwargs):
        package = config.model.lower() + '_th'
        package, cls_name = _TH_REPOSITORY.get(config.model, (package, None))
        try:
            module = importlib.import_module(f'.{package}', __package__)
        except ImportError:
            raise ModelNotFoundError(f'Model {config.model} not found')

        cls = None
        if cls_name is not None:
            cls = getattr(module, cls_name)
        else:
            try:
                cls = next(iter((x for name, x in vars(module).items() if name.lower() == config.model.lower())))
            except StopIteration:
                pass
        if cls is None:
            raise ModelNotFoundError(f'Model {config.model} not found')
        model = cls(config, **kwargs)
        return model


def load_config(config):
    def _build_dataclass(cls, kwargs):
        new_kwargs = dict()
        for field in dataclasses.fields(cls):
            if field.name not in kwargs:
                continue
            val = kwargs[field.name]
            if hasattr(field.type, 'from_str') and isinstance(val, str):
                val = field.type.from_str(val)
            elif dataclasses.is_dataclass(field.type):
                val = _build_dataclass(field.type, val)
            new_kwargs[field.name] = val
        return cls(**new_kwargs)

    model = config.pop('model')
    config_cls = supported_config_dict()[model]
    return _build_dataclass(config_cls, config)


def supported_config_dict():
    return ModelConfig.supported_config_dict()
