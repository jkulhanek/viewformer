import copy
from typing import Tuple, List, Optional
from aparse import Literal
from dataclasses import dataclass, fields, field, is_dataclass
from viewformer.utils.schedules import Schedule


ModelType = Literal['codebook', 'transformer']


def asdict(obj):
    dict_factory = dict

    def _asdict_inner(obj, dict_factory):
        if hasattr(obj, 'from_str'):
            return str(obj)
        elif is_dataclass(obj):
            result = []
            for f in fields(obj):
                value = _asdict_inner(getattr(obj, f.name), dict_factory)
                result.append((f.name, value))
            return dict_factory(result)
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)((_asdict_inner(k, dict_factory),
                              _asdict_inner(v, dict_factory))
                             for k, v in obj.items())
        else:
            return copy.deepcopy(obj)
        if not is_dataclass(obj):
            raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


@dataclass
class ModelConfig:
    model: str = field(init=False)

    def __post_init__(self):
        cls_name = type(self).__name__
        assert cls_name.endswith('Config')
        cls_name = cls_name[:-len('Config')]
        cls_name = cls_name.lower()
        self.model = cls_name

    def asdict(self):
        return asdict(self)

    @classmethod
    def supported_config_dict(cls):
        configs = {}
        if cls != ModelConfig:
            configs[cls.__name__.lower()[:-len('config')]] = cls
        for c in cls.__subclasses__():
            configs.update(c.supported_config_dict())
        return configs


@dataclass
class MIGTConfig(ModelConfig):
    n_embeddings: int = 1024
    n_head: int = 12
    d_model: int = 768
    dropout: float = 0.1
    n_layer: int = 12
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    learning_rate: float = 6.4e-4
    batch_size: int = 64
    gradient_clip_val: float = 0.0
    sequence_size: int = 20
    token_image_size: int = 8
    total_steps: int = 300000
    n_loss_skip: int = 4
    augment_poses: Literal['no', 'relative', 'simple', 'advanced'] = 'relative'
    use_dynamic_pose_loss: bool = False
    localization_weight: Schedule = Schedule.from_str('1')
    image_generation_weight: float = 1.

    pose_multiplier: float = 1.
    random_pose_multiplier: float = 1.

    @property
    def model_type(self):
        return 'transformer'


@dataclass
class VQGANConfig(ModelConfig):
    learning_rate: float = 1.584e-3
    embed_dim: int = 256
    n_embed: int = 1024
    z_channels: int = 256
    resolution: int = 256
    in_channels: int = 3
    out_ch: int = 3
    ch: int = 128
    num_res_blocks: int = 2
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    gradient_clip_val: float = .0
    batch_size: int = 352
    image_size: int = 128
    total_steps: int = 200000

    codebook_weight: float = 1.0
    pixelloss_weight: float = 1.0
    perceptual_weight: float = 1.0

    @property
    def stride(self):
        return 2 ** (len(self.ch_mult) - 1)

    @property
    def model_type(self):
        return 'codebook'
