import math
import inspect
import dataclasses


def default_from_str(cls, value):
    name = cls._get_name().lower()
    if not value.startswith(f'{name}(') or not value.endswith(')'):
        return None
    args_strs = [x.strip() for x in value[len(f'{name}('):-1].split(',')]
    args = [x for x in args_strs if '=' not in args_strs]
    kwargs = {k: v for k, v in (x.split('=', 1) for x in args if '=' in x)}
    params = inspect.signature(cls).parameters
    arg_types = [x.annotation for x in params.values() if x.kind in {inspect.Parameter.POSITIONAL_ONLY or inspect.Parameter.POSITIONAL_OR_KEYWORD}][:len(args)]
    assert len(arg_types) == len(args)
    kwarg_types = {x.name: x.annotation for x in params.values() if x.kind in {inspect.Parameter.KEYWORD_ONLY or inspect.Parameter.POSITIONAL_OR_KEYWORD}}
    for i, v in enumerate(args):
        args[i] = arg_types[i](v)
    for k, v in kwargs.items():
        kwargs[k] = kwarg_types[k](v)
    return cls(*args, **kwargs)


def with_default_from_str(cls=None):
    def wrap(cls):
        setattr(cls, '_from_str', classmethod(default_from_str))
        return cls
    if cls is not None:
        return wrap(cls)
    return wrap


def get_backend(tensor):
    if type(tensor).__module__.startswith('tensorflow'):
        return 'tf'
    return 'math'


class TFBackend:
    def __init__(self, tf):
        self._tf = tf
        for k in dir(tf):
            setattr(self, k, getattr(tf, k))
        self.min = tf.minimum
        self.max = tf.maximum

    def with_same_type(self, x, other):
        if not self._tf.is_tensor(x):
            x = self._tf.ones_like(other) * x
        return self._tf.cast(x, other.dtype)

    def cast(self, x, dtype):
        return self._tf.cast(x, dtype)


class MathBackend:
    def __init__(self):
        import math
        self._math = math
        for k in dir(math):
            setattr(self, k, getattr(math, k))
        self.min = min
        self.max = max

    def with_same_type(self, x, other):
        return float(x)

    def cast(self, x, dtype):
        return float(x)


class Schedule:
    def __call__(self, t: int, dtype: str = 'float32'):
        backend_name = get_backend(t)
        if backend_name == 'tf':
            import tensorflow as tf
            backend = TFBackend(tf)
        else:
            backend = MathBackend()
        inp_dtype = 'float32'
        if dtype == 'float64':
            inp_dtype = 'float64'
        t = backend.cast(t, inp_dtype)
        result = self.call(t, backend_name=backend_name, backend=backend)
        result = backend.cast(result, dtype)
        return result

    def __mul__(self, other):
        raise NotImplementedError()

    def __rmul__(self, other):
        return self.__mul__(other)

    def call(self, t: int, *, backend_name: str, backend) -> float:
        raise NotImplementedError()

    @classmethod
    def _get_name(cls):
        assert cls.__name__.endswith('Schedule')
        return cls.__name__[:-len('Schedule')]

    @classmethod
    def _from_str(cls, value):
        return None

    @classmethod
    def from_str(cls, value):
        for subclass in reversed(cls.__subclasses__()):
            obj = subclass.from_str(value)
            if obj is not None:
                return obj
        return cls._from_str(value)

    def with_total_steps(self, num_total_steps):
        if not hasattr(self, 'num_total_steps') or self.num_total_steps is not None:
            return self
        else:
            return dataclasses.replace(self, num_total_steps=num_total_steps)

    def is_zero(self):
        return False

    @staticmethod
    def zero():
        return ConstantSchedule(value=0)


@dataclasses.dataclass(frozen=True)
class ConstantSchedule(Schedule):
    value: float

    def call(self, value, *args, **kwargs):
        return (0 * value + 1) * self.value

    @classmethod
    def _from_str(cls, value):
        try:
            parsed_val = float(value)
            return cls(value=parsed_val)
        except Exception:
            pass
        return None

    def __str__(self):
        return str(self.value)

    def is_zero(self):
        return self.value == 0

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return dataclasses.replace(self,
                                       value=other * self.value)
        else:
            raise ValueError(f'Type {type(other)} is not supported')


@with_default_from_str
@dataclasses.dataclass(frozen=True)
class LinearSchedule(Schedule):
    initial_value: float
    final_value: float
    num_total_steps: int = None

    @classmethod
    def _from_str(cls, value):
        try:
            parsed_val = float(value)
            return cls(value=parsed_val)
        except Exception:
            pass
        return None

    def call(self, t, *, backend=None, **kwargs):
        return self.initial_value + backend.min(t / self.num_total_steps, backend.with_same_type(1., t)) * (self.final_value - self.initial_value)

    def __str__(self):
        return f'linear({self.initial_value},{self.final_value},{self.num_total_steps})'

    def is_zero(self):
        return self.initial_value == self.final_value == 0

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return dataclasses.replace(self,
                                       initial_value=other * self.initial_value,
                                       final_value=other * self.final_value)
        else:
            raise ValueError(f'Type {type(other)} is not supported')


@with_default_from_str
@dataclasses.dataclass(frozen=True)
class CosineSchedule(Schedule):
    initial_value: float
    final_value: float
    num_total_steps: int = None

    def call(self, t, *, backend=None, **kwargs):
        return self.final_value + (self.initial_value - self.final_value) * 0.5 * (
            backend.cos(backend.min(backend.with_same_type(1., t), t / self.num_total_steps) * backend.with_same_type(math.pi, t)) + 1)

    def __str__(self):
        return f'cosine({self.initial_value},{self.final_value},{self.num_total_steps})'

    def is_zero(self):
        return self.initial_value == self.final_value == 0

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return dataclasses.replace(self,
                                       initial_value=other * self.initial_value,
                                       final_value=other * self.final_value)
        else:
            raise ValueError(f'Type {type(other)} is not supported')


@dataclasses.dataclass(frozen=True)
class WarmupSchedule(Schedule):
    inner: Schedule
    warmup_steps: int

    def call(self, t, *, backend=None, **kwargs):
        warmup_time = backend.min(t, self.warmup_steps)
        rest_time = backend.max(t - self.warmup_steps, 0)
        return (warmup_time / self.warmup_steps) * self.inner.call(rest_time, backend=backend, **kwargs)

    def is_zero(self):
        return self.inner.is_zero()

    def __str__(self):
        return f'warmup({str(self.inner)},{self.warmup_steps})'

    def __mul__(self, other):
        return dataclasses.replace(self, inner=self.inner * other)

    @classmethod
    def _from_str(cls, value):
        if not value.startswith('warmup(') or not value.endswith(')') or ',' not in value:
            return None
        value = value[len('warmup('):-1]
        splitter = value.rindex(',')
        inner_str, wsteps = value[:splitter].strip(), value[splitter + 1:].strip()
        wsteps = int(wsteps)
        inner_schedule = Schedule.from_str(inner_str)
        if inner_schedule is None:
            return None
        return WarmupSchedule(inner=inner_schedule, warmup_steps=wsteps)
