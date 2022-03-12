import math
import numpy as np
import tensorflow as tf


def quaternion_multiply(quaternion1, quaternion2):
    w1, x1, y1, z1 = tf.unstack(quaternion1, 4, axis=-1)
    w2, x2, y2, z2 = tf.unstack(quaternion2, 4, axis=-1)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return tf.stack((w, x, y, z), -1)


def make_quaternion(axis, angle):
    angle = tf.convert_to_tensor(angle)
    w = tf.cos(angle / 2)[..., tf.newaxis]
    xyz = tf.sin(angle / 2)[..., tf.newaxis]
    xyz = xyz * axis
    return tf.concat([w, xyz], -1)


def make_quaternion_y(angle):
    angle = tf.convert_to_tensor(angle)
    axis = tf.constant([0, 1, 0], dtype=angle.dtype)
    return make_quaternion(axis, angle)


def make_quaternion_x(angle):
    angle = tf.convert_to_tensor(angle)
    axis = tf.constant([1, 0, 0], dtype=angle.dtype)
    return make_quaternion(axis, angle)


def assert_normalized(x, eps=None, axis=-1):
    if eps is None:
        eps = tf.constant(10.0 * np.finfo(x.dtype.as_numpy_dtype()).tiny, dtype=x.dtype)
    norm = tf.linalg.norm(x, axis=axis)
    one = tf.constant(1.0, dtype=norm.dtype)
    tf.debugging.assert_near(norm, one, atol=eps)


def quaternion_normalize(x, epsilon=1e-12):
    return tf.linalg.l2_normalize(x, axis=-1, epsilon=epsilon)


def quaternion_remove_sign(x):
    sign = 2 * tf.cast(x[..., :1] >= 0, x.dtype) - 1
    return x * sign


def quaternion_conjugate(quaternion):
    """Computes the conjugate of a quaternion.
    Note:
    In the following, A1 to An are optional batch dimensions.
    Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_conjugate".
    Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.
    Raises:
    ValueError: If the shape of `quaternion` is not supported.
    """
    w, xyz = tf.split(quaternion, (1, 3), axis=-1)
    return tf.concat((w, -xyz), axis=-1)


def quaternion_rotate(point, quaternion):
    """Rotates a point using a quaternion.
    Note:
    In the following, A1 to An are optional batch dimensions.
    Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.
    name: A name for this op that defaults to "quaternion_rotate".
    Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents a
    3d point.
    Raises:
    ValueError: If the shape of `point` or `quaternion` is not supported.
    """
    assert_normalized(quaternion)
    point = tf.concat([tf.zeros_like(point[..., :1]), point], -1)
    point = quaternion_multiply(quaternion, point)
    point = quaternion_multiply(point, quaternion_conjugate(quaternion))
    return point[..., 1:]


def quaternion_to_euler(quaternion):
    w, x, y, z = tf.unstack(quaternion, 4, axis=-1)

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * z + x * y)
    cosr_cosp = 1 - 2 * (z * z + x * x)
    roll = tf.math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * x - y * z)
    pitch = tf.where(
        tf.abs(sinp) >= 1,
        math.pi / 2 * tf.sign(sinp),
        tf.math.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * y + z * x)
    cosy_cosp = 1 - 2 * (x * x + y * y)
    yaw = tf.math.atan2(siny_cosp, cosy_cosp)
    return tf.stack([pitch, yaw, roll], -1)
