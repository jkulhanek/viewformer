import math
import numpy as np
import numpy.matlib as npm


# NOTE: we are using (w, x, y, z) order for quaternions


def safe_unsigned_div(a, b, eps=None):
    """Calculates a/b with b >= 0 safely.
    a: A `float` or a tensor of shape `[A1, ..., An]`, which is the nominator.
    b: A `float` or a tensor of shape `[A1, ..., An]`, which is the denominator.
    eps: A small `float`, to be added to the denominator. If left as `None`, its
      value is automatically selected using `b.dtype`.
    name: A name for this op. Defaults to 'safe_unsigned_div'.
    Raises:
     InvalidArgumentError: If tf-graphics debug flag is set and the division
       causes `NaN` or `Inf` values.
    Returns:
     A tensor of shape `[A1, ..., An]` containing the results of division.
    """
    if eps is None:
        eps = 10.0 * np.finfo(b.dtype).tiny
    return a / (b + eps)


# Taken from https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py#L290-L375
def rotation_matrix_to_quaternion(rotation_matrix):
    """Converts a rotation matrix representation to a quaternion.
    Warning:
        This function is not smooth everywhere.
    Note:
        In the following, A1 to An are optional batch dimensions.
    Args:
        rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
            dimensions represent a rotation matrix.
        name: A name for this op that defaults to "quaternion_from_rotation_matrix".
    Returns:
        A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
        a normalized quaternion.
    Raises:
        ValueError: If the shape of `rotation_matrix` is not supported.
    """
    trace = np.trace(rotation_matrix, axis1=-2, axis2=-1)
    eps_addition = 2.0 * np.finfo(rotation_matrix.dtype).eps
    rows = list(np.moveaxis(rotation_matrix, -2, 0))
    entries = [list(np.moveaxis(row, -1, 0)) for row in rows]

    def tr_positive():
        sq = np.sqrt(trace + 1.0) * 2.    # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_unsigned_div(entries[2][1] - entries[1][2], sq)
        qy = safe_unsigned_div(entries[0][2] - entries[2][0], sq)
        qz = safe_unsigned_div(entries[1][0] - entries[0][1], sq)
        return np.stack((qw, qx, qy, qz), -1)

    def cond_1():
        sq = np.sqrt(1.0 + entries[0][0] - entries[1][1] - entries[2][2] + eps_addition) * 2.    # sq = 4 * qx.
        qw = safe_unsigned_div(entries[2][1] - entries[1][2], sq)
        qx = 0.25 * sq
        qy = safe_unsigned_div(entries[0][1] + entries[1][0], sq)
        qz = safe_unsigned_div(entries[0][2] + entries[2][0], sq)
        return np.stack((qw, qx, qy, qz), axis=-1)

    def cond_2():
        sq = np.sqrt(1.0 + entries[1][1] - entries[0][0] - entries[2][2] + eps_addition) * 2.    # sq = 4 * qy.
        qw = safe_unsigned_div(entries[0][2] - entries[2][0], sq)
        qx = safe_unsigned_div(entries[0][1] + entries[1][0], sq)
        qy = 0.25 * sq
        qz = safe_unsigned_div(entries[1][2] + entries[2][1], sq)
        return np.stack((qw, qx, qy, qz), axis=-1)

    def cond_3():
        sq = np.sqrt(1.0 + entries[2][2] - entries[0][0] - entries[1][1] + eps_addition) * 2.    # sq = 4 * qz.
        qw = safe_unsigned_div(entries[1][0] - entries[0][1], sq)
        qx = safe_unsigned_div(entries[0][2] + entries[2][0], sq)
        qy = safe_unsigned_div(entries[1][2] + entries[2][1], sq)
        qz = 0.25 * sq
        return np.stack((qw, qx, qy, qz), axis=-1)

    def cond_idx(cond):
        cond = np.expand_dims(cond, -1)
        cond = np.tile(cond, [1] * (len(rotation_matrix.shape) - 2) + [4])
        return cond

    where_2 = np.where(
        cond_idx(entries[1][1] > entries[2][2]), cond_2(), cond_3())
    where_1 = np.where(
        cond_idx((entries[0][0] > entries[1][1]) & (entries[0][0] > entries[2][2])), cond_1(), where_2)
    quat = np.where(cond_idx(trace > 0), tr_positive(), where_1)
    return quat


def quaternion_multiply(quaternion1, quaternion2):
    w1, x1, y1, z1 = np.moveaxis(quaternion1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(quaternion2, -1, 0)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return np.stack((w, x, y, z), -1)


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / np.sqrt(np.maximum((x**2).sum(axis=axis, keepdims=True), epsilon))


def quaternion_normalize(x, epsilon=1e-12):
    x = l2_normalize(x, axis=-1, epsilon=epsilon)
    return x


def quaternion_remove_sign(x):
    sign = 2 * (x[..., :1] >= 0).astype(x.dtype) - 1
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
    w, xyz = np.split(quaternion, (1,), axis=-1)
    return np.concatenate((w, -xyz), axis=-1)


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
    point = np.concatenate([np.zeros_like(point[..., :1]), point], -1)
    point = quaternion_multiply(quaternion, point)
    point = quaternion_multiply(point, quaternion_conjugate(quaternion))
    return point[..., 1:]


def assert_normalized(x, eps=None, axis=-1):
    if eps is None:
        #     eps = np.array(10.0 * np.finfo(x.dtype).tiny, dtype=x.dtype)
        eps = 1e-7
    norm = np.linalg.norm(x, axis=axis)
    one = np.array(1.0, dtype=norm.dtype)
    np.testing.assert_allclose(norm, one, atol=eps)


def make_rotation_matrix_y(angle):
    sin, cos = np.sin(angle), np.cos(angle)
    zeros, ones = np.zeros_like(angle), np.ones_like(angle)
    block = np.stack([
        np.stack([cos, zeros, sin], -1),
        np.stack([zeros, ones, zeros], -1),
        np.stack([-sin, zeros, cos], -1)
    ], -2)
    return block


def make_rotation_matrix_x(angle):
    sin, cos = np.sin(angle), np.cos(angle)
    zeros, ones = np.zeros_like(angle), np.ones_like(angle)
    block = np.stack([
        np.stack([ones, zeros, zeros], -1),
        np.stack([zeros, cos, -sin], -1),
        np.stack([zeros, sin, cos], -1)
    ], -2)
    return block


def make_quaternion(axis, angle):
    w = np.cos(angle / 2)[..., np.newaxis]
    xyz = np.sin(angle / 2)[..., np.newaxis]
    xyz = xyz * axis
    return np.concatenate([w, xyz], -1)


def make_quaternion_y(angle):
    axis = np.array([0, 1, 0], dtype=angle.dtype)
    return make_quaternion(axis, angle)


def make_quaternion_x(angle):
    axis = np.array([1, 0, 0], dtype=angle.dtype)
    return make_quaternion(axis, angle)


def cameras_to_pose_euler(pose):
    xyz, quaternion = np.split(pose, (3,), -1)
    angles = quaternion_to_euler(quaternion)
    return np.concatenate((xyz, angles), -1)


def look_at_to_cameras(camera_position, look_at, up_vector):
    '''
    Converts look_at to cameras (x, y, z, wr, xr, yr, zr), where (wr, xr, yr, zr) is
    the rotation represented by a quaternion, z faces away from camera, y points down and x points to right
    in the right-handed system
    '''
    z_axis = l2_normalize(look_at - camera_position)
    x_axis = l2_normalize(np.cross(z_axis, up_vector))
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([y_axis, -x_axis, z_axis], -1)
    quaternion = quaternion_normalize(rotation_matrix_to_quaternion(R))
    t = camera_position
    return np.concatenate((t, quaternion), -1)


def quaternion_to_euler(quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * z + x * y)
    cosr_cosp = 1 - 2 * (z * z + x * x)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * x - y * z)
    pitch = np.where(
        np.abs(sinp) >= 1,
        math.pi / 2 * np.sign(sinp),
        np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * y + z * x)
    cosy_cosp = 1 - 2 * (x * x + y * y)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([pitch, yaw, roll], -1)


def quaternion_to_rotation_matrix(quaternion):
    assert_normalized(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = np.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)  # pyformat: disable
    return matrix.reshape(quaternion.shape[:-1] + (3, 3))


# https://ntrs.nasa.gov/citations/20070017872
# https://jp.mathworks.com/matlabcentral/fileexchange/40098-tolgabirdal-averaging_quaternions
def quaternion_average(quaternion, axis=-2):
    # q * qT
    quaternion = quaternion_remove_sign(quaternion)
    M = quaternion[..., np.newaxis, :] * quaternion[..., :, np.newaxis]
    M = M.mean(axis - 1)
    eig_val, eig_vec = np.linalg.eig(M)
    largest_eigvec = np.take_along_axis(eig_vec, np.argmax(eig_val, -1)[..., np.newaxis, np.newaxis], -2).squeeze(-2)
    return np.real(largest_eigvec)
