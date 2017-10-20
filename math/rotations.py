import numpy


def obtain_matrix_from_quaternion(q: numpy.ndarray) -> numpy.ndarray:
    if q.size != 4:
        raise ValueError('Wrong number of elements on the array (q.size != 4)!')

    m = numpy.zeros((3, 3), dtype=numpy.float64)
    m[0, 0] = 1.0 - 2.0 * q[2] * q[2] - 2.0 * q[3] * q[3]
    m[0, 1] = 2.0 * q[1] * q[2] - 2.0 * q[3] * q[0]
    m[0, 2] = 2.0 * q[1] * q[3] + 2.0 * q[2] * q[0]
    m[1, 0] = 2.0 * q[1] * q[2] + 2.0 * q[3] * q[0]
    m[1, 1] = 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[3] * q[3]
    m[1, 2] = 2.0 * q[2] * q[3] - 2.0 * q[1] * q[0]
    m[2, 0] = 2.0 * q[1] * q[3] - 2.0 * q[2] * q[0]
    m[2, 1] = 2.0 * q[2] * q[3] + 2.0 * q[1] * q[0]
    m[2, 2] = 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2]

    return m


def obtain_angle_vector_from_matrix(m: numpy.ndarray) -> tuple:
    v = numpy.array([
        m[2, 1] - m[1, 2],
        m[0, 2] - m[2, 0],
        m[1, 0] - m[0, 1]
        ], dtype=numpy.float64)
    a = numpy.linalg.norm(v)
    v /= a
    a = numpy.arcsin(a / 2.0)

    return (a, v)


def obtain_quaternion_from_angle_vector(a: float, v: numpy.ndarray) -> numpy.ndarray:
    v_magnitude = numpy.linalg.norm(v)
    q = numpy.array((1.0, 0.0, 0.0, 0.0), dtype=numpy.float64)
    if v_magnitude > 0.0 and abs(a) > 0.0:
        q = numpy.hstack((numpy.cos(a / 2.0), numpy.sin(a / 2.0) * v / v_magnitude))

    return q


def calculate_hamilton_product(q1: numpy.ndarray, q2: numpy.ndarray) -> numpy.ndarray:
    p = numpy.zeros((4), dtype=numpy.float64)

    p[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    p[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    p[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    p[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]

    return p


def obtain_quaternion_conjugate(q: numpy.ndarray)->numpy.ndarray:
    p = q.copy()
    p[1:] *= -1

    return p


def calculate_rotated_vector_by_quaternion(
  v: numpy.ndarray,
  q: numpy.ndarray
  ) -> numpy.ndarray:
    p = numpy.hstack((0.0, v))
    p = calculate_hamilton_product(
            calculate_hamilton_product(q, p),
            obtain_quaternion_conjugate(q)
            )

    return p[1:]


def calculate_rotation_matrix_from_bases(
  b1: numpy.ndarray,
  b2: numpy.ndarray,
  b3: numpy.ndarray,
  v1: numpy.ndarray,
  v2: numpy.ndarray,
  v3: numpy.ndarray,
  ) -> numpy.ndarray:

    b1n = b1 / numpy.linalg.norm(b1)
    b2n = b2 / numpy.linalg.norm(b2)
    b3n = b3 / numpy.linalg.norm(b3)
    v1n = v1 / numpy.linalg.norm(v1)
    v2n = v2 / numpy.linalg.norm(v2)
    v3n = v3 / numpy.linalg.norm(v3)

    V = numpy.matmul(v1n.reshape(3, 1), b1n.reshape((1, 3))) \
        + numpy.matmul(v2n.reshape(3, 1), b2n.reshape((1, 3))) \
        + numpy.matmul(v3n.reshape(3, 1), b3n.reshape((1, 3)))

    U, Q, W = numpy.linalg.svd(V)

    R = numpy.matmul(U, W)

    return R
