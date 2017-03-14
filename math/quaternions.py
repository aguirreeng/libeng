import math
import numpy

def MatrixFromQuaternion(q):
    m = numpy.zeros((3,3), dtype=numpy.float64)
    m[0,0] = 1.0 - 2.0 * q[2] * q[2] - 2.0 * q[3] * q[3]
    m[0,1] = 2.0 * q[1] * q[2] - 2.0 * q[3] * q[0]
    m[0,2] = 2.0 * q[1] * q[3] + 2.0 * q[2] * q[0]
    m[1,0] = 2.0 * q[1] * q[2] + 2.0 * q[3] * q[0]
    m[1,1] = 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[3] * q[3]
    m[1,2] = 2.0 * q[2] * q[3] - 2.0 * q[1] * q[0]
    m[2,0] = 2.0 * q[1] * q[3] - 2.0 * q[2] * q[0]
    m[2,1] = 2.0 * q[2] * q[3] + 2.0 * q[1] * q[0]
    m[2,2] = 1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2]
    
    return m


def AngleVectorFromMatrix(m):
    v = numpy.array([
        m[2][1] - m[1][2], 
        m[0][2] - m[2][0],
        m[1][0] - m[0][1]
        ], dtype = numpy.float64)
    a = numpy.linalg.norm(v)
    v /= a
    a = math.asin(a / 2.0)
    
    return (a, v)


def CreateQuaternionRotationAngleVector(a, v):
    v_magnitude = numpy.linalg.norm(v)
    q = numpy.array((1.0, 0.0, 0.0, 0.0), dtype = numpy.float64)
    if v_magnitude > 0.0 and abs(a) > 0.0:
        q = numpy.hstack((math.cos(a / 2.0), math.sin(a / 2.0) * v / v_magnitude))

    return q


def HamiltonProduct(q1, q2):
    p = numpy.zeros((4), dtype = numpy.float64)

    p[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    p[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    p[2] = q1[0] * q2[2] + q1[2] * q2[0] - q1[1] * q2[3] + q1[3] * q2[1]
    p[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]

    return p


def Conjugate(q):
    p = numpy.array(q, dtype = numpy.float64)
    p[1:] *= -1

    return p 


def RotateVectorByQuaternion(v, q):
    p = numpy.hstack((0.0, v))
    p = HamiltonProduct(HamiltonProduct(q, p), Conjugate(q))
    
    return p[1:]
