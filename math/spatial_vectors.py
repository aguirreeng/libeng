# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:55:36 2017

@author: aguirre
"""
import numpy

def obtain_skew_symmetric_matrix_from_vector(v: numpy.ndarray) -> numpy.ndarray:
    '''
    Function to generate the skew-symmetric matrix, the product of
    this matrix, which is generated based on a vector, by another
    vector is equivalent to the cross product (vector product) of the
    two vectors.
    '''
    m = numpy.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [v[1], v[0], 0.0]],
        dtype=numpy.float64
        )
    return m


def obtain_space_transform_matrices(
  position_vector: numpy.ndarray,
  rotation_matrix: numpy.ndarray
  ) -> tuple:
    '''
    Function to combine the frame translation vector and
    transformation matrix to generate the 6D transformation matrix.
    Valid for 3D or 2D spaces.
    '''
    # Generate auxiliary matrices.
    zero_matrix = numpy.zeros_like(rotation_matrix, dtype=numpy.float64)
    translation_skew_symmetric_matrix = obtain_skew_symmetric_matrix_from_vector(position_vector)

    # Calculate the space transform matrices (for both, motion and force spaces).
    motion_space_transform = numpy.block(
        [[rotation_matrix, zero_matrix],
         [numpy.matmul(-rotation_matrix, translation_skew_symmetric_matrix), rotation_matrix]]     
    )

    force_space_transform = numpy.block(
        [[rotation_matrix, numpy.matmul(-rotation_matrix, translation_skew_symmetric_matrix)],
         [zero_matrix, rotation_matrix]]
    )

    return (motion_space_transform, force_space_transform)


def obtain_spatial_inertia(
  mass: float,
  mass_center: numpy.ndarray,
  inertia_matrix: numpy.ndarray
  ) -> numpy.ndarray:
    identity_matrix = numpy.identity(inertia_matrix.shape[0], dtype=numpy.float64)
    mass_center_skew_symmetric_matrix = obtain_skew_symmetric_matrix_from_vector(mass_center)
    spatial_inertia = numpy.block(
        [[inertia_matrix + mass * numpy.matmul(
                mass_center_skew_symmetric_matrix,
                mass_center_skew_symmetric_matrix.transpose()),
            mass * mass_center_skew_symmetric_matrix],
            [mass * mass_center_skew_symmetric_matrix.transpose(), mass * identity_matrix]]
    )

    return spatial_inertia


def obtain_motion_skew_symmetric_matrix(v: numpy.ndarray) -> tuple:
    l = v.size // 2
    omega = v[:l]
    v0 = v[l:]
    omega_cross = obtain_skew_symmetric_matrix_from_vector(omega)
    v0_cross = obtain_skew_symmetric_matrix_from_vector(v0)
    zero = numpy.zeros_like(v0_cross)
    v_cross = numpy.block([[omega_cross, zero], [v0_cross, omega_cross]])
    v_cross_star = numpy.block([[omega_cross, v0_cross], [zero, omega_cross]])

    return v_cross, v_cross_star
