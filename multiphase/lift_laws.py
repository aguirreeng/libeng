"""
Module with functions to calculate lift coefficients for flows with
particles, droplets and bubbles around fixed bodies.

.. module:: lift_laws
    :synopsis: Lift laws for particles, droplets and bubbles.

.. moduleauthor:: Joao Aguirre <aguirre.eng@gmail.com>
"""

import sys
import numpy

def Saffman1968(reynolds_omega):
    '''
    .. py:function:: Saffman1968(reynolds_omega)
    
    Function to calculate the lift coefficient using the correlation by
    Saffman (1968) in the form given by Mei (1992).
    
    :param reynolds_omega: Reynolds number based on fluid properties
    and velocity rotational magnitude [ ].
    
    :return: Saffman (1965) lift coefficient [ ].
    
    references:
    . MEI, Renwei. An Approximate Expression for the Shear Lift Force on a Spherical Particle at
      Finite Reynolds Number. International Journal of Multiphase Flow, volume 18, number 1,
      pages 145-147, brief communication, 1992.
    . CROWE, Clayton T.; SCHWARZKOPF, John D.; SOMMERFELD, Martin; TSUJI, Yutaka.
      Multiphase Flows with Droplets and Particles. 2nd Edition, Boca Raton, FL: CRC Press,
      ISBN 978-1-4398-4051-1, 2012.
    . ANSYS, Inc. Fluent Theory Guide. Release 16.0. Canonsburg, PA, 2014.
    
    notes:
    . Valid for spherical particles.
    . Valid for Re << sqrt(Re_omega) << 1.
    . calling function has to deal with the case of Re_omega = 0.0!
    '''
    
    C_L = 0.0
    
    # MEI (1992), equation (1)
    # CROWE et al. (2012), equation (4-142)
    # ANSYS (2014), section 17.5.7.1.2, equation (17-258)
    # equation modified to the non-dimensional form and using the lift
    # coefficient definition */
    C_L = 6.46 * 3.0 / (2.0 * numpy.pi * numpy.sqrt(reynolds_omega))
    
    return C_L

def Mei1992(reynolds, reynolds_omega):
    '''
    .. py:function:: Mei1992(reynolds, re_omega)
    
    Function to calculate the lift coefficient using the correlation
    by Mei (1992).
    
    :param reynolds: Reynolds number based on fluid properties
	and relative velocity [ ].
    :param reynolds_omega: Reynolds number based on fluid phase 
	velocity gradient magnitude [ ].
    
    :return: Mei (1992) lift coefficient [ ].
    
    references:
    . MEI, Renwei. An Approximate Expression for the Shear Lift Force on a Spherical Particle at
      Finite Reynolds Number. International Journal of Multiphase Flow, volume 18, number 1,
      pages 145-147, brief communication, 1992.
    . CROWE, Clayton T.; SCHWARZKOPF, John D.; SOMMERFELD, Martin; TSUJI, Yutaka.
      Multiphase Flows with Droplets and Particles. 2nd Edition, Boca Raton, FL: CRC Press,
      ISBN 978-1-4398-4051-1, 2012.
    . ANSYS, Inc. Fluent Theory Guide. Release 16.0. Canonsburg, PA, 2014.
    
    notes:
    . Valid for spherical particles.
    . Valid for 0.1 <= Re <= 100, 0.005 <= alpha <= 0.4, or 0.1 <= epsilon <= 20,
      with epsilon = sqrt(Re_Omega)/Re and alpha = 1/2 * Re * epsilon ^ 2.
    . calling function has to deal with the case of Re = 0.0 and Re_Omega = 0.0!
    '''
	
    C_L = 0.0
    alpha = 0.0

    # MEI (1992) equations (5) and (6)
    alpha = 0.5 * reynolds_omega / reynolds

    # MEI (1992) equation (8)
    # CROWE et al. (2012), equations (4-145) and (4.146)
    # ANSYS (2014), section 17.5.7.1.2, equation (17-259)
    if reynolds <= 40.0:
        C_L = (1.0 - 0.3314 * numpy.sqrt(alpha)) * numpy.exp(-reynolds / 10.0) + 0.3314 * numpy.sqrt(alpha)
    else:
        C_L = 0.0524 * numpy.sqrt(alpha * reynolds)

    C_L *= 6.46 * 3.0 / (2.0 * numpy.pi * numpy.sqrt(reynolds_omega))

    return C_L
