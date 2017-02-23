# -*- coding: utf-8 -*-
"""
Module with functions to calculate drag coefficients for flows with particles, droplets and bubbles
or around fixed bodies.

Joao Aguirre
"""

import numpy as np


def SchillerNaumann1933(Re, modified = True):
    '''
    SchillerNaumann1933 - Function to calculate drag coefficient using the correlation by
    Schiller & Naumann (1933) with the modification found at ANSYS Inc. (2013).
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . modified = boolean to define the usage of the modified version of the Schiller & Naumann 
                 correlation (including the inertial range Re > 1000)

    returns:
    . C_D = Schiller & Naumann (1993) drag coefficient

    references:
    . CROWE, Clayton T.; SCHWARZKOPF, John D.; SOMMERFELD, Martin; TSUJI, Yutaka.
      Multiphase Flows with Droplets and Particles. 2. ed. Boca Raton, FL: CRC Press, 2012.
    . GIDASPOW, Dimitri. Multiphase Flow and Fluidization: Continuum and Kinetic Theory 
      Descriptions. San Diego, CA: Academic Press, 1993.
    . ANSYS, Inc. CFX Solver Theory Guide. Canonsburg, PA, 2013.

    TODO:
    . make the function valid only for Re > 0.0!
    '''
    
    # CROWE et al. (2012), page 74, equation (4.55)
    C_D = 24. / Re * (1. + 0.15 * np.power(Re, 0.687))
    
    # compute the modified version for Re > 1000.0
    # ANSYS (2013), section 5.5.2.1.1, equation (5-47)
    if modified:
        C_D = np.maximum(C_D, np.full_like(C_D, 0.44))
        
    return C_D


def HaiderLevenspiel1989(Re, phi = 1.0):
    '''
    HaiderLevenspiel1989 - Function to calculate drag coefficient of spherical and non-spherical 
    particles using the correlation by Haider & Levenspiel (1989).
    
    parameters:
    . Re = Reynolds number based on fluid properties, relative velocity and a equivalent spherical 
           particle diameter
    . phi = particle sphericity (ratio of the area of a spherical particle with the same volume of 
            the actual particle by the area of the actual particle)
    
    references:
    . HAIDER, A.; LEVENSPIEL, O. Drag Coefficient and Terminal Velocity of Spherical and 
      Nonspherical Particles. Powder Technology, v. 58, p. 63-70, 1989.
    . ANSYS, Inc. Fluent Theory Guide. Canonsburg, PA, 2013.
    
    returns:
    . C_D = Haider & Levenspiel (1989) drag coefficient
    
    notes:
    . correlation for spherical particles:
      - extracted from 408 experimental data points
      - valid for Re < 2.6e+05
      - RMS deviation = 2.4%
    . correlation for nonspherical particles:
      - extracted from 419 isometric experimental data points and 87 disk data points
      - valid for Re < 25000 for isometric particles and for Re < 500 for disks
      - RMS deviation = 3.0% for isometric particles
    . correlation used by ANSYS FLUENT for nonspherical particles
    
    TODO:
    . make the function valid only for Re > 0.0!
    '''
    
    # if the particles are spherical uses the fixed coefficient values
    if(phi == 1.0):
        # HAIDER & LEVENSPIEL (1989), page 64, equation (6)
        A = 0.1806
        B = 0.6459
        C = 0.4251
        D = 6880.95
    # if the particles are non-spherical calculates the coefficients
    else:
        # HAIDER & LEVENSPIEL (1989), page 65, equations (10a), (10b), (10c) and (10d)
        A = np.exp(2.3288 - 6.4581 * phi + 2.4486 * np.power(phi, 2.0))
        B = 0.0964 + 0.5565 * phi
        C = np.exp(4.905 - 13.8944 * phi + 18.4222 * np.power(phi, 2.0) - 10.2599 * np.power(phi, 3.0))
        D = np.exp(1.4681 + 12.2584 * phi - 20.7322 * np.power(phi, 2.0) + 15.8855 * np.power(phi, 3.0))
    
    # drag coefficient calculation
    # HAIDER & LEVENSPIEL (1989), page 63, equation (4)
    C_D = 24.0 / Re * (1.0 + A * np.power(Re, B)) + C / (1.0 + D / Re)
    
    return C_D


def Ganser1993(Re, phi = 1.0, d_v = 1.0, d_n = 1.0, D = 1.0e+20):
    '''
    Ganser1993 - Function to calculate drag coefficient of spherical and non-spherical particles 
    using the correlation by Ganser (1993).
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . phi = particle sphericity (ratio of the area of a spherical particle with the same volume of 
            the actual particle by the area of the actual particle)
    . d_v = diameter of a spherical particle with the same volume of the actual particle
    . d_n = diameter of a spherical particle with the same projected area of the actual particle
            in the direction of the flow
    . D = diameter of the duct where the drag is beign calculated
    
    notes:
    . all dimensions provided via parameters are used in ratios, so their units do not need to be 
      in metric units, but they need to be consistent
    
    references:
    . GANSER, Gary H. A rational approach to drag prediction of spherical and nonspherical 
      particles. Powder Technology, v. 77, p. 143-152, 1993.
    
    TODO:
    . make the function valid only for Re > 0.0!
    '''
    
    # Stokes form factor
    # GANSER (1993), page 151, table 7
    K_1 = 1.0 / (0.333333333333 * d_n / d_v + 0.666666666667 / np.sqrt(phi)) - 2.25 * d_v / D    
    
    # Newton form factor
    # GANSER (1993), page 151, table 7
    K_2 = np.power(10.0, np.power(1.8147 * (- np.log10(phi)), 0.5743))    
    
    # Ganser drag coefficient
    # GANSER (1993), page 150, equation (18)
    ReK1K2 = Re * K_1 * K_2
    C_D = K_2 * (24.0 / ReK1K2 * (1.0 + 0.1118 * np.power(ReK1K2, 0.6567)) + 0.4305 / (1.0 + 3305 / ReK1K2))
    
    return C_D


def WenYu1966(Re, alpha_f = 1.0):
    '''
    WenYu1966 - Function to calculate the drag coefficient for dense dense particle
    flow using the correlation by Wen & Yu (1966). The correlation presents itself as
    a modification on the relative particle Reynolds number and over the drag coefficient
    for a single particle calculated using the Schiller & Naumann (1933) correlation.
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . alpha_f = volume fraction of the continuous phase (porosity)

    returns:
    . C_D = Yen & Yu (1966) drag coefficient

    references:
    . CROWE, Clayton T.; SCHWARZKOPF, John D.; SOMMERFELD, Martin; TSUJI, Yutaka.
      Multiphase Flows with Droplets and Particles. 2. ed. Boca Raton, FL: CRC Press, 2012.
    . GIDASPOW, Dimitri. Multiphase Flow and Fluidization: Continuum and Kinetic Theory 
      Descriptions. San Diego, CA: Academic Press, 1993.
    . ANSYS, Inc. Fluent Theory Guide. Canonsburg, PA, 2013.

    notes:
    . the Wen & Yu (1966) correlation is valid for alpha_f > 0.8!

    TODO:
    . make the function valid only for Re > 0.0!
    '''

    # GIDASPOW (1993), page 37, equations (2.12), (2.13), (2.14), (2.15) and (2.16)
    # ANSYS, Inc. (2013), section 17.5.6.2, equation (17-226) and (17-227)
    C_D = np.power(alpha_f, -1.65) * SchillerNaumann1933(Re * alpha_f)
    
    return C_D


def Ergun1958(Re, alpha_f, phi = 1.0):
    '''
    Ergun1958 - Function to calculate the drag coefficient for dense dense particle
    flow using the correlation derived from the Ergun (1958) equation. The correlation was 
    derived from the Ergun equation, that is a redefinition of the phases momentum transfer 
    coefficient K_sf, and the standard definition of the phases momentum transfer 
    coefficient for multiphase CFD applications, converting the new definition of K_sf 
    into a new definition of a drag coefficient (C_D).
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . alpha_f = volume fraction of the continuous phase (porosity)
    . phi = particles sphericity

    returns:
    . C_D = Ergun (1958) drag coefficient

    references:
    . GIDASPOW, Dimitri. Multiphase Flow and Fluidization: Continuum and Kinetic Theory 
      Descriptions. San Diego, CA: Academic Press, 1993.
    . ANSYS, Inc. Fluent Theory Guide. Canonsburg, PA, 2013.

    notes:
    . the Ergun (1958) correlation is valid for alpha_f < 0.8

    TODO:
    . make the function valid only for Re = 0.0 and alpha_f = 1.0!
    '''

    # GIDASPOW (1993), page 36, equation (2.11)
    # ANSYS, Inc. (2013), section 17.5.6.2, equation (17-230)
    # equation modified to define a C_D according to ANSYS, Inc. (2013)
    # sections 17.5.5 and 17.5.6.1, equations (17-161), (17-164), (17-165) and (17-166) 
    C_D = 200. * (1.0 - alpha_f) / (alpha_f * phi * phi * Re) + 2.333333333333 / phi
    
    return C_D


def GuidaspowBezburuahDing1992(Re, alpha_f, phi = 1.0):
    '''
    GuidaspowBezburuahDing1992 - Function to calculate the drag coefficient for dense dense 
    particle flow using the correlation from Guidaspow, Bezburuah & Ding (1992). The correlation 
    is a direct blend between the Wen & Yu (1966) correlation and the Ergun (1958) equation.
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . alpha_f = volume fraction of the continuous phase (porosity)
    . phi = particles sphericity

    returns:
    . C_D = Guidaspow, Bezburuah & Ding (1992) drag coefficient

    references:
    . GIDASPOW, Dimitri; BEZBURUAH, Rukmini; DING, J. Hydrodynamics of Circulating Fuidized Beds: 
      Kinetic Theory Approach. In: ENGINEERING FOUNDATION CONFERENCE ON FLUIDIZATION, 7, 1992, 
      Brisbane, Australia. Proceedings... New York, NY: Engineering Foundation, 1992. p. 75-82.
    . GIDASPOW, Dimitri. Multiphase Flow and Fluidization: Continuum and Kinetic Theory 
      Descriptions. San Diego, CA: Academic Press, 1993.
    . ANSYS, Inc. Fluent Theory Guide. Canonsburg, PA, 2013.

    notes:
    . the Wen & Yu (1966) correlation is valid for alpha_f > 0.8
    . the Ergun (1958) correlation is valid for alpha_f < 0.8
    . the Schiller & Naumann (1933) modified correlation used on the Wen & Yu (1966) correlation is
      implemented with the maximum function instead of the Re = 1000 criteria
    
    TODO:
    . make the function valid only for Re = 0.0 and alpha_f = 1.0!
    '''

    if(alpha_f > 0.8):
        # GIDASPOW, BEZBURUAH & DING (1992), page 78(4), equations (T1-7b), (T1-7c), (T1-7d) and (T1-7e)
        # ANSYS, Inc. (2013), section 17.5.6.2, equations (17-228) and (17-229)
        C_D = WenYu1966(Re, alpha_f)
    else:
        # GIDASPOW, BEZBURUAH & DING (1992), page 78(4), equation (T1-7a)
        # ANSYS, Inc. (2013), section 17.5.6.2, equation (17-230)
        C_D = Ergun1958(Re, alpha_f, phi)
    
    return C_D


def HuilinGuidaspow2003(Re, alpha_f, phi = 1.0):
    '''
    HuilinGuidaspow2003 - Function to calculate the drag coefficient for dense dense particle
    flow using the correlation from Huilin & Guidaspow (2003). The correlation was 
    derived to provide a smooth transition between the Wen & Yu (1966) correlation and the
    Ergun (1958) equation.
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . alpha_f = volume fraction of the continuous phase (porosity)
    . phi = particles sphericity

    returns:
    . C_D = Huilin & Guidaspow (2003) drag coefficient

    references:
    . HUILIN, Lu; GIDASPOW, Dimitri. Hydrodynamics of binary Fuidization in a riser: 
      CFD simulation using two granular temperatures. Chemical Engineering Science, 
      v. 58, p. 3777-3792, 2003.
    . ANSYS, Inc. Fluent Theory Guide. Canonsburg, PA, 2013.

    notes:
    . the Wen & Yu (1966) correlation is valid for alpha_f > 0.8
    . the Ergun (1958) correlation is valid for alpha_f < 0.8
    . the Schiller & Naumann (1933) modified correlation used on the Wen & Yu (1966) correlation is
      implemented with the maximum function instead of the Re = 1000 criteria

    TODO:
    . make the function valid only for Re = 0.0 and alpha_f = 1.0!
    '''

    C_D_WenYu = WenYu1966(Re, alpha_f)
    C_D_Ergun = Ergun1958(Re, alpha_f, phi)
    
    # HUILIN & GIDASPOW (2003), page 3781, equation (19)
    # ANSYS, Inc. (2013), section 17.5.6.2, equation (17-232)
    # modified to use the fluid volume fraction alpha_s = 1.0 - alpha_f
    psi = np.arctan(262.5 * (alpha_f - 0.8) / np.pi) + 0.5
      
    # HUILIN & GIDASPOW (2003), page 3781, equation (18)
    # ANSYS, Inc. (2013), section 17.5.6.2, equation (17-231)
    C_D = psi * C_D_Ergun + (1.0 - psi) * C_D_WenYu
    
    return C_D


def DiFelice1994(Re, C_D0, alpha_f = 1.0):
    '''
    DiFelice1994 - Function to calculate the drag coefficient for dense dense particle
    flow using the correlation from Di Felice (1994). The correlation was developed as a 
    correction to be applied over the drag coefficient of a single particle and can be 
    applied over drag coefficients of spherical and non-spherical particles.
    
    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity
    . C_D0 = drag coefficient of a single particle at the same Reynolds number 
    . alpha_f = volume fraction of the continuous phase (porosity)

    returns:
    . C_D = Di Felice (1994) drag coefficient

    references:
    . DI FELICE, R. The Voidage Function for Fluid-Particle Interaction Systems. 
      International Journal of Multiphase Flow, v. 20, n. 1, p. 153-159, 1994.


    notes:

    TODO:
    . make the function valid only for Re = 0.0!
    '''

    # DI FELICE (1994), page 159, equation (30)
    x = 1.5 - np.log10(Re)
    beta = 3.7 - 0.65 * np.exp(-(x * x) / 2.0);
    
    # DI FELICE (1994), pages 154 and 156, equations (6) and (21)
    C_D = C_D0 * np.power(alpha_f, -beta);
      
    return C_D