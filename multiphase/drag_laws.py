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


def Dallavalle1948(Re):
    '''
    Dallavalle1948 - Function to calculate drag coefficient using the correlation by
    Dallavalle (1948), this drag model is used by the standard Di Felice correlation
    for single particle drag coefficient calculation.

    parameters:
    . Re = Reynolds number based on fluid properties and relative velocity (physical)
	
    returns:
    . C_D = Dallavalle (1948) drag coefficient
	 
    references:
    . DALLAVALLE, J. M. Micromeritics: the Technology of Fine Particles, 2nd Edition,
    Pitman Publishing Corp., New York (1948)
    . DI FELICE, R.The Voidage Function for Fluid - Particle Interaction Systems.
    International Journal of Multiphase Flow, v. 20, n. 1, p. 153 - 159, 1994
	 
    notes:
    . calling function has to deal with the case of Re = 0.0!
    '''

    C_D = 0.0

    # DI FELICE (1994), page 154, equation (5) 
    C_D = np.power(0.63 + 4.8 / np.sqrt(Re), 2.0)

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


def GidaspowBezburuahDing1992(Re, alpha_f, phi = 1.0):
    '''
    GidaspowBezburuahDing1992 - Function to calculate the drag coefficient for dense dense 
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


def HuilinGidaspow2003(Re, alpha_f, phi = 1.0):
    '''
    HuilinGidaspow2003 - Function to calculate the drag coefficient for dense dense particle
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
    psi = np.arctan(262.5 * (0.8 - alpha_f)) / np.pi + 0.5
      
    # HUILIN & GIDASPOW (2003), page 3781, equation (18)
    # ANSYS, Inc. (2013), section 17.5.6.2, equation (17-231)
    C_D = psi * C_D_Ergun + (1.0 - psi) * C_D_WenYu
    
    return C_D


def DiFelice1994(Re, alpha_f = 1.0):
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
    
    C_D0 = Dallavalle1948(alpha_f * Re)

    # DI FELICE (1994), page 159, equation (30)
    x = 1.5 - np.log10(alpha_f * Re)
    beta = 3.7 - 0.65 * np.exp(-(x * x) / 2.0);
    
    # DI FELICE (1994), pages 154 and 156, equations (6) and (21)
    C_D = C_D0 * np.power(alpha_f, 2.0-beta);
      
    return C_D
    
    
def HillKochLadd2001(Re, alpha_f = 1.0):
    '''
    KochHillLadd2001 - Function to calculate the drag coefficient for dense dense particle
    flow using the correlation from Koch, Hill & Ladd (2001). 
    Using Lattice-Boltzmann simulation results, Koch, Hill and Ladd (2001) have calculated 
    drag exerted by a fluid flow on a collection of randomly dispersed, fixed particles for 
    various values of Reynolds numbers and solid volume fractions and reported a functional 
    representation which was precisely fit to this data. 
    However, Koch, Hill and Ladd (2001) did not came up with a composite formula that is 
    applicable to the entire range of Re numbers and solid volume fractions. 
    Since multiphase flow models need constitutive relations that smoothly cover a wide range 
    of Re and volume fractions without gaps, Benyahia, Syamlal and O'Brien analyzed the results 
    reported by Hill et al. and extended them to the full range of Reynolds numbers and solids 
    volume fraction, discussing some of the issues they found when blending formulas.

    parameters:
    . Re_H = Reynolds number based on fluid properties and relative velocity
    This Reynolds number is different than regular Re as it is based on radius instead of diameter 
    and fluid volume fraction
    . alpha_f = volume fraction of the continuous phase(porosity)
	 
    returns:
    . C_D = KochHillLadd (2001) drag coefficient
	 
    references:
    . R.J. Hill, D.L. Koch, J.C. Ladd, The first effects of fluid inertia on flows in ordered 
    and random arrays of spheres, J. Fluid Mech., v. 448, p. 213–241, 2001. 
    . R.J. Hill, D.L. Koch, J.C. Ladd, Moderate-Reynolds-number flows in ordered and random 
    arrays of spheres, J. Fluid Mech.,  v. 448, p. 243–278, 2001
    . Benyahia, Sofiane; Syamlal, Madhava; O'Brien, Thomas J., Extension of Hill–Koch–Ladd 
    drag correlation over all ranges of Reynolds number and solids volume fraction, 
    Powder Technology,  v. 162, p. 166–174, 2006
    
    notes:
    . Implemented by Lucilla.
    '''
    
    alpha_s = 1- alpha_f
    Re_H = Re*alpha_f/2.0
    
    w = np.exp(-10.0*(0.4 - alpha_s)/alpha_s)
    
    #calculating F0
    if alpha_s > 0.01 and alpha_s < 0.4:
        F0 = (1.0-w) * ((1.0+(3.0 * np.sqrt(alpha_s/2.0))
            + (135.0/64.0)*alpha_s*np.log(alpha_s) + 17.14*alpha_s )/(1.0 + 0.681 *alpha_s - 8.48 * alpha_s * alpha_s + 8.16 * alpha_s * alpha_s * alpha_s))\
            + w* (10* alpha_s/np.power((1.0 - alpha_s), 3.0))
    else:
        F0 = (10* alpha_s/np.power((1.0 - alpha_s), 3.0))
        
    #calculating F1  
    if alpha_s > 0.01 and alpha_s < 0.1:
        F1 = np.sqrt(2.0/alpha_s)/40
    else:
        F1 = 0.11 + 0.00051 * np.exp(11.6* alpha_s)
    
    #calculating F2
    if alpha_s< 0.4:
        F2 = (1.0-w) * ((1.0+(3.0 * np.sqrt(alpha_s/2.0))
            + (135.0/64.0)*alpha_s*np.log(alpha_s) + 17.89*alpha_s )/(1.0 + 0.681 *alpha_s - 11.03 * alpha_s * alpha_s + 15.41 * alpha_s * alpha_s * alpha_s))\
            + w* (10* alpha_s/np.power((1.0 - alpha_s), 3.0))
    else:
        F2 = (10* alpha_s/np.power((1.0 - alpha_s), 3.0))
    
    #calculating F3
    if alpha_s<0.00953:
        F3 = 0.9351*alpha_s+0.03667
    else:
        F3 = 0.0673 + 0.212*alpha_s+ 0.0232/np.power((1.0-alpha_s), 5.0)
    
    # calculating F
    
    if alpha_s<=0.01 and Re_H <= (F2-1.0)/(3.0/8.0 - F3):
        F = 1.0 + (3.0/8.0)*Re_H
        
    elif alpha_s<=0.01 and Re_H > (F2-1.0)/(3.0/8.0 - F3):   
        F = F2 + F3 * Re_H
        
    elif alpha_s > 0.01 and Re_H <= (F3 + np.sqrt(F3*F3 - 4*F1 *(F0 - F2)))/(2*F1):
        F = F0 + F1*Re_H*Re_H
        
    else:
        F = F2 + F3*Re_H
        
    #Calculating CD:
    #CD = (12.0*(1.0-alpha_s)*(1.0-alpha_s)/Re_H)*F
    CD = (24.0*(1.0-alpha_s)*(1.0-alpha_s)/Re)*F
        
    return CD