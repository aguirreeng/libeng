# -*- coding: utf-8 -*-
"""
Module with functions to calculate the Nusselt number and 
heat transfer coefficient for flows with particles, 
droplets and bubbles or around fixed bodies.

Joao Aguirre
"""



import numpy as np



def RanzMarshall1952(Re, Pr):
    '''
    RanzMarshall1952 - Function to calculate the Nusselt 
    (or Sherwood) number for the heat (or mass) transfer 
    using the Rans & Marshall (1952) correlation for 
    spherical particles.
    
    parameters:
    . Re = Reynolds number based on particle diameter and
      relative velocity
    . Pr = Prandtl number based on fluid properties 
      (or Schmidt number for mass transfer)
    
    returns:
    . Nu = Rans & Marshall (1952) Nusselt (or Sherwood) 
      number
    
    references:
    . BERGMAN, Theodore L.; LAVINE, Adrienne S.; 
      INCROPERA, Frank P. and DEWITT, David P.; 
      Fundamentals of Heat and Mass Transfer, 7th Edition, 
      John Wiley & Sons, 2011.
    . MICHAELIDES, E. E.; Particles, Bubbles & Drops, 
      Their Motion, Heat and Mass Transfer, 
      World Scientific, 2006.
    
    notes:
    . This correlation is valid for Re < 5*10^4.
    '''
    
    # BERGMAN (2011), section 7.5, page 465, equation (7.57)
    # MICHAELIDES (2006), section 4.2.2, page 121, equation (4.2.6)
    Nu = 2.0 + 0.6 * np.sqrt(Re) * np.power(Pr, 1.0 / 3.0)

    return Nu


def Whitaker1972(Re, Pr, mu_ratio = 1.0):
    '''
    Whitaker1972 - Function to calculate the Nusselt (or 
    Sherwood) number for the heat (or mass) transfer 
    using the Whitaker (1972) correlation for spherical 
    particles.
    
    parameters:
    . Re = Reynolds number based on particle diameter and 
      relative velocity
    . Pr = Prandtl number based on fluid properties 
      (or Schmidt number for mass transfer)
    . mu_ratio = ratio of the viscosity at bulk temperature 
      and wall temperature
    
    returns:
    . Nu = Whitaker (1972) Nusselt (or Sherwood) number
    
    references:
    . BERGMAN, Theodore L.; LAVINE, Adrienne S.; 
      INCROPERA, Frank P. and DEWITT, David P.;
      Fundamentals of Heat and Mass Transfer, 7th Edition, 
      John Wiley & Sons, 2011.
    . BEJAN, Adrian; Convection Heat Transfer, 3rd Edition, 
      John Wiley & Sons, 2004.
    
    notes:
    . This correlation is valid for 0.71 < Pr < 380, 
      3.5 < Re < 7.6*10^4 and 1 < mu_r < 3.2.
    '''
    
    # BERGMAN (2011), section 7.5, page 465, equation (7.56)
    # BEJAN (2004), section 7.9.2, page 365, equation (7.104)
    Nu = 2.0 + (0.4 * np.sqrt(Re) + 0.06 * np.power(Re, 2.0 / 3.0)) \
        * np.power(Pr, 0.4) * np.power(mu_ratio, 0.25)

    return Nu


def Gunn1978(Re, Pr, alpha_f):
    '''
    Gunn1978 - Function to calculate the Nusselt 
    (or Sherwood) number for the heat (or mass) transfer
    using the Gunn (1978) correlation for dense flow of
    spherical particles.
    
    parameters:
    . Re = Reynolds number based on particle diameter and 
      relative velocity
    . Pr = Prandtl number based on fluid properties 
      (or Schmidt number for mass transfer)
    . alpha_f = fluid volume (void) fraction
    
    returns:
    . Nu = Gunn (1978) Nusselt (or Sherwood) number
    
    references:
    . GUNN, D.; Transfer of Heat or Mass to Particles in 
      Fixed and Fluidized Beds, International Journal of 
      Heat and Mass Transfer, volume 21, issue 4, 
      pages 467-476, 1978.
    
    notes:
    . This correlation is valid for Re < 10^5 and 
      0.35 < alpha_f < 1.
    '''

    # GUNN (1978), page 473, equation (4.2)
    Nu = (7.0 - 10.0 * alpha_f + 5.0 * alpha_f * alpha_f) \
        * (1.0 + 0.7 * pow(Re, 0.2) * pow(Pr, 0.3333333333)) \
        + (1.33 - 2.4 * alpha_f + 1.2 * alpha_f * alpha_f) \
        * pow(Re, 0.7) * pow(Pr, 0.3333333333)

    return Nu
