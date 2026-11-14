"""
UNESCO (1981) Equation of State of Seawater.

Direct Python translation of UNESCO.f from the Gildor & Tziperman (2001) 8-box model:
    Gill, A., Atmosphere-Ocean Dynamics (1982), pp 599-600.
    UNESCO, 10th report of the joint panel on oceanographic tables and standards (1981).
"""

import numpy as np


def unesco_density(s: float, t_celsius: float, p_dbar: float = 0.0) -> float:
    """
    Compute seawater density anomaly sigma = rho - 1000.0 (kg/m^3).
    
    Args:
        s: Salinity in practical salinity units (psu)
        t_celsius: In-situ temperature in degrees Celsius (°C)
        p_dbar: Pressure in decibars (approx. meters of depth)
        
    Returns:
        sigma: Density minus 1000 kg/m^3
    """
    t = float(t_celsius)
    s = float(s)
    p = float(p_dbar * 0.1)  # convert from decibars to bars
    c1p5 = 1.5

    # Pure water density at atmospheric pressure
    rw = (9.99842594e2 
          + 6.793952e-2 * t 
          - 9.095290e-3 * t**2 
          + 1.001685e-4 * t**3 
          - 1.120083e-6 * t**4 
          + 6.536332e-9 * t**5)

    # Seawater density at atmospheric pressure
    rsto = (rw + (8.24493e-1 
                  - 4.0899e-3 * t 
                  + 7.6438e-5 * t**2 
                  - 8.2467e-7 * t**3 
                  + 5.3875e-9 * t**4) * s
            + (-5.72466e-3 
               + 1.0227e-4 * t 
               - 1.6546e-6 * t**2) * (s**c1p5)
            + 4.8314e-4 * s**2)

    # Secant bulk modulus of pure water
    xkw = (1.965221e4 
           + 1.484206e2 * t 
           - 2.327105 * t**2 
           + 1.360477e-2 * t**3 
           - 5.155288e-5 * t**4)

    # Secant bulk modulus at atmospheric pressure
    xksto = (xkw + (5.46746e1 
                    - 6.03459e-1 * t 
                    + 1.09987e-2 * t**2 
                    - 6.1670e-5 * t**3) * s
             + (7.944e-2 
                + 1.6483e-2 * t 
                - 5.3009e-4 * t**2) * (s**c1p5))

    # Secant bulk modulus at pressure p
    xkstp = (xksto + (3.239908 
                     + 1.43713e-3 * t 
                     + 1.16092e-4 * t**2 
                     - 5.77905e-7 * t**3) * p
             + (2.2838e-3 
                - 1.0981e-5 * t 
                - 1.6078e-6 * t**2) * p * s
             + 1.91075e-4 * p * (s**c1p5)
             + (8.50935e-5 
                - 6.12293e-6 * t 
                + 5.2787e-8 * t**2) * (p**2)
             + (-9.9348e-7 
                + 2.0816e-8 * t 
                + 9.1697e-10 * t**2) * (p**2) * s)

    # In-situ density
    rho = rsto / (1.0 - p / xkstp) if p != 0.0 else rsto
    return rho - 1000.0


def unesco_full_density(s: float, t_celsius: float, p_dbar: float = 0.0) -> float:
    """Return total density rho (kg/m^3)."""
    return unesco_density(s, t_celsius, p_dbar) + 1000.0
