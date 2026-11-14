"""
Wood et al. (2019) 5-Box Model of the Atlantic Meridional Overturning Circulation (AMOC)

Reference:
    Wood, R. A., Rodríguez, J. M., Smith, R. S., Jackson, L. C., & Hawkins, E. (2019).
    Observable, low-order dynamical controls on thresholds of the Atlantic meridional
    overturning circulation. Climate Dynamics, 53(11), 6815–6834.
    https://doi.org/10.1007/s00382-019-04956-1
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.integrate import odeint


# Seconds per standard year used in climate models (365.25 days)
SEC_PER_YEAR = 365.25 * 86400.0

# Conversion factor:
# Transports/Fluxes in Sv (10^6 m^3/s), Volumes in 10^16 m^3.
# Rate of change per second = (Sv / V) * 10^6 / 10^16 = 10^-10 s^-1.
# Rate of change per year = 10^-10 * SEC_PER_YEAR = 3.15576e-3 yr^-1.
CONV_PER_YEAR = SEC_PER_YEAR * 1e-10


class BoxModelParams:
    """Parameters for the 5-box AMOC model (Table 1 in Wood et al. 2019)."""

    def __init__(
        self,
        name: str,
        VN: float,
        VT: float,
        VS: float,
        VIP: float,
        VB: float,
        AN: float,
        AT: float,
        AS: float,
        AIP: float,
        FN: float,
        FS: float,
        FT: float,
        FIP: float,
        TS: float,
        T0: float,
        mu: float,
        lam: float,
        KN: float,
        KS: float,
        KIP: float,
        eta: float,
        gamma: float,
        alpha: float = 0.12,
        beta: float = 0.79,
        S0: float = 35.0,
        init_salinities: Optional[List[float]] = None
    ):
        self.name = name
        self.VN = float(VN)
        self.VT = float(VT)
        self.VS = float(VS)
        self.VIP = float(VIP)
        self.VB = float(VB)
        self.AN = float(AN)
        self.AT = float(AT)
        self.AS = float(AS)
        self.AIP = float(AIP)
        self.FN = float(FN)
        self.FS = float(FS)
        self.FT = float(FT)
        self.FIP = float(FIP)
        self.TS = float(TS)
        self.T0 = float(T0)
        self.mu = float(mu)
        self.lam = float(lam)
        self.KN = float(KN)
        self.KS = float(KS)
        self.KIP = float(KIP)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.S0 = float(S0)
        self.init_salinities = init_salinities


# All model calibrations from Table 1 of Wood et al. (2019)
CALIBRATIONS: Dict[str, BoxModelParams] = {
    "FAMOUSA_1xCO2": BoxModelParams(
        name="FAMOUSA (1xCO2)",
        VN=3.683, VT=5.151, VS=10.28, VIP=21.29, VB=88.12,
        AN=0.194, AT=0.597, AS=-0.226, AIP=-0.565,
        FN=0.375, FS=1.014, FT=-0.723, FIP=-0.666,
        TS=5.571, T0=3.26, mu=7.0, lam=2.66,
        KN=5.439, KS=1.880, KIP=89.778, eta=66.061,
        gamma=0.58,
        init_salinities=[35.088, 35.670, 34.441, 34.689, 34.577]
    ),
    "FAMOUSB_1xCO2": BoxModelParams(
        name="FAMOUSB (1xCO2)",
        VN=3.261, VT=7.777, VS=8.897, VIP=22.02, VB=86.490,
        AN=0.070, AT=0.752, AS=-0.257, AIP=-0.565,
        FN=0.384, FS=1.078, FT=-0.723, FIP=-0.739,
        TS=4.773, T0=2.65, mu=5.5, lam=2.79,
        KN=5.456, KS=5.447, KIP=96.817, eta=74.492,
        gamma=0.39,
        init_salinities=[35.247, 35.656, 34.428, 34.692, 34.568]
    ),
    "FAMOUSB_2xCO2": BoxModelParams(
        name="FAMOUSB (2xCO2)",
        VN=3.683, VT=5.418, VS=6.097, VIP=14.86, VB=99.25,
        AN=0.131, AT=0.696, AS=-0.263, AIP=-0.564,
        FN=0.486, FS=1.265, FT=-0.997, FIP=-0.754,
        TS=7.919, T0=3.87, mu=22.0, lam=1.62,
        KN=1.762, KS=1.872, KIP=99.977, eta=33.264,
        gamma=0.36,
        init_salinities=[35.418, 36.195, 34.670, 34.908, 34.789]
    ),
    "HadGEM2AO_1xCO2": BoxModelParams(
        name="HadGEM2-AO (1xCO2)",
        VN=3.557, VT=8.908, VS=10.330, VIP=19.219, VB=90.23,
        AN=0.117, AT=0.703, AS=-0.303, AIP=-0.517,
        FN=0.453, FS=0.901, FT=-0.798, FIP=-0.556,
        TS=6.456, T0=2.71, mu=1.4, lam=2.17,
        KN=5.601, KS=7.169, KIP=459.095, eta=3.758,
        gamma=0.85,
        init_salinities=[35.034, 35.617, 34.455, 34.662, 34.582]
    ),
    "HadGEM2AO_2xCO2": BoxModelParams(
        name="HadGEM2-AO (2xCO2)",
        VN=5.259, VT=7.400, VS=9.336, VIP=19.220, VB=89.90,
        AN=0.285, AT=0.522, AS=-0.299, AIP=-0.508,
        FN=0.496, FS=1.021, FT=-0.921, FIP=-0.596,
        TS=7.424, T0=3.29, mu=16.0, lam=1.66,
        KN=15.890, KS=6.828, KIP=1029.641, eta=9.871,
        gamma=0.73,
        init_salinities=[35.148, 35.802, 34.512, 34.715, 34.650]
    ),
    "HadGEM2AO_4xCO2": BoxModelParams(
        name="HadGEM2-AO (4xCO2)",
        VN=5.257, VT=7.454, VS=9.462, VIP=19.155, VB=90.78,
        AN=0.197, AT=0.620, AS=-0.326, AIP=-0.491,
        FN=0.577, FS=1.114, FT=-1.099, FIP=-0.592,
        TS=8.710, T0=3.70, mu=28.0, lam=1.28,
        KN=20.954, KS=8.384, KIP=477.332, eta=6.773,
        gamma=0.39,
        init_salinities=[35.302, 36.035, 34.580, 34.821, 34.754]
    ),
    "DePreSys_1999_2008": BoxModelParams(
        name="DePreSys (1999-2008)",
        VN=4.854, VT=7.583, VS=17.247, VIP=38.856, VB=73.55,
        AN=0.194, AT=0.608, AS=-0.282, AIP=-0.519,
        FN=0.531, FS=0.849, FT=-0.743, FIP=-0.637,
        TS=4.385, T0=2.12, mu=2.7, lam=3.53,
        KN=17.070, KS=3.546, KIP=192.649, eta=19.689,
        gamma=0.33,
        init_salinities=[35.120, 35.610, 34.480, 34.710, 34.600]
    )
}


class WoodBoxModel:
    """
    Five-compartment global ocean box model of the AMOC from Wood et al. (2019).
    
    Boxes:
        - N: North Atlantic Deep Water formation region and Arctic
        - T: Atlantic thermocline
        - S: Southern Ocean surface / Antarctic Intermediate Water
        - IP: Indo-Pacific thermocline
        - B: Bottom waters (southward NADW and Circumpolar Deep Water)
    """

    def __init__(self, params: BoxModelParams):
        self.p = params

    def q(self, SN: float, SS: float) -> float:
        """
        Compute AMOC flow strength q in Sverdrups (Sv).
        
        From Eq. (1) and Eq. (12) (or Eq. 15 in Wood et al. 2019):
            q = λ * [α * (TS - T0) + β * (SN - SS)] / (1 + λ * α * μ)
            
        With units:
            λ in 10^7 m^6 kg^-1 s^-1 (giving 10 Sv per unit),
            μ in 10^-8 °C m^-3 s,
            so the denominator term λ * α * μ has factor 1e-1.
        """
        p = self.p
        denom = 1.0 + 1e-1 * p.lam * p.alpha * p.mu
        dT = p.TS - p.T0
        dS = SN - SS
        # Multiply by 10 to convert from 10^7 m^3/s to Sv (10^6 m^3/s)
        return (p.lam / denom) * (p.alpha * dT + p.beta * dS) * 10.0

    def TN(self, q: float) -> float:
        """North Atlantic box temperature (°C), Eq. (12): TN = mu * q + T0."""
        # Convert q (Sv) to SI transport if applying mu directly:
        # mu is in 10^-8 °C / (m^3/s). q in Sv = 10^6 m^3/s.
        # mu * 10^-8 * q * 10^6 = mu * q * 0.01
        return 0.01 * self.p.mu * q + self.p.T0

    def derivatives(self, S: List[float], t: float, H: float) -> List[float]:
        """
        Compute dS/dt (psu/yr) for the 5 boxes at given salinities and hosing H.
        
        Equations (2)-(6) for q >= 0, and (7)-(11) for q < 0.
        """
        p = self.p
        SN, ST, SS, SIP, SB = S
        q_val = self.q(SN, SS)

        # Effective surface freshwater fluxes including hosing:
        # Fi_eff = Fi + Ai * H
        FN_eff = p.FN + p.AN * H
        FT_eff = p.FT + p.AT * H
        FS_eff = p.FS + p.AS * H
        FIP_eff = p.FIP + p.AIP * H
        S0 = p.S0

        if q_val >= 0.0:
            # Equations (2) - (6)
            dSN = (q_val * (ST - SN) + p.KN * (ST - SN) - FN_eff * S0) / p.VN
            dST = (q_val * (p.gamma * SS + (1.0 - p.gamma) * SIP - ST)
                   + p.KS * (SS - ST) + p.KN * (SN - ST) - FT_eff * S0) / p.VT
            dSS = (p.gamma * q_val * (SB - SS) + p.KIP * (SIP - SS)
                   + p.KS * (ST - SS) + p.eta * (SB - SS) - FS_eff * S0) / p.VS
            dSIP = ((1.0 - p.gamma) * q_val * (SB - SIP)
                    + p.KIP * (SS - SIP) - FIP_eff * S0) / p.VIP
            dSB = (q_val * (SN - SB) + p.eta * (SS - SB)) / p.VB
        else:
            # Equations (7) - (11): Reverse overturning cell (|q|)
            abs_q = abs(q_val)
            dSN = (abs_q * (SB - SN) + p.KN * (ST - SN) - FN_eff * S0) / p.VN
            dST = (abs_q * (SN - ST) + p.KS * (SS - ST) + p.KN * (SN - ST) - FT_eff * S0) / p.VT
            dSS = (p.gamma * abs_q * (ST - SS) + p.KIP * (SIP - SS)
                   + p.KS * (ST - SS) + p.eta * (SB - SS) - FS_eff * S0) / p.VS
            dSIP = ((1.0 - p.gamma) * abs_q * (ST - SIP)
                    + p.KIP * (SS - SIP) - FIP_eff * S0) / p.VIP
            dSB = (p.gamma * abs_q * SS + (1.0 - p.gamma) * abs_q * SIP
                   - abs_q * SB + p.eta * (SS - SB)) / p.VB

        # Convert from s^-1 to yr^-1
        return [
            dSN * CONV_PER_YEAR,
            dST * CONV_PER_YEAR,
            dSS * CONV_PER_YEAR,
            dSIP * CONV_PER_YEAR,
            dSB * CONV_PER_YEAR,
        ]

    def solve_equilibrium(self, total_salt: Optional[float] = None) -> List[float]:
        """
        Integrate forward in time with H=0 to find the unperturbed equilibrium state.
        """
        S_curr = self.p.init_salinities or [35.1, 35.6, 34.4, 34.7, 34.6]
        # Integrate forward 10,000 years with H = 0 to reach steady-state
        t_span = np.linspace(0, 10000, 1001)
        res = odeint(lambda s, t: self.derivatives(s, t, H=0.0), S_curr, t_span)
        return res[-1].tolist()

    def freshwater_transports(self, SN: float, ST: float, SS: float, SIP: float, SB: float) -> Dict[str, float]:
        """
        Calculate overturning freshwater transport diagnostics (Eq. 16, etc.):
            - NOV: transport into North Atlantic box across ~40°N
            - TOV: transport into Atlantic thermocline
            - BOV: overturning transport into combined Atlantic/NADW system
        """
        q_val = self.q(SN, SS)
        S0 = self.p.S0
        NOV = -q_val * (ST - SN) / S0
        gamma = self.p.gamma
        BOV = q_val * (gamma * (SB - SS) + (1.0 - gamma) * (SB - SIP)) / S0
        return {"q": q_val, "NOV": NOV, "BOV": BOV}

    def run_hosing_experiment(
        self,
        ramp_rate: float = 5e-4,     # Sv/year
        H_max: float = 1.0,          # Sv
        H_min: float = -0.4,         # Sv
        dt: float = 1.0,             # year
        initial_salinities: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform standard quasi-equilibrium hosing hysteresis experiment as in Wood et al. 2019:
            - Phase 1 (Ramp-up): H increases linearly from 0 to H_max at ramp_rate.
            - Phase 2 (Ramp-down): H decreases linearly from H_max to H_min at ramp_rate.
        """
        # Time durations
        t_ramp_up = H_max / ramp_rate
        t_ramp_down = (H_max - H_min) / ramp_rate
        total_time = t_ramp_up + t_ramp_down

        t = np.arange(0, total_time + dt, dt)

        # Vectorized or callable H(t)
        def H_func(time_yr: float) -> float:
            if time_yr <= t_ramp_up:
                return ramp_rate * time_yr
            else:
                return H_max - ramp_rate * (time_yr - t_ramp_up)

        H_series = np.array([H_func(ti) for ti in t])

        # ODE system for scipy.integrate.odeint
        def rhs(S, ti):
            H_now = H_func(ti)
            return self.derivatives(S, ti, H_now)

        S_start = initial_salinities or self.p.init_salinities or self.solve_equilibrium()
        sol = odeint(rhs, S_start, t)

        q_series = np.array([self.q(sol[i, 0], sol[i, 2]) for i in range(len(t))])

        # Find collapse threshold (first crossing of q <= 0 during ramp-up)
        H_crit_collapse = None
        t_collapse = None
        for i, ti in enumerate(t):
            if ti <= t_ramp_up and q_series[i] <= 0.0:
                H_crit_collapse = H_series[i]
                t_collapse = ti
                break

        # Find recovery threshold (crossing of q > 0 during ramp-down)
        H_crit_recovery = None
        t_recovery = None
        for i, ti in enumerate(t):
            if ti > t_ramp_up and q_series[i] > 0.0:
                H_crit_recovery = H_series[i]
                t_recovery = ti
                break

        return {
            "time": t,
            "H": H_series,
            "q": q_series,
            "SN": sol[:, 0],
            "ST": sol[:, 1],
            "SS": sol[:, 2],
            "SIP": sol[:, 3],
            "SB": sol[:, 4],
            "t_ramp_up": t_ramp_up,
            "H_crit_collapse": H_crit_collapse,
            "t_collapse": t_collapse,
            "H_crit_recovery": H_crit_recovery,
            "t_recovery": t_recovery,
        }
