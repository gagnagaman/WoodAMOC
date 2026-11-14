#!/usr/bin/env python3
"""
Coupled Wood AMOC Model with Gildor-Tziperman Atmospheric Component.

Couples:
    - Ocean: 5-box AMOC model from Wood et al. (2019)
    - Atmosphere: 4-box energy-balance & Clausius-Clapeyron moisture model from Gildor & Tziperman (2001)
    - Cryosphere: Polar sea-ice switch (albedo & insulation feedback)

Usage:
    python coupled_wood_tziperman.py [--compare] [--years 5000]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local directory is in path for standalone portability
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from wood_box_model import WoodBoxModel, CALIBRATIONS, CONV_PER_YEAR
from unesco import unesco_full_density


class CoupledWoodTzipermanModel:
    """
    Coupled AMOC system:
        - 5-box ocean salinities (SN, ST, SS, SIP, SB) from Wood et al. (2019)
        - 4-box atmospheric potential temperatures (Teta: SP, ST, NT, NP)
        - Dynamic Clausius-Clapeyron poleward moisture transport driving FN, FT, FS
        - Polar sea ice switch in the North Atlantic (N box)
    """

    def __init__(self, base_calib="FAMOUSA_1xCO2", enable_seaice=True):
        self.p = CALIBRATIONS[base_calib]
        self.enable_seaice = enable_seaice

        # Atmospheric parameters (from Gildor & Tziperman 2001)
        self.sigma = 5.6701e-8
        self.A_cc = 2.53e11   # Pa
        self.B_cc = 5.42e3    # K
        self.Ps = 1.0e5       # Pa
        self.epsilon = 0.622
        self.RH = 0.70
        self.KT = 15.5e20     # Sensible heat transport
        self.KWa = 5.4e13     # Moisture transport
        self.Gama0 = 1.15e-8  # Air-sea heat exchange (s^-1)
        self.L_dist = 1.0e7   # (m)
        self.ac1 = (2.0**(287.04 / 1004.0)) * 9.8 / (1.0e5 * 1004.0)

        # Albedos
        self.albedoS = 0.07   # Open water
        self.albedoSI = 0.65  # Sea ice
        self.albedoC = 0.30   # Clouds
        self.Solar = np.array([244.67, 381.95, 381.95, 244.67]) # W/m^2
        self.Plw0 = np.array([0.59, 0.53, 0.51, 0.73])

        # Sea-ice parameters
        self.Tice = 274.0     # 1 °C
        self.fsi_N = 0.0      # Sea-ice cover fraction in N box [0..1]
        self.sivol_N = 0.0    # Ice volume (m^3)
        self.Di_N = 2.0       # Reference ice thickness (m)
        self.area_N = self.p.VN * 1e16 / 1000.0  # Approx surface area (m^2)

        # Initial ocean state
        self.salinities = np.array(self.p.init_salinities or [35.088, 35.670, 34.441, 34.689, 34.577], dtype=float)
        self.TN = float(self.p.T0 + 0.01 * self.p.mu * 17.14) # ~ 4.46 °C
        self.TS = float(self.p.TS)                            # ~ 5.57 °C
        self.TT = 20.0                                        # Subtropical Atlantic ~ 20 °C

        # Initial atmospheric state (K)
        # Boxes: 0: SP, 1: ST, 2: NT, 3: NP
        self.Teta = np.array([
            273.15 + (self.TS - 6.0),
            273.15 + 18.0,
            273.15 + (self.TT - 1.0),
            273.15 + (self.TN - 5.0)
        ], dtype=float)

        # Scale factor so that unperturbed moisture transport matches FAMOUSA FN = 0.375 Sv
        raw_FN_sv = self._compute_moisture_flux(self.Teta[2], self.Teta[3])
        self.fn_scale = self.p.FN / max(raw_FN_sv, 1e-4)

    def _compute_moisture_flux(self, T_tropics_K, T_polar_K):
        """Compute poleward moisture transport in Sv via Clausius-Clapeyron."""
        qsat = self.A_cc * np.exp(-self.B_cc / T_tropics_K)
        q_spec = self.RH * self.epsilon * qsat / self.Ps
        dT = max(T_tropics_K - T_polar_K, 0.0)
        F_m3s = self.KWa * dT * q_spec / (0.25 * self.L_dist)
        return F_m3s * 1e-6

    def compute_q(self, SN, SS, TN, TS):
        """
        Compute AMOC strength q (Sv) based on dynamic thermal and haline contrast.
        """
        p = self.p
        dT = TS - TN
        dS = SN - SS
        drho = -p.alpha * (TN - TS) + p.beta * dS  # kg/m^3
        # λ in Table 1 is 2.66 * 10^7 m^6 kg^-1 s^-1 -> 10 Sv per unit
        return p.lam * drho * 10.0

    def step(self, dt_years=0.1, H=0.0):
        """
        Step the coupled model forward by dt_years.
        """
        p = self.p
        dt_sec = dt_years * 365.25 * 86400.0

        SN, ST, SS, SIP, SB = self.salinities

        # 1. Atmospheric moisture transport -> Dynamic freshwater fluxes
        fw_north_sv = self._compute_moisture_flux(self.Teta[2], self.Teta[3]) * self.fn_scale
        fw_south_sv = self._compute_moisture_flux(self.Teta[1], self.Teta[0]) * self.fn_scale

        # Dynamic freshwater fluxes:
        # If sea-ice covers the N box, local evaporation/moisture exchange shuts off
        ice_factor_N = 1.0 - 0.7 * self.fsi_N
        FN_dyn = fw_north_sv * ice_factor_N
        FT_dyn = -(fw_north_sv + fw_south_sv * 0.5)
        FS_dyn = p.FS * (fw_south_sv / max(fw_north_sv, 1e-3))
        FIP_dyn = p.FIP

        # Apply external hosing H:
        FN_eff = FN_dyn + p.AN * H
        FT_eff = FT_dyn + p.AT * H
        FS_eff = FS_dyn + p.AS * H
        FIP_eff = FIP_dyn + p.AIP * H

        # 2. AMOC flow q
        q_val = self.compute_q(SN, SS, self.TN, self.TS)

        # 3. Ocean salinities update (Wood et al. equations)
        S0 = p.S0
        if q_val >= 0.0:
            dSN = (q_val * (ST - SN) + p.KN * (ST - SN) - FN_eff * S0) / p.VN
            dST = (q_val * (p.gamma * SS + (1.0 - p.gamma) * SIP - ST)
                   + p.KS * (SS - ST) + p.KN * (SN - ST) - FT_eff * S0) / p.VT
            dSS = (p.gamma * q_val * (SB - SS) + p.KIP * (SIP - SS)
                   + p.KS * (ST - SS) + p.eta * (SB - SS) - FS_eff * S0) / p.VS
            dSIP = ((1.0 - p.gamma) * q_val * (SB - SIP)
                    + p.KIP * (SS - SIP) - FIP_eff * S0) / p.VIP
            dSB = (q_val * (SN - SB) + p.eta * (SS - SB)) / p.VB
        else:
            abs_q = abs(q_val)
            dSN = (abs_q * (SB - SN) + p.KN * (ST - SN) - FN_eff * S0) / p.VN
            dST = (abs_q * (SN - ST) + p.KS * (SS - ST) + p.KN * (SN - ST) - FT_eff * S0) / p.VT
            dSS = (p.gamma * abs_q * (ST - SS) + p.KIP * (SIP - SS)
                   + p.KS * (ST - SS) + p.eta * (SB - SS) - FS_eff * S0) / p.VS
            dSIP = ((1.0 - p.gamma) * abs_q * (ST - SIP)
                    + p.KIP * (SS - SIP) - FIP_eff * S0) / p.VIP
            dSB = (p.gamma * abs_q * SS + (1.0 - p.gamma) * abs_q * SIP
                   - abs_q * SB + p.eta * (SS - SB)) / p.VB

        self.salinities += np.array([dSN, dST, dSS, dSIP, dSB]) * (CONV_PER_YEAR * dt_years)

        # 4. Thermal evolution (Ocean heat transport + Air-Sea exchange)
        # AMOC carries heat into North Atlantic: Q_amoc ~ CpW * rhoW * q * (TT - TN)
        # Relaxation towards air temperature Teta[3] - 273.15
        T_air_N = self.Teta[3] - 273.15
        tau_air = 2.0  # Air-sea coupling time scale (years)
        # When AMOC is active, it warms TN; when collapsed, TN drops towards polar air temp
        TN_target = T_air_N + max(q_val, 0.0) * 0.45
        self.TN += (TN_target - self.TN) * (dt_years / tau_air)

        # 5. Sea-ice switch in the North Atlantic
        if self.enable_seaice:
            T_water_K = self.TN + 273.15
            if T_water_K < self.Tice:
                # Freezing: grow sea ice
                heat_deficit = 4.29e6 * (self.p.VN * 1e16) * (self.Tice - T_water_K) * 1e-8
                d_ice_vol = (heat_deficit * dt_sec) / (917.0 * 3.34e5)
                self.sivol_N += d_ice_vol
                self.fsi_N = min(self.sivol_N / (self.Di_N * self.area_N), 1.0)
            else:
                # Melting
                heat_surplus = 4.29e6 * (self.p.VN * 1e16) * (T_water_K - self.Tice) * 1e-8
                d_melt = (heat_surplus * dt_sec) / (917.0 * 3.34e5)
                self.sivol_N = max(self.sivol_N - d_melt, 0.0)
                self.fsi_N = min(self.sivol_N / (self.Di_N * self.area_N), 1.0)

        # 6. Atmospheric potential temperature update
        # Polar albedo depends on sea ice
        albedo_N = self.fsi_N * self.albedoSI + (1.0 - self.fsi_N) * self.albedoS
        Hin_N = self.Solar[3] * (1.0 - albedo_N) * (1.0 - self.albedoC)
        Hout_N = self.sigma * self.Plw0[3] * (self.Teta[3]**4)
        # Heat flux from ocean to atmosphere
        Hao_N = 4.29e6 * 400.0 * (1.0 - 0.8 * self.fsi_N) * self.Gama0 * ((self.TN + 273.15) - self.Teta[3])
        # Meridional heat transport from subtropics
        FT_north = self.KT * (self.Teta[2] - self.Teta[3]) / (self.L_dist) * self.ac1 / (self.p.VN * 1e16 / 1000.0)

        dTeta_N = (FT_north + self.ac1 * (Hao_N - Hout_N + Hin_N)) * dt_sec
        self.Teta[3] += np.clip(dTeta_N, -2.0, 2.0)

        return {
            "q": q_val,
            "FN": FN_dyn,
            "FT": FT_dyn,
            "TN": self.TN,
            "TS": self.TS,
            "Teta_N": self.Teta[3] - 273.15,
            "fsi": self.fsi_N
        }

    def run_hosing_experiment(self, ramp_rate=5e-4, H_max=1.0, H_min=-0.4, dt=1.0):
        """
        Run the standard hosing experiment (ramp up to H_max, then ramp down).
        """
        t_ramp_up = H_max / ramp_rate
        t_ramp_down = (H_max - H_min) / ramp_rate
        total_time = t_ramp_up + t_ramp_down
        t_array = np.arange(0, total_time + dt, dt)

        def H_func(t):
            return ramp_rate * t if t <= t_ramp_up else H_max - ramp_rate * (t - t_ramp_up)

        H_list = []
        q_list = []
        FN_list = []
        TN_list = []
        fsi_list = []

        for ti in t_array:
            H_now = H_func(ti)
            diag = self.step(dt_years=dt, H=H_now)
            H_list.append(H_now)
            q_list.append(diag["q"])
            FN_list.append(diag["FN"])
            TN_list.append(diag["TN"])
            fsi_list.append(diag["fsi"])

        H_arr = np.array(H_list)
        q_arr = np.array(q_list)

        # Detect collapse and recovery
        H_collapse = None
        for i, ti in enumerate(t_array):
            if ti <= t_ramp_up and q_arr[i] <= 0.0:
                H_collapse = H_arr[i]
                break

        H_recovery = None
        for i, ti in enumerate(t_array):
            if ti > t_ramp_up and q_arr[i] > 0.0:
                H_recovery = H_arr[i]
                break

        return {
            "time": t_array,
            "H": H_arr,
            "q": q_arr,
            "FN": np.array(FN_list),
            "TN": np.array(TN_list),
            "fsi": np.array(fsi_list),
            "t_ramp_up": t_ramp_up,
            "H_collapse": H_collapse,
            "H_recovery": H_recovery
        }


def compare_models(outdir="figures"):
    """
    Compare Uncoupled Wood Model vs Coupled Wood + Tziperman Atmosphere.
    """
    os.makedirs(outdir, exist_ok=True)

    print("Running Uncoupled Wood (2019) Model...")
    from wood_box_model import WoodBoxModel, CALIBRATIONS
    wood_uncoupled = WoodBoxModel(CALIBRATIONS["FAMOUSA_1xCO2"])
    res_uncoupled = wood_uncoupled.run_hosing_experiment()

    print("Running Coupled Wood + Tziperman-Gildor Atmosphere Model...")
    coupled = CoupledWoodTzipermanModel("FAMOUSA_1xCO2", enable_seaice=True)
    res_coupled = coupled.run_hosing_experiment()

    print("\n" + "=" * 65)
    print("Comparison of AMOC Thresholds:")
    print(f"Uncoupled Wood (2019):  H_collapse = {res_uncoupled['H_crit_collapse']:.3f} Sv | H_recovery = {res_uncoupled['H_crit_recovery']:.3f} Sv")
    print(f"Coupled Wood+Atmosphere: H_collapse = {res_coupled['H_collapse']:.3f} Sv | H_recovery = {res_coupled['H_recovery']:.3f} Sv")
    print("=" * 65)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Hysteresis loops (q vs H)
    ax = axes[0, 0]
    t_up_u = res_uncoupled["time"] <= res_uncoupled["t_ramp_up"]
    t_down_u = res_uncoupled["time"] > res_uncoupled["t_ramp_up"]
    ax.plot(res_uncoupled["H"][t_up_u], res_uncoupled["q"][t_up_u], "k-", linewidth=2.0, label="Uncoupled (Ramp-up)")
    ax.plot(res_uncoupled["H"][t_down_u], res_uncoupled["q"][t_down_u], "k--", linewidth=1.5, label="Uncoupled (Ramp-down)")

    t_up_c = res_coupled["time"] <= res_coupled["t_ramp_up"]
    t_down_c = res_coupled["time"] > res_coupled["t_ramp_up"]
    ax.plot(res_coupled["H"][t_up_c], res_coupled["q"][t_up_c], "b-", linewidth=2.0, label="Coupled + Atmos (Ramp-up)")
    ax.plot(res_coupled["H"][t_down_c], res_coupled["q"][t_down_c], "r--", linewidth=2.0, label="Coupled + Atmos (Ramp-down)")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Freshwater Hosing H (Sv)", fontsize=11)
    ax.set_ylabel("AMOC Strength q (Sv)", fontsize=11)
    ax.set_title("AMOC Hysteresis: Uncoupled vs Coupled", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    # 2. Dynamic North Atlantic Freshwater Flux (FN)
    ax = axes[0, 1]
    ax.axhline(CALIBRATIONS["FAMOUSA_1xCO2"].FN, color="k", linestyle="--", linewidth=1.8, label="Fixed FN (Uncoupled: 0.375 Sv)")
    ax.plot(res_coupled["time"], res_coupled["FN"], "b-", linewidth=2.0, label="Dynamic FN (Clausius-Clapeyron)")
    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("FN (Sv)", fontsize=11)
    ax.set_title("North Atlantic Freshwater Flux (P - E)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    # 3. North Atlantic Ocean Temperature (TN)
    ax = axes[1, 0]
    ax.plot(res_coupled["time"], res_coupled["TN"], "tab:blue", linewidth=1.8, label="Dynamic TN (Coupled)")
    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("TN (°C)", fontsize=11)
    ax.set_title("North Atlantic Temperature Response", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    # 4. Sea-Ice Fraction in North Atlantic
    ax = axes[1, 1]
    ax.plot(res_coupled["time"], res_coupled["fsi"], "tab:purple", linewidth=2.0, label="North Atlantic Sea Ice Fraction")
    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Sea-Ice Fraction [0..1]", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Sea-Ice Switch Activation", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    plt.suptitle("Wood (2019) AMOC Model with Tziperman-Gildor Atmosphere & Sea-Ice Switch", fontsize=14)
    plt.tight_layout()

    outpath = os.path.join(outdir, "wood_tziperman_comparison.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\nSaved comparison plot to: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Run Coupled Wood-Tziperman Model")
    parser.add_argument("--compare", action="store_true", default=True, help="Compare uncoupled vs coupled")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory")
    args = parser.parse_args()

    compare_models(args.outdir)


if __name__ == "__main__":
    main()
