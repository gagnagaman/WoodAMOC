#!/usr/bin/env python3
"""
Run Wood et al. (2019) AMOC 5-box model simulations and generate plots.

Usage:
    python simulate.py [--model FAMOUSA_1xCO2] [--plot]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from wood_box_model import WoodBoxModel, CALIBRATIONS


def plot_simulation_results(res: dict, model_name: str, output_dir: str):
    """Generate hysteresis loop and salinity evolution plots."""
    os.makedirs(output_dir, exist_ok=True)
    t = res["time"]
    H = res["H"]
    q = res["q"]
    t_ramp_up = res["t_ramp_up"]
    idx_up = t <= t_ramp_up
    idx_down = t > t_ramp_up

    # --- Plot 1: AMOC Hysteresis Curve (q vs H) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(H[idx_up], q[idx_up], "b-", linewidth=2.0, label="Ramp-up (dH > 0)")
    ax.plot(H[idx_down], q[idx_down], "r--", linewidth=2.0, label="Ramp-down (dH < 0)")
    
    # Mark unperturbed initial state
    ax.plot(H[0], q[0], "ko", markersize=8, label=f"Unperturbed (q0 = {q[0]:.1f} Sv)")
    
    # Threshold indicators
    if res["H_crit_collapse"] is not None:
        ax.axvline(res["H_crit_collapse"], color="blue", linestyle=":", alpha=0.7,
                   label=f"H_crit collapse = {res['H_crit_collapse']:.2f} Sv")
    if res["H_crit_recovery"] is not None:
        ax.axvline(res["H_crit_recovery"], color="red", linestyle=":", alpha=0.7,
                   label=f"H_crit recovery = {res['H_crit_recovery']:.2f} Sv")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Freshwater Hosing H (Sv)", fontsize=12)
    ax.set_ylabel("AMOC Strength q (Sv)", fontsize=12)
    ax.set_title(f"AMOC Hysteresis Loop - {model_name}\n(Wood et al. 2019 Box Model)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    hyst_path = os.path.join(output_dir, f"amoc_hysteresis_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(hyst_path, dpi=200)
    plt.close(fig)
    print(f"Saved hysteresis plot to: {hyst_path}")

    # --- Plot 2: Salinity and Forcing Evolution vs Time ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(t, res["SN"], "b-", label="SN (North Atlantic)")
    ax1.plot(t, res["ST"], "g-", label="ST (Thermocline)")
    ax1.plot(t, res["SS"], "r-", label="SS (Southern Ocean)")
    ax1.plot(t, res["SIP"], "m-", label="SIP (Indo-Pacific)")
    ax1.plot(t, res["SB"], "k-", label="SB (Bottom / Deep)")
    ax1.set_ylabel("Salinity (psu)", fontsize=11)
    ax1.set_title(f"Box Salinities and Forcing Evolution - {model_name}", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", ncol=3, fontsize=9)

    ax2.plot(t, q, "k-", linewidth=1.8, label="AMOC q (Sv)")
    ax2.set_ylabel("AMOC q (Sv)", fontsize=11, color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_h = ax2.twinx()
    ax2_h.plot(t, H, "c--", linewidth=1.5, label="Hosing H (Sv)")
    ax2_h.set_ylabel("Hosing H (Sv)", fontsize=11, color="tab:cyan")
    ax2_h.tick_params(axis="y", labelcolor="tab:cyan")

    ax2.set_xlabel("Time (years)", fontsize=11)
    plt.tight_layout()
    sal_path = os.path.join(output_dir, f"salinity_evolution_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(sal_path, dpi=200)
    plt.close(fig)
    print(f"Saved salinity evolution plot to: {sal_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate AMOC 5-box model (Wood et al. 2019)")
    parser.add_argument("--model", type=str, default="FAMOUSA_1xCO2",
                        choices=list(CALIBRATIONS.keys()) + ["all"],
                        help="Calibration key to run (default: FAMOUSA_1xCO2)")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Directory to save generated figures")
    args = parser.parse_args()

    models_to_run = list(CALIBRATIONS.keys()) if args.model == "all" else [args.model]

    print("=" * 65)
    print("Wood et al. (2019) AMOC 5-Box Model Simulation")
    print("=" * 65)

    summary_table = []

    for m_key in models_to_run:
        params = CALIBRATIONS[m_key]
        model = WoodBoxModel(params)
        print(f"\nRunning simulation for: {params.name}...")
        res = model.run_hosing_experiment()

        q0 = res["q"][0]
        hc = res["H_crit_collapse"]
        hr = res["H_crit_recovery"]

        print(f"  -> Initial AMOC strength q0: {q0:.2f} Sv")
        print(f"  -> Collapse threshold H_crit: {hc:.3f} Sv" if hc is not None else "  -> No collapse detected")
        print(f"  -> Recovery threshold H_crit: {hr:.3f} Sv" if hr is not None else "  -> No recovery detected")

        summary_table.append((params.name, q0, hc, hr))
        plot_simulation_results(res, params.name, args.outdir)

    print("\n" + "=" * 65)
    print("Summary of Thresholds:")
    print(f"{'Model Calibration':<25} | {'q0 (Sv)':<8} | {'H_collapse (Sv)':<16} | {'H_recovery (Sv)':<16}")
    print("-" * 72)
    for name, q0, hc, hr in summary_table:
        hc_str = f"{hc:.3f}" if hc is not None else "N/A"
        hr_str = f"{hr:.3f}" if hr is not None else "N/A"
        print(f"{name:<25} | {q0:<8.2f} | {hc_str:<16} | {hr_str:<16}")
    print("=" * 65)


if __name__ == "__main__":
    main()
