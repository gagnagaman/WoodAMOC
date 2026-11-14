#!/usr/bin/env python3
"""
Recreate Figure 5 from Wood et al. (2019):
Comparison of AMOC ramp-up threshold across climate model calibrations
(FAMOUSA, FAMOUSB, HadGEM2-AO, DePreSys).

Usage:
    python plot_comparison.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wood_box_model import WoodBoxModel, CALIBRATIONS


def main():
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # ----------------------------------------------------
    # Panel 1: FAMOUS comparisons (Fig. 5b in paper)
    # ----------------------------------------------------
    famous_cases = [
        ("FAMOUSA_1xCO2", "FAMOUSA (1xCO2)", "black", "-"),
        ("FAMOUSB_1xCO2", "FAMOUSB (1xCO2)", "blue", "-"),
        ("FAMOUSB_2xCO2", "FAMOUSB (2xCO2)", "brown", "--"),
    ]

    ax1 = axes[0]
    for key, label, color, lstyle in famous_cases:
        p = CALIBRATIONS[key]
        model = WoodBoxModel(p)
        res = model.run_hosing_experiment()
        # Only show ramp-up stage up to 1.0 Sv
        idx = res["time"] <= res["t_ramp_up"]
        H = res["H"][idx]
        q = res["q"][idx]
        ax1.plot(H, q, color=color, linestyle=lstyle, linewidth=2.0, label=label)
        if res["H_crit_collapse"] is not None:
            ax1.axvline(res["H_crit_collapse"], color=color, linestyle=":", alpha=0.6)

    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("Freshwater Hosing H (Sv)", fontsize=11)
    ax1.set_ylabel("AMOC Strength q (Sv)", fontsize=11)
    ax1.set_title("FAMOUS Calibrations (cf. Fig. 5b)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=10)

    # ----------------------------------------------------
    # Panel 2: HadGEM2-AO & DePreSys comparisons (Fig. 5c,d in paper)
    # ----------------------------------------------------
    hadgem_cases = [
        ("HadGEM2AO_1xCO2", "HadGEM2-AO (1xCO2)", "blue", "-"),
        ("HadGEM2AO_2xCO2", "HadGEM2-AO (2xCO2)", "green", "--"),
        ("HadGEM2AO_4xCO2", "HadGEM2-AO (4xCO2)", "purple", "-."),
        ("DePreSys_1999_2008", "DePreSys Reanalysis", "red", ":"),
    ]

    ax2 = axes[1]
    for key, label, color, lstyle in hadgem_cases:
        p = CALIBRATIONS[key]
        model = WoodBoxModel(p)
        res = model.run_hosing_experiment()
        idx = res["time"] <= res["t_ramp_up"]
        H = res["H"][idx]
        q = res["q"][idx]
        ax2.plot(H, q, color=color, linestyle=lstyle, linewidth=2.0, label=label)
        if res["H_crit_collapse"] is not None:
            ax2.axvline(res["H_crit_collapse"], color=color, linestyle=":", alpha=0.6)

    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("Freshwater Hosing H (Sv)", fontsize=11)
    ax2.set_title("HadGEM2-AO & Reanalysis (cf. Fig. 5c,d)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=10)

    fig.suptitle("Wood et al. (2019) Box Model: AMOC Threshold Shifts Under Climate Change", fontsize=13)
    plt.tight_layout()

    outpath = os.path.join(outdir, "figure5_comparison.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved multi-model comparison to: {outpath}")


if __name__ == "__main__":
    main()
